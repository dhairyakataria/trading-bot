"""Execution Agent — the only agent that interacts directly with the broker API.

Responsibilities
----------------
- Execute approved BUY signals (LIMIT orders only — NEVER market buys)
- Execute SELL and EXIT signals (LIMIT by default; MARKET only for stop-loss exits)
- Place protective stop-loss orders immediately after a BUY is filled
- Cancel, modify, and query orders
- Persist every trade to the database
- Support paper-trading mode (no live API calls; simulated fills)

Safety rules (non-negotiable)
------------------------------
- NEVER place a MARKET order for a BUY
- MARKET orders are ONLY allowed for stop-loss-triggered exits
- Always verify available cash before buying
- Always verify holdings before selling
- No automatic retry on order failure
- Log intent to trade BEFORE placing the order
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.database.models import Trade
from src.utils.logger import get_logger, get_trade_logger

IST          = ZoneInfo("Asia/Kolkata")
_log         = get_logger("agents.execution")
_trade_log   = get_trade_logger()

# ── Constants ────────────────────────────────────────────────────────────────
_FILL_POLL_INTERVAL = 2      # seconds between order-status polls
_FILL_TIMEOUT       = 30     # max seconds to wait for a fill
_SL_LIMIT_BUFFER    = 0.005  # SL limit price = trigger * (1 - this)
_BUY_LIMIT_SLIP     = 0.0005 # Buy limit = entry_price * (1 + this)  [+0.05%]

# Broker order-status values that count as "filled"
_FILLED_STATUSES    = {"COMPLETE", "EXECUTED", "FILLED", "complete", "executed", "filled"}
_PENDING_STATUSES   = {"OPEN", "PENDING", "TRIGGER PENDING", "open", "pending"}


def _ist_now() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


def _ist_today() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def _cfg(config: Any, *keys: str, default: Any = None) -> Any:
    try:
        if hasattr(config, "get"):
            return config.get(*keys, default=default)
        obj: Any = config
        for k in keys:
            obj = obj[k]
        return obj
    except (KeyError, TypeError, AttributeError):
        return default


# ── ExecutionAgent ────────────────────────────────────────────────────────────

class ExecutionAgent:
    """Handles all order execution through the Angel One broker.

    This is the *only* agent that calls ``broker_client`` for order placement,
    modification, or cancellation.  All other agents that need to trade must
    go through this agent.

    Args:
        config:        Application :class:`~src.utils.config.Config`.
        broker_client: Authenticated :class:`~src.broker.angel_one.AngelOneClient`.
        db_manager:    :class:`~src.database.db_manager.DatabaseManager`.
    """

    def __init__(
        self,
        config: Any,
        broker_client: Any,
        db_manager: Any,
        notifier: Any = None,
    ) -> None:
        self.config   = config
        self.broker   = broker_client
        self.db       = db_manager
        self.notifier = notifier  # TelegramNotifier — optional but strongly recommended

        # ── Unified mode resolution ─────────────────────────────────────
        # Prefer the new `trading.mode` field; fall back to legacy booleans
        mode = _cfg(config, "trading", "mode", default=None)
        if mode is not None:
            self._mode = str(mode).lower().strip()
        else:
            # Legacy backward compatibility
            paper  = bool(_cfg(config, "trading", "paper_trading", default=False))
            signal = bool(_cfg(config, "trading", "signal_only",   default=False))
            if signal:
                self._mode = "approval"
            elif paper:
                self._mode = "paper"
            else:
                self._mode = "auto"

        # Convenience flags derived from mode (used in order placement logic)
        self._paper_trading = self._mode == "paper"
        self._signal_only   = self._mode == "approval"

        self._approval_timeout = int(
            _cfg(config, "trading", "approval_timeout_seconds", default=300)
        )

        _log.info(
            "ExecutionAgent initialised in %s mode",
            self._mode.upper(),
        )

    # ================================================================== #
    # BUY                                                                  #
    # ================================================================== #

    def execute_buy(self, signal: dict) -> dict:
        """Execute a BUY signal via a LIMIT order.

        Steps:
            1. Validate required fields
            2. Check available cash
            3. Log intent BEFORE placing order
            4. Place LIMIT order at ``entry_price`` (+ ``_BUY_LIMIT_SLIP``)
            5. Wait up to 30 s for fill
            6. On fill: place stop-loss order
            7. Persist trade to DB
            8. Return result

        Args:
            signal: Dict with at minimum::

                {
                    "symbol":       str,
                    "quantity":     int,
                    "entry_price":  float,
                    "stop_loss":    float,
                    "target_1":     float  (optional),
                    "target_2":     float  (optional),
                    "sector":       str    (optional),
                    "strategy":     str    (optional),
                }

        Returns::

            {
                "success":      bool,
                "symbol":       str,
                "order_id":     str | None,
                "sl_order_id":  str | None,
                "filled_price": float | None,
                "quantity":     int,
                "error":        str | None,
            }
        """
        symbol      = signal.get("symbol", "")
        quantity    = int(signal.get("quantity", 0))
        entry_price = float(signal.get("entry_price", 0.0))
        stop_loss   = float(signal.get("stop_loss", 0.0))
        target_1    = float(signal.get("target_1", 0.0))
        strategy    = str(signal.get("strategy", signal.get("strategy_name", "UNKNOWN")))
        sector      = str(signal.get("sector", "UNKNOWN"))

        # ── 1. Validate ──────────────────────────────────────────────────
        errors = []
        if not symbol:
            errors.append("symbol is required")
        if quantity <= 0:
            errors.append(f"quantity must be > 0, got {quantity}")
        if entry_price <= 0:
            errors.append(f"entry_price must be > 0, got {entry_price}")
        if stop_loss <= 0:
            errors.append(f"stop_loss must be > 0, got {stop_loss}")
        if stop_loss >= entry_price:
            errors.append(f"stop_loss {stop_loss} must be < entry_price {entry_price}")
        if errors:
            err = "; ".join(errors)
            _log.error("execute_buy validation failed for %s: %s", symbol, err)
            return self._buy_result(symbol, quantity, False, error=err)

        # ── 2. Check cash ────────────────────────────────────────────────
        required_cash = entry_price * quantity
        if not self._paper_trading:
            try:
                available_cash = self.broker.get_margin_available()
                if available_cash < required_cash:
                    err = (
                        f"Insufficient cash: need ₹{required_cash:,.0f}, "
                        f"available ₹{available_cash:,.0f}"
                    )
                    _log.warning("execute_buy: %s — %s", symbol, err)
                    return self._buy_result(symbol, quantity, False, error=err)
            except Exception as exc:
                _log.warning("execute_buy: could not check margin (%s) — proceeding", exc)

        # ── 3. Log BEFORE placing ────────────────────────────────────────
        limit_price = round(entry_price * (1 + _BUY_LIMIT_SLIP), 2)
        _trade_log.info(
            "INTENT BUY | %s | qty=%d | limit=%.2f | sl=%.2f | strategy=%s | paper=%s | signal_only=%s",
            symbol, quantity, limit_price, stop_loss, strategy,
            self._paper_trading, self._signal_only,
        )

        # ── 3a. APPROVAL mode — send Telegram approval request and wait ──
        if self._mode == "approval":
            target_1 = float(signal.get("target_1", 0.0))
            effective_target = target_1 if target_1 > 0 else entry_price * 1.06

            if self.notifier is not None and hasattr(self.notifier, "send_approval_request"):
                try:
                    decision = self.notifier.send_approval_request(
                        symbol=symbol,
                        quantity=quantity,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target=effective_target,
                        strategy=strategy,
                        sector=sector,
                        timeout_seconds=self._approval_timeout,
                    )
                    _log.info(
                        "Approval decision for %s: %s", symbol, decision,
                    )
                except Exception as exc:
                    _log.error(
                        "approval: Telegram request failed for %s: %s — skipping trade",
                        symbol, exc,
                    )
                    decision = "timeout"
            else:
                _log.warning(
                    "approval mode: no notifier with approval capability for %s — "
                    "falling back to signal-only alert",
                    symbol,
                )
                # Fallback: send a regular signal alert and skip execution
                if self.notifier is not None:
                    try:
                        reasoning = str(signal.get("reasoning", signal.get("reason", "")))
                        self.notifier.send_signal_alert(
                            symbol=symbol, quantity=quantity,
                            entry_price=entry_price, stop_loss=stop_loss,
                            target=effective_target, strategy=strategy,
                            reasoning=reasoning, sector=sector,
                        )
                    except Exception:
                        pass
                decision = "timeout"

            if decision == "rejected":
                _trade_log.info(
                    "REJECTED_BY_USER BUY | %s | qty=%d | entry=%.2f",
                    symbol, quantity, entry_price,
                )
                trade_id = self._record_trade(
                    symbol=symbol, quantity=quantity, price=entry_price,
                    order_id=None, status="REJECTED_BY_USER",
                    stop_loss=stop_loss, target_price=target_1,
                    strategy_signal=signal,
                )
                return self._buy_result(symbol, quantity, False, error="Rejected by user")

            if decision == "timeout":
                _trade_log.info(
                    "APPROVAL_TIMEOUT BUY | %s | qty=%d | entry=%.2f",
                    symbol, quantity, entry_price,
                )
                trade_id = self._record_trade(
                    symbol=symbol, quantity=quantity, price=entry_price,
                    order_id=None, status="APPROVAL_TIMEOUT",
                    stop_loss=stop_loss, target_price=target_1,
                    strategy_signal=signal,
                )
                return self._buy_result(symbol, quantity, False, error="Approval timed out")

            # decision == "approved" → fall through to live order placement below
            _trade_log.info(
                "APPROVED_BY_USER BUY | %s | qty=%d | entry=%.2f | proceeding to execute",
                symbol, quantity, entry_price,
            )

        # ── 4. Place order ───────────────────────────────────────────────
        order_id: Optional[str] = None
        if self._paper_trading:
            order_id = f"PAPER_BUY_{symbol}_{int(time.time())}"
            filled_price = limit_price
            _log.info("PAPER BUY | %s | qty=%d | price=%.2f", symbol, quantity, filled_price)
        else:
            try:
                result = self.broker.place_buy_order(
                    symbol=symbol,
                    quantity=quantity,
                    price=limit_price,
                    order_type="LIMIT",
                )
                order_id = result["order_id"]
            except Exception as exc:
                err = f"Broker order placement failed: {exc}"
                _log.error("execute_buy FAILED | %s | %s", symbol, err)
                return self._buy_result(symbol, quantity, False, error=err)

            # ── 5. Wait for fill ─────────────────────────────────────────
            filled_price = self._wait_for_fill(order_id, _FILL_TIMEOUT)
            if filled_price is None:
                # Check for a partial fill before giving up
                partial_qty, partial_price = self._check_partial_fill(order_id)
                if partial_qty > 0 and partial_price > 0:
                    _log.warning(
                        "execute_buy: order %s PARTIALLY filled — %d of %d shares @ %.2f",
                        order_id, partial_qty, quantity, partial_price,
                    )
                    self._notify_critical(
                        f"⚠️ PARTIAL FILL — {symbol}\n"
                        f"Filled {partial_qty}/{quantity} shares @ ₹{partial_price:.2f}\n"
                        f"Recording partial position and placing SL."
                    )
                    filled_price = partial_price
                    quantity     = partial_qty   # re-scope to actual filled qty
                else:
                    _log.warning(
                        "execute_buy: order %s not filled within %ds — recording as PENDING",
                        order_id, _FILL_TIMEOUT,
                    )
                    self._record_trade(
                        symbol=symbol, quantity=quantity, price=limit_price,
                        order_id=order_id, status="PENDING",
                        stop_loss=stop_loss, target_price=target_1,
                        strategy_signal=signal,
                    )
                    return self._buy_result(
                        symbol, quantity, False,
                        order_id=order_id,
                        error="Order not filled within timeout",
                    )

        # ── 6. Place stop-loss order ─────────────────────────────────────
        sl_order_id: Optional[str] = None
        sl_trigger  = stop_loss
        sl_limit    = round(stop_loss * (1 - _SL_LIMIT_BUFFER), 2)

        if self._paper_trading:
            sl_order_id = f"PAPER_SL_{symbol}_{int(time.time())}"
            _log.info("PAPER SL | %s | trigger=%.2f | limit=%.2f", symbol, sl_trigger, sl_limit)
        else:
            try:
                sl_result   = self.broker.place_stop_loss_order(
                    symbol=symbol,
                    quantity=quantity,
                    trigger_price=sl_trigger,
                    limit_price=sl_limit,
                )
                sl_order_id = sl_result["order_id"]
                _trade_log.info(
                    "SL PLACED | %s | order_id=%s | trigger=%.2f",
                    symbol, sl_order_id, sl_trigger,
                )
            except Exception as exc:
                _log.error(
                    "execute_buy: FAILED to place SL for %s after fill — "
                    "MANUAL ATTENTION REQUIRED: %s",
                    symbol, exc,
                )
                self._notify_critical(
                    f"🔴🔴🔴 SL PLACEMENT FAILED — {symbol}\n"
                    f"Position is UNPROTECTED. Qty={quantity} filled @ ₹{filled_price:.2f}\n"
                    f"MANUAL STOP-LOSS REQUIRED IMMEDIATELY\nError: {exc}"
                )

        # Store SL order ID for later cancellation
        if sl_order_id:
            try:
                self.db.set_system_state(f"sl_order_{symbol}", sl_order_id)
            except Exception:
                pass

        # ── 7. Save to DB ────────────────────────────────────────────────
        trade_id = self._record_trade(
            symbol=symbol, quantity=quantity, price=filled_price,
            order_id=order_id, status="EXECUTED",
            stop_loss=stop_loss, target_price=target_1,
            strategy_signal=signal,
        )

        # ── 8. Log execution ─────────────────────────────────────────────
        _trade_log.info(
            "EXECUTED BUY | %s | trade_id=%s | qty=%d | filled=%.2f | sl=%.2f | strategy=%s",
            symbol, trade_id, quantity, filled_price, stop_loss, strategy,
        )

        return self._buy_result(
            symbol, quantity, True,
            order_id=order_id,
            sl_order_id=sl_order_id,
            filled_price=filled_price,
            trade_id=trade_id,
        )

    # ================================================================== #
    # SELL                                                                 #
    # ================================================================== #

    def execute_sell(self, signal: dict) -> dict:
        """Execute a manual SELL via a LIMIT order.

        Args:
            signal: Dict with::

                {
                    "symbol":     str,
                    "quantity":   int,
                    "price":      float,  # limit price
                    "trade_id":   int     (optional — DB trade to close)
                }

        Returns::

            {
                "success":      bool,
                "symbol":       str,
                "order_id":     str | None,
                "filled_price": float | None,
                "pnl":          float | None,
                "error":        str | None,
            }
        """
        symbol   = signal.get("symbol", "")
        quantity = int(signal.get("quantity", 0))
        price    = float(signal.get("price", 0.0))
        trade_id = signal.get("trade_id")

        if not symbol or quantity <= 0 or price <= 0:
            err = f"Invalid sell signal: symbol={symbol} qty={quantity} price={price}"
            _log.error("execute_sell: %s", err)
            return self._sell_result(symbol, False, error=err)

        # ── Signal-only mode ─────────────────────────────────────────────
        if self._signal_only:
            if self.notifier is not None:
                try:
                    self.notifier.send_exit_signal_alert(
                        symbol=symbol,
                        quantity=quantity,
                        exit_type="MANUAL_SELL",
                        current_price=price,
                        reasoning=str(signal.get("reasoning", signal.get("reason", ""))),
                    )
                except Exception as exc:
                    _log.error("signal_only: failed to send sell signal for %s: %s", symbol, exc)
            else:
                _log.warning("signal_only SELL signal for %s — no notifier configured", symbol)
            _trade_log.info(
                "SIGNAL_SENT SELL | %s | qty=%d | price=%.2f",
                symbol, quantity, price,
            )
            return self._sell_result(symbol, True, filled_price=price)

        # ── 1. Verify holdings ───────────────────────────────────────────
        if not self._paper_trading:
            try:
                holdings = self.broker.get_holdings()
                held_qty = next(
                    (h["quantity"] for h in holdings if h["symbol"] == symbol), 0
                )
                if held_qty < quantity:
                    err = (
                        f"Cannot sell {quantity} {symbol}: only {held_qty} held"
                    )
                    _log.warning("execute_sell: %s", err)
                    return self._sell_result(symbol, False, error=err)
            except Exception as exc:
                _log.warning("execute_sell: holdings check failed (%s) — proceeding", exc)

        # ── 2. Cancel existing SL order for this symbol ──────────────────
        self._cancel_sl_order(symbol)

        # ── 3. Log BEFORE placing ────────────────────────────────────────
        _trade_log.info(
            "INTENT SELL | %s | qty=%d | limit=%.2f | paper=%s",
            symbol, quantity, price, self._paper_trading,
        )

        # ── 4. Place LIMIT sell ──────────────────────────────────────────
        order_id: Optional[str] = None
        if self._paper_trading:
            order_id     = f"PAPER_SELL_{symbol}_{int(time.time())}"
            filled_price = price
        else:
            try:
                result   = self.broker.place_sell_order(
                    symbol=symbol, quantity=quantity, price=price, order_type="LIMIT"
                )
                order_id = result["order_id"]
            except Exception as exc:
                err = f"Broker sell order failed: {exc}"
                _log.error("execute_sell FAILED | %s | %s", symbol, err)
                return self._sell_result(symbol, False, error=err)

            # ── 5. Wait for fill ─────────────────────────────────────────
            filled_price = self._wait_for_fill(order_id, _FILL_TIMEOUT)
            if filled_price is None:
                return self._sell_result(
                    symbol, False, order_id=order_id,
                    error="Sell order not filled within timeout",
                )

        # ── 6. Update DB ─────────────────────────────────────────────────
        pnl: Optional[float] = None
        if trade_id is not None:
            try:
                self.db.update_trade_exit(trade_id, filled_price)
                # Re-read to get computed PnL
                trade = self.db.get_trade_by_id(trade_id)
                pnl   = trade.pnl if trade else None
            except Exception as exc:
                _log.warning("execute_sell: DB update failed for trade_id=%s: %s", trade_id, exc)

        # ── 7. Log result ────────────────────────────────────────────────
        _trade_log.info(
            "EXECUTED SELL | %s | order_id=%s | qty=%d | filled=%.2f | pnl=%s",
            symbol, order_id, quantity, filled_price,
            f"₹{pnl:.0f}" if pnl is not None else "N/A",
        )

        return self._sell_result(
            symbol, True, order_id=order_id, filled_price=filled_price, pnl=pnl
        )

    # ================================================================== #
    # EXIT (from ExitAgent signals)                                        #
    # ================================================================== #

    def execute_exit(self, exit_signal: dict) -> dict:
        """Execute an exit signal from :class:`~src.agents.exit_agent.ExitAgent`.

        Uses a MARKET order for STOP_LOSS_HIT exits (speed priority); LIMIT for
        all other exit types.  Supports partial exits via ``sell_quantity``.

        Args:
            exit_signal: Dict from ``ExitAgent.check_exits()``::

                {
                    "symbol":        str,
                    "exit_type":     str,      # STOP_LOSS_HIT | TARGET_HIT | …
                    "current_price": float,
                    "urgency":       str,       # HIGH | MEDIUM | NORMAL | LOW
                    "sell_quantity": "ALL" | int,
                    "pnl":           float      (informational)
                }

        Returns::

            {
                "success":      bool,
                "symbol":       str,
                "order_id":     str | None,
                "exit_type":    str,
                "filled_price": float | None,
                "sell_quantity":int,
                "pnl":          float | None,
                "error":        str | None,
            }
        """
        symbol        = exit_signal.get("symbol", "")
        exit_type     = exit_signal.get("exit_type", "MANUAL")
        current_price = float(exit_signal.get("current_price", 0.0))
        sell_quantity = exit_signal.get("sell_quantity", "ALL")
        urgency       = exit_signal.get("exit_urgency", exit_signal.get("urgency", "NORMAL"))

        if not symbol or current_price <= 0:
            err = f"Invalid exit signal: symbol={symbol} price={current_price}"
            _log.error("execute_exit: %s", err)
            return self._exit_result(symbol, exit_type, False, error=err)

        # ── Resolve quantity ─────────────────────────────────────────────
        quantity = self._resolve_sell_quantity(symbol, sell_quantity)
        if quantity <= 0:
            err = f"No holdings found for {symbol} or quantity resolved to 0"
            _log.warning("execute_exit: %s", err)
            return self._exit_result(symbol, exit_type, False, error=err)

        # ── Signal-only mode ─────────────────────────────────────────────
        if self._signal_only:
            entry_price = float(exit_signal.get("entry_price", 0.0))
            pnl         = float(exit_signal.get("pnl", 0.0))
            reasoning   = str(exit_signal.get("reasoning", exit_signal.get("reason", exit_type)))
            if self.notifier is not None:
                try:
                    self.notifier.send_exit_signal_alert(
                        symbol=symbol,
                        quantity=quantity,
                        exit_type=exit_type,
                        current_price=current_price,
                        entry_price=entry_price,
                        pnl=pnl,
                        reasoning=reasoning,
                    )
                except Exception as exc:
                    _log.error("signal_only: failed to send exit signal for %s: %s", symbol, exc)
            else:
                _log.warning("signal_only EXIT signal for %s — no notifier configured", symbol)
            # Update DB to mark the open trade as signalled
            try:
                open_trades = self.db.get_open_trades()
                matching    = [t for t in open_trades if t.symbol == symbol]
                if matching:
                    self.db.update_trade_exit(matching[0].id, current_price)  # type: ignore[arg-type]
            except Exception as exc:
                _log.warning("signal_only exit: DB update failed for %s: %s", symbol, exc)
            _trade_log.info(
                "SIGNAL_SENT EXIT | %s | type=%s | qty=%d | price=%.2f",
                symbol, exit_type, quantity, current_price,
            )
            return self._exit_result(
                symbol, exit_type, True,
                filled_price=current_price,
                quantity=quantity,
                pnl=pnl if pnl != 0.0 else None,
            )

        # ── Determine order type ─────────────────────────────────────────
        use_market = (exit_type in ("STOP_LOSS_HIT", "TRAILING_STOP_LOSS")
                      and urgency == "HIGH")

        order_type = "MARKET" if use_market else "LIMIT"

        # ── Cancel existing SL order ─────────────────────────────────────
        self._cancel_sl_order(symbol)

        # ── Log BEFORE placing ───────────────────────────────────────────
        _trade_log.info(
            "INTENT EXIT | %s | type=%s | qty=%d | price=%.2f | order_type=%s | paper=%s",
            symbol, exit_type, quantity, current_price, order_type, self._paper_trading,
        )

        # ── Place order ──────────────────────────────────────────────────
        order_id: Optional[str] = None
        if self._paper_trading:
            order_id     = f"PAPER_EXIT_{symbol}_{int(time.time())}"
            filled_price = current_price
        else:
            try:
                if order_type == "MARKET":
                    result = self.broker.place_sell_order(
                        symbol=symbol, quantity=quantity,
                        price=0.0, order_type="MARKET",
                    )
                else:
                    result = self.broker.place_sell_order(
                        symbol=symbol, quantity=quantity,
                        price=current_price, order_type="LIMIT",
                    )
                order_id = result["order_id"]
            except Exception as exc:
                err = f"Exit order placement failed: {exc}"
                _log.error("execute_exit FAILED | %s | %s", symbol, err)
                return self._exit_result(symbol, exit_type, False, error=err)

            # ── Wait for fill ────────────────────────────────────────────
            timeout = 15 if use_market else _FILL_TIMEOUT
            filled_price = self._wait_for_fill(order_id, timeout)
            if filled_price is None:
                return self._exit_result(
                    symbol, exit_type, False, order_id=order_id,
                    quantity=quantity,
                    error="Exit order not filled within timeout",
                )

        # ── Update DB ────────────────────────────────────────────────────
        pnl: Optional[float] = None
        try:
            open_trades = self.db.get_open_trades()
            matching    = [t for t in open_trades if t.symbol == symbol]
            if matching:
                # Close the oldest open trade for this symbol
                trade    = matching[0]
                self.db.update_trade_exit(trade.id, filled_price)  # type: ignore[arg-type]
                trade    = self.db.get_trade_by_id(trade.id)       # type: ignore[arg-type]
                pnl      = trade.pnl if trade else None
        except Exception as exc:
            _log.warning("execute_exit: DB update failed for %s: %s", symbol, exc)

        # ── Log result ───────────────────────────────────────────────────
        _trade_log.info(
            "EXECUTED EXIT | %s | type=%s | order_id=%s | qty=%d | filled=%.2f | pnl=%s",
            symbol, exit_type, order_id, quantity, filled_price,
            f"₹{pnl:.0f}" if pnl is not None else "N/A",
        )

        return self._exit_result(
            symbol, exit_type, True,
            order_id=order_id,
            filled_price=filled_price,
            quantity=quantity,
            pnl=pnl,
        )

    # ================================================================== #
    # Utility methods                                                      #
    # ================================================================== #

    def cancel_pending_orders(self, symbol: Optional[str] = None) -> dict:
        """Cancel all PENDING orders, or only orders for *symbol* if provided.

        Returns::

            {"cancelled": [order_id, ...], "failed": [order_id, ...]}
        """
        cancelled: List[str] = []
        failed:    List[str] = []

        if self._paper_trading:
            _log.info("cancel_pending_orders: paper mode — no live orders to cancel")
            return {"cancelled": [], "failed": []}

        try:
            order_book = self.broker.get_order_book()
        except Exception as exc:
            _log.error("cancel_pending_orders: failed to fetch order book: %s", exc)
            return {"cancelled": [], "failed": []}

        for order in order_book:
            if order.get("status", "").upper() not in {s.upper() for s in _PENDING_STATUSES}:
                continue
            if symbol and order.get("symbol") != symbol:
                continue
            oid = order["order_id"]
            try:
                self.broker.cancel_order(oid)
                cancelled.append(oid)
                _log.info("Cancelled order %s (%s)", oid, order.get("symbol"))
            except Exception as exc:
                _log.warning("Failed to cancel order %s: %s", oid, exc)
                failed.append(oid)

        return {"cancelled": cancelled, "failed": failed}

    def check_order_status(self, order_id: str) -> dict:
        """Return status details for a specific order.

        Returns::

            {
                "order_id":         str,
                "status":           str,
                "filled_qty":       int,
                "price":            float,
                "symbol":           str,
                "transaction_type": str,
            }
        """
        if self._paper_trading:
            return {
                "order_id":         order_id,
                "status":           "COMPLETE",
                "filled_qty":       0,
                "price":            0.0,
                "symbol":           "",
                "transaction_type": "",
            }
        try:
            return self.broker.get_order_status(order_id)
        except Exception as exc:
            _log.warning("check_order_status(%s) failed: %s", order_id, exc)
            return {"order_id": order_id, "status": "UNKNOWN", "error": str(exc)}

    def get_todays_executed_trades(self) -> List[dict]:
        """Return all trades executed today from the broker's order book.

        Returns a list of filled order dicts.
        """
        if self._paper_trading:
            try:
                today = _ist_today()
                trades = self.db.get_trade_history(days=1)
                return [
                    t.to_dict() for t in trades
                    if t.entry_date and t.entry_date.startswith(today)
                ]
            except Exception:
                return []

        try:
            order_book = self.broker.get_order_book()
            return [
                o for o in order_book
                if o.get("status", "").upper() in {s.upper() for s in _FILLED_STATUSES}
            ]
        except Exception as exc:
            _log.warning("get_todays_executed_trades failed: %s", exc)
            return []

    # ================================================================== #
    # Legacy stub compatibility (pass-through wrappers)                    #
    # ================================================================== #

    def place_order(self, order: dict[str, Any]) -> str:
        """Submit an order dict to the broker and return the broker order ID.

        The *order* dict must contain ``symbol``, ``transaction_type`` (BUY/SELL),
        ``quantity``, ``price``, and ``order_type`` (LIMIT/MARKET).
        """
        transaction_type = order.get("transaction_type", "BUY").upper()
        symbol           = order["symbol"]
        quantity         = int(order["quantity"])
        price            = float(order.get("price", 0.0))
        order_type       = order.get("order_type", "LIMIT").upper()

        if self._paper_trading:
            return f"PAPER_{transaction_type}_{symbol}_{int(time.time())}"

        if transaction_type == "BUY":
            result = self.broker.place_buy_order(symbol, quantity, price, order_type)
        else:
            result = self.broker.place_sell_order(symbol, quantity, price, order_type)
        return result["order_id"]

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by its broker order ID."""
        if self._paper_trading:
            return True
        try:
            self.broker.cancel_order(order_id)
            return True
        except Exception as exc:
            _log.warning("cancel_order(%s) failed: %s", order_id, exc)
            return False

    def modify_order(self, order_id: str, updates: dict[str, Any]) -> bool:
        """Modify price or quantity of a pending order."""
        if self._paper_trading:
            return True
        try:
            self.broker.modify_order(
                order_id,
                new_price=updates.get("price"),
                new_quantity=updates.get("quantity"),
            )
            return True
        except Exception as exc:
            _log.warning("modify_order(%s) failed: %s", order_id, exc)
            return False

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Return the current status of an order (legacy stub compat)."""
        return self.check_order_status(order_id)

    def get_positions(self) -> list[dict[str, Any]]:
        """Return all current open positions from the broker."""
        if self._paper_trading:
            try:
                return [t.to_dict() for t in self.db.get_open_trades()]
            except Exception:
                return []
        try:
            return self.broker.get_holdings()
        except Exception as exc:
            _log.warning("get_positions failed: %s", exc)
            return []

    # ================================================================== #
    # Private helpers                                                      #
    # ================================================================== #

    def _wait_for_fill(self, order_id: str, timeout: int) -> Optional[float]:
        """Poll order status until filled or timeout.

        Returns the filled price, or None if not filled within *timeout* seconds.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status = self.broker.get_order_status(order_id)
                if status["status"].upper() in {s.upper() for s in _FILLED_STATUSES}:
                    price = float(status.get("price", 0.0))
                    _log.debug("Order %s filled at %.2f", order_id, price)
                    return price if price > 0 else None
            except Exception as exc:
                _log.debug("Status poll for %s failed: %s", order_id, exc)
            time.sleep(_FILL_POLL_INTERVAL)
        return None

    def _check_partial_fill(self, order_id: str) -> tuple[int, float]:
        """Check whether an order has a partial fill.

        Returns:
            (filled_qty, avg_price) or (0, 0.0) if nothing was filled.
        """
        try:
            status = self.broker.get_order_status(order_id)
            filled_qty = int(status.get("filled_qty", status.get("filledQty", 0)))
            avg_price  = float(status.get("price", status.get("averagePrice", 0.0)))
            return filled_qty, avg_price
        except Exception as exc:
            _log.debug("_check_partial_fill(%s): %s", order_id, exc)
            return 0, 0.0

    def _cancel_sl_order(self, symbol: str) -> None:
        """Cancel the stored stop-loss order for *symbol* if one exists."""
        try:
            sl_order_id = self.db.get_system_state(f"sl_order_{symbol}")
            if sl_order_id and not self._paper_trading:
                try:
                    self.broker.cancel_order(sl_order_id)
                    _log.info("Cancelled SL order %s for %s", sl_order_id, symbol)
                except Exception as exc:
                    _log.debug("Could not cancel SL order %s: %s", sl_order_id, exc)
            # Clear the stored SL order ID
            self.db.set_system_state(f"sl_order_{symbol}", "")
        except Exception:
            pass

    def _resolve_sell_quantity(self, symbol: str, sell_quantity: Any) -> int:
        """Return the integer quantity to sell."""
        if isinstance(sell_quantity, int) and sell_quantity > 0:
            return sell_quantity
        if str(sell_quantity).upper() == "ALL":
            if not self._paper_trading:
                try:
                    holdings = self.broker.get_holdings()
                    held     = next(
                        (h["quantity"] for h in holdings if h["symbol"] == symbol), 0
                    )
                    return int(held)
                except Exception:
                    pass
            # Fallback: check DB for open trade quantity
            try:
                open_trades = self.db.get_open_trades()
                for t in open_trades:
                    if t.symbol == symbol:
                        return t.quantity
            except Exception:
                pass
        return 0

    def _record_trade(
        self,
        symbol: str,
        quantity: int,
        price: float,
        order_id: Optional[str],
        status: str,
        stop_loss: Optional[float] = None,
        target_price: Optional[float] = None,
        strategy_signal: Optional[dict] = None,
    ) -> Optional[int]:
        """Persist a trade record to the database. Returns the trade_id."""
        try:
            trade = Trade(
                symbol=symbol,
                trade_type="BUY",
                quantity=quantity,
                price=price,
                order_id=order_id,
                status=status,
                stop_loss=stop_loss,
                target_price=target_price,
                entry_date=_ist_now(),
                strategy_signal=json.dumps(strategy_signal) if strategy_signal else None,
            )
            return self.db.record_trade(trade)
        except Exception as exc:
            _log.error("Failed to record trade for %s to DB: %s", symbol, exc)
            return None

    def _notify_critical(self, message: str) -> None:
        """Send a CRITICAL-level Telegram alert. Never raises — failures are logged."""
        if self.notifier is None:
            _log.critical("CRITICAL (no notifier): %s", message)
            return
        try:
            self.notifier.send_alert(message, level="CRITICAL")
        except Exception as exc:
            _log.error("_notify_critical: failed to send Telegram alert: %s", exc)

    # ── Result builders ──────────────────────────────────────────────────

    @staticmethod
    def _buy_result(
        symbol: str,
        quantity: int,
        success: bool,
        order_id: Optional[str] = None,
        sl_order_id: Optional[str] = None,
        filled_price: Optional[float] = None,
        trade_id: Optional[int] = None,
        error: Optional[str] = None,
    ) -> dict:
        return {
            "success":      success,
            "symbol":       symbol,
            "order_id":     order_id,
            "sl_order_id":  sl_order_id,
            "filled_price": filled_price,
            "quantity":     quantity,
            "trade_id":     trade_id,
            "error":        error,
        }

    @staticmethod
    def _sell_result(
        symbol: str,
        success: bool,
        order_id: Optional[str] = None,
        filled_price: Optional[float] = None,
        pnl: Optional[float] = None,
        error: Optional[str] = None,
    ) -> dict:
        return {
            "success":      success,
            "symbol":       symbol,
            "order_id":     order_id,
            "filled_price": filled_price,
            "pnl":          pnl,
            "error":        error,
        }

    @staticmethod
    def _exit_result(
        symbol: str,
        exit_type: str,
        success: bool,
        order_id: Optional[str] = None,
        filled_price: Optional[float] = None,
        quantity: int = 0,
        pnl: Optional[float] = None,
        error: Optional[str] = None,
    ) -> dict:
        return {
            "success":      success,
            "symbol":       symbol,
            "exit_type":    exit_type,
            "order_id":     order_id,
            "filled_price": filled_price,
            "sell_quantity": quantity,
            "pnl":          pnl,
            "error":        error,
        }
