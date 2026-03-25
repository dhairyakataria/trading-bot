"""Exit Agent — monitors open positions and generates exit signals.

Exit types (in priority order):

    STOP_LOSS_HIT          — price <= stop_loss (non-negotiable, HIGH urgency)
    TRAILING_STOP_LOSS     — trailing stop triggered (HIGH urgency)
    TARGET_HIT             — partial (50%) or full (100%) target reached (NORMAL)
    TECHNICAL_DETERIORATION — 2+ bearish technical signals (MEDIUM urgency)
    TIME_BASED_EXIT        — held too long without hitting target (LOW urgency)
    NEWS_TRIGGERED_EXIT    — negative news confirmed by research agent (HIGH)

The agent does NOT execute trades — it produces signals consumed by the
Execution Agent.  Every urgency-HIGH signal should be acted on immediately,
not queued for the next orchestrator cycle.

All exit signals are logged at WARNING level.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, List, Optional
from zoneinfo import ZoneInfo

from src.utils.logger import get_logger

IST = ZoneInfo("Asia/Kolkata")
_log = get_logger("agents.exit_agent")

# ── Trailing stop constants ──────────────────────────────────────────────────
_TRAILING_ACTIVATION_PCT = 0.02   # Trailing stop activates after 2% profit
_TRAILING_ATR_MULTIPLIER = 1.5    # Trailing stop = highest_price − 1.5 × ATR

# ── Time-based exit thresholds (trading days) ────────────────────────────────
_TIME_REVIEW_DAYS      = 15
_TIME_FORCE_EXIT_DAYS  = 20


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ist_today() -> date:
    return datetime.now(IST).date()


def _business_days_between(start_str: str, end_date: date) -> int:
    """Count Mon–Fri business days between *start_str* (ISO) and *end_date* (exclusive)."""
    if not start_str:
        return 0
    try:
        start = datetime.fromisoformat(start_str).date()
    except (ValueError, TypeError):
        return 0
    count = 0
    current = start
    while current < end_date:
        if current.weekday() < 5:   # Mon=0 … Fri=4
            count += 1
        current += timedelta(days=1)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# ExitAgent
# ─────────────────────────────────────────────────────────────────────────────

class ExitAgent:
    """Monitors open positions and generates exit signals.

    Args:
        config:               Application config object (``config.get(...)``).
        broker_client:        :class:`~src.broker.angel_one.AngelOneClient`.
        technical_indicators: :class:`~src.tools.technical_indicators.TechnicalIndicators`.
        research_agent:       :class:`~src.agents.research_agent.ResearchAgent`.
        db_manager:           :class:`~src.database.db_manager.DatabaseManager`.
        llm_router:           :class:`~src.llm.router.LLMRouter` (reserved for
                              future direct LLM reasoning; LLM is currently
                              invoked via ``research_agent``).
    """

    def __init__(
        self,
        config: Any,
        broker_client: Any,
        technical_indicators: Any,
        research_agent: Any,
        db_manager: Any,
        llm_router: Any,
    ) -> None:
        self.config         = config
        self.broker         = broker_client
        self.ti             = technical_indicators
        self.research_agent = research_agent
        self.db             = db_manager
        self.llm_router     = llm_router

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def check_exits(
        self,
        morning_briefing_data: Optional[dict] = None,
    ) -> List[dict]:
        """Run all exit checks on every open position.

        Called every scan cycle (every 30–60 min during market hours).

        Args:
            morning_briefing_data: Output of
                :meth:`~src.agents.research_agent.ResearchAgent.morning_briefing`.
                Used to decide whether to trigger news-based exits.

        Returns:
            List of exit signal dicts.  Each dict has the shape::

                {
                    "symbol":        "TCS",
                    "action":        "SELL",
                    "exit_type":     "TARGET_HIT",
                    "current_price": 3965.00,
                    "entry_price":   3850.50,
                    "pnl":           114.50,
                    "pnl_pct":       2.97,
                    "holding_days":  5,
                    "urgency":       "NORMAL",
                    "reasoning":     "Price hit target_1 at ₹3,960. Book profits.",
                    "sell_quantity": "ALL",   # or int for partial exits
                }

            Urgency-HIGH signals should be executed immediately.
        """
        open_trades = self.db.get_open_trades()
        if not open_trades:
            _log.debug("check_exits: no open trades")
            return []

        # Fetch LTP for every symbol once; skip any that fail.
        prices: dict[str, float] = {}
        for trade in open_trades:
            if trade.symbol in prices:
                continue
            try:
                prices[trade.symbol] = float(self.broker.get_ltp(trade.symbol)["ltp"])
            except Exception as exc:
                _log.error("check_exits: LTP failed for %s: %s", trade.symbol, exc)

        # Update trailing highs (uses cached prices where available).
        self.update_trailing_stops(open_trades, _prices=prices)

        signals: List[dict] = []
        signalled_symbols: set = set()  # prevent >1 exit signal per symbol per cycle
        today = _ist_today()

        for trade in open_trades:
            symbol = trade.symbol
            current_price = prices.get(symbol)
            if current_price is None:
                continue   # LTP unavailable — skip this symbol this cycle

            entry_price   = float(trade.price)
            stop_loss     = float(trade.stop_loss) if trade.stop_loss else None
            holding_days  = _business_days_between(trade.entry_date or "", today)
            target_1, target_2 = self._parse_targets(trade)

            trade_ctx: dict = {
                "id":           trade.id,
                "symbol":       symbol,
                "entry_price":  entry_price,
                "stop_loss":    stop_loss,
                "target_1":     target_1,
                "target_2":     target_2,
                "entry_date":   trade.entry_date,
                "quantity":     trade.quantity,
                "holding_days": holding_days,
            }

            # ── 1. Stop-loss (non-negotiable — skip all further checks) ──────
            sig = self._check_stop_loss(trade_ctx, current_price)
            if sig:
                _log.warning(
                    "EXIT SIGNAL [STOP_LOSS_HIT] %s @ ₹%.2f (entry=₹%.2f sl=₹%.2f)",
                    symbol, current_price, entry_price, stop_loss or 0,
                )
                signals.append(sig)
                signalled_symbols.add(symbol)
                continue

            # ── 2. Trailing stop ─────────────────────────────────────────────
            sig = self._check_trailing_stop(trade_ctx, current_price)
            if sig:
                _log.warning(
                    "EXIT SIGNAL [TRAILING_STOP_LOSS] %s @ ₹%.2f", symbol, current_price
                )
                signals.append(sig)
                signalled_symbols.add(symbol)
                continue

            # ── 3. Target hit (partial exits allowed — don't skip further checks)
            sig = self._check_targets(trade_ctx, current_price)
            if sig:
                _log.warning(
                    "EXIT SIGNAL [TARGET_HIT] %s @ ₹%.2f", symbol, current_price
                )
                signals.append(sig)
                signalled_symbols.add(symbol)
                # Don't continue — also check technical deterioration on partial targets.

            # Skip lower-priority checks if a signal was already emitted for this symbol
            if symbol in signalled_symbols:
                continue

            # ── 4. Technical deterioration ────────────────────────────────────
            df = self._get_ohlcv(symbol)
            if df is not None and not df.empty:
                tech_sig = self._check_technical_deterioration(trade_ctx, df)
                if tech_sig:
                    _log.warning(
                        "EXIT SIGNAL [TECHNICAL_DETERIORATION] %s @ ₹%.2f",
                        symbol, current_price,
                    )
                    self._enrich_signal(tech_sig, current_price, trade.quantity)
                    signals.append(tech_sig)
                    signalled_symbols.add(symbol)
                    continue

            # ── 5. Time-based exit ────────────────────────────────────────────
            if symbol not in signalled_symbols:
                time_sig = self._check_time_based(trade_ctx)
                if time_sig:
                    _log.warning(
                        "EXIT SIGNAL [TIME_BASED_EXIT] %s (day %d)", symbol, holding_days
                    )
                    self._enrich_signal(time_sig, current_price, trade.quantity)
                    signals.append(time_sig)
                    signalled_symbols.add(symbol)

            # ── 6. News-triggered (LLM — only when anomalies detected) ────────
            if symbol not in signalled_symbols and self._should_check_news(trade_ctx, morning_briefing_data):
                news_sig = self._check_news_triggered(trade_ctx, morning_briefing_data)
                if news_sig:
                    _log.warning(
                        "EXIT SIGNAL [NEWS_TRIGGERED_EXIT] %s @ ₹%.2f",
                        symbol, current_price,
                    )
                    self._enrich_signal(news_sig, current_price, trade.quantity)
                    signals.append(news_sig)
                    signalled_symbols.add(symbol)

        if signals:
            _log.warning(
                "check_exits: %d exit signal(s) for %d open positions",
                len(signals), len(open_trades),
            )
        return signals

    def update_trailing_stops(
        self,
        holdings: List[Any],
        _prices: Optional[dict[str, float]] = None,
    ) -> None:
        """Update highest-price-seen for each holding and persist to system_state.

        Called every scan cycle before exit checks so trailing stop values
        are always current.  Persisted values survive restarts.

        Args:
            holdings: List of :class:`~src.database.models.Trade` objects *or*
                      plain dicts with at least ``symbol`` / ``entry_price`` (or
                      ``price``) fields.
            _prices:  Optional pre-fetched LTP cache (avoids duplicate API calls
                      when called from :meth:`check_exits`).
        """
        for holding in holdings:
            if hasattr(holding, "symbol"):
                symbol      = holding.symbol
                entry_price = float(holding.price)
            else:
                symbol      = holding["symbol"]
                entry_price = float(holding.get("entry_price", holding.get("price", 0)))

            # Resolve current price — use cache if available, else hit broker.
            if _prices and symbol in _prices:
                current_price = _prices[symbol]
            else:
                try:
                    current_price = float(self.broker.get_ltp(symbol)["ltp"])
                except Exception as exc:
                    _log.debug(
                        "update_trailing_stops: LTP failed for %s: %s", symbol, exc
                    )
                    continue

            # Trailing stop only activates after 2% profit.
            if current_price < entry_price * (1 + _TRAILING_ACTIVATION_PCT):
                continue

            key = f"trailing_high_{symbol}"
            prev_high_str = self.db.get_system_state(key)
            prev_high = float(prev_high_str) if prev_high_str else entry_price

            if current_price > prev_high:
                self.db.set_system_state(key, str(current_price))
                _log.debug(
                    "update_trailing_stops: %s new high ₹%.2f (was ₹%.2f)",
                    symbol, current_price, prev_high,
                )

    def get_current_atr(self, symbol: str) -> float:
        """Return the 14-day ATR for *symbol*.

        Falls back to ``1.0`` if data is unavailable so trailing-stop
        calculations never receive ``None``.
        """
        df = self._get_ohlcv(symbol)
        if df is None or df.empty:
            _log.warning(
                "get_current_atr: no OHLCV for %s — defaulting to 1.0", symbol
            )
            return 1.0
        result = self.ti.calculate_atr(df)
        if "error" in result:
            _log.warning(
                "get_current_atr: ATR error for %s: %s — defaulting to 1.0",
                symbol, result["error"],
            )
            return 1.0
        return float(result["atr"])

    def get_exit_summary(self) -> dict:
        """Return a Telegram-ready summary of all pending exit signals.

        Returns:
            Dict with keys ``total_signals``, ``high_urgency``,
            ``medium_urgency``, ``normal_urgency``, ``low_urgency``,
            ``signals``.
        """
        signals = self.check_exits()
        return {
            "total_signals":  len(signals),
            "high_urgency":   sum(1 for s in signals if s.get("urgency") == "HIGH"),
            "medium_urgency": sum(1 for s in signals if s.get("urgency") == "MEDIUM"),
            "normal_urgency": sum(1 for s in signals if s.get("urgency") == "NORMAL"),
            "low_urgency":    sum(1 for s in signals if s.get("urgency") == "LOW"),
            "signals":        signals,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Individual exit checks
    # ──────────────────────────────────────────────────────────────────────────

    def _check_stop_loss(
        self, trade: dict, current_price: float
    ) -> Optional[dict]:
        """Return a ``STOP_LOSS_HIT`` signal if ``current_price <= stop_loss``.

        Stop-loss is sacrosanct — no LLM reasoning, no delay.
        """
        stop_loss = trade.get("stop_loss")
        if stop_loss is None or current_price > stop_loss:
            return None

        entry_price = trade["entry_price"]
        qty         = trade["quantity"]
        pnl_pct     = round((current_price - entry_price) / entry_price * 100, 2)
        pnl         = round((current_price - entry_price) * qty, 2)

        return {
            "symbol":        trade["symbol"],
            "action":        "SELL",
            "exit_type":     "STOP_LOSS_HIT",
            "current_price": current_price,
            "entry_price":   entry_price,
            "pnl":           pnl,
            "pnl_pct":       pnl_pct,
            "holding_days":  trade["holding_days"],
            "urgency":       "HIGH",
            "reasoning": (
                f"Stop-loss breached. Price ₹{current_price:,.2f} ≤ "
                f"stop-loss ₹{stop_loss:,.2f}. Exit immediately — non-negotiable."
            ),
            "sell_quantity": "ALL",
        }

    def _check_targets(
        self, trade: dict, current_price: float
    ) -> Optional[dict]:
        """Return a ``TARGET_HIT`` signal (full or partial) if a target is reached.

        * ``target_2`` hit → sell 100 % of position.
        * ``target_1`` hit → sell 50 % of position (book partial profits).
        """
        target_1 = trade.get("target_1")
        target_2 = trade.get("target_2")
        entry_price = trade["entry_price"]
        qty         = trade["quantity"]

        # Full exit at target_2.
        if target_2 and current_price >= target_2:
            pnl_pct = round((current_price - entry_price) / entry_price * 100, 2)
            pnl     = round((current_price - entry_price) * qty, 2)
            return {
                "symbol":        trade["symbol"],
                "action":        "SELL",
                "exit_type":     "TARGET_HIT",
                "current_price": current_price,
                "entry_price":   entry_price,
                "pnl":           pnl,
                "pnl_pct":       pnl_pct,
                "holding_days":  trade["holding_days"],
                "urgency":       "NORMAL",
                "reasoning": (
                    f"Price ₹{current_price:,.2f} reached target_2 ₹{target_2:,.2f}. "
                    "Book 100 % of position."
                ),
                "sell_quantity": "ALL",
            }

        # Partial exit at target_1.
        if target_1 and current_price >= target_1:
            sell_qty = max(1, qty // 2)
            pnl_pct  = round((current_price - entry_price) / entry_price * 100, 2)
            pnl      = round((current_price - entry_price) * sell_qty, 2)
            return {
                "symbol":        trade["symbol"],
                "action":        "SELL",
                "exit_type":     "TARGET_HIT",
                "current_price": current_price,
                "entry_price":   entry_price,
                "pnl":           pnl,
                "pnl_pct":       pnl_pct,
                "holding_days":  trade["holding_days"],
                "urgency":       "NORMAL",
                "reasoning": (
                    f"Price ₹{current_price:,.2f} reached target_1 ₹{target_1:,.2f}. "
                    "Book 50 % of position to lock in profits."
                ),
                "sell_quantity": sell_qty,
            }

        return None

    def _check_trailing_stop(
        self, trade: dict, current_price: float
    ) -> Optional[dict]:
        """Return a ``TRAILING_STOP_LOSS`` signal if the trailing stop is triggered.

        The trailing stop only activates once the position is 2 % in profit.
        It locks in gains by following the price up: any new high updates the
        reference, and the stop floor is ``highest_price − 1.5 × ATR``.
        """
        symbol = trade["symbol"]

        # Trailing stop is active only when a high has been recorded by
        # update_trailing_stops (which itself gates on the 2 % threshold).
        key = f"trailing_high_{symbol}"
        prev_high_str = self.db.get_system_state(key)
        if not prev_high_str:
            return None   # Not yet activated — no high ever recorded.

        prev_high = float(prev_high_str)
        atr       = self.get_current_atr(symbol)

        # Guard: ATR must be at least 0.5% of the peak price to avoid
        # collapsing the trailing stop on data-unavailable fallbacks.
        min_atr = prev_high * 0.005
        if atr < min_atr:
            _log.warning(
                "_check_trailing_stop: %s ATR %.4f below floor %.4f — using floor",
                symbol, atr, min_atr,
            )
            atr = min_atr

        trailing_stop = prev_high - (_TRAILING_ATR_MULTIPLIER * atr)

        if current_price > trailing_stop:
            return None   # Still above the floor.

        entry_price = trade["entry_price"]
        qty         = trade["quantity"]
        pnl_pct     = round((current_price - entry_price) / entry_price * 100, 2)
        pnl         = round((current_price - entry_price) * qty, 2)

        return {
            "symbol":        symbol,
            "action":        "SELL",
            "exit_type":     "TRAILING_STOP_LOSS",
            "current_price": current_price,
            "entry_price":   entry_price,
            "pnl":           pnl,
            "pnl_pct":       pnl_pct,
            "holding_days":  trade["holding_days"],
            "urgency":       "HIGH",
            "reasoning": (
                f"Trailing stop triggered. Price ₹{current_price:,.2f} fell below "
                f"trailing stop ₹{trailing_stop:,.2f} "
                f"(high ₹{prev_high:,.2f} − 1.5 × ATR ₹{atr:,.2f}). "
                "Locking in profits."
            ),
            "sell_quantity": "ALL",
        }

    def _check_time_based(self, trade: dict) -> Optional[dict]:
        """Return a ``TIME_BASED_EXIT`` signal when the position is held too long.

        * ≥ 15 trading days → review (still LOW urgency).
        * ≥ 20 trading days → force exit (swing thesis has expired).

        ``current_price``, ``pnl``, and ``pnl_pct`` are filled in by the
        caller (:meth:`_enrich_signal`) once the LTP is known.
        """
        holding_days = trade["holding_days"]
        if holding_days < _TIME_REVIEW_DAYS:
            return None

        if holding_days >= _TIME_FORCE_EXIT_DAYS:
            reasoning = (
                f"Held for {holding_days} trading days (≥ {_TIME_FORCE_EXIT_DAYS}). "
                "Swing trade thesis has not played out — exit to free capital."
            )
        else:
            reasoning = (
                f"Held for {holding_days} trading days (≥ {_TIME_REVIEW_DAYS}). "
                "Swing trade should have worked by now — review and consider exiting."
            )

        return {
            "symbol":        trade["symbol"],
            "action":        "SELL",
            "exit_type":     "TIME_BASED_EXIT",
            "current_price": None,   # enriched by caller
            "entry_price":   trade["entry_price"],
            "pnl":           None,
            "pnl_pct":       None,
            "holding_days":  holding_days,
            "urgency":       "LOW",
            "reasoning":     reasoning,
            "sell_quantity": "ALL",
        }

    def _check_technical_deterioration(
        self, trade: dict, df: Any
    ) -> Optional[dict]:
        """Return a ``TECHNICAL_DETERIORATION`` signal if ≥ 2 bearish signals fire.

        Bearish conditions checked:
        1. RSI > 75 — overbought, take profit.
        2. MACD bearish crossover — momentum fading.
        3. Price below 20-day EMA — short-term trend broken.
        4. Volume spike on a down day — potential panic selling.

        Any **two** of the above triggers an exit signal.

        ``current_price``, ``pnl``, and ``pnl_pct`` are filled in by the
        caller (:meth:`_enrich_signal`).
        """
        triggers: List[str] = []

        # 1. RSI overbought.
        rsi = self.ti.calculate_rsi(df)
        if "error" not in rsi:
            val = rsi.get("value", 0)
            if val > 75:
                triggers.append(f"RSI overbought at {val:.1f} (> 75)")

        # 2. MACD bearish crossover.
        macd = self.ti.calculate_macd(df)
        if "error" not in macd and macd.get("signal") == "BEARISH_CROSSOVER":
            triggers.append("MACD bearish crossover — momentum fading")

        # 3. Price below 20-day EMA.
        ema = self.ti.calculate_ema(df, periods=[20])
        if "error" not in ema and ema.get("price_vs_ema_20") == "BELOW":
            triggers.append(f"Price below 20-day EMA (₹{ema.get('ema_20', 'N/A')})")

        # 4. Volume spike on a down day.
        vol = self.ti.calculate_volume_analysis(df)
        if "error" not in vol and vol.get("signal") == "HIGH_VOLUME":
            try:
                if (
                    len(df) >= 2
                    and float(df["close"].iloc[-1]) < float(df["close"].iloc[-2])
                ):
                    ratio = vol.get("volume_ratio", 1.0)
                    triggers.append(
                        f"Volume spike ({ratio:.1f}× avg) on down day — panic selling"
                    )
            except Exception:
                pass

        if len(triggers) < 2:
            return None

        return {
            "symbol":        trade["symbol"],
            "action":        "SELL",
            "exit_type":     "TECHNICAL_DETERIORATION",
            "current_price": None,   # enriched by caller
            "entry_price":   trade["entry_price"],
            "pnl":           None,
            "pnl_pct":       None,
            "holding_days":  trade["holding_days"],
            "urgency":       "MEDIUM",
            "reasoning": (
                f"Technical deterioration: {len(triggers)} bearish signal(s) — "
                + "; ".join(triggers)
                + ". Consider exiting to protect capital."
            ),
            "sell_quantity": "ALL",
        }

    def _check_news_triggered(
        self,
        trade: dict,
        morning_briefing_data: Optional[dict],
    ) -> Optional[dict]:
        """Return a ``NEWS_TRIGGERED_EXIT`` signal if research confirms negative news.

        Calls :meth:`~src.agents.research_agent.ResearchAgent.research_stock`
        to verify current news for the held symbol.  An exit is generated only
        when the recommendation is ``"AVOID"``.

        ``current_price``, ``pnl``, and ``pnl_pct`` are filled in by the
        caller (:meth:`_enrich_signal`).
        """
        symbol      = trade["symbol"]
        entry_price = trade["entry_price"]

        context = (
            f"We currently HOLD {symbol} (entered at ₹{entry_price:,.2f}). "
            "Check for negative news, regulatory issues, earnings disappointments, "
            "sector headwinds, or management problems that warrant immediate exit."
        )
        try:
            research = self.research_agent.research_stock(
                symbol=symbol,
                context=context,
                session_type="exit_news_check",
            )
        except Exception as exc:
            _log.warning(
                "_check_news_triggered: research failed for %s: %s", symbol, exc
            )
            return None

        if research.get("recommendation", "").upper() != "AVOID":
            return None

        risks   = research.get("risks", [])
        summary = research.get("research_summary", "Negative news detected.")

        return {
            "symbol":        symbol,
            "action":        "SELL",
            "exit_type":     "NEWS_TRIGGERED_EXIT",
            "current_price": None,   # enriched by caller
            "entry_price":   entry_price,
            "pnl":           None,
            "pnl_pct":       None,
            "holding_days":  trade["holding_days"],
            "urgency":       "HIGH",
            "reasoning": (
                f"Research agent recommends AVOID for {symbol}. "
                f"{summary} Risks: {'; '.join(risks[:3])}."
            ),
            "sell_quantity": "ALL",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Original stub interface (retained for Execution Agent compatibility)
    # ──────────────────────────────────────────────────────────────────────────

    def should_exit(self, position: dict[str, Any], current_price: float) -> bool:
        """Return ``True`` if any rule-based exit check fires for *position*.

        Used by the Execution Agent for a quick yes/no decision without
        generating a full signal dict.
        """
        today = _ist_today()
        trade_ctx: dict = {
            "id":           position.get("id"),
            "symbol":       position.get("symbol", ""),
            "entry_price":  float(position.get("entry_price", position.get("price", 0))),
            "stop_loss":    position.get("stop_loss"),
            "target_1":     position.get("target_1"),
            "target_2":     position.get("target_2"),
            "entry_date":   position.get("entry_date"),
            "quantity":     int(position.get("quantity", 1)),
            "holding_days": _business_days_between(
                position.get("entry_date") or "", today
            ),
        }
        return bool(
            self._check_stop_loss(trade_ctx, current_price)
            or self._check_trailing_stop(trade_ctx, current_price)
            or self._check_targets(trade_ctx, current_price)
            or self._check_time_based(trade_ctx)
        )

    def update_trailing_stop(
        self, position: dict[str, Any], current_price: float
    ) -> float:
        """Update and return the trailing stop level for a single position.

        This is the single-position variant used by the Execution Agent.
        The persisted highest price is updated when *current_price* is a new high.

        Returns:
            The new trailing stop price, or a conservative floor if the
            trailing stop has not yet activated.
        """
        symbol      = position.get("symbol", "")
        entry_price = float(position.get("entry_price", position.get("price", 0)))

        key = f"trailing_high_{symbol}"
        prev_high_str = self.db.get_system_state(key)

        if not prev_high_str:
            # Not yet activated.
            if current_price < entry_price * (1 + _TRAILING_ACTIVATION_PCT):
                return round(entry_price * 0.95, 2)   # Conservative floor.
            # First activation — record this price as the high.
            self.db.set_system_state(key, str(current_price))
            prev_high = current_price
        else:
            prev_high = float(prev_high_str)
            if current_price > prev_high:
                self.db.set_system_state(key, str(current_price))
                prev_high = current_price

        atr = self.get_current_atr(symbol)
        return round(prev_high - (_TRAILING_ATR_MULTIPLIER * atr), 2)

    def generate_exit_order(self, position: dict[str, Any]) -> dict[str, Any]:
        """Build and return a basic exit order dict for the Execution Agent.

        In practice, the Execution Agent uses the richer signal dicts returned
        by :meth:`check_exits`.  This method is provided for compatibility with
        the original stub interface.
        """
        return {
            "symbol":     position.get("symbol"),
            "action":     "SELL",
            "quantity":   position.get("quantity"),
            "order_type": "MARKET",
            "exit_type":  "MANUAL",
            "urgency":    "NORMAL",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_targets(self, trade: Any) -> tuple[Optional[float], Optional[float]]:
        """Extract ``target_1`` and ``target_2`` from a Trade object.

        Lookup order:
        1. ``strategy_signal`` JSON blob (keys ``"target_1"`` / ``"target_2"``).
        2. ``target_price`` as ``target_1``; ``target_2`` derived as
           ``entry + 2 × (target_1 − entry)``.
        """
        target_1: Optional[float] = None
        target_2: Optional[float] = None

        if trade.strategy_signal:
            try:
                sig = json.loads(trade.strategy_signal)
                if "target_1" in sig:
                    target_1 = float(sig["target_1"])
                if "target_2" in sig:
                    target_2 = float(sig["target_2"])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        if target_1 is None and trade.target_price:
            target_1 = float(trade.target_price)

        if target_2 is None and target_1 is not None:
            entry    = float(trade.price)
            target_2 = round(entry + 2.0 * (target_1 - entry), 2)

        return target_1, target_2

    def _get_ohlcv(self, symbol: str, days: int = 60) -> Optional[Any]:
        """Fetch *days* of daily OHLCV data for *symbol*.

        Returns a DataFrame or ``None`` on any error.
        """
        today = _ist_today()
        from_dt   = datetime(today.year, today.month, today.day) - timedelta(days=days)
        from_date = from_dt.strftime("%Y-%m-%d %H:%M")
        to_date   = datetime(today.year, today.month, today.day).strftime("%Y-%m-%d %H:%M")
        try:
            df = self.broker.get_historical_data(
                symbol=symbol,
                interval="ONE_DAY",
                from_date=from_date,
                to_date=to_date,
            )
            return df if (df is not None and not df.empty) else None
        except Exception as exc:
            _log.warning("_get_ohlcv: failed for %s: %s", symbol, exc)
            return None

    def _should_check_news(
        self,
        trade: dict,
        morning_briefing_data: Optional[dict],
    ) -> bool:
        """Return ``True`` if a news-triggered exit check should run.

        Triggers when morning briefing flags this symbol or its sector as risky.
        """
        if not morning_briefing_data:
            return False

        symbol = trade["symbol"]

        if symbol in morning_briefing_data.get("risky_symbols", []):
            return True

        sector_map = morning_briefing_data.get("symbol_sectors", {})
        sector     = sector_map.get(symbol)
        if sector and sector in morning_briefing_data.get("risky_sectors", []):
            return True

        return False

    @staticmethod
    def _enrich_signal(sig: dict, current_price: float, quantity: int) -> None:
        """Fill in ``current_price``, ``pnl``, and ``pnl_pct`` in-place.

        Used for exit types whose check methods don't have access to
        ``current_price`` (time-based, technical deterioration, news-triggered).
        """
        entry = sig.get("entry_price") or 0.0
        sig["current_price"] = current_price
        sig["pnl"]           = round((current_price - entry) * quantity, 2)
        sig["pnl_pct"]       = round(
            (current_price - entry) / entry * 100 if entry else 0.0, 2
        )
