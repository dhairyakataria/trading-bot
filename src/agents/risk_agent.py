"""Risk Manager Agent — safety gatekeeper for all trades.

Every proposed trade passes through :meth:`RiskManager.check_trade` before
execution.  Rules are hard-coded and **cannot** be overridden at runtime —
not even by the LLM layer.

Proposed trade dict expected keys
----------------------------------
symbol      : str   — NSE ticker
trade_type  : str   — "BUY" or "SELL"
quantity    : int   — number of shares
entry_price : float — limit price
stop_loss   : float — mandatory for BUY
target_1    : float — first target, used for R:R check
sector      : str   — used for sector-concentration check (optional)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
_log = logging.getLogger("risk_manager")


# --------------------------------------------------------------------------- #
# Module-level helpers                                                         #
# --------------------------------------------------------------------------- #

def _ist_now() -> datetime:
    return datetime.now(IST)


def _ist_today_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def _this_week_monday_str() -> str:
    """Return YYYY-MM-DD for the most-recent Monday (IST)."""
    today = datetime.now(IST).date()
    monday = today - timedelta(days=today.weekday())  # weekday() 0 = Monday
    return monday.strftime("%Y-%m-%d")


def _check(rule: str, passed: bool, detail: str) -> dict:
    return {"rule": rule, "passed": passed, "detail": detail}


# --------------------------------------------------------------------------- #
# RiskManager                                                                  #
# --------------------------------------------------------------------------- #

class RiskManager:
    """Safety gatekeeper: approves or rejects every proposed trade.

    Hard rules are enforced unconditionally.  Every rule runs even when an
    earlier rule has already failed — the caller receives the full picture.

    Parameters
    ----------
    config:
        Application config object (supports ``config.get("trading", "key")``).
    broker_client:
        Live broker wrapper (``AngelOneClient`` or compatible mock).
    db_manager:
        Database manager (``DatabaseManager`` or compatible mock).
    """

    # Safety constants — some now configurable via config.trading.*
    _MIN_RR_RATIO: float = 1.5         # minimum reward-to-risk ratio
    _MAX_SECTOR_POSITIONS: int = 2     # max stocks from one sector
    _MIN_TRADE_VALUE: float = 1_000.0  # minimum order value (INR)
    _MAX_RISK_PER_TRADE_PCT: float = 0.02  # 2% of capital per trade

    def __init__(self, config: Any, broker_client: Any, db_manager: Any) -> None:
        self.config = config
        self.broker = broker_client
        self.db = db_manager

        # Load configurable limits (with sensible defaults)
        def _cfg(*keys: str, default: Any) -> Any:
            return config.get(*keys, default=default)

        self._capital: float = _cfg("trading", "capital", default=50_000)
        # Config stores percentage as an integer (e.g. 5 means 5%)
        self._max_position_pct: float = _cfg("trading", "max_position_pct", default=5) / 100
        self._max_daily_loss_pct: float = _cfg("trading", "max_daily_loss_pct", default=2) / 100
        self._max_weekly_loss_pct: float = _cfg("trading", "max_weekly_loss_pct", default=5) / 100
        self._max_open_positions: int = int(_cfg("trading", "max_open_positions", default=5))
        # Cap for stop-loss distance (as fraction). Allows ATR-based SLs that often
        # exceed 5% for volatile names. Override via trading.max_stop_loss_pct.
        self._MAX_STOP_LOSS_PCT: float = _cfg("trading", "max_stop_loss_pct", default=8) / 100
        # Paper-trading mode: skip live broker calls; use configured capital.
        self._paper_trading: bool = bool(_cfg("trading", "paper_trading", default=False))

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def check_trade(self, proposed_trade: dict) -> dict:
        """Run all risk rules against a proposed trade.

        All 11 rules execute regardless of earlier failures so the caller
        receives a complete picture of every violation.

        Returns
        -------
        dict with keys:
            approved          : bool
            trade             : the original proposed_trade dict
            checks            : list of per-rule result dicts
            rejection_reason  : None or human-readable summary
            adjusted_quantity : None or int (when MAX_POSITION_SIZE triggered)
            risk_score        : float 0–1 (0 = safest)
        """
        symbol      = proposed_trade.get("symbol", "")
        trade_type  = str(proposed_trade.get("trade_type", "BUY")).upper()
        quantity    = int(proposed_trade.get("quantity", 0))
        entry_price = float(proposed_trade.get("entry_price", 0.0))
        stop_loss   = proposed_trade.get("stop_loss")
        target_1    = proposed_trade.get("target_1")
        sector      = str(proposed_trade.get("sector", "UNKNOWN"))

        is_buy = trade_type == "BUY"

        # Fetch live portfolio state once — shared across all rules
        port = self._get_portfolio_state()
        holdings               = port["holdings"]
        total_portfolio_value  = port["total_portfolio_value"]
        available_cash         = port["available_cash"]
        today_realized_pnl     = port["today_realized_pnl"]
        unrealized_pnl         = port["unrealized_pnl"]
        weekly_pnl             = port["weekly_pnl"]

        checks: list[dict] = []
        adjusted_quantity: Optional[int] = None

        # ---- Rule 1: CAPITAL_AVAILABLE (BUY only) -----------------------
        if is_buy:
            checks.append(
                self._rule_capital_available(quantity, entry_price, available_cash)
            )
        else:
            checks.append(_check("CAPITAL_AVAILABLE", True, "N/A for SELL trades"))

        # ---- Rule 2: MAX_POSITION_SIZE (BUY only) -----------------------
        if is_buy:
            r2, adjusted_quantity = self._rule_max_position_size(
                quantity, entry_price, total_portfolio_value
            )
            checks.append(r2)
        else:
            checks.append(_check("MAX_POSITION_SIZE", True, "N/A for SELL trades"))

        # ---- Rule 3: MAX_OPEN_POSITIONS (BUY only) ----------------------
        if is_buy:
            checks.append(self._rule_max_open_positions(holdings))
        else:
            checks.append(_check("MAX_OPEN_POSITIONS", True, "N/A for SELL trades"))

        # ---- Rule 4: DAILY_LOSS_LIMIT (BUY only) ------------------------
        if is_buy:
            checks.append(
                self._rule_daily_loss_limit(
                    today_realized_pnl, unrealized_pnl, total_portfolio_value
                )
            )
        else:
            checks.append(_check("DAILY_LOSS_LIMIT", True, "N/A for SELL trades"))

        # ---- Rule 5: WEEKLY_LOSS_LIMIT (BUY only) -----------------------
        if is_buy:
            checks.append(self._rule_weekly_loss_limit(weekly_pnl, total_portfolio_value))
        else:
            checks.append(_check("WEEKLY_LOSS_LIMIT", True, "N/A for SELL trades"))

        # ---- Rule 6: STOP_LOSS_MANDATORY (BUY only) ---------------------
        checks.append(
            self._rule_stop_loss_mandatory(trade_type, entry_price, stop_loss)
        )

        # ---- Rule 7: RISK_REWARD_MINIMUM (BUY only) ---------------------
        checks.append(
            self._rule_risk_reward_minimum(trade_type, entry_price, stop_loss, target_1)
        )

        # ---- Rule 8: SECTOR_CONCENTRATION (BUY only) --------------------
        if is_buy:
            checks.append(self._rule_sector_concentration(sector, holdings))
        else:
            checks.append(_check("SECTOR_CONCENTRATION", True, "N/A for SELL trades"))

        # ---- Rule 9: NO_TRADING_FIRST_LAST_15MIN (always) ---------------
        checks.append(self._rule_trading_time_window())

        # ---- Rule 10: DUPLICATE_TRADE_CHECK (BUY only) ------------------
        if is_buy:
            checks.append(self._rule_duplicate_trade(symbol, holdings))
        else:
            checks.append(_check("DUPLICATE_TRADE_CHECK", True, "N/A for SELL trades"))

        # ---- Rule 11: MINIMUM_TRADE_VALUE (always) ----------------------
        checks.append(self._rule_minimum_trade_value(quantity, entry_price))

        # ---- Aggregate --------------------------------------------------
        failed = [c for c in checks if not c["passed"]]
        approved = len(failed) == 0
        rejection_reason: Optional[str] = (
            "; ".join(c["detail"] for c in failed) if failed else None
        )

        if not approved:
            _log.warning(
                "Trade REJECTED | %s %s qty=%d @ %.2f | %s",
                trade_type, symbol, quantity, entry_price, rejection_reason,
            )

        # Critical alerts for loss-limit breaches
        daily_rule = next((c for c in checks if c["rule"] == "DAILY_LOSS_LIMIT"), None)
        weekly_rule = next((c for c in checks if c["rule"] == "WEEKLY_LOSS_LIMIT"), None)
        if daily_rule and not daily_rule["passed"]:
            _log.critical("DAILY LOSS LIMIT HIT — all new BUY orders blocked for today")
        if weekly_rule and not weekly_rule["passed"]:
            _log.critical("WEEKLY LOSS LIMIT HIT — all new BUY orders blocked for this week")

        risk_score = self._calculate_risk_score(
            checks, today_realized_pnl, unrealized_pnl, weekly_pnl,
            total_portfolio_value, len(holdings),
        )

        return {
            "approved":          approved,
            "trade":             proposed_trade,
            "checks":            checks,
            "rejection_reason":  rejection_reason,
            "adjusted_quantity": adjusted_quantity,
            "risk_score":        round(risk_score, 4),
        }

    # ------------------------------------------------------------------ #
    # Rule implementations                                                 #
    # ------------------------------------------------------------------ #

    def _rule_capital_available(
        self, quantity: int, entry_price: float, available_cash: float
    ) -> dict:
        required = quantity * entry_price
        passed = required <= available_cash
        if passed:
            detail = (
                f"Required ₹{required:,.0f} ≤ available ₹{available_cash:,.0f}"
            )
        else:
            detail = (
                f"Insufficient cash: need ₹{required:,.0f}, have ₹{available_cash:,.0f}"
            )
        return _check("CAPITAL_AVAILABLE", passed, detail)

    def _rule_max_position_size(
        self,
        quantity: int,
        entry_price: float,
        total_portfolio_value: float,
    ) -> tuple[dict, Optional[int]]:
        """Returns (check_result, adjusted_quantity_or_None)."""
        position_value = quantity * entry_price
        pf_value = total_portfolio_value if total_portfolio_value > 0 else self._capital
        position_pct = position_value / pf_value
        limit = self._max_position_pct

        if position_pct <= limit:
            detail = (
                f"Position ₹{position_value:,.0f} is "
                f"{position_pct * 100:.1f}% of portfolio (limit: {limit * 100:.0f}%)"
            )
            return _check("MAX_POSITION_SIZE", True, detail), None

        # Too large — propose an adjusted quantity
        max_value = pf_value * limit
        adjusted = max(0, int(max_value / entry_price)) if entry_price > 0 else 0
        detail = (
            f"Position ₹{position_value:,.0f} is {position_pct * 100:.1f}% of portfolio "
            f"(limit: {limit * 100:.0f}%); suggested adjusted qty = {adjusted}"
        )
        return _check("MAX_POSITION_SIZE", False, detail), adjusted

    def _rule_max_open_positions(self, holdings: list) -> dict:
        current = len(holdings)
        limit = self._max_open_positions
        passed = current < limit
        if passed:
            detail = f"Open positions: {current}/{limit}"
        else:
            detail = f"Max open positions reached: {current}/{limit}"
        return _check("MAX_OPEN_POSITIONS", passed, detail)

    def _rule_daily_loss_limit(
        self,
        today_realized_pnl: float,
        unrealized_pnl: float,
        total_portfolio_value: float = 0.0,
    ) -> dict:
        today_total = today_realized_pnl + unrealized_pnl
        # Use live portfolio value so limits scale as capital grows/shrinks
        effective_capital = total_portfolio_value if total_portfolio_value > 0 else self._capital
        limit = -(effective_capital * self._max_daily_loss_pct)
        passed = today_total >= limit
        if passed:
            detail = f"Today's P&L ₹{today_total:,.0f} (limit: ₹{limit:,.0f})"
        else:
            detail = (
                f"Daily loss limit breached: today P&L ₹{today_total:,.0f} "
                f"< limit ₹{limit:,.0f}"
            )
        return _check("DAILY_LOSS_LIMIT", passed, detail)

    def _rule_weekly_loss_limit(
        self,
        weekly_pnl: float,
        total_portfolio_value: float = 0.0,
    ) -> dict:
        effective_capital = total_portfolio_value if total_portfolio_value > 0 else self._capital
        limit = -(effective_capital * self._max_weekly_loss_pct)
        passed = weekly_pnl >= limit
        if passed:
            detail = f"Weekly P&L ₹{weekly_pnl:,.0f} (limit: ₹{limit:,.0f})"
        else:
            detail = (
                f"Weekly loss limit breached: weekly P&L ₹{weekly_pnl:,.0f} "
                f"< limit ₹{limit:,.0f}"
            )
        return _check("WEEKLY_LOSS_LIMIT", passed, detail)

    def _rule_stop_loss_mandatory(
        self, trade_type: str, entry_price: float, stop_loss: Any
    ) -> dict:
        if trade_type != "BUY":
            return _check("STOP_LOSS_MANDATORY", True, "N/A for SELL trades")

        if stop_loss is None or stop_loss <= 0:
            return _check(
                "STOP_LOSS_MANDATORY", False,
                f"Stop loss missing or invalid: {stop_loss}",
            )

        sl = float(stop_loss)
        ep = float(entry_price)

        if sl >= ep:
            return _check(
                "STOP_LOSS_MANDATORY", False,
                f"Stop loss ₹{sl:.2f} must be strictly below entry ₹{ep:.2f}",
            )

        sl_distance_pct = (ep - sl) / ep
        if sl_distance_pct > self._MAX_STOP_LOSS_PCT:
            return _check(
                "STOP_LOSS_MANDATORY", False,
                f"Stop loss ₹{sl:.2f} is {sl_distance_pct * 100:.1f}% below entry "
                f"(max allowed: {self._MAX_STOP_LOSS_PCT * 100:.0f}%)",
            )

        return _check(
            "STOP_LOSS_MANDATORY", True,
            f"Stop loss set at ₹{sl:.2f} ({sl_distance_pct * 100:.1f}% below entry)",
        )

    def _rule_risk_reward_minimum(
        self,
        trade_type: str,
        entry_price: float,
        stop_loss: Any,
        target_1: Any,
    ) -> dict:
        if trade_type != "BUY":
            return _check("RISK_REWARD_MINIMUM", True, "N/A for SELL trades")

        if stop_loss is None or target_1 is None:
            return _check(
                "RISK_REWARD_MINIMUM", False,
                "Cannot calculate R:R — stop_loss or target_1 missing",
            )

        ep = float(entry_price)
        sl = float(stop_loss)
        t1 = float(target_1)
        risk = ep - sl
        reward = t1 - ep

        if risk <= 0:
            return _check(
                "RISK_REWARD_MINIMUM", False,
                f"Invalid risk: entry ₹{ep:.2f}, stop ₹{sl:.2f}",
            )
        if reward <= 0:
            return _check(
                "RISK_REWARD_MINIMUM", False,
                f"Target ₹{t1:.2f} must be above entry ₹{ep:.2f}",
            )

        rr = reward / risk
        passed = rr >= self._MIN_RR_RATIO
        detail = (
            f"R:R = {rr:.2f} (risk ₹{risk:.2f}, reward ₹{reward:.2f}; "
            f"min: {self._MIN_RR_RATIO})"
        )
        if not passed:
            detail = f"R:R {rr:.2f} below minimum {self._MIN_RR_RATIO}: " + detail
        return _check("RISK_REWARD_MINIMUM", passed, detail)

    def _rule_sector_concentration(self, sector: str, holdings: list) -> dict:
        sector_upper = sector.strip().upper()
        if not sector_upper or sector_upper == "UNKNOWN":
            return _check(
                "SECTOR_CONCENTRATION", True,
                "Sector unknown — skipping concentration check",
            )

        count = sum(
            1 for h in holdings
            if str(h.get("sector", "")).strip().upper() == sector_upper
        )
        limit = self._MAX_SECTOR_POSITIONS
        passed = count < limit
        if passed:
            detail = f"Sector {sector_upper}: {count}/{limit} positions"
        else:
            detail = (
                f"Sector {sector_upper} already has {count} positions (limit: {limit})"
            )
        return _check("SECTOR_CONCENTRATION", passed, detail)

    def _rule_trading_time_window(self) -> dict:
        now = _ist_now()
        current_minutes = now.hour * 60 + now.minute

        opening_start = 9 * 60 + 15   # 09:15
        opening_end   = 9 * 60 + 30   # 09:30
        closing_start = 15 * 60 + 15  # 15:15
        closing_end   = 15 * 60 + 30  # 15:30

        time_str = now.strftime("%H:%M")

        if opening_start <= current_minutes < opening_end:
            return _check(
                "NO_TRADING_FIRST_LAST_15MIN", False,
                f"No trading during opening volatility window "
                f"(09:15–09:30 IST); current: {time_str}",
            )
        if closing_start <= current_minutes < closing_end:
            return _check(
                "NO_TRADING_FIRST_LAST_15MIN", False,
                f"No trading during closing rush window "
                f"(15:15–15:30 IST); current: {time_str}",
            )

        return _check(
            "NO_TRADING_FIRST_LAST_15MIN", True,
            f"Current time {time_str} IST is outside restricted windows",
        )

    def _rule_duplicate_trade(self, symbol: str, holdings: list) -> dict:
        held = {str(h.get("symbol", "")).upper() for h in holdings}
        sym_upper = symbol.upper()
        passed = sym_upper not in held
        if passed:
            detail = f"{sym_upper} not in current holdings"
        else:
            detail = f"Already holding {sym_upper} — no averaging down in Phase 1"
        return _check("DUPLICATE_TRADE_CHECK", passed, detail)

    def _rule_minimum_trade_value(self, quantity: int, entry_price: float) -> dict:
        value = quantity * entry_price
        passed = value >= self._MIN_TRADE_VALUE
        if passed:
            detail = (
                f"Trade value ₹{value:,.0f} meets minimum ₹{self._MIN_TRADE_VALUE:,.0f}"
            )
        else:
            detail = (
                f"Trade value ₹{value:,.0f} below minimum ₹{self._MIN_TRADE_VALUE:,.0f}"
            )
        return _check("MINIMUM_TRADE_VALUE", passed, detail)

    # ------------------------------------------------------------------ #
    # Position sizing                                                      #
    # ------------------------------------------------------------------ #

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        atr_pct: float = 0.0,
        available_cash: float = 0.0,
    ) -> dict:
        """Calculate recommended position size using dynamic risk-based sizing.

        Uses the *smallest* of three constraints:

        1. **Risk-per-trade** (2% of capital): ``capital * 0.02 / (entry − SL)``
        2. **Position-size limit** (10% of portfolio): ``capital * max_pos_pct / entry``
        3. **Cash constraint**: ``available_cash / entry_price``

        Additionally, if ``atr_pct > 3.0%`` (high volatility), the quantity
        is scaled down proportionally so that per-trade risk stays constant
        across different volatility regimes.

        Parameters
        ----------
        symbol:       Stock ticker.
        entry_price:  Proposed entry price.
        stop_loss:    Stop-loss price.
        atr_pct:      14-day ATR as a percentage of price (optional; 0 = no scaling).
        available_cash: Current available cash (optional; 0 = use capital estimate).

        Returns
        -------
        dict with keys:
            recommended_quantity, investment_amount, investment_pct,
            risk_amount, risk_pct, volatility_adjusted
            (and optional ``error`` key on failure)
        """
        per_share_risk = entry_price - stop_loss
        if per_share_risk <= 0:
            return {
                "recommended_quantity": 0,
                "investment_amount":    0.0,
                "investment_pct":       0.0,
                "risk_amount":          0.0,
                "risk_pct":             0.0,
                "volatility_adjusted":  False,
                "error": "stop_loss must be strictly below entry_price",
            }

        total_capital = self._get_total_capital()

        # Method 1: risk-based sizing
        max_risk_amount = total_capital * self._MAX_RISK_PER_TRADE_PCT
        qty_by_risk = int(max_risk_amount / per_share_risk)

        # Method 2: position-size-based sizing
        max_position_value = total_capital * self._max_position_pct
        qty_by_position = int(max_position_value / entry_price) if entry_price > 0 else 0

        # Method 3: cash constraint
        if available_cash > 0:
            qty_by_cash = int(available_cash / entry_price) if entry_price > 0 else 0
        else:
            qty_by_cash = qty_by_position  # No cash constraint provided

        recommended_qty = min(qty_by_risk, qty_by_position, qty_by_cash)

        # Volatility scaling: reduce qty for highly volatile stocks
        # Normalises to a 2% ATR baseline
        volatility_adjusted = False
        if atr_pct > 3.0:
            vol_factor = 2.0 / atr_pct
            recommended_qty = max(1, int(recommended_qty * vol_factor))
            volatility_adjusted = True

        investment_amount = recommended_qty * entry_price
        investment_pct    = (investment_amount / total_capital * 100) if total_capital else 0.0
        risk_amount       = recommended_qty * per_share_risk
        risk_pct          = (risk_amount / total_capital * 100) if total_capital else 0.0

        _log.debug(
            "Position size %s: qty_risk=%d, qty_pos=%d, qty_cash=%d, vol_adj=%s → recommended=%d",
            symbol, qty_by_risk, qty_by_position, qty_by_cash,
            volatility_adjusted, recommended_qty,
        )

        return {
            "recommended_quantity": recommended_qty,
            "investment_amount":    round(investment_amount, 2),
            "investment_pct":       round(investment_pct, 2),
            "risk_amount":          round(risk_amount, 2),
            "risk_pct":             round(risk_pct, 2),
            "volatility_adjusted":  volatility_adjusted,
        }

    # ------------------------------------------------------------------ #
    # Portfolio risk summary                                               #
    # ------------------------------------------------------------------ #

    def get_portfolio_risk_summary(self) -> dict:
        """Return the current risk state of the portfolio.

        Returns
        -------
        dict with keys:
            total_capital, invested_amount, available_cash, utilization_pct,
            open_positions, unrealized_pnl, today_realized_pnl, today_total_pnl,
            weekly_pnl, is_daily_limit_hit, is_weekly_limit_hit, can_trade,
            sector_exposure, largest_position_pct
        """
        port = self._get_portfolio_state()
        total_capital         = self._get_total_capital()
        holdings              = port["holdings"]
        total_portfolio_value = port["total_portfolio_value"]
        available_cash        = port["available_cash"]
        invested_amount       = port["invested_amount"]
        unrealized_pnl        = port["unrealized_pnl"]
        today_realized_pnl    = port["today_realized_pnl"]
        weekly_pnl            = port["weekly_pnl"]

        today_total_pnl = today_realized_pnl + unrealized_pnl

        utilization_pct = (
            invested_amount / total_portfolio_value * 100
            if total_portfolio_value > 0 else 0.0
        )

        daily_limit  = -(total_capital * self._max_daily_loss_pct)
        weekly_limit = -(total_capital * self._max_weekly_loss_pct)
        is_daily_limit_hit  = today_total_pnl < daily_limit
        is_weekly_limit_hit = weekly_pnl < weekly_limit

        # Sector exposure derived from enriched holdings
        sector_exposure: dict[str, int] = {}
        for h in holdings:
            s = str(h.get("sector", "UNKNOWN")).strip().upper()
            sector_exposure[s] = sector_exposure.get(s, 0) + 1

        # Largest single position as % of portfolio
        largest_position_pct = 0.0
        if total_portfolio_value > 0 and holdings:
            position_values = [
                (h.get("quantity") or 0) * (h.get("ltp") or h.get("avg_price") or 0)
                for h in holdings
            ]
            if position_values:
                largest_position_pct = max(position_values) / total_portfolio_value * 100

        if is_daily_limit_hit:
            _log.critical("DAILY LOSS LIMIT HIT — portfolio summary shows breached limit")
        if is_weekly_limit_hit:
            _log.critical("WEEKLY LOSS LIMIT HIT — portfolio summary shows breached limit")

        return {
            "total_capital":        total_capital,
            "invested_amount":      round(invested_amount, 2),
            "available_cash":       round(available_cash, 2),
            "utilization_pct":      round(utilization_pct, 2),
            "open_positions":       len(holdings),
            "unrealized_pnl":       round(unrealized_pnl, 2),
            "today_realized_pnl":   round(today_realized_pnl, 2),
            "today_total_pnl":      round(today_total_pnl, 2),
            "weekly_pnl":           round(weekly_pnl, 2),
            "is_daily_limit_hit":   is_daily_limit_hit,
            "is_weekly_limit_hit":  is_weekly_limit_hit,
            "can_trade":            not is_daily_limit_hit and not is_weekly_limit_hit,
            "sector_exposure":      sector_exposure,
            "largest_position_pct": round(largest_position_pct, 2),
        }

    # ------------------------------------------------------------------ #
    # Circuit breaker                                                      #
    # ------------------------------------------------------------------ #

    def is_market_safe_to_trade(self) -> dict:
        """Check broader market conditions (Nifty 50 and India VIX).

        Does **not** block trading by itself — adds to ``risk_score`` and
        logs warnings.  The caller can act on ``market_safe = False``.

        Returns
        -------
        dict with keys: market_safe, nifty_change_pct, vix, warnings
        """
        warnings: list[str] = []
        nifty_change_pct = 0.0
        vix = 0.0
        market_safe = True

        # --- Nifty 50 intraday change ------------------------------------
        try:
            nifty_ltp_data = self.broker.get_ltp("NIFTY 50", "NSE")
            nifty_ltp = float(nifty_ltp_data.get("ltp", 0) or 0)
            if nifty_ltp > 0:
                from datetime import date as _date
                today = _date.today()
                yesterday = (today - timedelta(days=1)).strftime("%Y%m%d")
                today_str = today.strftime("%Y%m%d")
                hist = self.broker.get_historical_data(
                    "NIFTY 50", "ONE_DAY", yesterday, today_str, "NSE"
                )
                if hist is not None and len(hist) >= 1:
                    # If one row returned it is today's; two rows → take first as yesterday
                    prev_close = (
                        float(hist.iloc[-2]["close"]) if len(hist) >= 2
                        else float(hist.iloc[-1]["close"])
                    )
                    if prev_close > 0:
                        nifty_change_pct = (nifty_ltp - prev_close) / prev_close * 100
                        if nifty_change_pct < -2.0:
                            msg = (
                                f"HIGH_RISK_DAY: Nifty 50 down "
                                f"{nifty_change_pct:.2f}% intraday"
                            )
                            warnings.append(msg)
                            market_safe = False
                            _log.warning(msg)
        except Exception as exc:  # noqa: BLE001
            _log.debug("Could not fetch Nifty 50 data for circuit breaker: %s", exc)

        # --- India VIX ---------------------------------------------------
        try:
            vix_data = self.broker.get_ltp("INDIA VIX", "NSE")
            vix = float(vix_data.get("ltp", 0) or 0)
            if vix > 25:
                msg = f"HIGH_VOLATILITY: India VIX = {vix:.1f} (threshold: 25)"
                warnings.append(msg)
                market_safe = False
                _log.warning(msg)
        except Exception as exc:  # noqa: BLE001
            _log.debug("Could not fetch India VIX for circuit breaker: %s", exc)

        return {
            "market_safe":       market_safe,
            "nifty_change_pct":  round(nifty_change_pct, 4),
            "vix":               round(vix, 2),
            "warnings":          warnings,
        }

    # ------------------------------------------------------------------ #
    # Backward-compatible stub interface (original RiskAgent methods)     #
    # ------------------------------------------------------------------ #

    def approve_trade(self, trade_proposal: dict) -> bool:
        """Return True if the trade passes all risk checks."""
        return self.check_trade(trade_proposal)["approved"]

    def check_daily_loss_limit(self) -> bool:
        """Return True if the daily loss limit has NOT been breached."""
        port = self._get_portfolio_state()
        today_total = port["today_realized_pnl"] + port["unrealized_pnl"]
        effective_capital = port["total_portfolio_value"] or self._capital
        return today_total >= -(effective_capital * self._max_daily_loss_pct)

    def check_weekly_loss_limit(self) -> bool:
        """Return True if the weekly loss limit has NOT been breached."""
        port = self._get_portfolio_state()
        effective_capital = port["total_portfolio_value"] or self._capital
        return port["weekly_pnl"] >= -(effective_capital * self._max_weekly_loss_pct)

    def check_open_positions(self) -> bool:
        """Return True if adding one more position stays within the configured max."""
        port = self._get_portfolio_state()
        return len(port["holdings"]) < self._max_open_positions

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_total_capital(self) -> float:
        """Return total portfolio value from broker, or fall back to config capital."""
        if self._paper_trading:
            return self._capital
        try:
            pv = self.broker.get_portfolio_value()
            total = pv.get("total_value")
            return float(total) if total else self._capital
        except Exception:
            return self._capital

    def _get_portfolio_state(self) -> dict:
        """Fetch live portfolio state; degrade gracefully on any failure."""
        holdings: list[dict] = []
        available_cash       = self._capital
        invested_amount      = 0.0
        total_portfolio_value = self._capital
        unrealized_pnl       = 0.0

        if self._paper_trading:
            # Paper mode: the real broker account is irrelevant. Build portfolio
            # state from DB open trades + configured capital so risk checks use
            # the simulated capital (e.g. ₹1,00,000) rather than the real cash.
            try:
                open_trades = self.db.get_open_trades()
                for t in open_trades:
                    qty   = int(getattr(t, "quantity", 0) or 0)
                    price = float(getattr(t, "price", 0.0) or 0.0)
                    invested_amount += qty * price
                    holdings.append({
                        "symbol":    t.symbol,
                        "quantity":  qty,
                        "avg_price": price,
                        "ltp":       price,
                        "pnl":       0.0,
                        "sector":    "UNKNOWN",
                    })
            except Exception as exc:
                _log.debug("Paper mode: could not fetch open trades from DB: %s", exc)
            holdings = self._enrich_holdings_with_sector(holdings) if holdings else holdings
            available_cash = max(0.0, self._capital - invested_amount)
            total_portfolio_value = self._capital  # mark-to-market ≈ capital w/ ltp=price
        else:
            # --- Broker: portfolio value ---
            try:
                pv = self.broker.get_portfolio_value()
                total_portfolio_value = float(pv.get("total_value")  or self._capital)
                available_cash        = float(pv.get("available_cash") or self._capital)
                invested_amount       = float(pv.get("invested")       or 0.0)
            except Exception as exc:
                _log.debug("Could not fetch portfolio value: %s", exc)

            # --- Broker: holdings (with fallback to DB open trades) ---
            try:
                raw_holdings = self.broker.get_holdings()
                for h in raw_holdings:
                    unrealized_pnl += float(h.get("pnl") or 0.0)
                holdings = self._enrich_holdings_with_sector(raw_holdings)
            except Exception as exc:
                _log.debug("Could not fetch broker holdings: %s", exc)
                try:
                    open_trades = self.db.get_open_trades()
                    for t in open_trades:
                        holdings.append({
                            "symbol":    t.symbol,
                            "quantity":  t.quantity,
                            "avg_price": t.price,
                            "ltp":       t.price,
                            "pnl":       0.0,
                            "sector":    "UNKNOWN",
                        })
                except Exception as exc2:
                    _log.debug("Could not fetch open trades from DB: %s", exc2)

        # --- DB: today's realized PnL ---
        today_realized_pnl = 0.0
        try:
            today_str    = _ist_today_str()
            recent_trades = self.db.get_trade_history(days=2)
            for trade in recent_trades:
                if (
                    trade.exit_date
                    and trade.exit_date.startswith(today_str)
                    and trade.pnl is not None
                ):
                    today_realized_pnl += trade.pnl
        except Exception as exc:
            _log.debug("Could not compute today's realized PnL: %s", exc)

        # --- DB: this week's cumulative PnL (Monday → today) ---
        weekly_pnl = 0.0
        try:
            monday_str    = _this_week_monday_str()
            weekly_trades = self.db.get_trade_history(days=7)
            for trade in weekly_trades:
                if (
                    trade.exit_date
                    and trade.exit_date >= monday_str
                    and trade.pnl is not None
                ):
                    weekly_pnl += trade.pnl
        except Exception as exc:
            _log.debug("Could not compute weekly PnL: %s", exc)

        return {
            "holdings":              holdings,
            "total_portfolio_value": total_portfolio_value,
            "available_cash":        available_cash,
            "invested_amount":       invested_amount,
            "unrealized_pnl":        unrealized_pnl,
            "today_realized_pnl":    today_realized_pnl,
            "weekly_pnl":            weekly_pnl,
        }

    def _enrich_holdings_with_sector(self, holdings: list[dict]) -> list[dict]:
        """Add sector info to each holding using the latest watchlist from DB.

        Only overwrites the sector field when the watchlist actually knows the
        symbol.  Pre-existing sector info (e.g. from broker response) is kept
        when the symbol is absent from the watchlist.
        """
        try:
            watchlist = self.db.get_latest_watchlist()
            sector_map = {
                item.symbol.upper(): (item.sector or "UNKNOWN")
                for item in watchlist
            }
            for h in holdings:
                sym = str(h.get("symbol", "")).upper()
                if sym in sector_map:
                    h["sector"] = sector_map[sym]
                elif not h.get("sector"):
                    h["sector"] = "UNKNOWN"
        except Exception:
            pass
        return holdings

    def _calculate_risk_score(
        self,
        checks: list[dict],
        today_realized_pnl: float,
        unrealized_pnl: float,
        weekly_pnl: float,
        total_portfolio_value: float,
        open_positions: int,
    ) -> float:
        """Composite risk score: 0 = safest, 1 = maximum risk."""
        score = 0.0

        # Component 1: failed rules (each failed rule → +0.10, max 0.40)
        failed_count = sum(1 for c in checks if not c["passed"])
        score += min(failed_count * 0.10, 0.40)

        # Component 2: proximity to daily loss limit (0–0.20)
        today_loss = -(today_realized_pnl + unrealized_pnl)
        daily_limit = self._capital * self._max_daily_loss_pct
        if today_loss > 0 and daily_limit > 0:
            score += min(today_loss / daily_limit, 1.0) * 0.20

        # Component 3: proximity to weekly loss limit (0–0.20)
        weekly_loss = -weekly_pnl
        weekly_limit = self._capital * self._max_weekly_loss_pct
        if weekly_loss > 0 and weekly_limit > 0:
            score += min(weekly_loss / weekly_limit, 1.0) * 0.20

        # Component 4: position utilization (0–0.20)
        utilization = open_positions / max(self._max_open_positions, 1)
        score += utilization * 0.20

        return min(score, 1.0)


# --------------------------------------------------------------------------- #
# Backward-compatible alias for the original stub class name                  #
# --------------------------------------------------------------------------- #

#: ``RiskAgent`` is a drop-in alias for :class:`RiskManager`.
#: Existing code that imports ``RiskAgent`` continues to work.
RiskAgent = RiskManager
