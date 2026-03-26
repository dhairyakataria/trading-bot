"""Safety Circuit Breaker — halts trading when loss or error thresholds are hit.

Six independent breakers are evaluated on every call to
:meth:`CircuitBreaker.check_all`:

1. BROKER_CONNECTION      — ping broker 3 times; trip on total failure
2. DAILY_LOSS_LIMIT       — today's realised + unrealised P&L < threshold → block new trades
3. WEEKLY_LOSS_LIMIT      — this-week P&L < threshold → block new trades
4. MARKET_CRASH_DETECTOR  — Nifty > 3 % drop → block new trades; > 5 % → warn to exit
5. SYSTEM_HEALTH          — DB connectivity check
6. OUTSIDE_MARKET_HOURS   — only 9:15 AM – 3:30 PM IST on NSE trading days

Paper-trading mode:
- BROKER_CONNECTION and MARKET_CRASH_DETECTOR still run (paper mode uses
  real market data — Angel One is logged in for live prices/OHLCV).
- OUTSIDE_MARKET_HOURS is skipped so the bot can be tested outside
  live market sessions.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.utils.logger import get_logger

IST = ZoneInfo("Asia/Kolkata")
_log = get_logger("circuit_breakers.safety")

# ── Market window (IST, inclusive) ─────────────────────────────────────────
_OPEN_H,  _OPEN_M  = 9, 15
_CLOSE_H, _CLOSE_M = 15, 30

# ── Symbols ─────────────────────────────────────────────────────────────────
_NIFTY_SYMBOL = "NIFTY"       # Nifty 50 index on Angel One
_PING_SYMBOL  = "RELIANCE"    # Liquid stock used for connectivity test

# ── NSE holidays (YYYY-MM-DD) ───────────────────────────────────────────────
_NSE_HOLIDAYS: frozenset[str] = frozenset({
    # 2025
    "2025-01-26", "2025-02-26", "2025-03-14", "2025-04-14",
    "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27",
    "2025-10-02", "2025-10-20", "2025-10-21", "2025-11-05", "2025-12-25",
    # 2026
    "2026-01-26", "2026-03-03", "2026-04-02", "2026-04-14",
    "2026-05-01", "2026-08-15", "2026-10-02", "2026-10-09",
    "2026-10-29", "2026-11-23", "2026-12-25",
})


# ── Helpers ─────────────────────────────────────────────────────────────────

def _cfg(config: Any, *keys: str, default: Any = None) -> Any:
    """Safely access nested config values, supporting both dict and Config objects."""
    try:
        if hasattr(config, "get"):
            return config.get(*keys, default=default)
        obj: Any = config
        for k in keys:
            obj = obj[k]
        return obj
    except (KeyError, TypeError, AttributeError):
        return default


# ── CircuitBreaker ──────────────────────────────────────────────────────────

class CircuitBreaker:
    """System-level circuit breaker that pauses or restricts trading on safety violations.

    If a *critical* breaker trips (``system_ok=False``), the orchestrator
    must pause the entire cycle — including exits.  Non-critical breakers
    (``can_open_new_positions=False``) allow exit processing but block new buys.

    Args:
        config:        Application :class:`~src.utils.config.Config` instance.
        broker_client: Authenticated :class:`~src.broker.angel_one.AngelOneClient`.
        db_manager:    :class:`~src.database.db_manager.DatabaseManager` instance.
    """

    _BROKER_PING_RETRIES = 3
    _CRASH_WARN_PCT      = 0.03   # 3 % Nifty drop → block new trades
    _CRASH_EXIT_PCT      = 0.05   # 5 % Nifty drop → warn to exit risky positions

    def __init__(
        self,
        config: Any,
        broker_client: Any,
        db_manager: Any,
    ) -> None:
        self.config = config
        self.broker = broker_client
        self.db     = db_manager

        capital    = float(_cfg(config, "trading", "capital",              default=50_000))
        daily_pct  =       _cfg(config, "trading", "max_daily_loss_pct",  default=2) / 100
        weekly_pct =       _cfg(config, "trading", "max_weekly_loss_pct", default=5) / 100

        self._capital           = capital
        self._daily_loss_limit  = -(capital * daily_pct)
        self._weekly_loss_limit = -(capital * weekly_pct)
        self._paper_trading     = bool(_cfg(config, "trading", "paper_trading", default=False))

        # Legacy SafetyCircuitBreaker state
        self._tripped: bool     = False
        self._trip_reason: str  = ""

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def check_all(self) -> dict:
        """Evaluate all circuit breakers and return a composite status dict.

        Returns::

            {
                "system_ok":              bool,   # False → pause everything
                "can_open_new_positions": bool,   # False → exits only
                "must_close_all":         bool,   # severe crash detected
                "breakers_tripped":       list[str],
                "warnings":               list[str],
            }
        """
        tripped:    List[str] = []
        warnings:   List[str] = []
        system_ok  = True
        can_open   = True
        must_close = False

        # ── 1. Broker connection ─────────────────────────────────────────
        # Runs in ALL modes — paper mode still uses Angel One for live
        # market data (prices, OHLCV).
        if not self._check_broker_connection():
            tripped.append("BROKER_CONNECTION")
            system_ok = False

        # ── 2. Outside market hours / holiday ───────────────────────────
        # Skipped in paper mode so the bot can be tested outside live
        # market sessions.
        if not self._paper_trading:
            hours_ok, hours_msg = self._check_market_hours()
            if not hours_ok:
                tripped.append("OUTSIDE_MARKET_HOURS")
                system_ok = False
                _log.info("CircuitBreaker: OUTSIDE_MARKET_HOURS — %s", hours_msg)

        # Only run P&L / market checks if the broker is reachable
        broker_reachable = system_ok

        if broker_reachable:
            # ── 3. Daily loss limit ─────────────────────────────────────
            daily_ok, daily_pnl = self._check_daily_loss()
            if not daily_ok:
                tripped.append("DAILY_LOSS_LIMIT")
                can_open = False
                _log.warning(
                    "CircuitBreaker: DAILY_LOSS_LIMIT — P&L ₹%.0f (limit ₹%.0f)",
                    daily_pnl, self._daily_loss_limit,
                )

            # ── 4. Weekly loss limit ────────────────────────────────────
            weekly_ok, weekly_pnl = self._check_weekly_loss()
            if not weekly_ok:
                tripped.append("WEEKLY_LOSS_LIMIT")
                can_open = False
                _log.warning(
                    "CircuitBreaker: WEEKLY_LOSS_LIMIT — P&L ₹%.0f (limit ₹%.0f)",
                    weekly_pnl, self._weekly_loss_limit,
                )

            # ── 5. Market crash detector ────────────────────────────────
            # Runs in ALL modes — paper mode uses real market data from
            # Angel One, so Nifty crash detection remains active.
            crash_ok, change_pct, crash_msg = self._check_market_crash()
            if not crash_ok:
                tripped.append("MARKET_CRASH_DETECTOR")
                can_open = False
                if change_pct is not None and abs(change_pct) >= self._CRASH_EXIT_PCT:
                    must_close = True
                    warnings.append(
                        f"Nifty down {change_pct * 100:.1f}% — "
                        "consider exiting risky positions"
                    )
            if crash_msg:
                warnings.append(crash_msg)

        # ── 6. System health (DB) ────────────────────────────────────────
        health_ok, health_msg = self._check_system_health()
        if not health_ok:
            tripped.append("SYSTEM_HEALTH")
            system_ok = False
            _log.error("CircuitBreaker: SYSTEM_HEALTH — %s", health_msg)
        elif health_msg:
            warnings.append(health_msg)

        # Sync legacy flag
        self._tripped = not system_ok

        result: dict = {
            "system_ok":              system_ok,
            "can_open_new_positions": can_open and system_ok,
            "must_close_all":         must_close,
            "breakers_tripped":       tripped,
            "warnings":               warnings,
        }
        _log.info(
            "CircuitBreaker.check_all → system_ok=%s can_open=%s tripped=%s",
            system_ok, result["can_open_new_positions"], tripped or "none",
        )
        return result

    # ------------------------------------------------------------------ #
    # Individual breaker checks                                            #
    # ------------------------------------------------------------------ #

    def _check_broker_connection(self) -> bool:
        """Ping broker up to 3 times; return False only if all attempts fail."""
        for attempt in range(1, self._BROKER_PING_RETRIES + 1):
            try:
                self.broker.get_ltp(_PING_SYMBOL)
                return True
            except Exception as exc:
                _log.debug(
                    "Broker ping %d/%d failed: %s",
                    attempt, self._BROKER_PING_RETRIES, exc,
                )
                if attempt < self._BROKER_PING_RETRIES:
                    time.sleep(1)
        _log.error(
            "Broker connection FAILED after %d ping attempts", self._BROKER_PING_RETRIES
        )
        return False

    def _check_market_hours(self) -> Tuple[bool, str]:
        """Return (True, '') if within NSE market hours on a trading day."""
        now      = datetime.now(IST)
        date_str = now.strftime("%Y-%m-%d")

        if now.weekday() >= 5:
            return False, f"Weekend ({now.strftime('%A')})"

        if date_str in _NSE_HOLIDAYS:
            return False, f"NSE holiday on {date_str}"

        open_dt  = now.replace(hour=_OPEN_H,  minute=_OPEN_M,  second=0, microsecond=0)
        close_dt = now.replace(hour=_CLOSE_H, minute=_CLOSE_M, second=0, microsecond=0)

        if not (open_dt <= now <= close_dt):
            return False, f"Outside market hours — current IST {now.strftime('%H:%M')}"

        return True, ""

    def _check_daily_loss(self) -> Tuple[bool, float]:
        """Return (within_limit, today_total_pnl)."""
        try:
            today_str = datetime.now(IST).strftime("%Y-%m-%d")
            trades    = self.db.get_trade_history(days=2)
            realized  = sum(
                t.pnl for t in trades
                if t.exit_date
                and t.exit_date.startswith(today_str)
                and t.pnl is not None
            )
            unrealized = 0.0
            if not self._paper_trading:
                try:
                    pv         = self.broker.get_portfolio_value()
                    unrealized = pv.get("total_pnl", 0.0) - realized
                except Exception:
                    pass
            total = realized + unrealized
            return total >= self._daily_loss_limit, total
        except Exception as exc:
            _log.debug("Daily loss check error (treated as OK): %s", exc)
            return True, 0.0

    def _check_weekly_loss(self) -> Tuple[bool, float]:
        """Return (within_limit, weekly_pnl)."""
        try:
            now        = datetime.now(IST)
            monday_str = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
            trades     = self.db.get_trade_history(days=7)
            weekly_pnl = sum(
                t.pnl for t in trades
                if t.exit_date and t.exit_date >= monday_str and t.pnl is not None
            )
            return weekly_pnl >= self._weekly_loss_limit, weekly_pnl
        except Exception as exc:
            _log.debug("Weekly loss check error (treated as OK): %s", exc)
            return True, 0.0

    def _check_market_crash(self) -> Tuple[bool, Optional[float], str]:
        """Check Nifty's intraday percentage move vs previous close.

        Returns:
            (is_ok, change_pct_or_None, warning_message)
        """
        try:
            ltp_data = self.broker.get_ltp(_NIFTY_SYMBOL)
            current  = float(ltp_data["ltp"])

            now     = datetime.now(IST)
            from_dt = (now - timedelta(days=3)).strftime("%Y-%m-%d %H:%M")
            to_dt   = now.strftime("%Y-%m-%d %H:%M")

            hist = self.broker.get_historical_data(
                _NIFTY_SYMBOL, "ONE_DAY", from_dt, to_dt
            )
            if hist is None or hist.empty or len(hist) < 2:
                return True, None, ""

            prev_close = float(hist.iloc[-2]["close"])
            change_pct = (current - prev_close) / prev_close  # negative = drop

            if change_pct <= -self._CRASH_EXIT_PCT:
                return (
                    False,
                    change_pct,
                    f"Nifty down {change_pct * 100:.1f}% — "
                    "consider exiting risky positions",
                )
            if change_pct <= -self._CRASH_WARN_PCT:
                return (
                    False,
                    change_pct,
                    f"Nifty down {change_pct * 100:.1f}% — no new trades",
                )
            if change_pct <= -0.02:
                return True, change_pct, f"Nifty down {change_pct * 100:.1f}% — watch closely"

            return True, change_pct, ""
        except Exception as exc:
            _log.debug("Market crash check error (treated as OK): %s", exc)
            return True, None, ""

    def _check_system_health(self) -> Tuple[bool, str]:
        """Verify DB is responsive via a lightweight read."""
        try:
            self.db.get_system_state("_health_ping")
            return True, ""
        except Exception as exc:
            return False, f"Database health check failed: {exc}"

    # ------------------------------------------------------------------ #
    # Legacy SafetyCircuitBreaker compatibility                            #
    # ------------------------------------------------------------------ #

    def is_tripped(self) -> bool:
        """Return True if the circuit breaker has been triggered (legacy API)."""
        return self._tripped

    def check_loss_limits(self, pnl: float) -> None:
        """Trip the breaker if *pnl* breaches the daily loss limit (legacy API)."""
        if pnl <= self._daily_loss_limit:
            self._tripped     = True
            self._trip_reason = (
                f"Daily loss limit exceeded: ₹{pnl:,.0f} ≤ ₹{self._daily_loss_limit:,.0f}"
            )
            _log.critical("CircuitBreaker TRIPPED: %s", self._trip_reason)

    def check_error_rate(self, error_count: int, window_seconds: int) -> None:
        """Trip the breaker if too many broker errors occur in a time window (legacy API)."""
        threshold = max(3, window_seconds // 60)
        if error_count >= threshold:
            self._tripped     = True
            self._trip_reason = (
                f"High error rate: {error_count} errors in {window_seconds}s"
            )
            _log.critical("CircuitBreaker TRIPPED: %s", self._trip_reason)

    def reset(self) -> None:
        """Manually reset the circuit breaker (authorised operator use only)."""
        self._tripped     = False
        self._trip_reason = ""
        _log.info("CircuitBreaker manually reset.")

    def get_status(self) -> dict[str, Any]:
        """Return current breaker state (legacy API)."""
        return {
            "tripped": self._tripped,
            "reason":  self._trip_reason,
        }


# Backward-compatibility alias — existing tests / scheduler may reference the old name
SafetyCircuitBreaker = CircuitBreaker
