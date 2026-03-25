"""Tests for RiskManager — all 11 rules, position sizing, and portfolio summary.

Strategy:
  - All broker and DB calls are replaced with lightweight MagicMock objects.
  - Tests are deterministic and run in < 1 s (no network or file I/O).
  - Time-window tests patch datetime so they run regardless of when CI executes.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.agents.risk_agent import RiskManager, RiskAgent

IST = ZoneInfo("Asia/Kolkata")


# --------------------------------------------------------------------------- #
# Shared fixtures and helpers                                                  #
# --------------------------------------------------------------------------- #

def _make_config(
    capital: float = 50_000,
    max_position_pct: float = 25,   # stored as int % in config
    max_daily_loss_pct: float = 2,
    max_weekly_loss_pct: float = 5,
    max_open_positions: int = 5,
) -> MagicMock:
    cfg = MagicMock()
    cfg.get.side_effect = lambda *keys, default=None: {
        ("trading", "capital"):             capital,
        ("trading", "max_position_pct"):    max_position_pct,
        ("trading", "max_daily_loss_pct"):  max_daily_loss_pct,
        ("trading", "max_weekly_loss_pct"): max_weekly_loss_pct,
        ("trading", "max_open_positions"):  max_open_positions,
    }.get(keys, default)
    return cfg


def _make_broker(
    portfolio_value: float = 50_000,
    available_cash:  float = 50_000,
    invested:        float = 0.0,
    holdings: list | None = None,
) -> MagicMock:
    broker = MagicMock()
    broker.get_portfolio_value.return_value = {
        "total_value":   portfolio_value,
        "available_cash": available_cash,
        "invested":       invested,
        "total_pnl":      0.0,
    }
    broker.get_holdings.return_value = holdings or []
    return broker


def _make_db(
    open_trades: list | None = None,
    trade_history: list | None = None,
    watchlist: list | None = None,
) -> MagicMock:
    db = MagicMock()
    db.get_open_trades.return_value = open_trades or []
    db.get_trade_history.return_value = trade_history or []
    db.get_latest_watchlist.return_value = watchlist or []
    return db


def _make_rm(
    capital: float = 50_000,
    max_position_pct: float = 25,
    max_daily_loss_pct: float = 2,
    max_weekly_loss_pct: float = 5,
    max_open_positions: int = 5,
    available_cash: float = 50_000,
    portfolio_value: float = 50_000,
    holdings: list | None = None,
    trade_history: list | None = None,
    open_trades: list | None = None,
    watchlist: list | None = None,
) -> RiskManager:
    cfg    = _make_config(capital, max_position_pct, max_daily_loss_pct,
                          max_weekly_loss_pct, max_open_positions)
    broker = _make_broker(portfolio_value, available_cash,
                          portfolio_value - available_cash, holdings)
    db     = _make_db(open_trades, trade_history, watchlist)
    return RiskManager(cfg, broker, db)


def _good_trade(**overrides) -> dict:
    """Return a default valid BUY trade that should pass all rules."""
    base = {
        "symbol":      "TCS",
        "trade_type":  "BUY",
        "quantity":    5,
        "entry_price": 3_800.0,
        "stop_loss":   3_705.0,    # 2.5% below entry — within 5% limit
        "target_1":    3_942.5,    # risk=95, reward=142.5 → R:R=1.5
        "sector":      "IT",
    }
    base.update(overrides)
    return base


def _passed(result: dict, rule: str) -> bool:
    for c in result["checks"]:
        if c["rule"] == rule:
            return c["passed"]
    raise KeyError(f"Rule {rule!r} not found in checks")


# --------------------------------------------------------------------------- #
# Rule 1 — CAPITAL_AVAILABLE                                                   #
# --------------------------------------------------------------------------- #

class TestCapitalAvailable:

    def test_passes_when_enough_cash(self):
        rm = _make_rm(available_cash=50_000)
        result = rm.check_trade(_good_trade(quantity=5, entry_price=3_800))
        assert _passed(result, "CAPITAL_AVAILABLE")

    def test_fails_when_insufficient_cash(self):
        # 5 * 3800 = 19000 > available 10000
        rm = _make_rm(available_cash=10_000, portfolio_value=50_000)
        result = rm.check_trade(_good_trade(quantity=5, entry_price=3_800))
        assert not _passed(result, "CAPITAL_AVAILABLE")
        assert result["approved"] is False

    def test_auto_passes_for_sell(self):
        rm = _make_rm(available_cash=100)          # almost no cash
        result = rm.check_trade(_good_trade(trade_type="SELL"))
        assert _passed(result, "CAPITAL_AVAILABLE")


# --------------------------------------------------------------------------- #
# Rule 2 — MAX_POSITION_SIZE                                                   #
# --------------------------------------------------------------------------- #

class TestMaxPositionSize:

    def test_passes_within_limit(self):
        # 5 * 3800 = 19000; 19000/50000 = 38% > 25% limit → actually fails
        # Let's use a small position: 1 * 3800 = 3800; 3800/50000 = 7.6% < 25%
        rm = _make_rm(max_position_pct=25, portfolio_value=50_000)
        result = rm.check_trade(_good_trade(quantity=1, entry_price=3_800))
        assert _passed(result, "MAX_POSITION_SIZE")

    def test_fails_and_provides_adjusted_quantity(self):
        # 100 * 3800 = 380000; way over 25% of 50000
        rm = _make_rm(max_position_pct=25, portfolio_value=50_000)
        result = rm.check_trade(_good_trade(quantity=100, entry_price=3_800))
        assert not _passed(result, "MAX_POSITION_SIZE")
        assert result["adjusted_quantity"] is not None
        assert isinstance(result["adjusted_quantity"], int)
        # adjusted_qty * 3800 should be <= 25% of 50000 = 12500
        assert result["adjusted_quantity"] * 3_800 <= 50_000 * 0.25

    def test_adjusted_quantity_is_zero_when_price_exceeds_limit(self):
        # max allowed = 25% of 10000 = 2500; price = 3800 → 0 shares
        rm = _make_rm(max_position_pct=25, portfolio_value=10_000)
        result = rm.check_trade(_good_trade(quantity=5, entry_price=3_800))
        assert not _passed(result, "MAX_POSITION_SIZE")
        assert result["adjusted_quantity"] == 0

    def test_auto_passes_for_sell(self):
        rm = _make_rm(portfolio_value=50_000)
        result = rm.check_trade(_good_trade(trade_type="SELL"))
        assert _passed(result, "MAX_POSITION_SIZE")


# --------------------------------------------------------------------------- #
# Rule 3 — MAX_OPEN_POSITIONS                                                  #
# --------------------------------------------------------------------------- #

class TestMaxOpenPositions:

    def test_passes_below_limit(self):
        holdings = [{"symbol": f"STOCK{i}", "sector": "IT"} for i in range(4)]
        rm = _make_rm(max_open_positions=5, holdings=holdings)
        result = rm.check_trade(_good_trade())
        assert _passed(result, "MAX_OPEN_POSITIONS")

    def test_fails_at_limit(self):
        holdings = [{"symbol": f"STOCK{i}", "sector": "IT"} for i in range(5)]
        rm = _make_rm(max_open_positions=5, holdings=holdings)
        result = rm.check_trade(_good_trade())
        assert not _passed(result, "MAX_OPEN_POSITIONS")
        assert result["approved"] is False

    def test_auto_passes_for_sell(self):
        holdings = [{"symbol": f"STOCK{i}", "sector": "IT"} for i in range(10)]
        rm = _make_rm(max_open_positions=5, holdings=holdings)
        result = rm.check_trade(_good_trade(trade_type="SELL"))
        assert _passed(result, "MAX_OPEN_POSITIONS")


# --------------------------------------------------------------------------- #
# Rule 4 — DAILY_LOSS_LIMIT                                                    #
# --------------------------------------------------------------------------- #

class TestDailyLossLimit:

    def _make_exit_trade(self, pnl: float) -> MagicMock:
        """Return a mock closed trade with today's exit_date."""
        today = datetime.now(IST).strftime("%Y-%m-%d")
        t = MagicMock()
        t.exit_date = f"{today} 14:00:00"
        t.pnl = pnl
        return t

    def test_passes_when_no_loss(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade())
        assert _passed(result, "DAILY_LOSS_LIMIT")

    def test_passes_just_below_limit(self):
        # Limit = 2% of 50000 = 1000; loss of 999 is fine
        trade = self._make_exit_trade(pnl=-999)
        rm = _make_rm(trade_history=[trade], max_daily_loss_pct=2)
        result = rm.check_trade(_good_trade())
        assert _passed(result, "DAILY_LOSS_LIMIT")

    def test_fails_when_daily_limit_exceeded(self):
        trade = self._make_exit_trade(pnl=-1_100)
        rm = _make_rm(trade_history=[trade], max_daily_loss_pct=2)
        result = rm.check_trade(_good_trade())
        assert not _passed(result, "DAILY_LOSS_LIMIT")
        assert result["approved"] is False

    def test_daily_limit_blocks_buy_not_sell(self):
        trade = self._make_exit_trade(pnl=-2_000)
        rm = _make_rm(trade_history=[trade], max_daily_loss_pct=2)

        buy_result  = rm.check_trade(_good_trade(trade_type="BUY"))
        sell_result = rm.check_trade(_good_trade(trade_type="SELL"))

        assert not _passed(buy_result, "DAILY_LOSS_LIMIT")
        assert _passed(sell_result, "DAILY_LOSS_LIMIT")

    def test_unrealized_loss_counts_towards_daily_limit(self):
        # Holdings with -1500 unrealized PnL
        holdings = [{"symbol": "TCS", "pnl": -1_500, "sector": "IT"}]
        rm = _make_rm(holdings=holdings, max_daily_loss_pct=2)
        result = rm.check_trade(_good_trade())
        assert not _passed(result, "DAILY_LOSS_LIMIT")


# --------------------------------------------------------------------------- #
# Rule 5 — WEEKLY_LOSS_LIMIT                                                   #
# --------------------------------------------------------------------------- #

class TestWeeklyLossLimit:

    def _make_week_trade(self, pnl: float) -> MagicMock:
        """Return a mock closed trade from this week."""
        from datetime import date, timedelta
        today = date.today()
        monday = today - timedelta(days=today.weekday())
        t = MagicMock()
        t.exit_date = f"{monday.strftime('%Y-%m-%d')} 14:00:00"
        t.pnl = pnl
        return t

    def test_passes_when_no_weekly_loss(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade())
        assert _passed(result, "WEEKLY_LOSS_LIMIT")

    def test_fails_when_weekly_limit_exceeded(self):
        # Limit = 5% of 50000 = 2500; weekly loss -3000
        trades = [self._make_week_trade(-3_000)]
        rm = _make_rm(trade_history=trades, max_weekly_loss_pct=5)
        result = rm.check_trade(_good_trade())
        assert not _passed(result, "WEEKLY_LOSS_LIMIT")
        assert result["approved"] is False

    def test_weekly_limit_blocks_buy_not_sell(self):
        trades = [self._make_week_trade(-3_000)]
        rm = _make_rm(trade_history=trades, max_weekly_loss_pct=5)
        sell_result = rm.check_trade(_good_trade(trade_type="SELL"))
        assert _passed(sell_result, "WEEKLY_LOSS_LIMIT")


# --------------------------------------------------------------------------- #
# Rule 6 — STOP_LOSS_MANDATORY                                                 #
# --------------------------------------------------------------------------- #

class TestStopLossMandatory:

    def test_passes_with_valid_stop_loss(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(entry_price=3_800, stop_loss=3_705))
        assert _passed(result, "STOP_LOSS_MANDATORY")

    def test_fails_with_no_stop_loss(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(stop_loss=None))
        assert not _passed(result, "STOP_LOSS_MANDATORY")

    def test_fails_with_zero_stop_loss(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(stop_loss=0))
        assert not _passed(result, "STOP_LOSS_MANDATORY")

    def test_fails_when_stop_loss_above_entry(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(entry_price=3_800, stop_loss=3_900))
        assert not _passed(result, "STOP_LOSS_MANDATORY")

    def test_fails_when_stop_loss_too_far_below_entry(self):
        # entry 3800, stop 3420 → (3800-3420)/3800 ≈ 10% > 5% limit
        rm = _make_rm()
        result = rm.check_trade(_good_trade(entry_price=3_800, stop_loss=3_420))
        assert not _passed(result, "STOP_LOSS_MANDATORY")

    def test_auto_passes_for_sell(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(trade_type="SELL", stop_loss=None))
        assert _passed(result, "STOP_LOSS_MANDATORY")


# --------------------------------------------------------------------------- #
# Rule 7 — RISK_REWARD_MINIMUM                                                 #
# --------------------------------------------------------------------------- #

class TestRiskRewardMinimum:

    def test_passes_with_exactly_1_5_rr(self):
        # risk=95, reward=142.5 → R:R=1.5
        rm = _make_rm()
        result = rm.check_trade(
            _good_trade(entry_price=3_800, stop_loss=3_705, target_1=3_942.5)
        )
        assert _passed(result, "RISK_REWARD_MINIMUM")

    def test_fails_with_low_rr(self):
        # risk=95, reward=90 → R:R=0.95 < 1.5
        rm = _make_rm()
        result = rm.check_trade(
            _good_trade(entry_price=3_800, stop_loss=3_705, target_1=3_890)
        )
        assert not _passed(result, "RISK_REWARD_MINIMUM")

    def test_fails_when_target_missing(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(target_1=None))
        assert not _passed(result, "RISK_REWARD_MINIMUM")

    def test_fails_when_target_below_entry(self):
        rm = _make_rm()
        result = rm.check_trade(
            _good_trade(entry_price=3_800, stop_loss=3_705, target_1=3_700)
        )
        assert not _passed(result, "RISK_REWARD_MINIMUM")

    def test_auto_passes_for_sell(self):
        rm = _make_rm()
        result = rm.check_trade(
            _good_trade(trade_type="SELL", stop_loss=None, target_1=None)
        )
        assert _passed(result, "RISK_REWARD_MINIMUM")


# --------------------------------------------------------------------------- #
# Rule 8 — SECTOR_CONCENTRATION                                                #
# --------------------------------------------------------------------------- #

class TestSectorConcentration:

    def test_passes_with_one_it_position(self):
        holdings = [
            {"symbol": "INFY", "sector": "IT", "quantity": 10, "ltp": 1500},
        ]
        rm = _make_rm(holdings=holdings)
        result = rm.check_trade(_good_trade(sector="IT"))  # TCS → 2nd IT stock
        assert _passed(result, "SECTOR_CONCENTRATION")

    def test_fails_when_two_it_already_held(self):
        holdings = [
            {"symbol": "INFY", "sector": "IT", "quantity": 5, "ltp": 1500},
            {"symbol": "WIPRO", "sector": "IT", "quantity": 5, "ltp": 450},
        ]
        rm = _make_rm(holdings=holdings)
        result = rm.check_trade(_good_trade(sector="IT"))  # 3rd IT → fail
        assert not _passed(result, "SECTOR_CONCENTRATION")
        assert result["approved"] is False

    def test_passes_when_sector_unknown(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(sector="UNKNOWN"))
        assert _passed(result, "SECTOR_CONCENTRATION")

    def test_passes_different_sector(self):
        holdings = [
            {"symbol": "INFY",  "sector": "IT",      "quantity": 5, "ltp": 1500},
            {"symbol": "WIPRO", "sector": "IT",       "quantity": 5, "ltp": 450},
            {"symbol": "HDFC",  "sector": "BANKING",  "quantity": 5, "ltp": 1700},
        ]
        rm = _make_rm(holdings=holdings)
        # Adding a PHARMA stock — no sector concentration violation
        result = rm.check_trade(_good_trade(sector="PHARMA"))
        assert _passed(result, "SECTOR_CONCENTRATION")


# --------------------------------------------------------------------------- #
# Rule 9 — NO_TRADING_FIRST_LAST_15MIN                                         #
# --------------------------------------------------------------------------- #

class TestTradingTimeWindow:

    def _rm(self) -> RiskManager:
        return _make_rm()

    def _patch_time(self, hour: int, minute: int):
        """Context manager that pins IST time to hour:minute."""
        fake_now = datetime(2026, 3, 18, hour, minute, 0, tzinfo=IST)
        return patch("src.agents.risk_agent._ist_now", return_value=fake_now)

    def test_passes_during_regular_hours(self):
        with self._patch_time(11, 0):
            result = self._rm().check_trade(_good_trade())
        assert _passed(result, "NO_TRADING_FIRST_LAST_15MIN")

    def test_fails_during_opening_window_9_15(self):
        with self._patch_time(9, 15):
            result = self._rm().check_trade(_good_trade())
        assert not _passed(result, "NO_TRADING_FIRST_LAST_15MIN")

    def test_fails_during_opening_window_9_25(self):
        with self._patch_time(9, 25):
            result = self._rm().check_trade(_good_trade())
        assert not _passed(result, "NO_TRADING_FIRST_LAST_15MIN")

    def test_passes_at_9_30_exactly(self):
        with self._patch_time(9, 30):
            result = self._rm().check_trade(_good_trade())
        assert _passed(result, "NO_TRADING_FIRST_LAST_15MIN")

    def test_fails_during_closing_window_15_15(self):
        with self._patch_time(15, 15):
            result = self._rm().check_trade(_good_trade())
        assert not _passed(result, "NO_TRADING_FIRST_LAST_15MIN")

    def test_fails_during_closing_window_15_25(self):
        with self._patch_time(15, 25):
            result = self._rm().check_trade(_good_trade())
        assert not _passed(result, "NO_TRADING_FIRST_LAST_15MIN")

    def test_passes_at_15_30_exactly(self):
        with self._patch_time(15, 30):
            result = self._rm().check_trade(_good_trade())
        assert _passed(result, "NO_TRADING_FIRST_LAST_15MIN")

    def test_applies_to_sell_trades_too(self):
        with self._patch_time(9, 20):
            result = self._rm().check_trade(_good_trade(trade_type="SELL"))
        assert not _passed(result, "NO_TRADING_FIRST_LAST_15MIN")


# --------------------------------------------------------------------------- #
# Rule 10 — DUPLICATE_TRADE_CHECK                                              #
# --------------------------------------------------------------------------- #

class TestDuplicateTradeCheck:

    def test_passes_when_not_holding_symbol(self):
        holdings = [{"symbol": "INFY", "sector": "IT"}]
        rm = _make_rm(holdings=holdings)
        result = rm.check_trade(_good_trade(symbol="TCS"))
        assert _passed(result, "DUPLICATE_TRADE_CHECK")

    def test_fails_when_already_holding_symbol(self):
        holdings = [{"symbol": "TCS", "sector": "IT"}]
        rm = _make_rm(holdings=holdings)
        result = rm.check_trade(_good_trade(symbol="TCS"))
        assert not _passed(result, "DUPLICATE_TRADE_CHECK")
        assert result["approved"] is False

    def test_case_insensitive(self):
        holdings = [{"symbol": "tcs", "sector": "IT"}]
        rm = _make_rm(holdings=holdings)
        result = rm.check_trade(_good_trade(symbol="TCS"))
        assert not _passed(result, "DUPLICATE_TRADE_CHECK")

    def test_auto_passes_for_sell(self):
        holdings = [{"symbol": "TCS", "sector": "IT"}]
        rm = _make_rm(holdings=holdings)
        result = rm.check_trade(_good_trade(symbol="TCS", trade_type="SELL"))
        assert _passed(result, "DUPLICATE_TRADE_CHECK")


# --------------------------------------------------------------------------- #
# Rule 11 — MINIMUM_TRADE_VALUE                                                #
# --------------------------------------------------------------------------- #

class TestMinimumTradeValue:

    def test_passes_above_minimum(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(quantity=1, entry_price=1_500))
        assert _passed(result, "MINIMUM_TRADE_VALUE")

    def test_passes_at_exactly_minimum(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(quantity=10, entry_price=100))
        assert _passed(result, "MINIMUM_TRADE_VALUE")

    def test_fails_below_minimum(self):
        rm = _make_rm()
        # 2 shares * 400 = 800 < 1000
        result = rm.check_trade(_good_trade(quantity=2, entry_price=400))
        assert not _passed(result, "MINIMUM_TRADE_VALUE")
        assert result["approved"] is False

    def test_applies_to_sell_trades(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(trade_type="SELL", quantity=1, entry_price=500))
        # 500 < 1000 → should fail
        assert not _passed(result, "MINIMUM_TRADE_VALUE")


# --------------------------------------------------------------------------- #
# Multiple simultaneous rule failures                                          #
# --------------------------------------------------------------------------- #

class TestMultipleFailures:

    def test_all_failures_reported_not_just_first(self):
        """All failing rules must appear in checks — not just the first one."""
        # 3 failures: no stop_loss, no target, and insufficient cash
        rm = _make_rm(available_cash=100, portfolio_value=50_000)
        trade = {
            "symbol":      "TCS",
            "trade_type":  "BUY",
            "quantity":    5,
            "entry_price": 3_800,
            "stop_loss":   None,   # Rule 6 fails
            "target_1":    None,   # Rule 7 fails
            "sector":      "IT",
        }
        result = rm.check_trade(trade)
        assert result["approved"] is False

        failed_rules = [c["rule"] for c in result["checks"] if not c["passed"]]
        assert "CAPITAL_AVAILABLE"   in failed_rules   # no cash
        assert "STOP_LOSS_MANDATORY" in failed_rules   # no stop loss
        assert "RISK_REWARD_MINIMUM" in failed_rules   # no target

    def test_rejection_reason_contains_all_failures(self):
        rm = _make_rm(available_cash=100)
        trade = _good_trade(stop_loss=None)
        result = rm.check_trade(trade)
        # rejection_reason should mention both failures
        assert result["rejection_reason"] is not None
        # At least two distinct failure messages are present
        assert len(result["rejection_reason"]) > 20

    def test_approved_is_true_only_when_all_rules_pass(self):
        rm = _make_rm(available_cash=50_000, portfolio_value=50_000)
        # Use a small enough position to pass MAX_POSITION_SIZE with 25% limit
        result = rm.check_trade(_good_trade(quantity=1, entry_price=3_800))
        all_passed = all(c["passed"] for c in result["checks"])
        assert result["approved"] == all_passed


# --------------------------------------------------------------------------- #
# Position sizing                                                               #
# --------------------------------------------------------------------------- #

class TestPositionSizing:

    def test_basic_position_sizing(self):
        rm = _make_rm(capital=50_000, max_position_pct=25)
        result = rm.calculate_position_size(
            symbol="TCS",
            entry_price=3_800,
            stop_loss=3_705,  # per-share risk = 95
        )
        assert "recommended_quantity" in result
        assert result["recommended_quantity"] > 0
        # Risk amount should not exceed 2% of capital = 1000
        assert result["risk_amount"] <= 1_000 + 1  # tiny float tolerance

    def test_position_bounded_by_max_position_pct(self):
        rm = _make_rm(capital=50_000, max_position_pct=25)
        result = rm.calculate_position_size("TCS", 3_800, 3_705)
        max_investment = 50_000 * 0.25
        assert result["investment_amount"] <= max_investment + 1

    def test_returns_zero_when_stop_loss_invalid(self):
        rm = _make_rm()
        result = rm.calculate_position_size("TCS", 3_800, 3_900)  # sl > entry
        assert result["recommended_quantity"] == 0
        assert "error" in result

    def test_tight_stop_loss_gives_more_shares(self):
        rm = _make_rm(capital=50_000, max_position_pct=100)  # remove position cap
        # Tight SL (10 per share): 1000/10 = 100 shares
        r_tight = rm.calculate_position_size("TCS", 1_000, 990)
        # Wide SL (200 per share): 1000/200 = 5 shares
        r_wide  = rm.calculate_position_size("TCS", 1_000, 800)
        assert r_tight["recommended_quantity"] > r_wide["recommended_quantity"]

    def test_result_keys_present(self):
        rm = _make_rm()
        result = rm.calculate_position_size("INFY", 1_500, 1_455)
        for key in ("recommended_quantity", "investment_amount",
                    "investment_pct", "risk_amount", "risk_pct"):
            assert key in result


# --------------------------------------------------------------------------- #
# Portfolio risk summary                                                        #
# --------------------------------------------------------------------------- #

class TestPortfolioRiskSummary:

    def test_no_positions_clean_state(self):
        rm = _make_rm(
            capital=50_000,
            available_cash=50_000,
            portfolio_value=50_000,
            holdings=[],
            trade_history=[],
        )
        summary = rm.get_portfolio_risk_summary()
        assert summary["open_positions"]      == 0
        assert summary["can_trade"]           is True
        assert summary["is_daily_limit_hit"]  is False
        assert summary["is_weekly_limit_hit"] is False
        assert summary["available_cash"]      == 50_000

    def test_daily_limit_reflected(self):
        from datetime import datetime, date
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        t = MagicMock()
        t.exit_date = f"{today_str} 14:00:00"
        t.pnl = -2_000  # 4% of 50000 > 2% daily limit

        rm = _make_rm(trade_history=[t])
        summary = rm.get_portfolio_risk_summary()
        assert summary["is_daily_limit_hit"] is True
        assert summary["can_trade"]          is False

    def test_weekly_limit_reflected(self):
        from datetime import date, timedelta
        monday = date.today() - timedelta(days=date.today().weekday())
        t = MagicMock()
        t.exit_date = f"{monday.strftime('%Y-%m-%d')} 10:00:00"
        t.pnl = -3_000  # 6% of 50000 > 5% weekly limit

        rm = _make_rm(trade_history=[t])
        summary = rm.get_portfolio_risk_summary()
        assert summary["is_weekly_limit_hit"] is True
        assert summary["can_trade"]           is False

    def test_sector_exposure_aggregated(self):
        holdings = [
            {"symbol": "TCS",   "sector": "IT",      "quantity": 5, "ltp": 3800, "pnl": 0},
            {"symbol": "INFY",  "sector": "IT",      "quantity": 5, "ltp": 1500, "pnl": 0},
            {"symbol": "HDFC",  "sector": "BANKING", "quantity": 5, "ltp": 1700, "pnl": 0},
        ]
        rm = _make_rm(holdings=holdings, portfolio_value=50_000)
        summary = rm.get_portfolio_risk_summary()
        assert summary["sector_exposure"].get("IT", 0)      == 2
        assert summary["sector_exposure"].get("BANKING", 0) == 1

    def test_largest_position_pct_calculated(self):
        holdings = [
            {"symbol": "TCS",  "quantity": 10, "ltp": 3_000, "pnl": 0, "sector": "IT"},
            {"symbol": "INFY", "quantity": 5,  "ltp": 1_500, "pnl": 0, "sector": "IT"},
        ]
        # TCS = 30000, INFY = 7500, total = 50000
        # Largest = 30000/50000 = 60%
        rm = _make_rm(holdings=holdings, portfolio_value=50_000)
        summary = rm.get_portfolio_risk_summary()
        assert abs(summary["largest_position_pct"] - 60.0) < 0.1

    def test_utilization_pct(self):
        rm = _make_rm(
            portfolio_value=50_000,
            available_cash=15_000,
        )
        # invested = 50000 - 15000 = 35000
        summary = rm.get_portfolio_risk_summary()
        # utilization = 35000/50000 = 70%
        assert abs(summary["utilization_pct"] - 70.0) < 0.5


# --------------------------------------------------------------------------- #
# Risk score                                                                    #
# --------------------------------------------------------------------------- #

class TestRiskScore:

    def test_clean_trade_has_low_risk_score(self):
        rm = _make_rm(available_cash=50_000, portfolio_value=50_000)
        result = rm.check_trade(_good_trade(quantity=1, entry_price=3_800))
        assert result["risk_score"] < 0.5

    def test_portfolio_with_losses_has_higher_risk_score_than_clean(self):
        # Clean state: no losses, no positions
        rm_safe = _make_rm(available_cash=50_000, portfolio_value=50_000)

        # Risky state: 4 open positions + 80% of daily loss already hit
        from datetime import datetime as _dt
        today_str = _dt.now(IST).strftime("%Y-%m-%d")
        t = MagicMock()
        t.exit_date = f"{today_str} 14:00:00"
        t.pnl = -800   # 80% of 2% daily limit (₹1,000)
        holdings = [{"symbol": f"S{i}", "sector": "IT"} for i in range(4)]
        rm_risky = _make_rm(
            available_cash=50_000, portfolio_value=50_000,
            holdings=holdings, trade_history=[t],
        )

        result_safe  = rm_safe.check_trade(_good_trade(quantity=1, entry_price=3_800))
        result_risky = rm_risky.check_trade(_good_trade(quantity=1, entry_price=3_800))
        assert result_risky["risk_score"] > result_safe["risk_score"]

    def test_risk_score_bounded_0_to_1(self):
        rm = _make_rm(available_cash=0, portfolio_value=0)
        result = rm.check_trade(_good_trade(quantity=1000, entry_price=10_000, stop_loss=None))
        assert 0.0 <= result["risk_score"] <= 1.0


# --------------------------------------------------------------------------- #
# check_trade response structure                                                #
# --------------------------------------------------------------------------- #

class TestResponseStructure:

    def test_result_has_all_required_keys(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(quantity=1, entry_price=3_800))
        for key in ("approved", "trade", "checks", "rejection_reason",
                    "adjusted_quantity", "risk_score"):
            assert key in result, f"Missing key: {key}"

    def test_trade_dict_preserved_in_result(self):
        rm = _make_rm()
        trade = _good_trade()
        result = rm.check_trade(trade)
        assert result["trade"] is trade

    def test_exactly_11_checks_returned(self):
        rm = _make_rm()
        result = rm.check_trade(_good_trade(quantity=1, entry_price=3_800))
        assert len(result["checks"]) == 11

    def test_rejection_reason_is_none_when_approved(self):
        rm = _make_rm(available_cash=50_000, portfolio_value=50_000)
        result = rm.check_trade(_good_trade(quantity=1, entry_price=3_800))
        if result["approved"]:
            assert result["rejection_reason"] is None

    def test_rejection_reason_present_when_rejected(self):
        rm = _make_rm(available_cash=100)
        result = rm.check_trade(_good_trade())
        assert result["approved"] is False
        assert result["rejection_reason"] is not None
        assert len(result["rejection_reason"]) > 0


# --------------------------------------------------------------------------- #
# Backward-compatible stub methods                                             #
# --------------------------------------------------------------------------- #

class TestBackwardCompat:

    def test_approve_trade_returns_bool(self):
        rm = _make_rm(available_cash=50_000, portfolio_value=50_000)
        result = rm.approve_trade(_good_trade(quantity=1, entry_price=3_800))
        assert isinstance(result, bool)

    def test_check_daily_loss_limit_true_when_no_loss(self):
        rm = _make_rm()
        assert rm.check_daily_loss_limit() is True

    def test_check_weekly_loss_limit_true_when_no_loss(self):
        rm = _make_rm()
        assert rm.check_weekly_loss_limit() is True

    def test_check_open_positions_true_when_below_limit(self):
        rm = _make_rm(holdings=[], max_open_positions=5)
        assert rm.check_open_positions() is True

    def test_check_open_positions_false_at_limit(self):
        holdings = [{"symbol": f"S{i}", "sector": "IT"} for i in range(5)]
        rm = _make_rm(holdings=holdings, max_open_positions=5)
        assert rm.check_open_positions() is False

    def test_risk_agent_alias_is_risk_manager(self):
        """RiskAgent must be a drop-in alias for RiskManager."""
        assert RiskAgent is RiskManager

    def test_risk_agent_instantiates_correctly(self):
        cfg    = _make_config()
        broker = _make_broker()
        db     = _make_db()
        agent  = RiskAgent(cfg, broker, db)
        assert isinstance(agent, RiskManager)


# --------------------------------------------------------------------------- #
# Broker / DB failure resilience                                               #
# --------------------------------------------------------------------------- #

class TestResilientOnBrokerFailure:

    def test_check_trade_still_runs_when_broker_raises(self):
        cfg    = _make_config()
        broker = MagicMock()
        broker.get_portfolio_value.side_effect = Exception("connection refused")
        broker.get_holdings.side_effect        = Exception("connection refused")
        db     = _make_db()

        rm = RiskManager(cfg, broker, db)
        # Should not raise — just use fallback capital values
        result = rm.check_trade(_good_trade(quantity=1, entry_price=3_800))
        assert "approved" in result

    def test_check_trade_still_runs_when_db_raises(self):
        cfg    = _make_config()
        broker = _make_broker()
        db     = MagicMock()
        db.get_open_trades.side_effect    = Exception("db locked")
        db.get_trade_history.side_effect  = Exception("db locked")
        db.get_latest_watchlist.side_effect = Exception("db locked")

        rm = RiskManager(cfg, broker, db)
        result = rm.check_trade(_good_trade(quantity=1, entry_price=3_800))
        assert "approved" in result
