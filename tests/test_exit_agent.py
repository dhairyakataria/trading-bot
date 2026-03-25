"""Tests for ExitAgent — exit signal generation.

Strategy:
  - All broker, DB, technical-indicators, and research-agent calls are replaced
    with MagicMock objects so tests are deterministic and run in < 1 s.
  - Each test targets one specific exit type or guard condition.
  - Time-sensitive tests use fixed entry_date strings rather than patching
    datetime so that _business_days_between is exercised realistically.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.agents.exit_agent import ExitAgent, _business_days_between
from src.database.models import Trade

IST = ZoneInfo("Asia/Kolkata")


# ─────────────────────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────────────────────

def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.get.return_value = None
    return cfg


def _make_broker(ltp: float = 3_850.0, df: Any = None) -> MagicMock:
    broker = MagicMock()
    broker.get_ltp.return_value = {"symbol": "TCS", "ltp": ltp}
    broker.get_historical_data.return_value = df if df is not None else pd.DataFrame(
        columns=["datetime", "open", "high", "low", "close", "volume"]
    )
    return broker


def _make_db(
    open_trades: list | None = None,
    system_state: dict | None = None,
) -> MagicMock:
    db = MagicMock()
    db.get_open_trades.return_value = open_trades or []
    state = system_state or {}
    db.get_system_state.side_effect = lambda key: state.get(key)
    db.set_system_state.return_value = None
    return db


def _make_ti(
    rsi_val: float = 50.0,
    macd_signal: str = "BULLISH",
    ema_vs_20: str = "ABOVE",
    volume_signal: str = "NORMAL_VOLUME",
    atr_val: float = 30.0,
) -> MagicMock:
    ti = MagicMock()
    ti.calculate_rsi.return_value      = {"value": rsi_val, "signal": "NEUTRAL"}
    ti.calculate_macd.return_value     = {"signal": macd_signal}
    ti.calculate_ema.return_value      = {"price_vs_ema_20": ema_vs_20, "ema_20": 3_800.0}
    ti.calculate_volume_analysis.return_value = {"signal": volume_signal, "volume_ratio": 1.0}
    ti.calculate_atr.return_value      = {"atr": atr_val, "atr_pct": 0.78}
    return ti


def _make_research(recommendation: str = "BUY") -> MagicMock:
    ra = MagicMock()
    ra.research_stock.return_value = {
        "recommendation":   recommendation,
        "research_summary": "No significant issues found.",
        "risks":            [],
    }
    return ra


def _make_agent(
    broker: Any = None,
    db: Any = None,
    ti: Any = None,
    research: Any = None,
    open_trades: list | None = None,
    system_state: dict | None = None,
    ltp: float = 3_850.0,
    ohlcv_df: Any = None,
) -> ExitAgent:
    if broker is None:
        broker = _make_broker(ltp=ltp, df=ohlcv_df)
    if db is None:
        db = _make_db(open_trades=open_trades, system_state=system_state)
    if ti is None:
        ti = _make_ti()
    if research is None:
        research = _make_research()
    return ExitAgent(
        config=_make_config(),
        broker_client=broker,
        technical_indicators=ti,
        research_agent=research,
        db_manager=db,
        llm_router=MagicMock(),
    )


def _trade(
    symbol: str = "TCS",
    entry_price: float = 3_800.0,
    stop_loss: float = 3_600.0,
    target_price: float = 3_980.0,
    quantity: int = 10,
    entry_date: str | None = None,
    strategy_signal: str | None = None,
) -> Trade:
    """Return a minimal Trade object for tests."""
    if entry_date is None:
        # Default: 3 business days ago.
        entry_date = (datetime.now(IST) - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
    return Trade(
        symbol=symbol,
        trade_type="BUY",
        quantity=quantity,
        price=entry_price,
        stop_loss=stop_loss,
        target_price=target_price,
        entry_date=entry_date,
        status="EXECUTED",
        strategy_signal=strategy_signal,
        id=1,
    )


def _ohlcv_df(
    n: int = 30,
    close_trend: float = 0.0,
    last_close: float | None = None,
    last_volume_ratio: float = 1.0,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for technical indicator tests."""
    base = 3_800.0
    closes = [base + i * close_trend for i in range(n)]
    if last_close is not None:
        closes[-1] = last_close
    avg_vol = 100_000
    volumes = [int(avg_vol)] * n
    volumes[-1] = int(avg_vol * last_volume_ratio)

    rows = []
    for i, (c, v) in enumerate(zip(closes, volumes)):
        rows.append({
            "datetime": datetime.now(IST) - timedelta(days=n - i),
            "open":  c - 10,
            "high":  c + 15,
            "low":   c - 15,
            "close": c,
            "volume": v,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Helper tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBusinessDays:
    def test_same_day_returns_zero(self):
        today = date.today()
        assert _business_days_between(today.isoformat(), today) == 0

    def test_five_calendar_days_mon_to_fri(self):
        # Find a Monday.
        d = date(2025, 1, 6)   # Monday
        end = date(2025, 1, 11)  # Saturday
        # Mon, Tue, Wed, Thu, Fri = 5 days.
        assert _business_days_between(d.isoformat(), end) == 5

    def test_skips_weekends(self):
        d = date(2025, 1, 10)   # Friday
        end = date(2025, 1, 13) # Monday
        # Friday + Saturday (skipped) + Sunday (skipped) → 1 business day
        assert _business_days_between(d.isoformat(), end) == 1

    def test_empty_string_returns_zero(self):
        assert _business_days_between("", date.today()) == 0

    def test_invalid_string_returns_zero(self):
        assert _business_days_between("not-a-date", date.today()) == 0


# ─────────────────────────────────────────────────────────────────────────────
# STOP_LOSS_HIT
# ─────────────────────────────────────────────────────────────────────────────

class TestStopLoss:
    def test_stop_loss_hit_returns_signal(self):
        trade  = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        agent  = _make_agent(ltp=3_550.0, open_trades=[trade])  # below SL
        sigs   = agent.check_exits()
        assert len(sigs) == 1
        assert sigs[0]["exit_type"] == "STOP_LOSS_HIT"

    def test_stop_loss_urgency_is_high(self):
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        agent = _make_agent(ltp=3_550.0, open_trades=[trade])
        sig   = agent.check_exits()[0]
        assert sig["urgency"] == "HIGH"

    def test_stop_loss_exact_boundary(self):
        """Price == stop_loss must trigger exit."""
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        agent = _make_agent(ltp=3_600.0, open_trades=[trade])
        sigs  = agent.check_exits()
        assert sigs[0]["exit_type"] == "STOP_LOSS_HIT"

    def test_above_stop_loss_no_signal(self):
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        agent = _make_agent(ltp=3_750.0, open_trades=[trade])
        sigs  = agent.check_exits()
        # No stop-loss signal; other checks may fire but not SL.
        assert all(s["exit_type"] != "STOP_LOSS_HIT" for s in sigs)

    def test_stop_loss_sell_quantity_is_all(self):
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        agent = _make_agent(ltp=3_500.0, open_trades=[trade])
        sig   = agent.check_exits()[0]
        assert sig["sell_quantity"] == "ALL"

    def test_stop_loss_pnl_is_negative(self):
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0, quantity=5)
        agent = _make_agent(ltp=3_550.0, open_trades=[trade])
        sig   = agent.check_exits()[0]
        assert sig["pnl"] < 0
        assert sig["pnl_pct"] < 0

    def test_stop_loss_skips_all_other_checks(self):
        """Once SL fires, no further signals should be generated for that symbol."""
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        agent = _make_agent(ltp=3_550.0, open_trades=[trade])
        sigs  = agent.check_exits()
        # Only one signal for this trade.
        tcs_sigs = [s for s in sigs if s["symbol"] == "TCS"]
        assert len(tcs_sigs) == 1

    def test_no_stop_loss_defined_no_signal(self):
        trade = Trade(
            symbol="TCS", trade_type="BUY", quantity=10,
            price=3_800.0, stop_loss=None, target_price=4_000.0,
            entry_date="2025-01-01 09:15:00", status="EXECUTED", id=1,
        )
        agent = _make_agent(ltp=3_700.0, open_trades=[trade])
        sigs  = [s for s in agent.check_exits() if s["exit_type"] == "STOP_LOSS_HIT"]
        assert not sigs


# ─────────────────────────────────────────────────────────────────────────────
# TARGET_HIT — partial at target_1
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetHitPartial:
    def test_partial_exit_at_target_1(self):
        trade = _trade(entry_price=3_800.0, target_price=3_980.0, quantity=10)
        # target_2 is derived as entry + 2*(t1-entry) = 3800 + 2*180 = 4160
        agent = _make_agent(ltp=3_990.0, open_trades=[trade])
        sigs  = [s for s in agent.check_exits() if s["exit_type"] == "TARGET_HIT"]
        assert len(sigs) == 1
        assert sigs[0]["sell_quantity"] == 5   # 50% of 10

    def test_partial_exit_urgency_normal(self):
        trade = _trade(entry_price=3_800.0, target_price=3_980.0, quantity=10)
        agent = _make_agent(ltp=3_990.0, open_trades=[trade])
        sig   = next(s for s in agent.check_exits() if s["exit_type"] == "TARGET_HIT")
        assert sig["urgency"] == "NORMAL"

    def test_partial_exit_pnl_is_positive(self):
        trade = _trade(entry_price=3_800.0, target_price=3_980.0, quantity=10)
        agent = _make_agent(ltp=3_990.0, open_trades=[trade])
        sig   = next(s for s in agent.check_exits() if s["exit_type"] == "TARGET_HIT")
        assert sig["pnl"] > 0
        assert sig["pnl_pct"] > 0

    def test_partial_exit_qty_floored_at_1(self):
        """Odd quantity 1 → sell 1 (max(1, 1//2))."""
        trade = _trade(entry_price=3_800.0, target_price=3_980.0, quantity=1)
        agent = _make_agent(ltp=3_990.0, open_trades=[trade])
        sig   = next(s for s in agent.check_exits() if s["exit_type"] == "TARGET_HIT")
        assert sig["sell_quantity"] == 1

    def test_below_target_1_no_target_signal(self):
        trade = _trade(entry_price=3_800.0, target_price=3_980.0)
        agent = _make_agent(ltp=3_900.0, open_trades=[trade])
        sigs  = [s for s in agent.check_exits() if s["exit_type"] == "TARGET_HIT"]
        assert not sigs


# ─────────────────────────────────────────────────────────────────────────────
# TARGET_HIT — full at target_2
# ─────────────────────────────────────────────────────────────────────────────

class TestTargetHitFull:
    def _trade_with_two_targets(self) -> Trade:
        signal = json.dumps({"target_1": 3_980.0, "target_2": 4_160.0})
        return _trade(
            entry_price=3_800.0,
            target_price=3_980.0,
            quantity=10,
            strategy_signal=signal,
        )

    def test_full_exit_at_target_2(self):
        trade = self._trade_with_two_targets()
        agent = _make_agent(ltp=4_200.0, open_trades=[trade])
        sigs  = [s for s in agent.check_exits() if s["exit_type"] == "TARGET_HIT"]
        assert len(sigs) == 1
        assert sigs[0]["sell_quantity"] == "ALL"

    def test_full_exit_urgency_normal(self):
        trade = self._trade_with_two_targets()
        agent = _make_agent(ltp=4_200.0, open_trades=[trade])
        sig   = next(s for s in agent.check_exits() if s["exit_type"] == "TARGET_HIT")
        assert sig["urgency"] == "NORMAL"

    def test_full_exit_pnl_calculated_on_full_quantity(self):
        trade = self._trade_with_two_targets()
        agent = _make_agent(ltp=4_200.0, open_trades=[trade])
        sig   = next(s for s in agent.check_exits() if s["exit_type"] == "TARGET_HIT")
        expected_pnl = round((4_200.0 - 3_800.0) * 10, 2)
        assert sig["pnl"] == expected_pnl

    def test_target_2_takes_priority_over_target_1(self):
        """When price >= target_2 we should get sell_quantity='ALL', not partial."""
        trade = self._trade_with_two_targets()
        agent = _make_agent(ltp=4_200.0, open_trades=[trade])
        sig   = next(s for s in agent.check_exits() if s["exit_type"] == "TARGET_HIT")
        assert sig["sell_quantity"] == "ALL"


# ─────────────────────────────────────────────────────────────────────────────
# TRAILING_STOP_LOSS
# ─────────────────────────────────────────────────────────────────────────────

class TestTrailingStop:
    def test_trailing_stop_not_active_when_no_high_recorded(self):
        """Trailing stop must NOT fire when no high has ever been recorded.

        update_trailing_stops only writes trailing_high_<SYM> once the price
        exceeds the 2% activation threshold.  If the key is absent, the stock
        never reached activation and the trailing stop is dormant.
        """
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        # price = 3_850 → 1.3% above entry — system_state is empty.
        agent = _make_agent(ltp=3_850.0, open_trades=[trade], system_state={})
        sigs  = [s for s in agent.check_exits() if s["exit_type"] == "TRAILING_STOP_LOSS"]
        assert not sigs

    def test_trailing_stop_activates_after_2pct_profit(self):
        """Once profit >= 2%, and price falls below trailing floor → EXIT."""
        entry = 3_800.0
        # 2% above entry = 3876; set high to 3_900 (2.6% profit).
        high          = 3_900.0
        atr           = 20.0
        trailing_stop = high - 1.5 * atr   # 3_870
        # Set current price just below trailing stop.
        current_price = trailing_stop - 1   # 3_869

        trade  = _trade(entry_price=entry, stop_loss=3_600.0)
        state  = {"trailing_high_TCS": str(high)}
        ti     = _make_ti(atr_val=atr)
        df     = _ohlcv_df()   # non-empty so get_current_atr uses ti, not fallback
        broker = _make_broker(ltp=current_price, df=df)
        db     = _make_db(open_trades=[trade], system_state=state)
        agent  = ExitAgent(
            config=_make_config(),
            broker_client=broker,
            technical_indicators=ti,
            research_agent=_make_research(),
            db_manager=db,
            llm_router=MagicMock(),
        )
        sigs = [s for s in agent.check_exits() if s["exit_type"] == "TRAILING_STOP_LOSS"]
        assert len(sigs) == 1
        assert sigs[0]["urgency"] == "HIGH"

    def test_trailing_stop_above_floor_no_signal(self):
        """Price above trailing stop floor → no signal."""
        entry = 3_800.0
        high  = 3_900.0
        atr   = 20.0
        # Trailing stop = 3_900 - 30 = 3_870; price = 3_880 > 3_870.
        current_price = 3_880.0

        trade  = _trade(entry_price=entry, stop_loss=3_600.0)
        state  = {"trailing_high_TCS": str(high)}
        ti     = _make_ti(atr_val=atr)
        df     = _ohlcv_df()   # non-empty so get_current_atr uses ti, not fallback
        broker = _make_broker(ltp=current_price, df=df)
        db     = _make_db(open_trades=[trade], system_state=state)
        agent  = ExitAgent(
            config=_make_config(),
            broker_client=broker,
            technical_indicators=ti,
            research_agent=_make_research(),
            db_manager=db,
            llm_router=MagicMock(),
        )
        sigs = [s for s in agent.check_exits() if s["exit_type"] == "TRAILING_STOP_LOSS"]
        assert not sigs

    def test_trailing_stop_no_high_recorded_no_signal(self):
        """No trailing_high in system_state → trailing stop not active yet."""
        entry = 3_800.0
        # Price is 2%+ above entry but no high recorded yet.
        current_price = entry * 1.03

        trade = _trade(entry_price=entry, stop_loss=3_600.0)
        ti    = _make_ti(atr_val=20.0)
        broker = _make_broker(ltp=current_price)
        db    = _make_db(open_trades=[trade], system_state={})
        agent = ExitAgent(
            config=_make_config(),
            broker_client=broker,
            technical_indicators=ti,
            research_agent=_make_research(),
            db_manager=db,
            llm_router=MagicMock(),
        )
        sigs = [s for s in agent.check_exits() if s["exit_type"] == "TRAILING_STOP_LOSS"]
        assert not sigs

    def test_update_trailing_stops_writes_new_high(self):
        """update_trailing_stops persists a new high to system_state."""
        entry         = 3_800.0
        current_price = entry * 1.05   # 5% above entry — above activation threshold.

        trade  = _trade(entry_price=entry)
        broker = _make_broker(ltp=current_price)
        db     = _make_db(open_trades=[trade], system_state={})
        agent  = _make_agent(broker=broker, db=db, open_trades=[trade])

        agent.update_trailing_stops([trade])

        db.set_system_state.assert_called_once_with(
            "trailing_high_TCS", str(current_price)
        )

    def test_update_trailing_stops_no_write_below_activation(self):
        """update_trailing_stops must NOT write when profit < 2%."""
        entry         = 3_800.0
        current_price = entry * 1.01   # only 1% profit — below threshold.

        trade  = _trade(entry_price=entry)
        broker = _make_broker(ltp=current_price)
        db     = _make_db(open_trades=[trade], system_state={})
        agent  = _make_agent(broker=broker, db=db, open_trades=[trade])

        agent.update_trailing_stops([trade])
        db.set_system_state.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# TIME_BASED_EXIT
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeBased:
    def _trade_held_for_n_days(self, n: int) -> Trade:
        """Return a Trade whose entry_date is *n* calendar days ago (weekdays only)."""
        # Walk back far enough to accumulate n business days.
        end   = datetime.now(IST).date()
        count = 0
        delta = 0
        while count < n:
            delta += 1
            d = end - timedelta(days=delta)
            if d.weekday() < 5:
                count += 1
        entry_date = (datetime.now(IST) - timedelta(days=delta + 1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return _trade(entry_price=3_800.0, stop_loss=3_600.0, entry_date=entry_date)

    def test_no_signal_below_review_threshold(self):
        trade = self._trade_held_for_n_days(10)
        agent = _make_agent(ltp=3_850.0, open_trades=[trade])
        sigs  = [s for s in agent.check_exits() if s["exit_type"] == "TIME_BASED_EXIT"]
        assert not sigs

    def test_signal_at_review_threshold(self):
        trade = self._trade_held_for_n_days(15)
        agent = _make_agent(ltp=3_850.0, open_trades=[trade])
        sigs  = [s for s in agent.check_exits() if s["exit_type"] == "TIME_BASED_EXIT"]
        assert len(sigs) == 1

    def test_signal_at_force_exit_threshold(self):
        trade = self._trade_held_for_n_days(20)
        agent = _make_agent(ltp=3_850.0, open_trades=[trade])
        sigs  = [s for s in agent.check_exits() if s["exit_type"] == "TIME_BASED_EXIT"]
        assert len(sigs) == 1

    def test_time_based_urgency_is_low(self):
        trade = self._trade_held_for_n_days(20)
        agent = _make_agent(ltp=3_850.0, open_trades=[trade])
        sig   = next(s for s in agent.check_exits() if s["exit_type"] == "TIME_BASED_EXIT")
        assert sig["urgency"] == "LOW"

    def test_time_based_current_price_enriched(self):
        trade = self._trade_held_for_n_days(20)
        agent = _make_agent(ltp=3_900.0, open_trades=[trade])
        sig   = next(s for s in agent.check_exits() if s["exit_type"] == "TIME_BASED_EXIT")
        assert sig["current_price"] == 3_900.0


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL_DETERIORATION
# ─────────────────────────────────────────────────────────────────────────────

class TestTechnicalDeterioration:
    def test_two_signals_trigger_exit(self):
        """RSI > 75 + MACD bearish crossover = 2 signals → EXIT."""
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        ti    = _make_ti(rsi_val=78.0, macd_signal="BEARISH_CROSSOVER")
        df    = _ohlcv_df()
        agent = _make_agent(
            ltp=3_850.0,
            open_trades=[trade],
            ti=ti,
            ohlcv_df=df,
        )
        sigs = [s for s in agent.check_exits() if s["exit_type"] == "TECHNICAL_DETERIORATION"]
        assert len(sigs) == 1

    def test_one_signal_does_not_trigger(self):
        """Only RSI overbought — not enough signals."""
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        ti    = _make_ti(rsi_val=78.0, macd_signal="BULLISH")
        df    = _ohlcv_df()
        agent = _make_agent(
            ltp=3_850.0,
            open_trades=[trade],
            ti=ti,
            ohlcv_df=df,
        )
        sigs = [s for s in agent.check_exits() if s["exit_type"] == "TECHNICAL_DETERIORATION"]
        assert not sigs

    def test_rsi_plus_price_below_ema_triggers(self):
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        ti    = _make_ti(rsi_val=77.0, macd_signal="BULLISH", ema_vs_20="BELOW")
        df    = _ohlcv_df()
        agent = _make_agent(
            ltp=3_850.0,
            open_trades=[trade],
            ti=ti,
            ohlcv_df=df,
        )
        sigs = [s for s in agent.check_exits() if s["exit_type"] == "TECHNICAL_DETERIORATION"]
        assert len(sigs) == 1

    def test_volume_spike_on_down_day_triggers(self):
        """MACD bearish crossover + volume spike on down day = 2 signals."""
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        ti    = _make_ti(
            rsi_val=50.0,
            macd_signal="BEARISH_CROSSOVER",
            volume_signal="HIGH_VOLUME",
        )
        ti.calculate_volume_analysis.return_value = {
            "signal": "HIGH_VOLUME",
            "volume_ratio": 2.5,
        }
        # Down candle: last close < previous close.
        df = _ohlcv_df(n=30, last_close=3_750.0)   # previous close ~3800
        agent = _make_agent(
            ltp=3_750.0,
            open_trades=[trade],
            ti=ti,
            ohlcv_df=df,
        )
        sigs = [s for s in agent.check_exits() if s["exit_type"] == "TECHNICAL_DETERIORATION"]
        assert len(sigs) == 1

    def test_technical_urgency_is_medium(self):
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        ti    = _make_ti(rsi_val=78.0, macd_signal="BEARISH_CROSSOVER")
        df    = _ohlcv_df()
        agent = _make_agent(
            ltp=3_850.0,
            open_trades=[trade],
            ti=ti,
            ohlcv_df=df,
        )
        sig = next(
            s for s in agent.check_exits()
            if s["exit_type"] == "TECHNICAL_DETERIORATION"
        )
        assert sig["urgency"] == "MEDIUM"

    def test_technical_current_price_enriched(self):
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        ti    = _make_ti(rsi_val=80.0, macd_signal="BEARISH_CROSSOVER")
        df    = _ohlcv_df()
        agent = _make_agent(
            ltp=3_870.0,
            open_trades=[trade],
            ti=ti,
            ohlcv_df=df,
        )
        sig = next(
            s for s in agent.check_exits()
            if s["exit_type"] == "TECHNICAL_DETERIORATION"
        )
        assert sig["current_price"] == 3_870.0


# ─────────────────────────────────────────────────────────────────────────────
# NEWS_TRIGGERED_EXIT
# ─────────────────────────────────────────────────────────────────────────────

class TestNewsTriggered:
    def _briefing_with_risk(self, symbol: str = "TCS") -> dict:
        return {
            "risky_symbols": [symbol],
            "risky_sectors": [],
            "symbol_sectors": {},
        }

    def test_avoid_recommendation_triggers_exit(self):
        trade    = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        research = _make_research(recommendation="AVOID")
        research.research_stock.return_value = {
            "recommendation":   "AVOID",
            "research_summary": "Negative regulatory news.",
            "risks":            ["Regulatory risk", "Management change"],
        }
        agent = _make_agent(
            ltp=3_850.0,
            open_trades=[trade],
            research=research,
        )
        briefing = self._briefing_with_risk("TCS")
        sigs = agent.check_exits(morning_briefing_data=briefing)
        news_sigs = [s for s in sigs if s["exit_type"] == "NEWS_TRIGGERED_EXIT"]
        assert len(news_sigs) == 1

    def test_news_exit_urgency_is_high(self):
        trade    = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        research = _make_research(recommendation="AVOID")
        research.research_stock.return_value = {
            "recommendation":   "AVOID",
            "research_summary": "Bad news.",
            "risks":            ["R1"],
        }
        agent    = _make_agent(ltp=3_850.0, open_trades=[trade], research=research)
        briefing = self._briefing_with_risk("TCS")
        sig = next(
            s for s in agent.check_exits(morning_briefing_data=briefing)
            if s["exit_type"] == "NEWS_TRIGGERED_EXIT"
        )
        assert sig["urgency"] == "HIGH"

    def test_buy_recommendation_no_exit(self):
        trade    = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        research = _make_research(recommendation="BUY")
        agent    = _make_agent(ltp=3_850.0, open_trades=[trade], research=research)
        briefing = self._briefing_with_risk("TCS")
        sigs     = [
            s for s in agent.check_exits(morning_briefing_data=briefing)
            if s["exit_type"] == "NEWS_TRIGGERED_EXIT"
        ]
        assert not sigs

    def test_no_briefing_data_no_news_check(self):
        """Research agent should NOT be called when morning_briefing_data is None."""
        trade    = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        research = _make_research(recommendation="AVOID")
        agent    = _make_agent(ltp=3_850.0, open_trades=[trade], research=research)
        agent.check_exits(morning_briefing_data=None)
        research.research_stock.assert_not_called()

    def test_symbol_not_in_risky_list_no_news_check(self):
        trade    = _trade(symbol="INFY", entry_price=1_500.0, stop_loss=1_400.0)
        research = _make_research(recommendation="AVOID")
        broker   = _make_broker(ltp=1_520.0)
        broker.get_ltp.return_value = {"symbol": "INFY", "ltp": 1_520.0}
        db       = _make_db(open_trades=[trade])
        agent    = ExitAgent(
            config=_make_config(),
            broker_client=broker,
            technical_indicators=_make_ti(),
            research_agent=research,
            db_manager=db,
            llm_router=MagicMock(),
        )
        briefing = {"risky_symbols": ["TCS"], "risky_sectors": [], "symbol_sectors": {}}
        agent.check_exits(morning_briefing_data=briefing)
        research.research_stock.assert_not_called()

    def test_research_exception_does_not_crash(self):
        trade    = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        research = MagicMock()
        research.research_stock.side_effect = RuntimeError("Network error")
        agent    = _make_agent(ltp=3_850.0, open_trades=[trade], research=research)
        briefing = self._briefing_with_risk("TCS")
        sigs     = agent.check_exits(morning_briefing_data=briefing)
        # Should not crash; news signal silently skipped.
        assert all(s["exit_type"] != "NEWS_TRIGGERED_EXIT" for s in sigs)


# ─────────────────────────────────────────────────────────────────────────────
# check_exits — general behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckExits:
    def test_no_open_trades_returns_empty(self):
        agent = _make_agent(open_trades=[])
        assert agent.check_exits() == []

    def test_ltp_failure_skips_symbol(self):
        trade  = _trade()
        broker = _make_broker()
        broker.get_ltp.side_effect = Exception("Network error")
        db     = _make_db(open_trades=[trade])
        agent  = ExitAgent(
            config=_make_config(),
            broker_client=broker,
            technical_indicators=_make_ti(),
            research_agent=_make_research(),
            db_manager=db,
            llm_router=MagicMock(),
        )
        # Should not raise; returns empty list.
        assert agent.check_exits() == []

    def test_multiple_signals_same_symbol(self):
        """Technical deterioration + time-based can co-exist for one position."""
        n_days = 20
        trade  = TestTimeBased()._trade_held_for_n_days(n_days)
        trade.stop_loss = 3_600.0
        ti     = _make_ti(rsi_val=78.0, macd_signal="BEARISH_CROSSOVER")
        df     = _ohlcv_df()
        agent  = _make_agent(
            ltp=3_850.0,
            open_trades=[trade],
            ti=ti,
            ohlcv_df=df,
        )
        types = {s["exit_type"] for s in agent.check_exits()}
        assert "TECHNICAL_DETERIORATION" in types
        assert "TIME_BASED_EXIT" in types

    def test_all_signals_have_required_keys(self):
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        agent = _make_agent(ltp=3_550.0, open_trades=[trade])
        for sig in agent.check_exits():
            for key in (
                "symbol", "action", "exit_type", "current_price",
                "entry_price", "pnl", "pnl_pct", "holding_days",
                "urgency", "reasoning", "sell_quantity",
            ):
                assert key in sig, f"Missing key '{key}' in signal: {sig}"

    def test_action_is_always_sell(self):
        trade = _trade(entry_price=3_800.0, stop_loss=3_600.0)
        agent = _make_agent(ltp=3_550.0, open_trades=[trade])
        for sig in agent.check_exits():
            assert sig["action"] == "SELL"


# ─────────────────────────────────────────────────────────────────────────────
# should_exit convenience method
# ─────────────────────────────────────────────────────────────────────────────

class TestShouldExit:
    def _position(self, entry: float, stop: float, target: float) -> dict:
        return {
            "symbol":       "TCS",
            "entry_price":  entry,
            "stop_loss":    stop,
            "target_1":     target,
            "quantity":     10,
            "entry_date":   "2025-01-01 09:15:00",
        }

    def test_returns_true_when_stop_hit(self):
        agent = _make_agent(open_trades=[])
        pos   = self._position(3_800.0, 3_600.0, 4_000.0)
        assert agent.should_exit(pos, current_price=3_550.0) is True

    def test_returns_false_no_trigger(self):
        agent = _make_agent(open_trades=[])
        pos   = self._position(3_800.0, 3_600.0, 4_000.0)
        # Use a recent entry_date so time-based exit does not fire.
        pos["entry_date"] = (datetime.now(IST) - timedelta(days=3)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        assert agent.should_exit(pos, current_price=3_850.0) is False

    def test_returns_true_when_target_hit(self):
        agent = _make_agent(open_trades=[])
        pos   = self._position(3_800.0, 3_600.0, 4_000.0)
        assert agent.should_exit(pos, current_price=4_050.0) is True


# ─────────────────────────────────────────────────────────────────────────────
# update_trailing_stop (single-position variant)
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateTrailingStopSingle:
    def test_returns_floor_below_activation(self):
        """Below 2% profit → return 5% below entry as a conservative floor."""
        entry = 3_800.0
        db    = _make_db(system_state={})
        agent = _make_agent(db=db, open_trades=[])
        pos   = {"symbol": "TCS", "entry_price": entry, "quantity": 10}
        result = agent.update_trailing_stop(pos, current_price=entry * 1.01)
        assert result == round(entry * 0.95, 2)

    def test_persists_new_high(self):
        db    = _make_db(system_state={})
        agent = _make_agent(db=db, open_trades=[])
        pos   = {"symbol": "TCS", "entry_price": 3_800.0, "quantity": 10}
        agent.update_trailing_stop(pos, current_price=3_900.0)
        db.set_system_state.assert_called_once_with("trailing_high_TCS", str(3_900.0))

    def test_trailing_stop_value_is_high_minus_atr(self):
        high = 3_900.0
        atr  = 20.0
        db     = _make_db(system_state={"trailing_high_TCS": str(high)})
        ti     = _make_ti(atr_val=atr)
        df     = _ohlcv_df()   # non-empty so get_current_atr calls ti, not fallback
        broker = _make_broker(ltp=3_870.0, df=df)
        agent  = ExitAgent(
            config=_make_config(),
            broker_client=broker,
            technical_indicators=ti,
            research_agent=_make_research(),
            db_manager=db,
            llm_router=MagicMock(),
        )
        pos    = {"symbol": "TCS", "entry_price": 3_800.0, "quantity": 10}
        result = agent.update_trailing_stop(pos, current_price=3_870.0)
        assert result == round(high - 1.5 * atr, 2)   # 3_870.0
