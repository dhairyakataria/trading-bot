"""Tests for QuantAgent — technical analysis signal generator.

Strategy checkers receive pre-built ``inds`` dicts (mocked indicator output)
so tests are fast (<< 1 s each) and deterministic — no broker calls required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.quant_agent import QuantAgent


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 60,
    close: float = 1000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with constant close price."""
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    closes = np.full(n, close) + rng.normal(0, 1, n)
    closes[-1] = close  # pin last bar to exact value
    high   = closes + 15.0
    low    = closes - 15.0
    return pd.DataFrame(
        {
            "datetime": dates,
            "open":     closes + 3.0,
            "high":     high,
            "low":      low,
            "close":    closes,
            "volume":   np.full(n, 2_000_000.0),
        }
    )


def _inds(
    rsi_val: float = 50.0,
    macd_sig: str = "BULLISH",
    macd_hist: float = 1.0,
    ema_20: float = 980.0,
    ema_50: float = 950.0,
    price_vs_20: str = "ABOVE",
    price_vs_50: str = "ABOVE",
    price_vs_200: str = "ABOVE",
    vol_ratio: float = 1.0,
    atr_val: float = 25.0,
    atr_pct: float = 2.5,
    resistance_1: float = 1050.0,
    resistance_2: float = 1100.0,
    support_1: float = 950.0,
    support_2: float = 920.0,
) -> dict:
    """Build a minimal ``indicators`` dict accepted by all strategy checkers."""
    return {
        "rsi": {"value": rsi_val, "signal": "NEUTRAL"},
        "macd": {
            "signal":    macd_sig,
            "histogram": macd_hist,
            "macd_line": 2.0,
            "signal_line": 1.5,
        },
        "ema": {
            "ema_20": ema_20,
            "ema_50": ema_50,
            "ema_200": 900.0,
            "price_vs_ema_20":  price_vs_20,
            "price_vs_ema_50":  price_vs_50,
            "price_vs_ema_200": price_vs_200,
            "signal": "STRONG_UPTREND",
        },
        "volume": {
            "volume_ratio": vol_ratio,
            "current_volume": int(vol_ratio * 2_000_000),
            "avg_volume": 2_000_000,
            "signal": "NORMAL_VOLUME",
        },
        "atr": {"atr": atr_val, "atr_pct": atr_pct},
        "bollinger": {"signal": "ABOVE_MIDDLE"},
        "support_resistance": {
            "resistance_1": resistance_1,
            "resistance_2": resistance_2,
            "support_1":    support_1,
            "support_2":    support_2,
            "current_price": 1000.0,
        },
        "vwap": {"vwap": 998.0, "signal": "ABOVE_VWAP"},
    }


@pytest.fixture
def agent() -> QuantAgent:
    ti_mock = MagicMock()
    db_mock = MagicMock()
    db_mock.record_signal.return_value = 1
    return QuantAgent(
        config={},
        broker_client=MagicMock(),
        technical_indicators=ti_mock,
        db_manager=db_mock,
    )


# ---------------------------------------------------------------------------
# Risk management helpers
# ---------------------------------------------------------------------------


class TestCalculateRiskReward:
    def test_standard_2_to_1(self, agent):
        assert agent.calculate_risk_reward(1000, 950, 1100) == pytest.approx(2.0)

    def test_below_1(self, agent):
        assert agent.calculate_risk_reward(1000, 950, 1030) == pytest.approx(0.6)

    def test_zero_risk_returns_zero(self, agent):
        assert agent.calculate_risk_reward(1000, 1000, 1100) == 0.0

    def test_negative_risk_returns_zero(self, agent):
        # stop_loss above entry — degenerate input
        assert agent.calculate_risk_reward(1000, 1050, 1100) == 0.0


class TestCalculateStopLoss:
    def test_rsi_oversold_bounce(self, agent):
        # price - 2*ATR
        assert agent.calculate_stop_loss("RSI_OVERSOLD_BOUNCE", 1000, 50) == pytest.approx(900.0)

    def test_ema_pullback_uses_ema50(self, agent):
        # ema_50 * 0.995 = 950 * 0.995 = 945.25  <  price - 2*atr = 1000 - 100 = 900
        # → min(900, 945.25) = 900
        sl = agent.calculate_stop_loss("EMA_PULLBACK", 1000, 50, ema_50=950)
        assert sl == pytest.approx(900.0)

    def test_ema_pullback_ema50_closer_than_atr(self, agent):
        # ema_50=990 → 990*0.995=985.05   price-2*atr=1000-10=990 → min(990,985.05)=985.05
        sl = agent.calculate_stop_loss("EMA_PULLBACK", 1000, 5, ema_50=990)
        assert sl == pytest.approx(985.05)

    def test_volume_breakout(self, agent):
        sl = agent.calculate_stop_loss("VOLUME_BREAKOUT", 1020, 20, resistance_1=1010)
        assert sl == pytest.approx(1010 * 0.995)

    def test_trend_following_uses_ema20(self, agent):
        sl = agent.calculate_stop_loss("TREND_FOLLOWING", 1100, 30, ema_20=1060)
        assert sl == pytest.approx(1060.0)

    def test_unknown_strategy_fallback(self, agent):
        assert agent.calculate_stop_loss("UNKNOWN", 1000, 40) == pytest.approx(920.0)


class TestCalculateTargets:
    def test_rsi_oversold_bounce(self, agent):
        t1, t2 = agent.calculate_targets("RSI_OVERSOLD_BOUNCE", 1000, 50, {})
        assert t1 == pytest.approx(1150.0)   # price + 3×ATR
        assert t2 == pytest.approx(1250.0)   # price + 5×ATR
        assert t2 > t1

    def test_ema_pullback_uses_resistance(self, agent):
        t1, t2 = agent.calculate_targets(
            "EMA_PULLBACK", 1000, 50, {"resistance_1": 1080.0, "resistance_2": 1150.0}
        )
        assert t1 == pytest.approx(1080.0)
        assert t2 == pytest.approx(1150.0)

    def test_ema_pullback_fallback_when_no_resistance(self, agent):
        t1, t2 = agent.calculate_targets("EMA_PULLBACK", 1000, 50, {})
        assert t1 == pytest.approx(1125.0)   # price + 2.5×ATR
        assert t2 == pytest.approx(1200.0)   # price + 4.0×ATR

    def test_volume_breakout_uses_r2(self, agent):
        t1, t2 = agent.calculate_targets(
            "VOLUME_BREAKOUT", 1020, 20, {"resistance_2": 1100.0}
        )
        assert t1 == pytest.approx(1100.0)

    def test_t2_always_greater_than_t1(self, agent):
        # edge case where ATR is tiny — t2 bump kicks in
        t1, t2 = agent.calculate_targets("RSI_OVERSOLD_BOUNCE", 1000, 0.01, {})
        assert t2 > t1


# ---------------------------------------------------------------------------
# Strategy 1 — RSI_OVERSOLD_BOUNCE
# ---------------------------------------------------------------------------


class TestRsiOversoldBounce:
    def test_valid_setup_returns_buy(self, agent):
        df = _make_df(close=1000.0)
        inds = _inds(rsi_val=28.5, macd_sig="BULLISH_CROSSOVER", macd_hist=1.5, vol_ratio=1.5)
        result = agent.check_rsi_oversold_bounce(df, "TEST", inds)
        assert result is not None
        assert result["signal"] == "BUY"
        assert result["strategy_name"] == "RSI_OVERSOLD_BOUNCE"
        assert result["stop_loss"] < result["entry_price"]
        assert result["target_1"] > result["entry_price"]
        assert result["target_2"] > result["target_1"]

    def test_rejects_rsi_at_35(self, agent):
        df = _make_df(close=1000.0)
        inds = _inds(rsi_val=35.0, macd_sig="BULLISH_CROSSOVER", vol_ratio=1.5)
        assert agent.check_rsi_oversold_bounce(df, "TEST", inds) is None

    def test_rejects_rsi_above_35(self, agent):
        df = _make_df(close=1000.0)
        inds = _inds(rsi_val=42.0, macd_sig="BULLISH_CROSSOVER", vol_ratio=1.5)
        assert agent.check_rsi_oversold_bounce(df, "TEST", inds) is None

    def test_rejects_price_below_ema50(self, agent):
        df = _make_df(close=1000.0)
        inds = _inds(rsi_val=28.0, price_vs_50="BELOW", vol_ratio=1.5)
        assert agent.check_rsi_oversold_bounce(df, "TEST", inds) is None

    def test_rejects_bearish_macd(self, agent):
        df = _make_df(close=1000.0)
        inds = _inds(rsi_val=28.0, macd_sig="BEARISH", macd_hist=-1.0, vol_ratio=1.5)
        assert agent.check_rsi_oversold_bounce(df, "TEST", inds) is None

    def test_rejects_low_volume(self, agent):
        df = _make_df(close=1000.0)
        inds = _inds(rsi_val=28.0, macd_sig="BULLISH_CROSSOVER", vol_ratio=1.1)
        assert agent.check_rsi_oversold_bounce(df, "TEST", inds) is None

    def test_accepts_positive_histogram_without_crossover(self, agent):
        """Positive histogram alone satisfies the MACD condition."""
        df = _make_df(close=1000.0)
        inds = _inds(rsi_val=30.0, macd_sig="BULLISH", macd_hist=0.5, vol_ratio=1.3)
        assert agent.check_rsi_oversold_bounce(df, "TEST", inds) is not None

    def test_stop_loss_is_price_minus_2atr(self, agent):
        df = _make_df(close=1000.0)
        inds = _inds(rsi_val=28.0, macd_sig="BULLISH_CROSSOVER", vol_ratio=1.5, atr_val=30.0)
        result = agent.check_rsi_oversold_bounce(df, "TEST", inds)
        assert result["stop_loss"] == pytest.approx(1000.0 - 2 * 30.0)


# ---------------------------------------------------------------------------
# Strategy 2 — EMA_PULLBACK
# ---------------------------------------------------------------------------


class TestEmaPullback:
    def test_valid_setup_returns_buy(self, agent):
        # price=1005, ema_20=1000 → 0.5% apart
        df = _make_df(close=1005.0)
        inds = _inds(
            rsi_val=48.0, ema_20=1000.0, ema_50=950.0,
            price_vs_20="ABOVE", price_vs_50="ABOVE", vol_ratio=0.7,
        )
        result = agent.check_ema_pullback(df, "TEST", inds)
        assert result is not None
        assert result["signal"] == "BUY"
        assert result["strategy_name"] == "EMA_PULLBACK"

    def test_rejects_price_too_far_from_ema20(self, agent):
        df = _make_df(close=1020.0)  # 2% above EMA20=1000
        inds = _inds(rsi_val=48.0, ema_20=1000.0, ema_50=950.0, price_vs_50="ABOVE", vol_ratio=0.7)
        assert agent.check_ema_pullback(df, "TEST", inds) is None

    def test_rejects_price_below_ema50(self, agent):
        df = _make_df(close=1005.0)
        inds = _inds(rsi_val=48.0, ema_20=1000.0, ema_50=1100.0, price_vs_50="BELOW", vol_ratio=0.7)
        assert agent.check_ema_pullback(df, "TEST", inds) is None

    def test_rejects_rsi_above_55(self, agent):
        df = _make_df(close=1005.0)
        inds = _inds(rsi_val=60.0, ema_20=1000.0, ema_50=950.0, price_vs_50="ABOVE", vol_ratio=0.7)
        assert agent.check_ema_pullback(df, "TEST", inds) is None

    def test_rejects_rsi_below_40(self, agent):
        df = _make_df(close=1005.0)
        inds = _inds(rsi_val=35.0, ema_20=1000.0, ema_50=950.0, price_vs_50="ABOVE", vol_ratio=0.7)
        assert agent.check_ema_pullback(df, "TEST", inds) is None

    def test_rejects_high_volume_pullback(self, agent):
        """Volume at or above average = distribution, not healthy pullback."""
        df = _make_df(close=1005.0)
        inds = _inds(rsi_val=48.0, ema_20=1000.0, ema_50=950.0, price_vs_50="ABOVE", vol_ratio=1.0)
        assert agent.check_ema_pullback(df, "TEST", inds) is None

    def test_uses_resistance_for_target(self, agent):
        df = _make_df(close=1005.0)
        inds = _inds(
            rsi_val=48.0, ema_20=1000.0, ema_50=950.0,
            price_vs_50="ABOVE", vol_ratio=0.7,
            resistance_1=1060.0, resistance_2=1120.0,
        )
        result = agent.check_ema_pullback(df, "TEST", inds)
        assert result is not None
        assert result["target_1"] == pytest.approx(1060.0)
        assert result["target_2"] == pytest.approx(1120.0)


# ---------------------------------------------------------------------------
# Strategy 3 — VOLUME_BREAKOUT
# ---------------------------------------------------------------------------


class TestVolumeBreakout:
    def test_valid_breakout_returns_buy(self, agent):
        df = _make_df(close=1020.0)
        inds = _inds(
            rsi_val=58.0, macd_sig="BULLISH", macd_hist=2.0, vol_ratio=2.5,
            resistance_1=1010.0, resistance_2=1060.0,
        )
        result = agent.check_volume_breakout(df, "TEST", inds)
        assert result is not None
        assert result["signal"] == "BUY"
        assert result["strategy_name"] == "VOLUME_BREAKOUT"
        assert result["target_1"] == pytest.approx(1060.0)

    def test_rejects_price_below_resistance(self, agent):
        df = _make_df(close=990.0)  # below resistance_1 = 1010
        inds = _inds(rsi_val=55.0, macd_sig="BULLISH", vol_ratio=2.5, resistance_1=1010.0)
        assert agent.check_volume_breakout(df, "TEST", inds) is None

    def test_rejects_price_equal_to_resistance(self, agent):
        df = _make_df(close=1010.0)  # exactly at resistance — not a breakout
        inds = _inds(rsi_val=55.0, macd_sig="BULLISH", vol_ratio=2.5, resistance_1=1010.0)
        assert agent.check_volume_breakout(df, "TEST", inds) is None

    def test_rejects_low_volume(self, agent):
        df = _make_df(close=1020.0)
        inds = _inds(rsi_val=55.0, macd_sig="BULLISH", vol_ratio=1.8, resistance_1=1010.0)
        assert agent.check_volume_breakout(df, "TEST", inds) is None

    def test_rejects_overbought_rsi(self, agent):
        df = _make_df(close=1020.0)
        inds = _inds(rsi_val=67.0, macd_sig="BULLISH", vol_ratio=2.5, resistance_1=1010.0)
        assert agent.check_volume_breakout(df, "TEST", inds) is None

    def test_rejects_bearish_macd(self, agent):
        df = _make_df(close=1020.0)
        inds = _inds(rsi_val=55.0, macd_sig="BEARISH", vol_ratio=2.5, resistance_1=1010.0)
        assert agent.check_volume_breakout(df, "TEST", inds) is None

    def test_stop_loss_is_below_resistance(self, agent):
        df = _make_df(close=1020.0)
        inds = _inds(rsi_val=55.0, macd_sig="BULLISH", vol_ratio=2.5, resistance_1=1010.0)
        result = agent.check_volume_breakout(df, "TEST", inds)
        assert result["stop_loss"] == pytest.approx(1010.0 * 0.995)
        assert result["stop_loss"] < result["entry_price"]

    def test_accepts_bullish_crossover_macd(self, agent):
        df = _make_df(close=1020.0)
        inds = _inds(
            rsi_val=55.0, macd_sig="BULLISH_CROSSOVER", vol_ratio=2.5, resistance_1=1010.0
        )
        assert agent.check_volume_breakout(df, "TEST", inds) is not None


# ---------------------------------------------------------------------------
# Strategy 4 — TREND_FOLLOWING
# ---------------------------------------------------------------------------


class TestTrendFollowing:
    def test_valid_setup_returns_buy(self, agent):
        df = _make_df(close=1100.0)
        inds = _inds(
            rsi_val=58.0, macd_hist=3.0, macd_sig="BULLISH",
            ema_20=1070.0, ema_50=1040.0,
            price_vs_20="ABOVE", price_vs_50="ABOVE",
            atr_val=30.0, atr_pct=2.7,
        )
        result = agent.check_trend_following(df, "TEST", inds)
        assert result is not None
        assert result["signal"] == "BUY"
        assert result["strategy_name"] == "TREND_FOLLOWING"

    def test_rejects_ema20_below_ema50(self, agent):
        df = _make_df(close=1100.0)
        inds = _inds(
            rsi_val=58.0, macd_hist=3.0,
            ema_20=1040.0, ema_50=1070.0,  # EMA20 < EMA50
            price_vs_20="ABOVE", price_vs_50="ABOVE", atr_pct=2.7,
        )
        assert agent.check_trend_following(df, "TEST", inds) is None

    def test_rejects_price_below_ema20(self, agent):
        df = _make_df(close=1060.0)
        inds = _inds(
            rsi_val=58.0, macd_hist=3.0,
            ema_20=1070.0, ema_50=1040.0,
            price_vs_20="BELOW",  # price under EMA20
            price_vs_50="ABOVE", atr_pct=2.7,
        )
        assert agent.check_trend_following(df, "TEST", inds) is None

    def test_rejects_negative_macd_histogram(self, agent):
        df = _make_df(close=1100.0)
        inds = _inds(
            rsi_val=58.0, macd_hist=-0.5,
            ema_20=1070.0, ema_50=1040.0,
            price_vs_20="ABOVE", price_vs_50="ABOVE", atr_pct=2.7,
        )
        assert agent.check_trend_following(df, "TEST", inds) is None

    def test_rejects_rsi_above_65(self, agent):
        df = _make_df(close=1100.0)
        inds = _inds(
            rsi_val=67.0, macd_hist=3.0,
            ema_20=1070.0, ema_50=1040.0,
            price_vs_20="ABOVE", price_vs_50="ABOVE", atr_pct=2.7,
        )
        assert agent.check_trend_following(df, "TEST", inds) is None

    def test_rejects_rsi_below_50(self, agent):
        df = _make_df(close=1100.0)
        inds = _inds(
            rsi_val=47.0, macd_hist=3.0,
            ema_20=1070.0, ema_50=1040.0,
            price_vs_20="ABOVE", price_vs_50="ABOVE", atr_pct=2.7,
        )
        assert agent.check_trend_following(df, "TEST", inds) is None

    def test_rejects_low_atr(self, agent):
        df = _make_df(close=1100.0)
        inds = _inds(
            rsi_val=58.0, macd_hist=3.0,
            ema_20=1070.0, ema_50=1040.0,
            price_vs_20="ABOVE", price_vs_50="ABOVE",
            atr_pct=1.8,  # below 2%
        )
        assert agent.check_trend_following(df, "TEST", inds) is None

    def test_stop_loss_equals_ema20(self, agent):
        df = _make_df(close=1100.0)
        inds = _inds(
            rsi_val=58.0, macd_hist=3.0, macd_sig="BULLISH",
            ema_20=1070.0, ema_50=1040.0,
            price_vs_20="ABOVE", price_vs_50="ABOVE",
            atr_val=30.0, atr_pct=2.7,
        )
        result = agent.check_trend_following(df, "TEST", inds)
        assert result["stop_loss"] == pytest.approx(1070.0)


# ---------------------------------------------------------------------------
# Strategy 5 — EXIT_SIGNAL (SELL)
# ---------------------------------------------------------------------------


class TestExitSignals:
    def test_overbought_rsi_triggers_sell(self, agent):
        df = _make_df(close=1200.0)
        inds = _inds(rsi_val=75.0, macd_sig="BULLISH", macd_hist=1.0, vol_ratio=1.0)
        result = agent.check_exit_signals(df, "TEST", ["TEST"], inds)
        assert result is not None
        assert result["signal"] == "SELL"

    def test_macd_bearish_crossover_triggers_sell(self, agent):
        df = _make_df(close=1200.0)
        inds = _inds(rsi_val=55.0, macd_sig="BEARISH_CROSSOVER", macd_hist=-0.5, vol_ratio=1.0)
        result = agent.check_exit_signals(df, "TEST", ["TEST"], inds)
        assert result is not None
        assert result["signal"] == "SELL"

    def test_ema20_break_with_high_volume_triggers_sell(self, agent):
        df = _make_df(close=980.0)
        inds = _inds(
            rsi_val=45.0, macd_sig="BEARISH", macd_hist=-1.0,
            ema_20=1000.0, price_vs_20="BELOW", vol_ratio=1.8,
        )
        result = agent.check_exit_signals(df, "TEST", ["TEST"], inds)
        assert result is not None
        assert result["signal"] == "SELL"

    def test_multiple_conditions_accumulate_strength(self, agent):
        df = _make_df(close=1200.0)
        inds = _inds(
            rsi_val=75.0, macd_sig="BEARISH_CROSSOVER", macd_hist=-1.0, vol_ratio=1.0
        )
        result = agent.check_exit_signals(df, "TEST", ["TEST"], inds)
        # MACD crossover (0.40) + RSI overbought (0.35) = 0.75
        assert result["strength"] == pytest.approx(0.75)

    def test_no_exit_when_no_conditions(self, agent):
        df = _make_df(close=1100.0)
        inds = _inds(rsi_val=55.0, macd_sig="BULLISH", macd_hist=1.0, price_vs_20="ABOVE", vol_ratio=1.0)
        assert agent.check_exit_signals(df, "TEST", ["TEST"], inds) is None

    def test_not_held_suppressed_when_weak(self, agent):
        """Un-held stock with only RSI >70 (strength 0.35) — should be suppressed."""
        df = _make_df(close=1200.0)
        inds = _inds(rsi_val=72.0, macd_sig="BULLISH", macd_hist=1.0, vol_ratio=1.0)
        result = agent.check_exit_signals(df, "TEST", [], inds)  # not in holdings
        assert result is None

    def test_not_held_returned_when_strong(self, agent):
        """Un-held stock with RSI + MACD crossover (0.75 ≥ 0.70) — should be returned."""
        df = _make_df(close=1200.0)
        inds = _inds(rsi_val=75.0, macd_sig="BEARISH_CROSSOVER", macd_hist=-1.0, vol_ratio=1.0)
        result = agent.check_exit_signals(df, "TEST", [], inds)
        assert result is not None
        assert result["signal"] == "SELL"

    def test_strength_capped_at_1(self, agent):
        df = _make_df(close=980.0)
        inds = _inds(
            rsi_val=75.0, macd_sig="BEARISH_CROSSOVER", macd_hist=-1.0,
            price_vs_20="BELOW", vol_ratio=2.0,  # all three conditions
        )
        result = agent.check_exit_signals(df, "TEST", ["TEST"], inds)
        assert result["strength"] <= 1.0


# ---------------------------------------------------------------------------
# Signal strength combining
# ---------------------------------------------------------------------------


class TestCombineBuySignals:
    def _sig(self, strategy: str, strength: float) -> dict:
        return {
            "signal":           "BUY",
            "strength":         strength,
            "strategy_name":    strategy,
            "reasons":          [f"reason for {strategy}"],
            "entry_price":      1000.0,
            "stop_loss":        940.0,
            "target_1":         1090.0,
            "target_2":         1150.0,
            "risk_reward_ratio": 1.5,
            "indicators":       {},
            "timestamp":        "2026-01-01 10:00:00",
        }

    def test_single_signal_unchanged(self, agent):
        result = agent._combine_buy_signals([self._sig("RSI_OVERSOLD_BOUNCE", 0.70)])
        assert result["strength"] == pytest.approx(0.70)
        assert result["strategy_name"] == "RSI_OVERSOLD_BOUNCE"

    def test_two_signals_add_boost(self, agent):
        signals = [
            self._sig("RSI_OVERSOLD_BOUNCE", 0.70),
            self._sig("VOLUME_BREAKOUT",     0.75),
        ]
        result = agent._combine_buy_signals(signals)
        # Primary = 0.75 (strongest) + 0.30 boost = 1.05 → capped at 1.0
        assert result["strength"] == pytest.approx(1.0)

    def test_three_signals_capped_at_1(self, agent):
        signals = [
            self._sig("RSI_OVERSOLD_BOUNCE", 0.70),
            self._sig("EMA_PULLBACK",        0.65),
            self._sig("TREND_FOLLOWING",     0.60),
        ]
        result = agent._combine_buy_signals(signals)
        assert result["strength"] == pytest.approx(1.0)

    def test_strategy_name_concatenated(self, agent):
        signals = [
            self._sig("EMA_PULLBACK",    0.65),
            self._sig("TREND_FOLLOWING", 0.60),
        ]
        result = agent._combine_buy_signals(signals)
        assert "EMA_PULLBACK" in result["strategy_name"]
        assert "TREND_FOLLOWING" in result["strategy_name"]

    def test_all_reasons_merged(self, agent):
        signals = [
            self._sig("RSI_OVERSOLD_BOUNCE", 0.70),
            self._sig("VOLUME_BREAKOUT",     0.75),
        ]
        result = agent._combine_buy_signals(signals)
        assert len(result["reasons"]) == 2  # one reason per signal
        texts = " ".join(result["reasons"])
        assert "RSI_OVERSOLD_BOUNCE" in texts
        assert "VOLUME_BREAKOUT" in texts

    def test_primary_is_highest_strength(self, agent):
        """Primary signal (highest strength) should provide entry/stop data."""
        signals = [
            {**self._sig("EMA_PULLBACK", 0.65), "entry_price": 1000.0, "stop_loss": 940.0},
            {**self._sig("VOLUME_BREAKOUT", 0.75), "entry_price": 1010.0, "stop_loss": 980.0},
        ]
        result = agent._combine_buy_signals(signals)
        # Primary is VOLUME_BREAKOUT (0.75)
        assert result["entry_price"] == pytest.approx(1010.0)
        assert result["stop_loss"] == pytest.approx(980.0)


# ---------------------------------------------------------------------------
# scan_watchlist — integration / orchestration tests
# ---------------------------------------------------------------------------


class TestScanWatchlist:
    def _setup_agent_with_mock_analysis(self, agent, inds_dict: dict) -> None:
        """Patch generate_full_analysis to return *inds_dict* for any stock."""
        def mock_analysis(df, symbol):
            return {
                "symbol":   symbol,
                "price":    float(df["close"].iloc[-1]),
                "indicators": inds_dict,
                "overall_signal": "HOLD",
            }
        agent.ti.generate_full_analysis = mock_analysis
        agent.broker.get_historical_data.return_value = _make_df(n=60)

    def test_hold_stocks_not_returned(self, agent):
        """When no strategy fires, result list is empty."""
        # RSI=50, MACD bullish but histogram tiny, vol ratio normal — nothing fires
        self._setup_agent_with_mock_analysis(
            agent, _inds(rsi_val=50.0, macd_hist=0.5, vol_ratio=1.0, atr_pct=1.5)
        )
        results = agent.scan_watchlist([{"symbol": "TCS"}, {"symbol": "INFY"}])
        assert results == []

    def test_held_stock_no_buy_signal(self, agent):
        """BUY signals must be suppressed for stocks in current_holdings."""
        # Conditions perfect for RSI_OVERSOLD_BOUNCE
        self._setup_agent_with_mock_analysis(
            agent,
            _inds(rsi_val=28.0, macd_sig="BULLISH_CROSSOVER", macd_hist=2.0, vol_ratio=1.5),
        )
        results = agent.scan_watchlist([{"symbol": "TCS"}], current_holdings=["TCS"])
        buys = [r for r in results if r["signal"] == "BUY"]
        assert buys == []

    def test_buy_signal_returned_for_unheld_stock(self, agent):
        """A qualifying BUY is returned when the stock is not held."""
        self._setup_agent_with_mock_analysis(
            agent,
            _inds(rsi_val=28.0, macd_sig="BULLISH_CROSSOVER", macd_hist=2.0, vol_ratio=1.5),
        )
        results = agent.scan_watchlist([{"symbol": "TCS"}], current_holdings=[])
        buys = [r for r in results if r["signal"] == "BUY"]
        assert len(buys) == 1
        assert buys[0]["symbol"] == "TCS"

    def test_buy_rejected_when_rr_below_1(self, agent):
        """Signals with risk:reward ≤ 1.0 must NOT be returned."""
        df = _make_df(close=1000.0)
        agent.broker.get_historical_data.return_value = df

        bad_signal = {
            "symbol": "TEST", "signal": "BUY", "strength": 0.70,
            "entry_price": 1000.0, "stop_loss": 990.0, "target_1": 1005.0,
            "target_2": 1010.0, "risk_reward_ratio": 0.5,
            "reasons": [], "indicators": {}, "strategy_name": "RSI_OVERSOLD_BOUNCE",
            "timestamp": "2026-01-01 10:00:00",
        }
        agent.check_rsi_oversold_bounce = MagicMock(return_value=bad_signal)
        agent.check_ema_pullback        = MagicMock(return_value=None)
        agent.check_volume_breakout     = MagicMock(return_value=None)
        agent.check_trend_following     = MagicMock(return_value=None)
        agent.check_exit_signals        = MagicMock(return_value=None)
        agent.ti.generate_full_analysis = MagicMock(return_value={
            "symbol": "TEST", "price": 1000.0,
            "indicators": _inds(), "overall_signal": "BUY",
        })

        results = agent.scan_watchlist([{"symbol": "TEST"}])
        assert results == []

    def test_signal_stored_in_db(self, agent):
        """Every returned signal must be persisted via record_signal."""
        self._setup_agent_with_mock_analysis(
            agent,
            _inds(rsi_val=28.0, macd_sig="BULLISH_CROSSOVER", macd_hist=2.0, vol_ratio=1.5),
        )
        results = agent.scan_watchlist([{"symbol": "TCS"}])
        if results:  # only check if a signal was actually generated
            agent.db.record_signal.assert_called()

    def test_duplicate_buy_for_same_symbol_not_returned(self, agent):
        """Each symbol appears at most once with a BUY in a single scan."""
        self._setup_agent_with_mock_analysis(
            agent,
            _inds(rsi_val=28.0, macd_sig="BULLISH_CROSSOVER", macd_hist=2.0, vol_ratio=1.5),
        )
        # Same symbol twice in watchlist
        results = agent.scan_watchlist([{"symbol": "TCS"}, {"symbol": "TCS"}])
        buys = [r for r in results if r["signal"] == "BUY" and r["symbol"] == "TCS"]
        assert len(buys) <= 1

    def test_missing_data_skipped_gracefully(self, agent):
        """Symbols with insufficient data are silently skipped."""
        # Broker returns empty DataFrame
        agent.broker.get_historical_data.return_value = pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        results = agent.scan_watchlist([{"symbol": "TCS"}])
        assert results == []

    def test_symbol_without_key_skipped(self, agent):
        """Watchlist entries without a 'symbol' key are skipped."""
        results = agent.scan_watchlist([{"name": "TCS"}])  # wrong key
        assert results == []

    def test_multiple_stocks_independent(self, agent):
        """Signals for different stocks are independent."""
        def mock_data(symbol, **kwargs):
            return _make_df(close=1000.0)

        agent.broker.get_historical_data.side_effect = mock_data

        call_count = [0]
        def mock_analysis(df, symbol):
            call_count[0] += 1
            return {
                "symbol": symbol, "price": 1000.0,
                "indicators": _inds(rsi_val=28.0, macd_sig="BULLISH_CROSSOVER",
                                    macd_hist=2.0, vol_ratio=1.5),
                "overall_signal": "BUY",
            }

        agent.ti.generate_full_analysis = mock_analysis
        results = agent.scan_watchlist(
            [{"symbol": "TCS"}, {"symbol": "INFY"}, {"symbol": "HDFCBANK"}]
        )
        # Each stock analysed independently
        assert call_count[0] == 3
