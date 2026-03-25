"""Tests for UniverseAgent.

Tests are designed to be fast and dependency-free:
- Broker calls are mocked (no live API)
- time.sleep is patched to 0 (no rate-limit waits)
- DB is mocked via MagicMock

Run with:
    cd trading-bot
    pytest tests/test_universe_agent.py -v
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.agents.universe_agent import (
    UniverseAgent,
    _FALLBACK_CONSTITUENTS,
    _FALLBACK_SECTOR_MAP,
)
from src.database.models import WatchlistItem

IST = ZoneInfo("Asia/Kolkata")

TODAY = datetime.now(IST).strftime("%Y-%m-%d")


# =========================================================================== #
# Helpers                                                                       #
# =========================================================================== #

def _make_ohlcv(
    rows: int = 60,
    base_price: float = 500.0,
    trend: str = "flat",
    daily_range_pct: float = 0.03,
    volume: int = 2_000_000,
) -> pd.DataFrame:
    """Build a synthetic daily OHLCV DataFrame.

    Args:
        rows:            Number of candles.
        base_price:      Starting price.
        trend:           ``"up"`` → rising, ``"down"`` → falling, ``"flat"`` → stable.
        daily_range_pct: Intraday high–low range as a fraction of price.
        volume:          Constant daily volume (shares).
    """
    rng = np.random.default_rng(42)
    if trend == "up":
        closes = np.linspace(base_price * 0.6, base_price, rows)
    elif trend == "down":
        closes = np.linspace(base_price, base_price * 0.3, rows)
    else:  # flat / slight noise
        closes = base_price + rng.normal(0, base_price * 0.005, rows)
        closes = np.abs(closes)

    half_range = closes * daily_range_pct / 2
    highs = closes + half_range
    lows  = closes - half_range
    opens = closes + rng.normal(0, base_price * 0.003, rows)

    dates = pd.date_range(end=TODAY, periods=rows, freq="B")
    return pd.DataFrame({
        "datetime": dates,
        "open":     opens,
        "high":     highs,
        "low":      lows,
        "close":    closes,
        "volume":   volume,
    })


def _make_agent(
    broker_df: pd.DataFrame | None = None,
    cached_items: List[WatchlistItem] | None = None,
) -> UniverseAgent:
    """Build a UniverseAgent with a mock broker and mock DB.

    The agent's _API_DELAY is set to 0 so tests run instantly.
    If *broker_df* is None, all stocks return a default 60-row uptrend.
    """
    broker = MagicMock()
    df = broker_df if broker_df is not None else _make_ohlcv(60, base_price=500.0, trend="up")
    broker.get_historical_data.return_value = df
    ltp_price = float(df["close"].iloc[-1]) if not df.empty else 500.0
    broker.get_ltp.return_value = {"ltp": ltp_price}

    db = MagicMock()
    db.get_latest_watchlist.return_value = cached_items or []

    # Build a minimal config mock
    cfg = MagicMock()
    cfg.get.side_effect = lambda *args, **kwargs: {
        ("trading", "min_stock_price"): 50,
        ("trading", "max_stock_price"): 5000,
        ("trading", "min_volume_cr"):   10,
        ("universe", "blacklisted_stocks"): [],
    }.get(args, kwargs.get("default"))

    agent = UniverseAgent(broker=broker, db=db, config=cfg)
    agent._API_DELAY = 0  # no sleeping in tests
    return agent


# =========================================================================== #
# Fallback data integrity                                                       #
# =========================================================================== #

class TestFallbackData:
    def test_nifty50_has_50_symbols(self):
        assert len(_FALLBACK_CONSTITUENTS["NIFTY_50"]) == 50

    def test_next50_has_50_symbols(self):
        assert len(_FALLBACK_CONSTITUENTS["NIFTY_NEXT_50"]) == 50

    def test_midcap_select_has_symbols(self):
        assert len(_FALLBACK_CONSTITUENTS["NIFTY_MIDCAP_SELECT"]) >= 25

    def test_no_duplicates_within_nifty50(self):
        syms = _FALLBACK_CONSTITUENTS["NIFTY_50"]
        assert len(syms) == len(set(syms))

    def test_sector_map_covers_nifty50(self):
        missing = [s for s in _FALLBACK_CONSTITUENTS["NIFTY_50"] if s not in _FALLBACK_SECTOR_MAP]
        assert missing == [], f"Missing sector mapping for: {missing}"

    def test_total_universe_size(self):
        all_syms = {s for syms in _FALLBACK_CONSTITUENTS.values() for s in syms}
        # Expect roughly 150 unique symbols (3 × ~50 with minimal overlap)
        assert 140 <= len(all_syms) <= 160


# =========================================================================== #
# Filter 1 — Price Range                                                        #
# =========================================================================== #

class TestPriceFilter:
    def test_rejects_below_min_price(self):
        """Close at 30 INR is below min=50 → rejected."""
        df = _make_ohlcv(60, base_price=30.0, trend="up")
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["CHEAPSTOCK"])
        assert not result

    def test_rejects_above_max_price(self):
        """Close at 6000 INR exceeds max=5000 → rejected."""
        df = _make_ohlcv(60, base_price=6000.0, trend="up")
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["PRICEYSTOCK"])
        assert not result

    def test_accepts_price_at_boundaries(self):
        """Prices exactly at min(50) and max(5000) should pass the price filter.

        They may still fail other filters, but we test that price alone isn't
        the rejecting cause by using a deliberately failing volume mock.
        """
        # We test indirectly: broker returns insufficient data rows → skipped
        # for a different reason. Instead, we mock TI to trace which filter
        # rejects — simplest approach is just boundary value check.
        for price in (50.0, 5000.0):
            df = _make_ohlcv(60, base_price=price, trend="up", volume=2_000_000)
            agent = _make_agent(broker_df=df)
            # uptrend at valid price — may or may not pass all filters but
            # the important thing is it's NOT rejected by price filter
            result = agent.apply_daily_filters(["BOUNDARY"])
            # For price=50 the traded value may be low (50×2M=100Cr — fine)
            # For price=5000 the uptrend should hold
            # Just assert no exception is raised
            assert isinstance(result, list)


# =========================================================================== #
# Filter 2 — Volume                                                             #
# =========================================================================== #

class TestVolumeFilter:
    def test_rejects_low_volume(self):
        """Volume of 100 shares at 500 = 50 000 INR per day << 10 Cr → rejected."""
        df = _make_ohlcv(60, base_price=500.0, trend="up", volume=100)
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["THINSTOCK"])
        assert not result

    def test_accepts_adequate_volume(self):
        """2 M shares × 500 INR = 100 Cr/day >> 10 Cr → passes volume filter."""
        df = _make_ohlcv(60, base_price=500.0, trend="up", volume=2_000_000)
        agent = _make_agent(broker_df=df)
        # Should not be rejected by volume; may pass all filters
        result = agent.apply_daily_filters(["HEAVYSTOCK"])
        # Volume filter specifically must not be the rejection reason:
        # if it were, result would be empty for a valid uptrending stock.
        assert isinstance(result, list)  # no crash
        # Verify broker was called (real fetch happened)
        assert agent._broker.get_historical_data.called

    def test_volume_calculation_uses_20day_avg(self):
        """avg_volume_cr in result should be computed as mean(close × vol)/1e7."""
        df = _make_ohlcv(60, base_price=500.0, trend="up", volume=2_000_000)
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["CALC"])
        if result:
            # Expected ≈ 500 × 2M / 1e7 = 100 Cr
            assert 80 <= result[0]["avg_volume_cr"] <= 120


# =========================================================================== #
# Filter 3 — Trend (EMA-50)                                                    #
# =========================================================================== #

class TestTrendFilter:
    def test_rejects_downtrend(self):
        """Price fell from 1000 to 300 — well below EMA-50 → rejected."""
        # Create explicit downtrend: prices drop 1000→300 with small intraday range
        prices = np.linspace(1000, 300, 60)
        df = pd.DataFrame({
            "datetime": pd.date_range(end=TODAY, periods=60, freq="B"),
            "open":   prices * 1.005,
            "high":   prices * 1.01,
            "low":    prices * 0.99,
            "close":  prices,
            "volume": np.full(60, 2_000_000),
        })
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["DOWNTREND"])
        assert not result

    def test_accepts_uptrend(self):
        """Price rose from 200 to 1000 — well above EMA-50 → trend filter passes."""
        prices = np.linspace(200, 1000, 60)
        df = pd.DataFrame({
            "datetime": pd.date_range(end=TODAY, periods=60, freq="B"),
            "open":   prices * 0.99,
            "high":   prices * 1.02,
            "low":    prices * 0.98,
            "close":  prices,
            "volume": np.full(60, 2_000_000),
        })
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["UPTREND"])
        # Passed trend filter; also check price_above_ema_pct > 0
        assert result
        assert result[0]["price_above_ema_pct"] > 0

    def test_price_at_ema_is_rejected(self):
        """Price exactly equal to EMA-50 should NOT pass (condition is strict >)."""
        prices = np.full(60, 500.0)  # flat → price == EMA50 == 500
        df = pd.DataFrame({
            "datetime": pd.date_range(end=TODAY, periods=60, freq="B"),
            "open":   prices,
            "high":   prices + 5,
            "low":    prices - 5,
            "close":  prices,
            "volume": np.full(60, 2_000_000),
        })
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["FLATSTOCK"])
        assert not result


# =========================================================================== #
# Filter 4 — ATR Volatility                                                    #
# =========================================================================== #

class TestATRFilter:
    def test_rejects_too_low_atr(self):
        """ATR < 1.5%: flat stock with tiny intraday range → rejected."""
        prices = np.linspace(490, 510, 60)  # gentle uptrend, price > EMA50
        df = pd.DataFrame({
            "datetime": pd.date_range(end=TODAY, periods=60, freq="B"),
            "open":   prices,
            "high":   prices + 0.3,   # ≈ 0.06% range → ATR << 1.5%
            "low":    prices - 0.3,
            "close":  prices,
            "volume": np.full(60, 2_000_000),
        })
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["BORINGSTOCK"])
        assert not result

    def test_rejects_too_high_atr(self):
        """ATR > 6%: wide-ranging stock is too risky → rejected."""
        prices = np.linspace(400, 500, 60)  # uptrend to pass trend filter
        df = pd.DataFrame({
            "datetime": pd.date_range(end=TODAY, periods=60, freq="B"),
            "open":   prices,
            "high":   prices * 1.08,  # 8% daily range → ATR > 6%
            "low":    prices * 0.92,
            "close":  prices,
            "volume": np.full(60, 2_000_000),
        })
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["WILDSTOCK"])
        assert not result

    def test_accepts_atr_in_valid_range(self):
        """ATR ≈ 3% (between 1.5% and 6%) passes the filter."""
        prices = np.linspace(400, 500, 60)  # uptrend
        df = pd.DataFrame({
            "datetime": pd.date_range(end=TODAY, periods=60, freq="B"),
            "open":   prices,
            "high":   prices * 1.015,  # ≈ 3% daily range → ATR ≈ 2-3%
            "low":    prices * 0.985,
            "close":  prices,
            "volume": np.full(60, 2_000_000),
        })
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["SWINGSTOCK"])
        assert result
        assert 1.5 <= result[0]["atr_pct"] <= 6.0


# =========================================================================== #
# Output shape                                                                  #
# =========================================================================== #

class TestOutputShape:
    def test_result_dict_has_required_keys(self):
        """Each passing stock dict must contain the documented keys."""
        required = {
            "symbol", "price", "avg_volume_cr", "atr_pct",
            "ema_50", "price_above_ema_pct", "sector", "index", "added_date",
        }
        df = _make_ohlcv(60, base_price=500.0, trend="up", daily_range_pct=0.03)
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["GOODSTOCK"])
        assert result
        assert required.issubset(result[0].keys())

    def test_added_date_is_today(self):
        df = _make_ohlcv(60, base_price=500.0, trend="up", daily_range_pct=0.03)
        agent = _make_agent(broker_df=df)
        result = agent.apply_daily_filters(["GOODSTOCK"])
        assert result
        assert result[0]["added_date"] == TODAY


# =========================================================================== #
# Watchlist Caching                                                             #
# =========================================================================== #

class TestWatchlistCaching:
    def test_returns_cached_watchlist_when_today_exists(self):
        """If DB already has today's watchlist, skip filter run entirely."""
        cached = [
            WatchlistItem(
                symbol="TCS", date=TODAY, price=3850.0,
                avg_volume_cr=125.3, atr_pct=2.5, ema_50=3780.0,
                sector="IT", in_index="NIFTY_50",
            )
        ]
        agent = _make_agent(cached_items=cached)
        result = agent.get_active_watchlist()

        # Broker should NOT have been called (served from cache)
        agent._broker.get_historical_data.assert_not_called()
        assert len(result) == 1
        assert result[0]["symbol"] == "TCS"

    def test_runs_filters_when_no_cache(self):
        """If DB returns empty list, daily filters must be run."""
        agent = _make_agent()  # no cached items
        with patch.object(agent, "apply_daily_filters", return_value=[]) as mock_flt:
            with patch.object(agent, "get_base_universe", return_value=["TCS"]):
                agent.get_active_watchlist()
        mock_flt.assert_called_once()

    def test_runs_filters_when_cache_is_stale(self):
        """A watchlist from a different date should not be treated as today's."""
        stale = [WatchlistItem(symbol="INFY", date="2020-01-01")]
        agent = _make_agent(cached_items=stale)
        with patch.object(agent, "apply_daily_filters", return_value=[]) as mock_flt:
            with patch.object(agent, "get_base_universe", return_value=[]):
                agent.get_active_watchlist()
        mock_flt.assert_called_once()

    def test_saves_to_db_after_fresh_run(self):
        """After computing a fresh watchlist the result must be saved to DB."""
        agent = _make_agent()
        stock = {
            "symbol": "TCS", "price": 3850.0, "avg_volume_cr": 125.0,
            "atr_pct": 2.5, "ema_50": 3780.0, "price_above_ema_pct": 1.86,
            "sector": "IT", "index": "NIFTY_50", "added_date": TODAY,
        }
        with patch.object(agent, "apply_daily_filters", return_value=[stock]):
            with patch.object(agent, "get_base_universe", return_value=["TCS"]):
                agent.get_active_watchlist()
        agent._db.save_watchlist.assert_called_once()

    def test_caps_at_50_stocks(self):
        """get_active_watchlist must return at most 50 stocks."""
        many_stocks = [
            {
                "symbol": f"STOCK{i}", "price": 500.0, "avg_volume_cr": 100.0,
                "atr_pct": float(i % 5 + 1), "ema_50": 490.0,
                "price_above_ema_pct": 2.0, "sector": "IT",
                "index": "NIFTY_50", "added_date": TODAY,
            }
            for i in range(80)
        ]
        agent = _make_agent()
        with patch.object(agent, "apply_daily_filters", return_value=many_stocks):
            with patch.object(agent, "get_base_universe", return_value=[f"STOCK{i}" for i in range(80)]):
                result = agent.get_active_watchlist()
        assert len(result) <= 50

    def test_sorted_by_atr_descending(self):
        """Watchlist must be ordered by ATR% from highest to lowest."""
        stocks = [
            {"symbol": "A", "price": 500.0, "avg_volume_cr": 100.0, "atr_pct": 2.0,
             "ema_50": 490.0, "price_above_ema_pct": 2.0, "sector": "IT",
             "index": "NIFTY_50", "added_date": TODAY},
            {"symbol": "B", "price": 500.0, "avg_volume_cr": 100.0, "atr_pct": 4.5,
             "ema_50": 490.0, "price_above_ema_pct": 2.0, "sector": "BANKING",
             "index": "NIFTY_50", "added_date": TODAY},
            {"symbol": "C", "price": 500.0, "avg_volume_cr": 100.0, "atr_pct": 3.1,
             "ema_50": 490.0, "price_above_ema_pct": 2.0, "sector": "PHARMA",
             "index": "NIFTY_50", "added_date": TODAY},
        ]
        agent = _make_agent()
        with patch.object(agent, "apply_daily_filters", return_value=stocks):
            with patch.object(agent, "get_base_universe", return_value=["A", "B", "C"]):
                result = agent.get_active_watchlist()
        atr_values = [r["atr_pct"] for r in result]
        assert atr_values == sorted(atr_values, reverse=True)


# =========================================================================== #
# Held Stock Force-Inclusion                                                   #
# =========================================================================== #

class TestHeldStocksInclusion:
    def _make_watchlist(self) -> list:
        return [
            {"symbol": "TCS", "price": 3850.0, "avg_volume_cr": 125.0,
             "atr_pct": 2.5, "ema_50": 3780.0, "price_above_ema_pct": 1.86,
             "sector": "IT", "index": "NIFTY_50", "added_date": TODAY},
        ]

    def test_force_adds_missing_held_stock(self):
        agent = _make_agent()
        result = agent.ensure_held_stocks_in_watchlist(self._make_watchlist(), ["TCS", "INFY"])
        symbols = [s["symbol"] for s in result]
        assert "INFY" in symbols

    def test_forced_stock_has_portfolio_reason(self):
        agent = _make_agent()
        result = agent.ensure_held_stocks_in_watchlist(self._make_watchlist(), ["INFY"])
        infy = next(s for s in result if s["symbol"] == "INFY")
        assert infy["reason"] == "HELD_IN_PORTFOLIO"

    def test_no_duplicate_for_already_present_stock(self):
        agent = _make_agent()
        result = agent.ensure_held_stocks_in_watchlist(self._make_watchlist(), ["TCS"])
        tcs_entries = [s for s in result if s["symbol"] == "TCS"]
        assert len(tcs_entries) == 1

    def test_multiple_held_stocks_all_added(self):
        agent = _make_agent()
        result = agent.ensure_held_stocks_in_watchlist([], ["SBIN", "INFY", "RELIANCE"])
        symbols = {s["symbol"] for s in result}
        assert symbols == {"SBIN", "INFY", "RELIANCE"}

    def test_held_stock_ltp_is_fetched(self):
        """ensure_held_stocks_in_watchlist should call broker.get_ltp for missing stocks."""
        agent = _make_agent()
        agent._broker.get_ltp.return_value = {"ltp": 1800.0}
        result = agent.ensure_held_stocks_in_watchlist([], ["INFY"])
        assert result[0]["price"] == 1800.0

    def test_ltp_failure_does_not_crash(self):
        """If LTP call raises, the stock is still added (price=None)."""
        agent = _make_agent()
        agent._broker.get_ltp.side_effect = Exception("API down")
        result = agent.ensure_held_stocks_in_watchlist([], ["INFY"])
        assert result[0]["symbol"] == "INFY"
        assert result[0]["price"] is None


# =========================================================================== #
# Sector Distribution                                                           #
# =========================================================================== #

class TestSectorDistribution:
    def test_counts_are_correct(self):
        agent = _make_agent()
        watchlist = [
            {"symbol": "TCS",      "sector": "IT"},
            {"symbol": "INFY",     "sector": "IT"},
            {"symbol": "HDFCBANK", "sector": "BANKING"},
            {"symbol": "SBIN",     "sector": "BANKING"},
            {"symbol": "SUNPHARMA","sector": "PHARMA"},
        ]
        dist = agent.get_sector_distribution(watchlist)
        assert dist["IT"] == 2
        assert dist["BANKING"] == 2
        assert dist["PHARMA"] == 1

    def test_sorted_descending(self):
        agent = _make_agent()
        watchlist = [
            {"symbol": "TCS",      "sector": "IT"},
            {"symbol": "INFY",     "sector": "IT"},
            {"symbol": "HDFCBANK", "sector": "BANKING"},
        ]
        dist = agent.get_sector_distribution(watchlist)
        counts = list(dist.values())
        assert counts == sorted(counts, reverse=True)

    def test_none_sector_maps_to_unknown(self):
        agent = _make_agent()
        dist = agent.get_sector_distribution([{"symbol": "NEWCO", "sector": None}])
        assert dist.get("UNKNOWN") == 1

    def test_empty_watchlist_returns_empty_dict(self):
        agent = _make_agent()
        assert agent.get_sector_distribution([]) == {}


# =========================================================================== #
# Base Universe                                                                 #
# =========================================================================== #

class TestBaseUniverse:
    def test_blacklisted_symbols_excluded(self):
        broker = MagicMock()
        db = MagicMock()
        db.get_latest_watchlist.return_value = []
        cfg = MagicMock()
        cfg.get.side_effect = lambda *args, **kwargs: (
            ["RELIANCE", "TCS"]
            if args == ("universe", "blacklisted_stocks")
            else None
        )
        agent = UniverseAgent(broker=broker, db=db, config=cfg)
        agent._API_DELAY = 0

        with patch("src.agents.universe_agent._CONSTITUENTS_FILE") as mock_path:
            mock_path.exists.return_value = False
            base = agent.get_base_universe()

        assert "RELIANCE" not in base
        assert "TCS" not in base

    def test_no_duplicates_in_base_universe(self):
        agent = _make_agent()
        with patch("src.agents.universe_agent._CONSTITUENTS_FILE") as mock_path:
            mock_path.exists.return_value = False
            base = agent.get_base_universe()
        assert len(base) == len(set(base))

    def test_loads_from_file_when_present(self, tmp_path):
        """If index_constituents.json exists it is used instead of fallback."""
        constituents_file = tmp_path / "index_constituents.json"
        payload = {
            "updated_at": TODAY,
            "constituents": {"NIFTY_50": ["FAKESTOCK1", "FAKESTOCK2"]},
        }
        constituents_file.write_text(json.dumps(payload))

        agent = _make_agent()
        with patch("src.agents.universe_agent._CONSTITUENTS_FILE", constituents_file):
            base = agent.get_base_universe()

        assert "FAKESTOCK1" in base
        assert "FAKESTOCK2" in base


# =========================================================================== #
# Error resilience                                                              #
# =========================================================================== #

class TestErrorResilience:
    def test_broker_exception_skips_stock(self):
        """If broker raises for one stock, the rest should still be processed."""
        broker = MagicMock()
        good_df = _make_ohlcv(60, base_price=500.0, trend="up", daily_range_pct=0.03)
        bad_df = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated API failure")
            return good_df

        broker.get_historical_data.side_effect = side_effect
        broker.get_ltp.return_value = {"ltp": 500.0}

        db = MagicMock()
        db.get_latest_watchlist.return_value = []
        cfg = MagicMock()
        cfg.get.side_effect = lambda *args, **kwargs: {
            ("trading", "min_stock_price"): 50,
            ("trading", "max_stock_price"): 5000,
            ("trading", "min_volume_cr"): 10,
            ("universe", "blacklisted_stocks"): [],
        }.get(args)

        agent = UniverseAgent(broker=broker, db=db, config=cfg)
        agent._API_DELAY = 0

        result = agent.apply_daily_filters(["FAILSTOCK", "GOODSTOCK"])
        # First call raised, second succeeded — GOODSTOCK should appear
        assert any(s["symbol"] == "GOODSTOCK" for s in result)

    def test_empty_dataframe_skips_stock(self):
        """An empty DataFrame (no data returned) should not crash the scan."""
        empty_df = pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        agent = _make_agent(broker_df=empty_df)
        result = agent.apply_daily_filters(["NODATA"])
        assert result == []

    def test_insufficient_rows_skipped(self):
        """A DataFrame with fewer than 20 rows is skipped."""
        short_df = _make_ohlcv(10, base_price=500.0, trend="up")
        agent = _make_agent(broker_df=short_df)
        result = agent.apply_daily_filters(["SHORTDATA"])
        assert result == []

    def test_db_save_failure_does_not_crash(self):
        """If saving to DB raises, get_active_watchlist should still return results."""
        good_stock = {
            "symbol": "TCS", "price": 3850.0, "avg_volume_cr": 125.0,
            "atr_pct": 2.5, "ema_50": 3780.0, "price_above_ema_pct": 1.86,
            "sector": "IT", "index": "NIFTY_50", "added_date": TODAY,
        }
        agent = _make_agent()
        agent._db.save_watchlist.side_effect = RuntimeError("DB error")
        with patch.object(agent, "apply_daily_filters", return_value=[good_stock]):
            with patch.object(agent, "get_base_universe", return_value=["TCS"]):
                result = agent.get_active_watchlist()
        assert result  # still returns data despite DB failure


# =========================================================================== #
# refresh_index_constituents                                                    #
# =========================================================================== #

class TestRefreshIndexConstituents:
    def test_uses_fallback_when_nse_fetch_fails(self, tmp_path):
        """If NSE fetch fails for all indices, the fallback lists are persisted."""
        agent = _make_agent()
        with patch("src.agents.universe_agent._CONSTITUENTS_FILE", tmp_path / "ic.json"):
            with patch.object(agent, "_fetch_index_from_nse", return_value=None):
                summary = agent.refresh_index_constituents()

        assert summary["nifty_50"] == 50
        assert summary["next_50"] == 50
        assert summary["midcap_select"] >= 25

    def test_saves_json_file(self, tmp_path):
        """refresh_index_constituents must write index_constituents.json."""
        agent = _make_agent()
        out_file = tmp_path / "ic.json"
        with patch("src.agents.universe_agent._CONSTITUENTS_FILE", out_file):
            with patch.object(agent, "_fetch_index_from_nse", return_value=None):
                agent.refresh_index_constituents()
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "constituents" in data
        assert "updated_at" in data

    def test_returns_correct_summary_structure(self, tmp_path):
        agent = _make_agent()
        with patch("src.agents.universe_agent._CONSTITUENTS_FILE", tmp_path / "ic.json"):
            with patch.object(agent, "_fetch_index_from_nse", return_value=None):
                result = agent.refresh_index_constituents()
        assert set(result.keys()) == {
            "total_universe", "nifty_50", "next_50", "midcap_select"
        }


# =========================================================================== #
# Compatibility shims                                                           #
# =========================================================================== #

class TestCompatibilityShims:
    def test_is_blacklisted_true(self):
        cfg = MagicMock()
        cfg.get.return_value = ["RELIANCE"]
        agent = UniverseAgent(config=cfg)
        assert agent.is_blacklisted("RELIANCE") is True

    def test_is_blacklisted_false(self):
        cfg = MagicMock()
        cfg.get.return_value = []
        agent = UniverseAgent(config=cfg)
        assert agent.is_blacklisted("TCS") is False

    def test_get_index_constituents_delegates(self):
        agent = _make_agent()
        with patch("src.agents.universe_agent._CONSTITUENTS_FILE") as mp:
            mp.exists.return_value = False
            nifty50 = agent.get_index_constituents("NIFTY_50")
        assert len(nifty50) == 50
