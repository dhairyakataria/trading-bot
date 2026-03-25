"""Tests for src/tools — TechnicalIndicators, WebSearchTool, NewsFetcher."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.llm.budget_manager import BudgetManager
from src.tools.news_fetcher import NewsFetcher
from src.tools.technical_indicators import TechnicalIndicators
from src.tools.web_search import WebSearchTool


# ===========================================================================
# Fixtures / helpers
# ===========================================================================


def make_ohlcv(n: int = 100, start: float = 3500.0, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic OHLCV DataFrame with *n* rows."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2026-01-01", periods=n, freq="D")
    close = start + np.cumsum(rng.normal(0, 30, n))
    high = close + rng.uniform(5, 50, n)
    low = close - rng.uniform(5, 50, n)
    open_ = close + rng.normal(0, 15, n)
    volume = rng.integers(1_000_000, 10_000_000, n)
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume.astype(float),
        }
    )


def make_declining(n: int = 60) -> pd.DataFrame:
    """Strictly declining close prices → RSI should be very low."""
    close = np.linspace(4000, 2800, n)
    high = close + 10
    low = close - 10
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": close + 5,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 2_000_000.0),
        }
    )


def make_rising(n: int = 60) -> pd.DataFrame:
    """Strongly rising prices (with small noise) → RSI should be > 70.

    Pure all-gains series causes avg_loss = 0 which can make some TA
    libraries return NaN.  Small noise ensures a handful of down-bars
    exist so the calculation is numerically stable.
    """
    rng = np.random.default_rng(123)
    trend = np.linspace(2800, 4000, n)
    # noise std (8) much smaller than daily trend step (~20) → net uptrend
    close = trend + rng.normal(0, 8, n)
    close = np.maximum(close, 100)  # ensure positive prices
    high = close + 10
    low = close - 10
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": close - 5,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 2_000_000.0),
        }
    )


class MockConfig:
    """Minimal config stub that returns fake API keys."""

    def get(self, *keys, default=None):  # noqa: ANN001
        mapping = {
            ("apis", "tavily", "api_key"): "test_tavily_key",
            ("apis", "newsapi", "api_key"): "test_newsapi_key",
            ("apis", "serpapi", "api_key"): "test_serpapi_key",
        }
        return mapping.get(keys, default)


@pytest.fixture()
def ti():
    return TechnicalIndicators()


@pytest.fixture()
def sample_df():
    return make_ohlcv()


@pytest.fixture()
def budget():
    return BudgetManager(db=None)


@pytest.fixture()
def web_search(budget):
    return WebSearchTool(config=MockConfig(), budget_manager=budget)


@pytest.fixture()
def news_fetcher(budget):
    return NewsFetcher(config=MockConfig(), budget_manager=budget)


# ===========================================================================
# TechnicalIndicators
# ===========================================================================


class TestRSI:
    def test_returns_expected_keys(self, ti, sample_df):
        result = ti.calculate_rsi(sample_df)
        assert "value" in result
        assert "signal" in result
        assert "interpretation" in result

    def test_value_in_range(self, ti, sample_df):
        result = ti.calculate_rsi(sample_df)
        assert 0 <= result["value"] <= 100

    def test_oversold_on_declining_prices(self, ti):
        result = ti.calculate_rsi(make_declining())
        assert result["signal"] == "OVERSOLD"
        assert result["value"] < 30

    def test_overbought_on_rising_prices(self, ti):
        result = ti.calculate_rsi(make_rising())
        assert result["signal"] == "OVERBOUGHT"
        assert result["value"] > 70

    def test_insufficient_data_returns_error(self, ti):
        result = ti.calculate_rsi(make_ohlcv(5))
        assert "error" in result


class TestMACD:
    def test_returns_expected_keys(self, ti, sample_df):
        result = ti.calculate_macd(sample_df)
        for key in ("macd_line", "signal_line", "histogram", "signal", "interpretation"):
            assert key in result

    def test_signal_is_valid(self, ti, sample_df):
        result = ti.calculate_macd(sample_df)
        assert result["signal"] in (
            "BULLISH_CROSSOVER",
            "BEARISH_CROSSOVER",
            "BULLISH",
            "BEARISH",
        )

    def test_insufficient_data_returns_error(self, ti):
        result = ti.calculate_macd(make_ohlcv(20))
        assert "error" in result

    def test_histogram_equals_macd_minus_signal(self, ti, sample_df):
        result = ti.calculate_macd(sample_df)
        assert abs(result["histogram"] - (result["macd_line"] - result["signal_line"])) < 0.01


class TestBollingerBands:
    def test_returns_expected_keys(self, ti, sample_df):
        result = ti.calculate_bollinger_bands(sample_df)
        for key in ("upper", "middle", "lower", "current_price", "signal", "interpretation"):
            assert key in result

    def test_band_ordering(self, ti, sample_df):
        result = ti.calculate_bollinger_bands(sample_df)
        assert result["upper"] > result["middle"] > result["lower"]

    def test_signal_is_valid(self, ti, sample_df):
        result = ti.calculate_bollinger_bands(sample_df)
        assert result["signal"] in (
            "NEAR_LOWER_BAND",
            "NEAR_UPPER_BAND",
            "ABOVE_MIDDLE",
            "BELOW_MIDDLE",
        )

    def test_insufficient_data_returns_error(self, ti):
        result = ti.calculate_bollinger_bands(make_ohlcv(10))
        assert "error" in result


class TestEMA:
    def test_returns_expected_keys(self, ti, sample_df):
        result = ti.calculate_ema(sample_df)
        for key in ("ema_20", "ema_50", "ema_200", "signal", "interpretation"):
            assert key in result

    def test_price_vs_ema_values(self, ti, sample_df):
        result = ti.calculate_ema(sample_df)
        for p in (20, 50, 200):
            assert result.get(f"price_vs_ema_{p}") in (
                "ABOVE",
                "BELOW",
                "INSUFFICIENT_DATA",
            )

    def test_strong_uptrend_on_rising_prices(self, ti):
        result = ti.calculate_ema(make_rising(250))
        assert result["signal"] in ("STRONG_UPTREND", "UPTREND")

    def test_custom_periods(self, ti, sample_df):
        result = ti.calculate_ema(sample_df, periods=[10, 30])
        assert "ema_10" in result
        assert "ema_30" in result


class TestVWAP:
    def test_returns_expected_keys(self, ti, sample_df):
        result = ti.calculate_vwap(sample_df)
        assert "vwap" in result
        assert "current_price" in result
        assert "signal" in result

    def test_signal_is_valid(self, ti, sample_df):
        result = ti.calculate_vwap(sample_df)
        assert result["signal"] in ("ABOVE_VWAP", "BELOW_VWAP")

    def test_zero_volume_returns_error(self, ti):
        df = make_ohlcv(20)
        df["volume"] = 0.0
        result = ti.calculate_vwap(df)
        assert "error" in result


class TestATR:
    def test_returns_expected_keys(self, ti, sample_df):
        result = ti.calculate_atr(sample_df)
        for key in ("atr", "atr_pct", "interpretation"):
            assert key in result

    def test_atr_is_positive(self, ti, sample_df):
        result = ti.calculate_atr(sample_df)
        assert result["atr"] > 0
        assert result["atr_pct"] > 0

    def test_insufficient_data_returns_error(self, ti):
        result = ti.calculate_atr(make_ohlcv(5))
        assert "error" in result


class TestVolumeAnalysis:
    def test_returns_expected_keys(self, ti, sample_df):
        result = ti.calculate_volume_analysis(sample_df)
        for key in ("current_volume", "avg_volume", "volume_ratio", "signal", "interpretation"):
            assert key in result

    def test_high_volume_signal(self, ti):
        df = make_ohlcv(30)
        df.loc[df.index[-1], "volume"] = float(df["volume"].mean() * 3)
        result = ti.calculate_volume_analysis(df)
        assert result["signal"] == "HIGH_VOLUME"

    def test_low_volume_signal(self, ti):
        df = make_ohlcv(30)
        df.loc[df.index[-1], "volume"] = float(df["volume"].mean() * 0.1)
        result = ti.calculate_volume_analysis(df)
        assert result["signal"] == "LOW_VOLUME"


class TestSupportResistance:
    def test_returns_expected_keys(self, ti, sample_df):
        result = ti.calculate_support_resistance(sample_df)
        for key in ("support_1", "support_2", "resistance_1", "resistance_2", "current_price"):
            assert key in result

    def test_resistance_above_support(self, ti, sample_df):
        result = ti.calculate_support_resistance(sample_df)
        assert result["resistance_1"] > result["support_1"]

    def test_insufficient_data_returns_error(self, ti):
        result = ti.calculate_support_resistance(make_ohlcv(5))
        assert "error" in result


class TestGenerateFullAnalysis:
    def test_returns_expected_top_level_keys(self, ti, sample_df):
        result = ti.generate_full_analysis(sample_df, "TEST")
        for key in (
            "symbol",
            "timestamp",
            "price",
            "indicators",
            "overall_signal",
            "signal_strength",
            "score",
            "reasons",
        ):
            assert key in result

    def test_symbol_is_preserved(self, ti, sample_df):
        result = ti.generate_full_analysis(sample_df, "RELIANCE")
        assert result["symbol"] == "RELIANCE"

    def test_overall_signal_valid(self, ti, sample_df):
        result = ti.generate_full_analysis(sample_df, "TEST")
        assert result["overall_signal"] in ("BUY", "SELL", "HOLD")

    def test_signal_strength_in_range(self, ti, sample_df):
        result = ti.generate_full_analysis(sample_df, "TEST")
        assert 0.0 <= result["signal_strength"] <= 1.0

    def test_indicators_dict_has_all_keys(self, ti, sample_df):
        result = ti.generate_full_analysis(sample_df, "TEST")
        ind = result["indicators"]
        for key in ("rsi", "macd", "bollinger", "ema", "vwap", "atr", "volume", "support_resistance"):
            assert key in ind

    def test_buy_signal_scoring(self, ti, sample_df):
        """Patch all indicators to all-bullish values → score > 3 → BUY."""
        bullish_rsi = {"value": 25.0, "signal": "OVERSOLD", "interpretation": ""}
        bullish_macd = {
            "macd_line": 5.0,
            "signal_line": 3.0,
            "histogram": 2.0,
            "signal": "BULLISH_CROSSOVER",
            "interpretation": "",
        }
        bullish_bb = {
            "upper": 4100,
            "middle": 3950,
            "lower": 3800,
            "current_price": 3820,
            "signal": "NEAR_LOWER_BAND",
            "interpretation": "",
        }
        bullish_ema = {
            "ema_20": 3700,
            "ema_50": 3650,
            "ema_200": 3500,
            "price_vs_ema_20": "ABOVE",
            "price_vs_ema_50": "ABOVE",
            "price_vs_ema_200": "ABOVE",
            "signal": "STRONG_UPTREND",
            "interpretation": "",
        }
        bullish_vol = {
            "current_volume": 5_000_000,
            "avg_volume": 2_000_000,
            "volume_ratio": 2.5,
            "signal": "HIGH_VOLUME",
            "interpretation": "",
        }
        neutral_sr = {
            "support_1": 3750,
            "support_2": 3680,
            "resistance_1": 4100,
            "resistance_2": 4200,
            "current_price": 3850,
        }

        with (
            patch.object(ti, "calculate_rsi", return_value=bullish_rsi),
            patch.object(ti, "calculate_macd", return_value=bullish_macd),
            patch.object(ti, "calculate_bollinger_bands", return_value=bullish_bb),
            patch.object(ti, "calculate_ema", return_value=bullish_ema),
            patch.object(ti, "calculate_vwap", return_value={"vwap": 3800, "current_price": 3850, "signal": "ABOVE_VWAP", "interpretation": ""}),
            patch.object(ti, "calculate_atr", return_value={"atr": 85, "atr_pct": 2.2, "interpretation": ""}),
            patch.object(ti, "calculate_volume_analysis", return_value=bullish_vol),
            patch.object(ti, "calculate_support_resistance", return_value=neutral_sr),
        ):
            result = ti.generate_full_analysis(sample_df, "TEST")

        assert result["overall_signal"] == "BUY"
        assert result["score"] > 3

    def test_sell_signal_scoring(self, ti, sample_df):
        """Patch all indicators to all-bearish values → score < -3 → SELL."""
        bearish_rsi = {"value": 78.0, "signal": "OVERBOUGHT", "interpretation": ""}
        bearish_macd = {
            "macd_line": -5.0,
            "signal_line": -3.0,
            "histogram": -2.0,
            "signal": "BEARISH_CROSSOVER",
            "interpretation": "",
        }
        neutral_bb = {
            "upper": 4100,
            "middle": 3950,
            "lower": 3800,
            "current_price": 4080,
            "signal": "NEAR_UPPER_BAND",
            "interpretation": "",
        }
        bearish_ema = {
            "ema_20": 4100,
            "ema_50": 4050,
            "ema_200": 4000,
            "price_vs_ema_20": "BELOW",
            "price_vs_ema_50": "BELOW",
            "price_vs_ema_200": "BELOW",
            "signal": "STRONG_DOWNTREND",
            "interpretation": "",
        }
        bearish_vol = {
            "current_volume": 5_000_000,
            "avg_volume": 2_000_000,
            "volume_ratio": 2.5,
            "signal": "HIGH_VOLUME",
            "interpretation": "",
        }
        bearish_sr = {
            "support_1": 3200,
            "support_2": 3000,
            "resistance_1": 3860,
            "resistance_2": 4000,
            "current_price": 3850,
        }

        with (
            patch.object(ti, "calculate_rsi", return_value=bearish_rsi),
            patch.object(ti, "calculate_macd", return_value=bearish_macd),
            patch.object(ti, "calculate_bollinger_bands", return_value=neutral_bb),
            patch.object(ti, "calculate_ema", return_value=bearish_ema),
            patch.object(ti, "calculate_vwap", return_value={"vwap": 3900, "current_price": 3850, "signal": "BELOW_VWAP", "interpretation": ""}),
            patch.object(ti, "calculate_atr", return_value={"atr": 85, "atr_pct": 2.2, "interpretation": ""}),
            patch.object(ti, "calculate_volume_analysis", return_value=bearish_vol),
            patch.object(ti, "calculate_support_resistance", return_value=bearish_sr),
        ):
            result = ti.generate_full_analysis(sample_df, "TEST")

        assert result["overall_signal"] == "SELL"
        assert result["score"] < -3

    def test_hold_signal_neutral_indicators(self, ti, sample_df):
        """Neutral indicators → HOLD."""
        neutral_rsi = {"value": 50.0, "signal": "NEUTRAL", "interpretation": ""}
        neutral_macd = {
            "macd_line": 1.0,
            "signal_line": 0.9,
            "histogram": 0.1,
            "signal": "BULLISH",
            "interpretation": "",
        }
        neutral_bb = {"upper": 4100, "middle": 3950, "lower": 3800, "current_price": 3900, "signal": "ABOVE_MIDDLE", "interpretation": ""}
        neutral_ema = {
            "ema_20": 3920,
            "ema_50": 3880,
            "ema_200": 3800,
            "price_vs_ema_20": "BELOW",
            "price_vs_ema_50": "ABOVE",
            "price_vs_ema_200": "ABOVE",
            "signal": "MIXED",
            "interpretation": "",
        }
        neutral_vol = {"current_volume": 2_000_000, "avg_volume": 2_000_000, "volume_ratio": 1.0, "signal": "NORMAL_VOLUME", "interpretation": ""}
        neutral_sr = {"support_1": 3600, "support_2": 3500, "resistance_1": 4200, "resistance_2": 4400, "current_price": 3900}

        with (
            patch.object(ti, "calculate_rsi", return_value=neutral_rsi),
            patch.object(ti, "calculate_macd", return_value=neutral_macd),
            patch.object(ti, "calculate_bollinger_bands", return_value=neutral_bb),
            patch.object(ti, "calculate_ema", return_value=neutral_ema),
            patch.object(ti, "calculate_vwap", return_value={"vwap": 3900, "current_price": 3900, "signal": "ABOVE_VWAP", "interpretation": ""}),
            patch.object(ti, "calculate_atr", return_value={"atr": 85, "atr_pct": 2.2, "interpretation": ""}),
            patch.object(ti, "calculate_volume_analysis", return_value=neutral_vol),
            patch.object(ti, "calculate_support_resistance", return_value=neutral_sr),
        ):
            result = ti.generate_full_analysis(sample_df, "TEST")

        assert result["overall_signal"] == "HOLD"


# ===========================================================================
# WebSearchTool
# ===========================================================================


def _mock_tavily(search_response: dict) -> dict:
    """Return a sys.modules patch dict for tavily."""
    import sys  # noqa: PLC0415

    mock = MagicMock()
    mock.TavilyClient.return_value.search.return_value = search_response
    return {"tavily": mock}


def _mock_ddg(text_results: list) -> dict:
    """Return a sys.modules patch dict for both ddgs and duckduckgo_search."""
    mock = MagicMock()
    mock.DDGS.return_value.__enter__.return_value.text.return_value = text_results
    return {"ddgs": mock, "duckduckgo_search": mock}


def _mock_serpapi(organic_results: list) -> dict:
    """Return a sys.modules patch dict for serpapi."""
    import sys  # noqa: PLC0415

    mock = MagicMock()
    mock.GoogleSearch.return_value.get_dict.return_value = {
        "organic_results": organic_results
    }
    return {"serpapi": mock}


class TestWebSearchTavily:
    def test_returns_list_of_dicts(self, web_search):
        import sys  # noqa: PLC0415

        mock_response = {
            "results": [
                {
                    "title": "TCS Q4 Results",
                    "url": "https://example.com/tcs",
                    "content": "TCS reported strong earnings",
                    "published_date": "2026-03-17",
                }
            ]
        }
        with patch.dict(sys.modules, _mock_tavily(mock_response)):
            results = web_search.search_tavily("TCS NSE India")

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["title"] == "TCS Q4 Results"
        assert results[0]["url"] == "https://example.com/tcs"

    def test_budget_exhausted_returns_empty(self, web_search):
        web_search.budget._counters["tavily_search"] = 9999
        results = web_search.search_tavily("any query")
        assert results == []

    def test_missing_api_key_returns_empty(self, web_search):
        config = MockConfig()
        config.get = lambda *keys, default=None: (
            "${TAVILY_API_KEY}" if keys == ("apis", "tavily", "api_key") else default
        )
        tool = WebSearchTool(config=config, budget_manager=web_search.budget)
        results = tool.search_tavily("any query")
        assert results == []

    def test_tracks_budget_usage(self, web_search):
        import sys  # noqa: PLC0415

        initial = web_search.budget.get_remaining("tavily_search")
        mock_response = {
            "results": [
                {"title": "x", "url": "y", "content": "z", "published_date": ""}
            ]
        }
        with patch.dict(sys.modules, _mock_tavily(mock_response)):
            web_search.search_tavily("query")
        assert web_search.budget.get_remaining("tavily_search") == initial - 1


class TestWebSearchDuckDuckGo:
    def test_returns_expected_keys(self, web_search):
        import sys  # noqa: PLC0415

        mock_results = [
            {
                "title": "Nifty 50 Update",
                "href": "https://example.com",
                "body": "Market rose 1%",
            }
        ]
        with patch.dict(sys.modules, _mock_ddg(mock_results)):
            results = web_search.search_duckduckgo("Nifty 50")

        assert len(results) == 1
        assert results[0]["title"] == "Nifty 50 Update"
        assert results[0]["url"] == "https://example.com"
        assert "published_date" in results[0]

    def test_returns_empty_on_exception(self, web_search):
        import sys  # noqa: PLC0415

        mock_ddg_module = MagicMock()
        mock_ddg_module.DDGS.side_effect = Exception("network error")
        with patch.dict(sys.modules, {"ddgs": mock_ddg_module, "duckduckgo_search": mock_ddg_module}):
            results = web_search.search_duckduckgo("query")
        assert results == []


class TestWebSearchSerp:
    def test_returns_expected_keys(self, web_search):
        import sys  # noqa: PLC0415

        organic = [
            {
                "title": "BSE Sensex",
                "link": "https://bse.com",
                "snippet": "Markets up",
                "date": "2026-03-17",
            }
        ]
        with patch.dict(sys.modules, _mock_serpapi(organic)):
            results = web_search.search_serp("BSE Sensex")

        assert len(results) == 1
        assert results[0]["title"] == "BSE Sensex"

    def test_budget_exhausted_returns_empty(self, web_search):
        web_search.budget._counters["serp_api"] = 9999
        results = web_search.search_serp("query")
        assert results == []


class TestWebSearchMain:
    def test_falls_back_to_ddg_when_tavily_empty(self, web_search):
        """Tavily returns empty → falls through to DuckDuckGo."""
        ddg_results = [{"title": "DDG result", "url": "u", "snippet": "s", "published_date": ""}]
        with (
            patch.object(web_search, "search_tavily", return_value=[]),
            patch.object(web_search, "search_duckduckgo", return_value=ddg_results),
        ):
            results = web_search.search("any query")

        assert results == ddg_results

    def test_falls_back_to_serp_when_all_others_empty(self, web_search):
        serp_results = [{"title": "SERP result", "url": "u", "snippet": "s", "published_date": ""}]
        with (
            patch.object(web_search, "search_tavily", return_value=[]),
            patch.object(web_search, "search_duckduckgo", return_value=[]),
            patch.object(web_search, "search_serp", return_value=serp_results),
        ):
            results = web_search.search("any query")

        assert results == serp_results

    def test_stock_news_query_includes_nse(self, web_search):
        with patch.object(web_search, "search", return_value=[]) as mock_search:
            web_search.search_stock_news("TCS")
        called_query = mock_search.call_args[0][0]
        assert "TCS" in called_query
        assert "NSE" in called_query

    def test_sector_news_query_includes_sector(self, web_search):
        with patch.object(web_search, "search", return_value=[]) as mock_search:
            web_search.search_sector_news("IT")
        called_query = mock_search.call_args[0][0]
        assert "IT" in called_query


def _mock_newspaper(article_mock: MagicMock) -> dict:
    """Return a sys.modules patch dict for the newspaper package.

    newspaper3k's lxml.html.clean dependency was split into a separate package
    (lxml-html-clean).  Patching sys.modules avoids importing newspaper at all,
    so tests run regardless of whether lxml-html-clean is installed.
    """
    import sys  # noqa: PLC0415

    mock_module = MagicMock()
    mock_module.Article.return_value = article_mock
    return {"newspaper": mock_module}


class TestReadArticle:
    def test_returns_expected_keys(self, web_search):
        import sys  # noqa: PLC0415

        mock_article = MagicMock()
        mock_article.title = "Test Title"
        mock_article.text = "Article body " * 300  # > 3 000 chars
        mock_article.authors = ["Author One"]
        mock_article.publish_date = None

        with patch.dict(sys.modules, _mock_newspaper(mock_article)):
            result = web_search.read_article("https://example.com/article")

        assert result["title"] == "Test Title"
        assert len(result["text"]) <= 3000
        assert result["authors"] == ["Author One"]

    def test_text_truncated_to_3000(self, web_search):
        import sys  # noqa: PLC0415

        mock_article = MagicMock()
        mock_article.title = "T"
        mock_article.text = "x" * 5000
        mock_article.authors = []
        mock_article.publish_date = None

        with patch.dict(sys.modules, _mock_newspaper(mock_article)):
            result = web_search.read_article("https://example.com")

        assert len(result["text"]) == 3000

    def test_error_handled_gracefully(self, web_search):
        import sys  # noqa: PLC0415

        mock_module = MagicMock()
        mock_module.Article.side_effect = Exception("timeout")

        with patch.dict(sys.modules, {"newspaper": mock_module}):
            result = web_search.read_article("https://example.com")

        assert "error" in result
        assert result["title"] == ""

    def test_budget_exhausted_returns_error_dict(self, web_search):
        web_search.budget._counters["web_scrape"] = 9999
        result = web_search.read_article("https://example.com")
        assert "error" in result


# ===========================================================================
# NewsFetcher
# ===========================================================================


def _make_rss_entry(title: str, link: str, summary: str = "", published: str = "") -> MagicMock:
    entry = MagicMock()
    entry.title = title
    entry.link = link
    entry.summary = summary
    entry.published = published
    entry.source = {}
    return entry


def _make_feed(entries: list) -> MagicMock:
    feed = MagicMock()
    feed.entries = entries
    return feed


def _mock_feedparser(feed: MagicMock) -> dict:
    """Return a sys.modules patch dict for feedparser."""
    mock = MagicMock()
    mock.parse.return_value = feed
    return {"feedparser": mock}


class TestFetchGoogleNewsRSS:
    def test_returns_list_of_dicts(self, news_fetcher):
        import sys  # noqa: PLC0415

        entries = [
            _make_rss_entry(
                "Market rises", "https://example.com/1", "Nifty up 1%", "Mon, 17 Mar 2026"
            ),
            _make_rss_entry("RBI holds rates", "https://example.com/2"),
        ]
        with patch.dict(sys.modules, _mock_feedparser(_make_feed(entries))):
            results = news_fetcher.fetch_google_news_rss("Indian stock market")

        assert len(results) == 2
        assert results[0]["title"] == "Market rises"
        assert results[0]["url"] == "https://example.com/1"

    def test_max_results_respected(self, news_fetcher):
        import sys  # noqa: PLC0415

        entries = [
            _make_rss_entry(f"Article {i}", f"https://example.com/{i}")
            for i in range(20)
        ]
        with patch.dict(sys.modules, _mock_feedparser(_make_feed(entries))):
            results = news_fetcher.fetch_google_news_rss("query", max_results=5)
        assert len(results) == 5

    def test_error_handled_gracefully(self, news_fetcher):
        import sys  # noqa: PLC0415

        mock_fp = MagicMock()
        mock_fp.parse.side_effect = Exception("network error")
        with patch.dict(sys.modules, {"feedparser": mock_fp}):
            results = news_fetcher.fetch_google_news_rss("query")
        assert results == []


class TestFetchMoneyControlRSS:
    def test_returns_moneycontrol_source(self, news_fetcher):
        import sys  # noqa: PLC0415

        entries = [_make_rss_entry("Sensex gains", "https://moneycontrol.com/1")]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<fake rss>"
        with patch("requests.get", return_value=mock_resp):
            with patch.dict(sys.modules, _mock_feedparser(_make_feed(entries))):
                results = news_fetcher.fetch_moneycontrol_rss("market")

        assert len(results) == 1
        assert results[0]["source"] == "MoneyControl"

    def test_unknown_category_defaults_to_markets(self, news_fetcher):
        import sys  # noqa: PLC0415

        entries = [_make_rss_entry("Story", "https://mc.com/1")]
        mock_fp = MagicMock()
        mock_fp.parse.return_value = _make_feed(entries)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<fake rss>"
        with patch("requests.get", return_value=mock_resp) as mock_get:
            with patch.dict(sys.modules, {"feedparser": mock_fp}):
                news_fetcher.fetch_moneycontrol_rss("unknown_category")
        called_url = mock_get.call_args[0][0]
        assert "markets" in called_url


class TestFetchNewsAPI:
    def test_returns_articles(self, news_fetcher):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "Infosys Q4 Preview",
                    "url": "https://example.com/infosys",
                    "source": {"name": "Economic Times"},
                    "publishedAt": "2026-03-17T10:00:00Z",
                    "description": "Infosys set to report strong numbers",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(news_fetcher._session, "get", return_value=mock_response):
            results = news_fetcher.fetch_newsapi("Infosys NSE")

        assert len(results) == 1
        assert results[0]["title"] == "Infosys Q4 Preview"
        assert results[0]["source"] == "Economic Times"

    def test_budget_exhausted_returns_empty(self, news_fetcher):
        news_fetcher.budget._counters["news_api"] = 9999
        results = news_fetcher.fetch_newsapi("query")
        assert results == []

    def test_tracks_budget_usage(self, news_fetcher):
        initial = news_fetcher.budget.get_remaining("news_api")
        mock_response = MagicMock()
        mock_response.json.return_value = {"articles": []}
        mock_response.raise_for_status = MagicMock()

        with patch.object(news_fetcher._session, "get", return_value=mock_response):
            news_fetcher.fetch_newsapi("query")

        assert news_fetcher.budget.get_remaining("news_api") == initial - 1

    def test_missing_api_key_returns_empty(self, news_fetcher):
        config = MockConfig()
        config.get = lambda *keys, default=None: (
            "${NEWSAPI_API_KEY}" if keys == ("apis", "newsapi", "api_key") else default
        )
        fetcher = NewsFetcher(config=config, budget_manager=news_fetcher.budget)
        results = fetcher.fetch_newsapi("query")
        assert results == []


class TestDeduplication:
    def test_removes_duplicates(self, news_fetcher):
        articles = [
            {"title": "RBI holds rates steady today", "url": "u1", "source": "", "published_date": "", "snippet": ""},
            {"title": "RBI holds rates steady today", "url": "u2", "source": "", "published_date": "", "snippet": ""},
            {"title": "Different article here", "url": "u3", "source": "", "published_date": "", "snippet": ""},
        ]
        result = news_fetcher._deduplicate(articles)
        assert len(result) == 2

    def test_empty_titles_filtered(self, news_fetcher):
        articles = [
            {"title": "", "url": "u1", "source": "", "published_date": "", "snippet": ""},
            {"title": "Valid article", "url": "u2", "source": "", "published_date": "", "snippet": ""},
        ]
        result = news_fetcher._deduplicate(articles)
        assert len(result) == 1


class TestFetchMarketNews:
    def test_aggregates_multiple_sources(self, news_fetcher):
        rss_articles = [{"title": f"RSS {i}", "url": f"u{i}", "source": "G", "published_date": "", "snippet": ""} for i in range(5)]
        mc_articles = [{"title": f"MC {i}", "url": f"v{i}", "source": "MC", "published_date": "", "snippet": ""} for i in range(3)]

        with (
            patch.object(news_fetcher, "fetch_google_news_rss", return_value=rss_articles),
            patch.object(news_fetcher, "fetch_moneycontrol_rss", return_value=mc_articles),
            patch.object(news_fetcher, "fetch_newsapi", return_value=[]),
        ):
            results = news_fetcher.fetch_market_news(max_results=15)

        assert len(results) == 8  # 5 RSS + 3 MC, all unique


class TestFIIDIIData:
    def test_returns_expected_keys(self, news_fetcher):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "date": "17-Mar-2026",
                "fiiBUY": "5000.00",
                "fiiSELL": "6000.00",
                "fiiNET": "-1000.00",
                "diiBUY": "4000.00",
                "diiSELL": "3000.00",
                "diiNET": "1000.00",
            }
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("requests.Session") as MockSession:
            session_instance = MockSession.return_value
            session_instance.get.return_value = mock_response
            result = news_fetcher.get_fii_dii_data()

        assert "fii_buy" in result
        assert "fii_sell" in result
        assert "fii_net" in result
        assert "dii_buy" in result
        assert "source" in result

    def test_returns_safe_fallback_on_error(self, news_fetcher):
        with patch("requests.Session", side_effect=Exception("connection error")):
            result = news_fetcher.get_fii_dii_data()

        assert result["source"] == "unavailable"
        assert "error" in result
        assert result["fii_buy"] == 0.0


class TestGlobalMarketStatus:
    def test_returns_expected_keys(self, news_fetcher):
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 5500.0
        mock_fast_info.previous_close = 5450.0

        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = news_fetcher.get_global_market_status()

        assert "sp500" in result
        assert "nasdaq" in result
        assert "crude_oil" in result
        assert "gold" in result
        assert "usd_inr" in result
        assert "timestamp" in result

    def test_individual_ticker_failure_does_not_crash(self, news_fetcher):
        with patch("yfinance.Ticker", side_effect=Exception("yfinance error")):
            result = news_fetcher.get_global_market_status()
        # Should return error dict, not raise
        assert "error" in result or "timestamp" in result

    def test_change_pct_calculated(self, news_fetcher):
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 5500.0
        mock_fast_info.previous_close = 5000.0

        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = news_fetcher.get_global_market_status()

        assert result["sp500"]["change_pct"] == pytest.approx(10.0, rel=0.01)


class TestIndiaVIX:
    def test_high_fear_above_20(self, news_fetcher):
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 25.5

        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = news_fetcher.get_india_vix()

        assert result["vix"] == pytest.approx(25.5)
        assert result["signal"] == "HIGH_FEAR"

    def test_low_fear_below_15(self, news_fetcher):
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 12.3

        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = news_fetcher.get_india_vix()

        assert result["signal"] == "LOW_FEAR"

    def test_moderate_between_15_and_20(self, news_fetcher):
        mock_fast_info = MagicMock()
        mock_fast_info.last_price = 17.8

        mock_ticker = MagicMock()
        mock_ticker.fast_info = mock_fast_info

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = news_fetcher.get_india_vix()

        assert result["signal"] == "MODERATE"

    def test_error_handled_gracefully(self, news_fetcher):
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = news_fetcher.get_india_vix()
        assert "error" in result
        assert result["vix"] == 0.0
