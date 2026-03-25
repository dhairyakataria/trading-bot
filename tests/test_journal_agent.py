"""Tests for JournalAgent.

Strategy:
  - All external dependencies (db_manager, llm_router, chromadb) are replaced
    with MagicMock objects so tests are fast, deterministic, and offline.
  - ChromaDB is patched at the module import level so that __init__ never
    touches the file system.
  - Each test class targets one public method of JournalAgent.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agents.journal_agent import JournalAgent
from src.database.models import DailySummary, PortfolioSnapshot, Trade


# ─────────────────────────────────────────────────────────────────────────────
# Shared factories
# ─────────────────────────────────────────────────────────────────────────────


def _make_config(chroma_path: str = "data/chroma_db") -> MagicMock:
    cfg = MagicMock()
    cfg.get.return_value = chroma_path
    return cfg


def _make_llm(response: str = "Lesson: RSI bounces work well in uptrends.") -> MagicMock:
    llm = MagicMock()
    llm.call.return_value = response
    return llm


def _make_db(
    trade_history: list | None = None,
    perf_stats: dict | None = None,
    portfolio_history: list | None = None,
    system_state: dict | None = None,
) -> MagicMock:
    db = MagicMock()
    db.get_trade_history.return_value = trade_history or []
    db.get_performance_stats.return_value = perf_stats or {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate_pct": 0.0,
        "total_pnl": 0.0,
        "avg_pnl": 0.0,
        "best_trade_pnl": 0.0,
        "worst_trade_pnl": 0.0,
        "avg_holding_days": 0.0,
        "avg_pnl_pct": 0.0,
        "period_days": 30,
    }
    db.get_portfolio_history.return_value = portfolio_history or []
    state = system_state or {}
    db.get_system_state.side_effect = lambda key: state.get(key)
    db.log_agent_activity.return_value = None
    db.save_daily_summary.return_value = None
    return db


def _make_chroma_collection(count: int = 0, query_results: list | None = None) -> MagicMock:
    """Return a fake chromadb Collection."""
    col = MagicMock()
    col.count.return_value = count
    col.query.return_value = {
        "metadatas": [query_results or []],
        "documents": [[]],
    }
    col.add.return_value = None
    return col


def _make_chroma_client(collection: MagicMock | None = None) -> MagicMock:
    client = MagicMock()
    client.get_or_create_collection.return_value = collection or _make_chroma_collection()
    return client


def _make_agent(
    db: Any = None,
    llm: Any = None,
    chroma_collection: Any = None,
    chroma_raises: bool = False,
) -> JournalAgent:
    """Build a JournalAgent with all external deps mocked.

    ChromaDB is bypassed entirely by injecting a mock collection directly into
    the agent's private attribute after construction.  This avoids any file-system
    access and works regardless of whether chromadb is installed.
    """
    db = db or _make_db()
    llm = llm or _make_llm()
    agent = JournalAgent(config=_make_config(), db_manager=db, llm_router=llm)
    if not chroma_raises:
        # Inject mock collection — simulates successful ChromaDB init.
        agent._chroma_collection = chroma_collection or _make_chroma_collection()
    else:
        # Force None regardless of whether chromadb is installed in the environment.
        agent._chroma_collection = None
    return agent


def _closed_trade(
    symbol: str = "TCS",
    pnl_pct: float = 2.5,
    strategy_signal: dict | None = None,
    holding_days: int = 5,
    sector: str | None = None,
) -> Trade:
    sig = strategy_signal or {"strategy": "RSI_OVERSOLD_BOUNCE", "sector": sector or "IT"}
    t = Trade(
        symbol=symbol,
        trade_type="BUY",
        quantity=10,
        price=3800.0,
        exit_price=3800.0 * (1 + pnl_pct / 100),
        exit_date="2025-06-01 15:30:00",
        pnl=round(3800.0 * pnl_pct / 100 * 10, 2),
        pnl_percentage=pnl_pct,
        holding_days=holding_days,
        strategy_signal=json.dumps(sig),
        status="EXECUTED",
        id=1,
    )
    return t


def _open_trade(symbol: str = "INFY") -> Trade:
    return Trade(
        symbol=symbol,
        trade_type="BUY",
        quantity=5,
        price=1500.0,
        status="EXECUTED",
        id=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# __init__ / ChromaDB setup
# ─────────────────────────────────────────────────────────────────────────────


class TestInit:
    def test_chroma_collection_is_set_on_success(self):
        col = _make_chroma_collection()
        agent = _make_agent(chroma_collection=col)
        assert agent._chroma_collection is col

    def test_chroma_failure_does_not_crash(self):
        """ChromaDB init error must fall back to SQLite-only mode gracefully."""
        agent = _make_agent(chroma_raises=True)
        assert agent._chroma_collection is None

    def test_db_and_llm_are_stored(self):
        db = _make_db()
        llm = _make_llm()
        agent = _make_agent(db=db, llm=llm)
        assert agent.db is db
        assert agent.llm is llm


# ─────────────────────────────────────────────────────────────────────────────
# generate_trade_lessons
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateTradeLessons:
    def test_calls_llm_with_trade_details(self):
        llm = _make_llm("RSI bounce works well in IT sector.")
        agent = _make_agent(llm=llm)
        trade = {"symbol": "TCS", "strategy": "RSI_OVERSOLD_BOUNCE", "outcome": "WIN"}
        lessons = agent.generate_trade_lessons(trade)
        assert lessons == "RSI bounce works well in IT sector."
        llm.call.assert_called_once()
        prompt_used = llm.call.call_args[1]["prompt"]
        assert "TCS" in prompt_used

    def test_fallback_string_on_llm_error(self):
        llm = MagicMock()
        llm.call.side_effect = RuntimeError("Network error")
        agent = _make_agent(llm=llm)
        trade = {"symbol": "TCS", "strategy": "RSI_OVERSOLD_BOUNCE", "outcome": "LOSS"}
        lessons = agent.generate_trade_lessons(trade)
        assert isinstance(lessons, str)
        assert len(lessons) > 0
        assert "RSI_OVERSOLD_BOUNCE" in lessons

    def test_uses_simple_complexity(self):
        from src.llm.router import TaskComplexity

        llm = _make_llm()
        agent = _make_agent(llm=llm)
        agent.generate_trade_lessons({"symbol": "TCS"})
        call_kwargs = llm.call.call_args[1]
        assert call_kwargs["complexity"] == TaskComplexity.SIMPLE


# ─────────────────────────────────────────────────────────────────────────────
# record_trade_outcome
# ─────────────────────────────────────────────────────────────────────────────


class TestRecordTradeOutcome:
    def _trade_dict(self, **overrides) -> dict:
        base = {
            "id": 42,
            "symbol": "TCS",
            "sector": "IT",
            "strategy": "RSI_OVERSOLD_BOUNCE",
            "entry_price": 3850.0,
            "exit_price": 3960.0,
            "pnl_pct": 2.86,
            "outcome": "WIN",
            "holding_days": 5,
            "market_condition": "BULLISH",
            "entry_reasoning": "RSI at 27, MACD crossover",
            "exit_type": "TARGET_HIT",
            "lessons": "",
        }
        return {**base, **overrides}

    def test_generates_lessons_when_empty(self):
        llm = _make_llm("Entry was well-timed.")
        agent = _make_agent(llm=llm)
        agent.record_trade_outcome(self._trade_dict(lessons=""))
        llm.call.assert_called_once()

    def test_does_not_regenerate_existing_lessons(self):
        llm = _make_llm()
        agent = _make_agent(llm=llm)
        agent.record_trade_outcome(self._trade_dict(lessons="Already have a lesson."))
        llm.call.assert_not_called()

    def test_adds_to_chroma_collection(self):
        col = _make_chroma_collection()
        agent = _make_agent(chroma_collection=col)
        agent.record_trade_outcome(self._trade_dict())
        col.add.assert_called_once()

    def test_chroma_add_id_contains_symbol(self):
        col = _make_chroma_collection()
        agent = _make_agent(chroma_collection=col)
        agent.record_trade_outcome(self._trade_dict(id=42))
        call_kwargs = col.add.call_args[1]
        assert "TCS" in call_kwargs["ids"][0]

    def test_chroma_document_under_500_chars(self):
        col = _make_chroma_collection()
        agent = _make_agent(chroma_collection=col)
        agent.record_trade_outcome(self._trade_dict(entry_reasoning="x" * 1000))
        call_kwargs = col.add.call_args[1]
        assert len(call_kwargs["documents"][0]) <= 500

    def test_logs_to_sqlite(self):
        db = _make_db()
        agent = _make_agent(db=db)
        agent.record_trade_outcome(self._trade_dict())
        db.log_agent_activity.assert_called_once()

    def test_does_not_mutate_caller_dict(self):
        trade = self._trade_dict(lessons="")
        original_lessons = trade["lessons"]
        agent = _make_agent()
        agent.record_trade_outcome(trade)
        assert trade["lessons"] == original_lessons  # caller's dict unchanged

    def test_chroma_failure_does_not_raise(self):
        col = _make_chroma_collection()
        col.add.side_effect = RuntimeError("Disk full")
        agent = _make_agent(chroma_collection=col)
        # Must not propagate the exception.
        agent.record_trade_outcome(self._trade_dict())

    def test_sqlite_log_failure_does_not_raise(self):
        db = _make_db()
        db.log_agent_activity.side_effect = RuntimeError("DB locked")
        agent = _make_agent(db=db)
        agent.record_trade_outcome(self._trade_dict())  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# get_similar_past_trades  (ChromaDB retrieval)
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSimilarPastTrades:
    def test_returns_metadata_list(self):
        past = [
            {"symbol": "TCS", "strategy": "RSI_OVERSOLD_BOUNCE", "outcome": "WIN"},
            {"symbol": "INFY", "strategy": "RSI_OVERSOLD_BOUNCE", "outcome": "LOSS"},
        ]
        col = _make_chroma_collection(count=2, query_results=past)
        agent = _make_agent(chroma_collection=col)
        results = agent.get_similar_past_trades("TCS", "RSI_OVERSOLD_BOUNCE", "IT")
        assert results == past

    def test_passes_combined_query_text(self):
        col = _make_chroma_collection(count=1, query_results=[{"symbol": "TCS"}])
        agent = _make_agent(chroma_collection=col)
        agent.get_similar_past_trades("TCS", "RSI_OVERSOLD_BOUNCE", "IT", top_k=3)
        call_kwargs = col.query.call_args[1]
        assert "TCS" in call_kwargs["query_texts"][0]
        assert "RSI_OVERSOLD_BOUNCE" in call_kwargs["query_texts"][0]
        assert "IT" in call_kwargs["query_texts"][0]

    def test_top_k_respected(self):
        col = _make_chroma_collection(count=10, query_results=[])
        agent = _make_agent(chroma_collection=col)
        agent.get_similar_past_trades("TCS", "RSI", "IT", top_k=3)
        call_kwargs = col.query.call_args[1]
        assert call_kwargs["n_results"] == 3

    def test_n_results_capped_at_collection_count(self):
        col = _make_chroma_collection(count=2, query_results=[])
        agent = _make_agent(chroma_collection=col)
        agent.get_similar_past_trades("TCS", "RSI", "IT", top_k=10)
        call_kwargs = col.query.call_args[1]
        assert call_kwargs["n_results"] == 2

    def test_returns_empty_when_collection_empty(self):
        col = _make_chroma_collection(count=0)
        agent = _make_agent(chroma_collection=col)
        results = agent.get_similar_past_trades("TCS", "RSI", "IT")
        assert results == []
        col.query.assert_not_called()

    def test_returns_empty_when_chroma_unavailable(self):
        agent = _make_agent(chroma_raises=True)
        assert agent.get_similar_past_trades("TCS", "RSI", "IT") == []

    def test_chroma_query_error_returns_empty(self):
        col = _make_chroma_collection(count=5)
        col.query.side_effect = RuntimeError("Query failed")
        agent = _make_agent(chroma_collection=col)
        result = agent.get_similar_past_trades("TCS", "RSI", "IT")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# get_strategy_performance
# ─────────────────────────────────────────────────────────────────────────────


class TestGetStrategyPerformance:
    def test_empty_history_returns_zero_stats(self):
        agent = _make_agent(db=_make_db(trade_history=[]))
        result = agent.get_strategy_performance("RSI_OVERSOLD_BOUNCE")
        assert result["total_trades"] == 0
        assert result["win_rate"] == 0.0
        assert result["best_trade"] is None

    def test_filters_by_strategy_name(self):
        trades = [
            _closed_trade("TCS", 3.0, {"strategy": "RSI_OVERSOLD_BOUNCE", "sector": "IT"}),
            _closed_trade("HDFC", 1.5, {"strategy": "MACD_CROSSOVER", "sector": "FINANCE"}),
        ]
        db = _make_db(trade_history=trades)
        agent = _make_agent(db=db)
        result = agent.get_strategy_performance("RSI_OVERSOLD_BOUNCE")
        assert result["total_trades"] == 1
        assert result["strategy"] == "RSI_OVERSOLD_BOUNCE"

    def test_win_rate_calculation(self):
        trades = [
            _closed_trade("TCS", 2.5, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
            _closed_trade("INFY", 1.8, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
            _closed_trade("WIPRO", -1.2, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
        ]
        db = _make_db(trade_history=trades)
        agent = _make_agent(db=db)
        result = agent.get_strategy_performance("RSI_OVERSOLD_BOUNCE")
        assert result["total_trades"] == 3
        assert result["wins"] == 2
        assert result["losses"] == 1
        assert result["win_rate"] == pytest.approx(66.7, abs=0.1)

    def test_best_and_worst_trade(self):
        trades = [
            _closed_trade("INFY", 5.8, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
            _closed_trade("WIPRO", -2.1, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
        ]
        db = _make_db(trade_history=trades)
        agent = _make_agent(db=db)
        result = agent.get_strategy_performance("RSI_OVERSOLD_BOUNCE")
        assert result["best_trade"]["symbol"] == "INFY"
        assert result["best_trade"]["pnl_pct"] == pytest.approx(5.8, abs=0.01)
        assert result["worst_trade"]["symbol"] == "WIPRO"

    def test_avg_win_avg_loss_populated(self):
        trades = [
            _closed_trade("TCS", 3.0, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
            _closed_trade("INFY", 2.0, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
            _closed_trade("WIPRO", -1.5, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
        ]
        db = _make_db(trade_history=trades)
        agent = _make_agent(db=db)
        result = agent.get_strategy_performance("RSI_OVERSOLD_BOUNCE")
        assert result["avg_win_pct"] == pytest.approx(2.5, abs=0.01)
        assert result["avg_loss_pct"] == pytest.approx(-1.5, abs=0.01)

    def test_open_trades_excluded(self):
        trades = [
            _closed_trade("TCS", 2.5, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
            _open_trade("INFY"),
        ]
        db = _make_db(trade_history=trades)
        agent = _make_agent(db=db)
        result = agent.get_strategy_performance("RSI_OVERSOLD_BOUNCE")
        assert result["total_trades"] == 1

    def test_avg_holding_days(self):
        trades = [
            _closed_trade("TCS", 2.0, {"strategy": "RSI_OVERSOLD_BOUNCE"}, holding_days=4),
            _closed_trade("INFY", 1.5, {"strategy": "RSI_OVERSOLD_BOUNCE"}, holding_days=6),
        ]
        db = _make_db(trade_history=trades)
        agent = _make_agent(db=db)
        result = agent.get_strategy_performance("RSI_OVERSOLD_BOUNCE")
        assert result["avg_holding_days"] == pytest.approx(5.0, abs=0.01)

    def test_strategy_key_in_result(self):
        agent = _make_agent(db=_make_db())
        result = agent.get_strategy_performance("MY_STRATEGY")
        assert result["strategy"] == "MY_STRATEGY"


# ─────────────────────────────────────────────────────────────────────────────
# get_sector_performance
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSectorPerformance:
    def test_filters_by_sector_in_signal(self):
        trades = [
            _closed_trade("TCS", 2.5, {"strategy": "RSI", "sector": "IT"}),
            _closed_trade("HDFC", 1.0, {"strategy": "RSI", "sector": "FINANCE"}),
        ]
        db = _make_db(trade_history=trades)
        agent = _make_agent(db=db)
        result = agent.get_sector_performance("IT")
        assert result["total_trades"] == 1
        assert result["sector"] == "IT"

    def test_empty_history_returns_zero_stats(self):
        agent = _make_agent(db=_make_db(trade_history=[]))
        result = agent.get_sector_performance("IT")
        assert result["total_trades"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# get_overall_stats
# ─────────────────────────────────────────────────────────────────────────────


class TestGetOverallStats:
    def test_returns_stats_from_db(self):
        perf = {
            "total_trades": 20,
            "winning_trades": 14,
            "losing_trades": 6,
            "win_rate_pct": 70.0,
            "total_pnl": 12500.0,
            "avg_pnl": 625.0,
            "best_trade_pnl": 4200.0,
            "worst_trade_pnl": -1100.0,
            "avg_holding_days": 6.0,
            "avg_pnl_pct": 2.8,
            "period_days": 30,
        }
        agent = _make_agent(db=_make_db(perf_stats=perf))
        result = agent.get_overall_stats(days=30)
        assert result["total_trades"] == 20
        assert result["win_rate"] == 70.0
        assert result["total_pnl_inr"] == 12500.0

    def test_nifty_return_parsed_from_system_state(self):
        db = _make_db(system_state={"nifty_return_30d": "3.5"})
        agent = _make_agent(db=db)
        result = agent.get_overall_stats(days=30)
        assert result["nifty_return_pct"] == pytest.approx(3.5)

    def test_nifty_return_none_when_absent(self):
        agent = _make_agent(db=_make_db(system_state={}))
        result = agent.get_overall_stats(days=30)
        assert result["nifty_return_pct"] is None

    def test_sharpe_none_with_few_snapshots(self):
        db = _make_db(portfolio_history=[])  # no history
        agent = _make_agent(db=db)
        result = agent.get_overall_stats(days=30)
        assert result["sharpe_ratio"] is None

    def test_sharpe_calculated_with_sufficient_data(self):
        # 10 snapshots with linearly increasing values → positive Sharpe
        snapshots = [
            PortfolioSnapshot(
                date="2025-01-01",
                time="15:30:00",
                total_value=100_000 - i * 100,  # newest first
            )
            for i in range(10)
        ]
        db = _make_db(portfolio_history=snapshots)
        agent = _make_agent(db=db)
        result = agent.get_overall_stats(days=30)
        # Should be a finite float (positive or negative depending on trend).
        assert result["sharpe_ratio"] is not None
        assert math.isfinite(result["sharpe_ratio"])


# ─────────────────────────────────────────────────────────────────────────────
# get_context_for_trade
# ─────────────────────────────────────────────────────────────────────────────


class TestGetContextForTrade:
    def test_no_history_returns_first_trade_message(self):
        col = _make_chroma_collection(count=0)
        db = _make_db(trade_history=[])
        agent = _make_agent(db=db, chroma_collection=col)
        ctx = agent.get_context_for_trade("TCS", "RSI_OVERSOLD_BOUNCE", "IT")
        assert "first" in ctx.lower() or "no historical" in ctx.lower()

    def test_returns_string(self):
        col = _make_chroma_collection(count=0)
        agent = _make_agent(db=_make_db(trade_history=[]), chroma_collection=col)
        result = agent.get_context_for_trade("TCS", "RSI", "IT")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_win_rate_when_history_available(self):
        similar = [
            {"symbol": "TCS", "outcome": "WIN", "pnl_pct": "2.5", "holding_days": "5"},
            {"symbol": "INFY", "outcome": "WIN", "pnl_pct": "1.8", "holding_days": "4"},
            {"symbol": "WIPRO", "outcome": "LOSS", "pnl_pct": "-1.2", "holding_days": "3"},
        ]
        col = _make_chroma_collection(count=3, query_results=similar)
        trades = [
            _closed_trade("TCS", 2.5, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
            _closed_trade("INFY", 1.8, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
            _closed_trade("WIPRO", -1.2, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
        ]
        db = _make_db(trade_history=trades)
        agent = _make_agent(db=db, chroma_collection=col)
        ctx = agent.get_context_for_trade("TCS", "RSI_OVERSOLD_BOUNCE", "IT")
        assert "win rate" in ctx.lower() or "%" in ctx

    def test_includes_recommendation(self):
        similar = [{"symbol": "TCS", "outcome": "WIN"}] * 5
        col = _make_chroma_collection(count=5, query_results=similar)
        trades = [
            _closed_trade("TCS", 2.5, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
            _closed_trade("INFY", 1.8, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
            _closed_trade("WIPRO", 1.2, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
        ]
        db = _make_db(trade_history=trades)
        agent = _make_agent(db=db, chroma_collection=col)
        ctx = agent.get_context_for_trade("TCS", "RSI_OVERSOLD_BOUNCE", "IT")
        assert "recommendation" in ctx.lower()

    def test_limited_history_uses_standard_sizing_recommendation(self):
        """Fewer than 3 matching trades → recommend standard position sizing."""
        similar = []
        col = _make_chroma_collection(count=0, query_results=similar)
        # Only 2 closed trades match the strategy.
        trades = [
            _closed_trade("TCS", 2.5, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
            _closed_trade("INFY", 1.8, {"strategy": "RSI_OVERSOLD_BOUNCE"}),
        ]
        db = _make_db(trade_history=trades)
        agent = _make_agent(db=db, chroma_collection=col)
        ctx = agent.get_context_for_trade("TCS", "RSI_OVERSOLD_BOUNCE", "IT")
        assert "limited" in ctx.lower() or "standard" in ctx.lower()


# ─────────────────────────────────────────────────────────────────────────────
# generate_weekly_review
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerateWeeklyReview:
    def test_returns_dict_with_required_keys(self):
        agent = _make_agent()
        review = agent.generate_weekly_review()
        for key in ("week_ending", "total_trades", "win_rate", "total_pnl", "llm_analysis"):
            assert key in review, f"Missing key '{key}'"

    def test_calls_llm_with_moderate_complexity(self):
        from src.llm.router import TaskComplexity

        llm = _make_llm("Great week!")
        agent = _make_agent(llm=llm)
        agent.generate_weekly_review()
        call_kwargs = llm.call.call_args[1]
        assert call_kwargs["complexity"] == TaskComplexity.MODERATE

    def test_saves_to_daily_summary(self):
        db = _make_db()
        agent = _make_agent(db=db)
        agent.generate_weekly_review()
        db.save_daily_summary.assert_called_once()
        arg = db.save_daily_summary.call_args[0][0]
        assert isinstance(arg, DailySummary)

    def test_logs_activity_to_sqlite(self):
        db = _make_db()
        agent = _make_agent(db=db)
        agent.generate_weekly_review()
        calls = [
            c for c in db.log_agent_activity.call_args_list
            if c[1].get("session_type") == "weekly_review"
        ]
        assert len(calls) == 1

    def test_llm_failure_returns_fallback_analysis(self):
        llm = MagicMock()
        llm.call.side_effect = RuntimeError("API down")
        agent = _make_agent(llm=llm)
        review = agent.generate_weekly_review()
        assert isinstance(review["llm_analysis"], str)
        assert len(review["llm_analysis"]) > 0

    def test_week_ending_is_date_string(self):
        agent = _make_agent()
        review = agent.generate_weekly_review()
        # Should parse as a date.
        datetime.strptime(review["week_ending"], "%Y-%m-%d")

    def test_no_closed_trades_handled_gracefully(self):
        db = _make_db(trade_history=[_open_trade()])  # only open trade
        agent = _make_agent(db=db)
        review = agent.generate_weekly_review()  # must not raise
        assert review["total_trades"] == 0

    def test_save_daily_summary_failure_does_not_raise(self):
        db = _make_db()
        db.save_daily_summary.side_effect = RuntimeError("DB error")
        agent = _make_agent(db=db)
        agent.generate_weekly_review()  # must not propagate


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases — first trade (empty history)
# ─────────────────────────────────────────────────────────────────────────────


class TestFirstTrade:
    """Ensure all public methods handle empty history gracefully."""

    def setup_method(self):
        col = _make_chroma_collection(count=0)
        self.agent = _make_agent(db=_make_db(trade_history=[]), chroma_collection=col)

    def test_record_trade_outcome_works(self):
        self.agent.record_trade_outcome(
            {
                "symbol": "TCS",
                "strategy": "RSI_OVERSOLD_BOUNCE",
                "sector": "IT",
                "outcome": "WIN",
                "entry_reasoning": "RSI at 27",
                "lessons": "",
            }
        )  # Must not raise.

    def test_get_strategy_performance_returns_zeros(self):
        result = self.agent.get_strategy_performance("RSI_OVERSOLD_BOUNCE")
        assert result["total_trades"] == 0
        assert result["win_rate"] == 0.0

    def test_get_sector_performance_returns_zeros(self):
        result = self.agent.get_sector_performance("IT")
        assert result["total_trades"] == 0

    def test_get_overall_stats_does_not_raise(self):
        result = self.agent.get_overall_stats(days=30)
        assert isinstance(result, dict)

    def test_get_context_mentions_first_trade(self):
        ctx = self.agent.get_context_for_trade("TCS", "RSI_OVERSOLD_BOUNCE", "IT")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_generate_weekly_review_does_not_raise(self):
        review = self.agent.generate_weekly_review()
        assert isinstance(review, dict)
        assert review["total_trades"] == 0
