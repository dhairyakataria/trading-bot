"""Tests for ResearchAgent — autonomous LLM-driven research agent.

All external I/O (LLM, web, news, DB) is mocked so tests are:
 - Fast (no network calls)
 - Deterministic (no real LLM randomness)
 - Isolated (no API keys required)

Test matrix
───────────
 1.  _parse_tool_call — normal formats
 2.  _parse_tool_call — edge-case / varied LLM formatting
 3.  _parse_tool_call — unparseable returns (None, "")
 4.  _parse_final_answer — clean JSON after marker
 5.  _parse_final_answer — JSON wrapped in markdown fences
 6.  _parse_final_answer — no marker → None
 7.  _parse_final_answer — invalid JSON → None
 8.  research_stock — happy path (2-step: tool then answer)
 9.  research_stock — search budget enforced (agent stops calling search tools)
10.  research_stock — article budget enforced
11.  research_stock — LLM unavailable → force conclusion
12.  research_stock — max steps reached → force_conclusion called
13.  research_stock — two consecutive bad JSON → force_conclusion
14.  research_stock — all tools fail (tool errors) → still returns dict
15.  research_stock — DB logging called with correct params
16.  research_stock — DB logging failure doesn't crash agent
17.  morning_briefing — happy path (tools → answer)
18.  morning_briefing — correct session type used
19.  _force_conclusion — LLM succeeds and returns parsed JSON
20.  _force_conclusion — LLM fails → minimal fallback dict
21.  _force_conclusion — LLM returns raw JSON (no FINAL_ANSWER marker)
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from src.agents.research_agent import ResearchAgent
from src.llm.budget_manager import BudgetManager, SessionBudget, SESSION_LIMITS
from src.llm.router import LLMUnavailableError, TaskComplexity


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_agent(
    llm_responses: list[str] | None = None,
    search_results: list[dict] | None = None,
    news_articles: list[dict] | None = None,
    fii_dii: dict | None = None,
    global_markets: dict | None = None,
    india_vix: dict | None = None,
    db: Any = None,
) -> tuple[ResearchAgent, MagicMock]:
    """Build a ResearchAgent with all dependencies mocked.

    Returns (agent, mock_llm_router) so individual tests can inspect calls.
    """
    config = {}

    # LLM router
    llm_router = MagicMock()
    responses = llm_responses or ["FINAL_ANSWER:\n" + json.dumps(_minimal_stock_answer())]
    llm_router.call.side_effect = responses

    # Budget manager — real in-memory instance (no DB)
    budget_manager = BudgetManager(db=None)

    # Web-search tool
    web_search = MagicMock()
    web_search.search.return_value = search_results or [
        {"title": "TCS Q3 results", "url": "https://ex.com/1", "snippet": "TCS beats estimates", "published_date": "2026-03-10"},
    ]
    web_search.read_article.return_value = {
        "title": "TCS beats estimates", "text": "TCS reported strong Q3 revenue...", "authors": [], "published_date": "2026-03-10",
    }

    # News fetcher
    news_fetcher = MagicMock()
    news_fetcher.fetch_stock_specific_news.return_value = news_articles or [
        {"title": "TCS stock rises", "url": "https://ex.com/2", "source": "ET", "published_date": "2026-03-10", "snippet": ""},
    ]
    news_fetcher.fetch_google_news_rss.return_value = news_articles or []
    news_fetcher.fetch_market_news.return_value = news_articles or []
    news_fetcher.get_fii_dii_data.return_value = fii_dii or {
        "date": "2026-03-17", "fii_net": 500.0, "dii_net": -200.0,
        "fii_buy": 2000.0, "fii_sell": 1500.0, "dii_buy": 800.0, "dii_sell": 1000.0,
        "source": "NSE",
    }
    news_fetcher.get_global_market_status.return_value = global_markets or {
        "sp500":   {"last_price": 5200.0, "change_pct": 0.3},
        "nasdaq":  {"last_price": 16100.0, "change_pct": -0.5},
        "usd_inr": {"last_price": 86.5,   "change_pct": 0.1},
        "crude_oil": {"last_price": 78.5, "change_pct": -1.2},
        "timestamp": "2026-03-18 08:00:00",
    }
    news_fetcher.get_india_vix.return_value = india_vix or {
        "vix": 14.8, "signal": "LOW_FEAR", "interpretation": "Low fear", "timestamp": "2026-03-18 08:00:00",
    }

    agent = ResearchAgent(
        config=config,
        llm_router=llm_router,
        budget_manager=budget_manager,
        web_search=web_search,
        news_fetcher=news_fetcher,
        db_manager=db,
    )
    return agent, llm_router


def _minimal_stock_answer(
    symbol: str = "TCS",
    recommendation: str = "PROCEED",
) -> dict:
    return {
        "symbol":           symbol,
        "research_summary": "TCS looks solid based on news and FII flows.",
        "sentiment":        "POSITIVE",
        "confidence":       8,
        "risks":            ["US tech weakness"],
        "opportunities":    ["FII buying"],
        "recommendation":   recommendation,
        "reasoning":        "Strong fundamentals, proceed with normal size.",
        "suggested_adjustments": {"reduce_position_size": False, "tighter_stop_loss": False, "reason": ""},
        "news_items_reviewed":   [{"title": "TCS Q3 beats", "impact": "POSITIVE"}],
    }


def _minimal_briefing_answer(date: str = "2026-03-18") -> dict:
    return {
        "date":                   date,
        "global_sentiment":       "NEUTRAL",
        "us_markets":             {"sp500_change_pct": 0.3, "nasdaq_change_pct": -0.5},
        "asian_markets_summary":  "Nikkei up 0.4%, Hang Seng flat",
        "fii_dii":                {"fii_net": 500, "dii_net": -200, "interpretation": "FII buying"},
        "india_vix":              {"value": 14.8, "interpretation": "Low fear"},
        "crude_oil":              {"price": 78.5, "change_pct": -1.2},
        "usd_inr":                {"rate": 86.5, "change_pct": 0.1},
        "key_events_today":       ["RBI MPC minutes at 11:30 AM"],
        "sectors_to_watch":       [{"sector": "IT", "direction": "CAUTIOUS", "reason": "Nasdaq dip"}],
        "overall_market_outlook": "Cautiously optimistic.",
        "risk_level":             "MEDIUM",
        "recommendation":         "Normal trading day.",
    }


# ---------------------------------------------------------------------------
# 1-3  _parse_tool_call
# ---------------------------------------------------------------------------

class TestParseToolCall:

    def setup_method(self):
        self.agent, _ = _make_agent()

    def test_standard_format_with_arg(self):
        resp = "I will search now.\nTOOL_CALL: web_search(TCS quarterly results 2025)"
        name, param = self.agent._parse_tool_call(resp)
        assert name == "web_search"
        assert param == "TCS quarterly results 2025"

    def test_quoted_param_stripped(self):
        resp = 'TOOL_CALL: web_search("TCS earnings Q3")'
        name, param = self.agent._parse_tool_call(resp)
        assert name == "web_search"
        assert param == "TCS earnings Q3"

    def test_no_arg_tool(self):
        resp = "TOOL_CALL: get_fii_dii_data()"
        name, param = self.agent._parse_tool_call(resp)
        assert name == "get_fii_dii_data"
        assert param == ""

    def test_no_arg_tool_no_parens(self):
        """Some LLMs omit the parentheses for no-arg tools."""
        resp = "TOOL_CALL: get_market_news"
        name, param = self.agent._parse_tool_call(resp)
        assert name == "get_market_news"
        assert param == ""

    def test_space_separated_no_parens(self):
        """Some models write: TOOL_CALL: web_search TCS stock news"""
        resp = "TOOL_CALL: web_search TCS stock news"
        name, param = self.agent._parse_tool_call(resp)
        assert name == "web_search"
        assert param == "TCS stock news"

    def test_url_param(self):
        resp = "TOOL_CALL: read_article(https://example.com/tcs-results)"
        name, param = self.agent._parse_tool_call(resp)
        assert name == "read_article"
        assert param == "https://example.com/tcs-results"

    def test_get_stock_news(self):
        resp = "TOOL_CALL: get_stock_news(INFY)"
        name, param = self.agent._parse_tool_call(resp)
        assert name == "get_stock_news"
        assert param == "INFY"

    def test_case_insensitive_prefix(self):
        """Prefix 'tool_call:' or 'TOOL_CALL:' both work."""
        resp = "tool_call: web_search(Reliance Industries news)"
        name, param = self.agent._parse_tool_call(resp)
        assert name == "web_search"
        assert param == "Reliance Industries news"

    def test_tool_call_in_middle_of_response(self):
        resp = "I need more information about the stock.\nTOOL_CALL: get_india_vix()\nThis will help me decide."
        name, param = self.agent._parse_tool_call(resp)
        assert name == "get_india_vix"
        assert param == ""

    def test_unparseable_returns_none(self):
        resp = "I think I should just answer directly."
        name, param = self.agent._parse_tool_call(resp)
        assert name is None
        assert param == ""

    def test_empty_response(self):
        name, param = self.agent._parse_tool_call("")
        assert name is None
        assert param == ""


# ---------------------------------------------------------------------------
# 4-7  _parse_final_answer
# ---------------------------------------------------------------------------

class TestParseFinalAnswer:

    def setup_method(self):
        self.agent, _ = _make_agent()

    def test_clean_json(self):
        resp = 'FINAL_ANSWER:\n{"symbol": "TCS", "recommendation": "PROCEED"}'
        result = self.agent._parse_final_answer(resp)
        assert result == {"symbol": "TCS", "recommendation": "PROCEED"}

    def test_json_with_markdown_fences(self):
        resp = "FINAL_ANSWER:\n```json\n{\"symbol\": \"TCS\"}\n```"
        result = self.agent._parse_final_answer(resp)
        assert result is not None
        assert result["symbol"] == "TCS"

    def test_json_with_plain_fences(self):
        resp = "FINAL_ANSWER:\n```\n{\"a\": 1}\n```"
        result = self.agent._parse_final_answer(resp)
        assert result == {"a": 1}

    def test_no_marker_returns_none(self):
        resp = '{"symbol": "TCS", "recommendation": "PROCEED"}'
        result = self.agent._parse_final_answer(resp)
        assert result is None

    def test_invalid_json_returns_none(self):
        resp = "FINAL_ANSWER:\n{invalid json here"
        result = self.agent._parse_final_answer(resp)
        assert result is None

    def test_json_after_preamble(self):
        """LLM sometimes adds prose before the JSON block."""
        resp = (
            "FINAL_ANSWER:\nHere is my assessment:\n"
            '{"symbol": "HDFC", "recommendation": "HOLD_OFF"}'
        )
        result = self.agent._parse_final_answer(resp)
        assert result is not None
        assert result["recommendation"] == "HOLD_OFF"

    def test_nested_json(self):
        payload = _minimal_stock_answer()
        resp = f"FINAL_ANSWER:\n{json.dumps(payload)}"
        result = self.agent._parse_final_answer(resp)
        assert result is not None
        assert result["recommendation"] == "PROCEED"
        assert "suggested_adjustments" in result


# ---------------------------------------------------------------------------
# 8  research_stock — happy path
# ---------------------------------------------------------------------------

class TestResearchStockHappyPath:

    def test_single_step_final_answer(self):
        """Agent gets a FINAL_ANSWER immediately (no tools needed)."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=["FINAL_ANSWER:\n" + json.dumps(answer)])

        result = agent.research_stock("TCS", "RSI oversold at 27")

        assert result["symbol"] == "TCS"
        assert result["recommendation"] == "PROCEED"
        assert result["confidence"] == 8
        llm.call.assert_called_once()

    def test_tool_call_then_final_answer(self):
        """Agent calls one tool, then produces FINAL_ANSWER."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: get_stock_news(TCS)",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        result = agent.research_stock("TCS", "MACD bullish crossover")

        assert result["recommendation"] == "PROCEED"
        assert llm.call.call_count == 2
        # web_search tool was NOT used, news_fetcher was
        agent.news_fetcher.fetch_stock_specific_news.assert_called_once_with("TCS")

    def test_multiple_tools_then_final_answer(self):
        """Agent calls several tools before concluding."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: get_global_markets()",
            "TOOL_CALL: get_fii_dii_data()",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        result = agent.research_stock("INFY", "Volume breakout detected")

        assert result["recommendation"] == "PROCEED"
        assert llm.call.call_count == 3
        agent.news_fetcher.get_global_market_status.assert_called_once()
        agent.news_fetcher.get_fii_dii_data.assert_called_once()

    def test_result_has_all_required_keys(self):
        """Output must contain every key specified in the task interface."""
        required_keys = {
            "symbol", "research_summary", "sentiment", "confidence",
            "risks", "opportunities", "recommendation", "reasoning",
            "suggested_adjustments", "news_items_reviewed",
            "tools_used", "search_calls_used", "articles_read",
        }
        answer = _minimal_stock_answer()
        agent, _ = _make_agent(llm_responses=["FINAL_ANSWER:\n" + json.dumps(answer)])
        result = agent.research_stock("TCS", "RSI oversold")
        assert required_keys.issubset(set(result.keys()))

    def test_recommendation_values_valid(self):
        for rec in ("PROCEED", "PROCEED_WITH_CAUTION", "HOLD_OFF", "AVOID"):
            ans = _minimal_stock_answer(recommendation=rec)
            agent, _ = _make_agent(llm_responses=["FINAL_ANSWER:\n" + json.dumps(ans)])
            result = agent.research_stock("TCS", "signal")
            assert result["recommendation"] == rec


# ---------------------------------------------------------------------------
# 9-10  Budget enforcement
# ---------------------------------------------------------------------------

class TestBudgetEnforcement:

    def test_search_budget_exhausted_mid_loop(self):
        """When search budget runs out, agent gets a note and should stop searching."""
        # Session 'stock_research' allows 3 search calls
        # We exhaust all 3, then the 4th search attempt gets blocked
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: web_search(TCS news)",       # search 1
            "TOOL_CALL: web_search(TCS sector)",     # search 2
            "TOOL_CALL: web_search(TCS risks)",      # search 3
            "TOOL_CALL: web_search(TCS more news)",  # search 4 — budget exhausted
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        result = agent.research_stock("TCS", "signal")

        # Should still return a valid result
        assert result["recommendation"] == "PROCEED"
        # The 4th web_search call should NOT have been made
        assert agent.web_search.search.call_count == 3

    def test_article_budget_exhausted(self):
        """read_article calls exceeding budget are blocked."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: read_article(https://ex.com/1)",  # article 1
            "TOOL_CALL: read_article(https://ex.com/2)",  # article 2
            "TOOL_CALL: read_article(https://ex.com/3)",  # article 3 — budget exhausted (stock_research allows 2)
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        result = agent.research_stock("TCS", "signal")

        assert result["recommendation"] == "PROCEED"
        # Only 2 articles should have been fetched (stock_research budget)
        assert agent.web_search.read_article.call_count == 2

    def test_no_budget_cost_for_free_tools(self):
        """get_fii_dii_data / get_global_markets / get_india_vix don't consume search budget."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: get_fii_dii_data()",
            "TOOL_CALL: get_global_markets()",
            "TOOL_CALL: get_india_vix()",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        session = agent.budget_manager.create_session("stock_research")
        initial_search = session.remaining_search_calls  # should stay at 3

        result = agent.research_stock("TCS", "signal")

        # These tools don't decrement search budget — verify search_calls_used = 0
        assert result["search_calls_used"] == 0
        assert result["articles_read"] == 0


# ---------------------------------------------------------------------------
# 11  LLM unavailable
# ---------------------------------------------------------------------------

class TestLLMUnavailable:

    def test_llm_unavailable_returns_fallback(self):
        """If the LLM is unavailable, research_stock returns a safe fallback dict."""
        agent, llm = _make_agent()
        llm.call.side_effect = LLMUnavailableError("All providers exhausted")

        result = agent.research_stock("TCS", "RSI oversold")

        assert "recommendation" in result
        assert result["recommendation"] in ("PROCEED_WITH_CAUTION", "HOLD_OFF", "AVOID")
        assert result["confidence"] <= 5  # low confidence from fallback

    def test_llm_unavailable_still_returns_symbol(self):
        agent, llm = _make_agent()
        llm.call.side_effect = LLMUnavailableError("down")

        result = agent.research_stock("RELIANCE", "signal")

        assert result["symbol"] == "RELIANCE"


# ---------------------------------------------------------------------------
# 12  Max steps → force_conclusion
# ---------------------------------------------------------------------------

class TestMaxSteps:

    def test_max_steps_triggers_force_conclusion(self):
        """Agent that never produces FINAL_ANSWER hits max steps and force-concludes."""
        # stock_research allows 4 LLM calls — fill them all with tool calls + one force call
        tool_response = "TOOL_CALL: get_market_news()"
        force_answer  = _minimal_stock_answer(recommendation="PROCEED_WITH_CAUTION")

        # 4 loop steps (each budgeted) + 1 force conclusion call = 5 LLM responses needed
        # But stock_research max_llm_calls = 4, so loop runs 4 times then force runs once
        responses = [tool_response] * 4 + ["FINAL_ANSWER:\n" + json.dumps(force_answer)]
        agent, llm = _make_agent(llm_responses=responses)

        result = agent.research_stock("TCS", "signal")

        # Should return some result (not crash)
        assert "recommendation" in result
        assert llm.call.call_count >= 4

    def test_force_conclusion_minimal_fallback_on_llm_failure(self):
        """If force-conclusion LLM call also fails, return minimal safe dict."""
        tool_resp   = "TOOL_CALL: get_market_news()"
        # 4 tool responses + force call raises
        responses = ([tool_resp] * 4)
        agent, llm = _make_agent(llm_responses=responses)
        # Make the 5th call (force conclusion) also fail
        original_side_effect = [tool_resp] * 4
        call_count = {"n": 0}
        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 4:
                return tool_resp
            raise LLMUnavailableError("All providers down")
        llm.call.side_effect = side_effect

        result = agent.research_stock("TCS", "signal")

        assert "recommendation" in result
        # Fallback has low confidence
        assert result.get("confidence", 0) <= 5


# ---------------------------------------------------------------------------
# 13  Two consecutive bad JSON → force_conclusion
# ---------------------------------------------------------------------------

class TestBadJsonHandling:

    def test_two_bad_json_triggers_force(self):
        """Two consecutive FINAL_ANSWER with invalid JSON → force_conclusion."""
        force_answer = _minimal_stock_answer()
        responses = [
            "FINAL_ANSWER:\n{invalid",          # bad JSON #1
            "FINAL_ANSWER:\n{also invalid",     # bad JSON #2 → triggers force
            "FINAL_ANSWER:\n" + json.dumps(force_answer),  # force conclusion answer
        ]
        agent, llm = _make_agent(llm_responses=responses)
        result = agent.research_stock("TCS", "signal")

        assert "recommendation" in result

    def test_valid_json_after_one_bad_attempt(self):
        """One bad JSON, then a valid one — agent recovers."""
        answer = _minimal_stock_answer()
        responses = [
            "FINAL_ANSWER:\n{bad json",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ]
        agent, llm = _make_agent(llm_responses=responses)
        result = agent.research_stock("TCS", "signal")

        assert result["recommendation"] == "PROCEED"


# ---------------------------------------------------------------------------
# 14  All tools fail — agent still returns a result
# ---------------------------------------------------------------------------

class TestToolFailures:

    def test_all_tools_return_errors(self):
        """Even if every tool raises or returns an error dict, agent still answers."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(
            llm_responses=[
                "TOOL_CALL: web_search(TCS news)",
                "TOOL_CALL: get_fii_dii_data()",
                "FINAL_ANSWER:\n" + json.dumps(answer),
            ]
        )
        # Make all tool calls fail
        agent.web_search.search.side_effect = Exception("network error")
        agent.news_fetcher.get_fii_dii_data.side_effect = Exception("timeout")

        result = agent.research_stock("TCS", "signal")

        # Agent should still return valid output
        assert "recommendation" in result

    def test_unknown_tool_name(self):
        """LLM requests a non-existent tool — agent notes error and continues."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: nonexistent_tool(param)",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        result = agent.research_stock("TCS", "signal")

        assert result["recommendation"] == "PROCEED"


# ---------------------------------------------------------------------------
# 15-16  DB logging
# ---------------------------------------------------------------------------

class TestDBLogging:

    def test_db_log_called_with_correct_params(self):
        """research_stock calls db.log_agent_activity with the right fields."""
        db = MagicMock()
        answer = _minimal_stock_answer()
        agent, _ = _make_agent(
            llm_responses=["FINAL_ANSWER:\n" + json.dumps(answer)],
            db=db,
        )

        agent.research_stock("TCS", "RSI oversold at 27")

        db.log_agent_activity.assert_called_once()
        call_kwargs = db.log_agent_activity.call_args
        kw = call_kwargs.kwargs if call_kwargs.kwargs else {}
        args = call_kwargs.args if call_kwargs.args else ()

        # Accept either positional or keyword args
        all_args = {**dict(enumerate(args)), **kw}
        assert any("research_agent" in str(v) for v in all_args.values())
        assert any("stock_research" in str(v) for v in all_args.values())

    def test_db_log_failure_does_not_crash(self):
        """A DB error during logging must never propagate to the caller."""
        db = MagicMock()
        db.log_agent_activity.side_effect = Exception("DB unavailable")

        answer = _minimal_stock_answer()
        agent, _ = _make_agent(
            llm_responses=["FINAL_ANSWER:\n" + json.dumps(answer)],
            db=db,
        )

        # Must not raise
        result = agent.research_stock("TCS", "signal")
        assert result["recommendation"] == "PROCEED"

    def test_db_none_skips_logging(self):
        """db_manager=None means no DB calls are made (no AttributeError)."""
        answer = _minimal_stock_answer()
        agent, _ = _make_agent(
            llm_responses=["FINAL_ANSWER:\n" + json.dumps(answer)],
            db=None,
        )
        # Should run without error
        result = agent.research_stock("TCS", "signal")
        assert "recommendation" in result


# ---------------------------------------------------------------------------
# 17-18  morning_briefing
# ---------------------------------------------------------------------------

class TestMorningBriefing:

    def test_happy_path(self):
        """Morning briefing runs tools and returns structured dict."""
        briefing = _minimal_briefing_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: get_global_markets()",
            "TOOL_CALL: get_fii_dii_data()",
            "TOOL_CALL: web_search(India market news overnight)",
            "FINAL_ANSWER:\n" + json.dumps(briefing),
        ])

        result = agent.morning_briefing()

        assert result["global_sentiment"] == "NEUTRAL"
        assert result["risk_level"] == "MEDIUM"
        assert "sectors_to_watch" in result
        assert isinstance(result["sectors_to_watch"], list)

    def test_required_keys_present(self):
        required = {
            "date", "global_sentiment", "us_markets", "asian_markets_summary",
            "fii_dii", "india_vix", "crude_oil", "usd_inr", "key_events_today",
            "sectors_to_watch", "overall_market_outlook", "risk_level", "recommendation",
        }
        briefing = _minimal_briefing_answer()
        agent, _ = _make_agent(llm_responses=["FINAL_ANSWER:\n" + json.dumps(briefing)])
        result = agent.morning_briefing()
        assert required.issubset(set(result.keys()))

    def test_uses_morning_briefing_session_type(self):
        """morning_briefing() must use the 'morning_briefing' session (higher budget)."""
        briefing = _minimal_briefing_answer()
        agent, _ = _make_agent(llm_responses=["FINAL_ANSWER:\n" + json.dumps(briefing)])

        # Patch create_session to verify the right type is used
        original_create = agent.budget_manager.create_session
        captured = []
        def patched_create(session_type):
            captured.append(session_type)
            return original_create(session_type)
        agent.budget_manager.create_session = patched_create

        agent.morning_briefing()
        assert "morning_briefing" in captured

    def test_defaults_applied_when_llm_fails(self):
        """If LLM is down, morning_briefing returns a safe dict with defaults."""
        agent, llm = _make_agent()
        llm.call.side_effect = LLMUnavailableError("all providers down")

        result = agent.morning_briefing()

        assert "global_sentiment" in result
        assert "risk_level" in result
        assert "recommendation" in result

    def test_morning_briefing_calls_correct_tools(self):
        """Verify global markets and FII/DII tools are called during briefing."""
        briefing = _minimal_briefing_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: get_global_markets()",
            "TOOL_CALL: get_fii_dii_data()",
            "FINAL_ANSWER:\n" + json.dumps(briefing),
        ])

        agent.morning_briefing()

        agent.news_fetcher.get_global_market_status.assert_called_once()
        agent.news_fetcher.get_fii_dii_data.assert_called_once()


# ---------------------------------------------------------------------------
# 19-21  _force_conclusion
# ---------------------------------------------------------------------------

class TestForceConclusion:

    def test_force_conclusion_with_history(self):
        """_force_conclusion makes one LLM call and parses the result."""
        force_answer = _minimal_stock_answer(recommendation="PROCEED_WITH_CAUTION")
        agent, llm = _make_agent(
            llm_responses=["FINAL_ANSWER:\n" + json.dumps(force_answer)]
        )

        history = [
            {"role": "tool_result", "content": json.dumps({"fii_net": -500, "dii_net": 800})},
            {"role": "tool_result", "content": "TCS Q3 results beat estimates."},
        ]

        result = agent._force_conclusion(history)

        assert result["recommendation"] == "PROCEED_WITH_CAUTION"
        llm.call.assert_called_once()

    def test_force_conclusion_llm_fails_returns_fallback(self):
        """If the force-conclusion LLM call fails, return minimal fallback."""
        agent, llm = _make_agent()
        llm.call.side_effect = LLMUnavailableError("all down")

        result = agent._force_conclusion([])

        assert "recommendation" in result
        assert result["confidence"] <= 5
        assert result["suggested_adjustments"]["reduce_position_size"] is True

    def test_force_conclusion_raw_json_no_marker(self):
        """LLM returns raw JSON without the FINAL_ANSWER: marker — should be parsed."""
        raw = _minimal_stock_answer(recommendation="HOLD_OFF")
        agent, llm = _make_agent(llm_responses=[json.dumps(raw)])

        result = agent._force_conclusion([])

        assert result["recommendation"] == "HOLD_OFF"

    def test_force_conclusion_empty_history(self):
        """Works with no history — uses a generic prompt."""
        force_ans = _minimal_stock_answer()
        agent, llm = _make_agent(
            llm_responses=["FINAL_ANSWER:\n" + json.dumps(force_ans)]
        )
        result = agent._force_conclusion(history=[])
        assert "recommendation" in result


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_no_tool_format_nudges_llm(self):
        """Response with neither TOOL_CALL nor FINAL_ANSWER gets a format reminder."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            "I think TCS looks good based on what I know.",   # no marker
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        result = agent.research_stock("TCS", "signal")
        assert result["recommendation"] == "PROCEED"
        assert llm.call.call_count == 2

    def test_web_search_used_for_web_search_tool(self):
        """web_search tool maps to web_search_tool.search()."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: web_search(TCS earnings 2025)",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        agent.research_stock("TCS", "signal")
        agent.web_search.search.assert_called_once_with("TCS earnings 2025")

    def test_get_sector_news_calls_google_rss(self):
        """get_sector_news(IT) calls news_fetcher.fetch_google_news_rss with IT query."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: get_sector_news(IT)",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        agent.research_stock("TCS", "signal")
        agent.news_fetcher.fetch_google_news_rss.assert_called_once()
        call_args = agent.news_fetcher.fetch_google_news_rss.call_args[0][0]
        assert "IT" in call_args

    def test_read_article_passes_url(self):
        """read_article(url) passes URL directly to web_search.read_article."""
        url = "https://example.com/tcs-results"
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            f"TOOL_CALL: read_article({url})",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        agent.research_stock("TCS", "signal")
        agent.web_search.read_article.assert_called_once_with(url)

    def test_search_calls_used_counter_correct(self):
        """search_calls_used in result matches actual search tool invocations."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: web_search(TCS news)",       # search 1
            "TOOL_CALL: get_stock_news(TCS)",         # search 2
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        result = agent.research_stock("TCS", "signal")
        assert result["search_calls_used"] == 2

    def test_articles_read_counter_correct(self):
        """articles_read in result matches actual read_article invocations."""
        answer = _minimal_stock_answer()
        agent, llm = _make_agent(llm_responses=[
            "TOOL_CALL: read_article(https://ex.com/1)",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])

        result = agent.research_stock("TCS", "signal")
        assert result["articles_read"] == 1

    def test_defaults_filled_when_llm_omits_keys(self):
        """LLM response missing optional keys still produces a complete result."""
        minimal = {"symbol": "TCS", "recommendation": "PROCEED"}
        agent, _ = _make_agent(llm_responses=["FINAL_ANSWER:\n" + json.dumps(minimal)])

        result = agent.research_stock("TCS", "signal")

        assert result["sentiment"] in ("POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED", "NEUTRAL")
        assert isinstance(result["risks"], list)
        assert isinstance(result["opportunities"], list)
        assert "suggested_adjustments" in result
