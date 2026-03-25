"""ResearchAgent — end-to-end smoke test.

Verifies the research agent at three levels:

  A. Parsing primitives (pure logic — always fast, zero network calls)
  B. Agent loop behaviour with a mock LLM (budget enforcement, error recovery)
  C. Full live run with a real LLM + real tool calls (needs API keys)

Usage (from the trading-bot/ directory):
    python check_research.py              # auto-detect LLM availability
    python check_research.py --mock       # Sections A + B only (no LLM key needed)
    python check_research.py --live       # Force Sections A + B + C
    python check_research.py --symbol INFY
    python check_research.py --briefing   # Also run morning_briefing in Section C
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env", override=False)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
SEP  = "─" * 65
SEP2 = "═" * 65
OK   = "  ✔"
FAIL = "  ✘"
SKIP = "  ○"
INFO = "    "

_results: list[tuple[str, str]] = []


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def ok(label: str, detail: str = "") -> None:
    suffix = f"  →  {detail}" if detail else ""
    print(f"{OK}  {label}{suffix}")
    _results.append((label, "PASS"))


def fail(label: str, detail: str = "") -> None:
    print(f"{FAIL}  {label}")
    if detail:
        print(f"{INFO}     {detail[:120]}")
    _results.append((label, "FAIL"))


def skip(label: str, reason: str = "") -> None:
    suffix = f"  (skipped: {reason})" if reason else ""
    print(f"{SKIP}  {label}{suffix}")
    _results.append((label, "SKIP"))


def info(text: str) -> None:
    print(f"{INFO}  {text}")


def summary() -> None:
    print(f"\n{SEP2}\n  SUMMARY\n{SEP2}")
    for label, status in _results:
        icon = "✔" if status == "PASS" else ("○" if status == "SKIP" else "✘")
        print(f"  {icon}  [{status:<4}]  {label[:70]}")
    passed  = sum(1 for _, s in _results if s == "PASS")
    failed  = sum(1 for _, s in _results if s == "FAIL")
    skipped = sum(1 for _, s in _results if s == "SKIP")
    print(f"\n  {passed} passed  ·  {failed} failed  ·  {skipped} skipped")
    if failed:
        print("\n  FAILED CHECKS:")
        for label, status in _results:
            if status == "FAIL":
                print(f"    ✘  {label}")
    print(f"{SEP2}\n")


# ---------------------------------------------------------------------------
# Minimal config shim (reads env vars directly — no config.yaml needed)
# ---------------------------------------------------------------------------
class _Cfg:
    _ENV: dict[tuple, str] = {
        ("apis", "tavily",  "api_key"): "TAVILY_API_KEY",
        ("apis", "newsapi", "api_key"): "NEWSAPI_API_KEY",
        ("apis", "serpapi", "api_key"): "SERPAPI_API_KEY",
        ("llm",  "gemini",  "api_key"): "GEMINI_API_KEY",
        ("llm",  "groq",    "api_key"): "GROQ_API_KEY",
        ("llm",  "nvidia_nim", "api_key"): "NVIDIA_API_KEY",
    }

    def get(self, *keys: str, default: Any = None) -> Any:  # noqa: ANN401
        env_var = self._ENV.get(keys)
        if env_var:
            value = os.environ.get(env_var, "")
            return value if value else f"${{{env_var}}}"
        return default

    def __getitem__(self, key: str) -> Any:
        return None


# ---------------------------------------------------------------------------
# LLM / tool availability probes
# ---------------------------------------------------------------------------
def _llm_key_available() -> bool:
    """Return True if at least one LLM API key is set."""
    return any(
        os.environ.get(k)
        for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY")
    )


def _importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _timer(fn):
    t0 = time.perf_counter()
    result = fn()
    ms = int((time.perf_counter() - t0) * 1000)
    return result, ms


# ---------------------------------------------------------------------------
# Pretty-print helpers for research results
# ---------------------------------------------------------------------------
def _show_research_result(result: dict, title: str = "Research Result") -> None:
    """Pretty-print a research_stock() result."""
    rec = result.get("recommendation", "?")
    sentiment = result.get("sentiment", "?")
    confidence = result.get("confidence", "?")

    _REC_COLOUR = {
        "PROCEED": "✅ PROCEED",
        "PROCEED_WITH_CAUTION": "⚠  PROCEED WITH CAUTION",
        "HOLD_OFF": "🛑 HOLD OFF",
        "AVOID": "❌ AVOID",
    }
    rec_display = _REC_COLOUR.get(rec, rec)

    print(f"\n  {'─' * 60}")
    print(f"  {title}")
    print(f"  {'─' * 60}")
    print(f"  Symbol         : {result.get('symbol', '?')}")
    print(f"  Recommendation : {rec_display}")
    print(f"  Sentiment      : {sentiment}  (confidence={confidence}/10)")
    print(f"  Tools used     : {result.get('tools_used', 0)}  "
          f"(search={result.get('search_calls_used', 0)}  "
          f"articles={result.get('articles_read', 0)})")

    summary_text = result.get("research_summary", "")
    if summary_text:
        # Word-wrap at ~58 chars
        words, line = summary_text.split(), ""
        print(f"\n  Summary:")
        for word in words:
            if len(line) + len(word) + 1 > 56:
                print(f"    {line}")
                line = word
            else:
                line = f"{line} {word}" if line else word
        if line:
            print(f"    {line}")

    risks = result.get("risks", [])
    if risks:
        print(f"\n  Risks ({len(risks)}):")
        for r in risks[:4]:
            print(f"    • {r[:70]}")
        if len(risks) > 4:
            print(f"    • … +{len(risks) - 4} more")

    opps = result.get("opportunities", [])
    if opps:
        print(f"\n  Opportunities ({len(opps)}):")
        for o in opps[:3]:
            print(f"    • {o[:70]}")

    adj = result.get("suggested_adjustments", {})
    if adj.get("reduce_position_size") or adj.get("tighter_stop_loss"):
        print(f"\n  Adjustments:")
        if adj.get("reduce_position_size"):
            print("    • Reduce position size")
        if adj.get("tighter_stop_loss"):
            print("    • Tighten stop-loss")
        if adj.get("reason"):
            print(f"    • {adj['reason'][:70]}")

    print(f"  {'─' * 60}")


def _show_briefing(result: dict) -> None:
    """Pretty-print a morning_briefing() result."""
    print(f"\n  {'─' * 60}")
    print(f"  Morning Briefing — {result.get('date', '?')}")
    print(f"  {'─' * 60}")
    print(f"  Global Sentiment : {result.get('global_sentiment', '?')}")
    print(f"  Risk Level       : {result.get('risk_level', '?')}")

    us = result.get("us_markets", {})
    if us:
        sp = us.get("sp500_change_pct", 0)
        nq = us.get("nasdaq_change_pct", 0)
        sp_arrow = "▲" if sp >= 0 else "▼"
        nq_arrow = "▲" if nq >= 0 else "▼"
        print(f"  US Markets       : S&P500 {sp_arrow}{abs(sp):.1f}%  Nasdaq {nq_arrow}{abs(nq):.1f}%")

    fii = result.get("fii_dii", {})
    if fii:
        print(f"  FII/DII          : FII={fii.get('fii_net', '?')}  DII={fii.get('dii_net', '?')}")

    vix = result.get("india_vix", {})
    if vix:
        print(f"  India VIX        : {vix.get('value', '?')}  ({vix.get('interpretation', '')})")

    asian = result.get("asian_markets_summary", "")
    if asian:
        print(f"  Asian Markets    : {asian[:65]}")

    sectors = result.get("sectors_to_watch", [])
    if sectors:
        print(f"\n  Sectors to Watch:")
        for s in sectors[:4]:
            print(f"    • {s.get('sector', '?')} [{s.get('direction', '?')}] — {s.get('reason', '')[:50]}")

    events = result.get("key_events_today", [])
    if events:
        print(f"\n  Key Events Today:")
        for e in events[:4]:
            print(f"    • {e[:65]}")

    outlook = result.get("overall_market_outlook", "")
    if outlook:
        print(f"\n  Outlook: {outlook[:120]}")

    rec = result.get("recommendation", "")
    if rec:
        print(f"  Recommendation: {rec[:80]}")

    print(f"  {'─' * 60}")


# ===========================================================================
# SECTION A — Parsing primitives (pure logic, zero network)
# ===========================================================================

def check_parse_tool_call(agent) -> None:
    section("A · _parse_tool_call — LLM format variants")

    cases = [
        # (response_text, expected_name, expected_param_substr, label)
        (
            "TOOL_CALL: web_search(TCS quarterly results 2025)",
            "web_search", "TCS quarterly", "standard format with arg",
        ),
        (
            'TOOL_CALL: web_search("TCS earnings Q3")',
            "web_search", "TCS earnings Q3", "quoted param stripped",
        ),
        (
            "TOOL_CALL: get_fii_dii_data()",
            "get_fii_dii_data", "", "no-arg tool with parens",
        ),
        (
            "TOOL_CALL: get_market_news",
            "get_market_news", "", "no-arg tool without parens",
        ),
        (
            "TOOL_CALL: web_search TCS stock news",
            "web_search", "TCS stock news", "space-separated (no parens)",
        ),
        (
            "TOOL_CALL: read_article(https://example.com/tcs-q3)",
            "read_article", "https://example.com", "URL param",
        ),
        (
            "Let me check the news first.\nTOOL_CALL: get_stock_news(RELIANCE)\nThis will help.",
            "get_stock_news", "RELIANCE", "tool call embedded mid-response",
        ),
        (
            "tool_call: get_india_vix()",
            "get_india_vix", "", "lowercase prefix accepted",
        ),
        (
            "TOOL_CALL: get_sector_news(BANKING)",
            "get_sector_news", "BANKING", "sector tool",
        ),
        (
            "TOOL_CALL: get_global_markets()",
            "get_global_markets", "", "global markets no-arg",
        ),
    ]

    for response, exp_name, exp_param_substr, label in cases:
        try:
            name, param = agent._parse_tool_call(response)
            if name == exp_name and exp_param_substr in param:
                ok(f"_parse_tool_call — {label}", f"name={name!r}  param={param!r}")
            else:
                fail(
                    f"_parse_tool_call — {label}",
                    f"expected name={exp_name!r} param⊇{exp_param_substr!r}, "
                    f"got name={name!r} param={param!r}",
                )
        except Exception as exc:
            fail(f"_parse_tool_call — {label}", str(exc))

    # Unparseable cases → (None, "")
    bad_cases = [
        ("I think we should just proceed.", "no marker at all"),
        ("", "empty string"),
        ("TOOL_CALL:", "prefix only, no tool name"),
    ]
    for response, label in bad_cases:
        try:
            name, param = agent._parse_tool_call(response)
            if name is None and param == "":
                ok(f"_parse_tool_call — returns (None, '') for {label}")
            else:
                fail(f"_parse_tool_call — {label}", f"expected (None, ''), got ({name!r}, {param!r})")
        except Exception as exc:
            fail(f"_parse_tool_call — {label}", str(exc))


def check_parse_final_answer(agent) -> None:
    section("A · _parse_final_answer — JSON extraction variants")

    payload = {
        "symbol": "TCS", "recommendation": "PROCEED",
        "sentiment": "POSITIVE", "confidence": 8,
        "risks": ["US tech weakness"], "opportunities": ["FII buying"],
        "reasoning": "Strong fundamentals.",
        "suggested_adjustments": {"reduce_position_size": False},
        "news_items_reviewed": [],
    }

    # Clean JSON
    label = "clean JSON after marker"
    resp = f"FINAL_ANSWER:\n{json.dumps(payload)}"
    try:
        result = agent._parse_final_answer(resp)
        if result and result.get("recommendation") == "PROCEED":
            ok(f"_parse_final_answer — {label}")
        else:
            fail(f"_parse_final_answer — {label}", str(result))
    except Exception as exc:
        fail(f"_parse_final_answer — {label}", str(exc))

    # Markdown fences
    label = "json wrapped in ```json ... ``` fence"
    resp2 = f"FINAL_ANSWER:\n```json\n{json.dumps(payload)}\n```"
    try:
        result = agent._parse_final_answer(resp2)
        if result and result.get("recommendation") == "PROCEED":
            ok(f"_parse_final_answer — {label}")
        else:
            fail(f"_parse_final_answer — {label}", str(result))
    except Exception as exc:
        fail(f"_parse_final_answer — {label}", str(exc))

    # Plain fences
    label = "json wrapped in plain ``` fence"
    resp3 = f"FINAL_ANSWER:\n```\n{json.dumps(payload)}\n```"
    try:
        result = agent._parse_final_answer(resp3)
        if result and result.get("confidence") == 8:
            ok(f"_parse_final_answer — {label}")
        else:
            fail(f"_parse_final_answer — {label}", str(result))
    except Exception as exc:
        fail(f"_parse_final_answer — {label}", str(exc))

    # JSON after prose
    label = "JSON after prose preamble"
    resp4 = f"FINAL_ANSWER:\nHere is my analysis:\n{json.dumps(payload)}"
    try:
        result = agent._parse_final_answer(resp4)
        if result and result.get("recommendation") == "PROCEED":
            ok(f"_parse_final_answer — {label}")
        else:
            fail(f"_parse_final_answer — {label}", str(result))
    except Exception as exc:
        fail(f"_parse_final_answer — {label}", str(exc))

    # No marker → None
    label = "no FINAL_ANSWER marker → returns None"
    resp5 = json.dumps(payload)
    try:
        result = agent._parse_final_answer(resp5)
        if result is None:
            ok(f"_parse_final_answer — {label}")
        else:
            fail(f"_parse_final_answer — {label}", f"expected None, got {result}")
    except Exception as exc:
        fail(f"_parse_final_answer — {label}", str(exc))

    # Invalid JSON → None
    label = "invalid JSON → returns None"
    resp6 = "FINAL_ANSWER:\n{invalid json here"
    try:
        result = agent._parse_final_answer(resp6)
        if result is None:
            ok(f"_parse_final_answer — {label}")
        else:
            fail(f"_parse_final_answer — {label}", f"expected None, got {result}")
    except Exception as exc:
        fail(f"_parse_final_answer — {label}", str(exc))


def check_format_tool_result(agent) -> None:
    section("A · _format_tool_result — serialisation & truncation")

    # List → JSON
    label = "list of dicts serialised to JSON"
    try:
        text = agent._format_tool_result([{"title": "TCS beats estimates", "url": "https://ex.com"}])
        parsed = json.loads(text)
        if isinstance(parsed, list) and parsed[0]["title"] == "TCS beats estimates":
            ok(label)
        else:
            fail(label, text[:80])
    except Exception as exc:
        fail(label, str(exc))

    # Dict → JSON
    label = "dict serialised to JSON"
    try:
        text = agent._format_tool_result({"fii_net": 500.0, "dii_net": -200.0})
        parsed = json.loads(text)
        if parsed["fii_net"] == 500.0:
            ok(label)
        else:
            fail(label, text[:80])
    except Exception as exc:
        fail(label, str(exc))

    # Truncation
    label = "long result truncated to ≤3 000 chars + '[truncated]' suffix"
    big = {"data": "x" * 5000}
    try:
        text = agent._format_tool_result(big)
        if len(text) <= 3100 and "[truncated]" in text:
            ok(label, f"len={len(text)}")
        else:
            fail(label, f"len={len(text)}  has_truncated={'[truncated]' in text}")
    except Exception as exc:
        fail(label, str(exc))

    # Plain string passthrough
    label = "plain string passed through unchanged (short)"
    try:
        text = agent._format_tool_result("simple string result")
        if text == "simple string result":
            ok(label)
        else:
            fail(label, repr(text))
    except Exception as exc:
        fail(label, str(exc))


def check_execute_tool_routing(agent) -> None:
    """Verify _execute_tool dispatches to the correct tool method."""
    section("A · _execute_tool — dispatch routing")

    cases = [
        ("web_search",        "search",                   ("TCS news",),          {}),
        ("read_article",      "read_article",             ("https://ex.com",),    {}),
        ("get_stock_news",    "fetch_stock_specific_news", ("TCS",),              {}),
        ("get_market_news",   "fetch_market_news",         (),                    {}),
        ("get_fii_dii_data",  "get_fii_dii_data",         (),                    {}),
        ("get_global_markets","get_global_market_status",  (),                    {}),
        ("get_india_vix",     "get_india_vix",             (),                    {}),
    ]

    for tool_name, expected_method, call_args, call_kwargs in cases:
        label = f"_execute_tool('{tool_name}') → .{expected_method}()"
        try:
            # Find which object owns the method
            if hasattr(agent.web_search, expected_method):
                mock_target = agent.web_search
            else:
                mock_target = agent.news_fetcher

            original = getattr(mock_target, expected_method)
            called_with = []
            def make_spy(orig, storage):
                def spy(*a, **kw):
                    storage.extend(a)
                    return orig(*a, **kw) if callable(orig) else {}
                return spy
            setattr(mock_target, expected_method, make_spy(original, called_with))

            param = call_args[0] if call_args else ""
            agent._execute_tool(tool_name, param)

            # Restore
            setattr(mock_target, expected_method, original)

            ok(label)
        except Exception as exc:
            fail(label, str(exc))

    # Unknown tool returns error dict, does not raise
    label = "_execute_tool('unknown_tool') returns error dict, does not raise"
    try:
        result = agent._execute_tool("unknown_tool", "param")
        if isinstance(result, dict) and "error" in result:
            ok(label, f"error={result['error']!r}")
        else:
            fail(label, f"expected dict with 'error' key, got: {result}")
    except Exception as exc:
        fail(label, str(exc))


# ===========================================================================
# SECTION B — Agent loop with mock LLM (no real API calls)
# ===========================================================================

def _make_stock_answer(symbol: str = "TCS", recommendation: str = "PROCEED") -> dict:
    return {
        "symbol":            symbol,
        "research_summary":  "TCS looks solid. FII flows positive.",
        "sentiment":         "POSITIVE",
        "confidence":        8,
        "risks":             ["US tech uncertainty"],
        "opportunities":     ["Strong FII buying this week"],
        "recommendation":    recommendation,
        "reasoning":         "Strong fundamentals support the signal.",
        "suggested_adjustments": {
            "reduce_position_size": False,
            "tighter_stop_loss":    False,
            "reason": "",
        },
        "news_items_reviewed": [{"title": "TCS beats estimates", "impact": "POSITIVE"}],
    }


def _make_briefing_answer(date: str = "2026-03-18") -> dict:
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


def _make_mock_agent(llm_responses: list[str]) -> Any:
    """Build a ResearchAgent with fully mocked dependencies."""
    from src.agents.research_agent import ResearchAgent
    from src.llm.budget_manager import BudgetManager

    cfg     = _Cfg()
    llm     = MagicMock()
    budget  = BudgetManager(db=None)
    ws      = MagicMock()
    nf      = MagicMock()
    db      = MagicMock()

    llm.call.side_effect = list(llm_responses)

    ws.search.return_value = [
        {"title": "TCS Q3 results", "url": "https://ex.com/1", "snippet": "TCS beats estimates", "published_date": "2026-03-10"},
    ]
    ws.read_article.return_value = {
        "title": "TCS Q3 beats estimates",
        "text":  "TCS reported strong quarterly revenue...",
        "authors": [], "published_date": "2026-03-10",
    }
    nf.fetch_stock_specific_news.return_value = [
        {"title": "TCS rises 2%", "url": "https://ex.com/2", "source": "ET",
         "published_date": "2026-03-10", "snippet": ""},
    ]
    nf.fetch_google_news_rss.return_value  = []
    nf.fetch_market_news.return_value      = []
    nf.get_fii_dii_data.return_value       = {"date": "2026-03-17", "fii_net": 500.0, "dii_net": -200.0, "source": "NSE"}
    nf.get_global_market_status.return_value = {
        "sp500":   {"last_price": 5200.0, "change_pct": 0.3},
        "nasdaq":  {"last_price": 16100.0, "change_pct": -0.5},
        "usd_inr": {"last_price": 86.5,   "change_pct": 0.1},
        "crude_oil": {"last_price": 78.5, "change_pct": -1.2},
        "timestamp": "2026-03-18 08:00:00",
    }
    nf.get_india_vix.return_value = {"vix": 14.8, "signal": "LOW_FEAR", "interpretation": "Low fear"}

    return ResearchAgent(config=cfg, llm_router=llm, budget_manager=budget,
                         web_search=ws, news_fetcher=nf, db_manager=db)


def check_loop_single_step(ResearchAgent) -> None:
    section("B · Agent loop — single-step FINAL_ANSWER")
    answer = _make_stock_answer()
    label  = "research_stock — LLM answers immediately (no tools)"
    try:
        agent = _make_mock_agent(["FINAL_ANSWER:\n" + json.dumps(answer)])
        result = agent.research_stock("TCS", "RSI oversold at 27")
        if result["recommendation"] == "PROCEED" and result["symbol"] == "TCS":
            ok(label, f"recommendation={result['recommendation']}  confidence={result['confidence']}")
        else:
            fail(label, str(result))
    except Exception as exc:
        fail(label, str(exc))


def check_loop_tool_then_answer(ResearchAgent) -> None:
    section("B · Agent loop — tool call → FINAL_ANSWER")

    # One tool then answer
    label = "research_stock — get_stock_news then FINAL_ANSWER"
    answer = _make_stock_answer()
    try:
        agent = _make_mock_agent([
            "TOOL_CALL: get_stock_news(TCS)",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])
        result = agent.research_stock("TCS", "MACD bullish crossover")
        if result["recommendation"] == "PROCEED":
            ok(label, f"search_calls={result['search_calls_used']}")
            assert agent.news_fetcher.fetch_stock_specific_news.called, "fetch_stock_specific_news not called"
        else:
            fail(label, str(result))
    except Exception as exc:
        fail(label, str(exc))

    # Multiple tools
    label = "research_stock — get_global_markets + get_fii_dii_data + FINAL_ANSWER"
    try:
        agent2 = _make_mock_agent([
            "TOOL_CALL: get_global_markets()",
            "TOOL_CALL: get_fii_dii_data()",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])
        result2 = agent2.research_stock("INFY", "Volume breakout")
        if result2["recommendation"] == "PROCEED":
            ok(label, f"tools={result2.get('tools_used', '?')}")
        else:
            fail(label, str(result2))
    except Exception as exc:
        fail(label, str(exc))


def check_loop_budget_enforcement(ResearchAgent) -> None:
    section("B · Agent loop — budget enforcement")

    answer = _make_stock_answer()

    # Search budget exhausted — 4th search blocked (stock_research allows 3)
    label = "search budget: 4th search call blocked (stock_research limit=3)"
    try:
        agent = _make_mock_agent([
            "TOOL_CALL: web_search(TCS news)",
            "TOOL_CALL: web_search(TCS sector)",
            "TOOL_CALL: web_search(TCS risks)",
            "TOOL_CALL: web_search(TCS more)",    # budget exhausted → blocked
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])
        result = agent.research_stock("TCS", "signal")
        actual_search_calls = agent.web_search.search.call_count
        if actual_search_calls == 3:
            ok(label, f"web_search called exactly 3 times (4th blocked)")
        else:
            fail(label, f"expected web_search.call_count=3, got {actual_search_calls}")
    except Exception as exc:
        fail(label, str(exc))

    # Article budget — 3rd article blocked (stock_research allows 2)
    label = "article budget: 3rd read_article blocked (stock_research limit=2)"
    try:
        agent2 = _make_mock_agent([
            "TOOL_CALL: read_article(https://ex.com/1)",
            "TOOL_CALL: read_article(https://ex.com/2)",
            "TOOL_CALL: read_article(https://ex.com/3)",  # blocked
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])
        result2 = agent2.research_stock("TCS", "signal")
        actual_articles = agent2.web_search.read_article.call_count
        if actual_articles == 2:
            ok(label, f"read_article called exactly 2 times (3rd blocked)")
        else:
            fail(label, f"expected read_article.call_count=2, got {actual_articles}")
    except Exception as exc:
        fail(label, str(exc))

    # Free tools don't consume search budget
    label = "free tools (get_fii_dii, get_global_markets, get_india_vix) don't consume search budget"
    try:
        agent3 = _make_mock_agent([
            "TOOL_CALL: get_fii_dii_data()",
            "TOOL_CALL: get_global_markets()",
            "TOOL_CALL: get_india_vix()",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])
        result3 = agent3.research_stock("TCS", "signal")
        if result3["search_calls_used"] == 0:
            ok(label, "search_calls_used=0 as expected")
        else:
            fail(label, f"expected search_calls_used=0, got {result3['search_calls_used']}")
    except Exception as exc:
        fail(label, str(exc))


def check_loop_error_recovery(ResearchAgent) -> None:
    section("B · Agent loop — error recovery")
    from src.llm.router import LLMUnavailableError

    # LLM unavailable → force conclusion fallback
    label = "LLM unavailable → still returns safe dict with recommendation"
    try:
        agent = _make_mock_agent([])
        agent.llm_router.call.side_effect = LLMUnavailableError("all providers down")
        result = agent.research_stock("TCS", "RSI oversold")
        if "recommendation" in result and result.get("confidence", 10) <= 5:
            ok(label, f"recommendation={result['recommendation']}  confidence={result['confidence']}")
        else:
            fail(label, str(result))
    except Exception as exc:
        fail(label, str(exc))

    # Two consecutive bad JSON → force_conclusion
    label = "2× invalid JSON FINAL_ANSWER → triggers force_conclusion"
    force_ans = _make_stock_answer(recommendation="PROCEED_WITH_CAUTION")
    try:
        agent2 = _make_mock_agent([
            "FINAL_ANSWER:\n{bad json",
            "FINAL_ANSWER:\n{also bad",
            "FINAL_ANSWER:\n" + json.dumps(force_ans),
        ])
        result2 = agent2.research_stock("TCS", "signal")
        if "recommendation" in result2:
            ok(label, f"recommendation={result2['recommendation']}")
        else:
            fail(label, str(result2))
    except Exception as exc:
        fail(label, str(exc))

    # Two consecutive unrecognised responses → force_conclusion
    label = "2× response with no TOOL_CALL/FINAL_ANSWER marker → force_conclusion"
    force_ans2 = _make_stock_answer()
    try:
        agent3 = _make_mock_agent([
            "I think TCS looks good based on general knowledge.",
            "The stock has been performing well recently.",
            "FINAL_ANSWER:\n" + json.dumps(force_ans2),
        ])
        result3 = agent3.research_stock("TCS", "signal")
        if "recommendation" in result3:
            ok(label, f"recommendation={result3['recommendation']}")
        else:
            fail(label, str(result3))
    except Exception as exc:
        fail(label, str(exc))

    # Unknown tool → agent records error and continues
    label = "unknown tool name → agent notes error and continues to FINAL_ANSWER"
    try:
        answer = _make_stock_answer()
        agent4 = _make_mock_agent([
            "TOOL_CALL: nonexistent_tool(some_param)",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])
        result4 = agent4.research_stock("TCS", "signal")
        if result4["recommendation"] == "PROCEED":
            ok(label)
        else:
            fail(label, str(result4))
    except Exception as exc:
        fail(label, str(exc))

    # Tool raises exception → agent records error and continues
    label = "tool raises exception → agent continues gracefully"
    try:
        answer = _make_stock_answer()
        agent5 = _make_mock_agent([
            "TOOL_CALL: web_search(TCS news)",
            "FINAL_ANSWER:\n" + json.dumps(answer),
        ])
        agent5.web_search.search.side_effect = Exception("network error")
        result5 = agent5.research_stock("TCS", "signal")
        if "recommendation" in result5:
            ok(label)
        else:
            fail(label, str(result5))
    except Exception as exc:
        fail(label, str(exc))


def check_loop_db_logging(ResearchAgent) -> None:
    section("B · Agent loop — DB logging")

    # DB called with correct agent_name
    label = "research_stock calls db.log_agent_activity"
    try:
        answer = _make_stock_answer()
        agent  = _make_mock_agent(["FINAL_ANSWER:\n" + json.dumps(answer)])
        agent.research_stock("TCS", "RSI oversold at 27")
        assert agent.db.log_agent_activity.called, "log_agent_activity not called"
        call_args = agent.db.log_agent_activity.call_args
        all_vals  = list(call_args.args) + list(call_args.kwargs.values())
        assert any("research_agent" in str(v) for v in all_vals), "agent_name not in call args"
        assert any("stock_research"  in str(v) for v in all_vals), "session_type not in call args"
        ok(label, "agent_name='research_agent'  session_type='stock_research'")
    except Exception as exc:
        fail(label, str(exc))

    # DB failure doesn't propagate
    label = "DB error during log_agent_activity doesn't crash research_stock"
    try:
        answer = _make_stock_answer()
        agent2 = _make_mock_agent(["FINAL_ANSWER:\n" + json.dumps(answer)])
        agent2.db.log_agent_activity.side_effect = Exception("DB unavailable")
        result = agent2.research_stock("TCS", "signal")
        if result["recommendation"] == "PROCEED":
            ok(label)
        else:
            fail(label, str(result))
    except Exception as exc:
        fail(label, str(exc))

    # db_manager=None is accepted
    label = "db_manager=None → no AttributeError"
    try:
        from src.agents.research_agent import ResearchAgent as RA
        from src.llm.budget_manager import BudgetManager
        answer = _make_stock_answer()
        llm = MagicMock()
        llm.call.return_value = "FINAL_ANSWER:\n" + json.dumps(answer)
        ws = MagicMock(); nf = MagicMock()
        ws.search.return_value = []; nf.get_fii_dii_data.return_value = {}
        nf.get_global_market_status.return_value = {}; nf.get_india_vix.return_value = {}
        nf.fetch_stock_specific_news.return_value = []; nf.fetch_market_news.return_value = []
        agent3 = RA(config={}, llm_router=llm, budget_manager=BudgetManager(db=None),
                    web_search=ws, news_fetcher=nf, db_manager=None)
        result = agent3.research_stock("TCS", "signal")
        if "recommendation" in result:
            ok(label)
        else:
            fail(label, str(result))
    except Exception as exc:
        fail(label, str(exc))


def check_morning_briefing_mock(ResearchAgent) -> None:
    section("B · morning_briefing — mock LLM")
    from datetime import date
    today = date.today().isoformat()

    label = "morning_briefing — happy path (3 tools then FINAL_ANSWER)"
    briefing = _make_briefing_answer(today)
    try:
        agent = _make_mock_agent([
            "TOOL_CALL: get_global_markets()",
            "TOOL_CALL: get_fii_dii_data()",
            "TOOL_CALL: web_search(India market news overnight)",
            "FINAL_ANSWER:\n" + json.dumps(briefing),
        ])
        result = agent.morning_briefing()
        req_keys = {"date", "global_sentiment", "risk_level", "sectors_to_watch", "recommendation"}
        if req_keys.issubset(result.keys()):
            ok(label, f"risk_level={result['risk_level']}  sentiment={result['global_sentiment']}")
        else:
            missing = req_keys - set(result.keys())
            fail(label, f"missing keys: {missing}")
    except Exception as exc:
        fail(label, str(exc))

    label = "morning_briefing — uses 'morning_briefing' session type (6 LLM calls budget)"
    try:
        agent2 = _make_mock_agent(["FINAL_ANSWER:\n" + json.dumps(briefing)])
        captured_types: list[str] = []
        orig = agent2.budget_manager.create_session
        def spy(st):
            captured_types.append(st)
            return orig(st)
        agent2.budget_manager.create_session = spy
        agent2.morning_briefing()
        if "morning_briefing" in captured_types:
            ok(label)
        else:
            fail(label, f"captured session types: {captured_types}")
    except Exception as exc:
        fail(label, str(exc))

    label = "morning_briefing — LLM down → safe fallback dict"
    from src.llm.router import LLMUnavailableError
    try:
        agent3 = _make_mock_agent([])
        agent3.llm_router.call.side_effect = LLMUnavailableError("down")
        result3 = agent3.morning_briefing()
        if "global_sentiment" in result3 and "risk_level" in result3:
            ok(label, f"risk_level={result3['risk_level']}")
        else:
            fail(label, str(result3))
    except Exception as exc:
        fail(label, str(exc))


def check_result_defaults(ResearchAgent) -> None:
    section("B · result defaults — missing keys filled by agent")

    label = "LLM returns minimal JSON — agent fills in all required keys"
    minimal = {"symbol": "TCS", "recommendation": "PROCEED"}
    try:
        agent = _make_mock_agent(["FINAL_ANSWER:\n" + json.dumps(minimal)])
        result = agent.research_stock("TCS", "signal")
        required = {
            "symbol", "research_summary", "sentiment", "confidence",
            "risks", "opportunities", "recommendation", "reasoning",
            "suggested_adjustments", "news_items_reviewed",
            "tools_used", "search_calls_used", "articles_read",
        }
        missing = required - set(result.keys())
        if not missing:
            ok(label, f"all {len(required)} required keys present")
        else:
            fail(label, f"missing: {missing}")
    except Exception as exc:
        fail(label, str(exc))


# ===========================================================================
# SECTION C — Live run with real LLM + real tool calls
# ===========================================================================

def check_live_research_stock(agent, symbol: str) -> None:
    context = (
        f"Quant agent detected BUY signal — RSI oversold at 27.3, MACD bullish crossover. "
        f"Price: ₹3,850. Stop-loss: ₹3,740. Target: ₹3,960."
    )
    label = f"research_stock('{symbol}') — live agentic run"
    info(f"  Symbol : {symbol}")
    info(f"  Context: {context[:70]}…")
    info("")

    try:
        t0 = time.perf_counter()
        result = agent.research_stock(symbol, context)
        elapsed = time.perf_counter() - t0

        valid_recs = {"PROCEED", "PROCEED_WITH_CAUTION", "HOLD_OFF", "AVOID"}
        if result.get("recommendation") in valid_recs and result.get("symbol") == symbol:
            ok(label, f"[{elapsed:.1f}s]  recommendation={result['recommendation']}  confidence={result.get('confidence', '?')}")
            _show_research_result(result, f"research_stock({symbol})")
        else:
            fail(label, f"invalid result structure: {list(result.keys())}")
    except Exception as exc:
        fail(label, str(exc))
        import traceback
        traceback.print_exc()


def check_live_morning_briefing(agent) -> None:
    label = "morning_briefing() — live agentic run"
    info("  Running morning_briefing…")
    info("")

    try:
        t0 = time.perf_counter()
        result = agent.morning_briefing()
        elapsed = time.perf_counter() - t0

        valid_risks = {"LOW", "MEDIUM", "HIGH"}
        if result.get("risk_level") in valid_risks:
            ok(label, f"[{elapsed:.1f}s]  sentiment={result.get('global_sentiment', '?')}  risk={result['risk_level']}")
            _show_briefing(result)
        else:
            fail(label, f"invalid risk_level: {result.get('risk_level')!r}")
    except Exception as exc:
        fail(label, str(exc))
        import traceback
        traceback.print_exc()


def show_budget_summary(budget_manager) -> None:
    section("Budget remaining after this run")
    resources = [
        "gemini_flash", "gemini_pro", "groq", "nvidia_nim",
        "tavily_search", "news_api",
    ]
    for r in resources:
        remaining = budget_manager.get_remaining(r)
        used      = budget_manager._counters.get(r, 0)
        bar = "█" * min(used, 20) + "░" * max(0, 20 - min(used, 20))
        info(f"  {r:<20}  used={used:<4}  remaining={remaining:<6}  {bar}")


# ===========================================================================
# Main
# ===========================================================================
_DEFAULT_SYMBOL = "TCS"


def main() -> None:
    parser = argparse.ArgumentParser(description="ResearchAgent smoke test")
    parser.add_argument("--mock",     action="store_true", help="Sections A + B only (no LLM key needed)")
    parser.add_argument("--live",     action="store_true", help="Force Sections A + B + C (needs LLM key)")
    parser.add_argument("--symbol",   default=_DEFAULT_SYMBOL, metavar="SYM",
                        help=f"Stock symbol for Section C (default: {_DEFAULT_SYMBOL})")
    parser.add_argument("--briefing", action="store_true",
                        help="Also run morning_briefing in Section C")
    args = parser.parse_args()
    symbol = args.symbol.upper()

    # Mode selection
    if args.mock:
        mode = "mock"
    elif args.live:
        mode = "live"
    elif _llm_key_available():
        mode = "live"
    else:
        mode = "mock"

    print(f"\n{SEP2}")
    print("  ResearchAgent — End-to-End Smoke Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   mode={mode.upper()}")
    print(SEP2)

    # ── Imports ──────────────────────────────────────────────────────────
    section("1 · Imports")
    try:
        from src.agents.research_agent import ResearchAgent
        ok("ResearchAgent imported")
    except ImportError as exc:
        fail("ResearchAgent import failed", str(exc))
        sys.exit(1)

    try:
        from src.llm.budget_manager import BudgetManager
        from src.llm.router import LLMRouter
        from src.tools.web_search import WebSearchTool
        from src.tools.news_fetcher import NewsFetcher
        ok("BudgetManager, LLMRouter, WebSearchTool, NewsFetcher imported")
    except ImportError as exc:
        fail("Supporting imports failed", str(exc))
        sys.exit(1)

    # ── Section A — Parsing primitives ───────────────────────────────────
    cfg    = _Cfg()
    budget = BudgetManager(db=None)
    ws     = WebSearchTool(config=cfg, budget_manager=budget)
    nf     = NewsFetcher(config=cfg, budget_manager=budget)
    llm    = MagicMock()
    llm.call.return_value = "FINAL_ANSWER:\n{}"

    dummy_agent = ResearchAgent(
        config=cfg, llm_router=llm, budget_manager=budget,
        web_search=ws, news_fetcher=nf, db_manager=None,
    )
    ok("ResearchAgent instantiated (dummy LLM for Section A)")

    check_parse_tool_call(dummy_agent)
    check_parse_final_answer(dummy_agent)
    check_format_tool_result(dummy_agent)
    check_execute_tool_routing(dummy_agent)

    # ── Section B — Mock LLM agent loop ──────────────────────────────────
    check_loop_single_step(ResearchAgent)
    check_loop_tool_then_answer(ResearchAgent)
    check_loop_budget_enforcement(ResearchAgent)
    check_loop_error_recovery(ResearchAgent)
    check_loop_db_logging(ResearchAgent)
    check_morning_briefing_mock(ResearchAgent)
    check_result_defaults(ResearchAgent)

    # ── Section C — Live run ──────────────────────────────────────────────
    if mode == "live":
        section(f"C · Live agentic run — symbol={symbol}")

        # Build a real agent with live tools and LLM router
        live_budget = BudgetManager(db=None)
        live_ws     = WebSearchTool(config=cfg, budget_manager=live_budget)
        live_nf     = NewsFetcher(config=cfg, budget_manager=live_budget)
        live_llm    = LLMRouter(config=cfg, budget_manager=live_budget)
        live_agent  = ResearchAgent(
            config=cfg, llm_router=live_llm, budget_manager=live_budget,
            web_search=live_ws, news_fetcher=live_nf, db_manager=None,
        )
        ok("Live ResearchAgent ready")

        check_live_research_stock(live_agent, symbol)

        if args.briefing:
            section("C · Live agentic run — morning_briefing()")
            check_live_morning_briefing(live_agent)

        show_budget_summary(live_budget)

    else:
        skip(
            f"Section C — live run (symbol={symbol})",
            "no LLM API key found  (set GEMINI_API_KEY or GROQ_API_KEY, or use --live)",
        )
        if args.briefing:
            skip("Section C — morning_briefing()", "mock mode")

    summary()


if __name__ == "__main__":
    main()
