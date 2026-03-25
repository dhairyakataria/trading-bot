"""Research Agent — autonomous LLM-driven research for Indian equity markets.

Uses a simple ReAct-style agentic loop:
    LLM reasons → calls a tool → observes result → repeats
until it has enough information to produce a final structured JSON answer,
or until the session budget / step limit is reached.

Tool-call protocol (plain text, provider-agnostic):
    TOOL_CALL: tool_name(parameter)

Final answer protocol:
    FINAL_ANSWER:
    {json}
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from src.llm.budget_manager import BudgetManager, SessionBudget, SessionBudgetExhaustedError
from src.llm.router import LLMRouter, LLMUnavailableError, TaskComplexity
from src.tools.news_fetcher import NewsFetcher
from src.tools.web_search import WebSearchTool
from src.utils.logger import get_logger

IST = ZoneInfo("Asia/Kolkata")
_log = get_logger("agents.research_agent")

# Maximum chars of a single tool result inserted into the prompt
_TOOL_RESULT_TRUNCATE = 3000


# --------------------------------------------------------------------------- #
# Tool descriptions block (shared by both session prompts)                    #
# --------------------------------------------------------------------------- #

_TOOL_DESCRIPTIONS = """
AVAILABLE TOOLS — use exactly one per response step:

1.  web_search(query)         — Search the web. Returns titles, snippets, URLs.
2.  read_article(url)         — Read full article text (≤3 000 chars) from a URL.
3.  get_stock_news(symbol)    — Recent news for a specific NSE stock (e.g. TCS).
4.  get_sector_news(sector)   — Indian sector news (e.g. IT, BANKING, PHARMA).
5.  get_market_news()         — General Indian stock-market news (MC, ET, Google).
6.  get_fii_dii_data()        — FII & DII trading activity (institutional flows).
7.  get_global_markets()      — US/Asia indices, crude oil, gold, USD/INR.
8.  get_india_vix()           — India VIX (market fear / volatility index).

FORMAT FOR TOOL CALLS (no markdown, no code fences):
TOOL_CALL: tool_name(parameter)

Examples:
TOOL_CALL: web_search(TCS quarterly results 2025)
TOOL_CALL: get_stock_news(TCS)
TOOL_CALL: get_fii_dii_data()
TOOL_CALL: read_article(https://example.com/article)

FORMAT FOR FINAL ANSWER:
FINAL_ANSWER:
{valid JSON — no markdown fences}
""".strip()


# --------------------------------------------------------------------------- #
# ResearchAgent                                                                #
# --------------------------------------------------------------------------- #

class ResearchAgent:
    """Autonomous LLM-driven research agent for Indian equity markets.

    Implements a ReAct-style loop: the LLM reasons, invokes tools, observes
    results, and iterates until it produces a structured ``FINAL_ANSWER`` JSON
    or the session budget / step limit is exhausted.

    Args:
        config:          Project-wide config dict (passed through for extensibility).
        llm_router:      :class:`~src.llm.router.LLMRouter` that routes calls to
                         the best available free provider.
        budget_manager:  :class:`~src.llm.budget_manager.BudgetManager` that
                         tracks daily and session-level API quotas.
        web_search:      :class:`~src.tools.web_search.WebSearchTool` instance.
        news_fetcher:    :class:`~src.tools.news_fetcher.NewsFetcher` instance.
        db_manager:      Database manager used to persist agent-activity logs.
                         Pass ``None`` to disable persistence (useful in tests).
    """

    def __init__(
        self,
        config: Any,
        llm_router: LLMRouter,
        budget_manager: BudgetManager,
        web_search: WebSearchTool,
        news_fetcher: NewsFetcher,
        db_manager: Any,
    ) -> None:
        self.config         = config
        self.llm_router     = llm_router
        self.budget_manager = budget_manager
        self.web_search     = web_search
        self.news_fetcher   = news_fetcher
        self.db             = db_manager

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def research_stock(
        self,
        symbol: str,
        context: str,
        session_type: str = "stock_research",
    ) -> dict[str, Any]:
        """Research a stock that has a technical signal and assess trade safety.

        The agent autonomously decides which tools to call, collects evidence,
        and returns a structured assessment with a clear recommendation.

        Args:
            symbol:       NSE ticker, e.g. ``"TCS"``.
            context:      Quant-agent signal description, e.g.
                          ``"RSI oversold at 27.3, MACD bullish crossover…"``.
            session_type: Session key in :data:`~src.llm.budget_manager.SESSION_LIMITS`.

        Returns:
            Dict with keys: symbol, research_summary, sentiment, confidence,
            risks, opportunities, recommendation, reasoning,
            suggested_adjustments, news_items_reviewed,
            tools_used, search_calls_used, articles_read.
        """
        t0 = time.monotonic()
        session_budget = self.budget_manager.create_session(session_type)
        today = datetime.now(IST).strftime("%Y-%m-%d")

        system_prompt = f"""You are a senior financial research analyst specializing in Indian equity markets (NSE/BSE).
You are given a stock with a technical trading signal. Your job is to verify if it is safe to trade by researching news, market conditions, and any potential risks.

IMPORTANT RULES:
1. Be EFFICIENT with tool calls. Start broad, then go specific only if needed.
2. If your first search gives enough information, STOP. Do not over-research.
3. You have a LIMITED budget of tool calls. Use them wisely.
4. Focus on RISKS and RED FLAGS — do not just confirm positive bias.
5. Check for: upcoming earnings, regulatory issues, management problems, sector headwinds.
6. Consider global factors: if US tech fell, Indian IT stocks may follow.
7. Always check FII/DII data — institutional money flow matters.

{_TOOL_DESCRIPTIONS}

Your FINAL_ANSWER must be this exact JSON (no extra keys, no markdown):
{{
  "symbol": "{symbol}",
  "research_summary": "2-3 sentence summary of findings",
  "sentiment": "POSITIVE",
  "confidence": 7,
  "risks": ["risk 1", "risk 2"],
  "opportunities": ["opportunity 1"],
  "recommendation": "PROCEED",
  "reasoning": "explanation",
  "suggested_adjustments": {{
    "reduce_position_size": false,
    "tighter_stop_loss": false,
    "reason": "explanation"
  }},
  "news_items_reviewed": [
    {{"title": "article title", "impact": "POSITIVE"}}
  ]
}}
recommendation must be one of: PROCEED, PROCEED_WITH_CAUTION, HOLD_OFF, AVOID
sentiment must be one of: POSITIVE, NEGATIVE, NEUTRAL, MIXED"""

        user_prompt = (
            f"Research this trading signal and decide whether to proceed:\n\n"
            f"Stock: {symbol}\n"
            f"Signal Context: {context}\n"
            f"Today's date: {today}\n\n"
            f"Start by getting stock news or checking global markets, then make your FINAL_ANSWER."
        )

        result, llm_calls, search_calls, articles_read = self._run_agent_loop(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            session_budget=session_budget,
        )

        # Ensure required keys are present with sensible defaults
        result.setdefault("symbol", symbol)
        result.setdefault("research_summary", "Research completed with available data.")
        result.setdefault("sentiment", "NEUTRAL")
        result.setdefault("confidence", 5)
        result.setdefault("risks", [])
        result.setdefault("opportunities", [])
        result.setdefault("recommendation", "PROCEED_WITH_CAUTION")
        result.setdefault("reasoning", "Insufficient data — recommend manual review before trading.")
        result.setdefault("suggested_adjustments", {
            "reduce_position_size": False,
            "tighter_stop_loss": False,
            "reason": "",
        })
        result.setdefault("news_items_reviewed", [])
        result["tools_used"]        = result.get("tools_used", llm_calls + search_calls + articles_read)
        result["search_calls_used"] = search_calls
        result["articles_read"]     = articles_read

        duration = time.monotonic() - t0
        _log.info(
            "research_stock(%s) — recommendation=%s confidence=%s "
            "llm_calls=%d search_calls=%d duration=%.1fs",
            symbol, result["recommendation"], result["confidence"],
            llm_calls, search_calls, duration,
        )

        self._persist_log(
            session_type=session_type,
            input_data={"symbol": symbol, "context": context},
            output_data=result,
            llm_calls_count=llm_calls,
            search_calls_count=search_calls,
            duration_seconds=duration,
        )

        return result

    def morning_briefing(self) -> dict[str, Any]:
        """Run the daily pre-market briefing analysis (call at ~08:00 IST).

        Autonomously checks global markets, FII/DII flows, overnight news,
        India VIX, and identifies sectors likely to move today.

        Returns:
            Structured briefing dict with keys: date, global_sentiment,
            us_markets, asian_markets_summary, fii_dii, india_vix,
            crude_oil, usd_inr, key_events_today, sectors_to_watch,
            overall_market_outlook, risk_level, recommendation.
        """
        t0 = time.monotonic()
        session_budget = self.budget_manager.create_session("morning_briefing")
        today = datetime.now(IST).strftime("%Y-%m-%d")

        system_prompt = f"""You are a senior market analyst preparing the daily pre-market briefing for Indian equity markets.
Today's date is {today}.

DO THIS IN ORDER:
1. Check global market status (US close, Asian markets).
2. Get FII/DII data from yesterday.
3. Search for any major overnight news affecting Indian markets.
4. Get India VIX for the market fear gauge.
5. Identify sectors that might move today and WHY.

Be EFFICIENT. If global markets are flat and no major news, keep it short.
If something significant happened, dig deeper.

{_TOOL_DESCRIPTIONS}

Your FINAL_ANSWER must be this exact JSON (no markdown):
{{
  "date": "{today}",
  "global_sentiment": "NEUTRAL",
  "us_markets": {{"sp500_change_pct": 0.3, "nasdaq_change_pct": -0.5}},
  "asian_markets_summary": "Mixed — Nikkei up 0.4%, Hang Seng down 0.8%",
  "fii_dii": {{"fii_net": -500, "dii_net": 800, "interpretation": "FII selling but DII buying"}},
  "india_vix": {{"value": 14.8, "interpretation": "Low fear — complacent market"}},
  "crude_oil": {{"price": 78.5, "change_pct": -1.2}},
  "usd_inr": {{"rate": 86.5, "change_pct": 0.1}},
  "key_events_today": ["event 1", "event 2"],
  "sectors_to_watch": [
    {{"sector": "IT", "direction": "CAUTIOUS", "reason": "Nasdaq weakness overnight"}}
  ],
  "overall_market_outlook": "one paragraph summary",
  "risk_level": "MEDIUM",
  "recommendation": "Normal trading day. Avoid over-leveraging in IT sector."
}}
risk_level must be one of: LOW, MEDIUM, HIGH
global_sentiment must be one of: POSITIVE, NEGATIVE, NEUTRAL, MIXED"""

        user_prompt = (
            f"Prepare the pre-market briefing for Indian equity markets for {today}.\n\n"
            "Start with get_global_markets(), then get_fii_dii_data(), then search for key overnight news.\n"
            "Provide FINAL_ANSWER when done."
        )

        result, llm_calls, search_calls, articles_read = self._run_agent_loop(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            session_budget=session_budget,
        )

        # Defaults for any missing keys
        result.setdefault("date", today)
        result.setdefault("global_sentiment", "NEUTRAL")
        result.setdefault("us_markets", {})
        result.setdefault("asian_markets_summary", "Data unavailable.")
        result.setdefault("fii_dii", {})
        result.setdefault("india_vix", {})
        result.setdefault("crude_oil", {})
        result.setdefault("usd_inr", {})
        result.setdefault("key_events_today", [])
        result.setdefault("sectors_to_watch", [])
        result.setdefault("overall_market_outlook", "Briefing generated with limited data.")
        result.setdefault("risk_level", "MEDIUM")
        result.setdefault("recommendation", "Normal trading day. Monitor conditions carefully.")
        result["tools_used"]        = result.get("tools_used", llm_calls + search_calls + articles_read)
        result["search_calls_used"] = search_calls
        result["articles_read"]     = articles_read

        duration = time.monotonic() - t0
        _log.info(
            "morning_briefing() — sentiment=%s risk=%s llm_calls=%d duration=%.1fs",
            result["global_sentiment"], result["risk_level"], llm_calls, duration,
        )

        self._persist_log(
            session_type="morning_briefing",
            input_data={"date": today},
            output_data=result,
            llm_calls_count=llm_calls,
            search_calls_count=search_calls,
            duration_seconds=duration,
        )

        return result

    # ------------------------------------------------------------------ #
    # Core agentic loop                                                    #
    # ------------------------------------------------------------------ #

    def _run_agent_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        session_budget: SessionBudget,
    ) -> tuple[dict[str, Any], int, int, int]:
        """ReAct-style loop: reason → tool → observe → repeat → final answer.

        Returns:
            (result_dict, llm_calls_used, search_calls_used, articles_read)
        """
        history: list[dict[str, str]] = []
        max_steps        = max(1, session_budget.remaining_llm_calls)
        llm_calls        = 0
        search_calls     = 0
        articles_read    = 0
        tool_calls_total = 0
        consecutive_bad  = 0  # consecutive steps with no parseable output

        for step in range(max_steps):

            # ---- LLM call ------------------------------------------------
            if session_budget.remaining_llm_calls <= 0:
                _log.warning("LLM session budget exhausted at step %d — ending loop.", step)
                break

            full_prompt = self._build_prompt(user_prompt, history)
            try:
                response = self.llm_router.call(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    complexity=TaskComplexity.MODERATE,
                )
                session_budget.use_llm()
                llm_calls += 1
            except SessionBudgetExhaustedError:
                _log.warning("LLM session budget exhausted mid-loop at step %d.", step)
                break
            except LLMUnavailableError as exc:
                _log.error("All LLM providers unavailable at step %d: %s", step, exc)
                break
            except Exception as exc:
                _log.error("Unexpected LLM error at step %d: %s", step, exc)
                break

            _log.debug("Step %d response (%.300s)", step, response)

            # ---- Parse -------------------------------------------------------
            if "FINAL_ANSWER:" in response:
                answer = self._parse_final_answer(response)
                if answer is not None:
                    _log.info("Agent reached FINAL_ANSWER at step %d.", step + 1)
                    return answer, llm_calls, search_calls, articles_read

                # Bad JSON — give the LLM one correction nudge
                consecutive_bad += 1
                _log.warning("FINAL_ANSWER present but JSON invalid (attempt #%d).", consecutive_bad)
                if consecutive_bad >= 2:
                    _log.warning("Two consecutive bad outputs — forcing conclusion.")
                    break
                history.append({
                    "role": "system_note",
                    "content": (
                        "Your FINAL_ANSWER contained invalid JSON. "
                        "Respond ONLY with FINAL_ANSWER: followed by a valid JSON object. "
                        "No markdown, no code fences, no extra text."
                    ),
                })
                continue

            if "TOOL_CALL:" in response:
                consecutive_bad = 0
                tool_name, param = self._parse_tool_call(response)

                if tool_name is None:
                    _log.warning("Could not parse TOOL_CALL at step %d: %.200s", step, response)
                    history.append({
                        "role": "system_note",
                        "content": (
                            "Could not parse your TOOL_CALL. "
                            "Use the exact format:  TOOL_CALL: tool_name(parameter)"
                        ),
                    })
                    continue

                # ---- Budget checks for search / article tools ----------------
                is_search  = tool_name in ("web_search", "get_stock_news", "get_sector_news", "get_market_news")
                is_article = tool_name == "read_article"

                if is_search:
                    if session_budget.remaining_search_calls <= 0:
                        _log.warning("Search budget exhausted — skipping '%s'.", tool_name)
                        history.append({
                            "role": "system_note",
                            "content": (
                                "Search budget exhausted. "
                                "Provide your FINAL_ANSWER now based on the information you have."
                            ),
                        })
                        continue
                    try:
                        session_budget.use_search()
                        search_calls += 1
                    except SessionBudgetExhaustedError:
                        history.append({"role": "system_note", "content": "Search budget exhausted."})
                        continue

                if is_article:
                    if session_budget.remaining_article_reads <= 0:
                        _log.warning("Article budget exhausted — skipping read_article.")
                        history.append({
                            "role": "system_note",
                            "content": (
                                "Article reading budget exhausted. "
                                "Provide your FINAL_ANSWER now based on the information you have."
                            ),
                        })
                        continue
                    try:
                        session_budget.use_article()
                        articles_read += 1
                    except SessionBudgetExhaustedError:
                        history.append({"role": "system_note", "content": "Article reading budget exhausted."})
                        continue

                # ---- Execute tool --------------------------------------------
                tool_calls_total += 1
                _log.debug("Calling tool: %s(%r)", tool_name, param)
                tool_result = self._execute_tool(tool_name, param)

                history.append({"role": "tool_call",   "tool": tool_name, "params": param})
                history.append({"role": "tool_result", "content": self._format_tool_result(tool_result)})
                continue

            # ---- Neither marker found ----------------------------------------
            consecutive_bad += 1
            _log.warning("Response has no recognised marker (attempt #%d): %.200s", consecutive_bad, response)
            if consecutive_bad >= 2:
                _log.warning("Two consecutive unrecognised responses — forcing conclusion.")
                break
            history.append({
                "role": "system_note",
                "content": (
                    "Please respond with EITHER:\n"
                    "  TOOL_CALL: tool_name(parameter)\n"
                    "OR:\n"
                    "  FINAL_ANSWER:\n"
                    "  {json}"
                ),
            })

        # ---- Max steps reached or error — force a conclusion ----------------
        _log.warning("Agent loop ended without FINAL_ANSWER — forcing conclusion.")
        result = self._force_conclusion(history, system_prompt)
        return result, llm_calls, search_calls, articles_read

    # ------------------------------------------------------------------ #
    # Tool execution                                                       #
    # ------------------------------------------------------------------ #

    def _execute_tool(self, tool_name: str, param: str) -> Any:
        """Dispatch *tool_name* with *param* and return the result.

        Never raises — returns an error dict if the tool call fails.
        """
        try:
            if tool_name == "web_search":
                return self.web_search.search(param)

            if tool_name == "read_article":
                return self.web_search.read_article(param)

            if tool_name == "get_stock_news":
                return self.news_fetcher.fetch_stock_specific_news(param)

            if tool_name == "get_sector_news":
                query = f"India {param} sector stocks news"
                return self.news_fetcher.fetch_google_news_rss(query)

            if tool_name == "get_market_news":
                return self.news_fetcher.fetch_market_news()

            if tool_name == "get_fii_dii_data":
                return self.news_fetcher.get_fii_dii_data()

            if tool_name == "get_global_markets":
                return self.news_fetcher.get_global_market_status()

            if tool_name == "get_india_vix":
                return self.news_fetcher.get_india_vix()

            return {"error": f"Unknown tool '{tool_name}'"}

        except Exception as exc:
            _log.error("Tool '%s' raised: %s", tool_name, exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # Parsing helpers                                                      #
    # ------------------------------------------------------------------ #

    def _parse_tool_call(self, response: str) -> tuple[Optional[str], str]:
        """Extract ``(tool_name, parameter)`` from a ``TOOL_CALL:`` line.

        Handles multiple LLM formatting styles:
        - ``TOOL_CALL: tool_name(param)``
        - ``TOOL_CALL: tool_name("param")``
        - ``TOOL_CALL: tool_name param``  (no parens)
        - ``TOOL_CALL: tool_name``        (no-param tools)

        Returns ``(None, "")`` if parsing fails.
        """
        for line in response.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if not upper.startswith("TOOL_CALL:"):
                continue

            call_str = stripped[len("TOOL_CALL:"):].strip()

            # Pattern 1: tool_name(params) — capture up to first closing paren
            m = re.match(r"^(\w+)\s*\(([^)]*)\)", call_str)
            if m:
                name  = m.group(1)
                param = m.group(2).strip().strip("\"'")
                return name, param

            # Pattern 2: tool_name "param" or tool_name param (no parens)
            m2 = re.match(r"^(\w+)\s+(.+)$", call_str)
            if m2:
                name  = m2.group(1)
                param = m2.group(2).strip().strip("\"'")
                return name, param

            # Pattern 3: bare tool_name only (no-arg tools like get_market_news)
            m3 = re.match(r"^(\w+)\s*$", call_str)
            if m3:
                return m3.group(1), ""

        return None, ""

    def _parse_final_answer(self, response: str) -> Optional[dict[str, Any]]:
        """Extract and JSON-parse the block after ``FINAL_ANSWER:``.

        Returns ``None`` if no valid JSON is found.
        """
        idx = response.find("FINAL_ANSWER:")
        if idx == -1:
            return None

        after = response[idx + len("FINAL_ANSWER:"):].strip()

        # Strip markdown code fences if the LLM wrapped its JSON
        after = re.sub(r"^```(?:json)?\s*", "", after, flags=re.MULTILINE)
        after = re.sub(r"\s*```\s*$",        "", after, flags=re.MULTILINE)
        after = after.strip()

        # Try direct parse first
        try:
            return json.loads(after)
        except json.JSONDecodeError:
            pass

        # Fall back to extracting the first {...} block
        brace = re.search(r"\{.*\}", after, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group())
            except json.JSONDecodeError:
                pass

        return None

    def _build_prompt(
        self,
        user_prompt: str,
        history: list[dict[str, str]],
    ) -> str:
        """Reconstruct the full conversation prompt from *user_prompt* + *history*."""
        if not history:
            return user_prompt

        parts = [user_prompt, "\n\n--- Research history so far ---"]

        for entry in history:
            role = entry.get("role", "")
            if role == "tool_call":
                parts.append(f"\n[Tool called]: {entry.get('tool', '')}({entry.get('params', '')})")
            elif role == "tool_result":
                parts.append(f"\n[Tool result]:\n{entry.get('content', '')}")
            elif role == "system_note":
                parts.append(f"\n[Note]: {entry.get('content', '')}")

        parts.append(
            "\n\n---\n"
            "Continue your research. Call another tool with TOOL_CALL: if needed, "
            "or provide your FINAL_ANSWER: now."
        )
        return "".join(parts)

    def _format_tool_result(self, result: Any) -> str:
        """Serialise *result* to a prompt-friendly string, truncated to avoid
        blowing up the context window."""
        if isinstance(result, (list, dict)):
            text = json.dumps(result, indent=2, default=str)
        else:
            text = str(result)

        if len(text) > _TOOL_RESULT_TRUNCATE:
            text = text[:_TOOL_RESULT_TRUNCATE] + "\n... [truncated]"
        return text

    # ------------------------------------------------------------------ #
    # Force conclusion                                                     #
    # ------------------------------------------------------------------ #

    def _force_conclusion(
        self,
        history: list[dict[str, str]],
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """Ask the LLM to produce a FINAL_ANSWER from whatever is in *history*.

        Called when max_steps is exhausted.  Returns a minimal safe fallback
        dict if the LLM also fails or produces unparseable output.
        """
        # Gather tool results accumulated so far (most recent first)
        tool_results = [
            e["content"]
            for e in reversed(history)
            if e.get("role") == "tool_result"
        ]
        gathered = "\n\n".join(tool_results[:4])  # keep prompt compact

        force_prompt = (
            "You have reached the research step limit. "
            "Based on the information gathered below, provide your FINAL_ANSWER now.\n\n"
            f"Information gathered:\n{gathered[:2000] if gathered else 'No tool results available.'}\n\n"
            "Respond ONLY with FINAL_ANSWER: followed by valid JSON. No other text."
        )

        try:
            response = self.llm_router.call(
                prompt=force_prompt,
                system_prompt=system_prompt,
                complexity=TaskComplexity.MODERATE,
            )
            _log.debug("Force-conclusion response: %.300s", response)

            if "FINAL_ANSWER:" in response:
                answer = self._parse_final_answer(response)
                if answer is not None:
                    return answer

            # Try raw JSON extraction even without the marker
            brace = re.search(r"\{.*\}", response, re.DOTALL)
            if brace:
                try:
                    return json.loads(brace.group())
                except json.JSONDecodeError:
                    pass

        except Exception as exc:
            _log.error("Force-conclusion LLM call failed: %s", exc)

        # Absolute last-resort fallback — a minimal safe response
        _log.warning("Force conclusion failed — returning minimal fallback response.")
        return {
            "research_summary": (
                "Research reached step limit with inconclusive results. "
                "Manual verification recommended before trading."
            ),
            "sentiment":   "NEUTRAL",
            "confidence":  3,
            "risks":       ["Insufficient research data — manual review recommended"],
            "opportunities": [],
            "recommendation": "PROCEED_WITH_CAUTION",
            "reasoning": (
                "Agent reached its step limit. Recommend manual verification "
                "and reduced position size before trading."
            ),
            "suggested_adjustments": {
                "reduce_position_size": True,
                "tighter_stop_loss":    True,
                "reason": "Limited research — use smaller position with tighter risk controls",
            },
            "news_items_reviewed": [],
        }

    # ------------------------------------------------------------------ #
    # Database persistence                                                 #
    # ------------------------------------------------------------------ #

    def _persist_log(
        self,
        session_type: str,
        input_data: Any,
        output_data: Any,
        llm_calls_count: int,
        search_calls_count: int,
        duration_seconds: float,
    ) -> None:
        """Persist agent-session record to ``agent_logs``. Silently skips if
        ``db_manager`` is ``None`` or an error occurs."""
        if self.db is None:
            return
        try:
            self.db.log_agent_activity(
                agent_name="research_agent",
                session_type=session_type,
                input_data=input_data,
                output_data=output_data,
                llm_calls_count=llm_calls_count,
                search_calls_count=search_calls_count,
                duration_seconds=duration_seconds,
            )
        except Exception as exc:
            _log.warning("Failed to persist agent log to DB: %s", exc)
