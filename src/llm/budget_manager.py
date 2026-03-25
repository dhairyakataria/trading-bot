"""Budget Manager — tracks daily LLM/API request usage and enforces free-tier limits.

Persists counters to the database ``system_state`` table so usage survives
process restarts within the same calendar day.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from src.utils.logger import get_logger

IST = ZoneInfo("Asia/Kolkata")
_log = get_logger("llm.budget_manager")


# --------------------------------------------------------------------------- #
# Limits                                                                       #
# --------------------------------------------------------------------------- #

DAILY_LIMITS: dict[str, int] = {
    "gemini_flash":  1400,    # Gemini 2.0 Flash: 1500/day, keep 100 buffer
    "gemini_pro":      20,    # Gemini 2.5 Pro: 25/day, keep 5 buffer
    "groq":           800,    # Groq: 30 RPM but token-limited
    "nvidia_nim":     150,    # NVIDIA NIM: credit-based, conserve
    "ollama_local": 999999,   # Local: effectively unlimited
    "tavily_search":   30,    # Tavily: ~1000/month ÷ 22 trading days, with buffer
    "news_api":        80,    # NewsAPI: 100/day
    "serp_api":         4,    # SerpAPI: ~100/month ÷ 22 days
    "web_scrape":     200,    # Self-imposed polite limit
}

SESSION_LIMITS: dict[str, dict[str, int]] = {
    "morning_briefing":    {"max_llm_calls": 6, "max_search_calls": 5, "max_article_reads": 3},
    "stock_research":      {"max_llm_calls": 4, "max_search_calls": 3, "max_article_reads": 2},
    "trade_decision":      {"max_llm_calls": 3, "max_search_calls": 1, "max_article_reads": 1},
    "exit_analysis":       {"max_llm_calls": 3, "max_search_calls": 2, "max_article_reads": 1},
    "daily_summary":       {"max_llm_calls": 2, "max_search_calls": 0, "max_article_reads": 0},
    "conflict_resolution": {"max_llm_calls": 2, "max_search_calls": 0, "max_article_reads": 0},
}

_STATE_KEY = "llm_budget_daily"


# --------------------------------------------------------------------------- #
# SessionBudget                                                                #
# --------------------------------------------------------------------------- #

class SessionBudgetExhaustedError(Exception):
    """Raised when a session-level budget limit is exceeded."""


class SessionBudget:
    """Tracks usage within a single agent execution session.

    Created by :meth:`BudgetManager.create_session`; not persisted to the DB
    (sessions are short-lived within a single agent call).
    """

    def __init__(self, session_type: str, limits: dict[str, int]) -> None:
        self.session_type = session_type
        self.remaining_llm_calls: int     = limits["max_llm_calls"]
        self.remaining_search_calls: int  = limits["max_search_calls"]
        self.remaining_article_reads: int = limits["max_article_reads"]

    # -- usage ----------------------------------------------------------------

    def use_llm(self) -> None:
        """Consume one LLM call. Raises :exc:`SessionBudgetExhaustedError` if depleted."""
        if self.remaining_llm_calls <= 0:
            raise SessionBudgetExhaustedError(
                f"Session '{self.session_type}' has no LLM calls remaining."
            )
        self.remaining_llm_calls -= 1
        _log.debug(
            "Session '%s' — LLM call used; %d remaining",
            self.session_type, self.remaining_llm_calls,
        )

    def use_search(self) -> None:
        """Consume one search call."""
        if self.remaining_search_calls <= 0:
            raise SessionBudgetExhaustedError(
                f"Session '{self.session_type}' has no search calls remaining."
            )
        self.remaining_search_calls -= 1
        _log.debug(
            "Session '%s' — search call used; %d remaining",
            self.session_type, self.remaining_search_calls,
        )

    def use_article(self) -> None:
        """Consume one article read."""
        if self.remaining_article_reads <= 0:
            raise SessionBudgetExhaustedError(
                f"Session '{self.session_type}' has no article reads remaining."
            )
        self.remaining_article_reads -= 1
        _log.debug(
            "Session '%s' — article read used; %d remaining",
            self.session_type, self.remaining_article_reads,
        )

    def is_budget_exhausted(self) -> bool:
        """Return True when every category has been depleted."""
        return (
            self.remaining_llm_calls <= 0
            and self.remaining_search_calls <= 0
            and self.remaining_article_reads <= 0
        )


# --------------------------------------------------------------------------- #
# BudgetExceededError                                                          #
# --------------------------------------------------------------------------- #

class BudgetExceededError(Exception):
    """Raised when :meth:`BudgetManager.use` is called on an exhausted resource."""


# --------------------------------------------------------------------------- #
# BudgetManager                                                                #
# --------------------------------------------------------------------------- #

class BudgetManager:
    """Tracks and enforces daily call limits per LLM/API provider.

    Usage counters are persisted to the ``system_state`` table so that they
    survive process restarts within the same calendar day.  On startup the
    manager loads today's counters from the DB; at midnight the caller should
    invoke :meth:`reset_daily`.

    Args:
        db: A :class:`~src.database.db_manager.DatabaseManager` instance used
            to persist state.  Pass ``None`` to run in-memory only (useful in
            tests).
    """

    def __init__(self, db: Optional[Any] = None) -> None:
        self._db = db
        # counters: resource → calls_used_today
        self._counters: dict[str, int] = {k: 0 for k in DAILY_LIMITS}
        self._today: str = self._ist_today()
        self.load_state()

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _ist_today() -> str:
        return datetime.now(IST).strftime("%Y-%m-%d")

    def _check_rollover(self) -> None:
        """Reset counters automatically if the calendar date has changed."""
        today = self._ist_today()
        if today != self._today:
            _log.info("Date rolled over from %s to %s — resetting daily counters.", self._today, today)
            self.reset_daily()

    # -- public API -----------------------------------------------------------

    def can_use(self, resource: str) -> bool:
        """Return True if *resource* has remaining budget for today."""
        self._check_rollover()
        limit = DAILY_LIMITS.get(resource)
        if limit is None:
            _log.warning("Unknown resource '%s' — treating as unlimited.", resource)
            return True
        used = self._counters.get(resource, 0)
        return used < limit

    def use(self, resource: str) -> None:
        """Increment the usage counter for *resource*.

        Raises:
            BudgetExceededError: if the daily limit for *resource* is already
                reached before this call.
        """
        self._check_rollover()
        if not self.can_use(resource):
            raise BudgetExceededError(
                f"Daily budget exhausted for '{resource}' "
                f"(limit={DAILY_LIMITS.get(resource)})."
            )
        self._counters[resource] = self._counters.get(resource, 0) + 1
        _log.debug(
            "Budget — used '%s': %d/%d",
            resource,
            self._counters[resource],
            DAILY_LIMITS.get(resource, 0),
        )

    def get_remaining(self, resource: str) -> int:
        """Return the number of calls remaining today for *resource*."""
        self._check_rollover()
        limit = DAILY_LIMITS.get(resource, 0)
        used  = self._counters.get(resource, 0)
        return max(0, limit - used)

    def get_all_remaining(self) -> dict[str, int]:
        """Return a dict of ``resource → remaining_calls`` for all providers."""
        self._check_rollover()
        return {resource: self.get_remaining(resource) for resource in DAILY_LIMITS}

    def reset_daily(self) -> None:
        """Reset all counters to zero (call at midnight or start of day)."""
        self._today = self._ist_today()
        self._counters = {k: 0 for k in DAILY_LIMITS}
        _log.info("Daily budget counters reset for %s.", self._today)
        self.save_state()

    # -- persistence ----------------------------------------------------------

    def save_state(self) -> None:
        """Persist current counters to the ``system_state`` table."""
        if self._db is None:
            return
        payload = json.dumps({"date": self._today, "counters": self._counters})
        try:
            self._db.set_system_state(_STATE_KEY, payload)
            _log.debug("Budget state saved to DB.")
        except Exception as exc:
            _log.warning("Could not save budget state to DB: %s", exc)

    def load_state(self) -> None:
        """Restore today's counters from the ``system_state`` table.

        If the stored date differs from today, counters are reset to zero
        (a new trading day has started since the last run).
        """
        if self._db is None:
            return
        try:
            raw = self._db.get_system_state(_STATE_KEY)
        except Exception as exc:
            _log.warning("Could not load budget state from DB: %s", exc)
            return

        if not raw:
            return

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            _log.warning("Corrupt budget state in DB — ignoring.")
            return

        stored_date = data.get("date", "")
        today = self._ist_today()

        if stored_date != today:
            _log.info(
                "Stored budget date (%s) ≠ today (%s) — starting fresh.",
                stored_date, today,
            )
            return  # counters remain at 0

        stored_counters: dict[str, int] = data.get("counters", {})
        for resource, used in stored_counters.items():
            if resource in DAILY_LIMITS:
                self._counters[resource] = int(used)

        _log.info("Budget state loaded from DB for %s: %s", today, self._counters)

    # -- session budgets ------------------------------------------------------

    def create_session(self, session_type: str) -> SessionBudget:
        """Return a fresh :class:`SessionBudget` for *session_type*.

        Args:
            session_type: One of the keys in :data:`SESSION_LIMITS`.

        Raises:
            KeyError: if *session_type* is not recognised.
        """
        limits = SESSION_LIMITS.get(session_type)
        if limits is None:
            raise KeyError(
                f"Unknown session type '{session_type}'. "
                f"Valid types: {list(SESSION_LIMITS)}"
            )
        return SessionBudget(session_type, limits)

    # -- reporting ------------------------------------------------------------

    def get_usage_summary(self) -> dict[str, Any]:
        """Return today's usage stats for all providers."""
        self._check_rollover()
        summary: dict[str, Any] = {"date": self._today, "providers": {}}
        for resource, limit in DAILY_LIMITS.items():
            used = self._counters.get(resource, 0)
            summary["providers"][resource] = {
                "used":      used,
                "limit":     limit,
                "remaining": max(0, limit - used),
                "pct_used":  round(used / limit * 100, 1) if limit else 0.0,
            }
        return summary
