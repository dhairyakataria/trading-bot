"""Tests for src/llm/budget_manager.py and src/llm/router.py."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from src.llm.budget_manager import (
    DAILY_LIMITS,
    SESSION_LIMITS,
    BudgetExceededError,
    BudgetManager,
    SessionBudget,
    SessionBudgetExhaustedError,
)
from src.llm.router import LLMRouter, LLMUnavailableError, TaskComplexity


# =========================================================================== #
# Fixtures                                                                     #
# =========================================================================== #

@pytest.fixture()
def budget() -> BudgetManager:
    """BudgetManager with no DB (in-memory)."""
    return BudgetManager(db=None)


@pytest.fixture()
def mock_db() -> MagicMock:
    db = MagicMock()
    db.get_system_state.return_value = None
    return db


@pytest.fixture()
def budget_with_db(mock_db: MagicMock) -> BudgetManager:
    return BudgetManager(db=mock_db)


@pytest.fixture()
def router(budget: BudgetManager) -> LLMRouter:
    cfg = MagicMock()
    cfg.get.return_value = ""
    return LLMRouter(config=cfg, budget_manager=budget)


# =========================================================================== #
# BudgetManager — basic counting                                               #
# =========================================================================== #

class TestBudgetManagerCounting:
    def test_initial_remaining_equals_limit(self, budget):
        for resource, limit in DAILY_LIMITS.items():
            assert budget.get_remaining(resource) == limit

    def test_can_use_initially_true(self, budget):
        for resource in DAILY_LIMITS:
            assert budget.can_use(resource) is True

    def test_use_decrements_remaining(self, budget):
        budget.use("groq")
        assert budget.get_remaining("groq") == DAILY_LIMITS["groq"] - 1

    def test_use_multiple_times(self, budget):
        for _ in range(5):
            budget.use("news_api")
        assert budget.get_remaining("news_api") == DAILY_LIMITS["news_api"] - 5

    def test_remaining_never_negative(self, budget):
        # Drain gemini_pro (limit=20)
        for _ in range(DAILY_LIMITS["gemini_pro"]):
            budget.use("gemini_pro")
        assert budget.get_remaining("gemini_pro") == 0

    def test_get_all_remaining_returns_all_keys(self, budget):
        result = budget.get_all_remaining()
        assert set(result.keys()) == set(DAILY_LIMITS.keys())

    def test_get_all_remaining_values_correct(self, budget):
        budget.use("groq")
        budget.use("groq")
        result = budget.get_all_remaining()
        assert result["groq"] == DAILY_LIMITS["groq"] - 2


class TestBudgetManagerLimits:
    def test_can_use_false_when_exhausted(self, budget):
        limit = DAILY_LIMITS["serp_api"]  # 4 — fast to exhaust
        for _ in range(limit):
            budget.use("serp_api")
        assert budget.can_use("serp_api") is False

    def test_use_raises_when_exhausted(self, budget):
        limit = DAILY_LIMITS["serp_api"]
        for _ in range(limit):
            budget.use("serp_api")
        with pytest.raises(BudgetExceededError):
            budget.use("serp_api")

    def test_unknown_resource_treated_as_unlimited(self, budget):
        assert budget.can_use("nonexistent_resource") is True

    def test_ollama_has_effectively_unlimited_budget(self, budget):
        assert budget.can_use("ollama_local") is True
        assert budget.get_remaining("ollama_local") == 999999


class TestBudgetManagerReset:
    def test_reset_daily_clears_counters(self, budget):
        budget.use("groq")
        budget.use("gemini_flash")
        budget.reset_daily()
        assert budget.get_remaining("groq")        == DAILY_LIMITS["groq"]
        assert budget.get_remaining("gemini_flash") == DAILY_LIMITS["gemini_flash"]

    def test_can_use_true_after_reset(self, budget):
        limit = DAILY_LIMITS["serp_api"]
        for _ in range(limit):
            budget.use("serp_api")
        budget.reset_daily()
        assert budget.can_use("serp_api") is True


# =========================================================================== #
# BudgetManager — persistence                                                  #
# =========================================================================== #

class TestBudgetManagerPersistence:
    def test_save_state_calls_db(self, budget_with_db, mock_db):
        budget_with_db.use("groq")
        budget_with_db.save_state()
        mock_db.set_system_state.assert_called()
        key, value = mock_db.set_system_state.call_args[0]
        assert key == "llm_budget_daily"
        data = json.loads(value)
        assert data["counters"]["groq"] == 1

    def test_load_state_restores_counters(self, mock_db):
        today_str = BudgetManager._ist_today()
        payload = json.dumps({"date": today_str, "counters": {"groq": 42, "gemini_flash": 10}})
        mock_db.get_system_state.return_value = payload

        bm = BudgetManager(db=mock_db)
        assert bm.get_remaining("groq")        == DAILY_LIMITS["groq"] - 42
        assert bm.get_remaining("gemini_flash") == DAILY_LIMITS["gemini_flash"] - 10

    def test_load_state_old_date_resets_counters(self, mock_db):
        payload = json.dumps({"date": "2000-01-01", "counters": {"groq": 999}})
        mock_db.get_system_state.return_value = payload

        bm = BudgetManager(db=mock_db)
        assert bm.get_remaining("groq") == DAILY_LIMITS["groq"]

    def test_load_state_corrupt_json_is_ignored(self, mock_db):
        mock_db.get_system_state.return_value = "NOT_VALID_JSON"
        bm = BudgetManager(db=mock_db)
        assert bm.get_remaining("groq") == DAILY_LIMITS["groq"]

    def test_save_state_no_db_is_silent(self, budget):
        # Should not raise even without a DB
        budget.use("groq")
        budget.save_state()  # no-op


# =========================================================================== #
# SessionBudget                                                                #
# =========================================================================== #

class TestSessionBudget:
    def test_create_session_valid_type(self, budget):
        session = budget.create_session("morning_briefing")
        assert isinstance(session, SessionBudget)
        assert session.remaining_llm_calls    == SESSION_LIMITS["morning_briefing"]["max_llm_calls"]
        assert session.remaining_search_calls  == SESSION_LIMITS["morning_briefing"]["max_search_calls"]
        assert session.remaining_article_reads == SESSION_LIMITS["morning_briefing"]["max_article_reads"]

    def test_create_session_invalid_type(self, budget):
        with pytest.raises(KeyError):
            budget.create_session("unknown_session_type")

    def test_use_llm_decrements(self, budget):
        session = budget.create_session("trade_decision")
        initial = session.remaining_llm_calls
        session.use_llm()
        assert session.remaining_llm_calls == initial - 1

    def test_use_search_decrements(self, budget):
        session = budget.create_session("trade_decision")
        initial = session.remaining_search_calls
        session.use_search()
        assert session.remaining_search_calls == initial - 1

    def test_use_article_decrements(self, budget):
        session = budget.create_session("trade_decision")
        initial = session.remaining_article_reads
        session.use_article()
        assert session.remaining_article_reads == initial - 1

    def test_use_llm_raises_when_exhausted(self, budget):
        session = budget.create_session("daily_summary")  # max_llm_calls=2
        session.use_llm()
        session.use_llm()
        with pytest.raises(SessionBudgetExhaustedError):
            session.use_llm()

    def test_use_search_raises_when_zero(self, budget):
        session = budget.create_session("daily_summary")  # max_search_calls=0
        with pytest.raises(SessionBudgetExhaustedError):
            session.use_search()

    def test_is_budget_exhausted_true_when_all_zero(self, budget):
        session = budget.create_session("conflict_resolution")  # llm=2, search=0, article=0
        assert session.is_budget_exhausted() is False
        session.use_llm()
        session.use_llm()
        assert session.is_budget_exhausted() is True

    def test_is_budget_exhausted_false_if_any_remaining(self, budget):
        session = budget.create_session("trade_decision")  # llm=3, search=1, article=1
        session.use_llm()
        assert session.is_budget_exhausted() is False

    @pytest.mark.parametrize("session_type", list(SESSION_LIMITS.keys()))
    def test_all_session_types_creatable(self, budget, session_type):
        session = budget.create_session(session_type)
        assert session.session_type == session_type


# =========================================================================== #
# LLMRouter — routing logic                                                    #
# =========================================================================== #

class TestRoutingLogic:
    """Test that the correct provider chain is tried per complexity / priority."""

    def _make_router_with_mock_providers(self, budget):
        cfg = MagicMock()
        cfg.get.return_value = ""
        router = LLMRouter(config=cfg, budget_manager=budget)
        return router

    def test_simple_uses_groq_first(self, budget):
        router = self._make_router_with_mock_providers(budget)
        with patch.object(router, "_call_groq", return_value="answer") as mock_groq:
            result = router.call("hello", complexity=TaskComplexity.SIMPLE)
        assert result == "answer"
        mock_groq.assert_called_once()

    def test_moderate_uses_gemini_flash_first(self, budget):
        router = self._make_router_with_mock_providers(budget)
        with patch.object(router, "_call_gemini_flash", return_value="answer") as mock_gf:
            result = router.call("hello", complexity=TaskComplexity.MODERATE)
        assert result == "answer"
        mock_gf.assert_called_once()

    def test_complex_high_uses_gemini_pro_first(self, budget):
        router = self._make_router_with_mock_providers(budget)
        with patch.object(router, "_call_gemini_pro", return_value="answer") as mock_gp:
            result = router.call("hello", complexity=TaskComplexity.COMPLEX, priority="high")
        assert result == "answer"
        mock_gp.assert_called_once()

    def test_complex_normal_uses_gemini_flash_first(self, budget):
        router = self._make_router_with_mock_providers(budget)
        with patch.object(router, "_call_gemini_flash", return_value="answer") as mock_gf:
            result = router.call("hello", complexity=TaskComplexity.COMPLEX, priority="normal")
        assert result == "answer"
        mock_gf.assert_called_once()

    def test_bulk_uses_ollama_first(self, budget):
        router = self._make_router_with_mock_providers(budget)
        with patch.object(router, "_call_ollama", return_value="answer") as mock_ollama:
            result = router.call("hello", complexity=TaskComplexity.BULK)
        assert result == "answer"
        mock_ollama.assert_called_once()


# =========================================================================== #
# LLMRouter — fallback chain                                                   #
# =========================================================================== #

class TestFallbackChain:
    def test_falls_back_when_first_provider_exhausted(self, budget):
        # Exhaust groq budget
        for _ in range(DAILY_LIMITS["groq"]):
            budget._counters["groq"] = DAILY_LIMITS["groq"]

        cfg = MagicMock()
        cfg.get.return_value = ""
        router = LLMRouter(config=cfg, budget_manager=budget)

        with patch.object(router, "_call_gemini_flash", return_value="fallback"):
            result = router.call("q", complexity=TaskComplexity.SIMPLE)
        assert result == "fallback"

    def test_falls_back_on_rate_limit_error(self, budget):
        from src.llm.router import _ProviderSkip
        cfg = MagicMock()
        cfg.get.return_value = ""
        router = LLMRouter(config=cfg, budget_manager=budget)

        with patch.object(router, "_call_groq", side_effect=_ProviderSkip("rate limited")):
            with patch.object(router, "_call_gemini_flash", return_value="ok"):
                result = router.call("q", complexity=TaskComplexity.SIMPLE)
        assert result == "ok"

    def test_raises_when_all_providers_fail(self, budget):
        from src.llm.router import _ProviderSkip
        cfg = MagicMock()
        cfg.get.return_value = ""
        router = LLMRouter(config=cfg, budget_manager=budget)

        with patch.object(router, "_call_groq",         side_effect=_ProviderSkip("fail")):
            with patch.object(router, "_call_gemini_flash", side_effect=_ProviderSkip("fail")):
                with patch.object(router, "_call_ollama",   side_effect=_ProviderSkip("fail")):
                    with pytest.raises(LLMUnavailableError):
                        router.call("q", complexity=TaskComplexity.SIMPLE)

    def test_raises_when_all_providers_exhausted(self, budget):
        # Exhaust every provider in the SIMPLE chain
        for p in ["groq", "gemini_flash", "ollama_local"]:
            budget._counters[p] = DAILY_LIMITS[p]

        cfg = MagicMock()
        cfg.get.return_value = ""
        router = LLMRouter(config=cfg, budget_manager=budget)

        with pytest.raises(LLMUnavailableError):
            router.call("q", complexity=TaskComplexity.SIMPLE)

    def test_budget_incremented_only_on_success(self, budget):
        from src.llm.router import _ProviderSkip
        initial_groq = budget.get_remaining("groq")
        initial_flash = budget.get_remaining("gemini_flash")

        cfg = MagicMock()
        cfg.get.return_value = ""
        router = LLMRouter(config=cfg, budget_manager=budget)

        with patch.object(router, "_call_groq", side_effect=_ProviderSkip("fail")):
            with patch.object(router, "_call_gemini_flash", return_value="ok"):
                router.call("q", complexity=TaskComplexity.SIMPLE)

        assert budget.get_remaining("groq")        == initial_groq    # not decremented (failed)
        assert budget.get_remaining("gemini_flash") == initial_flash - 1  # decremented (succeeded)


# =========================================================================== #
# LLMRouter — response_format=json                                             #
# =========================================================================== #

class TestResponseFormat:
    def test_json_format_appends_instruction(self, budget):
        cfg = MagicMock()
        cfg.get.return_value = ""
        router = LLMRouter(config=cfg, budget_manager=budget)

        captured: list[str] = []

        def capture_prompt(prompt, system_prompt):
            captured.append(prompt)
            return "{}"

        with patch.object(router, "_call_groq", side_effect=capture_prompt):
            router.call("my prompt", complexity=TaskComplexity.SIMPLE, response_format="json")

        assert len(captured) == 1
        assert "JSON" in captured[0] or "json" in captured[0].lower()
        assert "my prompt" in captured[0]

    def test_text_format_does_not_append_instruction(self, budget):
        cfg = MagicMock()
        cfg.get.return_value = ""
        router = LLMRouter(config=cfg, budget_manager=budget)

        captured: list[str] = []

        def capture_prompt(prompt, system_prompt):
            captured.append(prompt)
            return "hello"

        with patch.object(router, "_call_groq", side_effect=capture_prompt):
            router.call("my prompt", complexity=TaskComplexity.SIMPLE, response_format="text")

        assert captured[0] == "my prompt"


# =========================================================================== #
# LLMRouter — lazy initialisation                                              #
# =========================================================================== #

class TestLazyInit:
    def test_clients_none_before_first_call(self, router):
        assert router._gemini_flash_client is None
        assert router._gemini_pro_client   is None
        assert router._groq_client         is None
        assert router._nvidia_client       is None

    def test_groq_client_not_created_for_bulk_task(self, budget):
        cfg = MagicMock()
        cfg.get.return_value = ""
        router = LLMRouter(config=cfg, budget_manager=budget)

        with patch.object(router, "_call_ollama", return_value="ok"):
            router.call("q", complexity=TaskComplexity.BULK)

        assert router._groq_client is None


# =========================================================================== #
# BudgetManager — usage summary                                                #
# =========================================================================== #

class TestUsageSummary:
    def test_summary_structure(self, budget):
        summary = budget.get_usage_summary()
        assert "date" in summary
        assert "providers" in summary
        for resource in DAILY_LIMITS:
            assert resource in summary["providers"]
            info = summary["providers"][resource]
            assert "used" in info
            assert "limit" in info
            assert "remaining" in info
            assert "pct_used" in info

    def test_summary_reflects_usage(self, budget):
        budget.use("groq")
        budget.use("groq")
        summary = budget.get_usage_summary()
        assert summary["providers"]["groq"]["used"] == 2
        assert summary["providers"]["groq"]["remaining"] == DAILY_LIMITS["groq"] - 2
