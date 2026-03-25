"""LLM Router — dispatches prompts to the cheapest available provider.

Provider priority per task complexity:

    SIMPLE   → Groq → Gemini Flash → Ollama local
    MODERATE → Gemini Flash → Groq → NVIDIA NIM → Ollama local
    COMPLEX + high priority → Gemini Pro → Gemini Flash → NVIDIA NIM
    COMPLEX + normal        → Gemini Flash → NVIDIA NIM → Groq
    BULK     → Ollama local → Groq

All provider clients are lazily initialised.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from src.llm.budget_manager import BudgetManager
from src.utils.logger import get_logger

_log = get_logger("llm.router")

_CALL_TIMEOUT = 30  # seconds


# --------------------------------------------------------------------------- #
# Enums / Exceptions                                                           #
# --------------------------------------------------------------------------- #

class TaskComplexity(Enum):
    SIMPLE   = "simple"    # extraction, formatting, classification
    MODERATE = "moderate"  # summarisation, sentiment analysis
    COMPLEX  = "complex"   # multi-step reasoning, trade decisions
    BULK     = "bulk"      # large-text processing, always local


class LLMUnavailableError(Exception):
    """Raised when every provider in the fallback chain has failed or is exhausted."""


# --------------------------------------------------------------------------- #
# Routing tables                                                               #
# --------------------------------------------------------------------------- #

_ROUTES: dict[str, list[str]] = {
    "simple":          ["groq",          "gemini_flash", "ollama_local"],
    "moderate":        ["gemini_flash",  "groq",         "nvidia_nim",   "ollama_local"],
    "complex_high":    ["gemini_pro",    "gemini_flash", "nvidia_nim"],
    "complex_normal":  ["gemini_flash",  "nvidia_nim",   "groq"],
    "bulk":            ["ollama_local",  "groq"],
}


def _route_key(complexity: TaskComplexity, priority: str) -> str:
    if complexity == TaskComplexity.SIMPLE:
        return "simple"
    if complexity == TaskComplexity.MODERATE:
        return "moderate"
    if complexity == TaskComplexity.BULK:
        return "bulk"
    # COMPLEX
    return "complex_high" if priority == "high" else "complex_normal"


# --------------------------------------------------------------------------- #
# LLMRouter                                                                    #
# --------------------------------------------------------------------------- #

class LLMRouter:
    """Routes LLM calls across Gemini, Groq, NVIDIA NIM, and Ollama.

    Args:
        config:         Application :class:`~src.utils.config.Config` instance.
        budget_manager: :class:`~src.llm.budget_manager.BudgetManager` that
                        tracks daily quotas.
    """

    def __init__(self, config: Any, budget_manager: BudgetManager) -> None:
        self._config  = config
        self._budget  = budget_manager

        # Lazy client holders
        self._gemini_flash_client: Optional[Any] = None
        self._gemini_pro_client:   Optional[Any] = None
        self._groq_client:         Optional[Any] = None
        self._nvidia_client:       Optional[Any] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def call(
        self,
        prompt: str,
        system_prompt: str = "",
        complexity: TaskComplexity = TaskComplexity.MODERATE,
        priority: str = "normal",
        response_format: str = "text",
    ) -> str:
        """Dispatch *prompt* to the best available provider and return the response.

        Args:
            prompt:          The user / task prompt.
            system_prompt:   Optional system instruction.
            complexity:      Task complexity hint used for provider selection.
            priority:        ``"high"`` or ``"normal"`` (only matters for COMPLEX tasks).
            response_format: ``"text"`` (default) or ``"json"`` (appends a JSON instruction).

        Returns:
            The model's response as a plain string.

        Raises:
            :exc:`LLMUnavailableError`: if every provider in the chain fails.
        """
        if response_format == "json":
            prompt = (
                prompt
                + "\n\nIMPORTANT: Respond with valid JSON only. "
                "Do not include any text outside the JSON object."
            )

        route = _route_key(complexity, priority)
        providers = _ROUTES[route]

        _log.debug(
            "LLMRouter.call — complexity=%s priority=%s route=%s providers=%s | prompt=%.200s",
            complexity.value, priority, route, providers, prompt,
        )

        errors: list[str] = []
        for provider in providers:
            if not self._budget.can_use(provider):
                _log.debug("Provider '%s' budget exhausted — skipping.", provider)
                continue

            try:
                t0 = time.monotonic()
                response = self._call_provider(provider, prompt, system_prompt)
                elapsed = time.monotonic() - t0

                self._budget.use(provider)
                self._budget.save_state()

                _log.debug(
                    "Provider '%s' responded in %.2fs | response=%.200s",
                    provider, elapsed, response,
                )
                return response

            except _ProviderSkip as exc:
                _log.warning("Provider '%s' skipped: %s", provider, exc)
                errors.append(f"{provider}: {exc}")
            except Exception as exc:
                _log.warning("Provider '%s' unexpected error: %s", provider, exc)
                errors.append(f"{provider}: {exc}")

        raise LLMUnavailableError(
            f"All providers exhausted for route '{route}'. Errors: {errors}"
        )

    def call_with_tools(
        self,
        prompt: str,
        system_prompt: str,
        tools: list[Any],
        complexity: TaskComplexity = TaskComplexity.MODERATE,
        priority: str = "normal",
    ) -> dict[str, Any]:
        """Agentic call — provider may invoke *tools* before returning.

        Uses a LangChain-style agent loop under the hood (Gemini Flash by
        default, falls back to Groq).  Returns a dict with keys:
        ``"output"`` (final text), ``"tool_calls"`` (list of tool invocations).
        """
        route = _route_key(complexity, priority)
        providers = _ROUTES[route]

        _log.debug(
            "LLMRouter.call_with_tools — complexity=%s tools=%d | prompt=%.200s",
            complexity.value, len(tools), prompt,
        )

        errors: list[str] = []
        for provider in providers:
            if not self._budget.can_use(provider):
                continue

            try:
                t0 = time.monotonic()
                result = self._call_provider_with_tools(
                    provider, prompt, system_prompt, tools
                )
                elapsed = time.monotonic() - t0

                self._budget.use(provider)
                self._budget.save_state()

                _log.debug(
                    "Provider '%s' tool-call responded in %.2fs", provider, elapsed
                )
                return result

            except _ProviderSkip as exc:
                _log.warning("Provider '%s' skipped: %s", provider, exc)
                errors.append(f"{provider}: {exc}")
            except Exception as exc:
                _log.warning("Provider '%s' unexpected error: %s", provider, exc)
                errors.append(f"{provider}: {exc}")

        raise LLMUnavailableError(
            f"All providers exhausted for tool call (route='{route}'). Errors: {errors}"
        )

    # ------------------------------------------------------------------ #
    # Provider dispatch                                                    #
    # ------------------------------------------------------------------ #

    def _call_provider(self, provider: str, prompt: str, system_prompt: str) -> str:
        dispatch = {
            "gemini_flash": self._call_gemini_flash,
            "gemini_pro":   self._call_gemini_pro,
            "groq":         self._call_groq,
            "nvidia_nim":   self._call_nvidia_nim,
            "ollama_local": self._call_ollama,
        }
        fn = dispatch.get(provider)
        if fn is None:
            raise _ProviderSkip(f"No implementation for provider '{provider}'")
        return fn(prompt, system_prompt)

    def _call_provider_with_tools(
        self,
        provider: str,
        prompt: str,
        system_prompt: str,
        tools: list[Any],
    ) -> dict[str, Any]:
        """Minimal tool-calling wrapper (Gemini + Groq supported)."""
        if provider in ("gemini_flash", "gemini_pro"):
            return self._call_gemini_with_tools(provider, prompt, system_prompt, tools)
        if provider == "groq":
            return self._call_groq_with_tools(prompt, system_prompt, tools)
        # Fallback: plain call, no tool invocations
        text = self._call_provider(provider, prompt, system_prompt)
        return {"output": text, "tool_calls": []}

    # ------------------------------------------------------------------ #
    # Gemini                                                               #
    # ------------------------------------------------------------------ #

    def _get_gemini_client(self) -> Any:
        """Lazy-init shared google.genai Client."""
        if self._gemini_flash_client is None:
            import os
            from google import genai  # type: ignore[import-untyped]
            api_key = os.getenv("GEMINI_API_KEY") or self._config.get("llm", "gemini", "api_key", default="")
            if not api_key or api_key.startswith("${"):
                raise _ProviderSkip("GEMINI_API_KEY not set.")
            self._gemini_flash_client = genai.Client(api_key=api_key)
        return self._gemini_flash_client

    def _call_gemini(self, model_name: str, prompt: str, system_prompt: str) -> str:
        from google.genai import types  # type: ignore[import-untyped]
        client = self._get_gemini_client()
        try:
            cfg = types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                http_options=types.HttpOptions(timeout=_CALL_TIMEOUT * 1000),
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=cfg,
            )
            return response.text
        except Exception as exc:
            exc_str = str(exc).lower()
            if any(k in exc_str for k in ("quota", "rate", "resource_exhausted", "429")):
                raise _ProviderSkip(f"Rate-limited: {exc}") from exc
            if any(k in exc_str for k in ("api_key", "auth", "permission", "403", "401")):
                raise _ProviderSkip(f"Auth error: {exc}") from exc
            if any(k in exc_str for k in ("timeout", "network", "connection", "socket")):
                raise _ProviderSkip(f"Network error: {exc}") from exc
            raise

    def _call_gemini_flash(self, prompt: str, system_prompt: str) -> str:
        return self._call_gemini("gemini-2.0-flash", prompt, system_prompt)

    def _call_gemini_pro(self, prompt: str, system_prompt: str) -> str:
        return self._call_gemini("gemini-2.5-pro", prompt, system_prompt)

    def _call_gemini_with_tools(
        self, provider: str, prompt: str, system_prompt: str, tools: list[Any]
    ) -> dict[str, Any]:
        from google.genai import types  # type: ignore[import-untyped]
        model_name = "gemini-2.0-flash" if provider == "gemini_flash" else "gemini-2.5-pro"
        client = self._get_gemini_client()
        try:
            cfg = types.GenerateContentConfig(
                tools=tools,
                system_instruction=system_prompt if system_prompt else None,
                http_options=types.HttpOptions(timeout=_CALL_TIMEOUT * 1000),
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=cfg,
            )
            tool_calls = []
            output_text = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_calls.append({
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args),
                    })
                elif hasattr(part, "text"):
                    output_text += part.text
            return {"output": output_text, "tool_calls": tool_calls}
        except Exception as exc:
            exc_str = str(exc).lower()
            if any(k in exc_str for k in ("quota", "rate", "429", "resource_exhausted")):
                raise _ProviderSkip(f"Rate-limited: {exc}") from exc
            if any(k in exc_str for k in ("api_key", "auth", "403", "401")):
                raise _ProviderSkip(f"Auth error: {exc}") from exc
            if any(k in exc_str for k in ("timeout", "network", "connection")):
                raise _ProviderSkip(f"Network error: {exc}") from exc
            raise

    # ------------------------------------------------------------------ #
    # Groq                                                                 #
    # ------------------------------------------------------------------ #

    def _get_groq(self) -> Any:
        if self._groq_client is None:
            import os
            from groq import Groq  # type: ignore[import-untyped]
            api_key = os.getenv("GROQ_API_KEY") or self._config.get("llm", "groq", "api_key", default="")
            if not api_key or api_key.startswith("${"):
                raise _ProviderSkip("GROQ_API_KEY not set.")
            self._groq_client = Groq(api_key=api_key)
        return self._groq_client

    def _call_groq(self, prompt: str, system_prompt: str) -> str:
        try:
            client = self._get_groq()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                timeout=_CALL_TIMEOUT,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            exc_str = str(exc).lower()
            if any(k in exc_str for k in ("rate", "429", "too many")):
                raise _ProviderSkip(f"Rate-limited: {exc}") from exc
            if any(k in exc_str for k in ("auth", "api_key", "401", "403")):
                raise _ProviderSkip(f"Auth error: {exc}") from exc
            if any(k in exc_str for k in ("timeout", "network", "connection")):
                raise _ProviderSkip(f"Network error: {exc}") from exc
            raise

    def _call_groq_with_tools(
        self, prompt: str, system_prompt: str, tools: list[Any]
    ) -> dict[str, Any]:
        try:
            client = self._get_groq()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                timeout=_CALL_TIMEOUT,
            )
            msg = response.choices[0].message
            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    import json as _json
                    tool_calls.append({
                        "name": tc.function.name,
                        "args": _json.loads(tc.function.arguments or "{}"),
                    })
            return {"output": msg.content or "", "tool_calls": tool_calls}
        except Exception as exc:
            exc_str = str(exc).lower()
            if any(k in exc_str for k in ("rate", "429")):
                raise _ProviderSkip(f"Rate-limited: {exc}") from exc
            if any(k in exc_str for k in ("auth", "401", "403")):
                raise _ProviderSkip(f"Auth error: {exc}") from exc
            if any(k in exc_str for k in ("timeout", "network", "connection")):
                raise _ProviderSkip(f"Network error: {exc}") from exc
            raise

    # ------------------------------------------------------------------ #
    # NVIDIA NIM                                                           #
    # ------------------------------------------------------------------ #

    def _get_nvidia(self) -> Any:
        if self._nvidia_client is None:
            import os
            from openai import OpenAI  # type: ignore[import-untyped]
            api_key = os.getenv("NVIDIA_API_KEY") or self._config.get("llm", "nvidia_nim", "api_key", default="")
            if not api_key or api_key.startswith("${"):
                raise _ProviderSkip("NVIDIA_API_KEY not set.")
            self._nvidia_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )
        return self._nvidia_client

    def _call_nvidia_nim(self, prompt: str, system_prompt: str) -> str:
        try:
            client = self._get_nvidia()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=messages,
                timeout=_CALL_TIMEOUT,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            exc_str = str(exc).lower()
            if any(k in exc_str for k in ("rate", "429", "too many")):
                raise _ProviderSkip(f"Rate-limited: {exc}") from exc
            if any(k in exc_str for k in ("auth", "api_key", "401", "403")):
                raise _ProviderSkip(f"Auth error: {exc}") from exc
            if any(k in exc_str for k in ("timeout", "network", "connection")):
                raise _ProviderSkip(f"Network error: {exc}") from exc
            raise

    # ------------------------------------------------------------------ #
    # Ollama (local)                                                       #
    # ------------------------------------------------------------------ #

    def _call_ollama(self, prompt: str, system_prompt: str) -> str:
        import requests  # type: ignore[import-untyped]
        base_url = self._config.get("llm", "ollama", "base_url", default="http://localhost:11434")
        model    = self._config.get("llm", "ollama", "model",    default="qwen2.5:7b")
        try:
            payload: dict[str, Any] = {
                "model":  model,
                "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
                "stream": False,
            }
            resp = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=_CALL_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.Timeout as exc:
            raise _ProviderSkip(f"Ollama timeout: {exc}") from exc
        except requests.exceptions.ConnectionError as exc:
            raise _ProviderSkip(f"Ollama not running: {exc}") from exc
        except requests.exceptions.HTTPError as exc:
            raise _ProviderSkip(f"Ollama HTTP error: {exc}") from exc


# --------------------------------------------------------------------------- #
# Internal sentinel                                                            #
# --------------------------------------------------------------------------- #

class _ProviderSkip(Exception):
    """Internal signal to skip to the next provider in the fallback chain."""
