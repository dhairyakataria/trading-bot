"""
LLM Provider Connectivity Test
================================
Tests every configured provider + model combination and reports
what works, what doesn't, and why.

Usage:
    cd e:/codes and projects/codes/Stock_Market_2
    python test_llm_providers.py

Requirements:
    pip install google-generativeai groq openai requests python-dotenv

The script loads API keys from trading-bot/.env automatically.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Load .env from trading-bot folder
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).parent / "trading-bot" / ".env"

try:
    from dotenv import load_dotenv
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
        print(f"[env] Loaded keys from {_ENV_FILE}\n")
    else:
        print(f"[env] WARNING: {_ENV_FILE} not found -relying on shell environment variables\n")
except ImportError:
    print("[env] WARNING: python-dotenv not installed. Run: pip install python-dotenv")
    print("[env]          Falling back to shell environment variables.\n")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    provider: str
    model: str
    status: str          # "OK" | "FAIL" | "SKIP"
    latency_ms: Optional[float] = None
    response_snippet: str = ""
    error: str = ""
    error_type: str = ""  # "auth" | "rate_limit" | "network" | "model_not_found" | "other"


results: list[TestResult] = []

PROMPT = "Reply with exactly: HELLO TEST OK"
TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(s: str) -> str:
    """Strip non-ASCII so Windows cp1252 terminal doesn't choke."""
    return s.encode("ascii", errors="replace").decode("ascii")


def _classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if any(k in msg for k in ("api_key", "auth", "permission", "401", "403", "invalid_api_key", "authentication")):
        return "auth"
    # Check model_not_found BEFORE rate_limit — 404 errors contain "limit" in Google's
    # help text ("Call ListModels ...") which would otherwise match rate_limit.
    if any(k in msg for k in ("not found", "does not exist", "not_found", "invalid_model", "decommissioned",
                               "no longer supported", "no longer available", "404")):
        return "model_not_found"
    if any(k in msg for k in ("quota", "rate", "429", "too many", "resource_exhausted", "limit")):
        return "rate_limit"
    if any(k in msg for k in ("timeout", "timed out", "connection", "network", "socket", "connect")):
        return "network"
    return "other"


def record(provider: str, model: str, t0: float, response: str) -> None:
    elapsed = (time.monotonic() - t0) * 1000
    snippet = response[:80].replace("\n", " ") if response else "(empty response)"
    results.append(TestResult(
        provider=provider,
        model=model,
        status="OK",
        latency_ms=round(elapsed, 1),
        response_snippet=snippet,
    ))
    print(_safe(f"  [OK]  {model}  ({elapsed:.0f} ms)  -> {snippet!r}"))


def fail(provider: str, model: str, exc: Exception) -> None:
    etype = _classify_error(exc)
    msg = str(exc)
    results.append(TestResult(
        provider=provider,
        model=model,
        status="FAIL",
        error=msg[:200],
        error_type=etype,
    ))
    print(_safe(f"  [FAIL] {model}  [{etype}]  {msg[:120]}"))


def skip(provider: str, model: str, reason: str) -> None:
    results.append(TestResult(provider=provider, model=model, status="SKIP", error=reason))
    print(_safe(f"  [SKIP] {model}  - {reason}"))


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ===========================================================================
# GEMINI
# ===========================================================================

GEMINI_MODELS = [
    "gemini-2.5-flash",   # newest flash (confirmed available on this key)
    "gemini-2.5-pro",     # newest pro (router.py pro model — replaces retired preview)
    "gemini-2.0-flash",   # router.py flash model (may be quota-limited on free tier)
    "gemini-flash-latest",  # always-latest flash alias
    "gemini-pro-latest",    # always-latest pro alias
]

def test_gemini() -> None:
    section("GEMINI  (GEMINI_API_KEY)")

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key.startswith("your_") or api_key.startswith("${"):
        for m in GEMINI_MODELS:
            skip("gemini", m, "GEMINI_API_KEY not set")
        return

    try:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore
    except ImportError:
        for m in GEMINI_MODELS:
            skip("gemini", m, "google-genai not installed  ->  pip install google-genai")
        return

    client = genai.Client(api_key=api_key)

    for model_name in GEMINI_MODELS:
        print(f"\n  Testing model: {model_name}")
        try:
            t0 = time.monotonic()
            response = client.models.generate_content(
                model=model_name,
                contents=PROMPT,
                config=genai_types.GenerateContentConfig(
                    http_options=genai_types.HttpOptions(timeout=TIMEOUT * 1000),
                ),
            )
            record("gemini", model_name, t0, response.text)
        except Exception as exc:
            fail("gemini", model_name, exc)


# ===========================================================================
# GROQ
# ===========================================================================

GROQ_MODELS = [
    "llama-3.3-70b-versatile",              # router.py model — confirmed working
    "llama-3.1-8b-instant",                 # fast/cheap variant
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Llama 4 Scout
    "qwen/qwen3-32b",                       # Qwen 3
    "moonshotai/kimi-k2-instruct",          # Kimi K2
]

def test_groq() -> None:
    section("GROQ  (GROQ_API_KEY)")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("your_") or api_key.startswith("${"):
        for m in GROQ_MODELS:
            skip("groq", m, "GROQ_API_KEY not set")
        return

    try:
        from groq import Groq  # type: ignore
    except ImportError:
        for m in GROQ_MODELS:
            skip("groq", m, "groq not installed  ->  pip install groq")
        return

    client = Groq(api_key=api_key)

    for model_name in GROQ_MODELS:
        print(f"\n  Testing model: {model_name}")
        try:
            t0 = time.monotonic()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": PROMPT}],
                timeout=TIMEOUT,
                max_tokens=50,
            )
            text = response.choices[0].message.content or ""
            record("groq", model_name, t0, text)
        except Exception as exc:
            fail("groq", model_name, exc)


# ===========================================================================
# NVIDIA NIM
# ===========================================================================

NVIDIA_MODELS = [
    "meta/llama-3.1-70b-instruct",          # router.py / config.yaml model
    "meta/llama-3.3-70b-instruct",          # newer Llama 3.3
    "meta/llama-3.1-8b-instruct",           # smaller/cheaper variant
    "mistralai/mixtral-8x7b-instruct-v0.1", # Mixtral on NIM
    "nvidia/llama-3.3-nemotron-super-49b-v1",  # NVIDIA Nemotron (updated ID)
]

def test_nvidia_nim() -> None:
    section("NVIDIA NIM  (NVIDIA_API_KEY)")

    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key or api_key.startswith("your_") or api_key.startswith("${"):
        for m in NVIDIA_MODELS:
            skip("nvidia_nim", m, "NVIDIA_API_KEY not set")
        return

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        for m in NVIDIA_MODELS:
            skip("nvidia_nim", m, "openai not installed  ->  pip install openai")
        return

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

    for model_name in NVIDIA_MODELS:
        print(f"\n  Testing model: {model_name}")
        try:
            t0 = time.monotonic()
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": PROMPT}],
                timeout=TIMEOUT,
                max_tokens=50,
            )
            text = response.choices[0].message.content or ""
            record("nvidia_nim", model_name, t0, text)
        except Exception as exc:
            fail("nvidia_nim", model_name, exc)


# ===========================================================================
# OLLAMA (local)
# ===========================================================================

OLLAMA_MODELS = [
    "qwen2.5:7b",    # router.py default
    "llama3.1:8b",   # config.yaml model
]

def test_ollama() -> None:
    section("OLLAMA  (local - no API key needed)")

    try:
        import requests  # type: ignore
    except ImportError:
        for m in OLLAMA_MODELS:
            skip("ollama_local", m, "requests not installed  ->  pip install requests")
        return

    base_url = "http://localhost:11434"

    # First check if Ollama is running at all
    print(f"\n  Checking Ollama at {base_url} ...")
    try:
        ping = requests.get(f"{base_url}/api/tags", timeout=5)
        ping.raise_for_status()
        available_models = [m["name"] for m in ping.json().get("models", [])]
        print(f"  Ollama is running. Installed models: {available_models or '(none)'}")
    except Exception as exc:
        for m in OLLAMA_MODELS:
            skip("ollama_local", m, f"Ollama not reachable at {base_url}: {exc}")
        return

    for model_name in OLLAMA_MODELS:
        print(f"\n  Testing model: {model_name}")
        # Check if model is actually pulled
        if available_models and not any(model_name in am for am in available_models):
            skip("ollama_local", model_name,
                 f"Model not pulled locally. Run: ollama pull {model_name}")
            continue
        try:
            t0 = time.monotonic()
            resp = requests.post(
                f"{base_url}/api/generate",
                json={"model": model_name, "prompt": PROMPT, "stream": False},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
            record("ollama_local", model_name, t0, text)
        except requests.exceptions.Timeout as exc:
            fail("ollama_local", model_name, exc)
        except Exception as exc:
            fail("ollama_local", model_name, exc)


# ===========================================================================
# Summary Report
# ===========================================================================

_STATUS_ICON = {"OK": "[OK]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}

_ERROR_ADVICE = {
    "auth": "Invalid or missing API key -check your .env file",
    "rate_limit": "Quota/rate-limit hit -wait or use a different key",
    "network": "Network/timeout -check internet connection or service status",
    "model_not_found": "Model ID invalid or not available on your plan",
    "other": "Unexpected error -see full error above",
}

def print_summary() -> None:
    section("SUMMARY")

    ok    = [r for r in results if r.status == "OK"]
    fails = [r for r in results if r.status == "FAIL"]
    skips = [r for r in results if r.status == "SKIP"]

    print(f"\n  Total tested : {len(results)}")
    print(f"  Working      : {len(ok)}")
    print(f"  Failed       : {len(fails)}")
    print(f"  Skipped      : {len(skips)}")

    if ok:
        print("\n  --- WORKING MODELS ---")
        for r in ok:
            print(f"  OK   {r.provider:<14}  {r.model:<45}  {r.latency_ms:>6.0f} ms")

    if fails:
        print("\n  --- FAILED MODELS ---")
        for r in fails:
            advice = _ERROR_ADVICE.get(r.error_type, "")
            print(f"  FAIL {r.provider:<14}  {r.model:<45}  [{r.error_type}]")
            if advice:
                print(f"       -> {advice}")

    if skips:
        print("\n  --- SKIPPED (API key not set or package missing) ---")
        # Group by provider+reason
        seen: set[str] = set()
        for r in skips:
            key = f"{r.provider}|{r.error}"
            if key not in seen:
                seen.add(key)
                print(f"  SKIP {r.provider:<14}  {r.error}")

    # Per-provider working models quick list
    print("\n  --- PER-PROVIDER WORKING MODELS ---")
    providers = dict.fromkeys(r.provider for r in results)
    for prov in providers:
        working = [r.model for r in results if r.provider == prov and r.status == "OK"]
        if working:
            print(f"  {prov}:")
            for m in working:
                print(f"    * {m}")
        else:
            not_skipped = [r for r in results if r.provider == prov and r.status != "SKIP"]
            if not_skipped:
                print(f"  {prov}:  (none working)")
            else:
                print(f"  {prov}:  (all skipped -API key not configured)")

    print()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("LLM Provider Test")
    print("=" * 60)
    print(f"Prompt sent to each model: {PROMPT!r}")
    print(f"Timeout per call: {TIMEOUT}s")

    test_gemini()
    test_groq()
    test_nvidia_nim()
    test_ollama()
    print_summary()
