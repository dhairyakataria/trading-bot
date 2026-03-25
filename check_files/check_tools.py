"""Smoke-test script for WebSearchTool and NewsFetcher.

Exercises every public method using free providers (DuckDuckGo, Google News
RSS, MoneyControl RSS, yfinance).  Paid providers (Tavily, SerpAPI, NewsAPI)
are tested automatically if their API keys are found in the environment.
No trades are placed and no data is written to the database.

Usage (from the trading-bot/ directory):
    python check_tools.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: ensure src.* imports resolve from this directory
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Auto-add venv site-packages so the script works without activating the venv.
# Covers both Windows (.venv/Lib/site-packages) and Linux (.venv/lib/pythonX.Y/site-packages).
_venv_root = HERE.parent / ".venv"
if _venv_root.exists():
    # Windows layout
    _win_sp = _venv_root / "Lib" / "site-packages"
    if _win_sp.exists() and str(_win_sp) not in sys.path:
        sys.path.insert(1, str(_win_sp))
    # Unix layout (glob over pythonX.Y dirs)
    for _unix_sp in (_venv_root / "lib").glob("python*/site-packages"):
        if str(_unix_sp) not in sys.path:
            sys.path.insert(1, str(_unix_sp))

# Force UTF-8 on Windows consoles (cp1252 can't render box-drawing / tick chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env", override=False)
except Exception:
    pass   # dotenv not critical — env vars may already be set


# ---------------------------------------------------------------------------
# Minimal config wrapper (reads API keys from environment directly,
# so the script works even if config.yaml is not fully configured)
# ---------------------------------------------------------------------------
class _Cfg:
    """Light-weight config shim used only by this script."""

    _ENV = {
        ("apis", "tavily",  "api_key"): "TAVILY_API_KEY",
        ("apis", "newsapi", "api_key"): "NEWSAPI_API_KEY",
        ("apis", "serpapi", "api_key"): "SERPAPI_API_KEY",
    }

    def get(self, *keys: str, default: Any = None) -> Any:  # noqa: ANN401
        env_var = self._ENV.get(keys)
        if env_var:
            value = os.environ.get(env_var, "")
            # Return placeholder-style string when key absent so tools
            # recognise it as "not configured" (same as config.yaml behaviour)
            return value if value else f"${{{env_var}}}"
        return default


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from src.llm.budget_manager import BudgetManager
    from src.tools.web_search import WebSearchTool
    from src.tools.news_fetcher import NewsFetcher
except ImportError as exc:
    print(f"\n[ERROR] Could not import tools: {exc}")
    print("        Make sure you're running from the trading-bot/ directory.\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Shared instances
# ---------------------------------------------------------------------------
_cfg = _Cfg()
_budget = BudgetManager(db=None)   # in-memory only — no DB needed
_ws = WebSearchTool(config=_cfg, budget_manager=_budget)
_nf = NewsFetcher(config=_cfg, budget_manager=_budget)


# ---------------------------------------------------------------------------
# Output helpers  (matches check_api.py style)
# ---------------------------------------------------------------------------
SEP  = "─" * 65
SEP2 = "═" * 65
PASS_SYM = "  ✔"
FAIL_SYM = "  ✘"
SKIP_SYM = "  ○"


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _ok(label: str, detail: str = "") -> None:
    suffix = f"  →  {detail}" if detail else ""
    print(f"{PASS_SYM}  {label}{suffix}")


def _fail(label: str, err: str = "") -> None:
    print(f"{FAIL_SYM}  {label}")
    if err:
        print(f"       {err[:120]}")


def _skip(label: str, reason: str = "") -> None:
    print(f"{SKIP_SYM}  {label}  (skipped: {reason})")


def _show_articles(items: list[dict], max_show: int = 3) -> None:
    """Pretty-print the first few articles."""
    for i, a in enumerate(items[:max_show], 1):
        title  = (a.get("title") or "")[:60]
        url    = (a.get("url") or "")[:65]
        source = a.get("source", "")
        date   = (a.get("published_date") or "")[:16]
        meta   = "  |  ".join(filter(None, [source, date]))
        print(f"       {i}. {title}")
        if meta:
            print(f"          [{meta}]")
        print(f"          {url}")


def _timer(fn):
    """Run fn(), return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn()
    ms = int((time.perf_counter() - t0) * 1000)
    return result, ms


# Result ledger for the final summary
_results: list[tuple[str, str]] = []   # (label, "PASS"|"FAIL"|"SKIP")


def _record(label: str, status: str) -> None:
    _results.append((label, status))


# ---------------------------------------------------------------------------
# ── SECTION A: WebSearchTool ────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Optional-package pre-flight  (imported once, used by all checks below)
# ---------------------------------------------------------------------------
def _importable(module: str) -> bool:
    """Return True if *module* can be imported."""
    import importlib.util
    return importlib.util.find_spec(module) is not None

_HAS_DDG       = _importable("ddgs") or _importable("duckduckgo_search")
_HAS_FEEDPARSER = _importable("feedparser")
_HAS_TAVILY    = _importable("tavily")
_HAS_SERPAPI   = _importable("serpapi")
_HAS_NEWSPAPER = _importable("newspaper")


def check_duckduckgo() -> None:
    label = "DuckDuckGo — search('RELIANCE NSE India stock news')"
    if not _HAS_DDG:
        _skip(label, "ddgs not installed")
        _record(label, "SKIP")
        print("        Install with:  pip install ddgs")
        return
    try:
        results, ms = _timer(lambda: _ws.search_duckduckgo("RELIANCE NSE India stock news", max_results=3))
        if results:
            _ok(label, f"{len(results)} result(s)  [{ms} ms]")
            _show_articles(results)
            _record(label, "PASS")
        else:
            _fail(label, "returned empty list")
            _record(label, "FAIL")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_search_main() -> list[dict]:
    """Main search() fallback chain — returns results for later use."""
    label = "search() — fallback chain (Tavily→DDG→SerpAPI)"
    if not (_HAS_TAVILY or _HAS_DDG or _HAS_SERPAPI):
        _skip(label, "no search provider installed")
        _record(label, "SKIP")
        return []
    try:
        results, ms = _timer(lambda: _ws.search("TCS NSE India latest news", max_results=3))
        if results:
            _ok(label, f"{len(results)} result(s)  [{ms} ms]")
            _show_articles(results)
            _record(label, "PASS")
            return results
        else:
            _fail(label, "all providers returned empty")
            _record(label, "FAIL")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")
    return []


def check_stock_news() -> None:
    label = "search_stock_news('INFY')"
    if not (_HAS_TAVILY or _HAS_DDG or _HAS_SERPAPI):
        _skip(label, "no search provider installed")
        _record(label, "SKIP")
        return
    try:
        results, ms = _timer(lambda: _ws.search_stock_news("INFY", days=3))
        if results:
            _ok(label, f"{len(results)} result(s)  [{ms} ms]")
            _show_articles(results, max_show=2)
            _record(label, "PASS")
        else:
            _fail(label, "returned empty list")
            _record(label, "FAIL")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_sector_news() -> None:
    label = "search_sector_news('IT')"
    if not (_HAS_TAVILY or _HAS_DDG or _HAS_SERPAPI):
        _skip(label, "no search provider installed")
        _record(label, "SKIP")
        return
    try:
        results, ms = _timer(lambda: _ws.search_sector_news("IT"))
        if results:
            _ok(label, f"{len(results)} result(s)  [{ms} ms]")
            _show_articles(results, max_show=2)
            _record(label, "PASS")
        else:
            _fail(label, "returned empty list")
            _record(label, "FAIL")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_read_article(url: str) -> None:
    label = f"read_article('{url[:45]}…')" if url else "read_article()"
    if not url:
        _skip(label, "no URL available from previous search")
        _record(label, "SKIP")
        return
    if not _HAS_NEWSPAPER:
        _skip(label, "newspaper3k / lxml-html-clean not installed")
        _record(label, "SKIP")
        print("        Install with:  pip install newspaper3k lxml-html-clean")
        return
    try:
        result, ms = _timer(lambda: _ws.read_article(url))
        if "error" in result:
            _fail(label, result["error"][:100])
            _record(label, "FAIL")
        else:
            title = (result.get("title") or "")[:55]
            text_len = len(result.get("text") or "")
            _ok(label, f"title='{title}'  text={text_len} chars  [{ms} ms]")
            _record(label, "PASS")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_tavily() -> None:
    label = "Tavily — search_tavily('Nifty 50 today')"
    if not _HAS_TAVILY:
        _skip(label, "tavily-python not installed  →  pip install tavily-python")
        _record(label, "SKIP")
        return
    api_key = _cfg.get("apis", "tavily", "api_key")
    if not api_key or api_key.startswith("${"):
        _skip(label, "TAVILY_API_KEY not set in .env")
        _record(label, "SKIP")
        return
    try:
        results, ms = _timer(lambda: _ws.search_tavily("Nifty 50 today", max_results=3))
        if results:
            _ok(label, f"{len(results)} result(s)  [{ms} ms]")
            _show_articles(results, max_show=2)
            _record(label, "PASS")
        else:
            _fail(label, "returned empty list")
            _record(label, "FAIL")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_serpapi() -> None:
    label = "SerpAPI — search_serp('BSE Sensex news')"
    if not _HAS_SERPAPI:
        _skip(label, "google-search-results not installed  →  pip install google-search-results")
        _record(label, "SKIP")
        return
    api_key = _cfg.get("apis", "serpapi", "api_key")
    if not api_key or api_key.startswith("${"):
        _skip(label, "SERPAPI_API_KEY not set in .env")
        _record(label, "SKIP")
        return
    # Validate key before running the full search to give a clear SKIP reason
    try:
        from serpapi import GoogleSearch as _GS
        _probe = _GS({"q": "test", "api_key": api_key, "num": 1, "engine": "google"}).get_dict()
        if "error" in _probe:
            _skip(label, f"key invalid — {_probe['error']}")
            _record(label, "SKIP")
            return
    except Exception:
        pass  # If probe itself fails, let the real call show the error
    try:
        results, ms = _timer(lambda: _ws.search_serp("BSE Sensex news", max_results=3))
        if results:
            _ok(label, f"{len(results)} result(s)  [{ms} ms]")
            _show_articles(results, max_show=2)
            _record(label, "PASS")
        else:
            _fail(label, "returned empty list")
            _record(label, "FAIL")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


# ---------------------------------------------------------------------------
# ── SECTION B: NewsFetcher ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def check_google_news_rss() -> None:
    label = "Google News RSS — 'Indian stock market NSE today'  (no API key needed)"
    if not _HAS_FEEDPARSER:
        _skip(label, "feedparser not installed  →  pip install feedparser")
        _record(label, "SKIP")
        return
    try:
        results, ms = _timer(
            lambda: _nf.fetch_google_news_rss("Indian stock market NSE today", max_results=5)
        )
        if results:
            _ok(label, f"{len(results)} article(s)  [{ms} ms]")
            _show_articles(results, max_show=3)
            _record(label, "PASS")
        else:
            _fail(label, "returned empty list (feedparser installed?)")
            _record(label, "FAIL")
    except ImportError as exc:
        _skip(label, f"feedparser not installed: {exc}")
        _record(label, "SKIP")
        print("        Install with:  pip install feedparser")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_moneycontrol_rss() -> None:
    label = "MoneyControl RSS — category='market'  (no API key needed)"
    if not _HAS_FEEDPARSER:
        _skip(label, "feedparser not installed  →  pip install feedparser")
        _record(label, "SKIP")
        return
    try:
        results, ms = _timer(lambda: _nf.fetch_moneycontrol_rss("market"))
        if results:
            _ok(label, f"{len(results)} article(s)  [{ms} ms]")
            _show_articles(results, max_show=3)
            _record(label, "PASS")
        else:
            _fail(label, "returned empty list")
            _record(label, "FAIL")
    except ImportError as exc:
        _skip(label, f"feedparser not installed: {exc}")
        _record(label, "SKIP")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_newsapi() -> None:
    label = "NewsAPI — 'India NSE BSE stock market'"
    api_key = _cfg.get("apis", "newsapi", "api_key")
    if not api_key or api_key.startswith("${"):
        _skip(label, "NEWSAPI_API_KEY not set in .env")
        _record(label, "SKIP")
        return
    try:
        results, ms = _timer(
            lambda: _nf.fetch_newsapi("India NSE BSE stock market", page_size=5)
        )
        if results:
            _ok(label, f"{len(results)} article(s)  [{ms} ms]")
            _show_articles(results, max_show=2)
            _record(label, "PASS")
        else:
            _fail(label, "returned empty list")
            _record(label, "FAIL")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_market_news() -> None:
    label = "fetch_market_news() — aggregated (RSS + NewsAPI)"
    try:
        results, ms = _timer(lambda: _nf.fetch_market_news(max_results=10))
        if results:
            _ok(label, f"{len(results)} deduplicated article(s)  [{ms} ms]")
            _show_articles(results, max_show=3)
            _record(label, "PASS")
        else:
            _fail(label, "returned empty list")
            _record(label, "FAIL")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_stock_specific_news() -> None:
    label = "fetch_stock_specific_news('TCS', 'Tata Consultancy')"
    try:
        results, ms = _timer(
            lambda: _nf.fetch_stock_specific_news("TCS", "Tata Consultancy")
        )
        if results:
            _ok(label, f"{len(results)} article(s)  [{ms} ms]")
            _show_articles(results, max_show=3)
            _record(label, "PASS")
        else:
            _fail(label, "returned empty list")
            _record(label, "FAIL")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_fii_dii() -> None:
    label = "get_fii_dii_data() — NSE FII/DII activity"
    try:
        data, ms = _timer(_nf.get_fii_dii_data)
        if data.get("source") == "unavailable":
            _fail(label, data.get("error", "NSE API blocked"))
            _record(label, "FAIL")
            print("       (NSE often blocks automated requests — this is expected outside market hours)")
        else:
            _ok(label, f"source={data['source']}  [{ms} ms]")
            print(f"       Date     : {data.get('date', 'N/A')}")
            print(f"       FII Buy  : ₹{data.get('fii_buy', 0):>12,.2f} Cr")
            print(f"       FII Sell : ₹{data.get('fii_sell', 0):>12,.2f} Cr")
            print(f"       FII Net  : ₹{data.get('fii_net', 0):>+12,.2f} Cr")
            print(f"       DII Net  : ₹{data.get('dii_net', 0):>+12,.2f} Cr")
            _record(label, "PASS")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_global_markets() -> None:
    label = "get_global_market_status() — yfinance"
    try:
        data, ms = _timer(_nf.get_global_market_status)
        if "error" in data:
            _fail(label, data["error"])
            _record(label, "FAIL")
            return
        _ok(label, f"fetched {len(data) - 1} instruments  [{ms} ms]")   # -1 for timestamp

        _LABELS = {
            "sp500":     "S&P 500    ",
            "nasdaq":    "Nasdaq     ",
            "dow_jones": "Dow Jones  ",
            "nikkei":    "Nikkei 225 ",
            "hang_seng": "Hang Seng  ",
            "crude_oil": "Crude Oil  ",
            "gold":      "Gold       ",
            "usd_inr":   "USD/INR    ",
        }
        for key, lbl in _LABELS.items():
            info = data.get(key, {})
            if "error" in info:
                print(f"       {lbl}  (error: {info['error'][:40]})")
            else:
                chg = info.get("change_pct", 0)
                arrow = "▲" if chg >= 0 else "▼"
                print(
                    f"       {lbl}  {info.get('last_price', 0):>10,.2f}"
                    f"   {arrow} {abs(chg):.2f}%"
                )
        _record(label, "PASS")
    except ImportError as exc:
        _skip(label, f"yfinance not installed: {exc}")
        _record(label, "SKIP")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


def check_india_vix() -> None:
    label = "get_india_vix() — yfinance ^INDIAVIX"
    try:
        data, ms = _timer(_nf.get_india_vix)
        if "error" in data:
            _fail(label, data["error"])
            _record(label, "FAIL")
        else:
            _ok(
                label,
                f"VIX={data['vix']}  signal={data['signal']}  [{ms} ms]",
            )
            print(f"       {data.get('interpretation', '')}")
            _record(label, "PASS")
    except Exception as exc:
        _fail(label, str(exc))
        _record(label, "FAIL")


# ---------------------------------------------------------------------------
# Budget display
# ---------------------------------------------------------------------------
def show_budget() -> None:
    section("Budget Remaining (after this run)")
    resources = ["tavily_search", "news_api", "serp_api", "web_scrape"]
    for r in resources:
        remaining = _budget.get_remaining(r)
        used = _budget._counters.get(r, 0)
        print(f"       {r:<20}  used={used}  remaining={remaining}")


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
def summary() -> None:
    print(f"\n{SEP2}")
    print("  SUMMARY")
    print(SEP2)
    passed  = [l for l, s in _results if s == "PASS"]
    failed  = [l for l, s in _results if s == "FAIL"]
    skipped = [l for l, s in _results if s == "SKIP"]

    for label, status in _results:
        icon = "✔" if status == "PASS" else ("○" if status == "SKIP" else "✘")
        print(f"  {icon}  [{status:<4}]  {label[:70]}")

    print(f"\n  {len(passed)} passed  ·  {len(failed)} failed  ·  {len(skipped)} skipped")

    if failed:
        print("\n  FAILED CHECKS:")
        for f in failed:
            print(f"    ✘  {f}")

    if skipped:
        print("\n  SKIPPED (optional packages / API keys not configured):")
        for s in skipped:
            print(f"    ○  {s}")

    print(f"\n{SEP2}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"\n{SEP2}")
    print("  WebSearchTool + NewsFetcher — Smoke Test")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(SEP2)

    # ── A. WebSearchTool ──────────────────────────────────────────────
    section("A · WebSearchTool — DuckDuckGo (free, no key)")
    check_duckduckgo()
    search_results = check_search_main()
    first_url = search_results[0].get("url", "") if search_results else ""

    section("A · WebSearchTool — Stock / Sector helpers")
    check_stock_news()
    check_sector_news()

    section("A · WebSearchTool — Article reader (newspaper3k)")
    check_read_article(first_url)

    section("A · WebSearchTool — Paid providers (skipped if no key)")
    check_tavily()
    check_serpapi()

    # ── B. NewsFetcher ────────────────────────────────────────────────
    section("B · NewsFetcher — Google News RSS (free, no key)")
    check_google_news_rss()

    section("B · NewsFetcher — MoneyControl RSS (free, no key)")
    check_moneycontrol_rss()

    section("B · NewsFetcher — NewsAPI (skipped if no key)")
    check_newsapi()

    section("B · NewsFetcher — Aggregated market news")
    check_market_news()
    check_stock_specific_news()

    section("B · NewsFetcher — Market data (yfinance, free)")
    check_fii_dii()
    check_global_markets()
    check_india_vix()

    show_budget()
    summary()


if __name__ == "__main__":
    main()
