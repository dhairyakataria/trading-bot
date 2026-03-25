"""Web Search — Tavily, DuckDuckGo, and SerpAPI with budget management.

Provider priority: Tavily (1 000 free/month) → DuckDuckGo (unlimited, no key)
                  → SerpAPI (100 free/month, use sparingly).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from src.llm.budget_manager import BudgetManager
from src.utils.logger import get_logger

_log = get_logger("tools.web_search")

# Standard result shape returned by every provider
_EMPTY: list[dict] = []


def _truncate(text: str, max_chars: int = 500) -> str:
    return text[:max_chars] if text else ""


class WebSearchTool:
    """Provides web search capabilities using multiple free providers.

    Falls back through providers when one fails or its budget is exhausted.
    All search methods return a list of dicts with keys:
        title, url, snippet, published_date
    """

    def __init__(self, config: Any, budget_manager: BudgetManager) -> None:
        self.config = config
        self.budget = budget_manager

    # ------------------------------------------------------------------
    # Tavily  (1 000 free/month)
    # ------------------------------------------------------------------

    def search_tavily(self, query: str, max_results: int = 5) -> list[dict]:
        """Search via Tavily API."""
        if not self.budget.can_use("tavily_search"):
            _log.warning("Tavily daily budget exhausted — skipping")
            return _EMPTY

        api_key = self.config.get("apis", "tavily", "api_key", default="")
        if not api_key or api_key.startswith("${"):
            _log.warning("Tavily API key not configured")
            return _EMPTY

        try:
            from tavily import TavilyClient

            _log.info("Tavily search: %r (max_results=%d)", query, max_results)
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query, max_results=max_results, search_depth="advanced"
            )
            self.budget.use("tavily_search")

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": _truncate(r.get("content", "")),
                    "published_date": r.get("published_date", ""),
                }
                for r in response.get("results", [])
            ]

        except Exception as exc:
            _log.error("Tavily search failed: %s", exc)
            return _EMPTY

    # ------------------------------------------------------------------
    # DuckDuckGo  (free, no API key, unlimited)
    # ------------------------------------------------------------------

    def search_duckduckgo(self, query: str, max_results: int = 5) -> list[dict]:
        """Search via DuckDuckGo (duckduckgo-search library, no key required)."""
        try:
            try:
                from ddgs import DDGS  # new package name (pip install ddgs)
            except ImportError:
                from duckduckgo_search import DDGS  # legacy fallback

            _log.info("DuckDuckGo search: %r (max_results=%d)", query, max_results)
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": _truncate(r.get("body", "")),
                            "published_date": "",
                        }
                    )
            return results

        except Exception as exc:
            _log.error("DuckDuckGo search failed: %s", exc)
            return _EMPTY

    # ------------------------------------------------------------------
    # SerpAPI  (100 free/month — use sparingly)
    # ------------------------------------------------------------------

    def search_serp(self, query: str, max_results: int = 5) -> list[dict]:
        """Search via SerpAPI (Google)."""
        if not self.budget.can_use("serp_api"):
            _log.warning("SerpAPI daily budget exhausted — skipping")
            return _EMPTY

        api_key = self.config.get("apis", "serpapi", "api_key", default="")
        if not api_key or api_key.startswith("${"):
            _log.warning("SerpAPI key not configured")
            return _EMPTY

        try:
            from serpapi import GoogleSearch

            _log.info("SerpAPI search: %r (max_results=%d)", query, max_results)
            params = {
                "q": query,
                "api_key": api_key,
                "num": max_results,
                "engine": "google",
            }
            search = GoogleSearch(params)
            raw = search.get_dict()

            api_error = raw.get("error", "")
            if api_error:
                _log.warning("SerpAPI error: %s", api_error)
                return _EMPTY

            self.budget.use("serp_api")
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "snippet": _truncate(r.get("snippet", "")),
                    "published_date": r.get("date", ""),
                }
                for r in raw.get("organic_results", [])[:max_results]
            ]

        except Exception as exc:
            _log.error("SerpAPI search failed: %s", exc)
            return _EMPTY

    # ------------------------------------------------------------------
    # Main search  (Tavily → DuckDuckGo → SerpAPI)
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Execute a search using the best available provider.

        Tries providers in priority order: Tavily → DuckDuckGo → SerpAPI.
        Returns as soon as a provider returns results.
        """
        results = self.search_tavily(query, max_results)
        if results:
            return results

        results = self.search_duckduckgo(query, max_results)
        if results:
            return results

        return self.search_serp(query, max_results)

    # ------------------------------------------------------------------
    # Article reading
    # ------------------------------------------------------------------

    def read_article(self, url: str) -> dict:
        """Fetch full article text from a URL using newspaper3k.

        Returns a dict with keys: title, text (≤3 000 chars), authors, published_date.
        Always returns a dict — never raises.
        """
        if not self.budget.can_use("web_scrape"):
            _log.warning("Web-scrape budget exhausted — cannot fetch article")
            return {
                "error": "Budget exhausted",
                "title": "",
                "text": "",
                "authors": [],
                "published_date": "",
            }

        _log.info("Fetching article: %s", url)
        try:
            from newspaper import Article

            article = Article(url)
            article.download()
            article.parse()
            self.budget.use("web_scrape")

            return {
                "title": article.title or "",
                "text": (article.text or "")[:3000],
                "authors": article.authors or [],
                "published_date": (
                    str(article.publish_date) if article.publish_date else ""
                ),
            }

        except Exception as exc:
            _log.warning("Article fetch failed for %s: %s", url, exc)
            return {
                "error": str(exc),
                "title": "",
                "text": "",
                "authors": [],
                "published_date": "",
            }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def search_stock_news(self, symbol: str, days: int = 3) -> list[dict]:
        """Search for news about an Indian stock on NSE.

        Automatically appends 'NSE India stock news' to the symbol query.
        """
        query = f"{symbol} NSE India stock news"
        _log.info("Stock news search: symbol=%s, days=%d", symbol, days)
        return self.search(query)

    def search_sector_news(self, sector: str) -> list[dict]:
        """Search for Indian sector news."""
        year = datetime.now().year
        query = f"Indian {sector} sector stock market news {year}"
        _log.info("Sector news search: %s", sector)
        return self.search(query)
