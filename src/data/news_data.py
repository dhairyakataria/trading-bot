"""News Data — aggregates financial news from multiple sources."""
from __future__ import annotations

from typing import Any


class NewsDataFetcher:
    """Collects and normalises financial news for Indian equities.

    Responsibilities:
    - Fetch news from NewsAPI, Tavily, and SerpAPI
    - Deduplicate articles across sources
    - Filter articles by symbol, sector, or keyword
    - Return articles in a consistent dict schema
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def get_news_for_symbol(
        self, symbol: str, days: int = 7
    ) -> list[dict[str, str]]:
        """Return recent news articles related to the given stock symbol."""
        raise NotImplementedError

    def get_market_news(self, days: int = 1) -> list[dict[str, str]]:
        """Return broad Indian market news for the given period."""
        raise NotImplementedError

    def get_sector_news(
        self, sector: str, days: int = 3
    ) -> list[dict[str, str]]:
        """Return news articles filtered to a specific market sector."""
        raise NotImplementedError

    def deduplicate(
        self, articles: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Remove duplicate articles based on URL or title similarity."""
        raise NotImplementedError
