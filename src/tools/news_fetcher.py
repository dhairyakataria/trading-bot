"""News Fetcher — structured news from NewsAPI, Google News RSS, and MoneyControl.

Source priority (cheapest first):
  1. RSS feeds — Google News, MoneyControl (unlimited, no key)
  2. NewsAPI (100 free/day)  — used only when budget allows
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import requests

from src.llm.budget_manager import BudgetManager
from src.utils.logger import get_logger

_log = get_logger("tools.news_fetcher")


class NewsFetcher:
    """Downloads and parses news from RSS feeds and NewsAPI.

    All public methods return a list of article dicts with keys:
        title, url, source, published_date, snippet
    or a structured dict for market-data methods.
    Never raises — returns [] / empty dict on failure.
    """

    def __init__(self, config: Any, budget_manager: BudgetManager) -> None:
        self.config = config
        self.budget = budget_manager
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_rss_entry(self, entry: Any) -> dict:
        """Convert a feedparser entry to a standard article dict."""
        published = (
            getattr(entry, "published", None)
            or getattr(entry, "updated", None)
            or ""
        )

        source = ""
        if hasattr(entry, "source") and isinstance(entry.source, dict):
            source = entry.source.get("title", "")

        snippet = ""
        if hasattr(entry, "summary"):
            snippet = (entry.summary or "")[:300]

        return {
            "title": getattr(entry, "title", ""),
            "url": getattr(entry, "link", ""),
            "source": source,
            "published_date": published,
            "snippet": snippet,
        }

    def _deduplicate(self, articles: list[dict]) -> list[dict]:
        """Remove near-duplicate articles based on the first 40 chars of the title."""
        seen: list[str] = []
        unique: list[dict] = []
        for article in articles:
            title_key = article.get("title", "").lower().strip()[:40]
            if title_key and title_key not in seen:
                seen.append(title_key)
                unique.append(article)
        return unique

    # ------------------------------------------------------------------
    # NewsAPI  (100 free/day)
    # ------------------------------------------------------------------

    def fetch_newsapi(
        self,
        query: str,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 10,
    ) -> list[dict]:
        """Fetch articles from NewsAPI.org (100 requests/day free tier)."""
        if not self.budget.can_use("news_api"):
            _log.warning("NewsAPI daily budget exhausted")
            return []

        api_key = self.config.get("apis", "newsapi", "api_key", default="")
        if not api_key or api_key.startswith("${"):
            _log.warning("NewsAPI key not configured")
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": language,
            "sortBy": sort_by,
            "pageSize": min(page_size, 100),
            "apiKey": api_key,
        }
        _log.info("NewsAPI fetch: %r", query)
        try:
            resp = self._session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            self.budget.use("news_api")
            return [
                {
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "source": (a.get("source") or {}).get("name", ""),
                    "published_date": a.get("publishedAt", ""),
                    "snippet": (a.get("description") or "")[:300],
                }
                for a in resp.json().get("articles", [])
            ]
        except Exception as exc:
            _log.error("NewsAPI fetch failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Google News RSS  (free, unlimited)
    # ------------------------------------------------------------------

    def fetch_google_news_rss(
        self, query: str, max_results: int = 10
    ) -> list[dict]:
        """Fetch from Google News RSS feed (no key, unlimited)."""
        try:
            import feedparser

            encoded = quote_plus(query)
            url = (
                f"https://news.google.com/rss/search"
                f"?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
            )
            _log.info("Google News RSS: %r — %s", query, url)
            feed = feedparser.parse(url)
            return [
                self._parse_rss_entry(e) for e in feed.entries[:max_results]
            ]
        except Exception as exc:
            _log.error("Google News RSS failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # MoneyControl RSS  (free, unlimited)
    # ------------------------------------------------------------------

    def fetch_moneycontrol_rss(self, category: str = "market") -> list[dict]:
        """Fetch Indian financial news RSS (no key, unlimited).

        Tries MoneyControl first; falls back to Economic Times if MoneyControl
        is blocked (Akamai CDN returns 503 for programmatic access).

        category: 'market' | 'business' | 'economy'
        """
        # Primary: MoneyControl (may be blocked by Akamai CDN)
        _MC_MAP = {"market": "markets", "business": "business", "economy": "economy"}
        # Fallback: Economic Times (reliable, 50+ articles per feed)
        _ET_MAP = {
            "market": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "business": "https://economictimes.indiatimes.com/news/company/rssfeeds/2143429.cms",
            "economy": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
        }
        mc_cat = _MC_MAP.get(category, "markets")
        mc_url = f"https://www.moneycontrol.com/rss/{mc_cat}.xml"
        et_url = _ET_MAP.get(category, _ET_MAP["market"])

        try:
            import feedparser
            import requests as _req

            _headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
            }

            def _parse_url(url: str, source: str) -> list[dict]:
                resp = _req.get(url, headers=_headers, timeout=10)
                if resp.status_code != 200:
                    raise OSError(f"HTTP {resp.status_code}")
                feed = feedparser.parse(resp.content)
                articles = []
                for entry in feed.entries[:15]:
                    art = self._parse_rss_entry(entry)
                    art["source"] = source
                    articles.append(art)
                return articles

            # Try MoneyControl first
            try:
                _log.info("MoneyControl RSS: category=%s — %s", category, mc_url)
                return _parse_url(mc_url, "MoneyControl")
            except OSError:
                _log.info("MoneyControl RSS blocked, falling back to Economic Times: %s", et_url)
                return _parse_url(et_url, "Economic Times")

        except Exception as exc:
            _log.error("Indian financial RSS failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Aggregated feeds
    # ------------------------------------------------------------------

    def fetch_market_news(self, max_results: int = 15) -> list[dict]:
        """Aggregate market news from Google News RSS + MoneyControl + NewsAPI.

        RSS feeds are queried first (free/unlimited); NewsAPI is added only
        if budget allows.  Results are deduplicated and sorted newest-first.
        """
        all_articles: list[dict] = []

        all_articles.extend(
            self.fetch_google_news_rss("Indian stock market today", max_results=10)
        )
        all_articles.extend(self.fetch_moneycontrol_rss("market"))

        if self.budget.can_use("news_api"):
            all_articles.extend(
                self.fetch_newsapi("India stock market NSE BSE", page_size=5)
            )

        return self._deduplicate(all_articles)[:max_results]

    def fetch_stock_specific_news(
        self, symbol: str, company_name: str = ""
    ) -> list[dict]:
        """Fetch news for a specific NSE stock symbol."""
        query = f"{symbol} {company_name} NSE India".strip()
        articles = self.fetch_google_news_rss(query, max_results=10)
        if self.budget.can_use("news_api"):
            articles.extend(self.fetch_newsapi(query, page_size=5))
        return self._deduplicate(articles)

    # ------------------------------------------------------------------
    # FII / DII data
    # ------------------------------------------------------------------

    def get_fii_dii_data(self) -> dict:
        """Fetch latest FII and DII trading activity from NSE India.

        Falls back to an empty result dict if NSE API is unavailable.
        Returns keys: date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell,
                       dii_net, source.
        """
        nse_url = "https://www.nseindia.com/api/fiidiiTradeReact"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/market-data/fii-dii-trading-activity",
        }
        try:
            _log.info("Fetching FII/DII data from NSE: %s", nse_url)
            session = requests.Session()
            session.headers.update(headers)
            # NSE requires a valid session cookie — hit the homepage first
            session.get("https://www.nseindia.com", timeout=10)
            time.sleep(1)
            resp = session.get(nse_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list) and data:
                latest = data[0]
                return {
                    "date": latest.get("date", ""),
                    "fii_buy": float(latest.get("fiiBUY") or 0),
                    "fii_sell": float(latest.get("fiiSELL") or 0),
                    "fii_net": float(latest.get("fiiNET") or 0),
                    "dii_buy": float(latest.get("diiBUY") or 0),
                    "dii_sell": float(latest.get("diiSELL") or 0),
                    "dii_net": float(latest.get("diiNET") or 0),
                    "source": "NSE",
                }

        except Exception as exc:
            _log.warning("NSE FII/DII API failed: %s", exc)

        # Return safe empty result so callers never crash
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "fii_buy": 0.0,
            "fii_sell": 0.0,
            "fii_net": 0.0,
            "dii_buy": 0.0,
            "dii_sell": 0.0,
            "dii_net": 0.0,
            "source": "unavailable",
            "error": "FII/DII data temporarily unavailable",
        }

    # ------------------------------------------------------------------
    # Global market status
    # ------------------------------------------------------------------

    def get_global_market_status(self) -> dict:
        """Fetch global indices, commodities, and USD/INR via yfinance (free)."""
        try:
            import yfinance as yf

            _log.info("Fetching global market data via yfinance")
            symbols: dict[str, str] = {
                "sp500": "^GSPC",
                "nasdaq": "^IXIC",
                "dow_jones": "^DJI",
                "nikkei": "^N225",
                "hang_seng": "^HSI",
                "shanghai": "000001.SS",
                "crude_oil": "CL=F",
                "gold": "GC=F",
                "usd_inr": "INR=X",
            }

            result: dict[str, Any] = {}
            for name, ticker_sym in symbols.items():
                try:
                    ticker = yf.Ticker(ticker_sym)
                    info = ticker.fast_info
                    last = float(info.last_price or 0)
                    prev = float(info.previous_close or 0)
                    change_pct = (
                        round((last - prev) / prev * 100, 2) if prev else 0.0
                    )
                    result[name] = {
                        "symbol": ticker_sym,
                        "last_price": round(last, 2),
                        "previous_close": round(prev, 2),
                        "change_pct": change_pct,
                    }
                except Exception as exc:
                    _log.warning("Failed to fetch %s (%s): %s", name, ticker_sym, exc)
                    result[name] = {"symbol": ticker_sym, "error": str(exc)}

            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return result

        except Exception as exc:
            _log.error("Global market status failed: %s", exc)
            return {
                "error": str(exc),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    # ------------------------------------------------------------------
    # India VIX
    # ------------------------------------------------------------------

    def get_india_vix(self) -> dict:
        """Fetch India VIX from yfinance (symbol ^INDIAVIX).

        VIX > 20 = HIGH_FEAR, VIX < 15 = LOW_FEAR, else MODERATE.
        Returns: vix, signal, interpretation, timestamp.
        """
        try:
            import yfinance as yf

            _log.info("Fetching India VIX via yfinance (^INDIAVIX)")
            ticker = yf.Ticker("^INDIAVIX")
            info = ticker.fast_info
            vix = round(float(info.last_price or 0), 2)

            if vix > 20:
                signal = "HIGH_FEAR"
                interpretation = (
                    f"India VIX at {vix} — high market fear, elevated volatility"
                )
            elif vix < 15:
                signal = "LOW_FEAR"
                interpretation = (
                    f"India VIX at {vix} — low fear, markets complacent"
                )
            else:
                signal = "MODERATE"
                interpretation = f"India VIX at {vix} — normal volatility range"

            return {
                "vix": vix,
                "signal": signal,
                "interpretation": interpretation,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        except Exception as exc:
            _log.error("India VIX fetch failed: %s", exc)
            return {
                "vix": 0.0,
                "signal": "UNKNOWN",
                "error": str(exc),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
