"""Universe Agent — filters the stock universe to a tradable watchlist.

Layer 1 (weekly): refresh_index_constituents() — downloads/caches index lists.
Layer 2 (daily):  apply_daily_filters()         — applies 4 sequential filters.
Entry point:      get_active_watchlist()         — returns today's watchlist (DB-cached).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.database.db_manager import DatabaseManager
from src.database.models import WatchlistItem
from src.tools.technical_indicators import TechnicalIndicators
from src.utils.config import get_config
from src.utils.logger import get_logger

IST = ZoneInfo("Asia/Kolkata")
_log = get_logger("agents.universe_agent")

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_CONSTITUENTS_FILE = _DATA_DIR / "index_constituents.json"
_SECTOR_MAP_FILE = _DATA_DIR / "sector_map.json"

# NSE India archive CSV URLs (may require Referer header to bypass bot-check)
_NSE_INDEX_URLS: Dict[str, str] = {
    "NIFTY_50":           "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY_NEXT_50":      "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "NIFTY_MIDCAP_SELECT":"https://nsearchives.nseindia.com/content/indices/ind_niftymidcapselect_list.csv",
}

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Referer":         "https://www.nseindia.com",
    "Connection":      "keep-alive",
}

# --------------------------------------------------------------------------- #
# Hardcoded fallback index constituents (update quarterly)                     #
# --------------------------------------------------------------------------- #

_FALLBACK_CONSTITUENTS: Dict[str, List[str]] = {
    "NIFTY_50": [
        "ADANIENT",   "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT",  "AXISBANK",
        "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BHARTIARTL",  "BPCL",
        "BRITANNIA",  "CIPLA",      "COALINDIA",  "DIVISLAB",    "DRREDDY",
        "EICHERMOT",  "ETERNAL",    "GRASIM",     "HCLTECH",     "HDFCBANK",
        "HDFCLIFE",   "HEROMOTOCO", "HINDALCO",   "HINDUNILVR",  "ICICIBANK",
        "INDUSINDBK", "INFY",       "ITC",        "JSWSTEEL",    "KOTAKBANK",
        "LT",         "M&M",        "MARUTI",     "NESTLEIND",   "NTPC",
        "ONGC",       "POWERGRID",  "RELIANCE",   "SBILIFE",     "SBIN",
        "SHRIRAMFIN", "SUNPHARMA",  "TATACONSUM", "TATAMOTORS",  "TATASTEEL",
        "TCS",        "TECHM",      "TITAN",      "ULTRACEMCO",  "WIPRO",
    ],
    "NIFTY_NEXT_50": [
        "ABB",        "ADANIGREEN", "ADANITOTAL", "ALKEM",       "AMBUJACEM",
        "BANKBARODA", "BEL",        "BERGERPAINTS","BOSCHLTD",   "CANBK",
        "CHOLAFIN",   "COLPAL",     "DABUR",      "DLF",         "DMART",
        "FEDERALBNK", "GODREJCP",   "GODREJPROP", "HAVELLS",     "HAL",
        "ICICIPRULI", "ICICIGI",    "INDUSTOWER", "IRFC",        "JINDALSTEL",
        "LUPIN",      "MCDOWELL-N", "MOTHERSON",  "MPHASIS",     "NAUKRI",
        "NMDC",       "OFSS",       "PAGEIND",    "PERSISTENT",  "PIIND",
        "RECLTD",     "SIEMENS",    "SRF",        "TATAPOWER",   "TORNTPHARM",
        "TRENT",      "TVSMOTOR",   "UPL",        "VBL",         "VEDL",
        "VOLTAS",     "WHIRLPOOL",  "YESBANK",    "ZYDUSLIFE",   "PGHH",
    ],
    "NIFTY_MIDCAP_SELECT": [
        "ACC",        "APLAPOLLO",  "AUBANK",     "ABCAPITAL",   "ABFRL",
        "ASTRAL",     "BANDHANBNK", "BIKAJI",     "CANFINHOME",  "COFORGE",
        "CUB",        "CUMMINSIND", "DELHIVERY",  "ELGIEQUIP",   "ESCORTS",
        "EXIDEIND",   "GLENMARK",   "GMRAIRPORT", "GNFC",        "HFCL",
        "HINDPETRO",  "IDBI",       "IIFL",       "INDIANB",     "INDHOTEL",
        "JKCEMENT",   "JUBLFOOD",   "KARURVYSYA", "KPITTECH",    "LALPATHLAB",
        "LTIM",       "MAXHEALTH",  "MCX",        "MGL",         "MFSL",
        "NATIONALUM", "NHPC",       "POLICYBZR",  "RAYMOND",     "SAIL",
        "SBICARD",    "SCHAEFFLER", "SOLARINDS",  "SUNDARMFIN",  "SUPREMEIND",
        "SUZLON",     "TIINDIA",    "TORNTPOWER", "UNIONBANK",   "ZEEL",
        "OBEROIRLTY", "OLECTRA",
    ],
}

# --------------------------------------------------------------------------- #
# Hardcoded sector map (update quarterly; overridden by data/sector_map.json)  #
# --------------------------------------------------------------------------- #

_FALLBACK_SECTOR_MAP: Dict[str, str] = {
    # --- Nifty 50 ---
    "ADANIENT":   "DIVERSIFIED",       "ADANIPORTS":  "INFRASTRUCTURE",
    "APOLLOHOSP": "HEALTHCARE",        "ASIANPAINT":  "CONSUMER_GOODS",
    "AXISBANK":   "BANKING",           "BAJAJ-AUTO":  "AUTO",
    "BAJAJFINSV": "FINANCIAL_SERVICES","BAJFINANCE":  "FINANCIAL_SERVICES",
    "BHARTIARTL": "TELECOM",           "BPCL":        "OIL_GAS",
    "BRITANNIA":  "FMCG",              "CIPLA":       "PHARMA",
    "COALINDIA":  "METALS_MINING",     "DIVISLAB":    "PHARMA",
    "DRREDDY":    "PHARMA",            "EICHERMOT":   "AUTO",
    "ETERNAL":    "TECH",              "GRASIM":      "DIVERSIFIED",
    "HCLTECH":    "IT",                "HDFCBANK":    "BANKING",
    "HDFCLIFE":   "INSURANCE",         "HEROMOTOCO":  "AUTO",
    "HINDALCO":   "METALS_MINING",     "HINDUNILVR":  "FMCG",
    "ICICIBANK":  "BANKING",           "INDUSINDBK":  "BANKING",
    "INFY":       "IT",                "ITC":         "FMCG",
    "JSWSTEEL":   "METALS_MINING",     "KOTAKBANK":   "BANKING",
    "LT":         "INFRASTRUCTURE",    "M&M":         "AUTO",
    "MARUTI":     "AUTO",              "NESTLEIND":   "FMCG",
    "NTPC":       "POWER",             "ONGC":        "OIL_GAS",
    "POWERGRID":  "POWER",             "RELIANCE":    "DIVERSIFIED",
    "SBILIFE":    "INSURANCE",         "SBIN":        "BANKING",
    "SHRIRAMFIN": "FINANCIAL_SERVICES","SUNPHARMA":   "PHARMA",
    "TATACONSUM": "FMCG",              "TATAMOTORS":  "AUTO",
    "TATASTEEL":  "METALS_MINING",     "TCS":         "IT",
    "TECHM":      "IT",                "TITAN":       "CONSUMER_GOODS",
    "ULTRACEMCO": "CEMENT",            "WIPRO":       "IT",
    # --- Nifty Next 50 ---
    "ABB":        "CAPITAL_GOODS",     "ADANIGREEN":  "POWER",
    "ADANITOTAL": "OIL_GAS",           "ALKEM":       "PHARMA",
    "AMBUJACEM":  "CEMENT",            "BANKBARODA":  "BANKING",
    "BEL":        "DEFENSE",           "BERGERPAINTS":"CONSUMER_GOODS",
    "BOSCHLTD":   "AUTO_ANCILLARY",    "CANBK":       "BANKING",
    "CHOLAFIN":   "FINANCIAL_SERVICES","COLPAL":      "FMCG",
    "DABUR":      "FMCG",              "DLF":         "REAL_ESTATE",
    "DMART":      "RETAIL",            "FEDERALBNK":  "BANKING",
    "GODREJCP":   "FMCG",              "GODREJPROP":  "REAL_ESTATE",
    "HAVELLS":    "CAPITAL_GOODS",     "HAL":         "DEFENSE",
    "ICICIPRULI": "INSURANCE",         "ICICIGI":     "INSURANCE",
    "INDUSTOWER": "TELECOM",           "IRFC":        "FINANCIAL_SERVICES",
    "JINDALSTEL": "METALS_MINING",     "LUPIN":       "PHARMA",
    "MCDOWELL-N": "FMCG",              "MOTHERSON":   "AUTO_ANCILLARY",
    "MPHASIS":    "IT",                "NAUKRI":      "TECH",
    "NMDC":       "METALS_MINING",     "OFSS":        "IT",
    "PAGEIND":    "TEXTILE",           "PERSISTENT":  "IT",
    "PIIND":      "AGRO_CHEMICALS",    "RECLTD":      "FINANCIAL_SERVICES",
    "SIEMENS":    "CAPITAL_GOODS",     "SRF":         "CHEMICALS",
    "TATAPOWER":  "POWER",             "TORNTPHARM":  "PHARMA",
    "TRENT":      "RETAIL",            "TVSMOTOR":    "AUTO",
    "UPL":        "AGRO_CHEMICALS",    "VBL":         "FMCG",
    "VEDL":       "METALS_MINING",     "VOLTAS":      "CAPITAL_GOODS",
    "WHIRLPOOL":  "CONSUMER_GOODS",    "YESBANK":     "BANKING",
    "ZYDUSLIFE":  "PHARMA",            "PGHH":        "FMCG",
    # --- Nifty Midcap Select ---
    "ACC":        "CEMENT",            "APLAPOLLO":   "METALS_MINING",
    "AUBANK":     "BANKING",           "ABCAPITAL":   "FINANCIAL_SERVICES",
    "ABFRL":      "TEXTILE",           "ASTRAL":      "CAPITAL_GOODS",
    "BANDHANBNK": "BANKING",           "BIKAJI":      "FMCG",
    "CANFINHOME": "FINANCIAL_SERVICES","COFORGE":     "IT",
    "CUB":        "BANKING",           "CUMMINSIND":  "CAPITAL_GOODS",
    "DELHIVERY":  "LOGISTICS",         "ELGIEQUIP":   "CAPITAL_GOODS",
    "ESCORTS":    "AUTO",              "EXIDEIND":    "AUTO_ANCILLARY",
    "GLENMARK":   "PHARMA",            "GMRAIRPORT":  "INFRASTRUCTURE",
    "GNFC":       "CHEMICALS",         "HFCL":        "TELECOM",
    "HINDPETRO":  "OIL_GAS",           "IDBI":        "BANKING",
    "IIFL":       "FINANCIAL_SERVICES","INDIANB":     "BANKING",
    "INDHOTEL":   "HOSPITALITY",       "JKCEMENT":    "CEMENT",
    "JUBLFOOD":   "RETAIL",            "KARURVYSYA":  "BANKING",
    "KPITTECH":   "IT",                "LALPATHLAB":  "HEALTHCARE",
    "LTIM":       "IT",                "MAXHEALTH":   "HEALTHCARE",
    "MCX":        "FINANCIAL_SERVICES","MGL":         "OIL_GAS",
    "MFSL":       "INSURANCE",         "NATIONALUM":  "METALS_MINING",
    "NHPC":       "POWER",             "POLICYBZR":   "INSURANCE",
    "RAYMOND":    "TEXTILE",           "SAIL":        "METALS_MINING",
    "SBICARD":    "FINANCIAL_SERVICES","SCHAEFFLER":  "AUTO_ANCILLARY",
    "SOLARINDS":  "CAPITAL_GOODS",     "SUNDARMFIN":  "FINANCIAL_SERVICES",
    "SUPREMEIND": "CAPITAL_GOODS",     "SUZLON":      "POWER",
    "TIINDIA":    "AUTO_ANCILLARY",    "TORNTPOWER":  "POWER",
    "UNIONBANK":  "BANKING",           "ZEEL":        "MEDIA",
    "OBEROIRLTY": "REAL_ESTATE",       "OLECTRA":     "AUTO",
}


# --------------------------------------------------------------------------- #
# UniverseAgent                                                                #
# --------------------------------------------------------------------------- #

class UniverseAgent:
    """Builds and maintains a curated watchlist of stocks for swing trading.

    Layer 1 (weekly): Downloads index constituents from NSE (with hardcoded
    fallback) and stores them in ``data/index_constituents.json``.

    Layer 2 (daily @ 8 AM): Fetches 50-day OHLCV data per stock and applies
    4 sequential filters — price, volume, trend (EMA-50), and ATR volatility.

    Usage::

        agent = UniverseAgent(broker=broker_client, db=db_manager)
        watchlist = agent.get_active_watchlist()
        watchlist = agent.ensure_held_stocks_in_watchlist(watchlist, held_symbols)
        dist = agent.get_sector_distribution(watchlist)
    """

    # Delay between broker API calls — patch to 0 in unit tests
    _API_DELAY: float = 0.3

    def __init__(
        self,
        broker: Any = None,
        db: Optional[DatabaseManager] = None,
        config: Any = None,
    ) -> None:
        self._broker = broker
        self._db = db
        self._config = config if config is not None else get_config()
        self._ti = TechnicalIndicators()
        self._sector_map: Dict[str, str] = self._load_sector_map()
        # In-memory OHLCV cache; reset each time apply_daily_filters is called.
        self._hist_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ #
    # Layer 1 — Index Constituent Management (runs WEEKLY)                #
    # ------------------------------------------------------------------ #

    def refresh_index_constituents(self) -> dict:
        """Download index lists from NSE and persist to data/index_constituents.json.

        Returns a summary dict::

            {"total_universe": 152, "nifty_50": 50, "next_50": 50, "midcap_select": 52}
        """
        constituents: Dict[str, List[str]] = {}

        for index_name in ("NIFTY_50", "NIFTY_NEXT_50", "NIFTY_MIDCAP_SELECT"):
            fetched = self._fetch_index_from_nse(index_name)
            if fetched:
                constituents[index_name] = fetched
                _log.info(
                    "Fetched %d symbols for %s from NSE India",
                    len(fetched), index_name,
                )
            else:
                fallback = _FALLBACK_CONSTITUENTS.get(index_name, [])
                constituents[index_name] = fallback
                _log.info(
                    "Using hardcoded fallback for %s (%d symbols)",
                    index_name, len(fallback),
                )

        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(IST).strftime("%Y-%m-%d"),
            "constituents": constituents,
        }
        with _CONSTITUENTS_FILE.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        all_symbols = {s for syms in constituents.values() for s in syms}
        result = {
            "total_universe":  len(all_symbols),
            "nifty_50":        len(constituents.get("NIFTY_50", [])),
            "next_50":         len(constituents.get("NIFTY_NEXT_50", [])),
            "midcap_select":   len(constituents.get("NIFTY_MIDCAP_SELECT", [])),
        }
        _log.info("Index constituents refreshed: %s", result)
        return result

    def get_base_universe(self) -> List[str]:
        """Return the deduplicated ~150-symbol universe, minus blacklisted stocks.

        Loads from ``data/index_constituents.json`` if present; falls back to
        the hardcoded lists otherwise.
        """
        constituents = self._load_constituents()
        seen: set = set()
        ordered: List[str] = []
        for index_name in ("NIFTY_50", "NIFTY_NEXT_50", "NIFTY_MIDCAP_SELECT"):
            for sym in constituents.get(index_name, []):
                if sym not in seen:
                    seen.add(sym)
                    ordered.append(sym)

        blacklisted: set = set(
            self._config.get("universe", "blacklisted_stocks") or []
        )
        base = [s for s in ordered if s not in blacklisted]
        _log.info(
            "Base universe: %d symbols (%d blacklisted)", len(base), len(blacklisted)
        )
        return base

    # ------------------------------------------------------------------ #
    # Layer 2 — Daily Filtering (runs DAILY at 8:00 AM)                  #
    # ------------------------------------------------------------------ #

    def apply_daily_filters(self, base_universe: List[str]) -> List[dict]:
        """Apply 4 sequential filters to *base_universe*; return passing stocks.

        Filters (applied in order — early rejection avoids unnecessary checks):
        1. Price range  : 50 ≤ current_price ≤ 5000 INR
        2. Volume       : 20-day avg daily traded value ≥ 10 Cr INR
        3. Trend        : current_price above 50-day EMA
        4. ATR          : 14-day ATR% between 1.5% and 6.0%

        Each stock's historical data is fetched once (cached in-memory) and
        the broker API call is followed by a short rate-limit delay.

        Returns a list of dicts with keys:
            symbol, price, avg_volume_cr, atr_pct, ema_50,
            price_above_ema_pct, sector, index, added_date
        """
        # Reset cache for the new scan day
        self._hist_cache = {}

        today = datetime.now(IST)
        # Fetch ~100 calendar days to get ≥70 trading days (needed for EMA-50)
        from_date = (today - timedelta(days=100)).strftime("%Y-%m-%d 09:15")
        to_date = today.strftime("%Y-%m-%d 15:30")

        min_price: float = float(
            self._config.get("trading", "min_stock_price") or 50
        )
        max_price: float = float(
            self._config.get("trading", "max_stock_price") or 5000
        )
        min_vol_cr: float = float(
            self._config.get("trading", "min_volume_cr") or 10
        )
        min_atr_pct: float = 1.5
        max_atr_pct: float = 6.0

        rejected: Dict[str, int] = {
            "price": 0, "volume": 0, "trend": 0, "atr": 0, "data_error": 0,
        }
        passed: List[dict] = []
        added_date = today.strftime("%Y-%m-%d")

        for symbol in base_universe:
            try:
                df = self._get_hist_data(symbol, from_date, to_date)
                if df.empty or len(df) < 20:
                    _log.debug(
                        "Skipping %s: insufficient data (%d rows)",
                        symbol, len(df),
                    )
                    rejected["data_error"] += 1
                    continue

                current_price = float(df["close"].iloc[-1])

                # ---- Filter 1: Price range --------------------------------
                if not (min_price <= current_price <= max_price):
                    rejected["price"] += 1
                    continue

                # ---- Filter 2: Volume (20-day avg traded value in Cr) -----
                recent = df.tail(20)
                daily_tv = recent["close"] * recent["volume"]
                avg_tv_cr = float(daily_tv.mean()) / 1e7  # rupees → crores
                if avg_tv_cr < min_vol_cr:
                    rejected["volume"] += 1
                    continue

                # ---- Filter 3: Trend — price above 50-day EMA ------------
                ema_result = self._ti.calculate_ema(df, periods=[50])
                if "error" in ema_result or ema_result.get("ema_50") is None:
                    rejected["trend"] += 1
                    continue
                ema_50: float = float(ema_result["ema_50"])
                if current_price <= ema_50:
                    rejected["trend"] += 1
                    continue

                # ---- Filter 4: ATR volatility 1.5% – 6.0% ----------------
                atr_result = self._ti.calculate_atr(df, period=14)
                if "error" in atr_result:
                    rejected["atr"] += 1
                    continue
                atr_pct: float = float(atr_result.get("atr_pct", 0.0))
                if not (min_atr_pct <= atr_pct <= max_atr_pct):
                    rejected["atr"] += 1
                    continue

                # ---- All filters passed -----------------------------------
                price_above_ema_pct = (
                    round((current_price - ema_50) / ema_50 * 100, 2)
                    if ema_50 > 0 else 0.0
                )
                passed.append({
                    "symbol":             symbol,
                    "price":              round(current_price, 2),
                    "avg_volume_cr":      round(avg_tv_cr, 2),
                    "atr_pct":            atr_pct,
                    "ema_50":             ema_50,
                    "price_above_ema_pct": price_above_ema_pct,
                    "sector":             self._sector_map.get(symbol, "UNKNOWN"),
                    "index":              self._get_index_for_symbol(symbol),
                    "added_date":         added_date,
                })

            except Exception as exc:  # noqa: BLE001
                _log.warning("Error processing %s: %s — skipping", symbol, exc)
                rejected["data_error"] += 1

        _log.info(
            "Filter results: %d/%d passed | rejected — "
            "price:%d volume:%d trend:%d atr:%d errors:%d",
            len(passed), len(base_universe),
            rejected["price"], rejected["volume"],
            rejected["trend"], rejected["atr"], rejected["data_error"],
        )
        return passed

    # ------------------------------------------------------------------ #
    # Watchlist Management                                                 #
    # ------------------------------------------------------------------ #

    def get_active_watchlist(self) -> List[dict]:
        """Return today's filtered watchlist (from DB cache or freshly computed).

        Steps:
        1. If today's watchlist exists in DB → return it immediately.
        2. Otherwise run apply_daily_filters(), save to DB, and return.
        3. Sort by ATR% descending (higher ATR = better swing candidate).
        4. Cap at 50 stocks.
        """
        today = datetime.now(IST).strftime("%Y-%m-%d")

        if self._db is not None:
            cached = self._load_cached_watchlist(today)
            if cached:
                _log.info(
                    "Returning cached watchlist for %s (%d stocks)", today, len(cached)
                )
                return cached

        _log.info("No cached watchlist for %s — running daily filters …", today)
        base = self.get_base_universe()
        filtered = self.apply_daily_filters(base)

        # Sort by ATR% descending; cap at 50
        filtered.sort(key=lambda x: x.get("atr_pct") or 0.0, reverse=True)
        filtered = filtered[:50]

        if self._db is not None:
            self._persist_watchlist(today, filtered)

        _log.info("Active watchlist ready: %d stocks", len(filtered))
        return filtered

    def ensure_held_stocks_in_watchlist(
        self,
        watchlist: List[dict],
        held_symbols: List[str],
    ) -> List[dict]:
        """Force-add currently held stocks that were filtered out.

        Held positions must always be monitored for exit signals regardless of
        whether they pass today's entry filters.  Force-added stocks carry a
        ``"reason": "HELD_IN_PORTFOLIO"`` flag.
        """
        present = {item["symbol"] for item in watchlist}
        additions: List[dict] = []

        for symbol in held_symbols:
            if symbol in present:
                continue  # already in watchlist — no action needed

            entry: dict = {
                "symbol":             symbol,
                "price":              None,
                "avg_volume_cr":      None,
                "atr_pct":            None,
                "ema_50":             None,
                "price_above_ema_pct": None,
                "sector":             self._sector_map.get(symbol, "UNKNOWN"),
                "index":              self._get_index_for_symbol(symbol),
                "added_date":         datetime.now(IST).strftime("%Y-%m-%d"),
                "reason":             "HELD_IN_PORTFOLIO",
            }

            # Best-effort LTP enrichment so the exit agent has a price
            if self._broker is not None:
                try:
                    ltp_data = self._broker.get_ltp(symbol)
                    entry["price"] = ltp_data.get("ltp")
                except Exception as exc:  # noqa: BLE001
                    _log.debug("Could not fetch LTP for held stock %s: %s", symbol, exc)

            additions.append(entry)
            _log.info("Force-adding held stock %s to watchlist", symbol)

        return watchlist + additions

    def get_sector_distribution(self, watchlist: List[dict]) -> dict:
        """Return a count of watchlist stocks per sector, sorted descending.

        Example::

            {"BANKING": 12, "IT": 8, "PHARMA": 5, ...}
        """
        dist: Dict[str, int] = {}
        for item in watchlist:
            sector = item.get("sector") or "UNKNOWN"
            dist[sector] = dist.get(sector, 0) + 1
        return dict(sorted(dist.items(), key=lambda kv: kv[1], reverse=True))

    # ------------------------------------------------------------------ #
    # Compatibility shims for the original stub interface                  #
    # ------------------------------------------------------------------ #

    def build_universe(self) -> List[str]:
        """Return tradable symbols (delegates to get_active_watchlist)."""
        return [item["symbol"] for item in self.get_active_watchlist()]

    def apply_filters(self, symbols: List[str]) -> List[str]:
        """Apply filters and return passing symbols (delegates to apply_daily_filters)."""
        return [item["symbol"] for item in self.apply_daily_filters(symbols)]

    def get_index_constituents(self, index_name: str) -> List[str]:
        """Return stored symbols for a given index name."""
        return self._load_constituents().get(index_name, [])

    def is_blacklisted(self, symbol: str) -> bool:
        """Return True if *symbol* is on the configured blacklist."""
        blacklisted = set(self._config.get("universe", "blacklisted_stocks") or [])
        return symbol in blacklisted

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_hist_data(
        self, symbol: str, from_date: str, to_date: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV with in-memory cache; sleeps after each real API call."""
        if symbol in self._hist_cache:
            return self._hist_cache[symbol]

        empty = pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        if self._broker is None:
            return empty

        try:
            df = self._broker.get_historical_data(
                symbol, "ONE_DAY", from_date, to_date
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to fetch data for %s: %s", symbol, exc)
            df = empty
        finally:
            time.sleep(self._API_DELAY)

        self._hist_cache[symbol] = df
        return df

    def _fetch_index_from_nse(self, index_name: str) -> Optional[List[str]]:
        """Attempt to download index constituents from NSE India archive CSV."""
        url = _NSE_INDEX_URLS.get(index_name)
        if not url:
            return None
        try:
            resp = requests.get(url, headers=_NSE_HEADERS, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            # NSE CSV has a "Symbol" column (sometimes "SYMBOL")
            symbol_col = next(
                (c for c in df.columns if c.strip().lower() == "symbol"), None
            )
            if symbol_col is None:
                _log.warning(
                    "NSE CSV for %s has no 'Symbol' column — columns: %s",
                    index_name, list(df.columns),
                )
                return None
            symbols = df[symbol_col].dropna().str.strip().tolist()
            return symbols
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "NSE fetch failed for %s (%s) — will use fallback",
                index_name, exc,
            )
            return None

    def _load_constituents(self) -> Dict[str, List[str]]:
        """Load index constituents from JSON file, or return hardcoded fallback."""
        if _CONSTITUENTS_FILE.exists():
            try:
                with _CONSTITUENTS_FILE.open(encoding="utf-8") as fh:
                    data = json.load(fh)
                return data.get("constituents", {})
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "Could not read %s (%s) — using hardcoded fallback",
                    _CONSTITUENTS_FILE, exc,
                )
        else:
            _log.info(
                "No index constituents file found at %s — using hardcoded fallback",
                _CONSTITUENTS_FILE,
            )
        return _FALLBACK_CONSTITUENTS

    def _load_cached_watchlist(self, date: str) -> List[dict]:
        """Return today's watchlist from DB if it exists; else empty list."""
        try:
            items = self._db.get_latest_watchlist()  # type: ignore[union-attr]
            if not items:
                return []
            if items[0].date != date:
                return []
            return [self._item_to_dict(item) for item in items]
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to load watchlist from DB: %s", exc)
            return []

    def _persist_watchlist(self, date: str, filtered: List[dict]) -> None:
        """Save the filtered watchlist to the database."""
        try:
            db_items = [
                WatchlistItem(
                    symbol=stock["symbol"],
                    date=date,
                    price=stock.get("price"),
                    avg_volume_cr=stock.get("avg_volume_cr"),
                    atr_pct=stock.get("atr_pct"),
                    ema_50=stock.get("ema_50"),
                    sector=stock.get("sector"),
                    in_index=stock.get("index"),
                )
                for stock in filtered
            ]
            self._db.save_watchlist(date, db_items)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            _log.error("Failed to save watchlist to DB: %s", exc)

    def _item_to_dict(self, item: WatchlistItem) -> dict:
        """Convert a WatchlistItem DB row to the output dict format."""
        d = item.to_dict()
        price = d.get("price")
        ema_50 = d.get("ema_50")
        d["price_above_ema_pct"] = (
            round((price - ema_50) / ema_50 * 100, 2)
            if price and ema_50 and ema_50 > 0
            else None
        )
        d["index"] = d.pop("in_index", None)
        d["added_date"] = d.pop("date", None)
        return d

    def _get_index_for_symbol(self, symbol: str) -> str:
        """Return the index name a symbol belongs to (Nifty 50 takes priority)."""
        constituents = self._load_constituents()
        for index_name in ("NIFTY_50", "NIFTY_NEXT_50", "NIFTY_MIDCAP_SELECT"):
            if symbol in constituents.get(index_name, []):
                return index_name
        return "UNKNOWN"

    def _load_sector_map(self) -> Dict[str, str]:
        """Load sector map from data/sector_map.json, falling back to built-in dict."""
        if _SECTOR_MAP_FILE.exists():
            try:
                with _SECTOR_MAP_FILE.open(encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "Could not read %s (%s) — using built-in sector map",
                    _SECTOR_MAP_FILE, exc,
                )
        return _FALLBACK_SECTOR_MAP
