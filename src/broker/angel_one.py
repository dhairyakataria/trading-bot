"""Angel One SmartAPI broker integration wrapper.

Provides :class:`AngelOneClient` — a thread-safe, retry-aware wrapper around
the SmartAPI Python SDK for Indian equity swing trading (DELIVERY / CNC).

Usage::

    from src.broker.angel_one import AngelOneClient
    from src.utils.config import get_config

    client = AngelOneClient(get_config())
    client.login()
    ltp = client.get_ltp("TCS")
"""
from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyotp
import requests

from src.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_INSTRUMENT_MASTER_URL = (
    "https://margincalculator.angelbroking.com"
    "/OpenAPI_File/files/OpenAPIScripMaster.json"
)
_INSTRUMENT_MASTER_CACHE = _PROJECT_ROOT / "data" / "instrument_master.json"
_CACHE_TTL_HOURS = 24

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class BrokerAuthError(Exception):
    """Raised on authentication failure (bad credentials, TOTP mismatch, etc.)."""


class BrokerAPIError(Exception):
    """Raised when the broker API returns an error response."""


class OrderRejectedError(Exception):
    """Raised when an order is rejected by the exchange or broker."""


class RateLimitError(Exception):
    """Raised when the broker API rate-limit is hit (AG8001 / AB1019 / AB1021)."""


# ---------------------------------------------------------------------------
# Token-bucket rate limiter
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Token-bucket: allows up to *rate* calls per *period* seconds."""

    def __init__(self, rate: int = 10, period: float = 1.0) -> None:
        self._rate = rate
        self._period = period
        self._tokens: float = float(rate)
        self._last: float = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(
                self._rate,
                self._tokens + elapsed * self._rate / self._period,
            )
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) * self._period / self._rate
                time.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


# ---------------------------------------------------------------------------
# AngelOneClient
# ---------------------------------------------------------------------------


class AngelOneClient:
    """Thread-safe wrapper around the Angel One SmartAPI Python SDK.

    Responsibilities
    ----------------
    - Authenticate with client ID, API key, and TOTP; transparently re-login
      when the session expires.
    - Provide typed methods for quotes, historical OHLCV, portfolio, and orders.
    - Enforce a 10 req/s rate limit and retry on transient failures.
    - Cache the instrument master locally to avoid repeated downloads.
    - Never log sensitive data (passwords, API keys, TOTP codes).
    """

    _MAX_LOGIN_RETRIES: int = 3
    _LOGIN_RETRY_WAIT: float = 5.0   # seconds between login attempts
    _API_MAX_RETRIES: int = 3
    _API_BASE_WAIT: float = 10.0     # seconds before first retry (historical API throttles hard)

    def __init__(self, config: Any) -> None:
        """Initialise from *config* without logging in.

        Args:
            config: The application :class:`~src.utils.config.Config` object
                    (or any dict-like object) providing ``broker.angel_one.*``.
        """
        broker_cfg: dict = config["broker"]["angel_one"]

        self._client_id: str = broker_cfg["client_id"]
        self._password: str = broker_cfg["password"]
        self._api_key: str = broker_cfg["api_key"]
        self._totp_secret: str = broker_cfg["totp_secret"]
        self._default_exchange: str = broker_cfg.get("default_exchange", "NSE")

        # SmartConnect instance and session tokens
        self._smart: Any = None
        self._jwt_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._feed_token: Optional[str] = None
        self._session_expiry: Optional[datetime] = None

        # Thread-safety locks
        self._session_lock = threading.Lock()       # guards session refresh
        self._instrument_lock = threading.Lock()    # guards master download

        # Rate limiter: max 10 calls / second
        self._rate_limiter = _RateLimiter(rate=10, period=1.0)

        # Instrument / token caches (populated from master on first use)
        self._symbol_to_token: Dict[str, str] = {}   # "TCS"    → "11536"
        self._token_to_symbol: Dict[str, str] = {}   # "11536"  → "TCS-EQ"
        self._trading_symbol: Dict[str, str] = {}    # "TCS"    → "TCS-EQ"
        self._instrument_master: Optional[pd.DataFrame] = None

        logger.info(
            "AngelOneClient initialised for client_id=%s***",
            self._client_id[:3] if self._client_id else "?",
        )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def login(self) -> bool:
        """Log in to Angel One, retrying up to 3 times on failure.

        Returns:
            ``True`` on success.

        Raises:
            :class:`BrokerAuthError`: if all attempts fail.
        """
        wait = self._LOGIN_RETRY_WAIT
        for attempt in range(1, self._MAX_LOGIN_RETRIES + 1):
            try:
                return self._do_login()
            except BrokerAuthError as exc:
                if attempt == self._MAX_LOGIN_RETRIES:
                    raise
                logger.warning(
                    "Login attempt %d/%d failed: %s — retrying in %.0fs",
                    attempt, self._MAX_LOGIN_RETRIES, exc, wait,
                )
                time.sleep(wait)
        return False  # unreachable; silences type-checkers

    def _do_login(self) -> bool:
        """Single login attempt.

        Raises:
            :class:`BrokerAuthError`: on any failure.
        """
        try:
            from SmartApi import SmartConnect  # type: ignore[import]  # v1.3.x
        except ImportError:
            try:
                from smartapi import SmartConnect  # type: ignore[import]  # v1.5.x
            except ImportError as exc:
                raise BrokerAuthError(
                    "smartapi-python not installed. Run: pip install smartapi-python"
                ) from exc

        # Generate TOTP — never logged
        totp_code = pyotp.TOTP(self._totp_secret).now()
        logger.info("Attempting login for client_id=%s***", self._client_id[:3])

        smart = SmartConnect(api_key=self._api_key)
        try:
            data = smart.generateSession(
                clientCode=self._client_id,
                password=self._password,
                totp=totp_code,
            )
        except Exception as exc:
            raise BrokerAuthError(
                f"SmartAPI generateSession raised {type(exc).__name__}: {exc}"
            ) from exc

        if not data or not data.get("status"):
            msg = data.get("message", "no response") if data else "empty response"
            raise BrokerAuthError(f"Login rejected: {msg}")

        session = data["data"]
        self._smart = smart
        self._jwt_token = session.get("jwtToken", "")
        self._refresh_token = session.get("refreshToken", "")
        self._feed_token = session.get("feedToken", "")
        # Angel One sessions last ~8 hours; expire conservatively at 7h 55m
        self._session_expiry = datetime.now() + timedelta(hours=7, minutes=55)

        logger.info(
            "Login successful — session valid until %s",
            self._session_expiry.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return True

    def logout(self) -> None:
        """Terminate the broker session cleanly."""
        if self._smart is None:
            logger.debug("logout: no active session")
            return
        try:
            self._rate_limiter.acquire()
            self._smart.terminateSession(self._client_id)
            logger.info("Session terminated for client_id=%s***", self._client_id[:3])
        except Exception as exc:
            logger.warning("logout: terminateSession raised %s (ignoring)", exc)
        finally:
            self._smart = None
            self._jwt_token = None
            self._refresh_token = None
            self._feed_token = None
            self._session_expiry = None

    def is_authenticated(self) -> bool:
        """Return ``True`` if a non-expired session is active."""
        return (
            self._smart is not None
            and self._jwt_token is not None
            and self._session_expiry is not None
            and datetime.now() < self._session_expiry
        )

    def _ensure_session(self) -> None:
        """Re-login if the current session has expired (thread-safe)."""
        if self.is_authenticated():
            return
        with self._session_lock:
            # Double-checked locking: another thread may have refreshed already
            if not self.is_authenticated():
                logger.info("Session expired — refreshing login automatically")
                self.login()

    # ------------------------------------------------------------------
    # Internal API call helper
    # ------------------------------------------------------------------

    def _call_api(
        self,
        fn: Any,
        *args: Any,
        context: str = "",
        **kwargs: Any,
    ) -> Any:
        """Call a SmartConnect method with rate-limiting and exponential retry.

        Retries up to ``_API_MAX_RETRIES`` times on :class:`RateLimitError`
        or transient :class:`BrokerAPIError`.

        Args:
            fn:      Bound SmartConnect method to call.
            *args:   Positional arguments forwarded to *fn*.
            context: Human-readable description for log messages.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            The parsed response dict from the SmartAPI.

        Raises:
            :class:`BrokerAPIError`:  On unrecoverable API errors.
            :class:`RateLimitError`:  If rate-limited on all retries.
        """
        wait = self._API_BASE_WAIT
        last_exc: Exception = BrokerAPIError(f"{context}: no attempts made")

        for attempt in range(1, self._API_MAX_RETRIES + 1):
            self._rate_limiter.acquire()
            try:
                resp = fn(*args, **kwargs)
            except Exception as exc:
                raise BrokerAPIError(
                    f"{context}: SDK raised {type(exc).__name__}: {exc}"
                ) from exc

            logger.debug("%s → %s", context, resp)

            if resp is None:
                raise BrokerAPIError(f"{context}: received None response")

            if not isinstance(resp, dict):
                # Some SmartAPI calls return a plain string on success
                return resp

            error_code = resp.get("errorcode", "")

            # Angel One uses several error codes for rate-limiting:
            # AG8001 — generic too-many-requests
            # AB1019 — historical data endpoint throttle
            # AB1021 — order/quote endpoint throttle
            _RATE_LIMIT_CODES = {"AG8001", "AB1019", "AB1021"}

            if error_code in _RATE_LIMIT_CODES:
                last_exc = RateLimitError(
                    f"{context}: rate limit hit ({error_code})"
                )
                if attempt < self._API_MAX_RETRIES:
                    logger.warning(
                        "%s: rate limited — retry %d/%d in %.1fs",
                        context, attempt, self._API_MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    wait *= 2.0
                    continue
                raise last_exc

            if not resp.get("status", True):
                msg = resp.get("message", "unknown error")
                raise BrokerAPIError(f"{context} failed: {msg}")

            return resp

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    def get_ltp(self, symbol: str, exchange: str = "NSE") -> dict:
        """Return the last traded price for *symbol*.

        Args:
            symbol:   Stock symbol, e.g. ``"TCS"``.
            exchange: ``"NSE"`` (default) or ``"BSE"``.

        Returns:
            ``{"symbol": "TCS", "ltp": 3850.50, "exchange": "NSE", "token": "11536"}``
        """
        self._ensure_session()
        token = self.symbol_to_token(symbol, exchange)
        trading_sym = self._trading_symbol.get(symbol, symbol)

        logger.debug("get_ltp: %s exchange=%s token=%s", symbol, exchange, token)
        resp = self._call_api(
            self._smart.ltpData,
            exchange,
            trading_sym,
            token,
            context=f"ltpData({symbol})",
        )
        ltp_val = float(resp["data"]["ltp"])
        logger.debug("LTP %s = %.2f", symbol, ltp_val)
        return {"symbol": symbol, "ltp": ltp_val, "exchange": exchange, "token": token}

    def get_ltp_batch(
        self, symbols: List[str], exchange: str = "NSE"
    ) -> List[dict]:
        """Return LTP for multiple symbols using the batch market-data API.

        Args:
            symbols:  List of stock symbols, e.g. ``["TCS", "INFY"]``.
            exchange: ``"NSE"`` (default) or ``"BSE"``.

        Returns:
            ``[{"symbol": "TCS", "ltp": 3850.50, "exchange": "NSE", "token": "11536"}, ...]``
            Symbols whose tokens cannot be resolved are silently skipped.
        """
        self._ensure_session()

        token_to_clean: Dict[str, str] = {}
        tokens: List[str] = []
        for sym in symbols:
            try:
                tok = self.symbol_to_token(sym, exchange)
                token_to_clean[tok] = sym
                tokens.append(tok)
            except BrokerAPIError:
                logger.warning("get_ltp_batch: no token for %s — skipping", sym)

        if not tokens:
            return []

        logger.debug(
            "get_ltp_batch: %d tokens exchange=%s", len(tokens), exchange
        )
        resp = self._call_api(
            self._smart.getMarketData,
            "LTP",
            {exchange: tokens},
            context="getMarketData(LTP)",
        )
        results: List[dict] = []
        for item in (resp.get("data") or {}).get("fetched", []):
            tok = str(item.get("symbolToken", ""))
            sym = token_to_clean.get(tok, tok)
            results.append({
                "symbol": sym,
                "ltp": float(item.get("ltp", 0)),
                "exchange": exchange,
                "token": tok,
            })
        return results

    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        from_date: str,
        to_date: str,
        exchange: str = "NSE",
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for *symbol*, paginating if the range is large.

        Args:
            symbol:    Stock symbol, e.g. ``"TCS"``.
            interval:  ``"ONE_DAY"``, ``"ONE_HOUR"``, ``"FIFTEEN_MINUTE"``,
                       ``"FIVE_MINUTE"``, etc.
            from_date: ``"YYYY-MM-DD HH:MM"`` (Angel One format).
            to_date:   ``"YYYY-MM-DD HH:MM"``.
            exchange:  ``"NSE"`` (default) or ``"BSE"``.

        Returns:
            DataFrame with columns ``[datetime, open, high, low, close, volume]``,
            sorted ascending. Empty DataFrame if no data is returned.
        """
        self._ensure_session()
        token = self.symbol_to_token(symbol, exchange)

        max_days = self._max_days_for_interval(interval)
        chunks = self._split_date_range(from_date, to_date, max_days)
        all_rows: List[list] = []

        for chunk_from, chunk_to in chunks:
            params = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": interval,
                "fromdate": chunk_from,
                "todate": chunk_to,
            }
            logger.debug(
                "get_historical_data: %s %s %s → %s",
                symbol, interval, chunk_from, chunk_to,
            )
            resp = self._call_api(
                self._smart.getCandleData,
                params,
                context=f"getCandleData({symbol},{interval})",
            )
            chunk_data = resp.get("data") or []
            all_rows.extend(chunk_data)

        if not all_rows:
            return pd.DataFrame(
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(
            all_rows,
            columns=["datetime", "open", "high", "low", "close", "volume"],
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Instrument Master & Symbol-Token Mapping
    # ------------------------------------------------------------------

    def download_instrument_master(self) -> pd.DataFrame:
        """Return the Angel One instrument master (NSE equity only).

        Downloads once per 24 hours and caches at
        ``data/instrument_master.json``.

        Returns:
            DataFrame with columns
            ``[token, symbol, name, exchange, segment, expiry,
               lot_size, instrument_type, tick_size]``.
        """
        with self._instrument_lock:
            if self._instrument_master is not None:
                return self._instrument_master

            if self._is_cache_fresh():
                logger.debug("Instrument master: loading from local cache")
                self._instrument_master = self._load_cached_master()
            else:
                logger.info(
                    "Instrument master: downloading fresh copy from Angel One"
                )
                self._instrument_master = self._fetch_and_cache_master()

            return self._instrument_master

    def _is_cache_fresh(self) -> bool:
        if not _INSTRUMENT_MASTER_CACHE.exists():
            return False
        age_seconds = time.time() - _INSTRUMENT_MASTER_CACHE.stat().st_mtime
        return age_seconds < _CACHE_TTL_HOURS * 3600

    def _fetch_and_cache_master(self) -> pd.DataFrame:
        try:
            resp = requests.get(_INSTRUMENT_MASTER_URL, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise BrokerAPIError(
                f"Instrument master download failed: {exc}"
            ) from exc

        raw: List[dict] = resp.json()
        _INSTRUMENT_MASTER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _INSTRUMENT_MASTER_CACHE.write_text(
            json.dumps(raw), encoding="utf-8"
        )
        logger.info(
            "Instrument master cached: %d records → %s",
            len(raw), _INSTRUMENT_MASTER_CACHE,
        )
        return self._parse_master(raw)

    def _load_cached_master(self) -> pd.DataFrame:
        raw = json.loads(_INSTRUMENT_MASTER_CACHE.read_text(encoding="utf-8"))
        return self._parse_master(raw)

    def _parse_master(self, raw: List[dict]) -> pd.DataFrame:
        """Normalise raw instrument list → NSE equity DataFrame + caches."""
        df = pd.DataFrame(raw)

        # Filter to NSE equities
        seg_col = "exch_seg"
        if seg_col not in df.columns:
            logger.warning("Instrument master has no '%s' column", seg_col)
            return df

        # Angel One API returns "nse_cm" (NSE Cash Market) for equities.
        # Older API versions may return plain "NSE".  Accept both; exclude
        # NFO/BSE segments so only regular equity rows are retained.
        # instrumenttype="" filters out indices (AMXIDX) and derivatives.
        _NSE_EQUITY_SEGMENTS = {"NSE", "nse_cm", "NSE_CM"}
        nse = df[
            df[seg_col].isin(_NSE_EQUITY_SEGMENTS) & (df["instrumenttype"] == "")
        ].copy()

        # Column renaming (keep only columns that exist)
        col_map = {
            "token": "token",
            "symbol": "symbol",
            "name": "name",
            seg_col: "segment",
            "expiry": "expiry",
            "lotsize": "lot_size",
            "instrumenttype": "instrument_type",
            "tick_size": "tick_size",
        }
        rename = {k: v for k, v in col_map.items() if k in nse.columns}
        nse = nse[list(rename)].rename(columns=rename)
        nse["exchange"] = "NSE"
        nse.reset_index(drop=True, inplace=True)

        # Rebuild caches
        self._symbol_to_token = {}
        self._token_to_symbol = {}
        self._trading_symbol = {}

        for _, row in nse.iterrows():
            tok = str(row.get("token", ""))
            trading_sym = str(row.get("symbol", ""))       # e.g. "TCS-EQ"
            clean_sym = trading_sym.replace("-EQ", "").replace("-BE", "")

            self._symbol_to_token[clean_sym] = tok
            self._symbol_to_token[trading_sym] = tok        # also accept full form
            self._token_to_symbol[tok] = trading_sym
            self._trading_symbol[clean_sym] = trading_sym

        logger.debug(
            "Instrument master parsed: %d NSE equity instruments", len(nse)
        )
        return nse

    def get_instrument_token(self, symbol: str, exchange: str = "NSE") -> str:
        """Return the numeric token string for *symbol*.

        Convenience alias for :meth:`symbol_to_token`.
        """
        return self.symbol_to_token(symbol, exchange)

    def symbol_to_token(self, symbol: str, exchange: str = "NSE") -> str:  # noqa: ARG002
        """Resolve *symbol* to its Angel One numeric token.

        Tries the symbol as-is, then with ``-EQ`` suffix.  Falls back to
        downloading the instrument master if the cache is empty.

        Raises:
            :class:`BrokerAPIError`: if symbol is not found.
        """
        # Fast path: already cached
        if symbol in self._symbol_to_token:
            return self._symbol_to_token[symbol]
        with_eq = f"{symbol}-EQ"
        if with_eq in self._symbol_to_token:
            return self._symbol_to_token[with_eq]

        # Slow path: load instrument master
        self.download_instrument_master()

        if symbol in self._symbol_to_token:
            return self._symbol_to_token[symbol]
        if with_eq in self._symbol_to_token:
            return self._symbol_to_token[with_eq]

        raise BrokerAPIError(
            f"Symbol {symbol!r} not found in Angel One instrument master "
            f"(tried '{symbol}' and '{with_eq}')."
        )

    def token_to_symbol(self, token: str) -> str:
        """Reverse lookup: numeric *token* → trading symbol (e.g. ``"TCS-EQ"``).

        Raises:
            :class:`BrokerAPIError`: if token is not found.
        """
        if token in self._token_to_symbol:
            return self._token_to_symbol[token]

        self.download_instrument_master()

        if token in self._token_to_symbol:
            return self._token_to_symbol[token]

        raise BrokerAPIError(f"Token {token!r} not found in instrument master.")

    # ------------------------------------------------------------------
    # Portfolio & Orders
    # ------------------------------------------------------------------

    def get_holdings(self) -> List[dict]:
        """Return current delivery (demat) holdings.

        Returns:
            ``[{"symbol": "TCS", "quantity": 5, "avg_price": 3800,
                "ltp": 3850, "pnl": 250, "token": "11536"}, ...]``
        """
        self._ensure_session()
        logger.debug("get_holdings: fetching")
        resp = self._call_api(
            self._smart.holding, context="holding()"
        )
        holdings: List[dict] = []
        for item in (resp.get("data") or []):
            trading_sym = str(item.get("tradingsymbol", ""))
            clean = trading_sym.replace("-EQ", "").replace("-BE", "")
            holdings.append(
                {
                    "symbol": clean,
                    "quantity": int(item.get("quantity", 0)),
                    "avg_price": float(item.get("averageprice", 0)),
                    "ltp": float(item.get("ltp", 0)),
                    "pnl": float(item.get("profitandloss", 0)),
                    "token": str(item.get("symboltoken", "")),
                }
            )
        return holdings

    def get_positions(self) -> List[dict]:
        """Return today's open intraday/carry-forward positions.

        Returns:
            ``[{"symbol": "TCS", "quantity": 5, "avg_price": 3800,
                "ltp": 3850, "pnl": 250, "product": "DELIVERY",
                "exchange": "NSE"}, ...]``
        """
        self._ensure_session()
        logger.debug("get_positions: fetching")
        resp = self._call_api(
            self._smart.position, context="position()"
        )
        positions: List[dict] = []
        for item in (resp.get("data") or []):
            trading_sym = str(item.get("tradingsymbol", ""))
            clean = trading_sym.replace("-EQ", "").replace("-BE", "")
            positions.append(
                {
                    "symbol": clean,
                    "quantity": int(item.get("netqty", 0)),
                    "avg_price": float(item.get("netprice", 0)),
                    "ltp": float(item.get("ltp", 0)),
                    "pnl": float(item.get("pnl", 0)),
                    "product": item.get("producttype", ""),
                    "exchange": item.get("exchange", ""),
                }
            )
        return positions

    def get_portfolio_value(self) -> dict:
        """Return a rolled-up portfolio summary.

        Returns:
            ``{"total_value": 52000.0, "invested": 45000.0,
               "available_cash": 7000.0, "total_pnl": 2000.0}``
        """
        holdings = self.get_holdings()
        cash = self.get_margin_available()

        invested = sum(h["avg_price"] * h["quantity"] for h in holdings)
        current = sum(h["ltp"] * h["quantity"] for h in holdings)
        pnl = current - invested

        return {
            "total_value": round(current + cash, 2),
            "invested": round(invested, 2),
            "available_cash": round(cash, 2),
            "total_pnl": round(pnl, 2),
        }

    def get_order_book(self) -> List[dict]:
        """Return all orders placed today.

        Returns:
            List of order dicts with keys:
            ``order_id, symbol, transaction_type, quantity, price,
              order_type, status, filled_qty, product, variety``.
        """
        self._ensure_session()
        logger.debug("get_order_book: fetching")
        resp = self._call_api(
            self._smart.orderBook, context="orderBook()"
        )
        orders: List[dict] = []
        for item in (resp.get("data") or []):
            trading_sym = str(item.get("tradingsymbol", ""))
            clean = trading_sym.replace("-EQ", "").replace("-BE", "")
            orders.append(
                {
                    "order_id": str(item.get("orderid", "")),
                    "symbol": clean,
                    "transaction_type": item.get("transactiontype", ""),
                    "quantity": int(item.get("quantity", 0)),
                    "price": float(item.get("price", 0)),
                    "order_type": item.get("ordertype", ""),
                    "status": item.get("status", ""),
                    "filled_qty": int(item.get("filledshares", 0)),
                    "product": item.get("producttype", ""),
                    "variety": item.get("variety", "NORMAL"),
                }
            )
        return orders

    def get_order_status(self, order_id: str) -> dict:
        """Return status details for a specific order.

        Args:
            order_id: Broker-assigned order ID.

        Returns:
            ``{"order_id": "...", "status": "EXECUTED", "filled_qty": 5,
               "price": 3850.50, "symbol": "TCS", "transaction_type": "BUY"}``

        Raises:
            :class:`BrokerAPIError`: if order_id is not in today's order book.
        """
        orders = self.get_order_book()
        for order in orders:
            if order["order_id"] == order_id:
                return {
                    "order_id": order_id,
                    "status": order["status"],
                    "filled_qty": order["filled_qty"],
                    "price": order["price"],
                    "symbol": order["symbol"],
                    "transaction_type": order["transaction_type"],
                }
        raise BrokerAPIError(
            f"Order {order_id!r} not found in today's order book."
        )

    # ------------------------------------------------------------------
    # Order Placement
    # ------------------------------------------------------------------

    def place_buy_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        order_type: str = "LIMIT",
    ) -> dict:
        """Place a DELIVERY (CNC) buy order.

        Args:
            symbol:     Stock symbol, e.g. ``"TCS"``.
            quantity:   Number of shares to buy.
            price:      Limit price (ignored for MARKET orders).
            order_type: ``"LIMIT"`` (default) or ``"MARKET"``.

        Returns:
            ``{"order_id": "...", "status": "PENDING"}``
        """
        logger.info(
            "place_buy_order: %s qty=%d price=%.2f type=%s",
            symbol, quantity, price, order_type,
        )
        return self._place_order(
            symbol=symbol,
            quantity=quantity,
            price=price,
            transaction_type="BUY",
            order_type=order_type,
            variety="NORMAL",
        )

    def place_sell_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        order_type: str = "LIMIT",
    ) -> dict:
        """Place a DELIVERY (CNC) sell order.

        Args:
            symbol:     Stock symbol.
            quantity:   Number of shares to sell.
            price:      Limit price (ignored for MARKET orders).
            order_type: ``"LIMIT"`` (default) or ``"MARKET"``.

        Returns:
            ``{"order_id": "...", "status": "PENDING"}``
        """
        logger.info(
            "place_sell_order: %s qty=%d price=%.2f type=%s",
            symbol, quantity, price, order_type,
        )
        return self._place_order(
            symbol=symbol,
            quantity=quantity,
            price=price,
            transaction_type="SELL",
            order_type=order_type,
            variety="NORMAL",
        )

    def place_stop_loss_order(
        self,
        symbol: str,
        quantity: int,
        trigger_price: float,
        limit_price: float,
    ) -> dict:
        """Place a SL-LIMIT sell order for risk management.

        Args:
            symbol:        Stock symbol.
            quantity:      Number of shares to exit.
            trigger_price: Price at which the SL order is triggered.
            limit_price:   Minimum acceptable execution price (≤ trigger).

        Returns:
            ``{"order_id": "...", "status": "PENDING"}``
        """
        logger.info(
            "place_stop_loss_order: %s qty=%d trigger=%.2f limit=%.2f",
            symbol, quantity, trigger_price, limit_price,
        )
        self._ensure_session()
        token = self.symbol_to_token(symbol)
        trading_sym = self._trading_symbol.get(symbol, symbol)

        params: dict = {
            "variety": "STOPLOSS",
            "tradingsymbol": trading_sym,
            "symboltoken": token,
            "transactiontype": "SELL",
            "exchange": self._default_exchange,
            "ordertype": "STOPLOSS_LIMIT",
            "producttype": "DELIVERY",
            "duration": "DAY",
            "price": str(limit_price),
            "triggerprice": str(trigger_price),
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(quantity),
        }
        return self._submit_order(params, context=f"SL-LIMIT {symbol}")

    def _place_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        transaction_type: str,
        order_type: str,
        variety: str,
    ) -> dict:
        """Build and submit a regular (non-SL) order."""
        self._ensure_session()
        token = self.symbol_to_token(symbol)
        trading_sym = self._trading_symbol.get(symbol, symbol)

        params: dict = {
            "variety": variety,
            "tradingsymbol": trading_sym,
            "symboltoken": token,
            "transactiontype": transaction_type,
            "exchange": self._default_exchange,
            "ordertype": order_type,
            "producttype": "DELIVERY",
            "duration": "DAY",
            "price": str(price),
            "squareoff": "0",
            "stoploss": "0",
            "quantity": str(quantity),
        }
        return self._submit_order(
            params,
            context=f"{transaction_type} {symbol} qty={quantity} price={price}",
        )

    def _submit_order(self, params: dict, context: str) -> dict:
        """Send *params* to placeOrder and return normalised result.

        Raises:
            :class:`OrderRejectedError`: if the API rejects the order.
        """
        resp = self._call_api(
            self._smart.placeOrder, params, context=f"placeOrder({context})"
        )
        # placeOrder returns {"status": True, "data": {"orderid": "..."}}
        raw_data = resp.get("data") or {}
        if isinstance(raw_data, dict):
            order_id = str(raw_data.get("orderid", ""))
        else:
            order_id = str(raw_data)

        if not order_id:
            raise OrderRejectedError(
                f"placeOrder returned no order ID ({context}): {resp}"
            )

        logger.info("Order placed: %s → order_id=%s", context, order_id)
        return {"order_id": order_id, "status": "PENDING"}

    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_quantity: Optional[int] = None,
    ) -> dict:
        """Modify a pending order's price or quantity.

        Looks up the current order from the order book to obtain the symbol,
        token, and existing values for fields that are not being changed.

        Args:
            order_id:     Broker-assigned order ID to modify.
            new_price:    New limit price (``None`` keeps existing).
            new_quantity: New quantity (``None`` keeps existing).

        Returns:
            ``{"order_id": "...", "status": "MODIFIED"}``

        Raises:
            :class:`BrokerAPIError`: if the order is not found or the call fails.
        """
        self._ensure_session()

        # Fetch current order to get symbol, token, existing price/qty/variety
        current_orders = self.get_order_book()
        current = next(
            (o for o in current_orders if o["order_id"] == order_id), None
        )
        if current is None:
            raise BrokerAPIError(
                f"modify_order: order {order_id!r} not found in order book."
            )

        sym = current["symbol"]
        token = self.symbol_to_token(sym)
        trading_sym = self._trading_symbol.get(sym, sym)
        price = new_price if new_price is not None else current["price"]
        qty = new_quantity if new_quantity is not None else current["quantity"]

        params: dict = {
            "variety": current.get("variety", "NORMAL"),
            "orderid": order_id,
            "ordertype": current.get("order_type", "LIMIT"),
            "producttype": "DELIVERY",
            "duration": "DAY",
            "tradingsymbol": trading_sym,
            "symboltoken": token,
            "exchange": self._default_exchange,
            "price": str(price),
            "quantity": str(qty),
        }
        logger.info(
            "modify_order: %s price=%s qty=%s", order_id, new_price, new_quantity
        )
        self._call_api(
            self._smart.modifyOrder,
            params,
            context=f"modifyOrder({order_id})",
        )
        return {"order_id": order_id, "status": "MODIFIED"}

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending order.

        Args:
            order_id: Broker-assigned order ID.

        Returns:
            ``{"order_id": "...", "status": "CANCELLED"}``
        """
        self._ensure_session()
        logger.info("cancel_order: %s", order_id)
        self._call_api(
            self._smart.cancelOrder,
            "NORMAL",
            order_id,
            context=f"cancelOrder({order_id})",
        )
        return {"order_id": order_id, "status": "CANCELLED"}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def calculate_quantity(self, symbol: str, capital_to_deploy: float) -> int:
        """Calculate the number of shares purchasable with *capital_to_deploy*.

        Args:
            symbol:            Stock symbol.
            capital_to_deploy: Amount in INR to invest.

        Returns:
            Floor number of shares (0 if capital < price of 1 share).
        """
        try:
            info = self.get_ltp(symbol)
            price = info["ltp"]
        except BrokerAPIError as exc:
            logger.error(
                "calculate_quantity: cannot get LTP for %s: %s", symbol, exc
            )
            return 0

        if price <= 0:
            logger.warning(
                "calculate_quantity: LTP for %s is %.2f — returning 0", symbol, price
            )
            return 0

        qty = math.floor(capital_to_deploy / price)
        logger.debug(
            "calculate_quantity: %s capital=%.2f LTP=%.2f → qty=%d",
            symbol, capital_to_deploy, price, qty,
        )
        return qty

    def get_margin_available(self) -> float:
        """Return available cash / margin for trading.

        Returns:
            Available cash in INR.
        """
        self._ensure_session()
        logger.debug("get_margin_available: fetching RMS limits")
        resp = self._call_api(
            self._smart.rmsLimit, context="rmsLimit()"
        )
        data = resp.get("data") or {}
        # Angel One may use "availablecash" or "net" depending on SDK version
        cash = float(
            data.get("availablecash", data.get("net", 0))
        )
        logger.debug("Available cash: %.2f", cash)
        return cash

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_days_for_interval(interval: str) -> int:
        """Return max days per getCandleData request for *interval*."""
        mapping: Dict[str, int] = {
            "ONE_MINUTE":      10,
            "THREE_MINUTE":    10,
            "FIVE_MINUTE":     30,
            "TEN_MINUTE":      30,
            "FIFTEEN_MINUTE":  30,
            "THIRTY_MINUTE":   30,
            "ONE_HOUR":        30,
            "ONE_DAY":        365,
            "ONE_WEEK":       365,
            "ONE_MONTH":     1825,
        }
        return mapping.get(interval, 30)

    @staticmethod
    def _split_date_range(
        from_date: str, to_date: str, max_days: int
    ) -> List[Tuple[str, str]]:
        """Split [from_date, to_date] into chunks of *max_days* days.

        Args:
            from_date: ``"YYYY-MM-DD HH:MM"`` string.
            to_date:   ``"YYYY-MM-DD HH:MM"`` string.
            max_days:  Maximum days per chunk.

        Returns:
            List of ``(chunk_from, chunk_to)`` tuples in the same format.
        """
        fmt = "%Y-%m-%d %H:%M"
        start = datetime.strptime(from_date, fmt)
        end = datetime.strptime(to_date, fmt)
        chunks: List[Tuple[str, str]] = []
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=max_days), end)
            chunks.append(
                (current.strftime(fmt), chunk_end.strftime(fmt))
            )
            # Advance by one minute past chunk_end to avoid overlap
            current = chunk_end + timedelta(minutes=1)
        return chunks
