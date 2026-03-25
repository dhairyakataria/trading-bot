"""Market Data — OHLCV and live price feeds for Indian equities."""
from __future__ import annotations

from typing import Any

import pandas as pd


class MarketDataFetcher:
    """Fetches historical and live market data for NSE/BSE securities.

    Responsibilities:
    - Retrieve OHLCV data via Angel One API (primary) or yfinance (fallback)
    - Cache data locally to minimise redundant API calls
    - Normalise data into a consistent pandas DataFrame format
    - Support multiple intervals: 1m, 5m, 15m, 1h, 1d
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        period: int = 100,
    ) -> pd.DataFrame:
        """Return an OHLCV DataFrame for the given symbol and interval."""
        raise NotImplementedError

    def get_live_price(self, symbol: str) -> float:
        """Return the current market price for a symbol."""
        raise NotImplementedError

    def get_multiple_quotes(self, symbols: list[str]) -> dict[str, float]:
        """Return a symbol→price dict for a batch of symbols."""
        raise NotImplementedError

    def get_global_indices(self) -> dict[str, float]:
        """Return current levels of major global indices (SGX Nifty, Dow, etc.)."""
        raise NotImplementedError
