"""Tests for src/broker/angel_one.py.

All SmartAPI network calls are mocked so no real API credentials are needed.
Run with:  pytest tests/test_angel_one.py -v
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from src.broker.angel_one import (
    AngelOneClient,
    BrokerAPIError,
    BrokerAuthError,
    OrderRejectedError,
    RateLimitError,
    _RateLimiter,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MOCK_CONFIG: dict = {
    "broker": {
        "angel_one": {
            "client_id": "A12345",
            "password": "1234",
            "api_key": "test_api_key_xyz",
            "totp_secret": "JBSWY3DPEHPK3PXP",   # standard pyotp test secret
            "default_exchange": "NSE",
        }
    }
}

# Minimal instrument master rows (two NSE equity entries)
_MASTER_ROWS = [
    {
        "token": "11536",
        "symbol": "TCS-EQ",
        "name": "TATA CONSULTANCY",
        "exch_seg": "nse_cm",
        "expiry": "",
        "lotsize": "1",
        "instrumenttype": "",
        "tick_size": "5.0",
    },
    {
        "token": "1594",
        "symbol": "INFY-EQ",
        "name": "INFOSYS",
        "exch_seg": "nse_cm",
        "expiry": "",
        "lotsize": "1",
        "instrumenttype": "",
        "tick_size": "5.0",
    },
    # Non-NSE row — should be filtered out
    {
        "token": "9999",
        "symbol": "NIFTY2640CE",
        "name": "NIFTY CALL",
        "exch_seg": "nfo_opt",
        "expiry": "26APR2026",
        "lotsize": "25",
        "instrumenttype": "OPTIDX",
        "tick_size": "0.05",
    },
]

_SUCCESS_RESPONSE: dict = {"status": True, "message": "SUCCESS", "errorcode": ""}


def _ok(data: Any) -> dict:
    """Wrap *data* in a standard SmartAPI success envelope."""
    return {**_SUCCESS_RESPONSE, "data": data}


@pytest.fixture()
def client() -> AngelOneClient:
    """Bare client — not logged in."""
    return AngelOneClient(_MOCK_CONFIG)


@pytest.fixture()
def logged_in(client: AngelOneClient) -> tuple[AngelOneClient, MagicMock]:
    """Client with a fake active session; returns (client, mock_smart)."""
    mock_smart = MagicMock()
    client._smart = mock_smart
    client._jwt_token = "Bearer fake.jwt.token"
    client._refresh_token = "fake_refresh"
    client._feed_token = "fake_feed"
    client._session_expiry = datetime.now() + timedelta(hours=7)
    return client, mock_smart


@pytest.fixture()
def preloaded(logged_in: tuple) -> tuple[AngelOneClient, MagicMock]:
    """Client with instrument master already loaded into memory cache."""
    client, mock_smart = logged_in
    client._parse_master(_MASTER_ROWS)
    return client, mock_smart


# ---------------------------------------------------------------------------
# _RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_burst_without_blocking(self) -> None:
        rl = _RateLimiter(rate=10, period=1.0)
        t0 = time.monotonic()
        for _ in range(10):
            rl.acquire()
        elapsed = time.monotonic() - t0
        # All 10 tokens available immediately — should finish well under 1s
        assert elapsed < 1.0, f"Burst took {elapsed:.2f}s, expected < 1s"

    def test_throttles_after_burst(self) -> None:
        rl = _RateLimiter(rate=2, period=0.2)   # 2 calls per 0.2s = 10 rps
        rl.acquire()  # token 1
        rl.acquire()  # token 2  (tokens exhausted)
        t0 = time.monotonic()
        rl.acquire()  # must wait
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.05, f"Expected throttle, got {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# AngelOneClient — constructor
# ---------------------------------------------------------------------------

class TestInit:
    def test_stores_client_id(self, client: AngelOneClient) -> None:
        assert client._client_id == "A12345"

    def test_stores_api_key(self, client: AngelOneClient) -> None:
        assert client._api_key == "test_api_key_xyz"

    def test_not_authenticated_on_init(self, client: AngelOneClient) -> None:
        assert not client.is_authenticated()

    def test_smart_is_none_on_init(self, client: AngelOneClient) -> None:
        assert client._smart is None

    def test_does_not_log_password(
        self, client: AngelOneClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The password "1234" must never appear in log output
        assert "1234" not in caplog.text
        assert "test_api_key_xyz" not in caplog.text


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _fake_smartapi_module(mock_instance: MagicMock) -> MagicMock:
    """Build a fake SmartApi module whose SmartConnect() returns mock_instance."""
    mock_cls = MagicMock(return_value=mock_instance)
    fake_module = MagicMock()
    fake_module.SmartConnect = mock_cls
    return fake_module


class TestLogin:
    def _make_session_response(self) -> dict:
        return _ok(
            {
                "jwtToken": "Bearer fake",
                "refreshToken": "r_tok",
                "feedToken": "f_tok",
            }
        )

    def test_login_success(
        self, client: AngelOneClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_smart = MagicMock()
        mock_smart.generateSession.return_value = self._make_session_response()
        monkeypatch.setitem(sys.modules, "SmartApi", _fake_smartapi_module(mock_smart))

        with patch("src.broker.angel_one.pyotp.TOTP") as mock_totp:
            mock_totp.return_value.now.return_value = "123456"
            result = client.login()

        assert result is True
        assert client.is_authenticated()
        assert client._jwt_token == "Bearer fake"

    def test_login_retries_on_failure(
        self, client: AngelOneClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_smart = MagicMock()
        fail_resp = {"status": False, "message": "Invalid TOTP", "errorcode": "AB1002"}
        ok_resp = self._make_session_response()
        mock_smart.generateSession.side_effect = [fail_resp, ok_resp]
        monkeypatch.setitem(sys.modules, "SmartApi", _fake_smartapi_module(mock_smart))

        with patch("src.broker.angel_one.pyotp.TOTP") as mock_totp, \
             patch("src.broker.angel_one.time.sleep"):
            mock_totp.return_value.now.return_value = "654321"
            result = client.login()

        assert result is True
        assert mock_smart.generateSession.call_count == 2

    def test_login_raises_after_max_retries(
        self, client: AngelOneClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_smart = MagicMock()
        fail_resp = {"status": False, "message": "Bad creds", "errorcode": "AB1001"}
        mock_smart.generateSession.return_value = fail_resp
        monkeypatch.setitem(sys.modules, "SmartApi", _fake_smartapi_module(mock_smart))

        with patch("src.broker.angel_one.pyotp.TOTP") as mock_totp, \
             patch("src.broker.angel_one.time.sleep"):
            mock_totp.return_value.now.return_value = "000000"
            with pytest.raises(BrokerAuthError, match="Bad creds"):
                client.login()

        assert mock_smart.generateSession.call_count == 3

    def test_login_raises_on_sdk_exception(
        self, client: AngelOneClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_smart = MagicMock()
        mock_smart.generateSession.side_effect = RuntimeError("network error")
        monkeypatch.setitem(sys.modules, "SmartApi", _fake_smartapi_module(mock_smart))

        with patch("src.broker.angel_one.pyotp.TOTP") as mock_totp, \
             patch("src.broker.angel_one.time.sleep"):
            mock_totp.return_value.now.return_value = "111111"
            with pytest.raises(BrokerAuthError, match="network error"):
                client.login()

    def test_totp_secret_not_logged(
        self,
        client: AngelOneClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_smart = MagicMock()
        mock_smart.generateSession.return_value = self._make_session_response()
        monkeypatch.setitem(sys.modules, "SmartApi", _fake_smartapi_module(mock_smart))

        with patch("src.broker.angel_one.pyotp.TOTP") as mock_totp:
            mock_totp.return_value.now.return_value = "987654"
            client.login()

        assert "987654" not in caplog.text
        assert "JBSWY3DPEHPK3PXP" not in caplog.text


class TestLogout:
    def test_logout_calls_terminate(self, logged_in: tuple) -> None:
        client, mock_smart = logged_in
        client.logout()
        mock_smart.terminateSession.assert_called_once_with("A12345")
        assert not client.is_authenticated()

    def test_logout_noop_when_not_logged_in(
        self, client: AngelOneClient
    ) -> None:
        # Should not raise even with no session
        client.logout()

    def test_logout_clears_tokens(self, logged_in: tuple) -> None:
        client, _ = logged_in
        client.logout()
        assert client._jwt_token is None
        assert client._smart is None
        assert client._session_expiry is None


class TestIsAuthenticated:
    def test_true_when_session_active(self, logged_in: tuple) -> None:
        client, _ = logged_in
        assert client.is_authenticated()

    def test_false_when_no_smart(self, client: AngelOneClient) -> None:
        assert not client.is_authenticated()

    def test_false_when_session_expired(self, logged_in: tuple) -> None:
        client, _ = logged_in
        client._session_expiry = datetime.now() - timedelta(seconds=1)
        assert not client.is_authenticated()

    def test_false_when_no_jwt_token(self, logged_in: tuple) -> None:
        client, _ = logged_in
        client._jwt_token = None
        assert not client.is_authenticated()


# ---------------------------------------------------------------------------
# Instrument Master
# ---------------------------------------------------------------------------

class TestInstrumentMaster:
    def test_parse_filters_nse_equity(self, client: AngelOneClient) -> None:
        df = client._parse_master(_MASTER_ROWS)
        # Only 2 NSE equity rows (nse_cm), not the NFO row
        assert len(df) == 2
        assert set(df["symbol"].tolist()) == {"TCS-EQ", "INFY-EQ"}

    def test_parse_builds_symbol_to_token_cache(
        self, client: AngelOneClient
    ) -> None:
        client._parse_master(_MASTER_ROWS)
        # Clean symbol form
        assert client._symbol_to_token["TCS"] == "11536"
        assert client._symbol_to_token["INFY"] == "1594"

    def test_parse_builds_trading_symbol_form(
        self, client: AngelOneClient
    ) -> None:
        client._parse_master(_MASTER_ROWS)
        assert client._symbol_to_token["TCS-EQ"] == "11536"

    def test_parse_builds_token_to_symbol_cache(
        self, client: AngelOneClient
    ) -> None:
        client._parse_master(_MASTER_ROWS)
        assert client._token_to_symbol["11536"] == "TCS-EQ"
        assert client._token_to_symbol["1594"] == "INFY-EQ"

    def test_parse_builds_trading_symbol_map(
        self, client: AngelOneClient
    ) -> None:
        client._parse_master(_MASTER_ROWS)
        assert client._trading_symbol["TCS"] == "TCS-EQ"

    def test_download_uses_local_cache_when_fresh(
        self, client: AngelOneClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_path = tmp_path / "instrument_master.json"
        cache_path.write_text(json.dumps(_MASTER_ROWS), encoding="utf-8")

        monkeypatch.setattr(
            "src.broker.angel_one._INSTRUMENT_MASTER_CACHE", cache_path
        )
        with patch("src.broker.angel_one.requests.get") as mock_get:
            df = client.download_instrument_master()

        mock_get.assert_not_called()
        assert len(df) == 2

    def test_download_fetches_when_cache_stale(
        self, client: AngelOneClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_path = tmp_path / "instrument_master.json"
        cache_path.write_text(json.dumps(_MASTER_ROWS), encoding="utf-8")
        # Make the file appear old
        old_time = time.time() - 25 * 3600
        import os
        os.utime(cache_path, (old_time, old_time))

        monkeypatch.setattr(
            "src.broker.angel_one._INSTRUMENT_MASTER_CACHE", cache_path
        )
        mock_response = MagicMock()
        mock_response.json.return_value = _MASTER_ROWS
        mock_response.raise_for_status.return_value = None

        with patch("src.broker.angel_one.requests.get", return_value=mock_response):
            df = client.download_instrument_master()

        assert len(df) == 2

    def test_download_fetches_when_no_cache(
        self, client: AngelOneClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_path = tmp_path / "instrument_master.json"
        # Do NOT create the file
        monkeypatch.setattr(
            "src.broker.angel_one._INSTRUMENT_MASTER_CACHE", cache_path
        )
        mock_response = MagicMock()
        mock_response.json.return_value = _MASTER_ROWS
        mock_response.raise_for_status.return_value = None

        with patch("src.broker.angel_one.requests.get", return_value=mock_response):
            df = client.download_instrument_master()

        assert cache_path.exists()
        assert len(df) == 2

    def test_download_raises_on_network_error(
        self, client: AngelOneClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import requests as req_mod

        cache_path = tmp_path / "instrument_master.json"
        monkeypatch.setattr(
            "src.broker.angel_one._INSTRUMENT_MASTER_CACHE", cache_path
        )
        with patch(
            "src.broker.angel_one.requests.get",
            side_effect=req_mod.RequestException("timeout"),
        ):
            with pytest.raises(BrokerAPIError, match="timeout"):
                client.download_instrument_master()

    def test_master_cached_in_memory(
        self, client: AngelOneClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache_path = tmp_path / "instrument_master.json"
        monkeypatch.setattr(
            "src.broker.angel_one._INSTRUMENT_MASTER_CACHE", cache_path
        )
        mock_response = MagicMock()
        mock_response.json.return_value = _MASTER_ROWS
        mock_response.raise_for_status.return_value = None

        with patch(
            "src.broker.angel_one.requests.get", return_value=mock_response
        ) as mock_get:
            client.download_instrument_master()
            client.download_instrument_master()   # second call — should not re-download

        mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# Symbol-Token Mapping
# ---------------------------------------------------------------------------

class TestSymbolTokenMapping:
    def test_symbol_to_token_clean(self, client: AngelOneClient) -> None:
        client._parse_master(_MASTER_ROWS)
        assert client.symbol_to_token("TCS") == "11536"

    def test_symbol_to_token_with_eq_suffix(
        self, client: AngelOneClient
    ) -> None:
        client._parse_master(_MASTER_ROWS)
        assert client.symbol_to_token("TCS-EQ") == "11536"

    def test_symbol_to_token_tries_eq_fallback(
        self, client: AngelOneClient
    ) -> None:
        """If 'INFY' is not directly in cache but 'INFY-EQ' is, it should find it."""
        client._parse_master(_MASTER_ROWS)
        # Remove the clean "INFY" key to force the -EQ fallback path
        del client._symbol_to_token["INFY"]
        assert client.symbol_to_token("INFY") == "1594"

    def test_symbol_to_token_raises_for_unknown(
        self, client: AngelOneClient
    ) -> None:
        client._parse_master(_MASTER_ROWS)
        with pytest.raises(BrokerAPIError, match="UNKNOWN_SYM"):
            client.symbol_to_token("UNKNOWN_SYM")

    def test_token_to_symbol(self, client: AngelOneClient) -> None:
        client._parse_master(_MASTER_ROWS)
        assert client.token_to_symbol("11536") == "TCS-EQ"

    def test_token_to_symbol_raises_for_unknown(
        self, client: AngelOneClient
    ) -> None:
        client._parse_master(_MASTER_ROWS)
        with pytest.raises(BrokerAPIError, match="9999999"):
            client.token_to_symbol("9999999")

    def test_get_instrument_token_alias(self, client: AngelOneClient) -> None:
        client._parse_master(_MASTER_ROWS)
        assert client.get_instrument_token("INFY") == "1594"


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------

class TestGetLtp:
    def test_returns_ltp_dict(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.ltpData.return_value = _ok(
            {"ltp": 3850.5, "exchange": "NSE", "tradingsymbol": "TCS-EQ"}
        )
        result = client.get_ltp("TCS")
        assert result["symbol"] == "TCS"
        assert result["ltp"] == 3850.5
        assert result["token"] == "11536"
        assert result["exchange"] == "NSE"

    def test_calls_ltpdata_with_correct_args(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.ltpData.return_value = _ok({"ltp": 100.0})
        client.get_ltp("INFY")
        mock_smart.ltpData.assert_called_once_with("NSE", "INFY-EQ", "1594")

    def test_raises_on_api_error(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.ltpData.return_value = {
            "status": False,
            "message": "Symbol not found",
            "errorcode": "AB0001",
        }
        with pytest.raises(BrokerAPIError, match="Symbol not found"):
            client.get_ltp("TCS")


class TestGetLtpBatch:
    def test_returns_list_of_ltps(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.getMarketData.return_value = _ok(
            {
                "fetched": [
                    {"symbolToken": "11536", "ltp": 3850.5},
                    {"symbolToken": "1594", "ltp": 1520.3},
                ],
                "unfetched": [],
            }
        )
        result = client.get_ltp_batch(["TCS", "INFY"])
        symbols = {r["symbol"] for r in result}
        assert symbols == {"TCS", "INFY"}

    def test_skips_unknown_symbols(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.getMarketData.return_value = _ok(
            {"fetched": [{"symbolToken": "11536", "ltp": 3850.5}], "unfetched": []}
        )
        result = client.get_ltp_batch(["TCS", "UNKNOWN_XYZ"])
        assert len(result) == 1
        assert result[0]["symbol"] == "TCS"

    def test_empty_list_returns_empty(self, preloaded: tuple) -> None:
        client, _ = preloaded
        result = client.get_ltp_batch([])
        assert result == []


class TestGetHistoricalData:
    def _candle_rows(self) -> list:
        return [
            ["2026-01-02T09:15:00+05:30", 3800, 3870, 3780, 3850, 100000],
            ["2026-01-03T09:15:00+05:30", 3850, 3900, 3840, 3880, 120000],
        ]

    def test_returns_dataframe_with_correct_columns(
        self, preloaded: tuple
    ) -> None:
        client, mock_smart = preloaded
        mock_smart.getCandleData.return_value = _ok(self._candle_rows())
        df = client.get_historical_data(
            "TCS", "ONE_DAY", "2026-01-01 09:15", "2026-01-05 15:30"
        )
        assert list(df.columns) == [
            "datetime", "open", "high", "low", "close", "volume"
        ]
        assert len(df) == 2

    def test_sorted_by_datetime(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        # Return candles in reverse order
        reversed_rows = list(reversed(self._candle_rows()))
        mock_smart.getCandleData.return_value = _ok(reversed_rows)
        df = client.get_historical_data(
            "TCS", "ONE_DAY", "2026-01-01 09:15", "2026-01-05 15:30"
        )
        assert df["datetime"].is_monotonic_increasing

    def test_empty_data_returns_empty_df(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.getCandleData.return_value = _ok([])
        df = client.get_historical_data(
            "TCS", "ONE_DAY", "2026-01-01 09:15", "2026-01-02 15:30"
        )
        assert len(df) == 0
        assert "datetime" in df.columns

    def test_paginates_large_date_range(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.getCandleData.return_value = _ok(self._candle_rows())
        # ONE_MINUTE max is 10 days; requesting 25 days → should paginate
        client.get_historical_data(
            "TCS", "ONE_MINUTE", "2026-01-01 09:15", "2026-01-25 15:30"
        )
        assert mock_smart.getCandleData.call_count >= 3


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class TestGetHoldings:
    def test_returns_holdings_list(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.holding.return_value = _ok(
            [
                {
                    "tradingsymbol": "TCS-EQ",
                    "quantity": 5,
                    "averageprice": 3800.0,
                    "ltp": 3850.0,
                    "profitandloss": 250.0,
                    "symboltoken": "11536",
                }
            ]
        )
        holdings = client.get_holdings()
        assert len(holdings) == 1
        h = holdings[0]
        assert h["symbol"] == "TCS"
        assert h["quantity"] == 5
        assert h["pnl"] == 250.0

    def test_strips_eq_suffix(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.holding.return_value = _ok(
            [{"tradingsymbol": "INFY-EQ", "quantity": 10,
              "averageprice": 1500.0, "ltp": 1520.0,
              "profitandloss": 200.0, "symboltoken": "1594"}]
        )
        holdings = client.get_holdings()
        assert holdings[0]["symbol"] == "INFY"

    def test_empty_holdings(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.holding.return_value = _ok([])
        assert client.get_holdings() == []

    def test_null_data_returns_empty(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.holding.return_value = _ok(None)
        assert client.get_holdings() == []


class TestGetPortfolioValue:
    def test_portfolio_value_calculation(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.holding.return_value = _ok(
            [
                {
                    "tradingsymbol": "TCS-EQ",
                    "quantity": 10,
                    "averageprice": 3800.0,
                    "ltp": 3900.0,
                    "profitandloss": 1000.0,
                    "symboltoken": "11536",
                }
            ]
        )
        mock_smart.rmsLimit.return_value = _ok({"availablecash": "7000.0"})
        pv = client.get_portfolio_value()
        assert pv["invested"] == 38000.0
        assert pv["total_pnl"] == 1000.0     # 39000 - 38000
        assert pv["available_cash"] == 7000.0
        assert pv["total_value"] == 46000.0  # 39000 + 7000


# ---------------------------------------------------------------------------
# Order Placement
# ---------------------------------------------------------------------------

class TestPlaceBuyOrder:
    def test_returns_order_id(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.placeOrder.return_value = _ok({"orderid": "ORD001"})
        result = client.place_buy_order("TCS", 5, 3850.0)
        assert result["order_id"] == "ORD001"
        assert result["status"] == "PENDING"

    def test_uses_delivery_product(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.placeOrder.return_value = _ok({"orderid": "ORD002"})
        client.place_buy_order("TCS", 5, 3850.0)
        called_params = mock_smart.placeOrder.call_args[0][0]
        assert called_params["producttype"] == "DELIVERY"

    def test_uses_limit_order_by_default(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.placeOrder.return_value = _ok({"orderid": "ORD003"})
        client.place_buy_order("TCS", 5, 3850.0)
        called_params = mock_smart.placeOrder.call_args[0][0]
        assert called_params["ordertype"] == "LIMIT"

    def test_market_order_type(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.placeOrder.return_value = _ok({"orderid": "ORD004"})
        client.place_buy_order("TCS", 5, 0.0, order_type="MARKET")
        called_params = mock_smart.placeOrder.call_args[0][0]
        assert called_params["ordertype"] == "MARKET"

    def test_raises_order_rejected_on_failure(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.placeOrder.return_value = {
            "status": False,
            "message": "Insufficient funds",
            "errorcode": "AG1001",
        }
        with pytest.raises(BrokerAPIError, match="Insufficient funds"):
            client.place_buy_order("TCS", 5, 3850.0)


class TestPlaceSellOrder:
    def test_transaction_type_is_sell(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.placeOrder.return_value = _ok({"orderid": "ORD010"})
        client.place_sell_order("TCS", 5, 3900.0)
        called_params = mock_smart.placeOrder.call_args[0][0]
        assert called_params["transactiontype"] == "SELL"


class TestPlaceStopLossOrder:
    def test_sl_order_params(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.placeOrder.return_value = _ok({"orderid": "SL001"})
        result = client.place_stop_loss_order(
            "TCS", quantity=5, trigger_price=3700.0, limit_price=3680.0
        )
        assert result["order_id"] == "SL001"
        params = mock_smart.placeOrder.call_args[0][0]
        assert params["ordertype"] == "STOPLOSS_LIMIT"
        assert params["triggerprice"] == "3700.0"
        assert params["price"] == "3680.0"
        assert params["transactiontype"] == "SELL"
        assert params["variety"] == "STOPLOSS"


class TestModifyOrder:
    def _order_book(self) -> dict:
        return _ok(
            [
                {
                    "orderid": "ORD100",
                    "tradingsymbol": "TCS-EQ",
                    "transactiontype": "BUY",
                    "quantity": 5,
                    "price": 3800.0,
                    "ordertype": "LIMIT",
                    "status": "PENDING",
                    "filledshares": 0,
                    "producttype": "DELIVERY",
                    "variety": "NORMAL",
                }
            ]
        )

    def test_modify_price(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.orderBook.return_value = self._order_book()
        mock_smart.modifyOrder.return_value = _ok({"orderid": "ORD100"})
        result = client.modify_order("ORD100", new_price=3850.0)
        assert result["order_id"] == "ORD100"
        assert result["status"] == "MODIFIED"
        params = mock_smart.modifyOrder.call_args[0][0]
        assert params["price"] == "3850.0"
        assert params["quantity"] == "5"  # unchanged

    def test_raises_for_unknown_order(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.orderBook.return_value = _ok([])
        with pytest.raises(BrokerAPIError, match="not found"):
            client.modify_order("NONEXISTENT")


class TestCancelOrder:
    def test_cancel_returns_cancelled(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.cancelOrder.return_value = _ok("ORD200")
        result = client.cancel_order("ORD200")
        assert result["status"] == "CANCELLED"
        mock_smart.cancelOrder.assert_called_once_with("NORMAL", "ORD200")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class TestCalculateQuantity:
    def test_floor_division(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.ltpData.return_value = _ok({"ltp": 3333.33})
        qty = client.calculate_quantity("TCS", capital_to_deploy=10000.0)
        assert qty == 3   # floor(10000 / 3333.33)

    def test_returns_zero_when_capital_insufficient(
        self, preloaded: tuple
    ) -> None:
        client, mock_smart = preloaded
        mock_smart.ltpData.return_value = _ok({"ltp": 5000.0})
        qty = client.calculate_quantity("TCS", capital_to_deploy=4999.0)
        assert qty == 0

    def test_returns_zero_on_ltp_error(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.ltpData.return_value = {
            "status": False,
            "message": "Error",
            "errorcode": "AB9999",
        }
        qty = client.calculate_quantity("TCS", capital_to_deploy=50000.0)
        assert qty == 0


class TestGetMarginAvailable:
    def test_returns_available_cash(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.rmsLimit.return_value = _ok({"availablecash": "25000.50"})
        margin = client.get_margin_available()
        assert margin == 25000.50

    def test_fallback_to_net_key(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        mock_smart.rmsLimit.return_value = _ok({"net": "18000.0"})
        margin = client.get_margin_available()
        assert margin == 18000.0


# ---------------------------------------------------------------------------
# Error handling & rate-limit retry
# ---------------------------------------------------------------------------

class TestRateLimitRetry:
    def test_retries_on_rate_limit_code(self, preloaded: tuple) -> None:
        client, mock_smart = preloaded
        rate_limit_resp = {
            "status": False,
            "message": "Too many requests",
            "errorcode": "AG8001",
        }
        ok_resp = _ok({"ltp": 3850.0})
        mock_smart.ltpData.side_effect = [rate_limit_resp, ok_resp]

        with patch("src.broker.angel_one.time.sleep"):
            result = client.get_ltp("TCS")

        assert result["ltp"] == 3850.0
        assert mock_smart.ltpData.call_count == 2

    def test_raises_rate_limit_error_after_max_retries(
        self, preloaded: tuple
    ) -> None:
        client, mock_smart = preloaded
        rate_limit_resp = {
            "status": False,
            "message": "Too many requests",
            "errorcode": "AG8001",
        }
        mock_smart.ltpData.return_value = rate_limit_resp

        with patch("src.broker.angel_one.time.sleep"):
            with pytest.raises(RateLimitError):
                client.get_ltp("TCS")

        assert mock_smart.ltpData.call_count == client._API_MAX_RETRIES

    def test_raises_broker_api_error_on_sdk_exception(
        self, preloaded: tuple
    ) -> None:
        client, mock_smart = preloaded
        mock_smart.ltpData.side_effect = ConnectionError("socket closed")
        with pytest.raises(BrokerAPIError, match="socket closed"):
            client.get_ltp("TCS")


# ---------------------------------------------------------------------------
# Date-range splitting
# ---------------------------------------------------------------------------

class TestSplitDateRange:
    def test_single_chunk_within_limit(self) -> None:
        chunks = AngelOneClient._split_date_range(
            "2026-01-01 09:15", "2026-01-05 15:30", max_days=30
        )
        assert len(chunks) == 1
        assert chunks[0][0] == "2026-01-01 09:15"
        assert chunks[0][1] == "2026-01-05 15:30"

    def test_splits_across_max_days(self) -> None:
        # 35 days with max_days=10 → 4 chunks
        chunks = AngelOneClient._split_date_range(
            "2026-01-01 09:15", "2026-02-05 15:30", max_days=10
        )
        assert len(chunks) >= 3
        # Chunks must be contiguous (each chunk_from ≤ previous chunk_to + 1 min)
        for i in range(1, len(chunks)):
            assert chunks[i][0] >= chunks[i - 1][1]

    def test_no_empty_chunks(self) -> None:
        chunks = AngelOneClient._split_date_range(
            "2026-03-01 09:15", "2026-03-25 15:30", max_days=7
        )
        for start, end in chunks:
            assert start < end


class TestMaxDaysForInterval:
    def test_one_day(self) -> None:
        assert AngelOneClient._max_days_for_interval("ONE_DAY") == 365

    def test_one_minute(self) -> None:
        assert AngelOneClient._max_days_for_interval("ONE_MINUTE") == 10

    def test_fifteen_minute(self) -> None:
        assert AngelOneClient._max_days_for_interval("FIFTEEN_MINUTE") == 30

    def test_unknown_defaults_to_30(self) -> None:
        assert AngelOneClient._max_days_for_interval("UNKNOWN") == 30
