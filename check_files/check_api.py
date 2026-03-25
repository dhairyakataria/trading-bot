"""Angel One API connectivity checker.

Runs a series of read-only API calls to verify your credentials and
connections are working.  No orders are placed.

Usage (from the trading-bot/ directory):
    python check_api.py

Add ANGEL_ONE_PASSWORD to your .env file (your 4-digit MPIN), or the
script will prompt you for it at runtime.
"""
from __future__ import annotations

import getpass
import json
import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add project root to sys.path so src.* imports work
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv

load_dotenv(HERE / ".env", override=False)

# ---------------------------------------------------------------------------
# Credentials from environment
# ---------------------------------------------------------------------------

CLIENT_ID   = os.getenv("ANGEL_ONE_CLIENT_ID", "")
API_KEY     = os.getenv("ANGEL_ONE_API_KEY", "")
TOTP_SECRET = os.getenv("ANGEL_ONE_TOTP_SECRET", "")
PASSWORD    = os.getenv("ANGEL_ONE_PASSWORD", "")

# Validate that the mandatory fields are present
missing = [k for k, v in {
    "ANGEL_ONE_CLIENT_ID":   CLIENT_ID,
    "ANGEL_ONE_API_KEY":     API_KEY,
    "ANGEL_ONE_TOTP_SECRET": TOTP_SECRET,
}.items() if not v]

if missing:
    print(f"\n[ERROR] Missing env vars: {', '.join(missing)}")
    print("       Set them in your .env file and rerun.\n")
    sys.exit(1)

# MPIN is not in .env.example by default — prompt if absent
if not PASSWORD:
    PASSWORD = getpass.getpass("Enter your Angel One MPIN (not stored anywhere): ")

# ---------------------------------------------------------------------------
# Build a minimal config dict for AngelOneClient
# ---------------------------------------------------------------------------
CONFIG = {
    "broker": {
        "angel_one": {
            "client_id":       CLIENT_ID,
            "api_key":         API_KEY,
            "totp_secret":     TOTP_SECRET,
            "password":        PASSWORD,
            "default_exchange": "NSE",
        }
    }
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEPARATOR = "─" * 60
PASS = "  ✔"
FAIL = "  ✘"


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def ok(label: str, value: str = "") -> None:
    suffix = f"  →  {value}" if value else ""
    print(f"{PASS}  {label}{suffix}")


def fail(label: str, err: str) -> None:
    print(f"{FAIL}  {label}")
    print(f"       {err}")


def pretty(data: object, indent: int = 6) -> str:
    """JSON-pretty-print a dict/list, indented by *indent* spaces."""
    pad = " " * indent
    return json.dumps(data, indent=2, default=str).replace("\n", f"\n{pad}")


# ---------------------------------------------------------------------------
# Import broker
# ---------------------------------------------------------------------------
try:
    from src.broker.angel_one import (
        AngelOneClient,
        BrokerAuthError,
        BrokerAPIError,
    )
except ImportError as e:
    print(f"\n[ERROR] Could not import AngelOneClient: {e}")
    print("        Run:  pip install smartapi-python pyotp pandas requests\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Main check routine
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "═" * 60)
    print("  Angel One API Connectivity Checker")
    print("═" * 60)
    print(f"  Client ID : {CLIENT_ID[:3]}{'*' * (len(CLIENT_ID) - 3)}")
    print(f"  API Key   : {API_KEY[:4]}{'*' * 8}  (masked)")

    client = AngelOneClient(CONFIG)

    # ------------------------------------------------------------------
    # 1. Login
    # ------------------------------------------------------------------
    section("1 · Login")
    try:
        client.login()
        ok("Login successful")
        ok("Session active", str(client.is_authenticated()))
        ok("Session expires", client._session_expiry.strftime("%Y-%m-%d %H:%M:%S"))
    except BrokerAuthError as e:
        fail("Login FAILED", str(e))
        print("\n  Cannot continue without a valid session — exiting.\n")
        sys.exit(1)
    except Exception as e:
        fail("Login FAILED (unexpected error)", traceback.format_exc())
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Instrument master + symbol lookup
    # ------------------------------------------------------------------
    section("2 · Instrument Master & Symbol Lookup")
    try:
        master = client.download_instrument_master()
        ok("Instrument master loaded", f"{len(master):,} NSE equity instruments")

        test_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        found, missing_syms = [], []
        for sym in test_symbols:
            try:
                tok = client.symbol_to_token(sym)
                found.append(f"{sym}→{tok}")
            except BrokerAPIError:
                missing_syms.append(sym)

        ok("Token lookup", ",  ".join(found))
        if missing_syms:
            fail("Could not resolve", ", ".join(missing_syms))
    except Exception as e:
        fail("Instrument master failed", str(e))

    # ------------------------------------------------------------------
    # 3. Live LTP (single)
    # ------------------------------------------------------------------
    section("3 · Live LTP — Single Symbol")
    for sym in ["RELIANCE", "NIFTY 50"]:
        try:
            data = client.get_ltp(sym)
            ok(f"LTP({sym})", f"₹{data['ltp']:,.2f}  (token {data['token']})")
        except BrokerAPIError as e:
            fail(f"LTP({sym})", str(e))
        except Exception as e:
            fail(f"LTP({sym}) — unexpected", str(e))

    # ------------------------------------------------------------------
    # 4. Live LTP (batch)
    # ------------------------------------------------------------------
    section("4 · Live LTP — Batch")
    batch_symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "WIPRO"]
    try:
        results = client.get_ltp_batch(batch_symbols)
        ok(f"Batch returned {len(results)}/{len(batch_symbols)} symbols")
        for r in results:
            print(f"       {r['symbol']:<15} ₹{r['ltp']:>10,.2f}")
    except Exception as e:
        fail("Batch LTP failed", str(e))

    # ------------------------------------------------------------------
    # 5. Holdings (demat / delivery)
    # ------------------------------------------------------------------
    section("5 · Holdings (Demat / Delivery)")
    try:
        holdings = client.get_holdings()
        if holdings:
            ok(f"{len(holdings)} holding(s) found")
            for h in holdings:
                pnl_sign = "+" if h["pnl"] >= 0 else ""
                print(
                    f"       {h['symbol']:<15} "
                    f"qty={h['quantity']:<6} "
                    f"avg=₹{h['avg_price']:>9,.2f}  "
                    f"ltp=₹{h['ltp']:>9,.2f}  "
                    f"P&L={pnl_sign}₹{h['pnl']:,.2f}"
                )
        else:
            ok("Holdings fetched — portfolio is empty (no delivery positions)")
    except Exception as e:
        fail("Holdings failed", str(e))

    # ------------------------------------------------------------------
    # 6. Positions (intraday / open)
    # ------------------------------------------------------------------
    section("6 · Open Positions (Today)")
    try:
        positions = client.get_positions()
        if positions:
            ok(f"{len(positions)} open position(s)")
            for p in positions:
                print(
                    f"       {p['symbol']:<15} "
                    f"qty={p['quantity']:<6} "
                    f"avg=₹{p['avg_price']:>9,.2f}  "
                    f"product={p['product']}"
                )
        else:
            ok("Positions fetched — no open positions today")
    except Exception as e:
        fail("Positions failed", str(e))

    # ------------------------------------------------------------------
    # 7. Portfolio value summary
    # ------------------------------------------------------------------
    section("7 · Portfolio Value Summary")
    try:
        pv = client.get_portfolio_value()
        ok("Portfolio value fetched")
        print(f"       Total Value      : ₹{pv['total_value']:>12,.2f}")
        print(f"       Invested         : ₹{pv['invested']:>12,.2f}")
        print(f"       Available Cash   : ₹{pv['available_cash']:>12,.2f}")
        print(f"       Unrealised P&L   : ₹{pv['total_pnl']:>+12,.2f}")
    except Exception as e:
        fail("Portfolio value failed", str(e))

    # ------------------------------------------------------------------
    # 8. Available margin
    # ------------------------------------------------------------------
    section("8 · Available Margin / Cash")
    try:
        margin = client.get_margin_available()
        ok("Margin fetched", f"₹{margin:,.2f}")
    except Exception as e:
        fail("Margin failed", str(e))

    # ------------------------------------------------------------------
    # 9. Order book (today's orders)
    # ------------------------------------------------------------------
    section("9 · Order Book (Today)")
    try:
        orders = client.get_order_book()
        if orders:
            ok(f"{len(orders)} order(s) in today's book")
            for o in orders[:10]:   # show max 10 to keep output clean
                print(
                    f"       {o['order_id']:<15} "
                    f"{o['transaction_type']:<5} "
                    f"{o['symbol']:<15} "
                    f"qty={o['quantity']:<6} "
                    f"price=₹{o['price']:>8,.2f}  "
                    f"status={o['status']}"
                )
            if len(orders) > 10:
                print(f"       ... and {len(orders) - 10} more")
        else:
            ok("Order book fetched — no orders placed today")
    except Exception as e:
        fail("Order book failed", str(e))

    # ------------------------------------------------------------------
    # 10. Order status (first pending order, if any)
    # ------------------------------------------------------------------
    section("10 · Order Status (first order in book)")
    try:
        orders = client.get_order_book()
        if orders:
            first_id = orders[0]["order_id"]
            status = client.get_order_status(first_id)
            ok(f"Order {first_id}")
            print(f"       Status     : {status['status']}")
            print(f"       Symbol     : {status['symbol']}")
            print(f"       Type       : {status['transaction_type']}")
            print(f"       Filled qty : {status['filled_qty']}")
            print(f"       Price      : ₹{status['price']:,.2f}")
        else:
            ok("No orders to check status for — skipping")
    except Exception as e:
        fail("Order status failed", str(e))

    # ------------------------------------------------------------------
    # 11. Historical data (last 5 days of TCS daily candles)
    # ------------------------------------------------------------------
    section("11 · Historical Data (TCS · last 5 days · daily candles)")
    try:
        from datetime import datetime, timedelta
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=7)
        df = client.get_historical_data(
            symbol="TCS",
            interval="ONE_DAY",
            from_date=from_dt.strftime("%Y-%m-%d 09:15"),
            to_date=to_dt.strftime("%Y-%m-%d 15:30"),
        )
        if not df.empty:
            ok(f"Got {len(df)} candle(s)")
            print(f"       {'Date':<22} {'Open':>9} {'High':>9} {'Low':>9} {'Close':>9} {'Volume':>12}")
            print(f"       {'─'*22} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*12}")
            for _, row in df.tail(5).iterrows():
                print(
                    f"       {str(row['datetime']):<22} "
                    f"{row['open']:>9,.2f} "
                    f"{row['high']:>9,.2f} "
                    f"{row['low']:>9,.2f} "
                    f"{row['close']:>9,.2f} "
                    f"{int(row['volume']):>12,}"
                )
        else:
            fail("Historical data", "Empty response (market may be closed or symbol not found)")
    except Exception as e:
        fail("Historical data failed", str(e))

    # ------------------------------------------------------------------
    # 12. Quantity calculator
    # ------------------------------------------------------------------
    section("12 · Quantity Calculator")
    for sym, capital in [("RELIANCE", 25000), ("TCS", 50000), ("INFY", 10000)]:
        try:
            qty = client.calculate_quantity(sym, capital)
            ltp = client.get_ltp(sym)["ltp"]
            ok(
                f"Capital ₹{capital:,} ÷ {sym} LTP ₹{ltp:,.2f}",
                f"= {qty} share(s)  (cost ≈ ₹{qty * ltp:,.2f})",
            )
        except Exception as e:
            fail(f"calculate_quantity({sym})", str(e))

    # ------------------------------------------------------------------
    # Logout
    # ------------------------------------------------------------------
    section("Cleanup")
    try:
        client.logout()
        ok("Logged out cleanly")
    except Exception as e:
        fail("Logout raised (non-critical)", str(e))

    print(f"\n{'═' * 60}")
    print("  All checks complete.")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
