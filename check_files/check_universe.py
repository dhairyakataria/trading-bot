"""Universe Agent — end-to-end smoke test.

Runs the full UniverseAgent pipeline and prints every step so you can see
exactly what is happening.  Two modes:

  LIVE   — uses your Angel One credentials (.env) to fetch real NSE data.
  DEMO   — uses yfinance (free, no credentials needed) as a drop-in data source.

The script auto-detects which mode to use.  Pass --live or --demo to override.

Usage (from the trading-bot/ directory):
    python check_universe.py              # auto mode
    python check_universe.py --demo       # force yfinance / no credentials needed
    python check_universe.py --live       # force Angel One (will fail if no .env)
    python check_universe.py --sample 10  # test on N stocks instead of default 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env", override=False)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Output helpers  (same style as check_api.py)
# ---------------------------------------------------------------------------
SEP  = "─" * 65
SEP2 = "═" * 65
OK   = "  ✔"
FAIL = "  ✘"
SKIP = "  ○"
INFO = "    "

_results: list[tuple[str, str]] = []


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def ok(label: str, detail: str = "") -> None:
    suffix = f"  →  {detail}" if detail else ""
    print(f"{OK}  {label}{suffix}")
    _results.append((label, "PASS"))


def fail(label: str, detail: str = "") -> None:
    print(f"{FAIL}  {label}")
    if detail:
        print(f"{INFO}     {detail[:120]}")
    _results.append((label, "FAIL"))


def skip(label: str, reason: str = "") -> None:
    suffix = f"  (skipped: {reason})" if reason else ""
    print(f"{SKIP}  {label}{suffix}")
    _results.append((label, "SKIP"))


def info(text: str) -> None:
    print(f"{INFO}  {text}")


def summary() -> None:
    print(f"\n{SEP2}\n  SUMMARY\n{SEP2}")
    for label, status in _results:
        icon = "✔" if status == "PASS" else ("○" if status == "SKIP" else "✘")
        print(f"  {icon}  [{status:<4}]  {label}")
    passed  = sum(1 for _, s in _results if s == "PASS")
    failed  = sum(1 for _, s in _results if s == "FAIL")
    skipped = sum(1 for _, s in _results if s == "SKIP")
    print(f"\n  {passed} passed  ·  {failed} failed  ·  {skipped} skipped")
    print(f"{SEP2}\n")


# ---------------------------------------------------------------------------
# Demo broker — wraps yfinance so UniverseAgent thinks it has a real broker
# ---------------------------------------------------------------------------

class _YFinanceBroker:
    """Thin yfinance wrapper that mirrors AngelOneClient's interface."""

    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        from_date: str,
        to_date: str,
        exchange: str = "NSE",
    ):
        import pandas as pd
        try:
            import yfinance as yf
        except ImportError:
            return pd.DataFrame(
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )
        # Angel One uses "YYYY-MM-DD HH:MM" format
        from_dt = datetime.strptime(from_date, "%Y-%m-%d %H:%M").strftime("%Y-%m-%d")
        to_dt   = datetime.strptime(to_date,   "%Y-%m-%d %H:%M").strftime("%Y-%m-%d")

        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(start=from_dt, end=to_dt, interval="1d", auto_adjust=True)
        if df.empty:
            return pd.DataFrame(
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c), None)
        if date_col and date_col != "datetime":
            df = df.rename(columns={date_col: "datetime"})
        return df[["datetime", "open", "high", "low", "close", "volume"]].copy()

    def get_ltp(self, symbol: str, exchange: str = "NSE") -> dict:
        try:
            import yfinance as yf
            price = yf.Ticker(f"{symbol}.NS").fast_info.last_price or 0.0
            return {"symbol": symbol, "ltp": float(price)}
        except Exception:
            return {"symbol": symbol, "ltp": 0.0}


# ---------------------------------------------------------------------------
# Minimal config shim  (avoids needing a fully-configured config.yaml)
# ---------------------------------------------------------------------------

class _Cfg:
    _data = {
        ("trading", "min_stock_price"):   50,
        ("trading", "max_stock_price"):   5000,
        ("trading", "min_volume_cr"):     10,
        ("universe", "blacklisted_stocks"): [],
    }
    def get(self, *keys: str, default: Any = None) -> Any:
        return self._data.get(keys, default)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(value: int, total: int, width: int = 20) -> str:
    filled = round(width * value / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def _show_filter_diagnostics(agent: Any, sample_symbols: list) -> None:
    """Print per-stock filter diagnostics using data already in the agent's cache.

    Called when apply_daily_filters() returns 0 results so the user can see
    exactly which filter rejected each stock and by how much.
    """
    from src.tools.technical_indicators import TechnicalIndicators
    ti = TechnicalIndicators()

    min_price = float(agent._config.get("trading", "min_stock_price") or 50)
    max_price = float(agent._config.get("trading", "max_stock_price") or 5000)
    min_vol   = float(agent._config.get("trading", "min_volume_cr")   or 10)

    info("")
    info(f"  {'Symbol':<12} {'Price':>8}  {'EMA50':>8}  {'%vsEMA':>7}  {'Vol(Cr)':>7}  {'ATR%':>5}  Why rejected")
    info("  " + "─" * 78)

    for sym in sample_symbols:
        df = agent._hist_cache.get(sym)
        if df is None or df.empty or len(df) < 20:
            info(f"  {sym:<12}  (no data fetched)")
            continue

        price     = float(df["close"].iloc[-1])
        recent    = df.tail(20)
        avg_tv_cr = float((recent["close"] * recent["volume"]).mean()) / 1e7

        ema_res = ti.calculate_ema(df, periods=[50])
        ema50   = ema_res.get("ema_50")
        vs_ema  = f"{(price-ema50)/ema50*100:+.1f}%" if ema50 else "  N/A"

        atr_res = ti.calculate_atr(df, period=14)
        atr_pct = atr_res.get("atr_pct") if "error" not in atr_res else None
        atr_str = f"{atr_pct:.1f}%" if atr_pct is not None else " N/A"

        # Determine rejection reason (same order as filters)
        fails: list[str] = []
        if not (min_price <= price <= max_price):
            fails.append(f"price ₹{price:.0f} outside [{min_price:.0f}-{max_price:.0f}]")
        if not fails and avg_tv_cr < min_vol:
            fails.append(f"vol {avg_tv_cr:.1f}Cr < {min_vol}Cr min")
        if not fails and ema50 and price <= ema50:
            fails.append(f"below EMA50 by {abs((price-ema50)/ema50*100):.1f}%")
        if not fails and atr_pct is not None and not (1.5 <= atr_pct <= 6.0):
            dir_ = "low" if atr_pct < 1.5 else "high"
            fails.append(f"ATR {atr_pct:.1f}% too {dir_} [1.5-6.0%]")

        reason   = " | ".join(fails) if fails else "PASS"
        ema_disp = f"₹{ema50:>7,.0f}" if ema50 else "     N/A"
        info(
            f"  {sym:<12} ₹{price:>7,.0f}  {ema_disp}  {vs_ema:>7}  "
            f"{avg_tv_cr:>7.1f}  {atr_str:>5}  {reason}"
        )


def _yfinance_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("yfinance") is not None


def _angel_creds_available() -> bool:
    return all(
        os.getenv(k)
        for k in ("ANGEL_ONE_CLIENT_ID", "ANGEL_ONE_API_KEY", "ANGEL_ONE_TOTP_SECRET")
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Universe Agent smoke test")
    parser.add_argument("--live",   action="store_true", help="Force Angel One live mode")
    parser.add_argument("--demo",   action="store_true", help="Force yfinance demo mode")
    parser.add_argument("--sample", type=int, default=8,
                        help="Number of sample stocks to run filters on (default 8)")
    parser.add_argument("--symbols", nargs="+", metavar="SYM",
                        help="Specific symbols to test (overrides --sample)")
    args = parser.parse_args()

    # Decide mode
    if args.live:
        mode = "live"
    elif args.demo:
        mode = "demo"
    elif _angel_creds_available():
        mode = "live"
    elif _yfinance_available():
        mode = "demo"
    else:
        mode = "none"

    print(f"\n{SEP2}")
    print("  Universe Agent — End-to-End Smoke Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   mode={mode.upper()}")
    print(SEP2)

    if mode == "none":
        print("\n[ERROR] No data source available.")
        print("        Either configure Angel One credentials in .env (LIVE mode)")
        print("        or install yfinance:   pip install yfinance   (DEMO mode)\n")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 1. Imports                                                           #
    # ------------------------------------------------------------------ #
    section("1 · Imports")
    try:
        from src.agents.universe_agent import UniverseAgent, _FALLBACK_CONSTITUENTS
        from src.database.db_manager import DatabaseManager
        ok("UniverseAgent imported")
    except ImportError as exc:
        fail("Import failed", str(exc))
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Database                                                          #
    # ------------------------------------------------------------------ #
    section("2 · Database")
    db_path = HERE / "data" / "trading_bot.db"
    try:
        db = DatabaseManager(str(db_path))
        ok("DatabaseManager ready", str(db_path.relative_to(HERE)))
    except Exception as exc:
        fail("DatabaseManager failed", str(exc))
        db = None

    # ------------------------------------------------------------------ #
    # 3. Broker / data source                                              #
    # ------------------------------------------------------------------ #
    section("3 · Data Source")
    broker = None

    if mode == "live":
        try:
            from src.broker.angel_one import AngelOneClient, BrokerAuthError
            cfg_dict = {
                "broker": {
                    "angel_one": {
                        "client_id":       os.getenv("ANGEL_ONE_CLIENT_ID"),
                        "api_key":         os.getenv("ANGEL_ONE_API_KEY"),
                        "totp_secret":     os.getenv("ANGEL_ONE_TOTP_SECRET"),
                        "password":        os.getenv("ANGEL_ONE_PASSWORD", ""),
                        "default_exchange": "NSE",
                    }
                }
            }
            if not cfg_dict["broker"]["angel_one"]["password"]:
                import getpass
                cfg_dict["broker"]["angel_one"]["password"] = getpass.getpass(
                    "Enter your Angel One MPIN: "
                )
            broker = AngelOneClient(cfg_dict)
            broker.login()
            ok("Angel One login successful")
            # Quick probe
            ltp_probe = broker.get_ltp("TCS")
            ok("LTP probe — TCS", f"₹{ltp_probe['ltp']:,.2f}")
        except Exception as exc:
            fail("Angel One broker", str(exc))
            if _yfinance_available():
                info("Falling back to yfinance demo mode …")
                broker = _YFinanceBroker()
                mode = "demo"
            else:
                info("Install yfinance (pip install yfinance) to run in demo mode.")
                sys.exit(1)

    else:  # demo
        if not _yfinance_available():
            fail("yfinance not installed", "pip install yfinance")
            sys.exit(1)
        broker = _YFinanceBroker()
        # Quick probe
        try:
            probe = broker.get_ltp("TCS")
            ok("yfinance demo broker ready")
            ok("LTP probe — TCS", f"₹{probe['ltp']:,.2f}")
        except Exception as exc:
            fail("yfinance probe failed", str(exc))

    # ------------------------------------------------------------------ #
    # 4. Index Constituents                                                #
    # ------------------------------------------------------------------ #
    section("4 · Index Constituents")
    cfg = _Cfg()
    agent = UniverseAgent(broker=broker, db=db, config=cfg)

    try:
        t0 = time.perf_counter()
        result = agent.refresh_index_constituents()
        elapsed = time.perf_counter() - t0
        ok(
            "refresh_index_constituents()",
            f"{result['total_universe']} unique symbols  [{elapsed:.2f}s]",
        )
        info(f"  Nifty 50          : {result['nifty_50']} stocks")
        info(f"  Nifty Next 50     : {result['next_50']} stocks")
        info(f"  Nifty Midcap Sel. : {result['midcap_select']} stocks")
    except Exception as exc:
        fail("refresh_index_constituents()", str(exc))

    # ------------------------------------------------------------------ #
    # 5. Base Universe                                                     #
    # ------------------------------------------------------------------ #
    section("5 · Base Universe")
    try:
        base = agent.get_base_universe()
        ok("get_base_universe()", f"{len(base)} symbols")
        info(f"  First 10: {', '.join(base[:10])}")
        info(f"  Last  10: {', '.join(base[-10:])}")
    except Exception as exc:
        fail("get_base_universe()", str(exc))
        base = list(_FALLBACK_CONSTITUENTS.get("NIFTY_50", []))[:args.sample]

    # ------------------------------------------------------------------ #
    # 6. Daily Filters — sample run                                        #
    # ------------------------------------------------------------------ #

    # Default: a spread of well-known liquid Nifty 50 stocks that are
    # stable in price, high in volume, and often pass all 4 filters.
    # User can override with --symbols TCS INFY ... or --sample N.
    _GOOD_DEFAULTS = [
        "TCS", "INFY", "HDFCBANK", "ICICIBANK", "WIPRO",
        "AXISBANK", "SBIN", "RELIANCE", "BHARTIARTL", "KOTAKBANK",
    ]

    if args.symbols:
        sample_symbols = [s.upper() for s in args.symbols]
    elif args.sample <= len(_GOOD_DEFAULTS):
        sample_symbols = _GOOD_DEFAULTS[:args.sample]
    else:
        # User asked for more than 10 — extend with base universe
        extras = [s for s in base if s not in _GOOD_DEFAULTS]
        sample_symbols = _GOOD_DEFAULTS + extras[:args.sample - len(_GOOD_DEFAULTS)]

    section(f"6 · Daily Filters — sample of {len(sample_symbols)} stocks")
    info(f"  Testing: {', '.join(sample_symbols)}")
    info(f"  Tip: pass --symbols TCS INFY SBIN to test specific stocks")
    info("")

    t0 = time.perf_counter()
    try:
        filtered = agent.apply_daily_filters(sample_symbols)
        elapsed = time.perf_counter() - t0
        n = len(sample_symbols)

        if filtered:
            ok(
                f"apply_daily_filters() on {n} stocks",
                f"{len(filtered)}/{n} passed  [{elapsed:.1f}s]",
            )
            # Table
            info(
                f"  {'Symbol':<14} {'Price':>8} {'AvgVol(Cr)':>11} "
                f"{'EMA50':>8} {'%AbvEMA':>8} {'ATR%':>6}  Sector"
            )
            info("  " + "─" * 75)
            for s in filtered:
                info(
                    f"  {s['symbol']:<14} "
                    f"₹{s['price']:>7,.1f} "
                    f"{s['avg_volume_cr']:>10.1f} "
                    f"₹{s['ema_50']:>7,.1f} "
                    f"{s['price_above_ema_pct']:>+7.1f}% "
                    f"{s['atr_pct']:>5.1f}%"
                    f"  {s['sector']}"
                )
            rejected = [s for s in sample_symbols if s not in {r["symbol"] for r in filtered}]
            if rejected:
                info("")
                info(f"  Rejected ({len(rejected)}): {', '.join(rejected)}")
                info("  Per-stock breakdown for all tested stocks:")
                _show_filter_diagnostics(agent, sample_symbols)
        else:
            # All filtered out — this is valid market data, not a bug.
            # Show per-stock diagnostics so the user can see the exact numbers.
            fail(
                f"apply_daily_filters() on {n} stocks",
                f"0/{n} passed in {elapsed:.1f}s",
            )
            info("  All stocks failed the filters — per-stock breakdown:")
            _show_filter_diagnostics(agent, sample_symbols)
            info("")
            info("  This is real market data, not a bug.")
            info("  Sections 7-10 will run with a synthetic demo watchlist to")
            info("  verify the remaining pipeline logic.")
    except Exception as exc:
        fail("apply_daily_filters()", str(exc))
        filtered = []

    # If nothing passed, build a minimal synthetic watchlist for sections 7-10
    # so we can still exercise all code paths.
    if not filtered:
        info("")
        info("  Building synthetic demo watchlist …")
        filtered = [
            {
                "symbol": sym, "price": 1500.0, "avg_volume_cr": 80.0,
                "atr_pct": 2.5, "ema_50": 1450.0, "price_above_ema_pct": 3.4,
                "sector": agent._sector_map.get(sym, "UNKNOWN"),
                "index": "NIFTY_50", "added_date": datetime.now().strftime("%Y-%m-%d"),
            }
            for sym in sample_symbols[:3]
        ]
        info(f"  Demo watchlist: {[s['symbol'] for s in filtered]}")

    # ------------------------------------------------------------------ #
    # 7. Active Watchlist                                                  #
    # ------------------------------------------------------------------ #
    section("7 · Active Watchlist")

    # Patch apply_daily_filters to return what we already computed so we
    # don't make a second round of API calls.  This tests the orchestration
    # logic (sort, cap, DB save) while reusing the filter output from §6.
    from unittest.mock import patch

    try:
        temp_agent = UniverseAgent(broker=broker, db=None, config=cfg)
        temp_agent._API_DELAY = 0  # no sleeping — data already fetched
        with patch.object(temp_agent, "get_base_universe", return_value=sample_symbols):
            with patch.object(temp_agent, "apply_daily_filters", return_value=filtered):
                t0 = time.perf_counter()
                watchlist = temp_agent.get_active_watchlist()
                elapsed = time.perf_counter() - t0
        ok(
            "get_active_watchlist()",
            f"{len(watchlist)} stocks  sorted by ATR% desc, capped at 50  [{elapsed:.3f}s]",
        )
        if watchlist:
            info(f"  Top stock : {watchlist[0]['symbol']}  ATR%={watchlist[0]['atr_pct']}")
            info(f"  Last stock: {watchlist[-1]['symbol']}  ATR%={watchlist[-1]['atr_pct']}")
    except Exception as exc:
        fail("get_active_watchlist()", str(exc))
        watchlist = filtered

    # ------------------------------------------------------------------ #
    # 8. Sector Distribution                                               #
    # ------------------------------------------------------------------ #
    section("8 · Sector Distribution")
    if watchlist:
        try:
            dist = agent.get_sector_distribution(watchlist)
            ok("get_sector_distribution()", f"{len(dist)} sector(s)")
            total = sum(dist.values())
            info(f"  {'Sector':<22} {'Count':>5}  Distribution")
            info("  " + "─" * 50)
            for sector, count in dist.items():
                bar = _bar(count, total, width=16)
                info(f"  {sector:<22} {count:>5}  {bar}  {count/total*100:.0f}%")
        except Exception as exc:
            fail("get_sector_distribution()", str(exc))
    else:
        skip("get_sector_distribution()", "watchlist is empty")

    # ------------------------------------------------------------------ #
    # 9. Held Stock Force-Inclusion                                        #
    # ------------------------------------------------------------------ #
    section("9 · Held Stock Force-Inclusion")
    if watchlist:
        # Pretend we hold one stock from the watchlist and one that isn't in it
        in_list  = watchlist[0]["symbol"]
        not_in   = "RELIANCE" if in_list != "RELIANCE" else "TCS"
        held     = [in_list, not_in]
        info(f"  Simulating held positions: {held}")
        try:
            extended = agent.ensure_held_stocks_in_watchlist(watchlist, held)
            present  = {s["symbol"] for s in extended}
            if in_list in present:
                ok(f"'{in_list}' (already in list) — no duplicate")
            else:
                fail(f"'{in_list}' missing from extended watchlist")
            forced = next((s for s in extended if s["symbol"] == not_in), None)
            if forced:
                ok(
                    f"'{not_in}' (not in list) — force-added",
                    f"reason={forced.get('reason')}  price=₹{forced.get('price') or 'N/A'}",
                )
            else:
                fail(f"'{not_in}' was not force-added")
            # No duplicates
            symbols = [s["symbol"] for s in extended]
            dups = [s for s in symbols if symbols.count(s) > 1]
            if not dups:
                ok("No duplicate symbols in extended watchlist")
            else:
                fail("Duplicates found", str(set(dups)))
        except Exception as exc:
            fail("ensure_held_stocks_in_watchlist()", str(exc))
    else:
        skip("ensure_held_stocks_in_watchlist()", "watchlist is empty")

    # ------------------------------------------------------------------ #
    # 10. DB Persistence & Cache                                           #
    # ------------------------------------------------------------------ #
    section("10 · Database Persistence & Caching")
    if db and watchlist:
        try:
            from src.database.models import WatchlistItem
            today = datetime.now().strftime("%Y-%m-%d")

            # Save to DB
            items = [
                WatchlistItem(
                    symbol=s["symbol"],
                    date=today,
                    price=s.get("price"),
                    avg_volume_cr=s.get("avg_volume_cr"),
                    atr_pct=s.get("atr_pct"),
                    ema_50=s.get("ema_50"),
                    sector=s.get("sector"),
                    in_index=s.get("index"),
                )
                for s in watchlist
            ]
            db.save_watchlist(today, items)
            ok(f"Saved watchlist to DB", f"{len(items)} rows for {today}")

            # Reload from DB
            reloaded = db.get_latest_watchlist()
            if reloaded and reloaded[0].date == today:
                ok("Reloaded from DB", f"{len(reloaded)} rows match")
                # Verify caching path: agent should NOT call broker if DB has today's data
                fresh_agent = UniverseAgent(broker=broker, db=db, config=cfg)
                fresh_agent._API_DELAY = 0
                with patch.object(fresh_agent, "apply_daily_filters") as mock_flt:
                    with patch.object(fresh_agent, "get_base_universe", return_value=sample_symbols):
                        cached_result = fresh_agent.get_active_watchlist()
                if not mock_flt.called:
                    ok("Cache hit: apply_daily_filters() was NOT re-run")
                else:
                    fail("Cache miss: filters were re-run despite valid DB cache")
            else:
                fail("Reloaded watchlist doesn't match today's date")
        except Exception as exc:
            fail("DB persistence check", str(exc))
    elif not db:
        skip("DB persistence check", "DatabaseManager unavailable")
    else:
        skip("DB persistence check", "watchlist is empty")

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    if mode == "live" and hasattr(broker, "logout"):
        try:
            broker.logout()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    summary()


if __name__ == "__main__":
    main()
