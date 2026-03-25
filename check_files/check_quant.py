"""QuantAgent — end-to-end smoke test.

Verifies the quant agent at three levels:

  A. Risk management helpers (pure math — always fast, no data needed)
  B. Strategy detectors with synthetic DataFrames (no broker needed)
  C. Full scan against real market data via yfinance (DEMO) or Angel One (LIVE)

Usage (from the trading-bot/ directory):
    python check_quant.py              # auto-detect mode
    python check_quant.py --demo       # force yfinance (no credentials needed)
    python check_quant.py --live       # force Angel One (needs .env)
    python check_quant.py --symbols TCS INFY HDFCBANK
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
from unittest.mock import MagicMock

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
# Output helpers
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
# Demo broker — wraps yfinance to match AngelOneClient's interface
# ---------------------------------------------------------------------------
class _YFinanceBroker:
    """Thin yfinance wrapper that mirrors AngelOneClient.get_historical_data."""

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
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        from_dt = datetime.strptime(from_date, "%Y-%m-%d %H:%M").strftime("%Y-%m-%d")
        to_dt   = datetime.strptime(to_date,   "%Y-%m-%d %H:%M").strftime("%Y-%m-%d")
        ticker  = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(start=from_dt, end=to_dt, interval="1d", auto_adjust=True)
        if df.empty:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c), None)
        if date_col and date_col != "datetime":
            df = df.rename(columns={date_col: "datetime"})
        return df[["datetime", "open", "high", "low", "close", "volume"]].copy()


# ---------------------------------------------------------------------------
# Synthetic DataFrame builders used in Section B
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd


def _make_df(n: int = 60, close: float = 1000.0) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with constant close price."""
    rng   = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    closes = np.full(n, close) + rng.normal(0, 1, n)
    closes[-1] = close
    return pd.DataFrame({
        "datetime": dates,
        "open":     closes + 3.0,
        "high":     closes + 15.0,
        "low":      closes - 15.0,
        "close":    closes,
        "volume":   np.full(n, 2_000_000.0),
    })


def _inds(
    rsi_val: float = 50.0,
    macd_sig: str = "BULLISH",
    macd_hist: float = 1.0,
    ema_20: float = 980.0,
    ema_50: float = 950.0,
    price_vs_20: str = "ABOVE",
    price_vs_50: str = "ABOVE",
    vol_ratio: float = 1.0,
    atr_val: float = 25.0,
    atr_pct: float = 2.5,
    resistance_1: float = 1050.0,
    resistance_2: float = 1100.0,
) -> dict:
    return {
        "rsi":   {"value": rsi_val, "signal": "NEUTRAL"},
        "macd":  {"signal": macd_sig, "histogram": macd_hist, "macd_line": 2.0, "signal_line": 1.5},
        "ema": {
            "ema_20": ema_20, "ema_50": ema_50, "ema_200": 900.0,
            "price_vs_ema_20":  price_vs_20,
            "price_vs_ema_50":  price_vs_50,
            "price_vs_ema_200": "ABOVE",
            "signal": "STRONG_UPTREND",
        },
        "volume": {"volume_ratio": vol_ratio, "current_volume": int(vol_ratio * 2_000_000), "avg_volume": 2_000_000},
        "atr":   {"atr": atr_val, "atr_pct": atr_pct},
        "bollinger": {"signal": "ABOVE_MIDDLE"},
        "support_resistance": {
            "resistance_1": resistance_1, "resistance_2": resistance_2,
            "support_1": 950.0, "support_2": 920.0, "current_price": 1000.0,
        },
        "vwap": {"vwap": 998.0},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _angel_creds_available() -> bool:
    return all(
        os.getenv(k)
        for k in ("ANGEL_ONE_CLIENT_ID", "ANGEL_ONE_API_KEY", "ANGEL_ONE_TOTP_SECRET")
    )


def _yfinance_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("yfinance") is not None


def _show_signal(sig: dict) -> None:
    """Pretty-print a single signal dict."""
    arrow = "▲" if sig["signal"] == "BUY" else "▼"
    info(f"  {arrow} {sig['symbol']:<12}  strategy={sig['strategy_name']}")
    info(f"    signal={sig['signal']}  strength={sig['strength']:.2f}  rr={sig['risk_reward_ratio']:.2f}")
    info(f"    entry=₹{sig['entry_price']:,.2f}  stop=₹{sig['stop_loss']:,.2f}  "
         f"t1=₹{sig['target_1']:,.2f}  t2=₹{sig['target_2']:,.2f}")
    info(f"    RSI={sig['indicators']['rsi']}  "
         f"MACD_hist={sig['indicators']['macd_histogram']:.4f}  "
         f"vol_ratio={sig['indicators']['volume_ratio']:.2f}")
    for reason in sig["reasons"][:3]:
        info(f"    • {reason}")
    if len(sig["reasons"]) > 3:
        info(f"    • … +{len(sig['reasons']) - 3} more reason(s)")


# ---------------------------------------------------------------------------
# Section A — Risk management helpers (pure math)
# ---------------------------------------------------------------------------
def check_risk_math(agent) -> None:
    section("A · Risk Management Helpers (pure math)")

    cases = [
        # (entry, stop, target, expected_rr, label)
        (1000.0, 950.0, 1100.0, 2.0,  "Standard 2:1 setup"),
        (1000.0, 960.0, 1080.0, 2.0,  "40 risk / 80 reward"),
        (1000.0, 1000.0, 1100.0, 0.0, "Zero risk → RR=0"),
        (1000.0, 990.0, 1005.0, 0.5,  "Bad setup RR < 1"),
    ]
    for entry, sl, target, expected, label in cases:
        rr = agent.calculate_risk_reward(entry, sl, target)
        if abs(rr - expected) < 0.01:
            ok(f"calculate_risk_reward — {label}", f"RR={rr:.2f}")
        else:
            fail(f"calculate_risk_reward — {label}", f"expected {expected}, got {rr}")

    # Stop-loss per strategy
    sl_cases = [
        ("RSI_OVERSOLD_BOUNCE", 1000, 50, {}, 900.0,  "price - 2×ATR"),
        ("EMA_PULLBACK",        1000, 50, {"ema_50": 960}, 900.0,  "min(price-2ATR, EMA50×0.995)"),
        ("VOLUME_BREAKOUT",     1020, 20, {"resistance_1": 1010}, round(1010*0.995, 2), "resistance-0.5%"),
        ("TREND_FOLLOWING",     1100, 30, {"ema_20": 1060}, 1060.0, "EMA20"),
    ]
    for strategy, price, atr, kwargs, expected, note in sl_cases:
        sl = agent.calculate_stop_loss(strategy, price, atr, **kwargs)
        if abs(sl - expected) < 0.1:
            ok(f"calculate_stop_loss({strategy})", f"SL={sl:.2f}  ({note})")
        else:
            fail(f"calculate_stop_loss({strategy})", f"expected {expected}, got {sl}")

    # Target calculation
    t1, t2 = agent.calculate_targets("RSI_OVERSOLD_BOUNCE", 1000, 50, {})
    if t1 == 1150.0 and t2 == 1250.0 and t2 > t1:
        ok("calculate_targets(RSI_OVERSOLD_BOUNCE)", f"t1={t1}  t2={t2}")
    else:
        fail("calculate_targets(RSI_OVERSOLD_BOUNCE)", f"t1={t1}  t2={t2}")

    t1, t2 = agent.calculate_targets("EMA_PULLBACK", 1000, 50, {"resistance_1": 1080.0, "resistance_2": 1150.0})
    if t1 == 1080.0 and t2 == 1150.0:
        ok("calculate_targets(EMA_PULLBACK) uses resistance levels", f"t1={t1}  t2={t2}")
    else:
        fail("calculate_targets(EMA_PULLBACK)", f"t1={t1}  t2={t2}")


# ---------------------------------------------------------------------------
# Section B — Strategy detectors with synthetic data
# ---------------------------------------------------------------------------
def check_strategy_rsi_oversold(agent) -> None:
    label = "RSI_OVERSOLD_BOUNCE — valid setup (RSI=28, MACD bullish, vol 1.5×)"
    df   = _make_df(close=1000.0)
    inds = _inds(rsi_val=28.0, macd_sig="BULLISH_CROSSOVER", macd_hist=1.5, vol_ratio=1.5)
    try:
        sig = agent.check_rsi_oversold_bounce(df, "SYNTH", inds)
        if sig and sig["signal"] == "BUY" and sig["stop_loss"] < sig["entry_price"]:
            ok(label, f"SL=₹{sig['stop_loss']:.2f}  T1=₹{sig['target_1']:.2f}  RR={sig['risk_reward_ratio']:.2f}")
        else:
            fail(label, f"result={sig}")
    except Exception as exc:
        fail(label, str(exc))

    # Rejection cases
    for rsi_val, note in [(35.0, "RSI=35 (boundary)"), (45.0, "RSI=45 (neutral)")]:
        lbl = f"RSI_OVERSOLD_BOUNCE — rejects when {note}"
        df2 = _make_df(close=1000.0)
        inds2 = _inds(rsi_val=rsi_val, macd_sig="BULLISH_CROSSOVER", vol_ratio=1.5)
        try:
            sig2 = agent.check_rsi_oversold_bounce(df2, "SYNTH", inds2)
            if sig2 is None:
                ok(lbl)
            else:
                fail(lbl, f"expected None, got signal={sig2['signal']}")
        except Exception as exc:
            fail(lbl, str(exc))


def check_strategy_ema_pullback(agent) -> None:
    label = "EMA_PULLBACK — valid setup (price 0.5% above EMA20, RSI=48, vol=0.7)"
    df   = _make_df(close=1005.0)
    inds = _inds(
        rsi_val=48.0, ema_20=1000.0, ema_50=950.0,
        price_vs_20="ABOVE", price_vs_50="ABOVE", vol_ratio=0.7,
    )
    try:
        sig = agent.check_ema_pullback(df, "SYNTH", inds)
        if sig and sig["signal"] == "BUY":
            ok(label, f"SL=₹{sig['stop_loss']:.2f}  T1=₹{sig['target_1']:.2f}  RR={sig['risk_reward_ratio']:.2f}")
        else:
            fail(label, f"result={sig}")
    except Exception as exc:
        fail(label, str(exc))

    lbl = "EMA_PULLBACK — rejects when vol_ratio=1.0 (distribution, not healthy)"
    df2 = _make_df(close=1005.0)
    inds2 = _inds(rsi_val=48.0, ema_20=1000.0, ema_50=950.0, price_vs_50="ABOVE", vol_ratio=1.0)
    try:
        sig2 = agent.check_ema_pullback(df2, "SYNTH", inds2)
        ok(lbl) if sig2 is None else fail(lbl, f"got signal {sig2['signal']}")
    except Exception as exc:
        fail(lbl, str(exc))


def check_strategy_volume_breakout(agent) -> None:
    label = "VOLUME_BREAKOUT — valid setup (price>R1, vol=2.5×, RSI=58, MACD bullish)"
    df   = _make_df(close=1020.0)
    inds = _inds(
        rsi_val=58.0, macd_sig="BULLISH", macd_hist=2.0, vol_ratio=2.5,
        resistance_1=1010.0, resistance_2=1060.0,
    )
    try:
        sig = agent.check_volume_breakout(df, "SYNTH", inds)
        if sig and sig["signal"] == "BUY":
            ok(label, f"SL=₹{sig['stop_loss']:.2f}  T1=₹{sig['target_1']:.2f}  RR={sig['risk_reward_ratio']:.2f}")
        else:
            fail(label, f"result={sig}")
    except Exception as exc:
        fail(label, str(exc))

    lbl = "VOLUME_BREAKOUT — rejects when price below resistance (no breakout)"
    df2 = _make_df(close=990.0)  # below resistance_1=1010
    inds2 = _inds(rsi_val=58.0, macd_sig="BULLISH", vol_ratio=2.5, resistance_1=1010.0)
    try:
        sig2 = agent.check_volume_breakout(df2, "SYNTH", inds2)
        ok(lbl) if sig2 is None else fail(lbl, f"got signal {sig2['signal']}")
    except Exception as exc:
        fail(lbl, str(exc))


def check_strategy_trend_following(agent) -> None:
    label = "TREND_FOLLOWING — valid setup (stacked EMAs, RSI=58, ATR%=2.7%)"
    df   = _make_df(close=1100.0)
    inds = _inds(
        rsi_val=58.0, macd_sig="BULLISH", macd_hist=3.0,
        ema_20=1070.0, ema_50=1040.0,
        price_vs_20="ABOVE", price_vs_50="ABOVE",
        atr_val=30.0, atr_pct=2.7,
    )
    try:
        sig = agent.check_trend_following(df, "SYNTH", inds)
        if sig and sig["signal"] == "BUY":
            ok(label, f"SL=₹{sig['stop_loss']:.2f} (=EMA20)  RR={sig['risk_reward_ratio']:.2f}")
            if abs(sig["stop_loss"] - 1070.0) < 0.01:
                ok("TREND_FOLLOWING — stop-loss is exactly EMA20")
            else:
                fail("TREND_FOLLOWING — stop-loss should equal EMA20", f"got {sig['stop_loss']}")
        else:
            fail(label, f"result={sig}")
    except Exception as exc:
        fail(label, str(exc))

    lbl = "TREND_FOLLOWING — rejects when ATR%=1.5% (insufficient volatility)"
    df2 = _make_df(close=1100.0)
    inds2 = _inds(
        rsi_val=58.0, macd_hist=3.0,
        ema_20=1070.0, ema_50=1040.0, price_vs_20="ABOVE", price_vs_50="ABOVE",
        atr_pct=1.5,
    )
    try:
        sig2 = agent.check_trend_following(df2, "SYNTH", inds2)
        ok(lbl) if sig2 is None else fail(lbl, f"got signal {sig2['signal']}")
    except Exception as exc:
        fail(lbl, str(exc))


def check_strategy_exit(agent) -> None:
    # Overbought RSI
    label = "EXIT_SIGNAL — RSI overbought (RSI=75)"
    df   = _make_df(close=1200.0)
    inds = _inds(rsi_val=75.0, macd_sig="BULLISH", macd_hist=1.0, vol_ratio=1.0)
    try:
        sig = agent.check_exit_signals(df, "SYNTH", ["SYNTH"], inds)
        if sig and sig["signal"] == "SELL":
            ok(label, f"strength={sig['strength']:.2f}")
        else:
            fail(label, f"result={sig}")
    except Exception as exc:
        fail(label, str(exc))

    # MACD bearish crossover
    label2 = "EXIT_SIGNAL — MACD bearish crossover"
    inds2  = _inds(rsi_val=55.0, macd_sig="BEARISH_CROSSOVER", macd_hist=-0.5, vol_ratio=1.0)
    try:
        sig2 = agent.check_exit_signals(df, "SYNTH", ["SYNTH"], inds2)
        if sig2 and sig2["signal"] == "SELL":
            ok(label2, f"strength={sig2['strength']:.2f}")
        else:
            fail(label2, f"result={sig2}")
    except Exception as exc:
        fail(label2, str(exc))

    # Multiple conditions → combined strength
    label3 = "EXIT_SIGNAL — RSI + MACD crossover → combined strength ≥ 0.75"
    inds3  = _inds(rsi_val=75.0, macd_sig="BEARISH_CROSSOVER", macd_hist=-1.0, vol_ratio=1.0)
    try:
        sig3 = agent.check_exit_signals(df, "SYNTH", ["SYNTH"], inds3)
        if sig3 and sig3["strength"] >= 0.75:
            ok(label3, f"strength={sig3['strength']:.2f}")
        else:
            fail(label3, f"result={sig3}")
    except Exception as exc:
        fail(label3, str(exc))

    # Not held + weak → suppressed
    label4 = "EXIT_SIGNAL — suppressed for un-held stock with weak signal"
    inds4  = _inds(rsi_val=72.0, macd_sig="BULLISH", macd_hist=1.0, vol_ratio=1.0)
    try:
        sig4 = agent.check_exit_signals(df, "SYNTH", [], inds4)  # not in holdings
        if sig4 is None:
            ok(label4)
        else:
            fail(label4, f"expected None, got signal={sig4['signal']}")
    except Exception as exc:
        fail(label4, str(exc))


def check_signal_combining(agent) -> None:
    section("B · Signal Strength Combining")
    def _s(strategy, strength):
        return {
            "signal": "BUY", "strength": strength, "strategy_name": strategy,
            "reasons": [f"reason for {strategy}"],
            "entry_price": 1000.0, "stop_loss": 940.0,
            "target_1": 1090.0, "target_2": 1150.0,
            "risk_reward_ratio": 1.5, "indicators": {}, "timestamp": "2026-01-01 10:00:00",
        }

    # Single signal — strength unchanged
    combined = agent._combine_buy_signals([_s("RSI_OVERSOLD_BOUNCE", 0.70)])
    if combined["strength"] == 0.70:
        ok("Single signal — strength unchanged", f"strength={combined['strength']:.2f}")
    else:
        fail("Single signal — strength unchanged", f"got {combined['strength']}")

    # Two signals — strength boosted
    combined2 = agent._combine_buy_signals([
        _s("RSI_OVERSOLD_BOUNCE", 0.70),
        _s("VOLUME_BREAKOUT",     0.75),
    ])
    # Primary=0.75 + 0.30 boost = 1.05 → capped at 1.0
    if combined2["strength"] == 1.0:
        ok("Two signals — strength boosted & capped at 1.0", f"strength={combined2['strength']:.2f}")
    else:
        fail("Two signals — strength boost", f"expected 1.0, got {combined2['strength']}")

    # Strategy name includes both strategies
    if "RSI_OVERSOLD_BOUNCE" in combined2["strategy_name"] and "VOLUME_BREAKOUT" in combined2["strategy_name"]:
        ok("Combined strategy_name contains both names", combined2["strategy_name"])
    else:
        fail("Combined strategy_name", combined2["strategy_name"])

    # Primary provides entry/stop data (highest strength takes precedence)
    combined3 = agent._combine_buy_signals([
        {**_s("EMA_PULLBACK",    0.65), "entry_price": 1000.0},
        {**_s("VOLUME_BREAKOUT", 0.75), "entry_price": 1010.0},
    ])
    if combined3["entry_price"] == 1010.0:
        ok("Primary signal (highest strength) provides entry price", f"entry=₹{combined3['entry_price']:.2f}")
    else:
        fail("Primary signal entry price", f"expected 1010.0, got {combined3['entry_price']}")


# ---------------------------------------------------------------------------
# Section C — Full scan against real data
# ---------------------------------------------------------------------------
def check_full_scan(agent, broker, symbols: list[str]) -> None:
    section(f"C · Full Scan — {len(symbols)} stocks with real market data")
    info(f"  Symbols: {', '.join(symbols)}")
    info(f"  Fetching data and running all 4 BUY strategies + EXIT strategy …")
    info("")

    # Patch the broker into the agent
    agent.broker = broker

    # Current holdings: pretend we hold the last symbol to test exit logic
    fake_holdings = [symbols[-1]] if symbols else []
    if fake_holdings:
        info(f"  Simulated holding: {fake_holdings}  (will check exit signals for it)")

    t0 = time.perf_counter()
    watchlist = [{"symbol": s} for s in symbols]
    try:
        signals = agent.scan_watchlist(watchlist, current_holdings=fake_holdings)
        elapsed = time.perf_counter() - t0

        ok(
            f"scan_watchlist({len(symbols)} stocks)",
            f"{len(signals)} signal(s) generated  [{elapsed:.2f}s  ≈ {elapsed/len(symbols):.2f}s/stock]",
        )
        info("")

        if signals:
            info(f"  {'─'*60}")
            info(f"  {'Symbol':<12} {'Signal':<5} {'Strategy':<26} {'Strength':>8} {'RR':>5}")
            info(f"  {'─'*60}")
            for sig in signals:
                arrow = "▲ BUY" if sig["signal"] == "BUY" else "▼ SELL"
                strategy_short = sig["strategy_name"][:26]
                info(
                    f"  {sig['symbol']:<12} {arrow:<5} {strategy_short:<26}"
                    f" {sig['strength']:>8.2f} {sig['risk_reward_ratio']:>5.2f}"
                )
            info(f"  {'─'*60}")
            info("")

            # Detail view for first signal
            info("  Detail — first signal:")
            _show_signal(signals[0])

            # Verify constraints
            all_ok = True
            for sig in signals:
                if sig["signal"] == "BUY":
                    if sig["stop_loss"] >= sig["entry_price"]:
                        fail(f"{sig['symbol']} BUY stop_loss >= entry_price", str(sig))
                        all_ok = False
                    if sig["target_1"] <= sig["entry_price"]:
                        fail(f"{sig['symbol']} BUY target_1 <= entry_price", str(sig))
                        all_ok = False
                    if sig["risk_reward_ratio"] <= 1.0:
                        fail(f"{sig['symbol']} BUY risk_reward_ratio <= 1.0", f"rr={sig['risk_reward_ratio']}")
                        all_ok = False
                    if sig["symbol"] in fake_holdings:
                        fail(f"{sig['symbol']} BUY generated for a held stock — should be suppressed")
                        all_ok = False
            if all_ok:
                ok("All signal constraints verified (stop<entry, target>entry, rr>1 for BUY)")

        else:
            info("  No actionable signals today — all stocks scored HOLD.")
            info("  This is normal market data, not a bug.")
            ok(f"scan_watchlist completed successfully with 0 signals")

    except Exception as exc:
        fail(f"scan_watchlist({len(symbols)} stocks)", str(exc))
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
_DEFAULT_SYMBOLS = ["TCS", "INFY", "HDFCBANK", "ICICIBANK", "WIPRO", "RELIANCE", "SBIN"]


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantAgent smoke test")
    parser.add_argument("--live",    action="store_true", help="Force Angel One mode")
    parser.add_argument("--demo",    action="store_true", help="Force yfinance mode")
    parser.add_argument("--symbols", nargs="+", metavar="SYM", default=_DEFAULT_SYMBOLS,
                        help="Symbols to scan (default: 7 Nifty50 stocks)")
    args = parser.parse_args()
    symbols = [s.upper() for s in args.symbols]

    # Mode selection
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
    print("  QuantAgent — End-to-End Smoke Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   mode={mode.upper()}")
    print(SEP2)

    # ── 1. Imports ──────────────────────────────────────────────────────
    section("1 · Imports")
    try:
        from src.agents.quant_agent import QuantAgent
        from src.tools.technical_indicators import TechnicalIndicators
        from src.database.db_manager import DatabaseManager
        ok("QuantAgent imported")
        ok("TechnicalIndicators imported")
        ok("DatabaseManager imported")
    except ImportError as exc:
        fail("Import failed", str(exc))
        sys.exit(1)

    # ── 2. Setup ─────────────────────────────────────────────────────────
    section("2 · Setup")

    # DB — in-memory mock (no file writes during smoke test)
    db_mock = MagicMock()
    db_mock.record_signal.return_value = 1

    # TechnicalIndicators — real instance
    ti = TechnicalIndicators()
    ok("TechnicalIndicators ready", f"pandas_ta={'yes' if ti._has_pandas_ta else 'no (fallback)'}")

    # Broker for Section C
    broker = None

    if mode == "live":
        try:
            from src.broker.angel_one import AngelOneClient
            cfg_dict = {
                "broker": {"angel_one": {
                    "client_id":  os.getenv("ANGEL_ONE_CLIENT_ID"),
                    "api_key":    os.getenv("ANGEL_ONE_API_KEY"),
                    "totp_secret": os.getenv("ANGEL_ONE_TOTP_SECRET"),
                    "password":   os.getenv("ANGEL_ONE_PASSWORD", ""),
                    "default_exchange": "NSE",
                }}
            }
            if not cfg_dict["broker"]["angel_one"]["password"]:
                import getpass
                cfg_dict["broker"]["angel_one"]["password"] = getpass.getpass("Enter Angel One MPIN: ")
            broker = AngelOneClient(cfg_dict)
            broker.login()
            ok("Angel One broker connected")
        except Exception as exc:
            fail("Angel One broker", str(exc))
            if _yfinance_available():
                info("  Falling back to yfinance demo mode …")
                broker = _YFinanceBroker()
                mode = "demo"
            else:
                info("  Install yfinance for demo mode: pip install yfinance")
    elif mode == "demo":
        if not _yfinance_available():
            fail("yfinance not installed", "pip install yfinance")
            sys.exit(1)
        broker = _YFinanceBroker()
        ok("yfinance demo broker ready")
    else:
        fail("No data source available")
        info("  Install yfinance:  pip install yfinance")
        sys.exit(1)

    # Build the agent — no broker yet (will be set per-section)
    agent = QuantAgent(
        config={},
        broker_client=MagicMock(),  # will be replaced before Section C
        technical_indicators=ti,
        db_manager=db_mock,
    )
    ok("QuantAgent instantiated")

    # ── Section A: Risk Management ──────────────────────────────────────
    check_risk_math(agent)

    # ── Section B: Strategy Detectors ───────────────────────────────────
    section("B · Strategy Detectors — Synthetic DataFrames (no broker needed)")

    check_strategy_rsi_oversold(agent)
    check_strategy_ema_pullback(agent)
    check_strategy_volume_breakout(agent)
    check_strategy_trend_following(agent)
    check_strategy_exit(agent)
    check_signal_combining(agent)

    # ── Section C: Full scan ──────────────────────────────────────────────
    if broker:
        check_full_scan(agent, broker, symbols)
    else:
        skip("Full scan", "no broker available")

    # ── Cleanup ───────────────────────────────────────────────────────────
    if mode == "live" and hasattr(broker, "logout"):
        try:
            broker.logout()
        except Exception:
            pass

    summary()


if __name__ == "__main__":
    main()
