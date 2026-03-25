"""Real-life scenario runner for RiskManager.

Run directly:
    python tests/test_risk_scenarios.py

Or via pytest (it will also collect the scenario functions as tests):
    pytest tests/test_risk_scenarios.py -v -s

Each scenario represents a situation a swing trader running a ₹50,000 account
would actually face.  The output is designed to be read and understood by a
human — every check result is printed with a clear pass/fail indicator.
"""
from __future__ import annotations

import sys
import os

# Ensure UTF-8 output on Windows consoles (handles the ₹ rupee symbol)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-16"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from datetime import datetime, date, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.risk_agent import RiskManager

IST = ZoneInfo("Asia/Kolkata")

# --------------------------------------------------------------------------- #
# ANSI colours (gracefully disabled when not a TTY)                            #
# --------------------------------------------------------------------------- #

_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

GREEN  = lambda t: _c("32", t)
RED    = lambda t: _c("31", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)

PASS = GREEN("✓ PASS")
FAIL = RED("✗ FAIL")
SEP  = DIM("─" * 72)


# --------------------------------------------------------------------------- #
# Fake broker & DB builders (realistic data, no network calls)                 #
# --------------------------------------------------------------------------- #

def _broker(
    total_value:    float = 50_000.0,
    available_cash: float = 50_000.0,
    holdings: list[dict] | None = None,
) -> MagicMock:
    b = MagicMock()
    b.get_portfolio_value.return_value = {
        "total_value":    total_value,
        "available_cash": available_cash,
        "invested":       total_value - available_cash,
        "total_pnl":      0.0,
    }
    b.get_holdings.return_value = holdings or []
    # Default: broker not connected for market-safety checks
    b.get_ltp.side_effect = ConnectionError("offline in scenario tests")
    b.get_historical_data.side_effect = ConnectionError("offline")
    return b


def _closed_trade(pnl: float, exit_date_str: str) -> MagicMock:
    t = MagicMock()
    t.exit_date = f"{exit_date_str} 14:30:00"
    t.pnl = pnl
    return t


def _db(
    trade_history: list | None = None,
    open_trades:   list | None = None,
    watchlist:     list | None = None,
) -> MagicMock:
    d = MagicMock()
    d.get_trade_history.return_value = trade_history or []
    d.get_open_trades.return_value   = open_trades   or []
    d.get_latest_watchlist.return_value = watchlist  or []
    return d


def _config(
    capital: float             = 50_000,
    max_position_pct: float    = 25,   # stored as integer percentage
    max_daily_loss_pct: float  = 2,
    max_weekly_loss_pct: float = 5,
    max_open_positions: int    = 5,
) -> MagicMock:
    cfg = MagicMock()
    cfg.get.side_effect = lambda *keys, default=None: {
        ("trading", "capital"):             capital,
        ("trading", "max_position_pct"):    max_position_pct,
        ("trading", "max_daily_loss_pct"):  max_daily_loss_pct,
        ("trading", "max_weekly_loss_pct"): max_weekly_loss_pct,
        ("trading", "max_open_positions"):  max_open_positions,
    }.get(keys, default)
    return cfg


def _make_rm(**kw) -> RiskManager:
    return RiskManager(_config(**kw), _broker(), _db())


# --------------------------------------------------------------------------- #
# Pretty-printing helpers                                                       #
# --------------------------------------------------------------------------- #

def _print_result(result: dict) -> None:
    status = BOLD(GREEN("APPROVED ✓")) if result["approved"] else BOLD(RED("REJECTED ✗"))
    trade = result["trade"]
    print(f"  Trade  : {trade.get('trade_type')} {trade.get('symbol')} "
          f"qty={trade.get('quantity')} @ ₹{trade.get('entry_price'):,.0f}")
    print(f"  Status : {status}")
    print(f"  Score  : {result['risk_score']:.2f}")
    print()
    for c in result["checks"]:
        icon = PASS if c["passed"] else FAIL
        print(f"  {icon}  [{c['rule']:<30}]  {DIM(c['detail'])}")
    if result["rejection_reason"]:
        print()
        print(f"  {RED('Rejection')} : {result['rejection_reason']}")
    if result["adjusted_quantity"] is not None:
        print(f"  {YELLOW('Adjusted qty')} : {result['adjusted_quantity']} shares")


def _header(title: str) -> None:
    print()
    print(SEP)
    print(BOLD(CYAN(f"  SCENARIO: {title}")))
    print(SEP)


def _subheader(text: str) -> None:
    print()
    print(YELLOW(f"  » {text}"))
    print()


# --------------------------------------------------------------------------- #
# ─────────────────────────── SCENARIOS ─────────────────────────────────────  #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Scenario 1 — Clean slate, textbook BUY setup                                 #
# --------------------------------------------------------------------------- #

def scenario_01_textbook_buy():
    """
    Monday morning. ₹50,000 idle. Quant agent fires a BUY signal on TCS.
    Stop loss 2.5% below entry, target gives R:R = 1.5. No existing holdings.
    Expected: APPROVED.
    """
    _header("01 — Textbook BUY on TCS (should be APPROVED)")

    rm = RiskManager(
        _config(),
        _broker(total_value=50_000, available_cash=50_000),
        _db(),
    )

    result = rm.check_trade({
        "symbol":      "TCS",
        "trade_type":  "BUY",
        "quantity":    3,            # 3 × ₹3,800 = ₹11,400 → 22.8% of portfolio
        "entry_price": 3_800.0,
        "stop_loss":   3_705.0,      # 2.5% below — within 5% hard limit
        "target_1":    3_942.5,      # risk ₹95, reward ₹142.5 → R:R 1.5
        "sector":      "IT",
    })
    _print_result(result)
    assert result["approved"], "Scenario 01 FAILED — expected APPROVED"
    return result


# --------------------------------------------------------------------------- #
# Scenario 2 — No stop loss (most common rookie mistake)                       #
# --------------------------------------------------------------------------- #

def scenario_02_no_stop_loss():
    """
    Research agent sends a BUY signal without attaching a stop loss.
    Expected: REJECTED (STOP_LOSS_MANDATORY fails).
    """
    _header("02 — BUY with no stop loss (should be REJECTED)")

    rm = RiskManager(
        _config(),
        _broker(total_value=50_000, available_cash=50_000),
        _db(),
    )

    result = rm.check_trade({
        "symbol":      "INFY",
        "trade_type":  "BUY",
        "quantity":    5,
        "entry_price": 1_500.0,
        "stop_loss":   None,         # ← missing
        "target_1":    1_650.0,
        "sector":      "IT",
    })
    _print_result(result)
    assert not result["approved"], "Scenario 02 FAILED — expected REJECTED"
    failing = [c["rule"] for c in result["checks"] if not c["passed"]]
    assert "STOP_LOSS_MANDATORY" in failing
    return result


# --------------------------------------------------------------------------- #
# Scenario 3 — Absurd stop loss (50% below entry)                              #
# --------------------------------------------------------------------------- #

def scenario_03_absurd_stop_loss():
    """
    Operator mistakenly sets stop loss at ₹1,000 on a ₹3,800 stock (74% away).
    Expected: REJECTED (STOP_LOSS_MANDATORY fails — exceeds 5% max distance).
    """
    _header("03 — Absurd stop loss 74% below entry (should be REJECTED)")

    rm = RiskManager(
        _config(),
        _broker(total_value=50_000, available_cash=50_000),
        _db(),
    )

    result = rm.check_trade({
        "symbol":      "RELIANCE",
        "trade_type":  "BUY",
        "quantity":    2,
        "entry_price": 2_800.0,
        "stop_loss":   700.0,        # ← 75% away — nonsensical
        "target_1":    3_200.0,
        "sector":      "OIL_GAS",
    })
    _print_result(result)
    assert not result["approved"]
    failing = [c["rule"] for c in result["checks"] if not c["passed"]]
    assert "STOP_LOSS_MANDATORY" in failing
    return result


# --------------------------------------------------------------------------- #
# Scenario 4 — Poor risk-reward (chasing a stock)                              #
# --------------------------------------------------------------------------- #

def scenario_04_poor_risk_reward():
    """
    Quant signal has stop loss 3% below entry but target only 2% above.
    R:R = 0.67 — well below the 1.5 minimum.
    Expected: REJECTED (RISK_REWARD_MINIMUM fails).
    """
    _header("04 — Poor R:R = 0.67 (should be REJECTED)")

    rm = RiskManager(
        _config(),
        _broker(total_value=50_000, available_cash=50_000),
        _db(),
    )

    entry  = 2_000.0
    sl     = entry * 0.97    # 3% stop → risk = ₹60
    target = entry * 1.02    # 2% target → reward = ₹40  → R:R ≈ 0.67

    result = rm.check_trade({
        "symbol":      "WIPRO",
        "trade_type":  "BUY",
        "quantity":    4,
        "entry_price": entry,
        "stop_loss":   sl,
        "target_1":    target,
        "sector":      "IT",
    })
    _print_result(result)
    assert not result["approved"]
    failing = [c["rule"] for c in result["checks"] if not c["passed"]]
    assert "RISK_REWARD_MINIMUM" in failing
    return result


# --------------------------------------------------------------------------- #
# Scenario 5 — Position too large, adjusted qty suggested                      #
# --------------------------------------------------------------------------- #

def scenario_05_oversized_position():
    """
    Signal suggests buying 20 shares of TCS @ ₹3,800 = ₹76,000.
    That's 152% of the ₹50,000 portfolio — way above the 25% cap.
    Expected: REJECTED, but adjusted_quantity shows the safe qty (3 shares).
    """
    _header("05 — Oversized position (REJECTED with adjusted_quantity)")

    rm = RiskManager(
        _config(max_position_pct=25),
        _broker(total_value=50_000, available_cash=50_000),
        _db(),
    )

    result = rm.check_trade({
        "symbol":      "TCS",
        "trade_type":  "BUY",
        "quantity":    20,
        "entry_price": 3_800.0,
        "stop_loss":   3_705.0,
        "target_1":    3_942.5,
        "sector":      "IT",
    })
    _print_result(result)
    assert not result["approved"]
    assert result["adjusted_quantity"] is not None
    # The suggested qty should fit within 25% of ₹50,000 = ₹12,500
    adj = result["adjusted_quantity"]
    assert adj * 3_800 <= 50_000 * 0.25, f"Adjusted qty {adj} still too large"
    _subheader(f"Suggested safe quantity: {adj} shares "
               f"(≈ ₹{adj * 3_800:,.0f} = {adj * 3_800 / 50_000 * 100:.1f}% of portfolio)")
    return result


# --------------------------------------------------------------------------- #
# Scenario 6 — Insufficient cash                                               #
# --------------------------------------------------------------------------- #

def scenario_06_insufficient_cash():
    """
    Bot only has ₹8,000 cash but the trade needs ₹19,000.
    Expected: REJECTED (CAPITAL_AVAILABLE fails).
    """
    _header("06 — Insufficient cash (should be REJECTED)")

    rm = RiskManager(
        _config(),
        _broker(total_value=50_000, available_cash=8_000),
        _db(),
    )

    result = rm.check_trade({
        "symbol":      "TCS",
        "trade_type":  "BUY",
        "quantity":    5,
        "entry_price": 3_800.0,
        "stop_loss":   3_705.0,
        "target_1":    3_942.5,
        "sector":      "IT",
    })
    _print_result(result)
    assert not result["approved"]
    failing = [c["rule"] for c in result["checks"] if not c["passed"]]
    assert "CAPITAL_AVAILABLE" in failing
    return result


# --------------------------------------------------------------------------- #
# Scenario 7 — Max open positions reached                                      #
# --------------------------------------------------------------------------- #

def scenario_07_max_positions_reached():
    """
    Already holding 5 stocks (the maximum). A new signal arrives.
    Expected: REJECTED (MAX_OPEN_POSITIONS fails).
    """
    _header("07 — Max open positions reached (should be REJECTED)")

    existing = [
        {"symbol": "TCS",      "sector": "IT",      "quantity": 3, "ltp": 3_800, "pnl": 150},
        {"symbol": "INFY",     "sector": "IT",      "quantity": 5, "ltp": 1_500, "pnl": -80},
        {"symbol": "HDFC",     "sector": "BANKING", "quantity": 4, "ltp": 1_700, "pnl": 200},
        {"symbol": "RELIANCE", "sector": "OIL_GAS", "quantity": 2, "ltp": 2_800, "pnl": 60},
        {"symbol": "ASIANPNT", "sector": "PAINTS",  "quantity": 6, "ltp": 3_200, "pnl": -40},
    ]
    invested = sum(h["quantity"] * h["ltp"] for h in existing)  # ≈ 47,200
    rm = RiskManager(
        _config(max_open_positions=5),
        _broker(total_value=50_000, available_cash=50_000 - invested, holdings=existing),
        _db(),
    )

    result = rm.check_trade({
        "symbol":      "BAJFINANCE",
        "trade_type":  "BUY",
        "quantity":    1,
        "entry_price": 6_800.0,
        "stop_loss":   6_630.0,
        "target_1":    7_055.0,
        "sector":      "NBFC",
    })
    _print_result(result)
    assert not result["approved"]
    failing = [c["rule"] for c in result["checks"] if not c["passed"]]
    assert "MAX_OPEN_POSITIONS" in failing
    return result


# --------------------------------------------------------------------------- #
# Scenario 8 — Daily loss limit hit                                            #
# --------------------------------------------------------------------------- #

def scenario_08_daily_loss_limit():
    """
    Two stop-losses were hit earlier today. Total realized loss = ₹1,200
    (2.4% of ₹50,000 — over the 2% daily limit of ₹1,000).
    Expected: REJECTED (DAILY_LOSS_LIMIT fails). SELL is still allowed.
    """
    _header("08 — Daily loss limit hit — BUY blocked, SELL allowed")

    today = datetime.now(IST).strftime("%Y-%m-%d")
    history = [
        _closed_trade(-700, today),
        _closed_trade(-500, today),
    ]  # total = -₹1,200

    rm = RiskManager(
        _config(max_daily_loss_pct=2),
        _broker(total_value=50_000, available_cash=48_800),
        _db(trade_history=history),
    )

    _subheader("Attempt 1 — BUY order")
    buy_result = rm.check_trade({
        "symbol":      "HCLTECH",
        "trade_type":  "BUY",
        "quantity":    2,
        "entry_price": 1_400.0,
        "stop_loss":   1_365.0,
        "target_1":    1_452.5,
        "sector":      "IT",
    })
    _print_result(buy_result)

    _subheader("Attempt 2 — SELL order on existing HCLTECH holding")
    sell_result = rm.check_trade({
        "symbol":      "HCLTECH",
        "trade_type":  "SELL",
        "quantity":    2,
        "entry_price": 1_390.0,
        "stop_loss":   None,
        "target_1":    None,
        "sector":      "IT",
    })
    _print_result(sell_result)

    assert not buy_result["approved"],   "BUY should be blocked when daily limit hit"
    assert sell_result["approved"],      "SELL should always be allowed"
    failing = [c["rule"] for c in buy_result["checks"] if not c["passed"]]
    assert "DAILY_LOSS_LIMIT" in failing
    return buy_result, sell_result


# --------------------------------------------------------------------------- #
# Scenario 9 — Weekly loss limit hit                                           #
# --------------------------------------------------------------------------- #

def scenario_09_weekly_loss_limit():
    """
    It's Wednesday. Three trades this week resulted in losses totalling ₹2,800
    (5.6% of ₹50,000 — over the 5% weekly limit of ₹2,500).
    Expected: REJECTED (WEEKLY_LOSS_LIMIT fails).
    """
    _header("09 — Weekly loss limit hit (should be REJECTED)")

    today  = date.today()
    monday = today - timedelta(days=today.weekday())
    history = [
        _closed_trade(-900,  monday.strftime("%Y-%m-%d")),
        _closed_trade(-1100, (monday + timedelta(1)).strftime("%Y-%m-%d")),
        _closed_trade(-800,  (monday + timedelta(2)).strftime("%Y-%m-%d")),
    ]  # total = -₹2,800 this week

    rm = RiskManager(
        _config(max_weekly_loss_pct=5),
        _broker(total_value=50_000, available_cash=47_200),
        _db(trade_history=history),
    )

    result = rm.check_trade({
        "symbol":      "MARUTI",
        "trade_type":  "BUY",
        "quantity":    1,
        "entry_price": 9_500.0,
        "stop_loss":   9_262.5,
        "target_1":    9_855.0,
        "sector":      "AUTO",
    })
    _print_result(result)
    assert not result["approved"]
    failing = [c["rule"] for c in result["checks"] if not c["passed"]]
    assert "WEEKLY_LOSS_LIMIT" in failing
    return result


# --------------------------------------------------------------------------- #
# Scenario 10 — Sector concentration (IT overweight)                           #
# --------------------------------------------------------------------------- #

def scenario_10_sector_concentration():
    """
    Already holding TCS (IT) and INFY (IT). A new signal arrives for WIPRO (IT).
    Adding a third IT stock violates the 2-stock-per-sector rule.
    Expected: REJECTED (SECTOR_CONCENTRATION fails).
    """
    _header("10 — Sector concentration (3rd IT stock rejected)")

    existing = [
        {"symbol": "TCS",  "sector": "IT", "quantity": 3, "ltp": 3_800, "pnl": 150},
        {"symbol": "INFY", "sector": "IT", "quantity": 5, "ltp": 1_500, "pnl": -80},
    ]
    invested = sum(h["quantity"] * h["ltp"] for h in existing)  # ≈ 18,900
    rm = RiskManager(
        _config(),
        _broker(total_value=50_000, available_cash=50_000 - invested, holdings=existing),
        _db(),
    )

    result = rm.check_trade({
        "symbol":      "WIPRO",
        "trade_type":  "BUY",
        "quantity":    6,
        "entry_price": 460.0,
        "stop_loss":   448.5,
        "target_1":    477.25,
        "sector":      "IT",
    })
    _print_result(result)
    assert not result["approved"]
    failing = [c["rule"] for c in result["checks"] if not c["passed"]]
    assert "SECTOR_CONCENTRATION" in failing
    return result


# --------------------------------------------------------------------------- #
# Scenario 11 — Averaging down (duplicate trade)                               #
# --------------------------------------------------------------------------- #

def scenario_11_averaging_down():
    """
    TCS was bought last week at ₹3,900. It has since fallen to ₹3,800.
    The orchestrator tries to buy more at the dip (averaging down).
    Phase 1 rule: NEVER average down.
    Expected: REJECTED (DUPLICATE_TRADE_CHECK fails).
    """
    _header("11 — Averaging down TCS (should be REJECTED)")

    existing = [
        {"symbol": "TCS", "sector": "IT", "quantity": 3, "ltp": 3_800,
         "avg_price": 3_900, "pnl": -300},
    ]
    rm = RiskManager(
        _config(),
        _broker(total_value=50_000, available_cash=38_600, holdings=existing),
        _db(),
    )

    result = rm.check_trade({
        "symbol":      "TCS",
        "trade_type":  "BUY",
        "quantity":    3,
        "entry_price": 3_800.0,
        "stop_loss":   3_705.0,
        "target_1":    3_942.5,
        "sector":      "IT",
    })
    _print_result(result)
    assert not result["approved"]
    failing = [c["rule"] for c in result["checks"] if not c["passed"]]
    assert "DUPLICATE_TRADE_CHECK" in failing
    return result


# --------------------------------------------------------------------------- #
# Scenario 12 — Tiny order (brokerage not worth it)                            #
# --------------------------------------------------------------------------- #

def scenario_12_tiny_order():
    """
    Signal fires on a ₹450 stock but only 2 shares are suggested.
    2 × ₹450 = ₹900 — below the ₹1,000 minimum order value.
    Expected: REJECTED (MINIMUM_TRADE_VALUE fails).
    """
    _header("12 — Order too small (₹900, min ₹1,000 — should be REJECTED)")

    rm = RiskManager(
        _config(),
        _broker(total_value=50_000, available_cash=50_000),
        _db(),
    )

    result = rm.check_trade({
        "symbol":      "WIPRO",
        "trade_type":  "BUY",
        "quantity":    2,
        "entry_price": 450.0,
        "stop_loss":   439.0,
        "target_1":    466.5,
        "sector":      "IT",
    })
    _print_result(result)
    assert not result["approved"]
    failing = [c["rule"] for c in result["checks"] if not c["passed"]]
    assert "MINIMUM_TRADE_VALUE" in failing
    return result


# --------------------------------------------------------------------------- #
# Scenario 13 — Multiple simultaneous failures                                 #
# --------------------------------------------------------------------------- #

def scenario_13_multiple_failures():
    """
    Worst-case trade proposal: no cash, no stop loss, no target, tiny quantity.
    All checks run — none short-circuit.  Caller gets the full picture.
    Expected: REJECTED with ALL relevant rules listed.
    """
    _header("13 — Multiple simultaneous failures (all rules still run)")

    rm = RiskManager(
        _config(),
        _broker(total_value=50_000, available_cash=200),  # almost no cash
        _db(),
    )

    result = rm.check_trade({
        "symbol":      "ZOMATO",
        "trade_type":  "BUY",
        "quantity":    1,
        "entry_price": 180.0,    # 1×180 = ₹180 < ₹1,000 minimum
        "stop_loss":   None,     # no stop loss
        "target_1":    None,     # no target
        "sector":      "FMCG",
    })
    _print_result(result)
    assert not result["approved"]
    failing = {c["rule"] for c in result["checks"] if not c["passed"]}
    # At minimum these three must all be in the failure set
    assert "STOP_LOSS_MANDATORY"  in failing
    assert "RISK_REWARD_MINIMUM"  in failing
    assert "MINIMUM_TRADE_VALUE"  in failing
    _subheader(f"Total rules failed: {len(failing)} — {', '.join(sorted(failing))}")
    return result


# --------------------------------------------------------------------------- #
# Scenario 14 — Position sizing calculator                                     #
# --------------------------------------------------------------------------- #

def scenario_14_position_sizing():
    """
    Use the position sizing calculator to determine how many shares to buy.
    Capital: ₹50,000  |  Entry: ₹3,800  |  Stop: ₹3,705 (risk/share = ₹95)
    Max risk per trade = 2% × 50,000 = ₹1,000 → 10 shares
    Max position = 25% × 50,000 = ₹12,500 → 3 shares
    Expected recommended: min(10, 3) = 3 shares.
    """
    _header("14 — Position sizing calculator")

    rm = RiskManager(
        _config(max_position_pct=25),
        _broker(),
        _db(),
    )

    result = rm.calculate_position_size(
        symbol="TCS",
        entry_price=3_800.0,
        stop_loss=3_705.0,
    )

    print(f"  Symbol         : TCS")
    print(f"  Entry          : ₹3,800")
    print(f"  Stop Loss      : ₹3,705  (risk per share: ₹95)")
    print()
    print(f"  {'Recommended qty':<22}: {result['recommended_quantity']} shares")
    print(f"  {'Investment':<22}: ₹{result['investment_amount']:,.0f}  "
          f"({result['investment_pct']:.1f}% of capital)")
    print(f"  {'Risk amount':<22}: ₹{result['risk_amount']:,.0f}  "
          f"({result['risk_pct']:.2f}% of capital)")

    assert result["recommended_quantity"] == 3, (
        f"Expected 3 shares, got {result['recommended_quantity']}"
    )
    assert result["risk_amount"] <= 1_000 + 1
    assert result["investment_amount"] <= 50_000 * 0.25 + 1
    return result


# --------------------------------------------------------------------------- #
# Scenario 15 — Portfolio risk summary                                         #
# --------------------------------------------------------------------------- #

def scenario_15_portfolio_summary():
    """
    Mid-day state: 3 open positions, one small realized loss today.
    Verify the summary correctly reports utilization, PnL, and trade eligibility.
    """
    _header("15 — Portfolio risk summary mid-day")

    today = datetime.now(IST).strftime("%Y-%m-%d")
    holdings = [
        {"symbol": "TCS",      "sector": "IT",      "quantity": 3, "ltp": 3_850, "pnl": 150},
        {"symbol": "HDFC",     "sector": "BANKING", "quantity": 4, "ltp": 1_720, "pnl": 80},
        {"symbol": "RELIANCE", "sector": "OIL_GAS", "quantity": 2, "ltp": 2_820, "pnl": 40},
    ]
    invested = sum(h["quantity"] * h["ltp"] for h in holdings)  # ≈ 25,390
    closed_today = [_closed_trade(-200, today)]  # small loss this morning

    rm = RiskManager(
        _config(),
        _broker(
            total_value    = 50_000,
            available_cash = 50_000 - invested,
            holdings       = holdings,
        ),
        _db(trade_history=closed_today),
    )

    summary = rm.get_portfolio_risk_summary()

    print(f"  Total capital        : ₹{summary['total_capital']:,.0f}")
    print(f"  Invested amount      : ₹{summary['invested_amount']:,.0f}")
    print(f"  Available cash       : ₹{summary['available_cash']:,.0f}")
    print(f"  Utilization          : {summary['utilization_pct']:.1f}%")
    print(f"  Open positions       : {summary['open_positions']}")
    print(f"  Unrealized PnL       : ₹{summary['unrealized_pnl']:,.0f}")
    print(f"  Today realized PnL   : ₹{summary['today_realized_pnl']:,.0f}")
    print(f"  Today total PnL      : ₹{summary['today_total_pnl']:,.0f}")
    print(f"  Weekly PnL           : ₹{summary['weekly_pnl']:,.0f}")
    print(f"  Daily limit hit?     : {summary['is_daily_limit_hit']}")
    print(f"  Weekly limit hit?    : {summary['is_weekly_limit_hit']}")
    print(f"  Can trade?           : {GREEN('YES') if summary['can_trade'] else RED('NO')}")
    print(f"  Largest position     : {summary['largest_position_pct']:.1f}%")
    print(f"  Sector exposure      : {summary['sector_exposure']}")

    assert summary["open_positions"]      == 3
    assert summary["can_trade"]           is True
    assert summary["is_daily_limit_hit"]  is False
    assert summary["unrealized_pnl"]      == 270.0   # 150+80+40
    assert summary["today_realized_pnl"]  == -200.0
    return summary


# --------------------------------------------------------------------------- #
# Scenario 16 — Full trading day simulation                                    #
# --------------------------------------------------------------------------- #

def scenario_16_full_day_simulation():
    """
    Simulate a realistic trading day:
      08:45 — Pre-market setup
      09:20 — Opening volatility window (trade blocked)
      10:30 — Good BUY signal (INFY) → approved
      11:15 — Bad BUY (no stop loss) → rejected
      14:00 — Good BUY (HDFC) → approved
      15:20 — Closing rush (trade blocked)
    """
    from unittest.mock import patch

    _header("16 — Full trading day simulation (time-controlled)")

    rm = RiskManager(
        _config(),
        _broker(total_value=50_000, available_cash=50_000),
        _db(),
    )

    def _trade_at(hour: int, minute: int, trade: dict, label: str) -> dict:
        fake_now = datetime(2026, 3, 18, hour, minute, 0, tzinfo=IST)
        with patch("src.agents.risk_agent._ist_now", return_value=fake_now):
            result = rm.check_trade(trade)
        status = GREEN("APPROVED") if result["approved"] else RED("REJECTED")
        print(f"  {hour:02d}:{minute:02d} IST  {label:<40} → {status}")
        return result

    good_trade_infy = {
        "symbol": "INFY", "trade_type": "BUY", "quantity": 4,
        "entry_price": 1_480.0, "stop_loss": 1_443.0, "target_1": 1_535.5,
        "sector": "IT",
    }
    bad_trade_tcs = {
        "symbol": "TCS", "trade_type": "BUY", "quantity": 2,
        "entry_price": 3_800.0, "stop_loss": None, "target_1": 3_900.0,
        "sector": "IT",
    }
    good_trade_hdfc = {
        "symbol": "HDFC", "trade_type": "BUY", "quantity": 3,
        "entry_price": 1_680.0, "stop_loss": 1_638.0, "target_1": 1_743.0,
        "sector": "BANKING",
    }

    print()
    r1 = _trade_at(9,  20, good_trade_infy, "BUY INFY (opening window)")
    r2 = _trade_at(10, 30, good_trade_infy, "BUY INFY (regular hours)")
    r3 = _trade_at(11, 15, bad_trade_tcs,   "BUY TCS no stop loss")
    r4 = _trade_at(14,  0, good_trade_hdfc, "BUY HDFC (regular hours)")
    r5 = _trade_at(15, 22, good_trade_hdfc, "BUY HDFC (closing window)")
    print()

    # Opening window → blocked
    assert not r1["approved"]
    assert not _rule_passed(r1, "NO_TRADING_FIRST_LAST_15MIN")

    # Regular hours, valid trade → approved
    assert r2["approved"]

    # Missing stop loss → rejected
    assert not r3["approved"]
    assert not _rule_passed(r3, "STOP_LOSS_MANDATORY")

    # Regular hours, valid trade → approved (HDFC is new sector)
    assert r4["approved"]

    # Closing window → blocked
    assert not r5["approved"]
    assert not _rule_passed(r5, "NO_TRADING_FIRST_LAST_15MIN")

    return [r1, r2, r3, r4, r5]


def _rule_passed(result: dict, rule: str) -> bool:
    for c in result["checks"]:
        if c["rule"] == rule:
            return c["passed"]
    return True


# --------------------------------------------------------------------------- #
# Entry point / runner                                                          #
# --------------------------------------------------------------------------- #

SCENARIOS = [
    ("01", "Textbook BUY — all rules pass",             scenario_01_textbook_buy),
    ("02", "No stop loss",                              scenario_02_no_stop_loss),
    ("03", "Absurd stop loss (74% away)",               scenario_03_absurd_stop_loss),
    ("04", "Poor R:R = 0.67",                           scenario_04_poor_risk_reward),
    ("05", "Oversized position → adjusted qty",         scenario_05_oversized_position),
    ("06", "Insufficient cash",                         scenario_06_insufficient_cash),
    ("07", "Max open positions reached",                scenario_07_max_positions_reached),
    ("08", "Daily loss limit hit",                      scenario_08_daily_loss_limit),
    ("09", "Weekly loss limit hit",                     scenario_09_weekly_loss_limit),
    ("10", "Sector concentration (3rd IT stock)",       scenario_10_sector_concentration),
    ("11", "Averaging down blocked",                    scenario_11_averaging_down),
    ("12", "Order too small (< ₹1,000)",                scenario_12_tiny_order),
    ("13", "Multiple simultaneous failures",            scenario_13_multiple_failures),
    ("14", "Position sizing calculator",                scenario_14_position_sizing),
    ("15", "Portfolio risk summary",                    scenario_15_portfolio_summary),
    ("16", "Full trading day simulation",               scenario_16_full_day_simulation),
]


def run_all() -> None:
    passed = 0
    failed = 0
    errors: list[tuple[str, str, str]] = []

    print()
    print(BOLD(CYAN("=" * 72)))
    print(BOLD(CYAN("  RiskManager — Real-Life Scenario Suite")))
    print(BOLD(CYAN("  Capital: ₹50,000  |  Exchange: NSE  |  Phase: 1")))
    print(BOLD(CYAN("=" * 72)))

    for num, title, fn in SCENARIOS:
        try:
            fn()
            print()
            print(BOLD(GREEN(f"  ✓ Scenario {num} passed")))
            passed += 1
        except AssertionError as exc:
            print()
            print(BOLD(RED(f"  ✗ Scenario {num} ASSERTION FAILED: {exc}")))
            failed += 1
            errors.append((num, title, str(exc)))
        except Exception as exc:
            print()
            print(BOLD(RED(f"  ✗ Scenario {num} ERROR: {type(exc).__name__}: {exc}")))
            failed += 1
            errors.append((num, title, f"{type(exc).__name__}: {exc}"))

    # Summary
    print()
    print(SEP)
    print(BOLD(f"  RESULTS: {GREEN(str(passed) + ' passed')}  "
               f"{(RED(str(failed) + ' failed')) if failed else DIM('0 failed')}  "
               f"/ {passed + failed} total"))
    if errors:
        print()
        for num, title, msg in errors:
            print(f"  {RED('FAIL')} Scenario {num} — {title}")
            print(f"       {DIM(msg)}")
    print(SEP)
    print()

    if failed:
        sys.exit(1)


# --------------------------------------------------------------------------- #
# pytest integration — each scenario is also a pytest test function            #
# --------------------------------------------------------------------------- #

def test_scenario_01(): scenario_01_textbook_buy()
def test_scenario_02(): scenario_02_no_stop_loss()
def test_scenario_03(): scenario_03_absurd_stop_loss()
def test_scenario_04(): scenario_04_poor_risk_reward()
def test_scenario_05(): scenario_05_oversized_position()
def test_scenario_06(): scenario_06_insufficient_cash()
def test_scenario_07(): scenario_07_max_positions_reached()
def test_scenario_08(): scenario_08_daily_loss_limit()
def test_scenario_09(): scenario_09_weekly_loss_limit()
def test_scenario_10(): scenario_10_sector_concentration()
def test_scenario_11(): scenario_11_averaging_down()
def test_scenario_12(): scenario_12_tiny_order()
def test_scenario_13(): scenario_13_multiple_failures()
def test_scenario_14(): scenario_14_position_sizing()
def test_scenario_15(): scenario_15_portfolio_summary()
def test_scenario_16(): scenario_16_full_day_simulation()


if __name__ == "__main__":
    run_all()
