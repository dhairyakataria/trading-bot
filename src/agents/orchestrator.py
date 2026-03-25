"""Orchestrator — central brain that coordinates all agents through the trading pipeline.

The orchestrator runs every 60 minutes during market hours and executes a
9-step cycle:

    0. Circuit-breaker check        — abort if system not OK
    1. Exit first (HIGH urgency)    — always prioritise open-position safety
    2. Gate on new-position flag    — skip buy steps if restricted
    3. Fetch watchlist              — from UniverseAgent
    4. Fetch holdings               — current open positions
    5. Scan for signals             — from QuantAgent
    6. Qualify BUY signals          — journal context → research → risk check
    7. Final trade selection        — sort, cap, diversify, max 2
    8. Execute approved trades      — via ExecutionAgent only
    9. Handle non-urgent exits      — NORMAL / LOW urgency signals

Scheduled routines
------------------
- Pre-market  (08:00 IST) — universe refresh, morning briefing, Telegram summary
- Post-market (15:45 IST) — portfolio snapshot, daily summary, journal update, report

Constraints
-----------
- Orchestrator NEVER calls broker directly — always via ExecutionAgent
- Max 2 trades per cycle; max 4 per day
- All exceptions are caught and logged — pipeline must not crash silently
- Every trade and decision is persisted to the database
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.utils.logger import get_logger, get_trade_logger

IST        = ZoneInfo("Asia/Kolkata")
_log       = get_logger("agents.orchestrator")
_tlog      = get_trade_logger()

# ── Trading limits ────────────────────────────────────────────────────────────
_MAX_TRADES_PER_CYCLE = 2
_MAX_TRADES_PER_DAY   = 4

# If two signals are within this strength delta, treat them as conflicting
_CONFLICT_STRENGTH_DELTA = 0.05

# Minimum signal strength to consider researching a stock
_MIN_SIGNAL_STRENGTH = 0.60

# Caution multiplier: reduce quantity when risk is moderate
_CAUTION_QTY_FACTOR = 0.70


def _cfg(config: Any, *keys: str, default: Any = None) -> Any:
    try:
        if hasattr(config, "get"):
            return config.get(*keys, default=default)
        obj: Any = config
        for k in keys:
            obj = obj[k]
        return obj
    except (KeyError, TypeError, AttributeError):
        return default


def _ist_now() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


def _ist_today() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


# ── Orchestrator ─────────────────────────────────────────────────────────────

class Orchestrator:
    """Top-level coordinator that runs the trading pipeline end-to-end.

    Initialises all agents internally using the provided *config*.  Pass the
    application :class:`~src.utils.config.Config` singleton.

    Args:
        config: Application :class:`~src.utils.config.Config` instance (or any
                dict-like object satisfying the same interface).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config        = config
        self._paper_trading = bool(_cfg(config, "trading", "paper_trading", default=False))
        self._capital       = float(_cfg(config, "trading", "capital", default=50_000))
        self._max_pos_pct   = _cfg(config, "trading", "max_position_pct", default=5) / 100

        _log.info(
            "Orchestrator starting up — capital=₹%.0f paper_trading=%s",
            self._capital, self._paper_trading,
        )

        self._init_infrastructure()
        self._init_agents()

        _log.info("Orchestrator ready.")

    # ================================================================== #
    # Initialisation                                                       #
    # ================================================================== #

    def _init_infrastructure(self) -> None:
        """Create broker, DB, LLM router, and shared tools."""
        from src.broker.angel_one import AngelOneClient
        from src.database.db_manager import DatabaseManager
        from src.llm.budget_manager import BudgetManager
        from src.llm.router import LLMRouter
        from src.tools.technical_indicators import TechnicalIndicators
        from src.tools.news_fetcher import NewsFetcher
        from src.tools.web_search import WebSearchTool

        # Database
        db_path   = _cfg(self.config, "database", "sqlite", "path", default="data/trading_bot.db")
        self.db   = DatabaseManager(db_path)

        # LLM
        self.budget_manager = BudgetManager(db=self.db)
        self.llm_router     = LLMRouter(self.config, self.budget_manager)

        # Broker
        self.broker = AngelOneClient(self.config)
        if not self._paper_trading:
            try:
                self.broker.login()
                _log.info("Broker login successful.")
            except Exception as exc:
                _log.critical(
                    "Broker login FAILED: %s — system will operate in degraded mode", exc
                )

        # Tools
        self.tech_indicators = TechnicalIndicators()
        self.news_fetcher    = NewsFetcher(self.config, self.budget_manager)
        self.web_search      = WebSearchTool(self.config, self.budget_manager)

    def _init_agents(self) -> None:
        """Create all trading agents in dependency order."""
        from src.agents.universe_agent import UniverseAgent
        from src.agents.quant_agent import QuantAgent
        from src.agents.research_agent import ResearchAgent
        from src.agents.risk_agent import RiskManager
        from src.agents.exit_agent import ExitAgent
        from src.agents.journal_agent import JournalAgent
        from src.agents.execution_agent import ExecutionAgent
        from src.circuit_breakers.safety import CircuitBreaker
        from src.notifications.telegram_bot import TelegramNotifier

        self.universe_agent  = UniverseAgent(broker=self.broker, db=self.db, config=self.config)
        self.quant_agent     = QuantAgent(self.config, self.broker, self.tech_indicators, self.db)
        self.research_agent  = ResearchAgent(
            self.config, self.llm_router, self.budget_manager,
            self.web_search, self.news_fetcher, self.db,
        )
        self.risk_manager    = RiskManager(self.config, self.broker, self.db)
        self.exit_agent      = ExitAgent(
            self.config, self.broker, self.tech_indicators,
            self.research_agent, self.db, self.llm_router,
        )
        self.journal_agent   = JournalAgent(self.config, self.db, self.llm_router)
        self.execution_agent = ExecutionAgent(self.config, self.broker, self.db, notifier=self.notifier)
        self.circuit_breaker = CircuitBreaker(self.config, self.broker, self.db)
        self.notifier        = TelegramNotifier(self.config)

    # ================================================================== #
    # Public lifecycle methods                                             #
    # ================================================================== #

    def run_cycle(self) -> dict:
        """Execute one full intra-day trading cycle (called every ~60 minutes).

        Returns::

            {
                "status":           "COMPLETE" | "PAUSED" | "EXITS_ONLY" | "ERROR",
                "signals_found":    int,
                "trades_approved":  int,
                "trades_executed":  int,
                "exits_executed":   int,
                "cycle_time_s":     float,
            }
        """
        t_start = time.monotonic()
        _log.info("═══ Trading cycle starting at %s ═══", _ist_now())

        result = {
            "status":          "COMPLETE",
            "signals_found":   0,
            "trades_approved": 0,
            "trades_executed": 0,
            "exits_executed":  0,
        }

        try:
            # ── Step 0: Circuit breaker ──────────────────────────────────
            safety = self._run_circuit_breaker()
            if not safety["system_ok"]:
                self._notify(
                    f"⛔ *System paused* — circuit breakers tripped:\n"
                    + "\n".join(f"• {b}" for b in safety["breakers_tripped"])
                )
                result["status"] = "PAUSED"
                return self._finish(result, t_start)

            if safety["must_close_all"]:
                self._notify(
                    "🔴🔴🔴 *MARKET CRASH DETECTED* — Nifty down ≥ 5%\n"
                    "Triggering immediate exit scan for ALL open positions."
                )
                _log.critical("must_close_all=True — running full exit scan immediately")
                exits_executed = self._process_exits(urgency_filter="ALL")
                result["exits_executed"] += exits_executed
                result["status"] = "EXITS_ONLY"
                return self._finish(result, t_start)

            # ── Step 1: Exit signals (HIGH urgency — PRIORITY) ───────────
            exits_executed = self._process_exits(urgency_filter="HIGH")
            result["exits_executed"] += exits_executed

            # ── Step 2: Gate on new-position flag ────────────────────────
            if not safety["can_open_new_positions"]:
                _log.info("Circuit breaker: new positions blocked — running exits only")
                result["status"] = "EXITS_ONLY"
                # Still handle non-urgent exits
                exits_executed = self._process_exits(urgency_filter="NON_HIGH")
                result["exits_executed"] += exits_executed
                return self._finish(result, t_start)

            # ── Step 3: Watchlist ────────────────────────────────────────
            watchlist = self._get_watchlist()
            if not watchlist:
                _log.warning("Empty watchlist — skipping signal scan")
                result["status"] = "COMPLETE"
                return self._finish(result, t_start)

            # ── Step 4: Holdings ─────────────────────────────────────────
            holdings = self._get_holdings()
            held_symbols = [h["symbol"] for h in holdings]

            # Ensure held stocks are in the watchlist (for exit checking)
            watchlist = self.universe_agent.ensure_held_stocks_in_watchlist(
                watchlist, held_symbols
            )

            # ── Step 5: Scan for signals ─────────────────────────────────
            signals = self._scan_signals(watchlist, held_symbols)
            buy_signals = [s for s in signals if s.get("signal") == "BUY"]
            result["signals_found"] = len(buy_signals)
            _log.info("Scan complete — %d BUY signal(s) found", len(buy_signals))

            # ── Step 6: Qualify BUY signals ──────────────────────────────
            approved_trades = self._qualify_buy_signals(
                buy_signals, watchlist, holdings
            )
            result["trades_approved"] = len(approved_trades)

            # ── Step 7: Final selection ───────────────────────────────────
            trades_today  = self._get_trades_executed_today()
            slots_left    = min(
                _MAX_TRADES_PER_CYCLE,
                max(0, _MAX_TRADES_PER_DAY - trades_today),
            )
            if slots_left <= 0:
                _log.info("Daily trade limit reached (%d/%d) — no new trades", trades_today, _MAX_TRADES_PER_DAY)
                result["status"] = "COMPLETE"
            else:
                final_trades = self._final_selection(
                    approved_trades, holdings, max_trades=slots_left
                )

                # ── Step 8: Execute ───────────────────────────────────────
                for trade in final_trades:
                    exec_result = self._execute_trade(trade)
                    if exec_result.get("success"):
                        result["trades_executed"] += 1

            # ── Step 9: Non-urgent exits ──────────────────────────────────
            exits_executed = self._process_exits(urgency_filter="NON_HIGH")
            result["exits_executed"] += exits_executed

        except Exception as exc:
            _log.exception("Orchestrator cycle error: %s", exc)
            result["status"] = "ERROR"
            self._notify(f"❌ *Orchestrator error:* {exc}")

        return self._finish(result, t_start)

    def run_pre_market(self) -> dict:
        """Run pre-market tasks at 08:00 IST.

        Tasks:
        1. Reset daily LLM budget
        2. Refresh stock universe watchlist
        3. Run morning research briefing
        4. Check global events / FII-DII data
        5. Send Telegram pre-market summary

        Returns summary dict.
        """
        _log.info("═══ Pre-market routine starting at %s ═══", _ist_now())
        result: dict = {"status": "COMPLETE", "watchlist_size": 0, "briefing_done": False}

        # 1. Reset today's trade counter
        try:
            self.db.set_system_state("trades_today_date",  _ist_today())
            self.db.set_system_state("trades_today_count", "0")
        except Exception as exc:
            _log.warning("Could not reset daily trade counter: %s", exc)

        # 2. Refresh watchlist
        try:
            self.universe_agent.refresh_index_constituents()
            watchlist = self.universe_agent.get_active_watchlist()
            result["watchlist_size"] = len(watchlist)
            _log.info("Watchlist refreshed — %d stocks", len(watchlist))
        except Exception as exc:
            _log.error("Watchlist refresh failed: %s", exc)

        # 3. Morning briefing (research)
        briefing: dict = {}
        try:
            briefing             = self.research_agent.morning_briefing()
            result["briefing_done"] = True
            _log.info("Morning briefing complete — outlook: %s", briefing.get("market_outlook", "N/A")[:80])
        except Exception as exc:
            _log.error("Morning briefing failed: %s", exc)

        # 4. Persist briefing to DB
        if briefing:
            try:
                self.db.set_system_state("morning_briefing", json.dumps(briefing))
            except Exception:
                pass

        # 5. Send Telegram summary
        if briefing:
            risky = briefing.get("risky_symbols", [])
            sectors = briefing.get("risky_sectors", [])
            msg = (
                f"🌅 *Pre-market brief* — {_ist_today()}\n"
                f"Outlook: {briefing.get('market_outlook', 'N/A')[:120]}\n"
                f"Watchlist: {result['watchlist_size']} stocks\n"
            )
            if risky:
                msg += f"⚠️ Risky symbols: {', '.join(risky[:5])}\n"
            if sectors:
                msg += f"⚠️ Risky sectors: {', '.join(sectors[:3])}"
            self._notify(msg)

        _log.info("Pre-market routine complete — %s", result)
        return result

    def run_post_market(self) -> dict:
        """Run post-market tasks at 15:45 IST.

        Tasks:
        1. Take portfolio snapshot
        2. Record trade outcomes in journal (for trades closed today)
        3. Build and save daily summary
        4. Send end-of-day Telegram report
        5. Weekly review (Fridays only)

        Returns summary dict.
        """
        _log.info("═══ Post-market routine starting at %s ═══", _ist_now())
        result: dict = {"status": "COMPLETE", "snapshot_saved": False, "report_sent": False}

        # 1. Portfolio snapshot
        try:
            self._save_portfolio_snapshot()
            result["snapshot_saved"] = True
        except Exception as exc:
            _log.error("Portfolio snapshot failed: %s", exc)

        # 2. Journal: record outcomes for trades closed today
        try:
            self._update_journal_for_closed_trades()
        except Exception as exc:
            _log.error("Journal update failed: %s", exc)

        # 3. Build daily summary
        try:
            summary = self._build_daily_summary()
        except Exception as exc:
            _log.error("Daily summary build failed: %s", exc)
            summary = {}

        # 4. Telegram EOD report
        try:
            self._send_eod_report(summary)
            result["report_sent"] = True
        except Exception as exc:
            _log.warning("EOD report send failed: %s", exc)

        # 5. Weekly review (Fridays)
        if datetime.now(IST).weekday() == 4:  # 4 = Friday
            try:
                weekly_review = self.journal_agent.generate_weekly_review()
                msg = f"📊 *Weekly Review*\n{json.dumps(weekly_review, indent=2)[:500]}"
                self._notify(msg)
            except Exception as exc:
                _log.warning("Weekly review failed: %s", exc)

        _log.info("Post-market routine complete — %s", result)
        return result

    def run_pipeline(self) -> None:
        """Determine the current trading phase and run the appropriate routine.

        Called by the scheduler when no specific routine is targeted.
        Routes to :meth:`run_pre_market`, :meth:`run_cycle`, or
        :meth:`run_post_market` based on current IST time.
        """
        now = datetime.now(IST)
        h, m = now.hour, now.minute

        if h == 8 and m < 30:
            self.run_pre_market()
        elif (h == 15 and m >= 45) or h > 15:
            self.run_post_market()
        else:
            self.run_cycle()

    def run_market_scan(self) -> None:
        """Intra-day scan alias for the scheduler (calls :meth:`run_cycle`)."""
        self.run_cycle()

    def shutdown(self) -> None:
        """Gracefully shut down all agents and close the broker connection."""
        _log.info("Orchestrator shutdown initiated.")
        try:
            if not self._paper_trading:
                self.broker.logout()
        except Exception as exc:
            _log.warning("Broker logout error during shutdown: %s", exc)
        _log.info("Orchestrator shutdown complete.")

    # ── Scheduler-facing aliases / lightweight routines ───────────────────────

    def run_morning_routine(self) -> dict:
        """Alias for run_pre_market — called by the scheduler at 08:00 IST."""
        return self.run_pre_market()

    def run_exit_check_only(self) -> int:
        """Lightweight 15-minute exit check — runs exits without a full cycle.

        Returns the number of exits executed.
        """
        _log.info("Exit-only check at %s", _ist_now())
        try:
            safety = self._run_circuit_breaker()
            if not safety["system_ok"]:
                _log.info("Circuit breakers tripped — skipping exit check")
                return 0
            return self._process_exits("ALL")
        except Exception as exc:
            _log.error("Exit-only check failed: %s", exc)
            return 0

    def run_eod_routine(self) -> dict:
        """Alias for run_post_market — called by the scheduler at 15:45 IST."""
        return self.run_post_market()

    def run_weekly_review(self) -> None:
        """Generate and send the weekly performance review via journal agent."""
        _log.info("Weekly review starting at %s", _ist_now())
        try:
            review = self.journal_agent.generate_weekly_review()
            self.notifier.send_weekly_report(review)
            _log.info("Weekly review sent")
        except Exception as exc:
            _log.error("Weekly review failed: %s", exc)
            self._notify(f"⚠️ *Weekly review failed:* {exc}")

    def reset_daily_state(self) -> None:
        """Reset daily budget and trade counters — called at midnight by the scheduler."""
        _log.info("Resetting daily state at %s", _ist_now())
        try:
            self.budget_manager.reset_daily_budgets()
        except Exception as exc:
            _log.warning("Budget reset failed: %s", exc)
        try:
            self.db.set_system_state("trades_today_date",  _ist_today())
            self.db.set_system_state("trades_today_count", "0")
            _log.info("Daily trade counter reset")
        except Exception as exc:
            _log.warning("Trade counter reset failed: %s", exc)

    def refresh_universe(self) -> None:
        """Refresh stock universe constituents — called on Sunday evenings."""
        _log.info("Universe refresh starting at %s", _ist_now())
        try:
            self.universe_agent.refresh_index_constituents()
            count = len(self.universe_agent.get_active_watchlist())
            _log.info("Universe refreshed — %d stocks in watchlist", count)
            self._notify(f"🔄 Universe refreshed — {count} stocks ready for next week")
        except Exception as exc:
            _log.error("Universe refresh failed: %s", exc)
            self._notify(f"⚠️ *Universe refresh failed:* {exc}")

    # ================================================================== #
    # Step implementations (private)                                       #
    # ================================================================== #

    def _run_circuit_breaker(self) -> dict:
        """Step 0: Run all circuit breakers and return safety status."""
        try:
            safety = self.circuit_breaker.check_all()
        except Exception as exc:
            _log.error("Circuit breaker check raised: %s — assuming system OK", exc)
            safety = {
                "system_ok": True,
                "can_open_new_positions": True,
                "must_close_all": False,
                "breakers_tripped": [],
                "warnings": [],
            }

        for warning in safety.get("warnings", []):
            _log.warning("CircuitBreaker warning: %s", warning)

        return safety

    def _process_exits(self, urgency_filter: str = "ALL") -> int:
        """Step 1 / Step 9: Process exit signals from ExitAgent.

        Args:
            urgency_filter: "HIGH" | "NON_HIGH" | "ALL"

        Returns:
            Number of exits executed.
        """
        executed = 0
        try:
            morning_briefing = self._load_morning_briefing()
            exit_signals     = self.exit_agent.check_exits(morning_briefing)
        except Exception as exc:
            _log.error("ExitAgent.check_exits failed: %s", exc)
            return 0

        for signal in exit_signals:
            urgency = signal.get("urgency", "NORMAL")

            if urgency_filter == "HIGH" and urgency != "HIGH":
                continue
            if urgency_filter == "NON_HIGH" and urgency == "HIGH":
                continue

            symbol = signal.get("symbol", "")
            try:
                exec_result = self.execution_agent.execute_exit(signal)
                if exec_result.get("success"):
                    executed += 1
                    pnl_str = f"₹{exec_result['pnl']:.0f}" if exec_result.get("pnl") else "N/A"
                    _log.info(
                        "EXIT executed | %s | type=%s | PnL=%s",
                        symbol, signal.get("exit_type"), pnl_str,
                    )
                    self._notify(
                        f"📤 *Exit* — {symbol}\n"
                        f"Type: {signal.get('exit_type')}\n"
                        f"PnL: {pnl_str}\n"
                        f"Reason: {signal.get('reasoning', '')[:80]}"
                    )
                else:
                    _log.warning(
                        "EXIT failed | %s | %s", symbol, exec_result.get("error")
                    )
            except Exception as exc:
                _log.error("Exit execution error for %s: %s", symbol, exc)

        return executed

    def _get_watchlist(self) -> List[dict]:
        """Step 3: Fetch today's active watchlist."""
        try:
            return self.universe_agent.get_active_watchlist()
        except Exception as exc:
            _log.error("Failed to fetch watchlist: %s", exc)
            return []

    def _get_holdings(self) -> List[dict]:
        """Step 4: Fetch current holdings from broker or DB."""
        if self._paper_trading:
            try:
                open_trades = self.db.get_open_trades()
                return [
                    {
                        "symbol":    t.symbol,
                        "quantity":  t.quantity,
                        "avg_price": t.price,
                        "ltp":       t.price,
                        "pnl":       0.0,
                    }
                    for t in open_trades
                ]
            except Exception:
                return []
        try:
            return self.broker.get_holdings()
        except Exception as exc:
            _log.error("Failed to fetch holdings: %s", exc)
            return []

    def _scan_signals(self, watchlist: List[dict], held_symbols: List[str]) -> List[dict]:
        """Step 5: Scan watchlist for technical signals."""
        try:
            signals = self.quant_agent.scan_watchlist(watchlist, held_symbols)
            _log.info("QuantAgent scanned %d stocks → %d signals", len(watchlist), len(signals))
            return signals
        except Exception as exc:
            _log.error("QuantAgent scan failed: %s", exc)
            return []

    def _qualify_buy_signals(
        self,
        buy_signals: List[dict],
        watchlist: List[dict],
        holdings: List[dict],
    ) -> List[dict]:
        """Step 6: For each BUY signal, run journal → research → risk checks.

        Returns a list of approved trade dicts (may have adjusted quantities).
        """
        approved: List[dict] = []

        # Build a sector lookup from watchlist
        sector_map: Dict[str, str] = {
            item["symbol"]: item.get("sector", "UNKNOWN")
            for item in watchlist
        }

        # Filter by minimum strength
        strong_signals = [
            s for s in buy_signals if s.get("strength", 0) >= _MIN_SIGNAL_STRENGTH
        ]
        _log.info(
            "Qualifying %d/%d signals (strength ≥ %.2f)",
            len(strong_signals), len(buy_signals), _MIN_SIGNAL_STRENGTH,
        )

        for signal in strong_signals:
            symbol      = signal.get("symbol", "")
            entry_price = float(signal.get("entry_price", 0.0))
            stop_loss   = float(signal.get("stop_loss",   0.0))
            target_1    = float(signal.get("target_1",    0.0))
            target_2    = float(signal.get("target_2",    0.0))
            strategy    = signal.get("strategy_name", "UNKNOWN")
            sector      = sector_map.get(symbol, "UNKNOWN")

            if entry_price <= 0:
                continue

            try:
                # ── 6a: Journal context ───────────────────────────────────
                context = ""
                try:
                    context = self.journal_agent.get_context_for_trade(
                        symbol, strategy, sector
                    )
                    _log.debug("Journal context for %s: %s", symbol, context[:80])
                except Exception as exc:
                    _log.debug("Journal context failed for %s: %s", symbol, exc)

                # ── 6b: Research ──────────────────────────────────────────
                research: dict = {}
                try:
                    research = self.research_agent.research_stock(
                        symbol, context=context, session_type="stock_research"
                    )
                except Exception as exc:
                    _log.warning("Research failed for %s: %s — using neutral sentiment", symbol, exc)
                    research = {"recommendation": "HOLD", "confidence": 0.5}

                # ── 6c: Skip if AVOID ─────────────────────────────────────
                if research.get("recommendation") == "AVOID":
                    _log.info(
                        "SKIP %s — research recommendation: AVOID (%s)",
                        symbol, research.get("reasoning", "")[:80],
                    )
                    self._record_signal_skipped(signal, "RESEARCH_AVOID")
                    continue

                # ── 6d: Calculate initial quantity ────────────────────────
                quantity = self._calculate_quantity(entry_price)
                if quantity <= 0:
                    _log.warning("SKIP %s — quantity resolved to 0 at price %.2f", symbol, entry_price)
                    continue

                proposed_trade = {
                    "symbol":       symbol,
                    "trade_type":   "BUY",
                    "quantity":     quantity,
                    "entry_price":  entry_price,
                    "stop_loss":    stop_loss,
                    "target_1":     target_1,
                    "target_2":     target_2,
                    "sector":       sector,
                    "strategy":     strategy,
                    "strategy_name": strategy,
                    "signal":       signal,
                    "research":     research,
                }

                # ── 6e: Risk check ────────────────────────────────────────
                try:
                    risk_check = self.risk_manager.check_trade(proposed_trade)
                except Exception as exc:
                    _log.error("RiskManager failed for %s: %s — skipping", symbol, exc)
                    self._record_signal_skipped(signal, f"RISK_CHECK_ERROR: {exc}")
                    continue

                if not risk_check.get("approved"):
                    reason = risk_check.get("rejection_reason", "UNKNOWN")
                    _log.info("SKIP %s — risk check rejected: %s", symbol, reason)
                    self._record_signal_skipped(signal, f"RISK_REJECTED: {reason}")
                    continue

                # ── 6f: Adjust quantity for CAUTION ──────────────────────
                final_qty = risk_check.get("adjusted_quantity") or quantity
                if risk_check.get("risk_score", 0) > 0.6:
                    final_qty = max(1, int(final_qty * _CAUTION_QTY_FACTOR))
                    _log.info(
                        "CAUTION adjustment for %s — reducing qty %d → %d (risk_score=%.2f)",
                        symbol, quantity, final_qty, risk_check["risk_score"],
                    )

                approved.append({
                    **proposed_trade,
                    "quantity":       final_qty,
                    "risk_check":     risk_check,
                    "research":       research,
                    "signal_strength": signal.get("strength", 0.0),
                })

                _log.info(
                    "APPROVED %s | qty=%d | strength=%.2f | risk_score=%.2f",
                    symbol, final_qty, signal.get("strength", 0), risk_check.get("risk_score", 0),
                )

            except Exception as exc:
                _log.error("Unexpected error qualifying %s: %s", symbol, exc)
                continue

        return approved

    def _final_selection(
        self,
        approved_trades: List[dict],
        current_holdings: List[dict],
        max_trades: int = _MAX_TRADES_PER_CYCLE,
    ) -> List[dict]:
        """Step 7: Select the best trades to execute this cycle.

        Rules (in order):
        1. Sort by composite score (signal strength × research confidence)
        2. Skip if we can't afford the trade
        3. Ensure sector diversity (≤ 1 new trade per sector per cycle)
        4. Cap at *max_trades*

        Returns:
            List of up to *max_trades* trade dicts ready for execution.
        """
        if not approved_trades:
            return []

        # Compute composite score
        def score(t: dict) -> float:
            strength   = t.get("signal_strength", 0.0)
            confidence = t.get("research", {}).get("confidence", 0.5)
            risk_score = t.get("risk_check", {}).get("risk_score", 0.5)
            return strength * confidence * (1.0 - risk_score * 0.5)

        sorted_trades = sorted(approved_trades, key=score, reverse=True)

        # Available cash check
        available_cash = self._get_available_cash()

        # Build a set of sectors already represented in current holdings
        held_sectors: set = {
            str(h.get("sector", "UNKNOWN")).upper() for h in current_holdings
        }
        # Also track sectors we're adding this cycle
        added_sectors: set = set()

        selected: List[dict] = []
        for trade in sorted_trades:
            if len(selected) >= max_trades:
                break

            symbol      = trade["symbol"]
            qty         = trade["quantity"]
            entry_price = trade["entry_price"]
            sector      = str(trade.get("sector", "UNKNOWN")).upper()
            cost        = qty * entry_price

            # Capital check
            if cost > available_cash:
                _log.info(
                    "SKIP %s — insufficient cash: need ₹%.0f, have ₹%.0f",
                    symbol, cost, available_cash,
                )
                continue

            # Sector diversity: don't add a second stock from the same sector in one cycle
            if sector in added_sectors:
                _log.info(
                    "SKIP %s — sector %s already being bought this cycle",
                    symbol, sector,
                )
                continue

            selected.append(trade)
            added_sectors.add(sector)
            available_cash -= cost  # Reserve cash for this trade

        _log.info(
            "Final selection: %d/%d approved trades selected",
            len(selected), len(approved_trades),
        )
        return selected

    def _execute_trade(self, trade: dict) -> dict:
        """Step 8: Execute a single approved trade via ExecutionAgent."""
        symbol = trade.get("symbol", "")
        _log.info(
            "Executing trade: %s qty=%d entry=%.2f sl=%.2f",
            symbol, trade["quantity"], trade["entry_price"], trade["stop_loss"],
        )
        try:
            exec_result = self.execution_agent.execute_buy(trade)
        except Exception as exc:
            _log.error("ExecutionAgent.execute_buy raised for %s: %s", symbol, exc)
            return {"success": False, "error": str(exc)}

        if exec_result.get("success"):
            self._increment_trades_today()
            pnl_str = "pending"
            self._notify(
                f"📈 *New Trade* — {symbol}\n"
                f"Strategy: {trade.get('strategy', 'N/A')}\n"
                f"Entry: ₹{exec_result.get('filled_price', trade['entry_price']):.2f}\n"
                f"Qty: {trade['quantity']}\n"
                f"SL: ₹{trade['stop_loss']:.2f}\n"
                f"Target: ₹{trade.get('target_1', 0):.2f}"
            )
            _tlog.info(
                "CYCLE TRADE | %s | qty=%d | entry=%.2f | sl=%.2f | strategy=%s",
                symbol, trade["quantity"], trade["entry_price"],
                trade["stop_loss"], trade.get("strategy"),
            )
        else:
            _log.warning("Trade execution failed for %s: %s", symbol, exec_result.get("error"))

        return exec_result

    # ================================================================== #
    # Post-market helpers                                                  #
    # ================================================================== #

    def _save_portfolio_snapshot(self) -> None:
        """Capture and persist the current portfolio state."""
        from src.database.models import PortfolioSnapshot

        now = datetime.now(IST)

        if self._paper_trading:
            open_trades   = self.db.get_open_trades()
            invested      = sum(t.price * t.quantity for t in open_trades)
            total_value   = self._capital  # Approximate for paper trading
            available_cash = total_value - invested
            unrealized_pnl = 0.0
            realized_today = sum(
                t.pnl for t in self.db.get_trade_history(days=1)
                if t.exit_date and t.exit_date.startswith(_ist_today()) and t.pnl
            )
        else:
            try:
                pv             = self.broker.get_portfolio_value()
                total_value    = pv.get("total_value", 0.0)
                available_cash = pv.get("available_cash", 0.0)
                invested       = pv.get("invested", 0.0)
                unrealized_pnl = pv.get("total_pnl", 0.0)
                realized_today = 0.0  # Will be computed from DB
                trades_today   = self.db.get_trade_history(days=1)
                today_str      = _ist_today()
                realized_today = sum(
                    t.pnl for t in trades_today
                    if t.exit_date and t.exit_date.startswith(today_str) and t.pnl
                )
            except Exception as exc:
                _log.error("Could not fetch portfolio value: %s", exc)
                return

        open_positions = len(self.db.get_open_trades())

        snapshot = PortfolioSnapshot(
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            total_value=round(total_value, 2),
            invested_amount=round(invested, 2),
            available_cash=round(available_cash, 2),
            unrealized_pnl=round(unrealized_pnl, 2),
            realized_pnl_today=round(realized_today, 2),
            open_positions=open_positions,
        )
        self.db.save_portfolio_snapshot(snapshot)
        _log.info(
            "Portfolio snapshot saved — total=₹%.0f cash=₹%.0f positions=%d",
            total_value, available_cash, open_positions,
        )

    def _update_journal_for_closed_trades(self) -> None:
        """Call journal_agent.record_trade_outcome for trades closed today."""
        today_str   = _ist_today()
        trades_today = self.db.get_trade_history(days=1)

        for trade in trades_today:
            if not trade.exit_date or not trade.exit_date.startswith(today_str):
                continue
            if trade.pnl is None:
                continue
            try:
                trade_dict = trade.to_dict()
                trade_dict["outcome"]     = "WIN" if (trade.pnl or 0) > 0 else "LOSS"
                trade_dict["pnl_pct"]     = trade.pnl_percentage or 0.0
                trade_dict["holding_days"] = trade.holding_days or 0
                self.journal_agent.record_trade_outcome(trade_dict)
                _log.debug("Journal updated for trade_id=%s %s", trade.id, trade.symbol)
            except Exception as exc:
                _log.warning("Journal update failed for trade %s: %s", trade.id, exc)

    def _build_daily_summary(self) -> dict:
        """Build and persist today's DailySummary to DB."""
        from src.database.models import DailySummary

        today_str   = _ist_today()
        stats       = self.db.get_performance_stats(days=1)
        perf_30d    = self.db.get_performance_stats(days=30)

        summary = DailySummary(
            date=today_str,
            trades_executed=stats.get("total_trades", 0),
            trades_profitable=stats.get("winning_trades", 0),
            total_pnl=stats.get("total_pnl", 0.0),
            agent_summary=(
                f"30d stats: {perf_30d.get('total_trades', 0)} trades, "
                f"{perf_30d.get('win_rate_pct', 0):.1f}% win rate, "
                f"₹{perf_30d.get('total_pnl', 0):,.0f} PnL"
            ),
        )

        # Nifty change
        if not self._paper_trading:
            try:
                from datetime import timedelta
                now     = datetime.now(IST)
                from_dt = (now - timedelta(days=2)).strftime("%Y-%m-%d %H:%M")
                to_dt   = now.strftime("%Y-%m-%d %H:%M")
                hist    = self.broker.get_historical_data("NIFTY", "ONE_DAY", from_dt, to_dt)
                if hist is not None and not hist.empty and len(hist) >= 2:
                    pct = (float(hist.iloc[-1]["close"]) - float(hist.iloc[-2]["close"])) / float(hist.iloc[-2]["close"]) * 100
                    summary.nifty_change_pct = round(pct, 2)
            except Exception:
                pass

        try:
            self.db.save_daily_summary(summary)
        except Exception as exc:
            _log.warning("Could not save daily summary: %s", exc)

        return summary.to_dict()

    def _send_eod_report(self, summary: dict) -> None:
        """Send end-of-day Telegram report."""
        today     = _ist_today()
        perf      = self.db.get_performance_stats(days=30)
        trades    = summary.get("trades_executed", 0)
        pnl       = summary.get("total_pnl", 0.0)
        nifty_str = (
            f"{summary.get('nifty_change_pct', 0):+.2f}%"
            if summary.get("nifty_change_pct") is not None else "N/A"
        )
        msg = (
            f"📊 *EOD Report — {today}*\n"
            f"Trades today: {trades}\n"
            f"Today P&L: ₹{pnl:,.0f}\n"
            f"Nifty: {nifty_str}\n"
            f"30d win rate: {perf.get('win_rate_pct', 0):.1f}%\n"
            f"30d P&L: ₹{perf.get('total_pnl', 0):,.0f}"
        )
        self._notify(msg)
        try:
            self.notifier.send_daily_summary(summary)
        except NotImplementedError:
            pass  # Notifier stub not yet implemented
        except Exception as exc:
            _log.warning("send_daily_summary failed: %s", exc)

    # ================================================================== #
    # Utility helpers                                                      #
    # ================================================================== #

    def _get_available_cash(self) -> float:
        """Return available trading cash."""
        if self._paper_trading:
            try:
                open_trades = self.db.get_open_trades()
                invested    = sum(t.price * t.quantity for t in open_trades)
                return max(0.0, self._capital - invested)
            except Exception:
                return self._capital
        try:
            return self.broker.get_margin_available()
        except Exception as exc:
            _log.warning("Could not fetch margin: %s — using capital estimate", exc)
            return self._capital * 0.5  # Conservative estimate

    def _calculate_quantity(self, entry_price: float) -> int:
        """Calculate how many shares to buy based on position sizing."""
        if entry_price <= 0:
            return 0
        max_trade_value = self._capital * self._max_pos_pct
        return max(0, int(max_trade_value / entry_price))

    def _get_trades_executed_today(self) -> int:
        """Return count of trades executed today (from system state)."""
        try:
            date_key  = self.db.get_system_state("trades_today_date")
            today_str = _ist_today()
            if date_key != today_str:
                # New day — reset counter
                self.db.set_system_state("trades_today_date",  today_str)
                self.db.set_system_state("trades_today_count", "0")
                return 0
            count_str = self.db.get_system_state("trades_today_count")
            return int(count_str or "0")
        except Exception:
            return 0

    def _increment_trades_today(self) -> None:
        """Increment today's trade counter."""
        try:
            current = self._get_trades_executed_today()
            self.db.set_system_state("trades_today_count", str(current + 1))
        except Exception as exc:
            _log.warning("Could not increment trades counter: %s", exc)

    def _load_morning_briefing(self) -> Optional[dict]:
        """Load today's morning briefing from DB system state."""
        try:
            raw = self.db.get_system_state("morning_briefing")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return None

    def _record_signal_skipped(self, signal: dict, reason: str) -> None:
        """Record a skipped BUY signal in the DB for analysis."""
        from src.database.models import Signal as DBSignal

        try:
            db_signal = DBSignal(
                symbol=signal.get("symbol", ""),
                date=_ist_today(),
                signal_type=signal.get("signal", "BUY"),
                signal_source="QUANT",
                strength=signal.get("strength"),
                indicators=json.dumps(signal.get("indicators", {})),
                was_acted_on=0,
                reason_if_skipped=reason,
            )
            self.db.record_signal(db_signal)
        except Exception as exc:
            _log.debug("Could not record skipped signal: %s", exc)

    def _notify(self, message: str) -> None:
        """Send a Telegram notification, silently ignoring failures."""
        try:
            self.notifier.send_message(message)
        except NotImplementedError:
            _log.debug("Telegram notifier not implemented yet — message suppressed")
        except Exception as exc:
            _log.debug("Telegram notification failed: %s", exc)

    @staticmethod
    def _finish(result: dict, t_start: float) -> dict:
        """Attach cycle timing and log final summary."""
        elapsed          = round(time.monotonic() - t_start, 1)
        result["cycle_time_s"] = elapsed
        _log.info(
            "═══ Cycle complete in %.1fs | status=%s trades=%d exits=%d ═══",
            elapsed,
            result.get("status"),
            result.get("trades_executed", 0),
            result.get("exits_executed", 0),
        )
        return result
