"""Journal Agent — records and analyses every trade for continuous improvement."""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from typing import Any, List, Optional
from zoneinfo import ZoneInfo

from src.database.models import DailySummary
from src.llm.router import LLMRouter, TaskComplexity

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_CHROMA_AVAILABLE = True
try:
    import chromadb  # type: ignore
except ImportError:  # pragma: no cover
    _CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed; falling back to SQLite-only mode.")


class JournalAgent:
    """Maintains a structured trade journal and generates performance reports.

    Responsibilities:
    - Record every closed trade with metadata and LLM-generated lessons.
    - Store trade memories in ChromaDB for semantic retrieval.
    - Calculate per-strategy and per-sector performance from SQLite.
    - Provide natural-language context summaries for the orchestrator.
    - Produce weekly LLM-assisted trading reviews.
    """

    def __init__(
        self,
        config: Any,
        db_manager: Any,
        llm_router: LLMRouter,
    ) -> None:
        self.config = config
        self.db = db_manager
        self.llm = llm_router
        self._chroma_collection: Any = None
        self._init_chroma()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_chroma(self) -> None:
        """Initialise the ChromaDB persistent collection, with graceful fallback."""
        if not _CHROMA_AVAILABLE:
            return
        try:
            chroma_path: str = (
                self.config.get("database", "chroma_path") or "data/chroma_db"
            )
            client = chromadb.PersistentClient(path=chroma_path)
            self._chroma_collection = client.get_or_create_collection(
                name="trade_memories",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB 'trade_memories' ready at %s", chroma_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ChromaDB init failed — falling back to SQLite-only mode: %s", exc
            )
            self._chroma_collection = None

    # ─────────────────────────────────────────────────────────────────────────
    # Recording
    # ─────────────────────────────────────────────────────────────────────────

    def generate_trade_lessons(self, trade: dict) -> str:
        """Ask the LLM to extract 1-2 key lessons from a completed trade."""
        prompt = (
            "Analyze this trade and extract 1-2 key lessons for future trades.\n"
            f"Trade details: {json.dumps(trade, default=str)}"
        )
        try:
            return self.llm.call(
                prompt=prompt,
                system_prompt=(
                    "You are an expert trading coach. Be concise and specific. "
                    "Focus on what can be learnt and applied to future trades."
                ),
                complexity=TaskComplexity.SIMPLE,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM lesson generation failed: %s", exc)
            outcome = trade.get("outcome", "UNKNOWN")
            strategy = trade.get("strategy", "UNKNOWN")
            return (
                f"{strategy} trade ended with {outcome}. "
                "Review entry conditions before the next similar setup."
            )

    def record_trade_outcome(self, trade: dict) -> None:
        """Persist a closed trade's outcome to ChromaDB and log it to SQLite.

        Generates LLM lessons if the 'lessons' field is empty, then stores the
        enriched trade in ChromaDB for future semantic retrieval.
        """
        # Enrich with lessons without mutating the caller's dict.
        trade = dict(trade)
        if not trade.get("lessons"):
            trade["lessons"] = self.generate_trade_lessons(trade)

        self._add_to_chroma(trade)

        try:
            self.db.log_agent_activity(
                agent_name="journal_agent",
                session_type="trade_outcome",
                input_data=trade,
                output_data={"lessons": trade.get("lessons", "")},
                llm_calls_count=1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log trade outcome to SQLite: %s", exc)

    def _add_to_chroma(self, trade: dict) -> None:
        """Insert a trade memory document into the ChromaDB collection."""
        if self._chroma_collection is None:
            return

        # Build a unique, stable document ID.
        raw_id = trade.get("id") or datetime.now(IST).strftime("%Y%m%d%H%M%S%f")
        chroma_id = f"trade_{trade.get('symbol', 'x')}_{raw_id}"

        # Keep document text concise (≤500 chars) — it's used for retrieval only.
        doc = (
            f"{trade.get('symbol', '')} {trade.get('strategy', '')} "
            f"{trade.get('sector', '')} {trade.get('outcome', '')} "
            f"{trade.get('entry_reasoning', '')}"
        )[:500]

        # ChromaDB metadata values must be str / int / float / bool.
        metadata: dict[str, Any] = {
            k: (str(v) if v is not None else "") for k, v in trade.items()
        }

        try:
            self._chroma_collection.add(
                documents=[doc],
                metadatas=[metadata],
                ids=[chroma_id],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaDB add failed for %s: %s", chroma_id, exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    def get_similar_past_trades(
        self,
        symbol: str,
        strategy: str,
        sector: str,
        top_k: int = 5,
    ) -> List[dict]:
        """Return the top-k most semantically similar past trades from ChromaDB.

        Falls back to an empty list when ChromaDB is unavailable or empty.
        """
        if self._chroma_collection is None:
            return []
        try:
            count: int = self._chroma_collection.count()
            if count == 0:
                return []
            results = self._chroma_collection.query(
                query_texts=[f"{symbol} {strategy} {sector}"],
                n_results=min(top_k, count),
            )
            metadatas: list[dict] = results.get("metadatas", [[]])[0]
            return metadatas
        except Exception as exc:  # noqa: BLE001
            logger.warning("ChromaDB query failed: %s", exc)
            return []

    def get_strategy_performance(self, strategy: str, days: int = 90) -> dict:
        """Calculate performance metrics for a strategy using SQLite trade history."""
        trades = self.db.get_trade_history(days)
        closed = self._filter_closed_by_strategy(trades, strategy)
        return self._calc_performance(closed, groupby="strategy", groupval=strategy)

    def get_sector_performance(self, sector: str, days: int = 90) -> dict:
        """Calculate performance metrics for a market sector using SQLite trade history."""
        trades = self.db.get_trade_history(days)
        closed = self._filter_closed_by_sector(trades, sector)
        return self._calc_performance(closed, groupby="sector", groupval=sector)

    def get_overall_stats(self, days: int = 30) -> dict:
        """Aggregate portfolio performance stats for the given number of days."""
        stats = self.db.get_performance_stats(days)

        nifty_return: Optional[float] = None
        try:
            raw = self.db.get_system_state(f"nifty_return_{days}d")
            if raw is not None:
                nifty_return = float(raw)
        except Exception:  # noqa: BLE001
            pass

        return {
            "period_days": days,
            "total_trades": stats.get("total_trades", 0),
            "wins": stats.get("winning_trades", 0),
            "losses": stats.get("losing_trades", 0),
            "win_rate": stats.get("win_rate_pct", 0.0),
            "total_pnl_inr": round(stats.get("total_pnl", 0.0), 2),
            "avg_pnl_pct": round(stats.get("avg_pnl_pct", 0.0), 2),
            "best_trade_pnl": stats.get("best_trade_pnl", 0.0),
            "worst_trade_pnl": stats.get("worst_trade_pnl", 0.0),
            "avg_holding_days": round(stats.get("avg_holding_days", 0.0), 1),
            "nifty_return_pct": nifty_return,
            "sharpe_ratio": self._calc_sharpe(days),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Context for Orchestrator
    # ─────────────────────────────────────────────────────────────────────────

    def get_context_for_trade(self, symbol: str, strategy: str, sector: str) -> str:
        """Return a natural-language summary for the orchestrator's trade reasoning.

        Combines semantic search results with aggregated strategy stats to produce
        a concise recommendation about whether historical conditions favour the setup.
        """
        similar = self.get_similar_past_trades(symbol, strategy, sector, top_k=12)
        perf = self.get_strategy_performance(strategy)

        if perf["total_trades"] == 0 and not similar:
            return (
                f"No historical data available for {strategy} on {sector} stocks. "
                "This would be the first recorded trade with this setup."
            )

        parts: list[str] = []

        if similar:
            sim_wins = sum(1 for t in similar if str(t.get("outcome", "")).upper() == "WIN")
            sim_total = len(similar)
            win_pct = round(sim_wins / sim_total * 100, 1)
            parts.append(
                f"Based on {sim_total} similar past trades: "
                f"{sim_wins} wins, {sim_total - sim_wins} losses ({win_pct}% win rate)."
            )
            last = similar[0]
            if last.get("symbol") and last.get("pnl_pct"):
                parts.append(
                    f"Last similar trade ({last['symbol']}) gained {last['pnl_pct']}% "
                    f"in {last.get('holding_days', '?')} days."
                )

        if perf["total_trades"] > 0:
            parts.append(
                f"{strategy} on {sector} stocks has a {perf['win_rate']}% win rate "
                f"({perf['total_trades']} trades), "
                f"avg gain {perf['avg_win_pct']}%, avg loss {perf['avg_loss_pct']}%, "
                f"expectancy {perf['expectancy']}."
            )

        # Recommendation
        if perf["total_trades"] >= 3:
            if perf["expectancy"] > 0 and perf["win_rate"] >= 50:
                parts.append("Recommendation: Conditions are favorable for this strategy.")
            elif perf["win_rate"] < 40:
                parts.append("Recommendation: Win rate is below threshold — exercise caution.")
            else:
                parts.append(
                    "Recommendation: Mixed results — proceed with standard risk controls."
                )
        else:
            parts.append(
                "Recommendation: Limited historical data — use standard position sizing."
            )

        return " ".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # Weekly Review
    # ─────────────────────────────────────────────────────────────────────────

    def generate_weekly_review(self) -> dict:
        """Produce an LLM-assisted weekly trading review and persist it."""
        trades = self.db.get_trade_history(days=7)
        stats = self.db.get_performance_stats(days=7)
        closed = [t for t in trades if t.exit_date]

        trade_lines: list[str] = []
        for t in closed:
            sig: dict = {}
            if t.strategy_signal:
                try:
                    sig = json.loads(t.strategy_signal)
                except json.JSONDecodeError:
                    pass
            pnl_pct = t.pnl_percentage or 0.0
            sign = "+" if pnl_pct > 0 else ""
            strat = sig.get("strategy") or sig.get("signal_source") or "unknown"
            trade_lines.append(f"{t.symbol}: {sign}{round(pnl_pct, 2)}% ({strat})")

        trades_text = "\n".join(trade_lines) if trade_lines else "No closed trades this week."

        prompt = (
            "Weekly Trading Review — last 7 days.\n"
            f"Performance stats: {json.dumps(stats, default=str)}\n"
            f"Closed trades:\n{trades_text}\n\n"
            "Provide a concise analysis covering:\n"
            "1. What worked well\n"
            "2. What didn't work\n"
            "3. Strategy adjustments for next week\n"
            "4. Risk management observations"
        )

        llm_analysis = ""
        try:
            llm_analysis = self.llm.call(
                prompt=prompt,
                system_prompt=(
                    "You are a professional trading coach reviewing weekly performance. "
                    "Be specific and actionable."
                ),
                complexity=TaskComplexity.MODERATE,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM weekly review generation failed: %s", exc)
            llm_analysis = "LLM review unavailable. Manual review recommended."

        week_ending = datetime.now(IST).strftime("%Y-%m-%d")
        review = {
            "week_ending": week_ending,
            "total_trades": stats.get("total_trades", 0),
            "win_rate": stats.get("win_rate_pct", 0.0),
            "total_pnl": stats.get("total_pnl", 0.0),
            "llm_analysis": llm_analysis,
        }

        try:
            self.db.save_daily_summary(
                DailySummary(
                    date=week_ending,
                    trades_executed=review["total_trades"],
                    trades_profitable=stats.get("winning_trades", 0),
                    total_pnl=review["total_pnl"],
                    agent_summary=llm_analysis,
                    market_outlook=f"Weekly review: {llm_analysis[:200]}",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save weekly review to daily_summary: %s", exc)

        try:
            self.db.log_agent_activity(
                agent_name="journal_agent",
                session_type="weekly_review",
                input_data={"days": 7, "closed_trades": len(closed)},
                output_data=review,
                llm_calls_count=1,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log weekly review activity: %s", exc)

        return review

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _filter_closed_by_strategy(self, trades: list, strategy: str) -> list:
        strategy_upper = strategy.upper()
        result = []
        for t in trades:
            if t.exit_date is None or t.pnl_percentage is None:
                continue
            if t.strategy_signal:
                try:
                    sig = json.loads(t.strategy_signal)
                    strat_val = (
                        sig.get("strategy")
                        or sig.get("strategy_name")
                        or sig.get("signal_source")
                        or ""
                    )
                    if strategy_upper in str(strat_val).upper():
                        result.append(t)
                except (json.JSONDecodeError, TypeError):
                    continue
        return result

    def _filter_closed_by_sector(self, trades: list, sector: str) -> list:
        sector_upper = sector.upper()
        result = []
        for t in trades:
            if t.exit_date is None or t.pnl_percentage is None:
                continue
            matched = False
            if t.strategy_signal:
                try:
                    sig = json.loads(t.strategy_signal)
                    sec_val = sig.get("sector", "")
                    if sector_upper in str(sec_val).upper():
                        matched = True
                except (json.JSONDecodeError, TypeError):
                    pass
            if not matched and t.research_summary:
                if sector_upper in t.research_summary.upper():
                    matched = True
            if matched:
                result.append(t)
        return result

    def _calc_performance(
        self, trades: list, groupby: str, groupval: str
    ) -> dict:
        empty = {
            groupby: groupval,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "expectancy": 0.0,
            "avg_holding_days": 0.0,
            "best_trade": None,
            "worst_trade": None,
        }
        if not trades:
            return empty

        wins = [t for t in trades if (t.pnl_percentage or 0.0) > 0]
        losses = [t for t in trades if (t.pnl_percentage or 0.0) <= 0]
        best = max(trades, key=lambda t: t.pnl_percentage or 0.0)
        worst = min(trades, key=lambda t: t.pnl_percentage or 0.0)

        avg_win = (
            sum(t.pnl_percentage for t in wins) / len(wins) if wins else 0.0
        )
        avg_loss = (
            sum(t.pnl_percentage for t in losses) / len(losses) if losses else 0.0
        )
        win_rate = len(wins) / len(trades) * 100
        loss_rate = 1 - win_rate / 100
        expectancy = round((win_rate / 100 * avg_win) + (loss_rate * avg_loss), 4)

        holding_vals = [t.holding_days for t in trades if t.holding_days is not None]
        avg_holding = sum(holding_vals) / len(holding_vals) if holding_vals else 0.0

        return {
            groupby: groupval,
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "expectancy": expectancy,
            "avg_holding_days": round(avg_holding, 1),
            "best_trade": {
                "symbol": best.symbol,
                "pnl_pct": round(best.pnl_percentage, 2),
            },
            "worst_trade": {
                "symbol": worst.symbol,
                "pnl_pct": round(worst.pnl_percentage, 2),
            },
        }

    def _calc_sharpe(self, days: int) -> Optional[float]:
        """Annualised Sharpe ratio from portfolio snapshots (returns None if insufficient data)."""
        try:
            history = self.db.get_portfolio_history(days)
            values = [s.total_value for s in history if s.total_value]
            if len(values) < 5:
                return None
            # Snapshots are newest-first; compute returns from adjacent pairs.
            returns = [
                (values[i] - values[i + 1]) / values[i + 1]
                for i in range(len(values) - 1)
            ]
            mean_r = sum(returns) / len(returns)
            variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            std_r = math.sqrt(variance)
            if std_r == 0:
                return None
            risk_free_daily = 0.065 / 252  # ~6.5% p.a.
            return round((mean_r - risk_free_daily) / std_r * math.sqrt(252), 2)
        except Exception:  # noqa: BLE001
            return None
