"""Database Manager — thread-safe SQLite persistence layer.

Usage::

    from src.database.db_manager import DatabaseManager
    db = DatabaseManager(db_path="data/trading_bot.db")

    trade_id = db.record_trade(trade)
    db.update_trade_exit(trade_id, exit_price=2650.0)
    open_trades = db.get_open_trades()
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator, List, Optional
from zoneinfo import ZoneInfo

from src.database.models import (
    AgentLog,
    DailySummary,
    PortfolioSnapshot,
    Signal,
    Trade,
    WatchlistItem,
)
from src.utils.logger import get_logger

IST = ZoneInfo("Asia/Kolkata")
_log = get_logger("database")


# --------------------------------------------------------------------------- #
# IST timestamp helpers                                                        #
# --------------------------------------------------------------------------- #

def _ist_now() -> str:
    """Current IST datetime as ``'YYYY-MM-DD HH:MM:SS'``."""
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


def _ist_today() -> str:
    """Current IST date as ``'YYYY-MM-DD'``."""
    return datetime.now(IST).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# DatabaseManager                                                              #
# --------------------------------------------------------------------------- #

class DatabaseManager:
    """Thread-safe SQLite persistence layer for the trading bot.

    A new connection is created and closed for every operation so that the
    manager can safely be used from multiple threads without sharing state.
    A :class:`threading.Lock` serialises writes to avoid ``SQLITE_BUSY``
    errors; reads also hold the lock so that PnL calculations stay consistent
    across a read-then-write pair.

    Args:
        db_path: Path to the SQLite file.  The parent directory is created
                 automatically if it does not exist.
    """

    # ------------------------------------------------------------------ #
    # DDL — all tables and indexes                                         #
    # ------------------------------------------------------------------ #

    _DDL = """
    CREATE TABLE IF NOT EXISTS trades (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol            TEXT    NOT NULL,
        trade_type        TEXT    NOT NULL,
        quantity          INTEGER NOT NULL,
        price             REAL    NOT NULL,
        order_id          TEXT,
        status            TEXT    DEFAULT 'PENDING',
        strategy_signal   TEXT,
        research_summary  TEXT,
        risk_check_result TEXT,
        stop_loss         REAL,
        target_price      REAL,
        entry_date        TEXT,
        exit_date         TEXT,
        exit_price        REAL,
        pnl               REAL,
        pnl_percentage    REAL,
        holding_days      INTEGER,
        created_at        TEXT    DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_trades_symbol     ON trades(symbol);
    CREATE INDEX IF NOT EXISTS idx_trades_entry_date ON trades(entry_date);

    CREATE TABLE IF NOT EXISTS watchlist (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        date          TEXT    NOT NULL,
        symbol        TEXT    NOT NULL,
        price         REAL,
        avg_volume_cr REAL,
        atr_pct       REAL,
        ema_50        REAL,
        sector        TEXT,
        in_index      TEXT,
        created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_watchlist_date_sym ON watchlist(date, symbol);

    CREATE TABLE IF NOT EXISTS signals (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        date              TEXT    NOT NULL,
        symbol            TEXT    NOT NULL,
        signal_type       TEXT,
        signal_source     TEXT,
        strength          REAL,
        indicators        TEXT,
        was_acted_on      INTEGER DEFAULT 0,
        reason_if_skipped TEXT,
        created_at        TEXT    DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_signals_date_sym ON signals(date, symbol);

    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        date               TEXT    NOT NULL,
        time               TEXT    NOT NULL,
        total_value        REAL,
        invested_amount    REAL,
        available_cash     REAL,
        unrealized_pnl     REAL,
        realized_pnl_today REAL,
        open_positions     INTEGER,
        snapshot_data      TEXT,
        created_at         TEXT    DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_portfolio_date ON portfolio_snapshots(date);

    CREATE TABLE IF NOT EXISTS agent_logs (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        date               TEXT    NOT NULL,
        agent_name         TEXT    NOT NULL,
        session_type       TEXT,
        input_data         TEXT,
        output_data        TEXT,
        llm_provider_used  TEXT,
        llm_calls_count    INTEGER DEFAULT 0,
        search_calls_count INTEGER DEFAULT 0,
        duration_seconds   REAL    DEFAULT 0.0,
        created_at         TEXT    DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_agent_logs_date ON agent_logs(date, agent_name);

    CREATE TABLE IF NOT EXISTS daily_summary (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        date                TEXT    NOT NULL UNIQUE,
        market_outlook      TEXT,
        trades_executed     INTEGER DEFAULT 0,
        trades_profitable   INTEGER DEFAULT 0,
        total_pnl           REAL    DEFAULT 0.0,
        portfolio_value_eod REAL,
        nifty_change_pct    REAL,
        agent_summary       TEXT,
        created_at          TEXT    DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS system_state (
        key        TEXT PRIMARY KEY,
        value      TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialise_schema()
        _log.info("DatabaseManager ready — %s", self._db_path)

    # ------------------------------------------------------------------ #
    # Connection context manager                                           #
    # ------------------------------------------------------------------ #

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a configured SQLite connection; commit on success, rollback on error."""
        conn = sqlite3.connect(str(self._db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Schema initialisation                                                #
    # ------------------------------------------------------------------ #

    def _initialise_schema(self) -> None:
        """Create all tables and indexes if they do not yet exist."""
        with self._lock:
            with self._get_connection() as conn:
                conn.executescript(self._DDL)
        _log.debug("Schema ready.")

    # ------------------------------------------------------------------ #
    # Trades                                                               #
    # ------------------------------------------------------------------ #

    def record_trade(self, trade: Trade) -> int:
        """Insert a new trade record and return its auto-generated id.

        Args:
            trade: :class:`Trade` dataclass.  ``entry_date`` defaults to
                   the current IST timestamp if not provided.

        Returns:
            The ``rowid`` of the newly inserted row.
        """
        sql = """
        INSERT INTO trades (
            symbol, trade_type, quantity, price, order_id, status,
            strategy_signal, research_summary, risk_check_result,
            stop_loss, target_price, entry_date, created_at
        ) VALUES (
            :symbol, :trade_type, :quantity, :price, :order_id, :status,
            :strategy_signal, :research_summary, :risk_check_result,
            :stop_loss, :target_price, :entry_date, :created_at
        )
        """
        params = {
            "symbol":            trade.symbol,
            "trade_type":        trade.trade_type,
            "quantity":          trade.quantity,
            "price":             trade.price,
            "order_id":          trade.order_id,
            "status":            trade.status,
            "strategy_signal":   trade.strategy_signal,
            "research_summary":  trade.research_summary,
            "risk_check_result": trade.risk_check_result,
            "stop_loss":         trade.stop_loss,
            "target_price":      trade.target_price,
            "entry_date":        trade.entry_date or _ist_now(),
            "created_at":        _ist_now(),
        }
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, params)
                trade_id: int = cursor.lastrowid  # type: ignore[assignment]
        _log.info(
            "Recorded trade id=%d | %s %s qty=%d @ %.2f",
            trade_id, trade.trade_type, trade.symbol, trade.quantity, trade.price,
        )
        return trade_id

    def update_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_date: Optional[str] = None,
    ) -> None:
        """Close an open trade: write exit price, auto-calculate PnL.

        PnL is computed as ``(exit_price - entry_price) * quantity`` (long
        trades).  ``pnl_percentage`` is relative to the entry price.
        ``holding_days`` counts calendar days between entry and exit dates.

        Args:
            trade_id:   Row id of the trade to close.
            exit_price: The price at which the position was closed.
            exit_date:  ISO datetime string (IST).  Defaults to now.
        """
        if exit_date is None:
            exit_date = _ist_now()

        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT price, quantity, entry_date FROM trades WHERE id = ?",
                    (trade_id,),
                ).fetchone()
                if row is None:
                    _log.error("update_trade_exit: trade id=%d not found", trade_id)
                    return

                entry_price: float = row["price"]
                quantity: int = row["quantity"]
                entry_date_str: Optional[str] = row["entry_date"]

                pnl = round((exit_price - entry_price) * quantity, 2)
                pnl_pct = round(
                    (exit_price - entry_price) / entry_price * 100
                    if entry_price else 0.0,
                    4,
                )

                holding_days: Optional[int] = None
                if entry_date_str:
                    try:
                        entry_dt = datetime.fromisoformat(entry_date_str)
                        exit_dt = datetime.fromisoformat(exit_date)
                        holding_days = max(0, (exit_dt.date() - entry_dt.date()).days)
                    except ValueError:
                        pass

                conn.execute(
                    """
                    UPDATE trades
                    SET exit_price     = :exit_price,
                        exit_date      = :exit_date,
                        pnl            = :pnl,
                        pnl_percentage = :pnl_pct,
                        holding_days   = :holding_days,
                        status         = 'EXECUTED'
                    WHERE id = :trade_id
                    """,
                    {
                        "exit_price":  exit_price,
                        "exit_date":   exit_date,
                        "pnl":         pnl,
                        "pnl_pct":     pnl_pct,
                        "holding_days": holding_days,
                        "trade_id":    trade_id,
                    },
                )

        _log.info(
            "Closed trade id=%d | exit=%.2f pnl=%.2f (%.2f%%)",
            trade_id, exit_price, pnl, pnl_pct,
        )

    def get_open_trades(self) -> List[Trade]:
        """Return all trades that have not yet been exited."""
        sql = """
        SELECT * FROM trades
        WHERE exit_date IS NULL
          AND status NOT IN ('CANCELLED', 'FAILED')
        ORDER BY entry_date
        """
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(sql).fetchall()
        return [Trade.from_dict(dict(r)) for r in rows]

    def get_trade_history(self, days: int = 30) -> List[Trade]:
        """Return all trades (open and closed) with entry_date in the last *days* days."""
        cutoff = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = """
        SELECT * FROM trades
        WHERE entry_date >= ?
        ORDER BY entry_date DESC
        """
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(sql, (cutoff,)).fetchall()
        return [Trade.from_dict(dict(r)) for r in rows]

    # ------------------------------------------------------------------ #
    # Watchlist                                                            #
    # ------------------------------------------------------------------ #

    def save_watchlist(self, date: str, stocks: List[WatchlistItem]) -> None:
        """Replace the watchlist snapshot for *date* with *stocks*.

        The previous snapshot for the same date is deleted first so that
        re-running the morning scan does not create duplicate rows.
        """
        sql_delete = "DELETE FROM watchlist WHERE date = ?"
        sql_insert = """
        INSERT INTO watchlist (
            date, symbol, price, avg_volume_cr, atr_pct, ema_50,
            sector, in_index, created_at
        ) VALUES (
            :date, :symbol, :price, :avg_volume_cr, :atr_pct, :ema_50,
            :sector, :in_index, :created_at
        )
        """
        ts = _ist_now()
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(sql_delete, (date,))
                for item in stocks:
                    row = item.to_dict()
                    row["date"] = date
                    row["created_at"] = ts
                    conn.execute(sql_insert, row)
        _log.info("Saved watchlist for %s (%d stocks)", date, len(stocks))

    def get_latest_watchlist(self) -> List[WatchlistItem]:
        """Return the most recently saved watchlist snapshot."""
        sql = """
        SELECT * FROM watchlist
        WHERE date = (SELECT MAX(date) FROM watchlist)
        ORDER BY symbol
        """
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(sql).fetchall()
        return [WatchlistItem.from_dict(dict(r)) for r in rows]

    # ------------------------------------------------------------------ #
    # Signals                                                              #
    # ------------------------------------------------------------------ #

    def record_signal(self, signal: Signal) -> int:
        """Insert a signal and return its row id."""
        sql = """
        INSERT INTO signals (
            date, symbol, signal_type, signal_source, strength,
            indicators, was_acted_on, reason_if_skipped, created_at
        ) VALUES (
            :date, :symbol, :signal_type, :signal_source, :strength,
            :indicators, :was_acted_on, :reason_if_skipped, :created_at
        )
        """
        params = signal.to_dict()
        params.pop("id", None)
        if not params.get("date"):
            params["date"] = _ist_today()
        params["created_at"] = _ist_now()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(sql, params)
                return cursor.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Portfolio snapshots                                                  #
    # ------------------------------------------------------------------ #

    def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """Insert a portfolio snapshot row."""
        sql = """
        INSERT INTO portfolio_snapshots (
            date, time, total_value, invested_amount, available_cash,
            unrealized_pnl, realized_pnl_today, open_positions,
            snapshot_data, created_at
        ) VALUES (
            :date, :time, :total_value, :invested_amount, :available_cash,
            :unrealized_pnl, :realized_pnl_today, :open_positions,
            :snapshot_data, :created_at
        )
        """
        params = snapshot.to_dict()
        params.pop("id", None)
        params["created_at"] = _ist_now()
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(sql, params)
        _log.debug("Portfolio snapshot saved — %s %s", snapshot.date, snapshot.time)

    def get_portfolio_history(self, days: int = 30) -> List[PortfolioSnapshot]:
        """Return portfolio snapshots for the last *days* days, newest first."""
        cutoff = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = """
        SELECT * FROM portfolio_snapshots
        WHERE date >= ?
        ORDER BY date DESC, time DESC
        """
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(sql, (cutoff,)).fetchall()
        return [PortfolioSnapshot.from_dict(dict(r)) for r in rows]

    # ------------------------------------------------------------------ #
    # Agent logs                                                           #
    # ------------------------------------------------------------------ #

    def log_agent_activity(
        self,
        agent_name: str,
        session_type: str,
        input_data: Any,
        output_data: Any,
        llm_provider_used: Optional[str] = None,
        llm_calls_count: int = 0,
        search_calls_count: int = 0,
        duration_seconds: float = 0.0,
    ) -> None:
        """Append one agent session record to ``agent_logs``.

        *input_data* and *output_data* are serialised to JSON automatically
        when they are not already strings.
        """
        sql = """
        INSERT INTO agent_logs (
            date, agent_name, session_type, input_data, output_data,
            llm_provider_used, llm_calls_count, search_calls_count,
            duration_seconds, created_at
        ) VALUES (
            :date, :agent_name, :session_type, :input_data, :output_data,
            :llm_provider_used, :llm_calls_count, :search_calls_count,
            :duration_seconds, :created_at
        )
        """
        params = {
            "date":               _ist_today(),
            "agent_name":         agent_name,
            "session_type":       session_type,
            "input_data":         (
                json.dumps(input_data)
                if not isinstance(input_data, str)
                else input_data
            ),
            "output_data":        (
                json.dumps(output_data)
                if not isinstance(output_data, str)
                else output_data
            ),
            "llm_provider_used":  llm_provider_used,
            "llm_calls_count":    llm_calls_count,
            "search_calls_count": search_calls_count,
            "duration_seconds":   duration_seconds,
            "created_at":         _ist_now(),
        }
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(sql, params)

    # ------------------------------------------------------------------ #
    # Daily summary                                                        #
    # ------------------------------------------------------------------ #

    def save_daily_summary(self, summary: DailySummary) -> None:
        """Upsert the daily summary for ``summary.date``.

        Uses ``INSERT OR REPLACE`` so re-running the post-market job is safe.
        """
        sql = """
        INSERT OR REPLACE INTO daily_summary (
            date, market_outlook, trades_executed, trades_profitable,
            total_pnl, portfolio_value_eod, nifty_change_pct,
            agent_summary, created_at
        ) VALUES (
            :date, :market_outlook, :trades_executed, :trades_profitable,
            :total_pnl, :portfolio_value_eod, :nifty_change_pct,
            :agent_summary, :created_at
        )
        """
        params = summary.to_dict()
        params.pop("id", None)
        params["created_at"] = _ist_now()
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(sql, params)
        _log.info("Daily summary saved for %s", summary.date)

    # ------------------------------------------------------------------ #
    # System state                                                         #
    # ------------------------------------------------------------------ #

    def get_system_state(self, key: str) -> Optional[str]:
        """Return the persisted string value for *key*, or ``None`` if absent."""
        sql = "SELECT value FROM system_state WHERE key = ?"
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(sql, (key,)).fetchone()
        return row["value"] if row else None

    def set_system_state(self, key: str, value: str) -> None:
        """Upsert *key* → *value* in ``system_state``."""
        sql = """
        INSERT INTO system_state (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE
            SET value      = excluded.value,
                updated_at = excluded.updated_at
        """
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(sql, (key, value, _ist_now()))

    # ------------------------------------------------------------------ #
    # Performance statistics                                               #
    # ------------------------------------------------------------------ #

    def get_performance_stats(self, days: int = 30) -> dict:
        """Return aggregated metrics for closed trades in the last *days* days.

        Keys in the returned dict:
            total_trades, winning_trades, losing_trades, win_rate_pct,
            total_pnl, avg_pnl, best_trade_pnl, worst_trade_pnl,
            avg_holding_days, avg_pnl_pct, period_days
        """
        cutoff = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = """
        SELECT
            COUNT(*)                                        AS total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)      AS winning_trades,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END)      AS losing_trades,
            COALESCE(SUM(pnl),            0.0)             AS total_pnl,
            COALESCE(AVG(pnl),            0.0)             AS avg_pnl,
            MAX(pnl)                                        AS best_trade_pnl,
            MIN(pnl)                                        AS worst_trade_pnl,
            COALESCE(AVG(holding_days),   0.0)             AS avg_holding_days,
            COALESCE(AVG(pnl_percentage), 0.0)             AS avg_pnl_pct
        FROM trades
        WHERE exit_date IS NOT NULL
          AND entry_date >= ?
        """
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(sql, (cutoff,)).fetchone()

        stats: dict = dict(row) if row else {}
        total = stats.get("total_trades") or 0
        winning = stats.get("winning_trades") or 0
        stats["win_rate_pct"] = round(winning / total * 100, 2) if total > 0 else 0.0
        stats["period_days"] = days
        return stats
