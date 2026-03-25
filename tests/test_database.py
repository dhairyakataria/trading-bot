"""Unit tests for the database layer (models + DatabaseManager).

Run from the project root:
    pytest tests/test_database.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.database.db_manager import DatabaseManager
from src.database.models import (
    AgentLog,
    DailySummary,
    PortfolioSnapshot,
    Signal,
    Trade,
    WatchlistItem,
)


# --------------------------------------------------------------------------- #
# Shared fixture                                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """Fresh DatabaseManager backed by a temporary SQLite file."""
    return DatabaseManager(db_path=str(tmp_path / "test_bot.db"))


# --------------------------------------------------------------------------- #
# 1. DB creation and table setup                                               #
# --------------------------------------------------------------------------- #

class TestSchemaCreation:
    """All seven tables must be present after __init__."""

    def test_all_tables_exist(self, db: DatabaseManager) -> None:
        expected = {
            "trades",
            "watchlist",
            "signals",
            "portfolio_snapshots",
            "agent_logs",
            "daily_summary",
            "system_state",
        }
        with db._get_connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        actual = {r["name"] for r in rows}
        assert expected.issubset(actual)

    def test_indexes_exist(self, db: DatabaseManager) -> None:
        expected_indexes = {
            "idx_trades_symbol",
            "idx_trades_entry_date",
            "idx_watchlist_date_sym",
            "idx_signals_date_sym",
            "idx_portfolio_date",
            "idx_agent_logs_date",
        }
        with db._get_connection() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        actual = {r["name"] for r in rows}
        assert expected_indexes.issubset(actual)

    def test_second_init_is_idempotent(self, tmp_path: Path) -> None:
        """Re-creating DatabaseManager against the same file must not raise."""
        path = str(tmp_path / "idempotent.db")
        DatabaseManager(db_path=path)
        DatabaseManager(db_path=path)  # must not raise


# --------------------------------------------------------------------------- #
# 2. Insert and retrieve a trade                                               #
# --------------------------------------------------------------------------- #

class TestInsertRetrieveTrade:
    def test_record_trade_returns_positive_id(self, db: DatabaseManager) -> None:
        trade = Trade(
            symbol="RELIANCE",
            trade_type="BUY",
            quantity=10,
            price=2500.0,
            stop_loss=2425.0,
            target_price=2625.0,
            research_summary="Breakout above 52-week high.",
            entry_date="2026-03-16 09:30:00",
        )
        trade_id = db.record_trade(trade)
        assert isinstance(trade_id, int)
        assert trade_id > 0

    def test_open_trades_returns_inserted_trade(self, db: DatabaseManager) -> None:
        trade = Trade(
            symbol="INFY",
            trade_type="BUY",
            quantity=5,
            price=1800.0,
            entry_date="2026-03-16 10:00:00",
        )
        db.record_trade(trade)

        open_trades = db.get_open_trades()
        assert len(open_trades) == 1
        fetched = open_trades[0]
        assert fetched.symbol == "INFY"
        assert fetched.quantity == 5
        assert fetched.price == pytest.approx(1800.0)
        assert fetched.exit_date is None
        assert fetched.status == "PENDING"

    def test_multiple_open_trades(self, db: DatabaseManager) -> None:
        for sym in ("TCS", "WIPRO", "HCL"):
            db.record_trade(Trade(sym, "BUY", 2, 1000.0))
        assert len(db.get_open_trades()) == 3

    def test_trade_appears_in_history(self, db: DatabaseManager) -> None:
        db.record_trade(Trade("HDFC", "BUY", 3, 1500.0, entry_date="2026-03-15 09:30:00"))
        history = db.get_trade_history(days=30)
        assert any(t.symbol == "HDFC" for t in history)


# --------------------------------------------------------------------------- #
# 3. Update trade exit — PnL calculation                                      #
# --------------------------------------------------------------------------- #

class TestUpdateTradeExit:
    def test_pnl_calculated_correctly(self, db: DatabaseManager) -> None:
        """(exit - entry) * qty  and  (exit - entry) / entry * 100."""
        trade_id = db.record_trade(
            Trade("TCS", "BUY", 5, 4000.0, entry_date="2026-03-10 09:30:00")
        )
        db.update_trade_exit(
            trade_id=trade_id,
            exit_price=4200.0,
            exit_date="2026-03-16 14:00:00",
        )

        closed = db.get_trade_history(days=30)
        assert len(closed) == 1
        t = closed[0]
        assert t.exit_price == pytest.approx(4200.0)
        assert t.pnl == pytest.approx(1000.0)          # (4200-4000)*5
        assert t.pnl_percentage == pytest.approx(5.0)  # 200/4000*100
        assert t.holding_days == 6
        assert t.status == "EXECUTED"

    def test_loss_trade_negative_pnl(self, db: DatabaseManager) -> None:
        trade_id = db.record_trade(
            Trade("ZOMATO", "BUY", 100, 200.0, entry_date="2026-03-15 09:30:00")
        )
        db.update_trade_exit(trade_id=trade_id, exit_price=180.0)
        t = db.get_trade_history(days=30)[0]
        assert t.pnl == pytest.approx(-2000.0)   # (180-200)*100
        assert t.pnl_percentage < 0

    def test_closed_trade_not_in_open_trades(self, db: DatabaseManager) -> None:
        trade_id = db.record_trade(Trade("SBIN", "BUY", 10, 600.0))
        db.update_trade_exit(trade_id=trade_id, exit_price=630.0)
        assert db.get_open_trades() == []

    def test_update_nonexistent_trade_is_safe(self, db: DatabaseManager) -> None:
        """Calling update_trade_exit on a missing id must not raise."""
        db.update_trade_exit(trade_id=9999, exit_price=100.0)  # no exception

    def test_holding_days_same_day_is_zero(self, db: DatabaseManager) -> None:
        trade_id = db.record_trade(
            Trade("ITC", "BUY", 20, 450.0, entry_date="2026-03-16 09:30:00")
        )
        db.update_trade_exit(
            trade_id=trade_id,
            exit_price=460.0,
            exit_date="2026-03-16 15:00:00",
        )
        t = db.get_trade_history(days=30)[0]
        assert t.holding_days == 0


# --------------------------------------------------------------------------- #
# 4. Portfolio snapshot save and retrieve                                      #
# --------------------------------------------------------------------------- #

class TestPortfolioSnapshot:
    def test_save_and_retrieve_snapshot(self, db: DatabaseManager) -> None:
        per_stock = {"RELIANCE": {"qty": 5, "value": 12_500.0}}
        snapshot = PortfolioSnapshot(
            date="2026-03-16",
            time="15:30:00",
            total_value=52_500.0,
            invested_amount=12_500.0,
            available_cash=40_000.0,
            unrealized_pnl=250.0,
            realized_pnl_today=500.0,
            open_positions=1,
            snapshot_data=json.dumps(per_stock),
        )
        db.save_portfolio_snapshot(snapshot)

        history = db.get_portfolio_history(days=10)
        assert len(history) == 1
        s = history[0]
        assert s.date == "2026-03-16"
        assert s.time == "15:30:00"
        assert s.total_value == pytest.approx(52_500.0)
        assert s.available_cash == pytest.approx(40_000.0)
        assert s.open_positions == 1
        assert json.loads(s.snapshot_data)["RELIANCE"]["qty"] == 5  # type: ignore[arg-type]

    def test_multiple_snapshots_same_day(self, db: DatabaseManager) -> None:
        for t in ("09:30:00", "12:00:00", "15:30:00"):
            db.save_portfolio_snapshot(
                PortfolioSnapshot(date="2026-03-16", time=t, total_value=50_000.0)
            )
        history = db.get_portfolio_history(days=10)
        assert len(history) == 3

    def test_history_respects_days_filter(self, db: DatabaseManager) -> None:
        db.save_portfolio_snapshot(
            PortfolioSnapshot(date="2025-01-01", time="15:30:00", total_value=50_000.0)
        )
        # A snapshot from ~15 months ago should NOT appear in a 30-day window
        history = db.get_portfolio_history(days=30)
        assert len(history) == 0


# --------------------------------------------------------------------------- #
# 5. System state get / set                                                    #
# --------------------------------------------------------------------------- #

class TestSystemState:
    def test_missing_key_returns_none(self, db: DatabaseManager) -> None:
        assert db.get_system_state("nonexistent_key") is None

    def test_set_and_get_value(self, db: DatabaseManager) -> None:
        db.set_system_state("daily_loss_so_far", "1500.00")
        assert db.get_system_state("daily_loss_so_far") == "1500.00"

    def test_upsert_overwrites_value(self, db: DatabaseManager) -> None:
        db.set_system_state("is_trading_paused", "false")
        db.set_system_state("is_trading_paused", "true")
        assert db.get_system_state("is_trading_paused") == "true"

    def test_multiple_keys_are_independent(self, db: DatabaseManager) -> None:
        db.set_system_state("key_a", "alpha")
        db.set_system_state("key_b", "beta")
        assert db.get_system_state("key_a") == "alpha"
        assert db.get_system_state("key_b") == "beta"

    def test_special_characters_in_value(self, db: DatabaseManager) -> None:
        db.set_system_state("json_state", '{"paused": true, "reason": "loss limit"}')
        raw = db.get_system_state("json_state")
        parsed = json.loads(raw)  # type: ignore[arg-type]
        assert parsed["paused"] is True


# --------------------------------------------------------------------------- #
# Bonus: watchlist, signals, agent logs, daily summary, performance stats     #
# --------------------------------------------------------------------------- #

class TestWatchlist:
    def test_save_and_retrieve_watchlist(self, db: DatabaseManager) -> None:
        stocks = [
            WatchlistItem("RELIANCE", price=2500.0, sector="Energy", in_index="NIFTY_50"),
            WatchlistItem("TCS",      price=4000.0, sector="IT",     in_index="NIFTY_50"),
        ]
        db.save_watchlist("2026-03-16", stocks)
        latest = db.get_latest_watchlist()
        assert len(latest) == 2
        symbols = {s.symbol for s in latest}
        assert symbols == {"RELIANCE", "TCS"}

    def test_save_watchlist_replaces_existing(self, db: DatabaseManager) -> None:
        db.save_watchlist("2026-03-16", [WatchlistItem("RELIANCE", price=2500.0)])
        db.save_watchlist("2026-03-16", [WatchlistItem("TCS", price=4000.0)])
        latest = db.get_latest_watchlist()
        assert len(latest) == 1
        assert latest[0].symbol == "TCS"


class TestSignals:
    def test_record_signal_returns_id(self, db: DatabaseManager) -> None:
        signal = Signal(
            symbol="INFY",
            date="2026-03-16",
            signal_type="BUY",
            signal_source="QUANT",
            strength=0.85,
            indicators=json.dumps({"RSI": 62, "MACD": 0.5}),
        )
        sig_id = db.record_signal(signal)
        assert isinstance(sig_id, int)
        assert sig_id > 0


class TestAgentLogs:
    def test_log_agent_activity_does_not_raise(self, db: DatabaseManager) -> None:
        db.log_agent_activity(
            agent_name="quant",
            session_type="morning_briefing",
            input_data={"symbols": ["TCS", "INFY"]},
            output_data={"signals": 2},
            llm_provider_used="gemini_flash",
            llm_calls_count=3,
            search_calls_count=1,
            duration_seconds=4.2,
        )
        # Verify the row exists
        with db._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM agent_logs").fetchone()[0]
        assert count == 1


class TestDailySummary:
    def test_save_and_upsert_daily_summary(self, db: DatabaseManager) -> None:
        s = DailySummary(
            date="2026-03-16",
            trades_executed=3,
            trades_profitable=2,
            total_pnl=1200.0,
        )
        db.save_daily_summary(s)
        db.save_daily_summary(
            DailySummary(date="2026-03-16", trades_executed=4, total_pnl=1500.0)
        )
        # Only one row should exist (UNIQUE on date)
        with db._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_summary WHERE date='2026-03-16'"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["trades_executed"] == 4


class TestPerformanceStats:
    def test_empty_db_returns_zero_win_rate(self, db: DatabaseManager) -> None:
        stats = db.get_performance_stats(days=30)
        assert stats["win_rate_pct"] == 0.0

    def test_stats_with_mixed_trades(self, db: DatabaseManager) -> None:
        # Two winning trades, one losing trade
        for sym, entry, exit_ in [
            ("A", 100.0, 120.0),  # +20 each × 10 = +200
            ("B", 200.0, 210.0),  # +10 each × 10 = +100
            ("C", 150.0, 140.0),  # -10 each × 10 = -100
        ]:
            tid = db.record_trade(
                Trade(sym, "BUY", 10, entry, entry_date="2026-03-15 09:30:00")
            )
            db.update_trade_exit(tid, exit_, exit_date="2026-03-16 14:00:00")

        stats = db.get_performance_stats(days=30)
        assert stats["total_trades"] == 3
        assert stats["winning_trades"] == 2
        assert stats["losing_trades"] == 1
        assert stats["win_rate_pct"] == pytest.approx(66.67, abs=0.01)
        assert stats["total_pnl"] == pytest.approx(200.0)
