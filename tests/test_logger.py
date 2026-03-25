"""Tests for src/utils/logger.py."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

import src.utils.logger as logger_module
from src.utils.logger import get_logger, get_trade_logger, setup_logging


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_logger_state():
    """Reset module flag and clear all root + trades handlers before each test."""
    logger_module._initialized = False

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    trades = logging.getLogger("trades")
    for h in trades.handlers[:]:
        trades.removeHandler(h)
        h.close()

    yield

    logger_module._initialized = False
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    trades = logging.getLogger("trades")
    for h in trades.handlers[:]:
        trades.removeHandler(h)
        h.close()


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_creates_two_handlers(self, tmp_path: Path) -> None:
        setup_logging(log_file=tmp_path / "test.log")
        our_handlers = [
            h for h in logging.getLogger().handlers
            if type(h).__name__ != "LogCaptureHandler"
        ]
        assert len(our_handlers) == 2

    def test_handler_types(self, tmp_path: Path) -> None:
        setup_logging(log_file=tmp_path / "test.log")
        types = {type(h).__name__ for h in logging.getLogger().handlers}
        assert "StreamHandler" in types
        assert "RotatingFileHandler" in types

    def test_creates_log_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "sub" / "app.log"
        setup_logging(log_file=log_file)
        logging.getLogger("test").info("hello")
        assert log_file.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "app.log"
        setup_logging(log_file=deep)
        assert deep.parent.exists()

    def test_idempotent_multiple_calls(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        setup_logging(log_file=log_file)
        setup_logging(log_file=log_file)
        our_handlers = [
            h for h in logging.getLogger().handlers
            if type(h).__name__ != "LogCaptureHandler"
        ]
        assert len(our_handlers) == 2

    def test_sets_initialized_flag(self, tmp_path: Path) -> None:
        assert logger_module._initialized is False
        setup_logging(log_file=tmp_path / "test.log")
        assert logger_module._initialized is True

    def test_console_handler_level_is_info(self, tmp_path: Path) -> None:
        setup_logging(log_file=tmp_path / "test.log")
        stream_handlers = [
            h for h in logging.getLogger().handlers
            if type(h).__name__ == "StreamHandler"
        ]
        assert stream_handlers[0].level == logging.INFO

    def test_file_handler_respects_log_level(self, tmp_path: Path) -> None:
        setup_logging(log_level="WARNING", log_file=tmp_path / "test.log")
        file_handlers = [
            h for h in logging.getLogger().handlers
            if type(h).__name__ == "RotatingFileHandler"
        ]
        assert file_handlers[0].level == logging.WARNING

    def test_log_format_contains_name_and_level(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file)
        logging.getLogger("mymodule").warning("check format")
        content = log_file.read_text(encoding="utf-8")
        assert "mymodule" in content
        assert "WARNING" in content

    def test_rotating_file_max_bytes(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        setup_logging(log_file=log_file, max_bytes=512, backup_count=2)
        file_handlers = [
            h for h in logging.getLogger().handlers
            if type(h).__name__ == "RotatingFileHandler"
        ]
        assert file_handlers[0].maxBytes == 512
        assert file_handlers[0].backupCount == 2


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_named_logger(self, tmp_path: Path) -> None:
        setup_logging(log_file=tmp_path / "test.log")
        log = get_logger("agents.quant")
        assert log.name == "agents.quant"

    def test_returns_logging_logger_instance(self, tmp_path: Path) -> None:
        setup_logging(log_file=tmp_path / "test.log")
        assert isinstance(get_logger("some.module"), logging.Logger)

    def test_triggers_setup_if_not_initialized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_logger should call setup_logging with defaults if not yet set up."""
        monkeypatch.setattr(logger_module, "_PROJECT_ROOT", tmp_path)
        (tmp_path / "logs").mkdir(exist_ok=True)
        log = get_logger("lazy.init")
        assert logger_module._initialized is True
        assert isinstance(log, logging.Logger)

    def test_different_names_are_different_loggers(self, tmp_path: Path) -> None:
        setup_logging(log_file=tmp_path / "test.log")
        log_a = get_logger("agents.quant")
        log_b = get_logger("agents.risk")
        assert log_a is not log_b
        assert log_a.name != log_b.name

    def test_same_name_returns_same_logger(self, tmp_path: Path) -> None:
        setup_logging(log_file=tmp_path / "test.log")
        assert get_logger("agents.quant") is get_logger("agents.quant")


# ---------------------------------------------------------------------------
# get_trade_logger
# ---------------------------------------------------------------------------

class TestGetTradeLogger:
    def test_returns_trades_logger(self, tmp_path: Path) -> None:
        tlog = get_trade_logger(trade_log_file=tmp_path / "trades.log")
        assert tlog.name == "trades"

    def test_does_not_propagate(self, tmp_path: Path) -> None:
        tlog = get_trade_logger(trade_log_file=tmp_path / "trades.log")
        assert tlog.propagate is False

    def test_creates_trade_log_file(self, tmp_path: Path) -> None:
        trade_log = tmp_path / "trades.log"
        tlog = get_trade_logger(trade_log_file=trade_log)
        tlog.info("BUY RELIANCE 10 @ 2450.50")
        assert trade_log.exists()

    def test_trade_log_contains_message(self, tmp_path: Path) -> None:
        trade_log = tmp_path / "trades.log"
        tlog = get_trade_logger(trade_log_file=trade_log)
        tlog.info("BUY INFY 20 @ 1800")
        content = trade_log.read_text(encoding="utf-8")
        assert "BUY INFY" in content

    def test_idempotent_second_call(self, tmp_path: Path) -> None:
        trade_log = tmp_path / "trades.log"
        tlog1 = get_trade_logger(trade_log_file=trade_log)
        tlog2 = get_trade_logger(trade_log_file=trade_log)
        assert tlog1 is tlog2
        assert len(tlog1.handlers) == 1  # handler not doubled

    def test_does_not_write_to_root_log(self, tmp_path: Path) -> None:
        main_log = tmp_path / "main.log"
        trade_log = tmp_path / "trades.log"
        setup_logging(log_file=main_log)
        tlog = get_trade_logger(trade_log_file=trade_log)
        tlog.info("SELL TCS 5 @ 3500")
        main_content = main_log.read_text(encoding="utf-8") if main_log.exists() else ""
        # Trade message must NOT appear in the main log (propagate=False)
        assert "SELL TCS" not in main_content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        deep = tmp_path / "nested" / "dir" / "trades.log"
        get_trade_logger(trade_log_file=deep)
        assert deep.parent.exists()

    def test_rotating_handler_config(self, tmp_path: Path) -> None:
        trade_log = tmp_path / "trades.log"
        tlog = get_trade_logger(
            trade_log_file=trade_log, max_bytes=1024, backup_count=3
        )
        rh = tlog.handlers[0]
        assert rh.maxBytes == 1024
        assert rh.backupCount == 3
