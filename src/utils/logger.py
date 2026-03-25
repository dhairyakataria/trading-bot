"""Logging setup for the trading bot.

Call ``setup_logging()`` once at application startup.  Everywhere else, use
``get_logger(__name__)`` to obtain a named logger.  Trade-specific events
should use the logger returned by ``get_trade_logger()``.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

# Project root: src/utils/logger.py  →  src/utils/  →  src/  →  trading-bot/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level flag so setup_logging() is idempotent.
_initialized: bool = False


# ------------------------------------------------------------------ #
# Core setup                                                           #
# ------------------------------------------------------------------ #

def setup_logging(
    log_level: str = "INFO",
    log_file: str | Path = "logs/trading_bot.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> None:
    """Configure the root logger with a console handler and a rotating file handler.

    Should be called **once** at application startup (e.g. in ``main.py``).
    Subsequent calls are no-ops.

    Args:
        log_level:    Minimum level written to the log *file* (e.g. ``"DEBUG"``).
                      The console handler always uses ``INFO``.
        log_file:     Path to the main log file.  Relative paths are resolved
                      against the project root.
        max_bytes:    Maximum size of a single log file before rotation.
        backup_count: Number of rotated backup files to retain.
    """
    global _initialized
    if _initialized:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Handlers apply their own level filters.

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # --- Console handler: INFO and above ---
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # --- Rotating file handler: configured level (DEBUG by default) ---
    resolved = _resolve_path(log_file)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    file_level = getattr(logging, log_level.upper(), logging.DEBUG)
    rotating = logging.handlers.RotatingFileHandler(
        filename=resolved,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    rotating.setLevel(file_level)
    rotating.setFormatter(formatter)
    root.addHandler(rotating)

    _initialized = True


# ------------------------------------------------------------------ #
# Named logger factory                                                 #
# ------------------------------------------------------------------ #

def get_logger(name: str) -> logging.Logger:
    """Return a named :class:`logging.Logger`, triggering setup if needed.

    Args:
        name: Logger name — conventionally the module path, e.g.
              ``"agents.quant"`` or ``__name__``.

    Returns:
        A configured :class:`logging.Logger` instance.

    Example::

        from src.utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Universe built with %d symbols", len(symbols))
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)


# ------------------------------------------------------------------ #
# Trade-specific logger                                                #
# ------------------------------------------------------------------ #

def get_trade_logger(
    trade_log_file: str | Path = "logs/trades.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Return a logger that writes trade events exclusively to *trades.log*.

    This logger does **not** propagate to the root logger, keeping trade
    records in a dedicated file that is easy to audit independently.

    Args:
        trade_log_file: Path to the trade log file.
        max_bytes:      Maximum size per file before rotation.
        backup_count:   Number of rotated backup files to retain.

    Returns:
        Logger named ``"trades"``.

    Example::

        tlog = get_trade_logger()
        tlog.info("ENTRY | RELIANCE | qty=10 | price=2450.50 | sl=2377.50")
    """
    logger = logging.getLogger("trades")

    # Already configured — return immediately to avoid duplicate handlers.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Keep trade logs out of the main log file.

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    trade_path = _resolve_path(trade_log_file)
    trade_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        filename=trade_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _resolve_path(path: str | Path) -> Path:
    """Return an absolute path, resolving relative paths against project root."""
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p
