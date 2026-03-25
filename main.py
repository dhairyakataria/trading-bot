"""
Indian Stock Market Swing Trading Bot
Multi-Agent System with Angel One SmartAPI
"""
from __future__ import annotations

import sys
import time


def main() -> None:
    """Boot sequence for the trading bot.

    Steps
    -----
    1.  Load configuration (config.yaml + .env)
    2.  Setup logging
    3-9. Initialize Orchestrator — internally creates:
         - DatabaseManager
         - BudgetManager + LLMRouter
         - AngelOneClient (logs in unless paper_trading=True)
         - TechnicalIndicators, NewsFetcher, WebSearchTool
         - All agents (Universe, Quant, Research, Risk, Exit, Journal, Execution)
         - CircuitBreaker + TelegramNotifier
    10. Initialize TradingScheduler and register all jobs
    11. Start the scheduler and block the main thread
    """

    # ── Step 1: Config ────────────────────────────────────────────────────────
    try:
        from src.utils.config import get_config
        config = get_config()
    except Exception as exc:
        print(f"[CRITICAL] Failed to load config: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: Logging ───────────────────────────────────────────────────────
    from src.utils.logger import setup_logging, get_logger

    log_level = config.get("logging", "level", default="INFO")
    log_file  = config.get("logging", "file",  default="logs/trading_bot.log")
    setup_logging(log_level=log_level, log_file=log_file)

    logger = get_logger("main")
    logger.info("=" * 60)
    logger.info("TRADING BOT STARTING")
    logger.info("=" * 60)

    # ── Steps 3–9: Orchestrator ───────────────────────────────────────────────
    # The Orchestrator creates every component internally in dependency order.
    # A broker login failure is handled as degraded mode (paper trading fallback).
    from src.agents.orchestrator import Orchestrator

    try:
        orchestrator = Orchestrator(config)
    except Exception as exc:
        logger.critical(
            "Failed to initialize orchestrator: %s", exc, exc_info=True
        )
        sys.exit(1)

    logger.info("All components initialized successfully")

    # ── Step 10: Scheduler ────────────────────────────────────────────────────
    from src.scheduler import TradingScheduler

    scheduler = TradingScheduler(orchestrator, config)
    scheduler.setup_schedule()

    # ── Step 11: Start ────────────────────────────────────────────────────────
    try:
        logger.info("Starting scheduler...")
        orchestrator.notifier.send_alert("Trading Bot is ONLINE 🚀")
        scheduler.start()

        # Keep the main thread alive; APScheduler runs jobs in background threads
        while True:
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown signal received.")

    except Exception as exc:
        logger.critical("Unexpected fatal error: %s", exc, exc_info=True)

    finally:
        logger.info("Stopping bot...")
        scheduler.stop()
        orchestrator.shutdown()
        orchestrator.notifier.send_alert("Trading Bot is OFFLINE 🔴")
        logger.info("Trading bot stopped.")


if __name__ == "__main__":
    main()
