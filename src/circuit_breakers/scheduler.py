"""Scheduler — manages timed pipeline execution using APScheduler."""
from __future__ import annotations

from typing import Any, Callable


class TradingScheduler:
    """Schedules the trading pipeline according to market hours and config.

    Responsibilities:
    - Schedule pre-market, intra-day scan, and post-market jobs
    - Respect NSE trading calendar (skip weekends and public holidays)
    - Provide start/stop lifecycle methods for the scheduler process
    - Allow dynamic job registration for one-off or recurring tasks
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._scheduler: Any = None

    def start(self) -> None:
        """Start the APScheduler background scheduler."""
        raise NotImplementedError

    def stop(self) -> None:
        """Shut down the scheduler gracefully."""
        raise NotImplementedError

    def add_job(
        self,
        func: Callable[..., Any],
        trigger: str,
        **trigger_kwargs: Any,
    ) -> str:
        """Register a job and return its job ID."""
        raise NotImplementedError

    def remove_job(self, job_id: str) -> None:
        """Remove a previously registered job."""
        raise NotImplementedError

    def is_market_day(self) -> bool:
        """Return True if today is an NSE trading day."""
        raise NotImplementedError

    def is_market_hours(self) -> bool:
        """Return True if the current time is within NSE trading hours."""
        raise NotImplementedError
