"""Trading Scheduler — manages all timed agent activities using APScheduler."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

_log = logging.getLogger("scheduler")
_IST = ZoneInfo("Asia/Kolkata")


class TradingScheduler:
    """Manages the daily schedule of all agent activities.

    Uses APScheduler's BackgroundScheduler so all jobs run in daemon threads
    without blocking the main loop.

    Job error policy
    ----------------
    - If a job raises, log the error and send a Telegram alert, but never
      stop the scheduler.
    - Missed jobs (system was off during a scheduled window) are coalesced
      into a single catch-up run, not backfilled individually.
    """

    def __init__(self, orchestrator: Any, config: Any) -> None:
        self.orchestrator = orchestrator
        self.config = config
        self._scheduler = BackgroundScheduler(
            job_defaults={
                "coalesce":           True,   # collapse multiple missed runs into one
                "max_instances":      1,      # never run the same job in parallel
                "misfire_grace_time": 300,    # allow up to 5 min late start
            },
            timezone=_IST,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _wrap(self, func: Any, job_name: str):
        """Wrap a job callable with error handling and Telegram alerting."""
        def _inner() -> None:
            try:
                _log.info("Job starting: %s", job_name)
                func()
                _log.info("Job complete: %s", job_name)
            except Exception as exc:
                _log.error("Job '%s' failed: %s", job_name, exc, exc_info=True)
                try:
                    self.orchestrator.notifier.send_alert(
                        f"Scheduled job *{job_name}* failed:\n{exc}",
                        level="ERROR",
                    )
                except Exception:
                    pass  # notification failure must never crash the scheduler

        _inner.__name__ = job_name
        return _inner

    # ── Public API ─────────────────────────────────────────────────────────────

    def setup_schedule(self) -> None:
        """Register all scheduled jobs with the APScheduler instance."""
        o = self.orchestrator

        # Pre-market routine: 8:00 AM IST, Mon–Fri
        self._scheduler.add_job(
            self._wrap(o.run_morning_routine, "run_morning_routine"),
            CronTrigger(hour=8, minute=0, day_of_week="mon-fri", timezone=_IST),
            id="morning_routine",
            name="Pre-market routine (08:00)",
        )

        # Data initialization buffer: 9:00 AM IST, Mon–Fri
        # 15-minute buffer before market open for broker connectivity check,
        # price pre-caching, and indicator warm-up
        self._scheduler.add_job(
            self._wrap(o.run_data_init, "run_data_init"),
            CronTrigger(hour=9, minute=0, day_of_week="mon-fri", timezone=_IST),
            id="data_init",
            name="Pre-market data init (09:00)",
        )

        # Main trading cycle: every hour from 9:30 AM to 2:30 PM
        # fires at 9:30, 10:30, 11:30, 12:30, 13:30, 14:30
        self._scheduler.add_job(
            self._wrap(o.run_cycle, "run_cycle"),
            CronTrigger(hour="9-14", minute=30, day_of_week="mon-fri", timezone=_IST),
            id="trading_cycle",
            name="Intra-day trading cycle (9:30–14:30)",
        )

        # Final trading cycle: 3:00 PM (last signal scan before close)
        self._scheduler.add_job(
            self._wrap(o.run_cycle, "run_cycle_1500"),
            CronTrigger(hour=15, minute=0, day_of_week="mon-fri", timezone=_IST),
            id="trading_cycle_1500",
            name="Final trading cycle (15:00)",
        )

        # Exit monitor: every 15 minutes from 9:00 AM to 3:45 PM
        self._scheduler.add_job(
            self._wrap(o.run_exit_check_only, "run_exit_check_only"),
            CronTrigger(minute="*/15", hour="9-15", day_of_week="mon-fri", timezone=_IST),
            id="exit_monitor",
            name="15-min exit monitor",
        )

        # End-of-day routine: 3:45 PM IST
        self._scheduler.add_job(
            self._wrap(o.run_eod_routine, "run_eod_routine"),
            CronTrigger(hour=15, minute=45, day_of_week="mon-fri", timezone=_IST),
            id="eod_routine",
            name="End-of-day routine (15:45)",
        )

        # Weekly review: Saturday 10:00 AM
        self._scheduler.add_job(
            self._wrap(o.run_weekly_review, "run_weekly_review"),
            CronTrigger(hour=10, minute=0, day_of_week="sat", timezone=_IST),
            id="weekly_review",
            name="Saturday weekly review (10:00)",
        )

        # Daily state reset: midnight every day
        self._scheduler.add_job(
            self._wrap(o.reset_daily_state, "reset_daily_state"),
            CronTrigger(hour=0, minute=0, timezone=_IST),
            id="daily_reset",
            name="Midnight daily state reset",
        )

        # Universe refresh: Sunday 8:00 PM (prepare for next week)
        self._scheduler.add_job(
            self._wrap(o.refresh_universe, "refresh_universe"),
            CronTrigger(hour=20, minute=0, day_of_week="sun", timezone=_IST),
            id="universe_refresh",
            name="Sunday universe refresh (20:00)",
        )

        _log.info(
            "Schedule configured — %d jobs registered",
            len(self._scheduler.get_jobs()),
        )

    def start(self) -> None:
        """Start the background scheduler."""
        self._scheduler.start()
        _log.info("Scheduler started.")
        for job in self._scheduler.get_jobs():
            _log.info(
                "  %-45s next run: %s",
                job.name,
                job.next_run_time or "paused",
            )

    def stop(self) -> None:
        """Gracefully shut down the scheduler, waiting for any running job."""
        if self._scheduler.running:
            _log.info("Stopping scheduler (waiting for running jobs)…")
            self._scheduler.shutdown(wait=True)
            _log.info("Scheduler stopped.")

    def get_schedule_status(self) -> dict:
        """Return a dict of all jobs and their next run times."""
        return {
            job.id: {
                "name":          job.name,
                "next_run_time": str(job.next_run_time) if job.next_run_time else "paused",
                "trigger":       str(job.trigger),
            }
            for job in self._scheduler.get_jobs()
        }

    def run_now(self, job_id: str) -> None:
        """Manually trigger a scheduled job by its ID (useful for testing).

        Args:
            job_id: The job ID as registered in setup_schedule (e.g. "morning_routine").
        """
        job = self._scheduler.get_job(job_id)
        if job is None:
            available = [j.id for j in self._scheduler.get_jobs()]
            _log.error(
                "Job '%s' not found. Available job IDs: %s",
                job_id, available,
            )
            return
        job.modify(next_run_time=datetime.now(_IST))
        _log.info("Job '%s' manually triggered.", job_id)
