"""
Base abstract and core implementations for Preemption Watchdogs.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
import logging
import random
import time
from typing import Awaitable, Callable, List, Optional

from daemon.models import PreemptionEvent

logger = logging.getLogger("sgm.watchdog.base")

PreemptionCallback = Callable[[PreemptionEvent], Awaitable[None]]


class AbstractPreemptionWatchdog(ABC):
    """Abstract interface for hypervisor spot preemption watchers."""

    def __init__(self, poll_interval_ms: int = 250, timeout_ms: int = 100) -> None:
        self.poll_interval_s: float = poll_interval_ms / 1000.0
        self.timeout_s: float = timeout_ms / 1000.0
        self.callbacks: List[PreemptionCallback] = []
        self._running: bool = False
        self._task: Optional[asyncio.Task[None]] = None
        self._preemption_detected: bool = False

    def register_callback(self, callback: PreemptionCallback) -> None:
        """Register an async callback invoked immediately upon preemption detection."""
        self.callbacks.append(callback)

    async def notify_callbacks(self, event: PreemptionEvent) -> None:
        """Invokes all registered callbacks asynchronously within SLA target (<50ms)."""
        if not self.callbacks:
            logger.warning("Preemption event detected but no callbacks registered!")
            return

        logger.critical(
            "PREEMPTION SIGNAL TRIGGERED: provider=%s, deadline_seconds=%.1f. Notifying %d callbacks.",
            event.provider,
            event.deadline_seconds,
            len(self.callbacks),
        )

        results = await asyncio.gather(
            *(cb(event) for cb in self.callbacks),
            return_exceptions=True,
        )
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error("Error in preemption callback %r: %s", self.callbacks[idx], res)

    async def start(self) -> None:
        """Start the background non-blocking polling task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name=f"{self.__class__.__name__}-loop")
        logger.info(
            "%s started with poll_interval=%.3fs, timeout=%.3fs",
            self.__class__.__name__,
            self.poll_interval_s,
            self.timeout_s,
        )

    async def stop(self) -> None:
        """Gracefully stop the polling loop and release network resources."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("%s stopped.", self.__class__.__name__)

    async def _poll_loop(self) -> None:
        """Background loop executing check_once() with jitter and backoff on errors."""
        backoff_delay = self.poll_interval_s
        max_interval_s = 0.500  # Clamp to 500ms max to prevent missing preemption window

        while self._running:
            start_time = time.monotonic()
            try:
                event = await self.check_once()
                backoff_delay = self.poll_interval_s  # Reset backoff on success

                if event is not None and not self._preemption_detected:
                    self._preemption_detected = True
                    await self.notify_callbacks(event)
                    # Once preemption is detected, we can stop polling or remain active
                    break

            except asyncio.CancelledError:
                break
            except Exception as exc:
                # Exponential backoff with jitter on network blips
                jitter = random.uniform(0.01, 0.05)
                backoff_delay = min(max_interval_s, (backoff_delay * 1.5) + jitter)
                logger.debug(
                    "Watchdog check failed (%s). Backing off for %.3fs",
                    exc,
                    backoff_delay,
                )

            elapsed = time.monotonic() - start_time
            sleep_time = max(0.0, backoff_delay - elapsed)
            try:
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break

    @abstractmethod
    async def check_once(self) -> Optional[PreemptionEvent]:
        """Execute a single polling probe against metadata endpoint."""
        pass
