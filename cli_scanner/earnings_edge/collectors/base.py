"""Base classes for robust external data collection."""

from __future__ import annotations

import time
import logging
from typing import Callable, TypeVar, Optional
from datetime import datetime, timezone

T = TypeVar("T")

logger = logging.getLogger("earnings_edge.collectors")


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open (too many recent failures)."""
    pass


class BaseCollector:
    """
    Provides retry-with-backoff and circuit-breaker semantics for external calls.

    Circuit breaker:
    - CLOSED: normal operation, requests pass through
    - OPEN: after ``circuit_threshold`` consecutive failures, reject all calls
    - HALF_OPEN: after ``circuit_reset_secs``, allow one trial call
    """

    def __init__(
        self,
        name: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        circuit_threshold: int = 5,
        circuit_reset_secs: int = 60,
    ):
        self.name = name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.circuit_threshold = circuit_threshold
        self.circuit_reset_secs = circuit_reset_secs

        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_opened_at: Optional[float] = None
        self._last_success: Optional[datetime] = None

    def with_retry(self, fn: Callable[[], T]) -> T:
        """Execute fn with retry and circuit-breaker protection."""
        self._check_circuit()

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = fn()
                self._on_success()
                return result
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                    logger.warning(
                        "[%s] attempt %d/%d failed: %s, retrying in %.1fs",
                        self.name, attempt, self.max_retries, exc, delay,
                    )
                    time.sleep(delay)

        self._on_failure()
        assert last_exc is not None
        raise last_exc

    def _check_circuit(self) -> None:
        if not self._circuit_open:
            return
        elapsed = time.monotonic() - (self._circuit_opened_at or 0)
        if elapsed >= self.circuit_reset_secs:
            logger.info("[%s] circuit breaker half-open, trying request", self.name)
            self._circuit_open = False
            return
        raise CircuitBreakerOpen(
            f"[{self.name}] circuit breaker open, "
            f"reset in {self.circuit_reset_secs - elapsed:.0f}s"
        )

    def _on_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open = False
        self._last_success = datetime.now(timezone.utc)

    def _on_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_threshold:
            self._circuit_open = True
            self._circuit_opened_at = time.monotonic()
            logger.error(
                "[%s] circuit breaker opened after %d consecutive failures",
                self.name, self._consecutive_failures,
            )

    @property
    def is_healthy(self) -> bool:
        return not self._circuit_open
