"""Polygon.io API client with proper sliding-window rate limiting."""

from __future__ import annotations

import time
import logging
from collections import deque
from typing import Optional

import requests

from .base import BaseCollector
from ..settings import get_settings

logger = logging.getLogger("earnings_edge.collectors.polygon")

POLYGON_BASE = "https://api.polygon.io"


class PolygonClient(BaseCollector):
    """
    Polygon.io client with sliding-window rate limiting.

    Free tier: 5 API calls/minute. We track call timestamps and
    sleep when approaching the limit.
    """

    RATE_LIMIT_CALLS = 5
    RATE_LIMIT_WINDOW = 62  # seconds (safety margin)

    def __init__(self):
        super().__init__(
            name="polygon",
            max_retries=3,
            base_delay=15.0,
            circuit_threshold=5,
        )
        self._settings = get_settings()
        self._call_times: deque = deque()

    def _wait_for_rate_limit(self):
        """Block until we're within the rate limit window."""
        now = time.monotonic()
        # Evict timestamps outside the window
        while self._call_times and now - self._call_times[0] > self.RATE_LIMIT_WINDOW:
            self._call_times.popleft()

        if len(self._call_times) >= self.RATE_LIMIT_CALLS:
            # Wait until the oldest call exits the window
            wait = self.RATE_LIMIT_WINDOW - (now - self._call_times[0]) + 1
            logger.info("Polygon rate limit: waiting %.0fs", wait)
            time.sleep(max(wait, 1))
            self._wait_for_rate_limit()  # re-check

    def get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        """Make a rate-limited GET request to Polygon API."""
        api_key = self._settings.polygon_api_key
        if not api_key:
            logger.error("POLYGON_API_KEY not set")
            return None

        params = dict(params or {})
        params["apiKey"] = api_key

        def _fetch():
            self._wait_for_rate_limit()
            resp = requests.get(f"{POLYGON_BASE}{path}", params=params, timeout=15)
            self._call_times.append(time.monotonic())
            if resp.status_code == 429:
                logger.warning("Polygon 429: rate limited, retrying after backoff")
                time.sleep(self.base_delay)
                resp = requests.get(f"{POLYGON_BASE}{path}", params=params, timeout=15)
                self._call_times.append(time.monotonic())
            resp.raise_for_status()
            return resp.json()

        try:
            return self.with_retry(_fetch)
        except Exception as exc:
            logger.warning("Polygon GET %s failed: %s", path, exc)
            return None

    def get_daily_bars(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        """Fetch daily OHLCV bars."""
        data = self.get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
            {"adjusted": "true", "sort": "asc", "limit": 20},
        )
        if not data or data.get("resultsCount", 0) == 0:
            return []
        return data.get("results", [])
