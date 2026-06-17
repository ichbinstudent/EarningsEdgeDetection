"""Multi-source earnings calendar fetcher with fallback chain."""

from __future__ import annotations

import logging
import os
import random
from datetime import date
from typing import List

import requests
from bs4 import BeautifulSoup

from .base import BaseCollector
from ..models import EarningsCandidate
from ..settings import get_settings

logger = logging.getLogger("earnings_edge.collectors.earnings")

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/92.0.4515.107 Safari/537.36",
]


class EarningsCalendarCollector(BaseCollector):
    """Fetch earnings calendar from multiple sources with fallback."""

    def __init__(self):
        super().__init__(
            name="earnings_calendar",
            max_retries=2,
            base_delay=2.0,
            circuit_threshold=3,
        )
        self._settings = get_settings()

    def fetch(self, target_date: date) -> List[EarningsCandidate]:
        """
        Fetch earnings for target_date using fallback chain:
        1. Investing.com (primary, most complete)
        2. Finnhub (fallback if Investing.com fails/empty)
        Merge if both available.
        """
        investing = self._safe_fetch(self._investing_fetch, target_date, "investing")
        finnhub = self._safe_fetch(self._finnhub_fetch, target_date, "finnhub")

        if investing and finnhub:
            return self._merge(investing, finnhub)
        if investing:
            return investing
        if finnhub:
            return finnhub

        logger.warning("All earnings sources empty for %s", target_date)
        return []

    def _safe_fetch(self, fn, target_date: date, source_name: str) -> List[EarningsCandidate]:
        """Wrap a source fetch with circuit-breaker retry."""
        try:
            return self.with_retry(lambda: fn(target_date))
        except Exception as exc:
            logger.warning("[%s] fetch failed for %s: %s", source_name, target_date, exc)
            return []

    def _investing_fetch(self, target_date: date) -> List[EarningsCandidate]:
        """Scrape Investing.com earnings calendar."""
        url = "https://www.investing.com/earnings-calendar/Service/getCalendarFilteredData"
        headers = {
            "User-Agent": random.choice(_USER_AGENTS),
            "X-Requested-With": "XMLHttpRequest",
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://www.investing.com/earnings-calendar/",
        }
        payload = {
            "country[]": "5",
            "dateFrom": target_date.strftime("%Y-%m-%d"),
            "dateTo": target_date.strftime("%Y-%m-%d"),
            "currentTab": "custom",
            "limit_from": 0,
        }
        resp = requests.post(url, headers=headers, data=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "data" not in data:
            raise ValueError("Investing.com: missing 'data' key in response")

        soup = BeautifulSoup(data["data"], "html.parser")
        stocks: List[EarningsCandidate] = []
        for row in soup.find_all("tr"):
            if not row.find("span", class_="earnCalCompanyName"):
                continue
            try:
                ticker = row.find("a", class_="bold").text.strip()
                span = row.find("span", class_="genToolTip")
                tooltip = span.get("data-tooltip", "") if span and span.has_attr("data-tooltip") else ""
                timing = (
                    "Pre Market" if "Before" in tooltip
                    else "Post Market" if "After" in tooltip
                    else "During Market"
                )
                stocks.append(EarningsCandidate(
                    ticker=ticker, timing=timing, source="investing",
                ))
            except Exception as exc:
                logger.debug("Investing.com row parse error: %s", exc)

        if not stocks:
            raise ValueError("Investing.com: parsed 0 candidates (markup changed?)")
        return stocks

    def _finnhub_fetch(self, target_date: date) -> List[EarningsCandidate]:
        """Fetch from Finnhub API."""
        api_key = self._settings.finnhub_api_key
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not set")

        ds = target_date.strftime("%Y-%m-%d")
        resp = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={"from": ds, "to": ds, "token": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        entries = resp.json().get("earningsCalendar", [])

        stocks: List[EarningsCandidate] = []
        for e in entries:
            symbol = e.get("symbol")
            if not symbol:
                continue
            hour = e.get("hour", "").lower()
            timing = (
                "Pre Market" if hour == "bmo"
                else "Post Market" if hour == "amc"
                else "During Market"
            )
            stocks.append(EarningsCandidate(
                ticker=symbol, timing=timing, source="finnhub",
            ))
        return stocks

    @staticmethod
    def _merge(*lists: List[EarningsCandidate]) -> List[EarningsCandidate]:
        """Deduplicate by ticker, preferring non-Unknown timing."""
        merged: dict[str, EarningsCandidate] = {}
        for candidate_list in lists:
            for c in candidate_list:
                existing = merged.get(c.ticker)
                if existing is None:
                    merged[c.ticker] = c
                elif existing.timing == "Unknown" and c.timing != "Unknown":
                    merged[c.ticker] = c
        return list(merged.values())
