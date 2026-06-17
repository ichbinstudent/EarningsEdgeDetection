"""
Earnings-calendar data fetching from three sources:
Investing.com (HTTP), Finnhub (API), DoltHub (MySQL).
"""

import logging
import os
import random
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pytz
import requests
from bs4 import BeautifulSoup

from .config import get_logger
from .models import EarningsCandidate

logger = get_logger("earnings")

# ── Source: Investing.com ─────────────────────────────────────────────

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) Safari/604.1",
]


def _investing_earnings(date_: date) -> List[EarningsCandidate]:
    """Scrape earnings calendar from Investing.com."""
    url = "https://www.investing.com/earnings-calendar/Service/getCalendarFilteredData"
    headers = {
        "User-Agent": random.choice(_USER_AGENTS),
        "X-Requested-With": "XMLHttpRequest",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": "https://www.investing.com/earnings-calendar/",
    }
    payload = {
        "country[]": "5",
        "dateFrom": date_.strftime("%Y-%m-%d"),
        "dateTo": date_.strftime("%Y-%m-%d"),
        "currentTab": "custom",
        "limit_from": 0,
    }
    try:
        resp = requests.post(url, headers=headers, data=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "data" not in data:
            logger.warning("Invalid Investing.com response")
            return []
        soup = BeautifulSoup(data["data"], "html.parser")
    except (requests.RequestException, ValueError) as exc:
        logger.error(f"Investing.com fetch failed: {exc}")
        return []

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
            stocks.append(EarningsCandidate(ticker=ticker, timing=timing))
        except Exception as exc:
            logger.warning(f"Row parse error: {exc}")
    return stocks


# ── Source: Finnhub ───────────────────────────────────────────────────

def _finnhub_earnings(date_: date) -> List[EarningsCandidate]:
    """Fetch earnings from Finnhub API."""
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        logger.warning("FINNHUB_API_KEY not set — skipping Finnhub")
        return []

    ds = date_.strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={"from": ds, "to": ds, "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        entries = resp.json().get("earningsCalendar", [])
    except Exception as exc:
        logger.warning(f"Finnhub fetch failed: {exc}")
        return []

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
        stocks.append(EarningsCandidate(ticker=symbol, timing=timing))
    return stocks


# ── Source: DoltHub ───────────────────────────────────────────────────

def _dolthub_earnings(date_: date) -> List[EarningsCandidate]:
    """Fetch earnings from a local DoltHub MySQL instance."""
    try:
        import mysql.connector  # type: ignore
        from mysql.connector import errorcode  # type: ignore
    except ImportError:
        logger.warning("mysql-connector-python not installed — skipping DoltHub")
        return []

    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(
            host="localhost", port=3306, user="root", password="",
            database="earnings", connection_timeout=5, buffered=True,
            use_pure=True, autocommit=True,
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SET SESSION max_execution_time=5000")
        cursor.execute(
            "SELECT act_symbol, `when` FROM earnings_calendar WHERE date = %s",
            (date_.strftime("%Y-%m-%d"),),
        )
        rows = cursor.fetchall()
    except Exception as exc:
        logger.error(f"DoltHub error: {exc}")
        return []
    finally:
        for resource in (cursor, conn):
            try:
                if resource:
                    resource.close()
            except Exception:
                pass

    stocks: List[EarningsCandidate] = []
    for row in rows:
        sym = row.get("act_symbol")
        if not sym:
            continue
        when = row.get("when", "")
        timing = (
            "Pre Market" if when in ("Before market open", "bmo")
            else "Post Market" if when in ("After market close", "amc")
            else "Unknown" if when is None
            else "During Market"
        )
        stocks.append(EarningsCandidate(ticker=sym.strip(), timing=timing))
    return stocks


# ── Merge helpers ─────────────────────────────────────────────────────

def _merge(*lists: List[EarningsCandidate]) -> List[EarningsCandidate]:
    """Deduplicate by ticker, preferring non-Unknown timing."""
    merged: Dict[str, EarningsCandidate] = {}
    for candidate_list in lists:
        for c in candidate_list:
            existing = merged.get(c.ticker)
            if existing is None:
                merged[c.ticker] = c
            elif existing.timing == "Unknown" and c.timing != "Unknown":
                merged[c.ticker] = EarningsCandidate(
                    ticker=c.ticker, timing=c.timing, earnings_date=existing.earnings_date,
                )
    return list(merged.values())


# ── Public API ────────────────────────────────────────────────────────

def fetch_earnings(
    date_: date,
    use_dolthub: bool = False,
    use_finnhub: bool = False,
    all_sources: bool = False,
) -> List[EarningsCandidate]:
    """
    Return earnings for *date_* from the selected sources.

    Priority order when merging: DoltHub > Finnhub > Investing.com.
    """
    if all_sources:
        return _merge(
            _dolthub_earnings(date_),
            _finnhub_earnings(date_),
            _investing_earnings(date_),
        )

    if use_dolthub:
        merged = _merge(_dolthub_earnings(date_), _finnhub_earnings(date_))
        if merged:
            return merged
        logger.info("DoltHub+Finnhub empty — falling back to Investing.com")
        return _investing_earnings(date_)

    if use_finnhub:
        stocks = _finnhub_earnings(date_)
        return stocks if stocks else _investing_earnings(date_)

    # Default: Investing.com primary, Finnhub fallback
    stocks = _investing_earnings(date_)
    return stocks if stocks else _finnhub_earnings(date_)


def scan_dates(
    input_date: Optional[str],
    eastern_tz,
) -> tuple:
    """Compute (post_market_date, pre_market_date) from input or current time."""
    if input_date:
        try:
            post = datetime.strptime(input_date, "%m/%d/%Y").date()
        except ValueError as e:
            raise ValueError("Date must be MM/DD/YYYY") from e
        return post, post + timedelta(days=1)

    now = datetime.now(eastern_tz)
    cutoff = now.replace(hour=16, minute=0, second=0, microsecond=0)
    post = now.date() if now < cutoff else (now + timedelta(days=1)).date()
    return post, post + timedelta(days=1)
