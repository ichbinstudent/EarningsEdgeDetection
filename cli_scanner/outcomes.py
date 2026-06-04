"""
Post-earnings outcome tracker — fetches actual stock moves from Polygon.io
and writes them back to the ML database.

Run daily to label historical snapshots with actual outcomes.
"""

import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from earnings_edge.config import get_logger, setup_logging
from earnings_edge.db import get_connection, fetch_pending_outcomes, update_outcome

setup_logging()
logger = get_logger("outcomes")

POLYGON_BASE = "https://api.polygon.io"


RATE_LIMIT_SLEEP = float(os.environ.get("POLYGON_RATE_SLEEP", "1"))

def _polygon_get(path: str, params: dict) -> Optional[dict]:
    """Make a Polygon API GET request with 429 retry. Returns JSON or None."""
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        logger.error("POLYGON_API_KEY not set")
        return None
    params["apiKey"] = api_key
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(f"{POLYGON_BASE}{path}", params=params, timeout=15)
            if resp.status_code == 429:
                wait = RATE_LIMIT_SLEEP * attempt
                logger.warning(f"Polygon 429 (attempt {attempt}/{max_retries}), waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning(f"Polygon API error: {exc}")
            return None
    logger.error(f"Polygon API: all {max_retries} retries exhausted for {path}")
    return None


def get_daily_bars(ticker: str, from_date: str, to_date: str) -> list[dict]:
    """
    Fetch daily OHLCV bars for *ticker* from Polygon.
    Returns list of bar dicts with 'c' (close), 'h' (high), 'l' (low), 't' (timestamp).
    """
    data = _polygon_get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
        {"adjusted": "true", "sort": "asc", "limit": 10},
    )
    if not data or data.get("resultsCount", 0) == 0:
        return []
    return data.get("results", [])


def compute_outcome(ticker: str, earnings_date_str: str) -> Optional[dict]:
    """
    Compute actual earnings-move outcome for *ticker*.

    Looks at:
    - Close the day before earnings → pre_earnings_close
    - Close the day after earnings → post_earnings_close
    - Max intraday range on earnings day or day after

    Returns an outcome dict or None if data unavailable.
    """
    ed = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
    # Fetch a window: 3 days before to 3 days after
    from_d = (ed - timedelta(days=3)).isoformat()
    to_d = (ed + timedelta(days=3)).isoformat()

    bars = get_daily_bars(ticker, from_d, to_d)
    if len(bars) < 2:
        logger.debug(f"{ticker}: insufficient bars ({len(bars)})")
        return None

    # Find the bar ON or just after earnings_date
    ed_ms = int(datetime(ed.year, ed.month, ed.day).timestamp() * 1000)
    pre_bar = None
    post_bar = None
    earnings_bar = None

    for i, bar in enumerate(bars):
        bar_date = datetime.fromtimestamp(bar["t"] / 1000).date()
        if bar_date < ed and i < len(bars) - 1:
            pre_bar = bar
        if bar_date >= ed and post_bar is None:
            post_bar = bar
            earnings_bar = bar

    # If we didn't find a pre-bar, use the bar before the post_bar
    if pre_bar is None and post_bar is not None:
        idx = bars.index(post_bar)
        if idx > 0:
            pre_bar = bars[idx - 1]

    if pre_bar is None or post_bar is None:
        logger.debug(f"{ticker}: can't find pre/post bars")
        return None

    pre_close = pre_bar["c"]
    post_close = post_bar["c"]

    if pre_close <= 0:
        return None

    actual_move_pct = ((post_close - pre_close) / pre_close) * 100
    direction = "UP" if actual_move_pct > 0.5 else "DOWN" if actual_move_pct < -0.5 else "FLAT"

    # Max intraday range across earnings-day and day-after
    max_range_pct = 0.0
    earnings_idx = bars.index(earnings_bar) if earnings_bar else -1
    for j in range(max(0, earnings_idx), min(len(bars), earnings_idx + 3)):
        bar = bars[j]
        if pre_close > 0:
            range_pct = ((bar["h"] - bar["l"]) / pre_close) * 100
            max_range_pct = max(max_range_pct, range_pct)

    return {
        "pre_earnings_close": round(pre_close, 4),
        "post_earnings_close": round(post_close, 4),
        "actual_move_pct": round(actual_move_pct, 4),
        "actual_move_direction": direction,
        "max_intraday_range_pct": round(max_range_pct, 4),
        "outcome_fetched_at": datetime.utcnow().isoformat(),
    }


def run_outcomes(min_age_days: int = 2, limit: int = 0, max_retries: int = 2) -> int:
    """
    Process pending outcomes. Returns count of outcomes written.
    If limit > 0, process at most that many.
    After max_retries consecutive "no data" results for a ticker/earnings_date
    pair, mark it as permanently unavailable.
    """
    conn = get_connection()
    pending = fetch_pending_outcomes(conn, min_age_days=min_age_days)
    logger.info(f"{len(pending)} pending outcomes to fetch"
                f"{f' (processing up to {limit})' if limit else ''}")

    if limit > 0:
        pending = pending[:limit]

    updated = 0
    failed = 0

    for row in pending:
        ticker = row["ticker"]
        ed = row["earnings_date"]
        logger.info(f"  {ticker} ({ed})")

        outcome = compute_outcome(ticker, ed)
        if outcome:
            update_outcome(conn, row["id"], outcome)
            updated += 1
            logger.info(
                f"    → {outcome['actual_move_direction']} "
                f"{outcome['actual_move_pct']:.2f}% "
                f"(max range {outcome['max_intraday_range_pct']:.2f}%)"
            )
        else:
            failed += 1
            # Track how many times this earnings_date has been tried
            attempt_count = conn.execute(
                "SELECT outcome_attempt_count FROM snapshots WHERE id = ?",
                (row["id"],),
            ).fetchone()[0] or 0
            attempt_count += 1

            if attempt_count >= max_retries:
                # Mark as permanently unavailable
                conn.execute(
                    "UPDATE snapshots SET outcome_fetched_at = 'unavailable', "
                    "outcome_attempt_count = ? WHERE id = ?",
                    (attempt_count, row["id"]),
                )
                conn.commit()
                logger.info(f"    → no data (marked unavailable after {attempt_count} attempts)")
            else:
                conn.execute(
                    "UPDATE snapshots SET outcome_attempt_count = ? WHERE id = ?",
                    (attempt_count, row["id"]),
                )
                conn.commit()
                logger.info(f"    → no data (attempt {attempt_count}/{max_retries})")

        time.sleep(RATE_LIMIT_SLEEP)

    conn.close()
    logger.info(f"Outcomes complete: {updated} updated, {failed} no data")
    return updated


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Post-earnings outcome tracker")
    p.add_argument("--min-age", type=int, default=2, help="Min days past earnings to check")
    p.add_argument("--limit", type=int, default=0, help="Max outcomes to process (0=all)")
    p.add_argument("--max-retries", type=int, default=2, help="Attempts before marking unavailable")
    args = p.parse_args()
    run_outcomes(min_age_days=args.min_age, limit=args.limit, max_retries=args.max_retries)
