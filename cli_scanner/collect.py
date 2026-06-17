"""
Daily feature collector — gathers ALL available features for every
earnings candidate without applying any filter gates.

Run once (or twice) daily to accumulate training data for ML.
"""

import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pytz
import yfinance as yf

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from earnings_edge.analyzer import OptionsAnalyzer
from earnings_edge.config import get_logger, session
from earnings_edge.collectors.earnings_calendar import EarningsCalendarCollector
from earnings_edge.earnings import scan_dates
from earnings_edge.models import EarningsCandidate
from earnings_edge.db import get_connection, insert_snapshot

logger = get_logger("collector")


def collect_features(
    ticker: str,
    earnings_date: Optional[date],
) -> dict:
    """
    Collect ALL available features for *ticker*.
    Never raises — returns whatever was collected plus an error field.
    """
    row: dict = {"ticker": ticker, "collection_error": None}
    errors: list[str] = []

    yt = yf.Ticker(ticker, session=session)

    # ── Price ────────────────────────────────────────────────────────
    try:
        hist = yt.history(period="1d")
        if not hist.empty:
            row["price"] = float(hist["Close"].iloc[-1])
        else:
            errors.append("no price data")
    except Exception as exc:
        errors.append(f"price: {exc}")

    # ── Volume + 3-month history ─────────────────────────────────────
    try:
        hist_3m = yt.history(period="3mo")
        if not hist_3m.empty:
            row["avg_volume_30d"] = float(hist_3m["Volume"].rolling(30).mean().dropna().iloc[-1])
    except Exception as exc:
        errors.append(f"volume: {exc}")

    # ── Options availability ─────────────────────────────────────────
    try:
        opts = yt.options
        row["has_options"] = 1 if opts else 0
        if opts:
            row["nearest_expiry"] = opts[0]
            row["days_to_expiry"] = (
                datetime.strptime(opts[0], "%Y-%m-%d").date() - date.today()
            ).days

            # Open interest
            try:
                chain = yt.option_chain(opts[0])
                row["total_open_interest"] = int(
                    chain.calls["openInterest"].sum() + chain.puts["openInterest"].sum()
                )
            except Exception:
                pass
    except Exception as exc:
        errors.append(f"options: {exc}")
        row["has_options"] = 0

    # ── Full options analysis (IVs, term structure, greeks, etc.) ────
    if row.get("has_options"):
        try:
            analyzer = OptionsAnalyzer()
            result = analyzer.compute_recommendation(ticker, earnings_date)
            if result.ok:
                row["iv30_rv30"] = result.iv30_rv30
                row["term_slope"] = result.term_slope
                row["term_structure_valid"] = 1 if result.term_structure_valid else 0
                row["expected_move_pct"] = (
                    float(result.expected_move.strip("%")) if result.expected_move != "N/A" else None
                )
                if row.get("price") and row.get("expected_move_pct"):
                    row["expected_move_dollars"] = row["price"] * row["expected_move_pct"] / 100
                row["recommendation"] = result.recommendation
                row["sigma_baseline_1y"] = result.sigma_baseline_1y
                row["sigma_short_leg"] = result.sigma_short_leg
                row["sigma_short_leg_fair"] = result.sigma_short_leg_fair
                row["actual_to_fair_ratio"] = result.actual_to_fair_ratio
                row["atm_call_delta"] = result.atm_call_delta
                row["atm_put_delta"] = result.atm_put_delta
                row["atm_iv_near"] = result.atm_iv_near
                row["atm_call_iv"] = result.atm_call_iv
                row["atm_put_iv"] = result.atm_put_iv
                row["rv30"] = result.rv30
                row["hist_vol_3m"] = result.hist_vol_3m
            else:
                errors.append(f"analysis: {result.error}")
        except Exception as exc:
            errors.append(f"analysis: {exc}")

    if errors:
        row["collection_error"] = "; ".join(errors)

    return row


def run_collection(scan_date: Optional[str] = None) -> int:
    """
    Run the full daily collection. Returns number of snapshots stored.
    """
    eastern = pytz.timezone("US/Eastern")
    post_date, pre_date = scan_dates(scan_date, eastern)
    conn = get_connection()
    today_str = date.today().isoformat()
    total = 0

    for label, target_date in [("post-market", post_date), ("pre-market", pre_date)]:
        logger.info(f"Fetching {label} earnings for {target_date}")
        collector = EarningsCalendarCollector()
        candidates = collector.fetch(target_date)
        logger.info(f"  {len(candidates)} candidates for {target_date}")

        for i, cand in enumerate(candidates):
            row = collect_features(cand.ticker, target_date)
            row["earnings_date"] = target_date.isoformat()
            row["scan_date"] = today_str
            row["timing"] = cand.timing
            row["data_source"] = cand.source

            insert_snapshot(conn, row)
            total += 1

            if (i + 1) % 25 == 0:
                logger.info(f"  ... {i + 1}/{len(candidates)} processed")
                conn.commit()

            # Rate limit yfinance
            time.sleep(0.3)

        conn.commit()
        logger.info(f"  Stored {len(candidates)} snapshots for {target_date}")

    conn.close()
    logger.info(f"Collection complete: {total} total snapshots")
    return total


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Daily ML feature collector")
    p.add_argument("--date", "-d", help="Override scan date (MM/DD/YYYY)")
    args = p.parse_args()
    run_collection(scan_date=args.date)
