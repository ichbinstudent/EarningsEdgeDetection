#!/usr/bin/env python3
"""Backfill missing IV fields for snapshots that have options but no atm_iv_near."""

import sqlite3
import sys
import time
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from earnings_edge.analyzer import OptionsAnalyzer
from earnings_edge.db import get_connection

FIELDS_TO_UPDATE = [
    "atm_iv_near", "atm_call_iv", "atm_put_iv",
    "rv30", "hist_vol_3m", "iv30_rv30",
    "term_slope", "term_structure_valid",
    "expected_move_pct", "expected_move_dollars",
    "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair",
    "actual_to_fair_ratio", "atm_call_delta", "atm_put_delta",
]


def main():
    conn = get_connection()
    c = conn.cursor()

    # Get snapshots needing backfill
    c.execute("""
        SELECT id, ticker, earnings_date, price
        FROM snapshots
        WHERE has_options = 1 AND atm_iv_near IS NULL
        ORDER BY ticker, earnings_date
    """)
    rows = c.fetchall()
    print(f"{len(rows)} snapshots to backfill")

    # Group by ticker to minimize yfinance calls
    from collections import defaultdict
    by_ticker = defaultdict(list)
    for row in rows:
        by_ticker[row["ticker"]].append(row)

    print(f"{len(by_ticker)} unique tickers")
    analyzer = OptionsAnalyzer()
    updated = 0
    failed = 0

    for i, (ticker, ticker_rows) in enumerate(by_ticker.items()):
        # Use the earliest earnings_date for the analyzer call
        earnings_dates = {r["earnings_date"] for r in ticker_rows}
        # Run analyzer once per ticker with the most recent earnings date
        # The IV data is the same regardless of earnings date (it's current options data)
        latest_date = max(earnings_dates)

        try:
            result = analyzer.compute_recommendation(ticker, latest_date)
            if not result.ok:
                print(f"  [{i+1}/{len(by_ticker)}] {ticker}: {result.error}")
                failed += len(ticker_rows)
                continue

            # Update all rows for this ticker
            sets = []
            params = []
            for field in FIELDS_TO_UPDATE:
                val = getattr(result, field, None)
                if val is not None:
                    sets.append(f"{field} = ?")
                    params.append(val)

            if not sets:
                print(f"  [{i+1}/{len(by_ticker)}] {ticker}: no fields to update")
                failed += len(ticker_rows)
                continue

            for row in ticker_rows:
                p = params + [row["id"]]
                c.execute(f"UPDATE snapshots SET {', '.join(sets)} WHERE id = ?", p)

            conn.commit()
            updated += len(ticker_rows)
            print(f"  [{i+1}/{len(by_ticker)}] {ticker}: updated {len(ticker_rows)} rows (atm_iv={result.atm_iv_near:.3f})")

        except Exception as e:
            print(f"  [{i+1}/{len(by_ticker)}] {ticker}: EXCEPTION {e}")
            failed += len(ticker_rows)

        # Rate limit
        time.sleep(0.3)

    conn.close()
    print(f"\nDone: {updated} updated, {failed} failed")


if __name__ == "__main__":
    main()
