#!/usr/bin/env python3
"""
Backfill rv30 and hist_vol_3m from Polygon historical stock bars.
Only needs 1 API call per unique ticker (daily bars for 120 days).
13s rate limit.
"""
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from polygon_backfill import PolygonClient, realized_vol_30d, hist_vol
from earnings_edge.db import DEFAULT_DB_PATH

load_dotenv(Path(__file__).resolve().parent / ".env")


def main():
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set")

    pg = PolygonClient(key, sleep=13)
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get unique (ticker, scan_date) pairs that need rv30
    rows = conn.execute("""
        SELECT DISTINCT ticker, scan_date, MIN(id) as sample_id
        FROM snapshots
        WHERE scan_date >= '2026-05-01' AND has_options = 1 AND rv30 IS NULL
        GROUP BY ticker, scan_date
        ORDER BY ticker, scan_date
    """).fetchall()

    print(f"Unique (ticker, scan_date) pairs needing rv30: {len(rows)}")
    ok = 0
    failed = 0
    start = time.time()

    for i, row in enumerate(rows, 1):
        ticker = row["ticker"]
        sd = datetime.strptime(row["scan_date"], "%Y-%m-%d").date()
        remaining = (len(rows) - i) * 13 / 60

        print(f"[{i}/{len(rows)}] {ticker} scan={row['scan_date']} ~{remaining:.0f}min left")

        try:
            bars = pg.daily_bars(ticker, sd - timedelta(days=120), sd)
            if not bars:
                print(f"  SKIP: no stock bars")
                failed += 1
                continue

            rv = realized_vol_30d(bars)
            hv = hist_vol(bars, 63)

            if rv is None and hv is None:
                print(f"  SKIP: insufficient data")
                failed += 1
                continue

            # Update ALL snapshots for this ticker+scan_date
            conn.execute("""
                UPDATE snapshots SET rv30 = ?, hist_vol_3m = ?
                WHERE ticker = ? AND scan_date = ? AND scan_date >= '2026-05-01'
            """, (rv, hv, ticker, row["scan_date"]))

            # Also update iv30_rv30 where we now have both
            conn.execute("""
                UPDATE snapshots 
                SET iv30_rv30 = atm_iv_near / rv30
                WHERE ticker = ? AND scan_date = ? AND scan_date >= '2026-05-01'
                AND atm_iv_near IS NOT NULL AND rv30 IS NOT NULL AND rv30 > 0
                AND (iv30_rv30 IS NULL OR iv30_rv30 != atm_iv_near / rv30)
            """, (ticker, row["scan_date"]))

            conn.commit()
            ok += 1
            print(f"  OK: rv30={rv:.4f} hv3m={hv:.4f}" if rv and hv else f"  OK: rv30={rv} hv3m={hv}")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1

    elapsed = (time.time() - start) / 60
    print(f"\nDone in {elapsed:.1f}min: {ok} updated, {failed} failed")
    
    # Summary
    remaining = conn.execute("""
        SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND rv30 IS NULL
    """).fetchone()[0]
    print(f"Remaining with null rv30: {remaining}")
    conn.close()


if __name__ == "__main__":
    main()
