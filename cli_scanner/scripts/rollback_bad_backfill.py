#!/usr/bin/env python3
"""Roll back the yfinance-based IV backfill and prepare for Polygon-based re-backfill."""

import sqlite3
from pathlib import Path

DB = Path("data/earnings_ml.db")

conn = sqlite3.connect(DB)

# 1. NULL out all IV fields that were backfilled with yfinance current data
#    (May+ 2026 snapshots that had non-null IV after the bad backfill)
cur = conn.execute("""
    UPDATE snapshots 
    SET atm_iv_near = NULL, atm_call_iv = NULL, atm_put_iv = NULL, 
        rv30 = NULL, hist_vol_3m = NULL
    WHERE scan_date >= '2026-05-01' 
    AND (atm_iv_near IS NOT NULL OR rv30 IS NOT NULL)
""")
print(f"Nulled IV fields on {cur.rowcount} May+ rows")

# 2. Verify rollback
iv_before = conn.execute(
    "SELECT COUNT(*) FROM snapshots WHERE atm_iv_near IS NOT NULL AND scan_date < '2026-05-01'"
).fetchone()[0]
iv_after = conn.execute(
    "SELECT COUNT(*) FROM snapshots WHERE atm_iv_near IS NOT NULL AND scan_date >= '2026-05-01'"
).fetchone()[0]
total_may = conn.execute(
    "SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01'"
).fetchone()[0]
need_backfill = conn.execute(
    "SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL"
).fetchone()[0]

print(f"Pre-May rows with IV: {iv_before}")
print(f"Post-May rows with IV: {iv_after} (should be 0)")
print(f"Total May+ snapshots: {total_may}")
print(f"May+ snapshots needing Polygon backfill: {need_backfill}")

# 3. Calendar trades are fine (those use Polygon historical API)
trades = conn.execute("SELECT COUNT(*) FROM calendar_call_trades").fetchone()[0]
print(f"Calendar trades (untouched): {trades}")

conn.commit()
conn.close()
print("Rollback complete.")
