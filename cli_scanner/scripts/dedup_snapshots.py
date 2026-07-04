#!/usr/bin/env python3
"""Dedup snapshots by (ticker, earnings_date, scan_date, timing, data_source), keeping lowest id."""
import sqlite3
import shutil

db_path = "data/earnings_ml.db"
shutil.copy2(db_path, db_path + ".dedup.bak")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

rows = conn.execute("""
    SELECT MIN(id) AS keep_id, ticker, earnings_date, scan_date, timing, data_source
    FROM snapshots
    GROUP BY ticker, earnings_date, scan_date, timing, data_source
    HAVING COUNT(*) > 1
""").fetchall()
print(f"Duplicate groups to clean: {len(rows)}")

total_deleted = 0
for row in rows:
    keep_id = row["keep_id"]
    ticker, ed, sd, timing, ds = row["ticker"], row["earnings_date"], row["scan_date"], row["timing"], row["data_source"]
    deleted = conn.execute(
        "DELETE FROM snapshots WHERE id > ? AND ticker = ? AND earnings_date = ? AND scan_date = ? AND timing IS ? AND data_source IS ?",
        (keep_id, ticker, ed, sd, timing, ds)
    ).rowcount
    total_deleted += deleted

conn.commit()

# Verify
remaining = conn.execute("""
    SELECT COUNT(*) FROM (
        SELECT ticker, earnings_date, scan_date, timing, data_source
        FROM snapshots
        GROUP BY ticker, earnings_date, scan_date, timing, data_source
        HAVING COUNT(*) > 1
    )
""").fetchone()[0]
conn.close()
print(f"Deleted {total_deleted} duplicate rows")
print(f"Remaining duplicate groups: {remaining}")
