import sqlite3
conn = sqlite3.connect("data/earnings_ml.db")

# What snapshot fields are already populated for May+ rows?
rows = conn.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN price IS NOT NULL AND price > 0 THEN 1 ELSE 0 END) as has_price,
        SUM(CASE WHEN avg_volume_30d IS NOT NULL THEN 1 ELSE 0 END) as has_vol,
        SUM(CASE WHEN market_cap IS NOT NULL THEN 1 ELSE 0 END) as has_mcap,
        SUM(CASE WHEN expected_move_pct IS NOT NULL THEN 1 ELSE 0 END) as has_exp_move,
        SUM(CASE WHEN term_slope IS NOT NULL THEN 1 ELSE 0 END) as has_term,
        SUM(CASE WHEN straddle_price IS NOT NULL THEN 1 ELSE 0 END) as has_straddle
    FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1
""").fetchone()
print("May+ snapshots with has_options=1:")
print(f"  Total: {rows[0]}")
print(f"  Has price: {rows[1]}")
print(f"  Has avg_volume: {rows[2]}")
print(f"  Has market_cap: {rows[3]}")
print(f"  Has expected_move: {rows[4]}")
print(f"  Has term_slope: {rows[5]}")
print(f"  Has straddle_price: {rows[6]}")

# So some snapshots already have expected_move, straddle, term_slope from the live collector
# Those were populated correctly by collect.py using yfinance LIVE data
# Only the IV-specific fields (atm_iv_near, rv30, etc.) are missing

# How many have any IV-adjacent data already?
has_any_iv = conn.execute("""
    SELECT COUNT(*) FROM snapshots 
    WHERE scan_date >= '2026-05-01' AND has_options = 1
    AND (expected_move_pct IS NOT NULL OR straddle_price IS NOT NULL OR term_slope IS NOT NULL)
""").fetchone()[0]
print(f"  Has any IV-adjacent data: {has_any_iv}")

# Check what collect.py actually writes vs what polygon_backfill writes
print("\nSample snapshot (has_options=1, latest):")
sample = conn.execute("""
    SELECT * FROM snapshots 
    WHERE scan_date >= '2026-05-01' AND has_options = 1 AND price > 0
    ORDER BY scan_date DESC LIMIT 1
""").fetchone()
if sample:
    for key in sample.keys():
        print(f"  {key} = {sample[key]}")

conn.close()
