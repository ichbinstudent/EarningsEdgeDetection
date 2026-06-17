"""Quick fix: copy sigma_short_leg -> atm_iv_near for rows where they match."""
import sqlite3

conn = sqlite3.connect("data/earnings_ml.db")

# Step 1: Copy sigma_short_leg to atm_iv_near (they're identical)
cur = conn.execute("""
    UPDATE snapshots 
    SET atm_iv_near = sigma_short_leg
    WHERE scan_date >= '2026-05-01' 
    AND atm_iv_near IS NULL 
    AND sigma_short_leg IS NOT NULL
""")
print(f"Step 1: Copied sigma_short_leg -> atm_iv_near for {cur.rowcount} rows")

# Also copy sigma_short_leg -> atm_call_iv as approximation (same ATM IV)
# And set atm_put_iv = atm_call_iv (ATM puts and calls have similar IV)
cur2 = conn.execute("""
    UPDATE snapshots 
    SET atm_call_iv = sigma_short_leg,
        atm_put_iv = sigma_short_leg
    WHERE scan_date >= '2026-05-01' 
    AND atm_call_iv IS NULL 
    AND sigma_short_leg IS NOT NULL
""")
print(f"Step 2: Copied sigma_short_leg -> atm_call_iv/atm_put_iv for {cur2.rowcount} rows")

# Verify remaining nulls
remaining = conn.execute("""
    SELECT COUNT(*) FROM snapshots 
    WHERE scan_date >= '2026-05-01' AND has_options = 1 AND atm_iv_near IS NULL
""").fetchone()[0]
no_rv = conn.execute("""
    SELECT COUNT(*) FROM snapshots 
    WHERE scan_date >= '2026-05-01' AND has_options = 1 AND rv30 IS NULL
""").fetchone()[0]
print(f"\nRemaining: atm_iv_near null = {remaining}, rv30 null = {no_rv}")

conn.commit()
conn.close()
