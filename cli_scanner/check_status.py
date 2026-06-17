import sqlite3
conn = sqlite3.connect("data/earnings_ml.db")
total = conn.execute("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1").fetchone()[0]
iv_null = conn.execute("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND atm_iv_near IS NULL").fetchone()[0]
rv_null = conn.execute("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND rv30 IS NULL").fetchone()[0]
both_null = conn.execute("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND atm_iv_near IS NULL AND rv30 IS NULL").fetchone()[0]
print(f"Total has_options=1: {total}")
print(f"IV null: {iv_null}")
print(f"RV null: {rv_null}")
print(f"Both null: {both_null}")
conn.close()
