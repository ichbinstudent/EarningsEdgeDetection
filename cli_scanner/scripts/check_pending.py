import sqlite3
conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "earnings_ml.db"))
conn.row_factory = sqlite3.Row

total = conn.execute("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01'").fetchone()[0]
has_opts = conn.execute("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1").fetchone()[0]
no_opts = total - has_opts
has_error = conn.execute("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND collection_error IS NOT NULL").fetchone()[0]
pending = conn.execute("SELECT COUNT(*) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL").fetchone()[0]
tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM snapshots WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL").fetchone()[0]

print(f"May+ total: {total}")
print(f"  has_options=1: {has_opts}")
print(f"  has_options=0: {no_opts}")
print(f"  has collection_error: {has_error}")
print(f"  pending backfill: {pending}")
print(f"  unique tickers: {tickers}")
print(f"  est time @13s/ticker: {tickers * 13 / 60:.0f} min")
conn.close()
