import sqlite3
conn = sqlite3.connect("data/earnings_ml.db")

# Find some well-known tickers with pending backfill
rows = conn.execute("""
    SELECT DISTINCT ticker, earnings_date, scan_date FROM snapshots 
    WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL
    AND ticker IN ('AAPL','MSFT','NVDA','AMZN','META','GOOG','TSLA','AMD','NFLX','CRM')
    ORDER BY earnings_date LIMIT 5
""").fetchall()
for r in rows:
    print(f"{r[0]} ed={r[1]} scan={r[2]}")
if not rows:
    print("No big tech pending, checking volume > 1M...")
    rows2 = conn.execute("""
        SELECT DISTINCT s.ticker, s.earnings_date, s.scan_date FROM snapshots s
        WHERE s.scan_date >= '2026-05-01' AND s.has_options = 1 AND s.collection_error IS NULL AND s.atm_iv_near IS NULL
        AND s.avg_volume_30d > 1000000
        ORDER BY s.earnings_date LIMIT 10
    """).fetchall()
    for r in rows2:
        print(f"{r[0]} ed={r[1]} scan={r[2]}")
    if not rows2:
        # Just show first 10
        rows3 = conn.execute("""
            SELECT DISTINCT ticker, earnings_date, scan_date FROM snapshots 
            WHERE scan_date >= '2026-05-01' AND has_options = 1 AND collection_error IS NULL AND atm_iv_near IS NULL
            ORDER BY earnings_date LIMIT 10
        """).fetchall()
        for r in rows3:
            print(f"{r[0]} ed={r[1]} scan={r[2]}")
conn.close()
