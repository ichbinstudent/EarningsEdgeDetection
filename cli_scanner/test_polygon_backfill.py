import os, sys
from datetime import date
from dotenv import load_dotenv
load_dotenv(".env")
sys.path.insert(0, ".")
from polygon_backfill import PolygonClient, collect_polygon_features, EarningsEvent

pg = PolygonClient(os.environ["POLYGON_API_KEY"], sleep=13)
event = EarningsEvent("CRM", date(2026, 5, 27))
# Note: collect_polygon_features uses scan_offset_days from earnings_date,
# but we want as_of = 2026-05-26 (the scan_date). 27-1 = 26. Lucky.
features = collect_polygon_features(pg, event, scan_offset_days=1)
print(f"atm_iv_near={features.get('atm_iv_near')}")
print(f"rv30={features.get('rv30')}")
print(f"hist_vol_3m={features.get('hist_vol_3m')}")
print(f"error={features.get('collection_error')}")
print(f"price={features.get('price')}")
