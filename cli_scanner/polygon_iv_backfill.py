#!/usr/bin/env python3
"""
Polygon HISTORICAL IV backfill for May+ snapshots.

Deduplicates by (ticker, scan_date) to minimize API calls.
Uses as_of = snapshot.scan_date with Polygon's historical contracts API.

Polygon API: 13s rate limit between calls.
"""

import argparse
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from polygon_backfill import (
    PolygonClient,
    choose_atm_pair,
    delta,
    hist_vol,
    implied_vol,
    realized_vol_30d,
)
from earnings_edge.config import get_logger, setup_logging
from earnings_edge.db import DEFAULT_DB_PATH

setup_logging()
load_dotenv(Path(__file__).resolve().parent / ".env")
logger = get_logger("polygon_iv_backfill")


def collect_iv_for_snapshot(pg: PolygonClient, ticker: str, earnings_date: date, as_of: date) -> dict[str, Any]:
    """Collect IV features from Polygon as of a specific historical date."""
    bars = pg.daily_bars(ticker, as_of - timedelta(days=120), as_of)
    if not bars:
        return {"collection_error": "no stock bars"}

    last = bars[-1]
    price = float(last.get("c") or 0)
    if not price or price <= 0:
        return {"collection_error": "bad stock price"}

    rv30 = realized_vol_30d(bars)
    hist_vol_3m = hist_vol(bars, 63)

    contracts = pg.option_contracts(
        ticker, as_of=as_of,
        expiry_gte=earnings_date,
        expiry_lte=earnings_date + timedelta(days=70),
    )
    if not contracts:
        return {"collection_error": "no historical option contracts", "rv30": rv30, "hist_vol_3m": hist_vol_3m}

    expiries = sorted({
        datetime.strptime(c["expiration_date"], "%Y-%m-%d").date()
        for c in contracts if c.get("expiration_date")
    })
    if not expiries:
        return {"collection_error": "contracts missing expiries", "rv30": rv30, "hist_vol_3m": hist_vol_3m}

    near_exp = expiries[0]
    far_exp = expiries[1] if len(expiries) > 1 else None

    near_call, near_put = choose_atm_pair(contracts, price, near_exp)
    if not near_call or not near_put:
        return {"collection_error": "no ATM near pair", "rv30": rv30, "hist_vol_3m": hist_vol_3m}

    call_px = pg.option_close(near_call["ticker"], as_of)
    put_px = pg.option_close(near_put["ticker"], as_of)
    if call_px is None or put_px is None:
        return {"collection_error": "no ATM near option prices", "rv30": rv30, "hist_vol_3m": hist_vol_3m}

    T_near = max((near_exp - as_of).days / 365.0, 1 / 365)
    K_call = float(near_call["strike_price"])
    K_put = float(near_put["strike_price"])
    call_iv = implied_vol(call_px, price, K_call, T_near, "call")
    put_iv = implied_vol(put_px, price, K_put, T_near, "put")
    near_iv_vals = [x for x in [call_iv, put_iv] if x is not None]
    near_iv = float(np.mean(near_iv_vals)) if near_iv_vals else None

    result: dict[str, Any] = {
        "atm_call_iv": call_iv, "atm_put_iv": put_iv, "atm_iv_near": near_iv,
        "atm_call_delta": delta(price, K_call, T_near, call_iv, "call") if call_iv else None,
        "atm_put_delta": delta(price, K_put, T_near, put_iv, "put") if put_iv else None,
        "straddle_price": call_px + put_px,
        "expected_move_dollars": call_px + put_px,
        "expected_move_pct": ((call_px + put_px) / price) * 100,
        "rv30": rv30, "hist_vol_3m": hist_vol_3m,
        "iv30_rv30": (near_iv / rv30) if near_iv and rv30 else None,
    }

    far_iv = None
    if far_exp:
        far_call, far_put = choose_atm_pair(contracts, price, far_exp)
        if far_call and far_put:
            far_call_px = pg.option_close(far_call["ticker"], as_of)
            far_put_px = pg.option_close(far_put["ticker"], as_of)
            T_far = max((far_exp - as_of).days / 365.0, 1 / 365)
            far_ivs = []
            if far_call_px is not None:
                iv = implied_vol(far_call_px, price, float(far_call["strike_price"]), T_far, "call")
                if iv: far_ivs.append(iv)
            if far_put_px is not None:
                iv = implied_vol(far_put_px, price, float(far_put["strike_price"]), T_far, "put")
                if iv: far_ivs.append(iv)
            if near_iv and far_ivs:
                far_iv = float(np.mean(far_ivs))
                day_gap = max((far_exp - near_exp).days, 1)
                result["term_slope"] = (far_iv - near_iv) / day_gap
                result["term_structure_valid"] = 1 if result["term_slope"] <= -0.004 else 0

    sigma_short_leg_fair = None
    actual_to_fair_ratio = None
    if near_iv and far_iv and far_exp:
        T_short_days = max((near_exp - as_of).days, 1)
        T_long_days = max((far_exp - as_of).days, T_short_days + 1)
        baseline_iv = min(near_iv, far_iv)
        radicand = (far_iv ** 2 * T_long_days - baseline_iv ** 2 * (T_long_days - T_short_days)) / T_short_days
        if radicand > 0:
            sigma_short_leg_fair = float(np.sqrt(radicand))
            if sigma_short_leg_fair > 0:
                actual_to_fair_ratio = float((near_iv / sigma_short_leg_fair - 1) * 100)

    result["sigma_baseline_1y"] = min(near_iv, far_iv) if near_iv and far_iv else None
    result["sigma_short_leg"] = near_iv
    result["sigma_short_leg_fair"] = sigma_short_leg_fair
    result["actual_to_fair_ratio"] = actual_to_fair_ratio
    return result


def get_pending_groups(conn: sqlite3.Connection, limit: int = 0) -> list[dict]:
    """Get unique (ticker, earnings_date, scan_date) groups with their snapshot IDs."""
    sql = """
        SELECT ticker, earnings_date, scan_date, GROUP_CONCAT(id) as ids
        FROM snapshots
        WHERE scan_date >= '2026-05-01'
          AND has_options = 1
          AND atm_iv_near IS NULL
        GROUP BY ticker, earnings_date, scan_date
        ORDER BY earnings_date, ticker
    """
    if limit:
        sql += f" LIMIT {limit}"
    rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def update_snapshots_iv(conn: sqlite3.Connection, snapshot_ids: list[int], features: dict) -> None:
    fields = [
        "atm_iv_near", "atm_call_iv", "atm_put_iv",
        "rv30", "hist_vol_3m", "iv30_rv30",
        "term_slope", "term_structure_valid",
        "expected_move_pct", "expected_move_dollars",
        "straddle_price", "atm_call_delta", "atm_put_delta",
        "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair",
        "actual_to_fair_ratio",
    ]
    set_clause = ", ".join(f"{f} = ?" for f in fields)
    values = [features.get(f) for f in fields]
    placeholders = ",".join("?" * len(snapshot_ids))
    conn.execute(
        f"UPDATE snapshots SET {set_clause} WHERE id IN ({placeholders})",
        values + snapshot_ids,
    )


def main():
    parser = argparse.ArgumentParser(description="Polygon historical IV backfill")
    parser.add_argument("--limit", type=int, default=0, help="Max groups (0=all)")
    parser.add_argument("--rate-sleep", type=float, default=13.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set")

    pg = PolygonClient(key, sleep=args.rate_sleep)
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row

    groups = get_pending_groups(conn, args.limit)
    total_rows = sum(len(g["ids"].split(",")) for g in groups)
    print(f"Groups to backfill: {len(groups)} ({total_rows} snapshot rows)")

    ok = 0
    failed = 0
    skipped = 0
    rows_updated = 0
    start_time = time.time()

    for i, group in enumerate(groups, 1):
        ticker = group["ticker"]
        ed = datetime.strptime(group["earnings_date"], "%Y-%m-%d").date()
        sd = datetime.strptime(group["scan_date"], "%Y-%m-%d").date()
        sids = [int(x) for x in group["ids"].split(",")]

        elapsed_min = (time.time() - start_time) / 60
        remaining = (len(groups) - i) * args.rate_sleep * 5 / 60  # ~5 calls per group

        print(f"[{i}/{len(groups)}] {ticker} ed={group['earnings_date']} scan={group['scan_date']} "
              f"({len(sids)} rows) ok={ok} fail={failed} skip={skipped} ~{remaining:.0f}min left")

        if args.dry_run:
            skipped += 1
            continue

        try:
            features = collect_iv_for_snapshot(pg, ticker, ed, as_of=sd)

            if features.get("atm_iv_near") is None:
                err = features.get("collection_error", "no IV")
                print(f"  SKIP: {err}")
                skipped += 1
                placeholders = ",".join("?" * len(sids))
                conn.execute(
                    f"UPDATE snapshots SET collection_error = ? WHERE id IN ({placeholders})",
                    [f"polygon_backfill: {err}"] + sids,
                )
                conn.commit()
                # Still update rv30/hist_vol if we got them
                if features.get("rv30") or features.get("hist_vol_3m"):
                    partial_fields = []
                    partial_values = []
                    for f in ["rv30", "hist_vol_3m"]:
                        if features.get(f) is not None:
                            partial_fields.append(f"{f} = ?")
                            partial_values.append(features[f])
                    if partial_fields:
                        ph = ",".join("?" * len(sids))
                        conn.execute(
                            f"UPDATE snapshots SET {', '.join(partial_fields)} WHERE id IN ({ph})",
                            partial_values + sids,
                        )
                        conn.commit()
                continue

            update_snapshots_iv(conn, sids, features)
            conn.commit()
            ok += 1
            rows_updated += len(sids)
            print(f"  OK: iv={features['atm_iv_near']:.4f} rv={features.get('rv30', 0):.4f} -> {len(sids)} rows")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1
            logger.error("Backfill failed for %s: %s", ticker, exc)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f}min: {ok} groups ({rows_updated} rows) updated, {failed} failed, {skipped} skipped")
    conn.close()


if __name__ == "__main__":
    main()
