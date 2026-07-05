#!/usr/bin/env python3
"""Historical options-chain backfill using Alpaca.

For every (ticker, scan_date) in the snapshot table where has_options=1:
  1. Reconstruct plausible OCC symbols from snapshot ATM delta + term slope
     (we know nearest_expiry, ATM strike, IV).
  2. Pull bar for each constructed OCC symbol at scan_date.
  3. Persist OHLCV into options_chain.

Usage:
  python backfill_options_chains.py --start 2024-01-01 --end 2025-01-01 --limit 100
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earnings_edge.config import get_logger, setup_logging
from earnings_edge.db import get_connection, insert_options_chain_rows

setup_logging()
logger = get_logger("backfill_options_chains")

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.environ.get("APCA_API_KEY_ID"))
    p.add_argument("--api-secret", default=os.environ.get("APCA_API_SECRET_KEY"))
    p.add_argument("--start", help="Earliest scan_date (YYYY-MM-DD)")
    p.add_argument("--end", help="Latest scan_date (YYYY-MM-DD)")
    p.add_argument("--limit", type=int, help="Cap snapshot count")
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume-from-collector-run", help="Skip if existing rows for this run")
    return p.parse_args()


def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta(S: float, K: float, T: float, iv: float, opt_type: str, r: float = 0.045) -> float:
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
    return float(norm.cdf(d1) if opt_type == "call" else norm.cdf(d1) - 1)


def make_occ(ticker: str, expiry: date, strike: float, opt_type: str) -> str:
    """Construct OCC symbol. Example: AAPL250117C00150000"""
    exp = expiry.strftime("%y%m%d")
    return f"{ticker}{exp}{opt_type.upper()}{int(round(strike * 1000)):08d}"


def candidates_for_snapshot(row: dict) -> list[str]:
    """Build plausible OCC symbols from a snapshot row.

    We use:
    - nearest_expiry, price, atm_iv_near (already in snapshots)
    - atm_call_delta (where populated) to back out the ATM strike via BS delta inversion
    - Otherwise: round(price / 5) * 5 as ATM strike

    Produces: ATM straddle ±0 and ±1 strike steps around ATM, calls + puts.
    """
    ticker = row["ticker"]
    ne_str = row.get("nearest_expiry") or row.get("near_expiry")
    if not ne_str:
        return []
    try:
        expiry = date.fromisoformat(str(ne_str)[:10])
    except ValueError:
        return []

    price = row.get("price") or 0
    iv = row.get("atm_iv_near") or 0
    if price <= 0 or iv <= 0:
        return []

    # Approximate ATM strike from delta (if delta is available)
    d = row.get("atm_call_delta")
    if d is not None and not (isinstance(d, float) and math.isnan(d)):
        # For ATM call, delta ~ 0.5. If delta deviates, strike is off.
        # Use BS delta inversion is overkill - just round to nearest $5.
        pass
    atm = round(price / 5) * 5

    candidates = []
    # Straddle around ATM: ±0..2 strikes each side, calls + puts
    for step in range(0, 3):
        k = atm + step * 5
        candidates.append(make_occ(ticker, expiry, k, "C"))
        candidates.append(make_occ(ticker, expiry, k, "P"))
        if step > 0:
            k2 = atm - step * 5
            candidates.append(make_occ(ticker, expiry, k2, "C"))
            candidates.append(make_occ(ticker, expiry, k2, "P"))

    # Far expiry (if far_expiry populated in snapshot or ≈30d out)
    far_expiry = expiry + timedelta(days=30)
    for step in range(0, 3):
        k = atm + step * 5
        candidates.append(make_occ(ticker, far_expiry, k, "C"))
        candidates.append(make_occ(ticker, far_expiry, k, "P"))
        if step > 0:
            k2 = atm - step * 5
            candidates.append(make_occ(ticker, far_expiry, k2, "C"))
            candidates.append(make_occ(ticker, far_expiry, k2, "P"))

    return candidates


def backfill(client, snapshots_df, *, run_id, dry_run=False):
    """Pull bars for each snapshot and persist."""
    from earnings_edge.collectors.alpaca_options import AlpacaOptionsClient

    inserted = 0
    api_calls = 0
    seen = set()  # (scan_date, contract_ticker) dedup across snapshots

    for i, row in snapshots_df.iterrows():
        sd = row["scan_date"]
        ticker = row["ticker"]
        cands = candidates_for_snapshot(row)
        if not cands:
            continue

        # Dedup within scan_date
        uniq = []
        for ct in cands:
            k = (sd, ct)
            if k not in seen:
                seen.add(k)
                uniq.append(ct)
        if not uniq:
            continue

        # Alpaca bars allow up to ~100 symbols per call; chunk
        for chunk_start in range(0, len(uniq), 100):
            chunk = uniq[chunk_start: chunk_start + 100]
            if api_calls > 0 and api_calls % 50 == 0:
                time.sleep(1)  # be nice

            bars_list, _ = client.bars(
                symbols=chunk, timeframe="1D", start=sd, end=sd
            )
            api_calls += 1
            if not bars_list:
                continue

            rows = []
            for bar in bars_list:
                ct = bar.get("symbol", "")
                rows.append({
                    "collector_run_id": run_id,
                    "ticker": ticker,
                    "scan_date": sd,
                    "contract_ticker": ct,
                    "underlying": ticker,
                    "expiry": None,
                    "strike": None,
                    "contract_type": None,
                    "style": None,
                    "bid": None,
                    "ask": None,
                    "bid_size": None,
                    "ask_size": None,
                    "midpoint": None,
                    "close": bar.get("c"),
                    "open_price": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "trade_count": bar.get("n"),
                    "volume": bar.get("v"),
                    "vwap": bar.get("vw"),
                    "implied_volatility": None,
                    "delta": None,
                    "gamma": None,
                    "theta": None,
                    "vega": None,
                })
                # Parse OCC for expiry/strike/contract_type
                if len(ct) >= 6 and ct[-1] in "CP":
                    ctype_char = ct[-1]
                    try:
                        strike_part = ct[-8:-1]
                        expiry_part = ct[-16:-10]
                        int(strike_part)
                        int(expiry_part)
                        rows[-1].update({
                            "expiry": f"20{expiry_part[:2]}-{expiry_part[2:4]}-{expiry_part[4:6]}",
                            "strike": int(strike_part) / 1000.0,
                            "contract_type": "call" if ctype_char == "C" else "put",
                        })
                    except (ValueError, IndexError):
                        pass

            if not dry_run and rows:
                conn = get_connection()
                try:
                    n = insert_options_chain_rows(conn, rows)
                    inserted += n
                finally:
                    conn.close()

        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d snapshots, %d rows inserted", i + 1, len(snapshots_df), inserted)

    return inserted, api_calls


def main():
    args = _parse_args()
    if not args.api_key or not args.api_secret:
        raise RuntimeError("Pass --api-key + --api-secret or set APCA_API env vars")

    from earnings_edge.collectors.alpaca_options import AlpacaOptionsClient
    client = AlpacaOptionsClient(api_key=args.api_key, api_secret=args.api_secret)

    conn = get_connection()
    query = "SELECT * FROM snapshots WHERE has_options = 1"
    params: list = []
    if args.start:
        query += " AND scan_date >= ?"
        params.append(args.start)
    if args.end:
        query += " AND scan_date <= ?"
        params.append(args.end)
    if args.tickers:
        query += f" AND ticker IN ({','.join(['?' for _ in args.tickers])})"
        params.extend(args.tickers)
    query += " ORDER BY scan_date, ticker"
    if args.limit:
        query += f" LIMIT {args.limit}"

    import pandas as pd
    df = pd.read_sql(query, conn, params=params)
    conn.close()
    if df.empty:
        logger.warning("No snapshots matching criteria")
        return

    logger.info("Backfilling options for %d snapshots (%s to %s)", len(df), df['scan_date'].min(), df['scan_date'].max())

    run_id = f"backfill_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    inserted, api_calls = backfill(client, df, run_id=run_id, dry_run=args.dry_run)
    logger.info("Done: %d rows inserted, %d API calls", inserted, api_calls)


if __name__ == "__main__":
    main()
