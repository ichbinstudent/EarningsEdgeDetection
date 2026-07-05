#!/usr/bin/env python3
"""Live daily options-chain collector via Alpaca.

At market close (intended to be scheduled), call
/v1beta1/options/snapshots/{underlying} for every active underlying and
persist every row into options_chain.

Rate limit: Alpaca data tier allows ~5 req/sec sustained.
One call per underlying; ~15 req/s burst then 1 token per second refill.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earnings_edge.config import get_logger, setup_logging
from earnings_edge.db import get_connection, insert_options_chain_rows

setup_logging()
logger = get_logger("collect_options_snapshot")

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _parse_args():
    p = argparse.ArgumentParser(description="Live options-chain collector")
    p.add_argument("--api-key", default=os.environ.get("APCA_API_KEY_ID"))
    p.add_argument("--api-secret", default=os.environ.get("APCA_API_SECRET_KEY"))
    p.add_argument("--underlyings", nargs="*", default=[],
                   help="Underlying tickers; default from DB.")
    p.add_argument("--max-tickers", type=int, default=200,
                   help="Cap tickers to avoid long runs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Pull only; don't persist.")
    return p.parse_args()


def _row_for_contract(run_id: str, underlying: str, contract_ticker: str,
                       snap: dict) -> dict:
    """Convert one Alpaca chain-snapshot record into a dB-ready dict."""
    bar = snap.get("dailyBar") or {}
    q = snap.get("latestQuote") or {}
    bid = q.get("bp")
    ask = q.get("ap")
    midpoint = ((bid + ask) / 2) if (bid is not None and ask is not None) else None
    close = bar.get("c")

    # Parse OCC symbol for expiry, strike, contract_type
    expiry_str = ""
    strike_val = None
    contract_type = ""
    style = "american"
    # Example: AAPL250117C00150000
    if len(contract_ticker) >= 6 and contract_ticker[-1] in "CP":
        ctype_char = contract_ticker[-1]
        contract_type = "call" if ctype_char == "C" else "put"
        try:
            strike_part = contract_ticker[-8:-1]
            expiry_part = contract_ticker[-16:-10]
            int(strike_part)
            int(expiry_part)
            expiry_str = f"20{expiry_part[:2]}-{expiry_part[2:4]}-{expiry_part[4:6]}"
            strike_val = int(strike_part) / 1000.0
        except (ValueError, IndexError):
            pass  # leave expiry/strike as-is on parse failure

    return {
        "collector_run_id": run_id,
        "ticker": underlying,
        "scan_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "contract_ticker": contract_ticker,
        "underlying": underlying,
        "expiry": expiry_str,
        "strike": strike_val,
        "contract_type": contract_type,
        "style": style,
        "bid": bid,
        "ask": ask,
        "bid_size": q.get("bs"),
        "ask_size": q.get("as"),
        "midpoint": midpoint,
        "close": close,
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
    }


def collect(client, underlyings, *, run_id, dry_run=False):
    """Pull chain for each underlying, persist rows. Returns (inserted, api_calls)."""
    from earnings_edge.collectors.alpaca_options import AlpacaOptionsClient

    inserted = 0
    api_calls = 0
    for i, und in enumerate(underlyings):
        if i > 0:
            time.sleep(0.22)  # stay under ~5 req/sec sustained

        snap, _ = client.chain_snapshot(und)
        api_calls += 1
        contracts = snap.get("snapshots", {})
        if not contracts:
            logger.debug("No contracts for %s", und)
            continue

        rows = []
        for ct, s in contracts.items():
            rows.append(_row_for_contract(run_id, und, ct, s))

        if not dry_run and rows:
            conn = get_connection()
            try:
                n = insert_options_chain_rows(conn, rows)
                inserted += n
                logger.info("%s: inserted %d rows (out of %d)", und, n, len(rows))
            finally:
                conn.close()
        else:
            logger.info("%s: %d rows (dry-run: %s)", und, len(rows), dry_run)

    return inserted, api_calls


def main():
    args = _parse_args()
    if not args.api_key or not args.api_secret:
        raise RuntimeError("Must pass --api-key + --api-secret or set env APCA_API_KEY+SECRET")

    from earnings_edge.collectors.alpaca_options import AlpacaOptionsClient
    client = AlpacaOptionsClient(api_key=args.api_key, api_secret=args.api_secret)

    underlyings = args.underlyings
    if not underlyings:
        # Default: distinct tickers from snapshots that had options in last 30d
        conn = get_connection()
        r = conn.execute(
            "SELECT DISTINCT ticker FROM snapshots "
            "WHERE has_options = 1 "
              "AND outcome_fetched_at IS NOT NULL "
            "ORDER BY rowid DESC LIMIT ?", (args.max_tickers,)
        )
        underlyings = [row[0] for row in r.fetchall()]
        conn.close()

    underlyings = underlyings[: args.max_tickers]
    if not underlyings:
        logger.warning("No underlyings to collect")
        return

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger.info("Starting options-chain collection run=%s (%d underlyings)", run_id, len(underlyings))

    inserted, api_calls = collect(
        client, underlyings, run_id=run_id, dry_run=args.dry_run
    )
    logger.info(
        "Run %s complete: %d rows inserted, %d API calls", run_id, inserted, api_calls
    )


if __name__ == "__main__":
    main()
