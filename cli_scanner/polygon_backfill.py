#!/usr/bin/env python3
"""
Historical ML backfill using Polygon.io market/options data.

Important distinction:
- Feature/label data is computed from Polygon stock + options bars.
- Earnings event dates are pluggable. Polygon Benzinga earnings is supported,
  but may require an extra entitlement. Finnhub is the default fallback because
  this Polygon key currently returns 403 for /benzinga/v1/earnings.

The output rows are inserted into earnings_edge.db.snapshots so train.py can use
live-collected and backfilled data together.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from scipy.optimize import brentq
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from earnings_edge.config import get_logger, setup_logging
from earnings_edge.db import get_connection, insert_snapshot, update_outcome
from earnings_edge.earnings import _investing_earnings
from earnings_edge.services.outcome_service import OutcomeService

setup_logging()

load_dotenv(Path(__file__).resolve().parent / ".env")
logger = get_logger("polygon_backfill")

POLYGON_BASE = "https://api.polygon.io"
FINNHUB_BASE = "https://finnhub.io/api/v1"


@dataclass(frozen=True)
class EarningsEvent:
    ticker: str
    earnings_date: date
    timing: str = "historical"


class PolygonClient:
    def __init__(self, api_key: str, sleep: float = 12.5):
        self.api_key = api_key
        self.sleep = sleep
        self.session = requests.Session()

    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        p = dict(params or {})
        p["apiKey"] = self.api_key
        for attempt in range(3):
            try:
                r = self.session.get(f"{POLYGON_BASE}{path}", params=p, timeout=20)
                time.sleep(self.sleep)
                if r.status_code == 429:
                    wait = max(65.0, self.sleep * 5)
                    logger.warning("Polygon rate limited on %s; sleeping %.0fs", path, wait)
                    time.sleep(wait)
                    continue
                if r.status_code == 403:
                    logger.warning("Polygon not authorized for %s: %s", path, r.text[:160])
                    return None
                if r.status_code == 404:
                    logger.debug("Polygon 404 for %s", path)
                    return None
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                logger.warning("Polygon request failed %s: %s", path, exc)
                if attempt < 2:
                    time.sleep(max(10.0, self.sleep))
        return None

    def daily_bars(self, ticker: str, start: date, end: date, limit: int = 5000) -> list[dict]:
        data = self.get(
            f"/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}",
            {"adjusted": "true", "sort": "asc", "limit": limit},
        )
        return (data or {}).get("results", []) or []

    def option_contracts(
        self,
        underlying: str,
        as_of: date,
        expiry_gte: date,
        expiry_lte: date,
        contract_type: Optional[str] = None,
    ) -> list[dict]:
        """Fetch option contracts active as-of a historical date."""
        out: list[dict] = []
        params: dict[str, Any] = {
            "underlying_ticker": underlying,
            "as_of": as_of.isoformat(),
            "expiration_date.gte": expiry_gte.isoformat(),
            "expiration_date.lte": expiry_lte.isoformat(),
            "limit": 1000,
            "sort": "expiration_date",
        }
        if contract_type:
            params["contract_type"] = contract_type

        path = "/v3/reference/options/contracts"
        while True:
            data = self.get(path, params)
            if not data:
                return out
            out.extend(data.get("results", []) or [])
            next_url = data.get("next_url")
            if not next_url:
                return out
            # next_url already includes api key sometimes; easiest: call raw URL
            try:
                r = self.session.get(next_url, params={"apiKey": self.api_key}, timeout=20)
                time.sleep(self.sleep)
                r.raise_for_status()
                data = r.json()
                out.extend(data.get("results", []) or [])
                if not data.get("next_url"):
                    return out
                # Continue with next URL manually.
                next_url = data.get("next_url")
                while next_url:
                    r = self.session.get(next_url, params={"apiKey": self.api_key}, timeout=20)
                    time.sleep(self.sleep)
                    r.raise_for_status()
                    data = r.json()
                    out.extend(data.get("results", []) or [])
                    next_url = data.get("next_url")
                return out
            except Exception as exc:
                logger.warning("Polygon contract pagination failed: %s", exc)
                return out

    def option_close(self, option_ticker: str, as_of: date, lookback_days: int = 4) -> Optional[float]:
        """Get most recent option close on/before as_of."""
        start = as_of - timedelta(days=lookback_days)
        bars = self.daily_bars(option_ticker, start, as_of, limit=10)
        bars = [b for b in bars if b.get("c") is not None]
        return float(bars[-1]["c"]) if bars else None


def _dt_from_ms(ms: int) -> date:
    return datetime.fromtimestamp(ms / 1000).date()


def realized_vol_30d(bars: list[dict]) -> Optional[float]:
    closes = [float(b["c"]) for b in bars if b.get("c") and b["c"] > 0]
    if len(closes) < 22:
        return None
    rets = np.diff(np.log(closes[-31:]))
    if len(rets) < 20:
        return None
    return float(np.std(rets, ddof=1) * math.sqrt(252))


def hist_vol(bars: list[dict], days: int = 63) -> Optional[float]:
    closes = [float(b["c"]) for b in bars if b.get("c") and b["c"] > 0]
    if len(closes) < min(days, 20):
        return None
    rets = np.diff(np.log(closes[-days:]))
    if len(rets) < 10:
        return None
    return float(np.std(rets, ddof=1) * math.sqrt(252))


def bs_price(sigma: float, S: float, K: float, T: float, r: float, opt_type: str) -> float:
    if sigma <= 0 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price: float, S: float, K: float, T: float, opt_type: str, r: float = 0.045) -> Optional[float]:
    if not price or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(0.0, S - K) if opt_type == "call" else max(0.0, K - S)
    # Option close can be stale/bad; allow tiny tolerance around intrinsic.
    if price < intrinsic * 0.98:
        return None
    try:
        iv = brentq(lambda sig: bs_price(sig, S, K, T, r, opt_type) - price, 0.01, 5.0, maxiter=100)
        if 0.01 <= iv <= 5.0:
            return float(iv)
    except Exception:
        return None
    return None


def delta(S: float, K: float, T: float, iv: float, opt_type: str, r: float = 0.045) -> Optional[float]:
    if not iv or iv <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    return float(norm.cdf(d1) if opt_type == "call" else norm.cdf(d1) - 1)


def choose_atm_pair(contracts: list[dict], spot: float, expiry: date) -> tuple[Optional[dict], Optional[dict]]:
    same = [c for c in contracts if c.get("expiration_date") == expiry.isoformat()]
    calls = [c for c in same if c.get("contract_type") == "call"]
    puts = [c for c in same if c.get("contract_type") == "put"]
    if not calls or not puts:
        return None, None
    call = min(calls, key=lambda c: abs(float(c.get("strike_price") or 0) - spot))
    put = min(puts, key=lambda c: abs(float(c.get("strike_price") or 0) - spot))
    return call, put


def fetch_events_finnhub(start: date, end: date, tickers: Optional[set[str]] = None) -> list[EarningsEvent]:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY not set; cannot use --events-source finnhub")
    out: list[EarningsEvent] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=30), end)
        r = requests.get(
            f"{FINNHUB_BASE}/calendar/earnings",
            params={"from": cur.isoformat(), "to": chunk_end.isoformat(), "token": key},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json().get("earningsCalendar", []) or []
        for e in data:
            sym = e.get("symbol")
            d = e.get("date")
            if not sym or not d:
                continue
            if tickers and sym not in tickers:
                continue
            timing = e.get("hour") or "historical"
            out.append(EarningsEvent(sym.upper(), datetime.strptime(d, "%Y-%m-%d").date(), timing))
        cur = chunk_end + timedelta(days=1)
        time.sleep(0.25)
    # dedupe
    seen = set()
    deduped = []
    for e in out:
        k = (e.ticker, e.earnings_date)
        if k not in seen:
            seen.add(k)
            deduped.append(e)
    return deduped


def fetch_events_polygon_benzinga(pg: PolygonClient, start: date, end: date, tickers: Optional[set[str]] = None) -> list[EarningsEvent]:
    params: dict[str, Any] = {"date.gte": start.isoformat(), "date.lte": end.isoformat(), "limit": 1000}
    data = pg.get("/benzinga/v1/earnings", params)
    if not data:
        raise RuntimeError("Polygon Benzinga earnings unavailable/not authorized for this API key")
    out = []
    for e in data.get("results", []) or []:
        sym = (e.get("ticker") or e.get("symbol") or "").upper()
        d = e.get("date") or e.get("announce_date")
        if not sym or not d:
            continue
        if tickers and sym not in tickers:
            continue
        out.append(EarningsEvent(sym, datetime.strptime(d[:10], "%Y-%m-%d").date(), e.get("time") or "historical"))
    return out


def collect_polygon_features(pg: PolygonClient, event: EarningsEvent, scan_offset_days: int = 1) -> dict[str, Any]:
    """Collect pre-earnings features from Polygon as of the prior trading day/window."""
    ticker = event.ticker
    ed = event.earnings_date
    as_of = ed - timedelta(days=scan_offset_days)
    errors: list[str] = []
    row: dict[str, Any] = {
        "ticker": ticker,
        "earnings_date": ed.isoformat(),
        "scan_date": as_of.isoformat(),
        "timing": event.timing,
        "collection_error": None,
    }

    # Stock bars before earnings.
    bars = pg.daily_bars(ticker, as_of - timedelta(days=120), as_of)
    if not bars:
        row["collection_error"] = "no stock bars"
        return row
    last = bars[-1]
    price = float(last.get("c") or 0)
    row["price"] = price or None
    vols = [float(b.get("v") or 0) for b in bars[-30:]]
    row["avg_volume_30d"] = float(np.mean(vols)) if vols else None
    row["rv30"] = realized_vol_30d(bars)
    row["hist_vol_3m"] = hist_vol(bars, 63)

    if not price or price <= 0:
        row["collection_error"] = "bad stock price"
        return row

    # Contracts expiring soon after earnings and next monthly-ish expiry.
    contracts = pg.option_contracts(
        ticker,
        as_of=as_of,
        expiry_gte=ed,
        expiry_lte=ed + timedelta(days=70),
    )
    row["has_options"] = 1 if contracts else 0
    if not contracts:
        row["collection_error"] = "no historical option contracts"
        return row

    expiries = sorted({datetime.strptime(c["expiration_date"], "%Y-%m-%d").date() for c in contracts if c.get("expiration_date")})
    if not expiries:
        row["collection_error"] = "contracts missing expiries"
        return row
    near_exp = expiries[0]
    far_exp = expiries[1] if len(expiries) > 1 else None
    row["nearest_expiry"] = near_exp.isoformat()
    row["days_to_expiry"] = (near_exp - as_of).days

    # ATM near call/put.
    near_call, near_put = choose_atm_pair(contracts, price, near_exp)
    if not near_call or not near_put:
        row["collection_error"] = "no ATM near pair"
        return row

    call_px = pg.option_close(near_call["ticker"], as_of)
    put_px = pg.option_close(near_put["ticker"], as_of)
    if call_px is None or put_px is None:
        row["collection_error"] = "no ATM near option prices"
        return row

    T_near = max((near_exp - as_of).days / 365.0, 1 / 365)
    K_call = float(near_call["strike_price"])
    K_put = float(near_put["strike_price"])
    call_iv = implied_vol(call_px, price, K_call, T_near, "call")
    put_iv = implied_vol(put_px, price, K_put, T_near, "put")
    near_iv_vals = [x for x in [call_iv, put_iv] if x is not None]
    near_iv = float(np.mean(near_iv_vals)) if near_iv_vals else None

    row["atm_call_iv"] = call_iv
    row["atm_put_iv"] = put_iv
    row["atm_iv_near"] = near_iv
    row["atm_call_delta"] = delta(price, K_call, T_near, call_iv, "call") if call_iv else None
    row["atm_put_delta"] = delta(price, K_put, T_near, put_iv, "put") if put_iv else None
    row["straddle_price"] = call_px + put_px
    row["expected_move_dollars"] = call_px + put_px
    row["expected_move_pct"] = ((call_px + put_px) / price) * 100
    row["iv30_rv30"] = (near_iv / row["rv30"]) if near_iv and row.get("rv30") else None

    # Far expiry IV and term structure.
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
                if iv:
                    far_ivs.append(iv)
            if far_put_px is not None:
                iv = implied_vol(far_put_px, price, float(far_put["strike_price"]), T_far, "put")
                if iv:
                    far_ivs.append(iv)
            if near_iv and far_ivs:
                far_iv = float(np.mean(far_ivs))
                day_gap = max((far_exp - near_exp).days, 1)
                row["term_slope"] = (far_iv - near_iv) / day_gap
                row["term_structure_valid"] = 1 if row["term_slope"] <= -0.004 else 0

    # We cannot reliably reconstruct historical OI from basic Polygon aggregates.
    row["total_open_interest"] = None
    row["recommendation"] = None
    row["sigma_short_leg"] = near_iv

    # Fair IV for short leg: decompose event premium from term structure.
    # Uses the same variance-decomposition formula as the live analyzer:
    #   σ_fair² = (σ_far² · T_far − σ_baseline² · (T_far − T_near)) / T_near
    # Baseline = min(near_iv, far_iv) — the non-event IV floor.
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
    row["sigma_baseline_1y"] = min(near_iv, far_iv) if near_iv and far_iv else None
    row["sigma_short_leg_fair"] = sigma_short_leg_fair
    row["actual_to_fair_ratio"] = actual_to_fair_ratio

    if errors:
        row["collection_error"] = "; ".join(errors)
    return row


def fetch_events_investing(start: date, end: date, tickers: Optional[set[str]] = None, limit: Optional[int] = None) -> list[EarningsEvent]:
    """Fetch historical earnings events from the Investing.com calendar parser only."""
    out: list[EarningsEvent] = []
    cur = start
    stop = False
    while cur <= end and not stop:
        try:
            # Use Investing.com directly. fetch_earnings() falls back to Finnhub on
            # empty days, which is noisy and quickly hits Finnhub 429s during
            # historical day-by-day scans.
            candidates = _investing_earnings(cur)
            for c in candidates:
                sym = c.ticker.upper()
                # Skip preferred/warrant-ish symbols from Investing.com when a ticker filter is absent.
                if "_" in sym or "." in sym:
                    continue
                if tickers and sym not in tickers:
                    continue
                out.append(EarningsEvent(sym, cur, c.timing or "historical"))
                if limit and len(out) >= limit:
                    stop = True
                    break
        except Exception as exc:
            logger.warning("Investing earnings fetch failed for %s: %s", cur, exc)
        cur += timedelta(days=1)
        time.sleep(0.15)

    seen = set()
    deduped: list[EarningsEvent] = []
    for e in out:
        k = (e.ticker, e.earnings_date)
        if k not in seen:
            seen.add(k)
            deduped.append(e)
    return deduped


def parse_manual_events(values: Optional[list[str]]) -> list[EarningsEvent]:
    """Parse --event TICKER:YYYY-MM-DD[:TIMING] entries for exact smoke tests."""
    events: list[EarningsEvent] = []
    for value in values or []:
        parts = value.split(":", 2)
        if len(parts) < 2:
            raise ValueError(f"Bad --event {value!r}; expected TICKER:YYYY-MM-DD[:TIMING]")
        events.append(EarningsEvent(parts[0].upper(), datetime.strptime(parts[1], "%Y-%m-%d").date(), parts[2] if len(parts) == 3 else "manual"))
    return events


def snapshot_exists(conn, ticker: str, earnings_date: date) -> bool:
    row = conn.execute(
        "SELECT 1 FROM snapshots WHERE ticker = ? AND earnings_date = ? LIMIT 1",
        (ticker.upper(), earnings_date.isoformat()),
    ).fetchone()
    return row is not None


def run_backfill(
    start: date,
    end: date,
    events_source: str,
    tickers: Optional[set[str]],
    limit: Optional[int],
    dry_run: bool,
    rate_sleep: float,
    manual_events: Optional[list[EarningsEvent]] = None,
    insert_errors: bool = False,
) -> int:
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set")
    pg = PolygonClient(key, sleep=rate_sleep)

    if manual_events:
        events = [e for e in manual_events if start <= e.earnings_date <= end and (not tickers or e.ticker in tickers)]
    elif events_source == "investing":
        events = fetch_events_investing(start, end, tickers, limit)
    elif events_source == "finnhub":
        events = fetch_events_finnhub(start, end, tickers)
    elif events_source == "polygon-benzinga":
        events = fetch_events_polygon_benzinga(pg, start, end, tickers)
    else:
        raise ValueError(f"Unknown events source: {events_source}")

    events = sorted(events, key=lambda e: (e.earnings_date, e.ticker))
    if limit and not manual_events and events_source != "investing":
        events = events[:limit]
    logger.info("Backfilling %d earnings events from %s", len(events), events_source)

    conn = get_connection()
    inserted = 0
    for i, event in enumerate(events, start=1):
        logger.info("[%d/%d] %s %s", i, len(events), event.ticker, event.earnings_date)
        try:
            if not dry_run and snapshot_exists(conn, event.ticker, event.earnings_date):
                logger.info("  Skipping existing snapshot for %s %s", event.ticker, event.earnings_date)
                continue
            row = collect_polygon_features(pg, event)
            outcome = OutcomeService().compute_outcome(event.ticker, event.earnings_date.isoformat())
            if dry_run:
                logger.info("  DRY row: price=%s iv=%s em=%s err=%s outcome=%s",
                            row.get("price"), row.get("atm_iv_near"), row.get("expected_move_pct"),
                            row.get("collection_error"), outcome)
                continue
            if row.get("collection_error") and not insert_errors:
                logger.info("  Skipping failed snapshot for %s %s: %s", event.ticker, event.earnings_date, row.get("collection_error"))
                continue
            sid = insert_snapshot(conn, row)
            if outcome:
                update_outcome(conn, sid, outcome)
            inserted += 1
        except Exception as exc:
            logger.exception("  Failed %s %s: %s", event.ticker, event.earnings_date, exc)
        time.sleep(0.05)

    conn.close()
    logger.info("Backfill complete: inserted=%d", inserted)
    return inserted


def parse_tickers(value: Optional[str]) -> Optional[set[str]]:
    if not value:
        return None
    return {x.strip().upper() for x in value.split(",") if x.strip()}


def main() -> None:
    p = argparse.ArgumentParser(description="Historical Polygon options/stock backfill for ML training")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--events-source", choices=["investing", "finnhub", "polygon-benzinga"], default="investing")
    p.add_argument("--tickers", help="Comma-separated ticker filter, e.g. AAPL,MSFT,NVDA")
    p.add_argument("--limit", type=int, help="Max events to process")
    p.add_argument("--event", action="append",
                   help="Exact event to process as TICKER:YYYY-MM-DD[:TIMING]. Can be repeated; avoids calendar scraping.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--insert-errors", action="store_true",
                   help="Insert rows with collection_error. Default skips failed feature rows to keep training data clean.")
    p.add_argument("--rate-sleep", type=float, default=12.5,
                   help="Seconds to sleep after each Polygon request (free tier often needs ~12.5s)")
    args = p.parse_args()

    run_backfill(
        start=datetime.strptime(args.start, "%Y-%m-%d").date(),
        end=datetime.strptime(args.end, "%Y-%m-%d").date(),
        events_source=args.events_source,
        tickers=parse_tickers(args.tickers),
        limit=args.limit,
        dry_run=args.dry_run,
        rate_sleep=args.rate_sleep,
        manual_events=parse_manual_events(args.event),
        insert_errors=args.insert_errors,
    )


if __name__ == "__main__":
    main()
