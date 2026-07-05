#!/usr/bin/env python3
"""Multi-strike positional options strategies using real Alpaca chain data.

Fall back to Black-Scholes synthetic pricing (from snapshot ATM IV) when
options_chain table does not contain real bid/ask for a given contract.

Strategies:
- IronCondorReal  — sell OTM strangle at ~15-delta short strikes, buy OTM wings
- ButterflyReal    — long butterfly expecting earnings crush within body
- RiskReversalReal — buy OTM call, sell OTM put on skew mispricing
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from earnings_edge.strategies import Trade, StrategyResult, DataBundle
from earnings_edge.collectors.alpaca_options import AlpacaOptionsClient

# ---------------------------------------------------------------------------
# Black-Scholes helpers (same math as multi_strike.py)
# ---------------------------------------------------------------------------

def bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "call":
        return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def delta(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0 or math.isnan(sigma):
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d1) if opt_type == "call" else norm.cdf(d1) - 1)


def strike_for_delta(underlying: float, T: float, r: float, sigma: float, target_delta: float) -> float:
    """Find strike where |delta|=target_delta via Newton-Raphon."""
    # Approx: for ~15-delta call, K ≈ S + sigma * sqrt(T) * 1.04
    K = underlying * (1 + math.sqrt(T) * sigma * 1.04) if target_delta > 0 else underlying * (1 - math.sqrt(T) * sigma * 1.04)
    for _ in range(50):
        p = delta(underlying, K, T, r, sigma, "call" if target_delta > 0 else "put")
        if math.isnan(p):
            return K
        diff = abs(abs(p) - abs(target_delta))
        if diff < 0.001:
            break
        # Derivative of delta w.r.t. strike = -pdf/(K*sigma*sqrt(T))
        pdf = norm.pdf((math.log(underlying/K) + (r + 0.5*sigma**2)*T)/(sigma*math.sqrt(T)))
        deriv = -pdf / (K * sigma * math.sqrt(T)) if K > 0 else 0
        if abs(deriv) < 1e-9:
            break
        K -= (abs(p) - abs(target_delta)) / abs(deriv) * (-1 if target_delta > 0 else 1)
    return K


def nearest_strike(price: float, width: int = 5) -> float:
    return round(price / width) * width


def pull_bid_ask(chain_df: pd.DataFrame, contract_ticker: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (bid, ask, midpoint) for this contract ticker in the chain snapshot."""
    if chain_df.empty:
        return None, None, None
    row = chain_df[chain_df["contract_ticker"] == contract_ticker]
    if row.empty:
        return None, None, None
    bid = row.iloc[0].get("bid")
    ask = row.iloc[0].get("ask")
    bid = float(bid) if bid is not None else None
    ask = float(ask) if ask is not None else None
    midpoint = ((bid + ask) / 2) if (bid is not None and ask is not None) else None
    return bid, ask, midpoint


class AlpacaMultiStrike:
    """Provide bid/ask or BS fallback for a given (ticker, scan_date)."""

    def __init__(self, bundle: DataBundle, r: float = 0.045):
        self.bundle = bundle
        self.r = r

    def chain_for(self, ticker: str, scan_date: str) -> pd.DataFrame:
        df = self.bundle.options_chain
        if df.empty:
            return pd.DataFrame()
        return df[(df["ticker"] == ticker) & (df["scan_date"] == scan_date)]

    def bid_ask_or_bs(self, chain: pd.DataFrame, contract_ticker: str,
                       S: float, K: float, T: float, sigma: float, opt_type: str) -> tuple[float, float]:
        """Return (bid, ask) for the contract — either real or BS-synthesized."""
        r, a, m = pull_bid_ask(chain, contract_ticker)
        if m is not None and r is not None:
            return float(r), float(a)
        # Fallback: BS price ± 10% spread around mid
        mid = bs_price(S, K, T, self.r, sigma, opt_type)
        return mid * 0.9, mid * 1.1


# ---------------------------------------------------------------------------
# Iron Condor at 15-delta short strikes
# ---------------------------------------------------------------------------

class IronCondorReal:
    """Sell iron condor at ~15-delta short strikes.

    Short: 15-delta call and 15-delta put (=OTM wings)
    Long:  25-delta call and 25-delta put (far OTM wings)

    Credit received when stock stays between short strikes.
    Max loss when stock exceeds long strikes.
    """
    name = "iron_condor_real"

    def __init__(self, iv_rv_min: float = 1.2, min_expected_move: float = 6.0,
                 min_days_to_expiry: int = 3, max_days_to_expiry: int = 14,
                 risk_reward_min: float = 0.20):
        self.iv_rv_min = iv_rv_min
        self.min_expected_move = min_expected_move
        self.min_dte = min_days_to_expiry
        self.max_dte = max_days_to_expiry
        self.risk_reward_min = risk_reward_min

    def run(self, bundle: DataBundle) -> StrategyResult:
        trades = []
        snap = bundle.snapshots
        if snap.empty:
            return StrategyResult(self.name, [])

        mask = (
            snap["actual_move_pct"].notna()
            & snap["expected_move_pct"].notna()
            & snap["iv30_rv30"].notna()
            & (snap["iv30_rv30"] >= self.iv_rv_min)
            & (snap["expected_move_pct"] >= self.min_expected_move)
            & snap["price"].notna()
            & snap["atm_iv_near"].notna()
            & snap["nearest_expiry"].notna()
        )
        filtered = snap.loc[mask].copy()
        if filtered.empty:
            return StrategyResult(self.name, [])

        api = AlpacaMultiStrike(bundle)

        for _, row in filtered.iterrows():
            ticker = row["ticker"]
            scan_date = row["scan_date"]
            S = float(row["price"])
            iv = float(row["atm_iv_near"])
            try:
                expiry = date.fromisoformat(str(row["nearest_expiry"])[:10])
            except Exception:
                continue
            scan = date.fromisoformat(str(scan_date)[:10]) if scan_date else expiry - timedelta(days=7)
            T = max((expiry - scan).days / 365.0, 1/365)
            dte = int((expiry - scan).days)
            if dte < self.min_dte or dte > self.max_dte:
                continue

            chain = api.chain_for(ticker, str(scan_date))
            r = 0.045

            # Target ~15-delta short strikes, ~8-delta long wings
            sc_call = strike_for_delta(S, T, r, iv, 0.15)
            sp_put = strike_for_delta(S, T, r, iv, -0.15)
            lc_call = strike_for_delta(S, T, r, iv, 0.08)
            lp_put = strike_for_delta(S, T, r, iv, -0.08)

            # Snap to nearest 5
            sc_call_k = nearest_strike(sc_call)
            sp_put_k = nearest_strike(sp_put)
            lc_call_k = nearest_strike(lc_call)
            lp_put_k = nearest_strike(lp_put)

            # Get bid/ask or BS fallback
            sc_call_bid, sc_call_ask = api.bid_ask_or_bs(chain, f"{ticker}{expiry.strftime('%y%m%d')}C{int(sc_call_k*1000):08d}", S, sc_call_k, T, iv, "call")
            sp_put_bid, sp_put_ask = api.bid_ask_or_bs(chain, f"{ticker}{expiry.strftime('%y%m%d')}P{int(sp_put_k*1000):08d}", S, sp_put_k, T, iv, "put")
            lc_call_bid, lc_call_ask = api.bid_ask_or_bs(chain, f"{ticker}{expiry.strftime('%y%m%d')}C{int(lc_call_k*1000):08d}", S, lc_call_k, T, iv, "call")
            lp_put_bid, lp_put_ask = api.bid_ask_or_bs(chain, f"{ticker}{expiry.strftime('%y%m%d')}P{int(lp_put_k*1000):08d}", S, lp_put_k, T, iv, "put")

            # Credit received: sell OTM strangle, buy further OTM
            credit = sc_call_bid + sp_put_bid - lc_call_ask - lp_put_ask
            max_loss = max(sc_call_k - sp_put_k, lc_call_k - lp_put_k)  # wing width
            risk_reward = credit / max_loss if max_loss > 0 else 0

            if risk_reward < self.risk_reward_min:
                continue

            actual_move = float(row["actual_move_pct"])
            underlying_at_expiry = S + S * actual_move / 100.0

            # PnL
            call_loss = max(0.0, underlying_at_expiry - sc_call_k) - max(0.0, underlying_at_expiry - lc_call_k)
            put_loss = max(0.0, sp_put_k - underlying_at_expiry) - max(0.0, lp_put_k - underlying_at_expiry)
            pnl_percent = credit - call_loss - put_loss

            trades.append(Trade(
                ticker=ticker,
                earnings_date=row["earnings_date"] if isinstance(row["earnings_date"], date) else date.fromisoformat(str(row["earnings_date"])),
                scan_date=row["scan_date"] if isinstance(row["scan_date"], date) else date.fromisoformat(str(row["scan_date"])),
                strategy=self.name,
                side="IRON_CONDOR",
                entry_price=credit,
                exit_price=call_loss + put_loss,
                pnl=pnl_percent,
                pnl_pct=pnl_percent,
                features={"iv_rv": row.get("iv30_rv30"), "expected_move_pct": row.get("expected_move_pct"),
                          "risk_reward": risk_reward, "short_call": sc_call_k, "short_put": sp_put_k,
                          "long_call": lc_call_k, "long_put": lp_put_k, "dte": dte},
                ml_decision="TAKE",
                notes=f"sc={sc_call_k} sp={sp_put_k} lc={lc_call_k} lp={lp_put_k} rr={risk_reward:.2f} credit={credit:.2f} pnl={pnl_percent:.2f}"
            ))

        if not trades:
            return StrategyResult(self.name, [])

        taken = trades
        total_pnl = sum(t.pnl for t in taken)
        win_rate = sum(1 for t in taken if t.pnl > 0) / len(taken)
        return StrategyResult(self.name, trades, {
            "total": len(trades),
            "taken": len(taken),
            "avg_pnl": float(np.mean([t.pnl for t in taken])),
            "total_pnl": float(round(total_pnl, 4)),
            "win_rate": win_rate,
            "risk_reward_min": self.risk_reward_min,
        })


# ---------------------------------------------------------------------------
# Long Butterfly (earnings crush — stock stays near ATM)
# ---------------------------------------------------------------------------

class ButterflyReal:
    """Long butterfly: buy lower, sell 2× ATM, buy upper.

    Max payoff when stock = ATM at expiry; loss = net premium paid.
    """
    name = "butterfly_real"

    def __init__(self, iv_rv_min: float = 1.15, min_expected_move: float = 6.0,
                 width: int = 5):
        self.iv_rv_min = iv_rv_min
        self.min_expected_move = min_expected_move
        self.width = width

    def run(self, bundle: DataBundle) -> StrategyResult:
        trades = []
        snap = bundle.snapshots
        if snap.empty:
            return StrategyResult(self.name, [])

        mask = (
            snap["actual_move_pct"].notna()
            & snap["expected_move_pct"].notna()
            & snap["iv30_rv30"].notna()
            & (snap["iv30_rv30"] >= self.iv_rv_min)
            & (snap["expected_move_pct"] >= self.min_expected_move)
            & snap["price"].notna()
            & snap["atm_iv_near"].notna()
            & snap["nearest_expiry"].notna()
        )
        filtered = snap.loc[mask].copy()
        if filtered.empty:
            return StrategyResult(self.name, [])

        api = AlpacaMultiStrike(bundle)

        for _, row in filtered.iterrows():
            ticker = row["ticker"]
            scan_date = row["scan_date"]
            S = float(row["price"])
            iv = float(row["atm_iv_near"])
            try:
                expiry = date.fromisoformat(str(row["nearest_expiry"])[:10])
            except Exception:
                continue
            scan = date.fromisoformat(str(scan_date)[:10]) if scan_date else expiry - timedelta(days=7)
            T = max((expiry - scan).days / 365.0, 1/365)
            r = 0.045

            chain = api.chain_for(ticker, str(scan_date))
            atm = nearest_strike(S, self.width)
            lo = atm - self.width
            hi = atm + self.width

            lo_bid, lo_ask = api.bid_ask_or_bs(chain, f"{ticker}{expiry.strftime('%y%m%d')}C{int(lo*1000):08d}", S, lo, T, iv, "call")
            atm_bid, atm_ask = api.bid_ask_or_bs(chain, f"{ticker}{expiry.strftime('%y%m%d')}C{int(atm*1000):08d}", S, atm, T, iv, "call")
            hi_bid, hi_ask = api.bid_ask_or_bs(chain, f"{ticker}{expiry.strftime('%y%m%d')}C{int(hi*1000):08d}", S, hi, T, iv, "call")

            # Buy lower + upper, sell 2× ATM
            net_premium = lo_ask + hi_ask - 2 * atm_bid

            if net_premium <= 0:
                continue  # degenerate, skip

            actual_move = float(row["actual_move_pct"])
            underlying_at_expiry = S + S * actual_move / 100.0

            # PnL: payoff at expiry minus net premium
            lower_payoff = max(0.0, underlying_at_expiry - lo)
            atm_payoff = 2 * max(0.0, underlying_at_expiry - atm)
            upper_payoff = max(0.0, underlying_at_expiry - hi)
            pnl_percent = lower_payoff - atm_payoff + upper_payoff - net_premium

            trades.append(Trade(
                ticker=ticker,
                earnings_date=row["earnings_date"] if isinstance(row["earnings_date"], date) else date.fromisoformat(str(row["earnings_date"])),
                scan_date=row["scan_date"] if isinstance(row["scan_date"], date) else date.fromisoformat(str(row["scan_date"])),
                strategy=self.name,
                side="BUTTERFLY",
                entry_price=net_premium,
                exit_price=max(0, pnl_percent + net_premium),
                pnl=pnl_percent,
                pnl_pct=pnl_percent,
                features={"iv_rv": row.get("iv30_rv30"), "expected_move_pct": row.get("expected_move_pct"),
                          "atm": atm, "lo": lo, "hi": hi, "net_premium": net_premium},
                ml_decision="TAKE",
                notes=f"atm={atm} w={self.width} prem={net_premium:.2f} pnl={pnl_percent:.2f}"
            ))

        if not trades:
            return StrategyResult(self.name, [])

        taken = trades
        total_pnl = sum(t.pnl for t in taken)
        win_rate = sum(1 for t in taken if t.pnl > 0) / len(taken)
        return StrategyResult(self.name, trades, {
            "total": len(trades),
            "taken": len(taken),
            "avg_pnl": float(np.mean([t.pnl for t in taken])),
            "total_pnl": float(round(total_pnl, 4)),
            "win_rate": win_rate,
        })


# ---------------------------------------------------------------------------
# Risk Reversal (buy OTM call, sell OTM put — directional skew arb)
# ---------------------------------------------------------------------------

class RiskReversalReal:
    """Buy OTM call, sell OTM put for ~zero net premium.

    Profit when underlying rises; loss when underlying falls further than strike distance.
    """
    name = "risk_reversal_real"

    def __init__(self, iv_rv_max: float = 1.3, width: int = 10):
        self.iv_rv_max = iv_rv_max
        self.width = width

    def run(self, bundle: DataBundle) -> StrategyResult:
        trades = []
        snap = bundle.snapshots
        if snap.empty:
            return StrategyResult(self.name, [])

        mask = (
            snap["actual_move_pct"].notna()
            & snap["expected_move_pct"].notna()
            & snap["iv30_rv30"].notna()
            & (snap["iv30_rv30"] <= self.iv_rv_max)
            & snap["price"].notna()
            & snap["atm_iv_near"].notna()
            & snap["nearest_expiry"].notna()
        )
        filtered = snap.loc[mask].copy()
        if filtered.empty:
            return StrategyResult(self.name, [])

        api = AlpacaMultiStrike(bundle)

        for _, row in filtered.iterrows():
            ticker = row["ticker"]
            scan_date = row["scan_date"]
            S = float(row["price"])
            iv = float(row["atm_iv_near"])
            try:
                expiry = date.fromisoformat(str(row["nearest_expiry"])[:10])
            except Exception:
                continue
            scan = date.fromisoformat(str(scan_date)[:10]) if scan_date else expiry - timedelta(days=7)
            T = max((expiry - scan).days / 365.0, 1/365)
            r = 0.045

            chain = api.chain_for(ticker, str(scan_date))
            atm = nearest_strike(S, self.width)
            kc = atm + self.width
            kp = atm - self.width

            call_bid, call_ask = api.bid_ask_or_bs(chain, f"{ticker}{expiry.strftime('%y%m%d')}C{int(kc*1000):08d}", S, kc, T, iv, "call")
            put_bid, put_ask = api.bid_ask_or_bs(chain, f"{ticker}{expiry.strftime('%y%m%d')}P{int(kp*1000):08d}", S, kp, T, iv, "put")

            # Buy OTM call (pay ask), sell OTM put (receive bid)
            net_premium = call_ask - put_bid

            actual_move = float(row["actual_move_pct"])
            underlying_at_expiry = S + S * actual_move / 100.0

            call_payoff = max(0.0, underlying_at_expiry - kc)
            put_payoff = max(0.0, kp - underlying_at_expiry)
            pnl_percent = call_payoff - put_payoff - net_premium

            trades.append(Trade(
                ticker=ticker,
                earnings_date=row["earnings_date"] if isinstance(row["earnings_date"], date) else date.fromisoformat(str(row["earnings_date"])),
                scan_date=row["scan_date"] if isinstance(row["scan_date"], date) else date.fromisoformat(str(row["scan_date"])),
                strategy=self.name,
                side="RISK_REVERSAL",
                entry_price=net_premium,
                exit_price=max(0, pnl_percent + net_premium),
                pnl=pnl_percent,
                pnl_pct=pnl_percent,
                features={"iv_rv": row.get("iv30_rv30"), "expected_move_pct": row.get("expected_move_pct"),
                          "call_strike": kc, "put_strike": kp, "net_premium": net_premium},
                ml_decision="TAKE",
                notes=f"kc={kc} kp={kp} prem={net_premium:.2f} pnl={pnl_percent:.2f}"
            ))

        if not trades:
            return StrategyResult(self.name, [])

        taken = trades
        total_pnl = sum(t.pnl for t in taken)
        win_rate = sum(1 for t in taken if t.pnl > 0) / len(taken)
        return StrategyResult(self.name, trades, {
            "total": len(trades),
            "taken": len(taken),
            "avg_pnl": float(np.mean([t.pnl for t in taken])),
            "total_pnl": float(round(total_pnl, 4)),
            "win_rate": win_rate,
        })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MULTI_STRIKE_STRATEGIES = {
    IronCondorReal.name: IronCondorReal,
    ButterflyReal.name: ButterflyReal,
    RiskReversalReal.name: RiskReversalReal,
}


def run_multi_strike(bundle: DataBundle, strategies: Optional[list[str]] = None) -> dict[str, StrategyResult]:
    results = {}
    selected = strategies or list(MULTI_STRIKE_STRATEGIES.keys())
    for name in selected:
        cls = MULTI_STRIKE_STRATEGIES[name]
        instance = cls()
        results[name] = instance.run(bundle)
    return results
