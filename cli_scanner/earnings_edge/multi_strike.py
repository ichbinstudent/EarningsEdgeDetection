"""Multi-strike option-structure pricing from snapshot aggregates.

Uses Black-Scholes with ATM IV and a flat vol surface to synthesize
iron-condor, butterfly, and risk-reversal payoff profiles for each
snapshot with populated ATM data. These are backtest-calibration
tools — they evaluate whether snapshot features (IV/RV, term slope,
ATM IV level) have predictive power for common option structures.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes core (identical math to analyzer.py)
# ---------------------------------------------------------------------------

def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             opt_type: str = "call") -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if opt_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float,
             opt_type: str = "call") -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1)) if opt_type == "call" else float(norm.cdf(d1) - 1)


def _nearest_strike(price: float, width: int = 5) -> float:
    return round(price / width) * width


# ---------------------------------------------------------------------------
# Iron Condor
# ---------------------------------------------------------------------------

@dataclass
class IronCondor:
    """Sell OTM call + put, buy further-out wings.

    PnL when stock stays between short strikes = premium received minus
    wing costs (max credit). PnL when stock exceeds long strikes = max loss.
    """
    short_call: float
    long_call: float
    short_put: float
    long_put: float
    credit: float
    max_loss: float

    @classmethod
    def construct(cls, S: float, wing_width: int = 5, r: float = 0.045,
                  T: float = None, sigma: float = 0.5) -> "IronCondor":
        atm = _nearest_strike(S)
        sc = atm + wing_width
        lc = atm + 2 * wing_width
        sp = atm - wing_width
        lp = atm - 2 * wing_width

        if T is None:
            T = 7 / 365.0

        # Credit: sell OTM strangle, buy further OTM wings
        short_call_px = bs_price(S, sc, T, r, sigma, "call")
        long_call_px = bs_price(S, lc, T, r, sigma, "call")
        short_put_px = bs_price(S, sp, T, r, sigma, "put")
        long_put_px = bs_price(S, lp, T, r, sigma, "put")

        credit = short_call_px + short_put_px - long_call_px - long_put_px
        max_loss = wing_width - credit
        return cls(sc, lc, sp, lp, credit, max_loss)

    def payoff(self, underlying_at_expiry: float) -> float:
        # Call-side: loss when underlying > short_call
        call_loss = max(0.0, underlying_at_expiry - self.short_call) \
                    - max(0.0, underlying_at_expiry - self.long_call)
        # Put-side: loss when underlying < short_put
        put_loss = max(0.0, self.short_put - underlying_at_expiry) \
                   - max(0.0, self.long_put - underlying_at_expiry)
        return self.credit - call_loss - put_loss


# ---------------------------------------------------------------------------
# Butterfly
# ---------------------------------------------------------------------------

@dataclass
class Butterfly:
    """Long butterfly: buy lower, sell 2x ATM call, buy upper.

    Max profit when stock = ATM at expiry; max loss = net premium paid.
    """
    lower: float
    upper: float
    net_premium: float

    @classmethod
    def construct(cls, S: float, width: int = 5, r: float = 0.045,
                  T: float = None, sigma: float = 0.5) -> "Butterfly":
        atm = _nearest_strike(S)
        lo = atm - width
        hi = atm + width

        if T is None:
            T = 7 / 365.0

        lower_px = bs_price(S, lo, T, r, sigma, "call")
        atm_px = bs_price(S, atm, T, r, sigma, "call")
        upper_px = bs_price(S, hi, T, r, sigma, "call")

        net = lower_px + upper_px - 2 * atm_px
        return cls(lo, hi, net)

    def payoff(self, underlying_at_expiry: float) -> float:
        center = _nearest_strike((self.lower + self.upper) / 2)
        lower_payoff = max(0.0, underlying_at_expiry - self.lower)
        atm_payoff = 2 * max(0.0, underlying_at_expiry - center)
        upper_payoff = max(0.0, underlying_at_expiry - self.upper)
        return lower_payoff - atm_payoff + upper_payoff - self.net_premium


# ---------------------------------------------------------------------------
# Risk Reversal
# ---------------------------------------------------------------------------

@dataclass
class RiskReversal:
    """Buy OTM call, sell OTM put (or vice versa) — zero net premium, pure directional.

    When premium collected on put > premium paid on call, credit received;
    when call premium > put premium, debit paid. Zero-premium risk reversals
    are standard — but here we just compute actual premium.

    PnL at expiry = call payoff - put payoff - net_premium_paid.
    """
    call_strike: float
    put_strike: float
    net_premium: float

    @classmethod
    def construct(cls, S: float, width: int = 5, r: float = 0.045,
                  T: float = None, sigma: float = 0.5) -> "RiskReversal":
        atm = _nearest_strike(S)
        kc = atm + width
        kp = atm - width

        if T is None:
            T = 7 / 365.0

        call_px = bs_price(S, kc, T, r, sigma, "call")
        put_px = bs_price(S, kp, T, r, sigma, "put")

        # Buy call, sell put → net debit
        net = call_px - put_px
        return cls(kc, kp, net)

    def payoff(self, underlying_at_expiry: float) -> float:
        call_payoff = max(0.0, underlying_at_expiry - self.call_strike)
        put_payoff = max(0.0, self.put_strike - underlying_at_expiry)
        return call_payoff - put_payoff - self.net_premium
