"""Iron fly strategy strike calculation."""

import logging
from typing import Optional

import yfinance as yf

from .config import get_logger, session
from .models import IronFlyResult

logger = get_logger("iron_fly")


def calculate_iron_fly(ticker: str) -> IronFlyResult:
    """
    Compute iron fly strikes centred on the ATM option closest to 50δ.

    Returns an ``IronFlyResult`` or an error variant on failure.
    """
    try:
        t = yf.Ticker(ticker, session=session)
        if not t.options:
            return IronFlyResult(error="No options available")

        expiry = t.options[0]
        chain = t.option_chain(expiry)
        calls, puts = chain.calls, chain.puts
        price = t.history(period="1d")["Close"].iloc[-1]

        # Choose short strikes closest to 50 delta (fallback: ATM)
        if "delta" in calls.columns and "delta" in puts.columns:
            calls["dd"] = (calls["delta"].abs() - 0.5).abs()
            puts["dd"] = (puts["delta"].abs() - 0.5).abs()
        else:
            calls["dd"] = (calls["strike"] - price).abs()
            puts["dd"] = (puts["strike"] - price).abs()

        sc = calls.loc[calls["dd"].idxmin()]
        sp = puts.loc[puts["dd"].idxmin()]
        sc_strike = round(sc["strike"], 2)
        sp_strike = round(sp["strike"], 2)
        sc_prem = round((sc["bid"] + sc["ask"]) / 2, 2)
        sp_prem = round((sp["bid"] + sp["ask"]) / 2, 2)
        credit = round(sc_prem + sp_prem, 2)

        # Wing width = 3× credit
        wing = round(3 * credit, 2)
        target_lp = sp_strike - wing
        target_lc = sc_strike + wing

        avail_p = sorted(puts["strike"].unique())
        avail_c = sorted(calls["strike"].unique())
        lp_strike = round(min(avail_p, key=lambda x: abs(x - target_lp)), 2)
        lc_strike = round(min(avail_c, key=lambda x: abs(x - target_lc)), 2)

        lp_row = puts[puts["strike"] == lp_strike].iloc[0]
        lc_row = calls[calls["strike"] == lc_strike].iloc[0]
        lp_prem = round((lp_row["bid"] + lp_row["ask"]) / 2, 2)
        lc_prem = round((lc_row["bid"] + lc_row["ask"]) / 2, 2)
        debit = round(lp_prem + lc_prem, 2)
        net = round(credit - debit, 2)

        pw = round(sp_strike - lp_strike, 2)
        cw = round(lc_strike - sc_strike, 2)
        max_profit = net
        max_risk = round(min(pw, cw) - net, 2)
        rr = round(max_risk / max_profit, 1) if max_profit > 0 else float("inf")

        return IronFlyResult(
            short_call_strike=sc_strike, short_put_strike=sp_strike,
            long_call_strike=lc_strike, long_put_strike=lp_strike,
            short_call_premium=sc_prem, short_put_premium=sp_prem,
            long_call_premium=lc_prem, long_put_premium=lp_prem,
            total_credit=credit, total_debit=debit, net_credit=net,
            put_wing_width=pw, call_wing_width=cw,
            max_profit=max_profit, max_risk=max_risk,
            upper_breakeven=round(sc_strike + net, 2),
            lower_breakeven=round(sp_strike - net, 2),
            risk_reward_ratio=rr,
            expiration=expiry,
        )
    except Exception as exc:
        logger.warning(f"Iron fly calc failed for {ticker}: {exc}")
        return IronFlyResult(error=str(exc))
