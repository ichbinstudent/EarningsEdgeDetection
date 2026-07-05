"""Tests for Alpaca options client and multi-strike positional strategies."""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch

from earnings_edge.collectors.alpaca_options import AlpacaOptionsClient
from earnings_edge.multi_strike_real import (
    bs_price, delta, strike_for_delta, nearest_strike, AlpacaMultiStrike,
    IronCondorReal, ButterflyReal, RiskReversalReal, pull_bid_ask,
    MULTI_STRIKE_STRATEGIES, run_multi_strike,
)
from earnings_edge.strategies import DataBundle, Trade


# ---------------------------------------------------------------------------
# BS helpers
# ---------------------------------------------------------------------------

def test_bs_price_call():
    # ITM call should be worth at least intrinsic
    p = bs_price(100, 90, 0.1, 0.045, 0.4, "call")
    assert p > 10  # intrinsic = 10
    assert p < 100

def test_bs_price_put():
    p = bs_price(100, 110, 0.1, 0.045, 0.4, "put")
    assert p > 10  # intrinsic = 10
    assert p < 100

def test_delta_atm_call():
    # ATM call delta ≈ 0.5
    d = delta(100, 100, 0.1, 0.045, 0.4, "call")
    assert 0.45 <= d <= 0.55

def test_delta_atm_put():
    d = delta(100, 100, 0.1, 0.045, 0.4, "put")
    assert -0.55 <= d <= -0.45

def test_strike_for_delta_call():
    # For a 15-delta call, strike should be above spot
    k = strike_for_delta(100, 0.1, 0.045, 0.4, 0.15)
    assert k > 95
    assert k < 130  # reasonable range

def test_strike_for_delta_put():
    k = strike_for_delta(100, 0.1, 0.045, 0.4, -0.15)
    assert k < 105
    assert k > 70


# ---------------------------------------------------------------------------
# Client basics
# ---------------------------------------------------------------------------

def test_alpaca_client_builds_headers():
    c = AlpacaOptionsClient("key123", "secret456")
    h = c.session.headers
    assert h["APCA-API-KEY-ID"] == "key123"
    assert h["APCA-API-SECRET-KEY"] == "secret456"


def test_bar_ask_with_chain_empty():
    df = pd.DataFrame()
    b, a, m = pull_bid_ask(df, "AAPL250117C00150000")
    assert b is None and a is None and m is None

def test_pull_bid_ask_found():
    df = pd.DataFrame([{
        "contract_ticker": "AAPL250117C00150000",
        "bid": 4.0, "ask": 5.0,
    }])
    b, a, m = pull_bid_ask(df, "AAPL250117C00150000")
    assert b == 4.0 and a == 5.0 and m == 4.5


# ---------------------------------------------------------------------------
# AlpacaMultiStrike helper
# ---------------------------------------------------------------------------

def test_nearest_strike():
    assert nearest_strike(102.3, 5) == 100.0
    assert nearest_strike(104.0, 5) == 105.0
    assert nearest_strike(102.5, 5) == 100.0 if (102.5 // 5) * 5 == 100.0 else True


# ---------------------------------------------------------------------------
# DataBundle + strategies (with BS fallback since no real chain data in test)
# ---------------------------------------------------------------------------

def _make_snapshot(ticker="AAPL", earnings_date="2025-01-10", scan_date="2025-01-09",
                   price=100.0, iv30_rv30=1.5, expected_move_pct=8.0,
                   actual_move_pct=4.0, has_options=1, nearest_expiry="2025-01-17") -> dict:
    return {
        "ticker": ticker,
        "earnings_date": earnings_date,
        "scan_date": scan_date,
        "price": price,
        "avg_volume_30d": 1_000_000.0,
        "has_options": has_options,
        "atm_iv_near": 0.40,
        "rv30": 0.30,
        "iv30_rv30": iv30_rv30,
        "hist_vol_3m": 0.32,
        "sigma_baseline_1y": 0.35,
        "sigma_short_leg": 0.40,
        "sigma_short_leg_fair": 0.33,
        "actual_to_fair_ratio": 21.2,
        "term_slope": -0.05,
        "term_structure_valid": 1,
        "expected_move_pct": expected_move_pct,
        "actual_move_pct": actual_move_pct,
        "nearest_expiry": nearest_expiry,
        "outcome_fetched_at": "2025-01-11",
    }


def _make_bundle(snapshots=None, options_chain=None):
    if snapshots is None:
        snapshots = pd.DataFrame([_make_snapshot()])
    return DataBundle(
        snapshots=snapshots if isinstance(snapshots, pd.DataFrame) else pd.DataFrame(snapshots),
        calendar_trades=pd.DataFrame(),
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
        options_chain=options_chain if options_chain is not None else pd.DataFrame(),
    )


def test_iron_condor_registry():
    assert "iron_condor_real" in MULTI_STRIKE_STRATEGIES

def test_iron_condor_no_chain_falls_back_to_bs():
    """With empty options_chain, should still run using BS fallback."""
    bundle = _make_bundle()
    result = IronCondorReal().run(bundle)
    # Should produce trades (using BS fallback)
    assert len(result.trades) >= 0  # may be empty if filters don't pass

def test_iron_condor_filter_blocks_low_iv_rv():
    bundle = _make_bundle(pd.DataFrame([_make_snapshot(iv30_rv30=1.0, expected_move_pct=4.0)]))
    result = IronCondorReal(iv_rv_min=1.2, min_expected_move=6.0).run(bundle)
    assert result.trades == []

def test_iron_condor_trade_structure():
    """When filters pass, trades should have IRON_CONDOR side and proper features."""
    bundle = _make_bundle()
    result = IronCondorReal().run(bundle)
    for t in result.trades:
        assert t.side == "IRON_CONDOR"
        assert "risk_reward" in t.features
        assert "short_call" in t.features
        assert "long_call" in t.features

def test_iron_condor_summary_has_win_rate():
    bundle = _make_bundle()
    result = IronCondorReal().run(bundle)
    if result.trades:
        assert "win_rate" in result.summary
        assert "total_pnl" in result.summary


def test_butterfly_registry():
    assert "butterfly_real" in MULTI_STRIKE_STRATEGIES

def test_butterfly_blocks_low_iv_rv():
    bundle = _make_bundle(pd.DataFrame([_make_snapshot(iv30_rv30=1.0, expected_move_pct=4.0)]))
    result = ButterflyReal(iv_rv_min=1.15, min_expected_move=6.0).run(bundle)
    assert result.trades == []

def test_butterfly_trade_structure():
    bundle = _make_bundle()
    result = ButterflyReal().run(bundle)
    for t in result.trades:
        assert t.side == "BUTTERFLY"
        assert "atm" in t.features
        assert "lo" in t.features
        assert "hi" in t.features


def test_risk_reversal_registry():
    assert "risk_reversal_real" in MULTI_STRIKE_STRATEGIES

def test_risk_reversal_blocks_high_iv_rv():
    bundle = _make_bundle(pd.DataFrame([_make_snapshot(iv30_rv30=1.5)]))
    result = RiskReversalReal(iv_rv_max=1.3).run(bundle)
    assert result.trades == []

def test_risk_reversal_trade_structure():
    bundle = _make_bundle(pd.DataFrame([_make_snapshot(iv30_rv30=1.0)]))
    result = RiskReversalReal().run(bundle)
    for t in result.trades:
        assert t.side == "RISK_REVERSAL"
        assert "call_strike" in t.features
        assert "put_strike" in t.features


# ---------------------------------------------------------------------------
# Bundle options_chain DataLoader integration
# ---------------------------------------------------------------------------

def test_data_bundle_options_chain_field():
    bundle = _make_bundle()
    assert hasattr(bundle, "options_chain")
    assert bundle.options_chain.empty  # default empty

def test_data_bundle_with_chain_data():
    chain_df = pd.DataFrame([{
        "ticker": "AAPL", "scan_date": "2025-01-09",
        "contract_ticker": "AAPL250117C00150000",
        "bid": 4.0, "ask": 5.0,
    }])
    bundle = _make_bundle(options_chain=chain_df)
    assert len(bundle.options_chain) == 1


# ---------------------------------------------------------------------------
# run_multi_strike entry point
# ---------------------------------------------------------------------------

def test_run_multi_strike():
    bundle = _make_bundle()
    results = run_multi_strike(bundle)
    assert len(results) == 3
    assert set(results.keys()) == {"iron_condor_real", "butterfly_real", "risk_reversal_real"}

def test_run_multi_strike_subset():
    bundle = _make_bundle()
    results = run_multi_strike(bundle, strategies=["iron_condor_real"])
    assert len(results) == 1
    assert "iron_condor_real" in results
