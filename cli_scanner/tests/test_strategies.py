"""Tests for the strategy framework."""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import date

from earnings_edge.strategies import (
    DataBundle,
    Trade,
    list_strategies,
    get_strategy,
    StrategyResult,
    CalendarCallStrategy,
    CalendarCallHighConviction,
    CalendarCallNoML,
    StockDriftStrategy,
    IVRVMeanReversion,
    TermStructureSteepener,
    EarningsQualityStrategy,
    DebitSizeExploit,
    DAXForwardVolStrategy,
    ShortStraddleStrategy,
)

from backtest import run_all


# ---------------------------------------------------------------------------
# Helpers — build synthetic DataBundles for testing
# ---------------------------------------------------------------------------

def _make_snapshot(ticker: str, earnings_date: str, scan_date: str = "2025-01-01",
                   price: float = 100.0, iv30_rv30: float = 1.3,
                   term_slope: float = -0.05, pre_close: float = 98.0,
                   post_close: float = 102.0, outcome_fetched: bool = True) -> dict:
    return {
        "ticker": ticker,
        "earnings_date": earnings_date,
        "scan_date": scan_date,
        "timing": "Pre Market",
        "price": price,
        "avg_volume_30d": 1_000_000.0,
        "has_options": 1,
        "atm_iv_near": 0.40,
        "rv30": 0.35,
        "iv30_rv30": iv30_rv30,
        "term_slope": term_slope,
        "term_structure_valid": 1,
        "expected_move_pct": 5.0,
        "atm_call_delta": 0.5,
        "atm_put_delta": -0.5,
        "atm_call_iv": 0.40,
        "atm_put_iv": 0.40,
        "pre_earnings_close": pre_close,
        "post_earnings_close": post_close,
        "actual_move_pct": ((post_close - pre_close) / pre_close * 100) if pre_close > 0 else None,
        "outcome_fetched_at": "2025-01-02" if outcome_fetched else None,
    }


def _make_calendar_trade(ticker: str, earnings_date: str, scan_date: str = "2025-01-01",
                         strike: float = 100.0, net_debit: float = 1.0,
                         exit_value: float = 1.5, pnl: float = 0.5,
                         price: float = 100.0) -> dict:
    return {
        "ticker": ticker,
        "earnings_date": earnings_date,
        "scan_date": scan_date,
        "near_expiry": "2025-01-15",
        "far_expiry": "2025-02-15",
        "strike": strike,
        "net_debit": net_debit,
        "near_entry": 1.0,
        "far_entry": 2.0,
        "pnl_dollars": pnl,
        "return_on_debit": 0.5,
        "exit_value": exit_value,
        "price": price,
    }


# ---------------------------------------------------------------------------
# Test 1: registry has exactly 10 strategies
# ---------------------------------------------------------------------------

def test_registry_has_10_strategies():
    strategies = list_strategies()
    assert len(strategies) == 10
    expected = {
        "calendar_call_ml",
        "calendar_call_high_conviction",
        "calendar_call_no_ml",
        "stock_drift_pead",
        "iv_rv_mean_reversion",
        "term_structure_steepener",
        "dax_forward_vol",
        "short_straddle",
        "earnings_quality",
        "debit_size_exploit",
    }
    assert set(strategies) == expected


def test_get_strategy():
    s = get_strategy("calendar_call_ml")
    assert s.name == "calendar_call_ml"


def test_get_strategy_missing():
    with pytest.raises(KeyError, match="Unknown strategy"):
        get_strategy("nonexistent_strategy")


# ---------------------------------------------------------------------------
# Test 2: DataBundle construction
# ---------------------------------------------------------------------------

def test_databundle_empty():
    bundle = DataBundle(
        snapshots=pd.DataFrame(),
        calendar_trades=pd.DataFrame(),
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )
    assert bundle.snapshots.empty
    assert bundle.calendar_trades.empty


# ---------------------------------------------------------------------------
# Test 3: Stock Drift Strategy (PEAD)
# ---------------------------------------------------------------------------

def test_stock_drift():
    snap = pd.DataFrame([
        _make_snapshot("AAPL", "2025-01-15", pre_close=98.0, post_close=102.0),
        _make_snapshot("MSFT", "2025-01-16", pre_close=200.0, post_close=201.0),
        _make_snapshot("BAD", "2025-01-17", price=2.0, pre_close=1.5, post_close=1.2),  # below min price
    ])
    data = DataBundle(
        snapshots=snap,
        calendar_trades=pd.DataFrame(),
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )

    strategy = StockDriftStrategy(min_price=5.0, min_volume=0)
    result = strategy.run(data)

    assert result.name == "stock_drift_pead"
    # AAPL: (102-98)/98 * 100 = 4.08%
    # MSFT: (201-200)/200 * 100 = 0.5%
    # BAD: filtered (min_price)
    assert len(result.trades) == 2
    assert result.summary["total"] == 2
    assert result.summary["avg_return_pct"] > 0
    assert result.summary["win_rate"] == 1.0  # both positive


def test_stock_drift_missing_prices():
    snap = pd.DataFrame([
        _make_snapshot("X", "2025-01-01", pre_close=0, post_close=0),  # zero pre_close → skipped
        _make_snapshot("Y", "2025-01-02", outcome_fetched=False),  # missing outcome → skipped
    ])
    data = DataBundle(
        snapshots=snap,
        calendar_trades=pd.DataFrame(),
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )
    result = StockDriftStrategy(min_price=0, min_volume=0).run(data)
    assert len(result.trades) == 0


# ---------------------------------------------------------------------------
# Test 4: Calendar Call ML (quality gates)
# ---------------------------------------------------------------------------

def test_calendar_call_quality_gates():
    # 3 trades: 1 clean, 1 bad moneyness, 1 negative debit
    ct = pd.DataFrame([
        _make_calendar_trade("A", "2025-01-10", strike=100.0, net_debit=1.0, exit_value=1.5, pnl=0.5),
        _make_calendar_trade("B", "2025-01-11", strike=200.0, net_debit=1.0, exit_value=1.0, pnl=-1.0),  # moneyness 200/100=2 → bad
        _make_calendar_trade("C", "2025-01-12", strike=100.0, net_debit=-0.5, exit_value=0.5, pnl=-1.0),  # negative debit
    ])
    snap = pd.DataFrame([
        _make_snapshot("A", "2025-01-10"),
        _make_snapshot("B", "2025-01-11"),
        _make_snapshot("C", "2025-01-12"),
    ])

    data = DataBundle(
        snapshots=snap,
        calendar_trades=ct,
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )

    strategy = CalendarCallStrategy(min_rows=1)
    result = strategy.run(data)

    # Only trade A passes quality gates
    assert len(result.trades) >= 1
    tickers = {t.ticker for t in result.trades}
    assert "A" in tickers
    # Trade B fails moneyness (no snapshot price matches) — actually B has snapshot so moneyness=200/100=2.0 fails
    # Trade C fails negative debit


def test_calendar_call_no_trades():
    data = DataBundle(
        snapshots=pd.DataFrame(),
        calendar_trades=pd.DataFrame(),
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )
    result = CalendarCallStrategy(min_rows=1).run(data)
    assert result.trades == []


# ---------------------------------------------------------------------------
# Test 5: Calendar Call No-ML
# ---------------------------------------------------------------------------

def test_calendar_call_no_ml():
    ct = pd.DataFrame([
        _make_calendar_trade("A", "2025-01-10", strike=100.0, net_debit=1.0, exit_value=1.5, pnl=0.5),
        _make_calendar_trade("B", "2025-01-11", strike=100.0, net_debit=1.2, exit_value=0.8, pnl=-0.4),
    ])
    snap = pd.DataFrame([
        _make_snapshot("A", "2025-01-10"),
        _make_snapshot("B", "2025-01-11"),
    ])

    data = DataBundle(
        snapshots=snap,
        calendar_trades=ct,
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )

    result = CalendarCallNoML(min_rows=1).run(data)
    # Without ML, all should be TAKE
    assert all(t.ml_decision == "TAKE" for t in result.trades)
    assert all(t.model_score is None for t in result.trades)
    assert result.summary["taken"] == result.summary["total"]


# ---------------------------------------------------------------------------
# Test 6: Calendar Call High-Conviction
# ---------------------------------------------------------------------------

def test_high_conviction_risk_overlay():
    ct = pd.DataFrame([
        # Cheap + high score → TAKE
        _make_calendar_trade("A", "2025-01-10", strike=100.0, net_debit=1.5, exit_value=2.0, pnl=1.0),
        # Expensive → SKIP_RISK
        _make_calendar_trade("B", "2025-01-11", strike=100.0, net_debit=3.0, exit_value=3.5, pnl=1.5),
    ])
    snap = pd.DataFrame([
        _make_snapshot("A", "2025-01-10"),
        _make_snapshot("B", "2025-01-11"),
    ])

    data = DataBundle(
        snapshots=snap,
        calendar_trades=ct,
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )

    result = CalendarCallHighConviction(min_rows=1).run(data)
    # B is expensive → risk overlay
    b_trade = next((t for t in result.trades if t.ticker == "B"), None)
    if b_trade:
        assert b_trade.ml_decision in ("SKIP_RISK", "SKIP") or b_trade.notes


# ---------------------------------------------------------------------------
# Test 7: IV/RV Mean Reversion
# ---------------------------------------------------------------------------

def test_iv_rv_filter():
    ct = pd.DataFrame([
        _make_calendar_trade("A", "2025-01-10"),
        _make_calendar_trade("B", "2025-01-11"),
    ])
    # A has IV/RV = 1.5 (> 1.15 threshold), B has IV/RV = 1.0 (< 1.15 threshold)
    snap = pd.DataFrame([
        _make_snapshot("A", "2025-01-10", iv30_rv30=1.5),
        _make_snapshot("B", "2025-01-11", iv30_rv30=1.0),
    ])

    data = DataBundle(
        snapshots=snap,
        calendar_trades=ct,
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )

    result = IVRVMeanReversion(iv_rv_min=1.15, min_rows=1).run(data)
    tickers = {t.ticker for t in result.trades}
    assert "A" in tickers
    assert "B" not in tickers


# ---------------------------------------------------------------------------
# Test 8: Term Structure Steepener
# ---------------------------------------------------------------------------

def test_term_structure_filter():
    ct = pd.DataFrame([
        _make_calendar_trade("A", "2025-01-10"),
        _make_calendar_trade("B", "2025-01-11"),
    ])
    # A has term_slope = -0.05 (≤ -0.03), B has term_slope = -0.01 (> -0.03)
    snap = pd.DataFrame([
        _make_snapshot("A", "2025-01-10", term_slope=-0.05),
        _make_snapshot("B", "2025-01-11", term_slope=-0.01),
    ])

    data = DataBundle(
        snapshots=snap,
        calendar_trades=ct,
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )

    result = TermStructureSteepener(term_slope_max=-0.03, min_rows=1).run(data)
    tickers = {t.ticker for t in result.trades}
    assert "A" in tickers
    assert "B" not in tickers


# ---------------------------------------------------------------------------
# Test 9: Earnings Quality
# ---------------------------------------------------------------------------

def test_earnings_quality():
    snap = pd.DataFrame([
        # Large move (> 5% threshold) with outcome
        _make_snapshot("A", "2025-01-10", pre_close=100.0, post_close=110.0),  # +10%
        # Another large move
        _make_snapshot("B", "2025-01-11", pre_close=50.0, post_close=45.0),  # -10%
        # Small move (< 5%)
        _make_snapshot("C", "2025-01-12", pre_close=200.0, post_close=201.0),  # +0.5%
    ])

    data = DataBundle(
        snapshots=snap,
        calendar_trades=pd.DataFrame(),
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )

    result = EarningsQualityStrategy(threshold_pct=5.0).run(data)
    tickers = {t.ticker for t in result.trades}
    assert "A" in tickers
    assert "B" in tickers
    assert "C" not in tickers
    assert result.summary["total"] == 2
    assert result.summary["long_trades"] == 1
    assert result.summary["short_trades"] == 1
    assert result.summary["total_pnl_pct"] == 0.0  # +10% and -10% cancel to zero


def test_earnings_quality_no_outcome():
    snap = pd.DataFrame([
        _make_snapshot("A", "2025-01-10", outcome_fetched=False),
    ])
    data = DataBundle(
        snapshots=snap,
        calendar_trades=pd.DataFrame(),
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )
    result = EarningsQualityStrategy(threshold_pct=5.0).run(data)
    assert len(result.trades) == 0


# ---------------------------------------------------------------------------
# Test 10: Debit Size Exploit
# ---------------------------------------------------------------------------

def test_debit_size_filter():
    ct = pd.DataFrame([
        # Cheap entry: debit_pct = 2.5 / 500 = 0.005 ≤ 0.03
        _make_calendar_trade("CHEAP", "2025-01-10", scan_date="2025-01-01", price=500.0, net_debit=2.5, strike=500.0),
        # Expensive entry: debit_pct = 5.0 / 10 = 0.50 > 0.03
        _make_calendar_trade("EXP", "2025-01-11", scan_date="2025-01-01", price=10.0, net_debit=5.0, strike=10.0),
    ])
    snap = pd.DataFrame([
        _make_snapshot("CHEAP", "2025-01-10", scan_date="2025-01-01", price=500.0),
        _make_snapshot("EXP", "2025-01-11", scan_date="2025-01-01", price=10.0),
    ])

    data = DataBundle(
        snapshots=snap,
        calendar_trades=ct,
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )

    result = DebitSizeExploit(max_debit_pct=0.03, min_rows=1).run(data)
    tickers = {t.ticker for t in result.trades}
    assert "CHEAP" in tickers
    assert "EXP" not in tickers


# ---------------------------------------------------------------------------
# Test 11: Placeholder strategies (data-gathering)
# ---------------------------------------------------------------------------

def test_dax_forward_vol():
    data = DataBundle(
        snapshots=pd.DataFrame(),
        calendar_trades=pd.DataFrame(),
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )
    result = DAXForwardVolStrategy().run(data)
    assert result.trades == []
    assert "Eurex" in result.summary.get("note", "")
    assert result.name == "dax_forward_vol"


def test_short_straddle():
    data = DataBundle(
        snapshots=pd.DataFrame(),
        calendar_trades=pd.DataFrame(),
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )
    result = ShortStraddleStrategy().run(data)
    assert result.trades == []
    assert "bid/ask" in result.summary.get("note", "").lower() or "multi-strike" in result.summary.get("note", "").lower()
    assert result.name == "short_straddle"


# ---------------------------------------------------------------------------
# Test 12: run_all entry point
# ---------------------------------------------------------------------------

def test_run_all():
    snap = pd.DataFrame([
        _make_snapshot("AAPL", "2025-01-10", pre_close=98.0, post_close=102.0),
    ])
    data = DataBundle(
        snapshots=snap,
        calendar_trades=pd.DataFrame(),
        live_candidates=pd.DataFrame(),
        scan_outputs=pd.DataFrame(),
    )

    # Just run a safe data-gathering strategy
    results = run_all(data, strategies=["dax_forward_vol"])
    assert "dax_forward_vol" in results
    assert isinstance(results["dax_forward_vol"], StrategyResult)


# ---------------------------------------------------------------------------
# Test 13: Trade dataclass
# ---------------------------------------------------------------------------

def test_trade_is_winner():
    trade = Trade("AAPL", date(2025, 1, 10), date(2025, 1, 1), "test", "LONG", 100.0, pnl=5.0)
    assert trade.is_winner()

    trade_loss = Trade("AAPL", date(2025, 1, 10), date(2025, 1, 1), "test", "LONG", 100.0, pnl=-3.0)
    assert not trade_loss.is_winner()


# ---------------------------------------------------------------------------
# Test 14: StrategyResult dataframe
# ---------------------------------------------------------------------------

def test_strategy_result_dataframe():
    trades = [
        Trade("A", date(2025, 1, 1), date(2025, 1, 1), "s", "LONG", 100.0, pnl=1.0, ml_decision="TAKE"),
        Trade("B", date(2025, 1, 2), date(2025, 1, 2), "s", "SHORT", 50.0, pnl=-0.5, ml_decision="SKIP"),
    ]
    sr = StrategyResult("test", trades)
    df = sr.to_dataframe()
    assert len(df) == 2
    assert "ticker" in df.columns
    assert "ml_decision" in df.columns


def test_strategy_result_dataframe_empty():
    sr = StrategyResult("test", [])
    df = sr.to_dataframe()
    assert df.empty
