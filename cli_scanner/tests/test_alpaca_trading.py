"""Tests for Alpaca trading client & strategy bridge."""
from __future__ import annotations

import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch, PropertyMock

from earnings_edge.alpaca_trading import (
    AlpacaTradingClient,
    AlpacaError,
    AlpacaAuthError,
    AlpacaNotFoundError,
    OrderResult,
    PositionManager,
    create_client,
)
from earnings_edge.alpaca_bridge import (
    StrategyBridge,
    BridgeConfig,
    run_auto_trade,
    BEST_STRATEGIES,
    MAX_PCT_PER_TRADE,
)
from earnings_edge.strategies import Trade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    """AlpacaTradingClient with mocked session and methods."""
    with patch.dict("os.environ", {"APCA_API_KEY_ID": "test_key", "APCA_API_SECRET_KEY": "test_secret"}):
        client = AlpacaTradingClient(api_key="test_key", api_secret="test_secret")
    client.session = MagicMock()
    # Default mock: no positions exist
    client.has_position = MagicMock(return_value=False)
    client.buying_power = MagicMock(return_value=50000.0)
    client.get_option_contracts = MagicMock(return_value={"option_contracts": []})
    client.get_option_snapshot = MagicMock(return_value={})
    return client

@pytest.fixture
def bridge(mock_client):
    config = BridgeConfig(dry_run=True)
    return StrategyBridge(client=mock_client, config=config)


# ---------------------------------------------------------------------------
# Client basics
# ---------------------------------------------------------------------------

def test_client_env_vars():
    with patch.dict("os.environ", {"APCA_API_KEY_ID": "env_key", "APCA_API_SECRET_KEY": "env_secret"}):
        c = AlpacaTradingClient()
        assert c.api_key == "env_key"
        assert c.api_secret == "env_secret"

def test_client_explicit_args():
    c = AlpacaTradingClient(api_key="explicit", api_secret="args")
    assert c.api_key == "explicit"
    assert c.api_secret == "args"

def test_client_headers_set():
    with patch.dict("os.environ", {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"}):
        c = AlpacaTradingClient()
    h = c.session.headers
    assert h["APCA-API-KEY-ID"] == "k"
    assert h["APCA-API-SECRET-KEY"] == "s"

def test_account_parsing(mock_client):
    mock_client.session.request.return_value = MagicMock(
        status_code=200,
        json=lambda: {"buying_power": "50000.00", "portfolio_value": "100000.00"},
    )
    acct = mock_client.get_account()
    # API returns strings; buying_power() coerces to float
    assert float(acct["buying_power"]) == 50000.0
    assert mock_client.buying_power() == 50000.0

def test_auth_error_raises(mock_client):
    mock_client.session.request.return_value = MagicMock(
        status_code=401,
        json=lambda: {"message": "Unauthorized"},
    )
    with pytest.raises(AlpacaAuthError):
        mock_client.get_account()

def test_not_found_raises(mock_client):
    mock_client.session.request.return_value = MagicMock(
        status_code=404,
        json=lambda: {"message": "Not found"},
    )
    result = mock_client.get_position("AAPL250117C00150000")
    assert result is None


# ---------------------------------------------------------------------------
# OrderResult
# ---------------------------------------------------------------------------

def test_order_result_from_alpaca():
    raw = {
        "id": "ord-123",
        "client_order_id": "strat_ticker_12345",
        "symbol": "AAPL250117C00150000",
        "status": "filled",
        "filled_qty": "1",
        "filled_avg_price": "4.50",
        "created_at": "2025-01-10T14:00:00Z",
        "legs": [],
    }
    result = OrderResult.from_alpaca(raw, strategy="calendar_call_ml")
    assert result.order_id == "ord-123"
    assert result.strategy == "calendar_call_ml"
    assert result.status == "filled"
    assert result.filled_qty == 1


# ---------------------------------------------------------------------------
# Bridge: leg builders
# ---------------------------------------------------------------------------

def _make_trade(side="DIRECTIONAL_CALL", features=None):
    return Trade(
        ticker="AAPL",
        earnings_date=date(2025, 1, 10),
        scan_date=date(2025, 1, 9),
        strategy="test",
        side=side,
        entry_price=5.0,
        exit_price=4.0,
        pnl=1.0,
        pnl_pct=1.0,
        features=features or {"atm_strike": 150.0, "expiry": date(2025, 1, 17)},
        ml_decision="TAKE",
    )

def test_bridge_single_call_leg(bridge):
    trade = _make_trade("DIRECTIONAL_CALL")
    legs = bridge._build_legs(trade)
    assert len(legs) == 1
    assert legs[0]["side"] == "buy"
    assert legs[0]["option_type"] == "call"
    assert legs[0]["strike"] == 150.0
    assert "AAPL250117C" in legs[0]["symbol"]

def test_bridge_single_put_leg(bridge):
    trade = _make_trade("DIRECTIONAL_PUT")
    legs = bridge._build_legs(trade)
    assert len(legs) == 1
    assert legs[0]["side"] == "buy"
    assert legs[0]["option_type"] == "put"
    assert "AAPL250117P" in legs[0]["symbol"]

def test_bridge_short_straddle_legs(bridge):
    trade = _make_trade("SHORT_STRADDLE", {"atm_strike": 250.0, "expiry": date(2025, 1, 17)})
    legs = bridge._build_legs(trade)
    assert len(legs) == 2
    sides = {leg["side"] for leg in legs}
    types = {leg["option_type"] for leg in legs}
    assert sides == {"sell"}
    assert types == {"call", "put"}

def test_bridge_long_straddle_legs(bridge):
    trade = _make_trade("LONG_STRADDLE", {"atm_strike": 250.0, "expiry": date(2025, 1, 17)})
    legs = bridge._build_legs(trade)
    assert len(legs) == 2
    sides = {leg["side"] for leg in legs}
    assert sides == {"buy"}

def test_bridge_iron_condor_legs(bridge):
    trade = _make_trade("IRON_CONDOR", {
        "short_call": 160.0, "short_put": 140.0,
        "long_call": 165.0, "long_put": 135.0,
        "expiry": date(2025, 1, 17),
    })
    legs = bridge._build_legs(trade)
    assert len(legs) == 4
    assert legs[0]["side"] == "buy" and legs[0]["option_type"] == "put"
    assert legs[1]["side"] == "sell" and legs[1]["option_type"] == "put"
    assert legs[2]["side"] == "sell" and legs[2]["option_type"] == "call"
    assert legs[3]["side"] == "buy" and legs[3]["option_type"] == "call"

def test_bridge_butterfly_legs(bridge):
    trade = _make_trade("BUTTERFLY", {
        "atm": 150.0, "lo": 145.0, "hi": 155.0,
        "option_type": "call", "expiry": date(2025, 1, 17),
    })
    legs = bridge._build_legs(trade)
    assert len(legs) == 3
    # 1 long lo, 2 short atm, 1 long hi
    assert legs[0]["side"] == "buy" and legs[0]["strike"] == 145.0
    assert legs[1]["side"] == "sell" and legs[1]["strike"] == 150.0 and legs[1]["ratio_qty"] == 2
    assert legs[2]["side"] == "buy" and legs[2]["strike"] == 155.0

def test_bridge_risk_reversal_legs(bridge):
    trade = _make_trade("RISK_REVERSAL", {
        "call_strike": 420.0, "put_strike": 400.0, "expiry": date(2025, 1, 17),
    })
    legs = bridge._build_legs(trade)
    assert len(legs) == 2
    sides = {leg["side"] for leg in legs}
    assert sides == {"buy", "sell"}

def test_bridge_calendar_legs(bridge):
    trade = _make_trade("CALENDAR", {
        "near_strike": 150.0, "far_strike": 150.0,
        "near_expiry": date(2025, 1, 17), "far_expiry": date(2025, 2, 21),
    })
    legs = bridge._build_legs(trade)
    assert len(legs) == 2
    sides = {leg["side"] for leg in legs}
    assert sides == {"buy", "sell"}


# ---------------------------------------------------------------------------
# Bridge: OCC symbol construction
# ---------------------------------------------------------------------------

def test_occ_symbol_basic():
    b = StrategyBridge(client=MagicMock(), config=BridgeConfig(dry_run=True))
    assert b._occ_symbol("AAPL", date(2025, 1, 17), 150.0, "call") == "AAPL250117C00150000"
    assert b._occ_symbol("TSLA", date(2025, 6, 20), 250.0, "put") == "TSLA250620P00250000"

def test_occ_symbol_fractional_strike():
    b = StrategyBridge(client=MagicMock(), config=BridgeConfig(dry_run=True))
    assert b._occ_symbol("SPY", date(2025, 1, 17), 450.5, "call") == "SPY250117C00450500"


# ---------------------------------------------------------------------------
# Bridge: config & defaults
# ---------------------------------------------------------------------------

def test_bridge_config_defaults():
    config = BridgeConfig()
    assert config.dry_run is False
    assert config.order_type == "market"
    assert config.max_pct_per_trade == MAX_PCT_PER_TRADE
    assert config.skip_if_position_exists is True

def test_bridge_config_custom():
    config = BridgeConfig(dry_run=True, max_pct_per_trade=0.20)
    assert config.dry_run is True
    assert config.max_pct_per_trade == 0.20

def test_best_strategies_list():
    assert len(BEST_STRATEGIES) >= 5
    assert "short_straddle" in BEST_STRATEGIES
    assert "vol_risk_premium" in BEST_STRATEGIES


# ---------------------------------------------------------------------------
# Bridge: execute trade (dry run)
# ---------------------------------------------------------------------------

def test_execute_trade_dry_run_single(bridge):
    trade = _make_trade("DIRECTIONAL_CALL")
    result = bridge.execute_trade(trade)
    assert result is not None
    assert result.status == "dry_run"
    assert result.strategy == "test"
    assert len(result.legs) == 1

def test_execute_trade_dry_run_straddle(bridge):
    trade = _make_trade("SHORT_STRADDLE")
    result = bridge.execute_trade(trade)
    assert result is not None
    assert result.status == "dry_run"
    assert len(result.legs) == 2

def test_execute_trade_dry_run_iron_condor(bridge):
    trade = _make_trade("IRON_CONDOR", {
        "short_call": 160.0, "short_put": 140.0,
        "long_call": 165.0, "long_put": 135.0,
        "expiry": date(2025, 1, 17),
    })
    result = bridge.execute_trade(trade)
    assert result is not None
    assert result.status == "dry_run"
    assert len(result.legs) == 4


# ---------------------------------------------------------------------------
# Bridge: DTE filtering
# ---------------------------------------------------------------------------

def test_skip_too_short_dte(bridge):
    """Trades with DTE < min should be skipped."""
    trade = _make_trade("DIRECTIONAL_CALL", {
        "atm_strike": 150.0,
        "expiry": date(2025, 1, 10),  # DTE = 0
    })
    result = bridge.execute_trade(trade)
    assert result is None  # skipped

def test_skip_existing_position(mock_client):
    """Existing position should cause skip when skip_if_position_exists=True."""
    mock_client.has_position = MagicMock(return_value=True)
    bridge = StrategyBridge(client=mock_client, config=BridgeConfig(dry_run=True))
    trade = _make_trade("DIRECTIONAL_CALL")
    result = bridge.execute_trade(trade)
    assert result is None  # skipped


# ---------------------------------------------------------------------------
# PositionManager
# ---------------------------------------------------------------------------

def test_position_manager_filters_options(mock_client):
    mock_client.session.request.return_value = MagicMock(
        status_code=200,
        json=lambda: [
            {"symbol": "AAPL250117C00150000", "asset_class": "option", "qty": "1", "market_value": "450.0"},
            {"symbol": "AAPL", "asset_class": "stock", "qty": "10", "market_value": "1500.0"},
        ],
    )
    pm = PositionManager(mock_client)
    opts = pm.open_positions()
    assert len(opts) == 1
    assert opts[0]["symbol"] == "AAPL250117C00150000"

def test_position_manager_exposure_by_underlying(mock_client):
    mock_client.session.request.return_value = MagicMock(
        status_code=200,
        json=lambda: [
            {"symbol": "AAPL250117C00150000", "asset_class": "option", "qty": "1", "market_value": "450.0", "underlying_symbol": "AAPL"},
            {"symbol": "AAPL250221C00160000", "asset_class": "option", "qty": "1", "market_value": "300.0", "underlying_symbol": "AAPL"},
            {"symbol": "TSLA250117P00200000", "asset_class": "option", "qty": "1", "market_value": "200.0", "underlying_symbol": "TSLA"},
        ],
    )
    pm = PositionManager(mock_client)
    assert pm.option_exposure("AAPL") == 750.0
    assert pm.option_exposure("TSLA") == 200.0


# ---------------------------------------------------------------------------
# run_auto_trade (with mocked client)
# ---------------------------------------------------------------------------

def test_run_auto_trade_dry_run(mock_client):
    """Verify the pipeline runs end-to-end in dry-run mode with mocked client."""
    with patch("earnings_edge.alpaca_bridge.create_client", return_value=mock_client):
        summary = run_auto_trade(dry_run=True)
    assert summary["dry_run"] is True
    assert "timestamp" in summary
    assert "buying_power" in summary
    assert "strategies" in summary
    assert "orders" in summary
    # Should have processed some strategies
    assert len(summary["strategies"]) > 0

def test_run_auto_trade_subset(mock_client):
    """Verify strategy subset selection works."""
    with patch("earnings_edge.alpaca_bridge.create_client", return_value=mock_client):
        summary = run_auto_trade(strategies=["short_straddle"], dry_run=True)
    assert "short_straddle" in summary["strategies"]
    assert len(summary["strategies"]) == 1


# ---------------------------------------------------------------------------
# Symbol resolution (live — Alpaca paper API)
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_account():
    """Live test: test account status (requires APCA_API_KEY_ID env var)."""
    import os
    if not os.environ.get("APCA_API_KEY_ID"):
        pytest.skip("APCA_API_KEY_ID not set")
    client = AlpacaTradingClient()
    acct = client.get_account()
    assert acct["status"] == "ACTIVE"
    assert float(acct["buying_power"]) > 0

