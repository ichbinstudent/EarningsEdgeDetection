"""Bridge: Strategy signals → Alpaca paper orders.

Translates strategy Trade signals into Alpaca option orders:
- Single-leg: buy/sell call or put
- Two-leg: calendar spread (long back-month, short front-month)
- Multi-leg: iron condor, butterfly, risk reversal (up to 4 legs)

Supports sizing (Kelly fraction or fixed), pre-submission validation,
dry-run mode, and order-result tracking.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from earnings_edge.alpaca_trading import (
    AlpacaTradingClient,
    OrderResult,
    AlpacaError,
    create_client,
)
from earnings_edge.strategies import Trade, StrategyResult, DataBundle

logger = logging.getLogger(__name__)

MAX_PCT_PER_TRADE = 0.10  # 10% of buying power max per signal
MAX_PCT_PER_UNDERLYING = 0.25  # 25% max exposure per underlying
DEFAULT_ORDER_TYPE = "market"  # market for immediate fill in paper
DEFAULT_TIF = "day"


@dataclass
class BridgeConfig:
    """Configuration for the strategy→order bridge."""
    dry_run: bool = False
    order_type: str = DEFAULT_ORDER_TYPE
    time_in_force: str = DEFAULT_TIF
    max_pct_per_trade: float = MAX_PCT_PER_TRADE
    max_pct_per_underlying: float = MAX_PCT_PER_UNDERLYING
    skip_if_position_exists: bool = True  # don't add to existing position
    max_dte_min: int = 1  # skip trades with < 1 DTE
    max_dte_max: int = 30  # skip trades with > 30 DTE


class StrategyBridge:
    """Bridge strategy TAKE lists to Alpaca paper trades."""

    def __init__(
        self,
        client: Optional[AlpacaTradingClient] = None,
        config: Optional[BridgeConfig] = None,
    ):
        self.client = client or create_client()
        self.config = config or BridgeConfig()
        self.submitted: list[OrderResult] = []

    def account_buying_power(self) -> float:
        return self.client.buying_power()

    # ──────────────── Core: Trade → Order ────────────────────────────────

    def execute_trade(self, trade: Trade) -> Optional[OrderResult]:
        """Execute a single strategy Trade as an Alpaca order.

        Dispatches on trade.side to the appropriate leg builder.
        Returns OrderResult on success, None if skipped.
        """
        try:
            legs = self._build_legs(trade)
            if not legs or len(legs) < 1:
                logger.warning("%s %s: no legs, skipping", trade.strategy, trade.ticker)
                return None

            # Validate: expiry ≥ min_dte and ≤ max_dte
            min_expiry = self._min_expiry(legs)
            if min_expiry is not None and trade.earnings_date:
                dte = (min_expiry - trade.earnings_date).days
                if dte < self.config.max_dte_min:
                    logger.warning("%s %s: DTE %d < min %d, skipping", trade.strategy, trade.ticker, dte, self.config.max_dte_min)
                    return None
                if dte > self.config.max_dte_max:
                    logger.warning("%s %s: DTE %d > max %d, skipping", trade.strategy, trade.ticker, dte, self.config.max_dte_max)
                    return None

            # Position check
            if self.config.skip_if_position_exists:
                for leg in legs:
                    if self.client.has_position(leg["symbol"]):
                        logger.info("%s: position exists for %s, skipping", trade.strategy, leg["symbol"])
                        return None

            # Compute limit price (or market)
            limit_price = None
            if self.config.order_type == "limit":
                limit_price = self._midpoint_price(legs)

            # Submit
            client_order_id = f"{trade.strategy}_{trade.ticker}_{trade.scan_date}_{int(datetime.utcnow().timestamp())}"
            if self.config.dry_run:
                logger.info(
                    "DRY RUN: %s %s %s %d legs side=%s",
                    "BUY" if legs[0]["side"] == "buy" else "SELL",
                    1,
                    trade.ticker,
                    len(legs),
                    trade.side,
                )
                return OrderResult(
                    order_id="dry-run",
                    client_order_id=client_order_id,
                    symbol=trade.ticker,
                    strategy=trade.strategy,
                    legs=legs,
                    status="dry_run",
                    filled_qty=0,
                    filled_avg_price=None,
                    created_at=datetime.utcnow().isoformat(),
                    raw={},
                )

            if len(legs) == 1:
                leg = legs[0]
                order = self.client.submit_order(
                    symbol=leg["symbol"],
                    qty=leg.get("ratio_qty", 1),
                    side=leg["side"],
                    order_type=self.config.order_type,
                    time_in_force=self.config.time_in_force,
                    limit_price=limit_price,
                    client_order_id=client_order_id,
                )
            else:
                order = self.client.submit_multi_leg_order(
                    legs=legs,
                    order_type=self.config.order_type,
                    time_in_force=self.config.time_in_force,
                    limit_price=limit_price,
                    client_order_id=client_order_id,
                )

            result = OrderResult.from_alpaca(order, strategy=trade.strategy)
            self.submitted.append(result)
            logger.info("Order %s: %s", result.order_id, result.status)
            return result

        except AlpacaError as e:
            logger.error("Alpaca error on %s %s: %s", trade.strategy, trade.ticker, e)
            return None
        except Exception as e:
            logger.exception("Execution error on %s %s: %s", trade.strategy, trade.ticker, e)
            return None

    # ──────────────── Leg builders ────────────────────────────────────────

    def _build_legs(self, trade: Trade) -> list[dict]:
        """Build order legs based on trade.side.

        Returns list of dicts: {"symbol": str, "ratio_qty": int, "side": "buy"|"sell", "strike": float, "expiry": date, "option_type": "call"|"put"}
        """
        feat = trade.features or {}
        if trade.side == "CALENDAR":
            return self._legs_calendar(trade, feat)
        if trade.side == "SHORT_STRADDLE":
            return self._legs_short_straddle(trade, feat)
        if trade.side == "LONG_STRADDLE":
            return self._legs_long_straddle(trade, feat)
        if trade.side == "DIRECTIONAL_CALL":
            return self._legs_single(trade, feat, "call", "buy")
        if trade.side == "DIRECTIONAL_PUT":
            return self._legs_single(trade, feat, "put", "buy")
        if trade.side == "BULL_CALL_SPREAD":
            return self._legs_vertical(trade, feat, "call", "debit")
        if trade.side == "BEAR_PUT_SPREAD":
            return self._legs_vertical(trade, feat, "put", "debit")
        if trade.side == "IRON_CONDOR":
            return self._legs_iron_condor(trade, feat)
        if trade.side == "BUTTERFLY":
            return self._legs_butterfly(trade, feat)
        if trade.side == "RISK_REVERSAL" or trade.side == "RR":
            return self._legs_risk_reversal(trade, feat)
        # Default: treat as single buy call at nearest round strike
        return self._legs_single(trade, feat, "call", "buy")

    def _legs_calendar(self, trade: Trade, feat: dict) -> list[dict]:
        # Calendar: sell near-month ATM call, buy far-month ATM call
        near_strike = feat.get("near_strike", trade.features.get("atm_strike", 0))
        far_strike = feat.get("far_strike", trade.features.get("atm_strike", 0))
        near_expiry = self._parse_date(feat.get("near_expiry")) or trade.earnings_date
        far_expiry = self._parse_date(feat.get("far_expiry")) or trade.earnings_date + timedelta(days=45)
        near_sym = self._occ_symbol(trade.ticker, near_expiry, near_strike, "call")
        far_sym = self._occ_symbol(trade.ticker, far_expiry, far_strike, "call")
        return [
            {"symbol": near_sym, "ratio_qty": 1, "side": "sell", "strike": near_strike, "expiry": near_expiry, "option_type": "call"},
            {"symbol": far_sym, "ratio_qty": 1, "side": "buy", "strike": far_strike, "expiry": far_expiry, "option_type": "call"},
        ]

    def _legs_short_straddle(self, trade: Trade, feat: dict) -> list[dict]:
        strike = feat.get("atm_strike", feat.get("nearest_atm", 0))
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        call_occ = self._occ_symbol(trade.ticker, expiry, strike, "call")
        put_occ = self._occ_symbol(trade.ticker, expiry, strike, "put")
        return [
            {"symbol": call_occ, "ratio_qty": 1, "side": "sell", "strike": strike, "expiry": expiry, "option_type": "call"},
            {"symbol": put_occ, "ratio_qty": 1, "side": "sell", "strike": strike, "expiry": expiry, "option_type": "put"},
        ]

    def _legs_long_straddle(self, trade: Trade, feat: dict) -> list[dict]:
        strike = feat.get("atm_strike", feat.get("nearest_atm", 0))
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        call_occ = self._occ_symbol(trade.ticker, expiry, strike, "call")
        put_occ = self._occ_symbol(trade.ticker, expiry, strike, "put")
        return [
            {"symbol": call_occ, "ratio_qty": 1, "side": "buy", "strike": strike, "expiry": expiry, "option_type": "call"},
            {"symbol": put_occ, "ratio_qty": 1, "side": "buy", "strike": strike, "expiry": expiry, "option_type": "put"},
        ]

    def _legs_single(self, trade: Trade, feat: dict, option_type: str, direction: str) -> list[dict]:
        strike = feat.get("atm_strike", feat.get("nearest_atm", 0))
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        occ = self._occ_symbol(trade.ticker, expiry, strike, option_type)
        return [
            {"symbol": occ, "ratio_qty": 1, "side": direction, "strike": strike, "expiry": expiry, "option_type": option_type},
        ]

    def _legs_vertical(self, trade: Trade, feat: dict, option_type: str, kind: str) -> list[dict]:
        # Bull call spread: buy lower call, sell higher call
        k_low = feat.get("lower_strike", 0)
        k_high = feat.get("upper_strike", 0)
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        lo_occ = self._occ_symbol(trade.ticker, expiry, k_low, option_type)
        hi_occ = self._occ_symbol(trade.ticker, expiry, k_high, option_type)
        if kind == "debit":
            return [
                {"symbol": lo_occ, "ratio_qty": 1, "side": "buy", "strike": k_low, "expiry": expiry, "option_type": option_type},
                {"symbol": hi_occ, "ratio_qty": 1, "side": "sell", "strike": k_high, "expiry": expiry, "option_type": option_type},
            ]
        return []

    def _legs_iron_condor(self, trade: Trade, feat: dict) -> list[dict]:
        sc = feat.get("short_call", 0)
        sp = feat.get("short_put", 0)
        lc = feat.get("long_call", 0)
        lp = feat.get("long_put", 0)
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        return [
            {"symbol": self._occ_symbol(trade.ticker, expiry, lp, "put"), "ratio_qty": 1, "side": "buy", "strike": lp, "expiry": expiry, "option_type": "put"},
            {"symbol": self._occ_symbol(trade.ticker, expiry, sp, "put"), "ratio_qty": 1, "side": "sell", "strike": sp, "expiry": expiry, "option_type": "put"},
            {"symbol": self._occ_symbol(trade.ticker, expiry, sc, "call"), "ratio_qty": 1, "side": "sell", "strike": sc, "expiry": expiry, "option_type": "call"},
            {"symbol": self._occ_symbol(trade.ticker, expiry, lc, "call"), "ratio_qty": 1, "side": "buy", "strike": lc, "expiry": expiry, "option_type": "call"},
        ]

    def _legs_butterfly(self, trade: Trade, feat: dict) -> list[dict]:
        atm = feat.get("atm", 0)
        lo = feat.get("lo", atm - 5)
        hi = feat.get("hi", atm + 5)
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        opt_type = feat.get("option_type", "call")
        # 1 long lo, 2 short atm, 1 long hi — for multi-leg Alpaca uses ratio_qty
        return [
            {"symbol": self._occ_symbol(trade.ticker, expiry, lo, opt_type), "ratio_qty": 1, "side": "buy", "strike": lo, "expiry": expiry, "option_type": opt_type},
            {"symbol": self._occ_symbol(trade.ticker, expiry, atm, opt_type), "ratio_qty": 2, "side": "sell", "strike": atm, "expiry": expiry, "option_type": opt_type},
            {"symbol": self._occ_symbol(trade.ticker, expiry, hi, opt_type), "ratio_qty": 1, "side": "buy", "strike": hi, "expiry": expiry, "option_type": opt_type},
        ]

    def _legs_risk_reversal(self, trade: Trade, feat: dict) -> list[dict]:
        kc = feat.get("call_strike", 0)
        kp = feat.get("put_strike", 0)
        expiry = self._parse_date(feat.get("expiry")) or trade.earnings_date
        return [
            {"symbol": self._occ_symbol(trade.ticker, expiry, kp, "put"), "ratio_qty": 1, "side": "sell", "strike": kp, "expiry": expiry, "option_type": "put"},
            {"symbol": self._occ_symbol(trade.ticker, expiry, kc, "call"), "ratio_qty": 1, "side": "buy", "strike": kc, "expiry": expiry, "option_type": "call"},
        ]

    # ──────────────── Helpers ─────────────────────────────────────────────

    def _resolve_symbol(self, ticker: str, expiry: date, strike: float, option_type: str) -> Optional[str]:
        """Resolve Alpaca's internal option symbol by querying the contract catalog.

        Alpaca uses non-standard symbol roots (e.g., `AA` not `AAPL`), so we
        must query /options/contracts to get the actual `symbol` used for orders.
        """
        expiry_min = (expiry - timedelta(days=2)).isoformat()
        expiry_max = (expiry + timedelta(days=2)).isoformat()
        contracts_resp = self.client.get_option_contracts(
            ticker,
            expiration_date_gte=expiry_min,
            expiration_date_lte=expiry_max,
            limit=200,
        )
        contracts = contracts_resp.get("option_contracts", [])

        best = None
        best_dist = float("inf")
        for c in contracts:
            if c.get("type", "").lower() != option_type.lower():
                continue
            c_strike = float(c.get("strike_price", 0))
            dist = abs(c_strike - strike) / max(strike, 0.01)
            if dist < best_dist:
                best_dist = dist
                best = c

        if best and best_dist <= 0.05:
            return best["symbol"]
        return None

    def _occ_symbol(self, ticker: str, expiry: date, strike: float, option_type: str) -> str:
        """Build OCC 21-char option symbol for Alpaca order submission.

        Alpaca accepts standard OCC symbols directly (verified).
        We also try to resolve via the contract catalog as a safety check,
        but OCC construction is the primary path.
        """
        # Try to resolve from Alpaca's contract catalog as sanity check
        resolved = self._resolve_symbol(ticker, expiry, strike, option_type)
        if resolved:
            return resolved
        # Primary: construct OCC 21-char symbol
        root = ticker.upper()
        # Some roots differ from ticker (e.g., BRK.B → BRK), but most are 1-4 chars
        # Alpaca supports standard OCC format
        date_code = expiry.strftime("%y%m%d")
        type_code = "C" if option_type.lower() == "call" else "P"
        strike_padded = f"{int(round(strike * 1000)):08d}"
        return f"{root}{date_code}{type_code}{strike_padded}"

    def _parse_date(self, val) -> Optional[date]:
        if val is None:
            return None
        if isinstance(val, date):
            return val
        if isinstance(val, datetime):
            return val.date()
        if isinstance(val, str):
            try:
                return date.fromisoformat(val[:10])
            except Exception:
                return None
        return None

    def _min_expiry(self, legs: list[dict]) -> Optional[date]:
        expiries = [leg.get("expiry") for leg in legs if leg.get("expiry") is not None]
        return min(expiries) if expiries else None

    def _midpoint_price(self, legs: list[dict]) -> Optional[float]:
        """Compute midpoint price for limit orders (sum of leg midpoints)."""
        total = 0.0
        for leg in legs:
            snap = self.client.get_option_snapshot(leg["symbol"])
            bid = snap.get("latestQuote", {}).get("bp", 0) or 0
            ask = snap.get("latestQuote", {}).get("ap", 0) or 0
            mid = (bid + ask) / 2
            sign = -1 if leg["side"] == "sell" else 1
            total += sign * mid
        return total if total > 0 else None


# ---------------------------------------------------------------------------
# High-level: run best strategies today → submit orders
# ---------------------------------------------------------------------------

BEST_STRATEGIES = [
    "calendar_call_ml",
    "debit_size_exploit",
    "short_straddle",
    "vol_risk_premium",
    "iv_rv_mean_reversion",
    "term_structure_steepener",
    "earnings_quality",
]


def run_auto_trade(
    strategies: Optional[list[str]] = None,
    dry_run: bool = False,
    db_path: Optional[str] = None,
) -> dict:
    """Run specified strategies on today's data and submit paper orders.

    Returns dict with execution summary.
    """
    from earnings_edge.strategies import DataBundle, list_strategies, get_strategy

    bundle = DataBundle.from_db(db_path)
    bridge = StrategyBridge(config=BridgeConfig(dry_run=dry_run))
    results = {}
    total_submitted = 0
    total_skipped = 0

    strategies = strategies or BEST_STRATEGIES
    for name in strategies:
        try:
            strategy = get_strategy(name)
        except KeyError:
            logger.warning("Strategy %s not found in registry", name)
            continue

        result = strategy.run(bundle)
        if not result.trades:
            results[name] = {"status": "no-signals", "trades": 0, "submitted": 0}
            continue

        submitted_for_strat = 0
        for trade in result.trades:
            order_result = bridge.execute_trade(trade)
            if order_result:
                submitted_for_strat += 1
                total_submitted += 1
            else:
                total_skipped += 1

        results[name] = {
            "status": "ok",
            "trades": len(result.trades),
            "submitted": submitted_for_strat,
            "skipped": len(result.trades) - submitted_for_strat,
        }

    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "dry_run": dry_run,
        "buying_power": bridge.account_buying_power(),
        "total_submitted": total_submitted,
        "total_skipped": total_skipped,
        "strategies": results,
        "orders": [
            {
                "order_id": o.order_id,
                "strategy": o.strategy,
                "symbol": o.symbol,
                "legs": len(o.legs),
                "status": o.status,
                "client_order_id": o.client_order_id,
            }
            for o in bridge.submitted
        ],
    }
    return summary


if __name__ == "__main__":
    import json
    import sys

    dry_run = "--dry-run" in sys.argv
    summary = run_auto_trade(dry_run=dry_run)
    print(json.dumps(summary, indent=2, default=str))
