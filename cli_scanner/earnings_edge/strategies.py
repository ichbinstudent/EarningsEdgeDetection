"""
Trading strategies — backtestable units that emit Trade objects.

Each strategy takes snapshots + (option features) + (calendar_call_trades or outcomes)
and returns a list of Trade objects with fill prices and PnL.

The strategies are intentionally kept thin — they don't touch the DB or network.
All data is passed in via DataBundle so they can be unit-tested without real data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from earnings_edge.calendar_filter import (
    add_calendar_entry_features,
    data_quality_rejection_reasons,
    score_calendar_trade,
)


# ---------------------------------------------------------------------------
# Trade abstraction
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single strategy trade to be backtested or scored live."""
    ticker: str
    earnings_date: date
    scan_date: date
    strategy: str
    side: str                 # LONG, SHORT, SPREAD, CALENDAR
    entry_price: float        # raw cost basis (debit or mid-price or spread)
    exit_price: float = 0.0   # 0 = not yet closed
    pnl: float = 0.0          # absolute PnL (dollars for options, percent for stock)
    pnl_pct: float = 0.0      # return_on_debit for options, simple return for stock
    features: Dict[str, Any] = field(default_factory=dict)
    model_score: Optional[float] = None
    ml_decision: str = "SKIP"
    notes: str = ""

    def is_winner(self) -> bool:
        return self.pnl > 0


# ---------------------------------------------------------------------------
# DataBundle — everything a strategy needs (no DB/network calls)
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    snapshots: pd.DataFrame
    calendar_trades: pd.DataFrame
    live_candidates: pd.DataFrame
    scan_outputs: pd.DataFrame
    options_chain: pd.DataFrame = field(default_factory=pd.DataFrame)

    @classmethod
    def from_db(cls, db_path: str | None = None) -> DataBundle:
        from earnings_edge.db import DEFAULT_DB_PATH, get_connection

        resolved = None if db_path is None else Path(db_path)
        conn = get_connection(resolved)

        snapshots = pd.read_sql("SELECT * FROM snapshots", conn)
        calendar_trades = pd.read_sql("SELECT * FROM calendar_call_trades", conn)
        live_candidates = pd.read_sql("SELECT * FROM live_calendar_candidates", conn)
        scan_outputs = pd.read_sql("SELECT * FROM scanner_scan_outputs", conn)
        options_chain = pd.read_sql("SELECT * FROM options_chain", conn)

        conn.close()
        return cls(
            snapshots=snapshots,
            calendar_trades=calendar_trades,
            live_candidates=live_candidates,
            scan_outputs=scan_outputs,
            options_chain=options_chain,
        )


# ---------------------------------------------------------------------------
# Result of one strategy backtest
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    trades: List[Trade]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "ticker": t.ticker,
                "earnings_date": t.earnings_date,
                "scan_date": t.scan_date,
                "strategy": t.strategy,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "model_score": t.model_score,
                "ml_decision": t.ml_decision,
                "notes": t.notes,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Strategy protocol + registry
# ---------------------------------------------------------------------------

_STRATEGIES: Dict[str, Any] = {}


def register(strategy: Any) -> Any:
    _STRATEGIES[strategy.name] = strategy
    return strategy


def get_strategy(name: str) -> Any:
    if name not in _STRATEGIES:
        raise KeyError(f"Unknown strategy: {name}. Available: {list(_STRATEGIES)}")
    return _STRATEGIES[name]


def list_strategies() -> List[str]:
    return list(_STRATEGIES)


# ---------------------------------------------------------------------------
# Strategy 1: Calendar-call (existing baseline, with ML filter)
# ---------------------------------------------------------------------------

class CalendarCallStrategy:
    """Calendar-call trade with existing ML filter (ridge regression, target=return_on_debit)."""
    name = "calendar_call_ml"

    def __init__(self, model_path: str = "data/models/calendar_call_filter_ridge_allfeatures.joblib",
                 threshold: float = 0.0,
                 min_rows: int = 30):
        self.model_path = model_path
        self.threshold = threshold
        self.min_rows = min_rows

    def run(self, data: DataBundle) -> StrategyResult:
        trades = []
        ct = data.calendar_trades
        if ct.empty:
            return StrategyResult(self.name, trades)

        # Join trades with snapshots on (ticker, earnings_date, scan_date) to avoid
        # 1-to-many inflation when the same (ticker, earnings_date) has multiple scans.
        snap = data.snapshots
        key_cols = ["ticker", "earnings_date", "scan_date"]
        # Fall back to (ticker, earnings_date) if scan_date missing on either side
        if snap["scan_date"].notna().all() and ct["scan_date"].notna().all():
            merge_keys = ["ticker", "earnings_date", "scan_date"]
        else:
            merge_keys = ["ticker", "earnings_date"]

        merged = ct.merge(
            snap,
            left_on=merge_keys,
            right_on=merge_keys,
            how="left",
            suffixes=("_trade", ""),
        )

        # Drop rows missing critical snapshot features
        merged = merged.dropna(subset=["price", "net_debit"])
        if len(merged) < self.min_rows:
            return StrategyResult(self.name, trades)

        # Apply data-quality gates (pure filtering, not model)
        model = self._load_model()
        for _, row in merged.iterrows():
            row_dict = row.to_dict()
            reasons = data_quality_rejection_reasons(row_dict)
            if reasons:
                continue

            enriched = add_calendar_entry_features(row_dict)
            score = None
            model_reason = "no_model"
            if model is not None:
                try:
                    result = score_calendar_trade(
                        artifact={"features": model["features"], "pipeline": model["pipeline"], "score_kind": model.get("score_kind", "regression")},
                        row=enriched,
                        threshold=self.threshold,
                    )
                    score = result.probability
                    model_reason = result.reason
                except Exception as e:
                    model_reason = f"model_error: {e}"

            pnl = float(row.get("pnl_dollars") or 0.0)
            rod = float(row.get("return_on_debit") or 0.0)
            decision = "TAKE" if score is not None and score >= self.threshold else "SKIP"

            trades.append(Trade(
                ticker=row["ticker"],
                earnings_date=pd.to_datetime(row["earnings_date"]).date(),
                scan_date=pd.to_datetime(row["scan_date"]).date() if pd.notna(row.get("scan_date")) else pd.to_datetime(row["earnings_date"]).date(),
                strategy=self.name,
                side="CALENDAR",
                entry_price=row.get("net_debit", 0.0),
                pnl=pnl,
                pnl_pct=rod,
                features={"net_debit": row.get("net_debit"), "price": row.get("price")},
                model_score=score,
                ml_decision=decision,
                notes=model_reason,
            ))

        return StrategyResult(self.name, trades, self._summarize(trades))

    def _apply_quality_gates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "price" in df.columns:
            df["moneyness_err"] = (df["strike"] / df["price"] - 1.0).abs()
            df = df[df["moneyness_err"] <= 0.20]
        df = df[df["net_debit"] > 0]
        df = df[df["exit_value"] >= 0]
        return df

    def _load_model(self):
        import joblib
        from pathlib import Path
        path = Path(self.model_path)
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    def _summarize(self, trades: List[Trade]) -> Dict[str, Any]:
        taken = [t for t in trades if t.ml_decision == "TAKE"]
        return {
            "total": len(trades),
            "taken": len(taken),
            "pnl": sum(t.pnl for t in taken),
            "win_rate": float(np.mean([1 if t.is_winner() else 0 for t in taken])) if taken else 0.0,
        }


# ---------------------------------------------------------------------------
# Strategy 2: Calendar-call (high-conviction ML)
# ---------------------------------------------------------------------------

class CalendarCallHighConviction(CalendarCallStrategy):
    """Calendar-call with strict risk envelope: ML score >= 0.55 AND net_debit <= $2.00."""
    name = "calendar_call_high_conviction"

    def run(self, data: DataBundle) -> StrategyResult:
        base = super().run(data)
        for t in base.trades:
            if t.ml_decision == "TAKE" and (
                t.entry_price > 2.0 or (t.model_score is not None and t.model_score < 0.55)
            ):
                t.ml_decision = "SKIP_RISK"
                t.notes = "risk overlay: high debit or low score"
        base.summary = self._summarize(base.trades)
        base.name = self.name
        return base


# ---------------------------------------------------------------------------
# Strategy 3: Calendar-call (per-quarter, no ML)
# ---------------------------------------------------------------------------

class CalendarCallNoML(CalendarCallStrategy):
    """Calendar-call without ML — pure data-quality gates, equal weight."""
    name = "calendar_call_no_ml"

    def run(self, data: DataBundle) -> StrategyResult:
        base = super().run(data)
        for t in base.trades:
            t.ml_decision = "TAKE"
            t.model_score = None
        base.summary = self._summarize(base.trades)
        base.name = self.name
        return base


# ---------------------------------------------------------------------------
# Strategy 4: Stock drift (PEAD) — buy before earnings, sell after
# ---------------------------------------------------------------------------

class StockDriftStrategy:
    """Drift strategy: long stock 7 days pre-earnings, sell 1 day post. PnL = (post - pre) / pre * 100."""
    name = "stock_drift_pead"

    def __init__(self, pre_days: int = 7, post_days: int = 1,
                 min_price: float = 5.0, min_volume: float = 500_000):
        self.pre_days = pre_days
        self.post_days = post_days
        self.min_price = min_price
        self.min_volume = min_volume

    def run(self, data: DataBundle) -> StrategyResult:
        trades = []
        snap = data.snapshots
        if snap.empty:
            return StrategyResult(self.name, trades)

        snap = snap[snap["price"] >= self.min_price].copy()
        snap = snap[snap["avg_volume_30d"].fillna(0) >= self.min_volume]
        snap = snap[snap["outcome_fetched_at"].notna()]
        snap = snap[snap["pre_earnings_close"].notna()]
        snap = snap[snap["post_earnings_close"].notna()]

        for _, row in snap.iterrows():
            pre_close = row["pre_earnings_close"]
            post_close = row["post_earnings_close"]
            if pre_close <= 0:
                continue
            ret_pct = (post_close - pre_close) / pre_close * 100

            trades.append(Trade(
                ticker=row["ticker"],
                earnings_date=pd.to_datetime(row["earnings_date"]).date(),
                scan_date=pd.to_datetime(row["scan_date"]).date()
                if "scan_date" in row
                else pd.to_datetime(row["earnings_date"]).date(),
                strategy=self.name,
                side="LONG",
                entry_price=pre_close,
                exit_price=post_close,
                pnl=ret_pct,
                pnl_pct=ret_pct,
                features={
                    "pre_earnings_close": pre_close,
                    "post_earnings_close": post_close,
                    "expected_move_pct": row.get("expected_move_pct"),
                    "actual_move_pct": row.get("actual_move_pct"),
                    "timing": row.get("timing", ""),
                },
                ml_decision="TAKE",
            ))

        return StrategyResult(self.name, trades, self._summarize(trades))

    def _summarize(self, trades: List[Trade]) -> Dict[str, Any]:
        return {
            "total": len(trades),
            "avg_return_pct": float(np.mean([t.pnl for t in trades])) if trades else 0.0,
            "win_rate": float(np.mean([t.is_winner() for t in trades])) if trades else 0.0,
            "total_pnl_pct": float(sum(t.pnl for t in trades)),
        }


# ---------------------------------------------------------------------------
# Strategy 5: IV/RV mean-reversion (calendar-call proxy)
# ---------------------------------------------------------------------------

class IVRVMeanReversion(CalendarCallStrategy):
    """Only take calendar-calls where IV/RV > 1.15 (overpriced vol)."""
    name = "iv_rv_mean_reversion"

    def __init__(self, iv_rv_min: float = 1.15, min_rows: int = 30):
        super().__init__(threshold=0.0, min_rows=min_rows)
        self.iv_rv_min = iv_rv_min

    def run(self, data: DataBundle) -> StrategyResult:
        base = super().run(data)
        filtered = []
        for t in base.trades:
            snap = data.snapshots
            match = snap[
                (snap["ticker"] == t.ticker)
                & (snap["earnings_date"] == str(t.earnings_date))
            ]
            ratio = float(match["iv30_rv30"].iloc[0]) if not match.empty else None
            if ratio is not None and ratio >= self.iv_rv_min:
                t.features["iv30_rv30"] = ratio
                filtered.append(t)
        base.trades = filtered
        base.name = self.name
        base.summary = self._summarize(filtered)
        return base


# ---------------------------------------------------------------------------
# Strategy 6: Term-structure steepener
# ---------------------------------------------------------------------------

class TermStructureSteepener(CalendarCallStrategy):
    """Calendar-calls only where term_slope < -0.03 (downward-sloping IV curve)."""
    name = "term_structure_steepener"

    def __init__(self, term_slope_max: float = -0.03, min_rows: int = 30):
        super().__init__(threshold=0.0, min_rows=min_rows)
        self.term_slope_max = term_slope_max

    def run(self, data: DataBundle) -> StrategyResult:
        base = super().run(data)
        filtered = []
        for t in base.trades:
            snap = data.snapshots
            match = snap[
                (snap["ticker"] == t.ticker)
                & (snap["earnings_date"] == str(t.earnings_date))
            ]
            slope = float(match["term_slope"].iloc[0]) if not match.empty else None
            if slope is not None and slope <= self.term_slope_max:
                t.features["term_slope"] = slope
                filtered.append(t)
        base.trades = filtered
        base.name = self.name
        base.summary = self._summarize(filtered)
        return base


# ---------------------------------------------------------------------------
# Strategy 7: DAX Forward Volatility (data-gathering strategy)
# ---------------------------------------------------------------------------

class DAXForwardVolStrategy:
    """Placeholder: needs daily Eurex IV data collection via forward_volatility.py."""

    name = "dax_forward_vol"

    def run(self, data: DataBundle) -> StrategyResult:
        return StrategyResult(self.name, [], {
            "total": 0,
            "note": "Requires daily Eurex IV collection. Run: .venv/bin/python3.12 forward_volatility.py",
            "data_needed": "Eurex DAX options bid/ask IV by expiry, collected daily at 10:00 CET",
        })


# ---------------------------------------------------------------------------
# Strategy 8: Short straddle / iron fly (data-gathering strategy)
# ---------------------------------------------------------------------------

class ShortStraddleStrategy:
    """Placeholder: needs historical straddle pricing (bid/ask at multiple strikes)."""

    name = "short_straddle"

    def run(self, data: DataBundle) -> StrategyResult:
        return StrategyResult(self.name, [], {
            "total": 0,
            "note": "Requires multi-strike options chain data. Set up scheduler for daily collection.",
            "data_needed": "Near-ATM straddle bid/ask prices for liquid equities, collected at market close",
        })


# ---------------------------------------------------------------------------
# Strategy 9: Earnings-quality (beat/miss surprise)
# ---------------------------------------------------------------------------

class EarningsQualityStrategy:
    """Long/short based on actual move vs expected move (surprise)."""

    name = "earnings_quality"

    def __init__(self, threshold_pct: float = 5.0):
        self.threshold_pct = threshold_pct

    def run(self, data: DataBundle) -> StrategyResult:
        trades = []
        snap = data.snapshots
        if snap.empty:
            return StrategyResult(self.name, trades)

        snap = snap[snap["actual_move_pct"].notna()]
        for _, row in snap.iterrows():
            actual = row["actual_move_pct"]
            if abs(actual) > self.threshold_pct:
                direction = "UP" if actual > 0 else "DOWN"
                trades.append(Trade(
                    ticker=row["ticker"],
                    earnings_date=pd.to_datetime(row["earnings_date"]).date(),
                    scan_date=pd.to_datetime(row["scan_date"]).date()
                    if "scan_date" in row
                    else pd.to_datetime(row["earnings_date"]).date(),
                    strategy=self.name,
                    side="LONG_STOCK" if direction == "UP" else "SHORT_STOCK",
                    entry_price=row.get("pre_earnings_close", 0.0),
                    exit_price=row.get("post_earnings_close", 0.0),
                    pnl=actual,
                    pnl_pct=actual,
                    features={"direction": direction, "actual_move_pct": actual},
                    ml_decision="TAKE",
                ))

        return StrategyResult(self.name, trades, self._summarize(trades))

    def _summarize(self, trades: List[Trade]) -> Dict[str, Any]:
        return {
            "total": len(trades),
            "avg_return_pct": float(np.mean([t.pnl for t in trades])) if trades else 0.0,
            "win_rate": float(np.mean([t.is_winner() for t in trades])) if trades else 0.0,
            "total_pnl_pct": float(sum(t.pnl for t in trades)),
            "long_trades": len([t for t in trades if t.side == "LONG_STOCK"]),
            "short_trades": len([t for t in trades if t.side == "SHORT_STOCK"]),
        }


# ---------------------------------------------------------------------------
# Strategy 10: Debit-size exploit
# ---------------------------------------------------------------------------

class DebitSizeExploit(CalendarCallStrategy):
    """Calendar-calls only where debit_pct_price <= 0.03 (cheap relative to stock price)."""

    name = "debit_size_exploit"

    def __init__(self, max_debit_pct: float = 0.03, min_rows: int = 30):
        super().__init__(threshold=0.0, min_rows=min_rows)
        self.max_debit_pct = max_debit_pct

    def run(self, data: DataBundle) -> StrategyResult:
        base = super().run(data)
        filtered = []
        for t in base.trades:
            if "price" in t.features and t.features["price"] > 0:
                pct = t.entry_price / t.features["price"]
                if pct <= self.max_debit_pct:
                    t.features["debit_pct_price"] = pct
                    filtered.append(t)
        base.trades = filtered
        base.name = self.name
        base.summary = self._summarize(filtered)
        return base


# ---------------------------------------------------------------------------
# Register all strategies
# ---------------------------------------------------------------------------

def _register_all() -> None:
    """Register the built-in strategy set (call once at import time)."""
    register(CalendarCallStrategy())
    register(CalendarCallHighConviction())
    register(CalendarCallNoML())
    register(StockDriftStrategy())
    register(IVRVMeanReversion())
    register(TermStructureSteepener())
    register(DAXForwardVolStrategy())
    register(ShortStraddleStrategy())
    register(EarningsQualityStrategy())
    register(DebitSizeExploit())


_register_all()
