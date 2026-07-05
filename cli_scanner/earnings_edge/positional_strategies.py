"""
Non-calendar positional option strategies.

Unlike calendar-call strategies that sell a near and buy a far leg,
these strategies take outright directional or volatility positions:

  1. ShortStraddle   — sell premium when IV/RV is elevated and model predicts
                       small post-earnings move (overpriced vol).
  2. LongStraddle    — buy premium when model predicts large move that implied
                       vol has underpriced.
  3. DirectionalCall — buy calls on bullish model signal + magnitude filter.
  4. DirectionalPut  — buy puts on bearish model signal + magnitude filter.
  5. VolRiskPremium  — structural short vol when IV/RV is extreme; model
                       provides magnitude confirmation.
"""
from __future__ import annotations

from earnings_edge.strategies import Trade, StrategyResult, DataBundle

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger("positional_strategies")


# ---------------------------------------------------------------------------
# Short Straddle — sell premium when vol is overpriced
# ---------------------------------------------------------------------------

class ShortStraddle:
    """Sell a straddle when:
    - IV/RV ratio >= iv_rv_min (premium is rich)
    - Model predicts magnitude < expected move (low realized vol expected)
    - Expected move is at least min_expected_move (enough premium to collect)

    PnL (percent-of-price): expected_move_pct - |actual_move_pct|
    Positive when realized vol < implied vol.
    """
    name = "short_straddle"

    def __init__(self, iv_rv_min: float = 1.2, min_expected_move: float = 6.0,
                 model_threshold: float = 5.0, model_path: str = "data/models",
                 model_type: str = "gradient_boosting"):
        self.iv_rv_min = iv_rv_min
        self.min_expected_move = min_expected_move
        self.model_threshold = model_threshold
        self.model_path = model_path
        self.model_type = model_type
        self.model = self._load_model()

    def _load_model(self):
        try:
            return joblib.load(f"{self.model_path}/option_model_magnitude_{self.model_type}.joblib")
        except (FileNotFoundError, Exception):
            logger.warning("ShortStraddle: magnitude model unavailable; using filter-only mode")
            return None

    def run(self, data: DataBundle) -> StrategyResult:
        df = data.snapshots.copy()
        if df.empty:
            return StrategyResult(self.name, [])

        actual = "actual_move_pct"
        expected = "expected_move_pct"

        mask = (
            df[actual].notna()
            & df[expected].notna()
            & (df[expected] > 0)
            & df["iv30_rv30"].notna()
            & (df["iv30_rv30"] >= self.iv_rv_min)
            & (df[expected] >= self.min_expected_move)
        )
        filtered = df.loc[mask].copy()

        if filtered.empty:
            return StrategyResult(self.name, [])

        feature_cols = [c for c in ["price","avg_volume_30d","atm_iv_near","rv30","iv30_rv30",
                         "hist_vol_3m","sigma_baseline_1y","sigma_short_leg","sigma_short_leg_fair",
                         "actual_to_fair_ratio","term_slope","term_structure_valid"]
                         if c in filtered.columns]

        trades = []
        for _, row in filtered.iterrows():
            X = row[feature_cols].to_frame().T
            model_score = None

            if self.model is not None:
                try:
                    model_score = float(self.model.predict(X)[0])
                    if model_score >= self.model_threshold:
                        continue
                except Exception:
                    model_score = None

            actual_move = row[actual]
            expected_row = float(row[expected])
            pnl = expected_row - abs(actual_move)

            trades.append(Trade(
                ticker=row["ticker"],
                earnings_date=row["earnings_date"] if isinstance(row["earnings_date"], date) else date.fromisoformat(str(row["earnings_date"])),
                scan_date=row["scan_date"] if isinstance(row["scan_date"], date) else date.fromisoformat(str(row["scan_date"])),
                strategy=self.name,
                side="SHORT_STRADDLE",
                entry_price=abs(expected_row),
                exit_price=abs(actual_move),
                pnl=pnl,
                pnl_pct=pnl,
                features={"iv_rv": row.get("iv30_rv30"), "atm_iv": row.get("atm_iv_near")},
                model_score=model_score,
                ml_decision="TAKE",
                notes=f"ep={expected_row:.2f} am={actual_move:.2f}"
            ))

        if not trades:
            return StrategyResult(self.name, [])

        taken = trades
        total_pnl = sum(t.pnl for t in taken)
        win_rate = sum(1 for t in taken if t.pnl > 0) / len(taken)
        return StrategyResult(self.name, trades, {
            "total": len(trades),
            "taken": len(taken),
            "avg_pnl": np.mean([t.pnl for t in taken]),
            "total_pnl": float(round(total_pnl, 4)),
            "win_rate": win_rate,
            "iv_rv_min": self.iv_rv_min,
        })


# ---------------------------------------------------------------------------
# Long Straddle — buy premium when vol is cheap and model predicts big move
# ---------------------------------------------------------------------------

class LongStraddle:
    """Buy a straddle when:
    - IV/RV ratio <= iv_rv_max (premium is cheap)
    - Model predicts high post-earnings magnitude
    - Expected move is meaningful

    PnL (percent-of-price): |actual_move_pct| - expected_move_pct
    """
    name = "long_straddle"

    def __init__(self, iv_rv_max: float = 1.0, min_expected_move: float = 4.0,
                 model_threshold: float = 7.0, model_path: str = "data/models",
                 model_type: str = "gradient_boosting"):
        self.iv_rv_max = iv_rv_max
        self.min_expected_move = min_expected_move
        self.model_threshold = model_threshold
        self.model_path = model_path
        self.model_type = model_type
        self.model = self._load_model()

    def _load_model(self):
        try:
            return joblib.load(f"{self.model_path}/option_model_magnitude_{self.model_type}.joblib")
        except (FileNotFoundError, Exception):
            logger.warning("LongStraddle: magnitude model unavailable")
            return None

    def run(self, data: DataBundle) -> StrategyResult:
        df = data.snapshots.copy()
        if df.empty:
            return StrategyResult(self.name, [])

        actual = "actual_move_pct"
        expected = "expected_move_pct"

        mask = (
            df[actual].notna()
            & df[expected].notna()
            & (df[expected] > 0)
            & df["iv30_rv30"].notna()
            & (df["iv30_rv30"] <= self.iv_rv_max)
            & (df[expected] >= self.min_expected_move)
        )
        filtered = df.loc[mask].copy()
        if filtered.empty:
            return StrategyResult(self.name, [])

        feature_cols = [c for c in ["price","avg_volume_30d","atm_iv_near","rv30","iv30_rv30",
                         "hist_vol_3m","sigma_baseline_1y","sigma_short_leg","sigma_short_leg_fair",
                         "actual_to_fair_ratio","term_slope","term_structure_valid"]
                         if c in filtered.columns]

        trades = []
        for _, row in filtered.iterrows():
            X = row[feature_cols].to_frame().T
            model_score = None
            if self.model is not None:
                try:
                    model_score = float(self.model.predict(X)[0])
                    if model_score < self.model_threshold:
                        continue
                except Exception:
                    pass

            actual_move = row[actual]
            expected_row = float(row[expected])
            pnl = abs(actual_move) - expected_row

            trades.append(Trade(
                ticker=row["ticker"],
                earnings_date=row["earnings_date"] if isinstance(row["earnings_date"], date) else date.fromisoformat(str(row["earnings_date"])),
                scan_date=row["scan_date"] if isinstance(row["scan_date"], date) else date.fromisoformat(str(row["scan_date"])),
                strategy=self.name,
                side="LONG_STRADDLE",
                entry_price=abs(expected_row),
                exit_price=abs(actual_move),
                pnl=pnl,
                pnl_pct=pnl,
                features={"iv_rv": row.get("iv30_rv30")},
                model_score=model_score,
                ml_decision="TAKE",
                notes=f"ep={expected_row:.2f} am={actual_move:.2f}"
            ))

        if not trades:
            return StrategyResult(self.name, [])

        win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades)
        return StrategyResult(self.name, trades, {
            "total": len(trades),
            "taken": len(trades),
            "avg_pnl": np.mean([t.pnl for t in trades]),
            "total_pnl": round(sum(t.pnl for t in trades), 4),
            "win_rate": win_rate,
            "iv_rv_max": self.iv_rv_max,
        })


# ---------------------------------------------------------------------------
# Directional Call
# ---------------------------------------------------------------------------

class DirectionalCall:
    """Buy a call when direction model predicts UP and magnitude model
    predicts large move.
    PnL: actual_move_pct - expected_move_pct
    """
    name = "directional_call"

    def __init__(self, magnitude_threshold: float = 6.0,
                 iv_rv_max: float = 1.3,
                 model_path: str = "data/models",
                 model_type: str = "gradient_boosting"):
        self.magnitude_threshold = magnitude_threshold
        self.iv_rv_max = iv_rv_max
        self.magnitude_model = self._load_model("magnitude", model_path, model_type)
        self.direction_model = self._load_model("direction", model_path, model_type)

    def _load_model(self, target, model_path, model_type):
        try:
            return joblib.load(f"{model_path}/option_model_{target}_{model_type}.joblib")
        except (FileNotFoundError, Exception):
            logger.warning("DirectionalCall: %s model unavailable", target)
            return None

    def run(self, data: DataBundle) -> StrategyResult:
        df = data.snapshots.copy()
        if df.empty:
            return StrategyResult(self.name, [])

        actual = "actual_move_pct"
        expected = "expected_move_pct"

        mask = (
            df[actual].notna()
            & df[expected].notna()
            & (df["iv30_rv30"].notna())
            & (df["iv30_rv30"] <= self.iv_rv_max)
            & (df[expected] > 0)
        )
        filtered = df.loc[mask].copy()
        if filtered.empty:
            return StrategyResult(self.name, [])

        feature_cols = [c for c in ["price","avg_volume_30d","atm_iv_near","rv30","iv30_rv30",
                         "hist_vol_3m","sigma_baseline_1y","sigma_short_leg","sigma_short_leg_fair",
                         "actual_to_fair_ratio","term_slope","term_structure_valid"]
                         if c in filtered.columns]

        trades = []
        for _, row in filtered.iterrows():
            X = row[feature_cols].to_frame().T

            pred_dir = None
            if self.direction_model is not None:
                try:
                    pred_dir = int(self.direction_model.predict(X)[0])
                except Exception:
                    pass
            if pred_dir != 1:
                continue

            pred_mag = None
            if self.magnitude_model is not None:
                try:
                    pred_mag = float(self.magnitude_model.predict(X)[0])
                except Exception:
                    pass
            if pred_mag is None or pred_mag < self.magnitude_threshold:
                continue

            actual_move = row[actual]
            premium = float(row[expected])
            pnl = actual_move - premium

            trades.append(Trade(
                ticker=row["ticker"],
                earnings_date=row["earnings_date"] if isinstance(row["earnings_date"], date) else date.fromisoformat(str(row["earnings_date"])),
                scan_date=row["scan_date"] if isinstance(row["scan_date"], date) else date.fromisoformat(str(row["scan_date"])),
                strategy=self.name,
                side="LONG_CALL",
                entry_price=abs(premium),
                exit_price=actual_move,
                pnl=pnl,
                pnl_pct=pnl,
                features={"direction": pred_dir, "predicted_magnitude": pred_mag},
                model_score=pred_mag,
                ml_decision="TAKE",
                notes=f"dir=UP mag={pred_mag:.2f} premium={premium:.2f}"
            ))

        if not trades:
            return StrategyResult(self.name, [])

        win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades)
        return StrategyResult(self.name, trades, {
            "total": len(trades),
            "taken": len(trades),
            "avg_pnl": float(np.mean([t.pnl for t in trades])),
            "total_pnl": round(sum(t.pnl for t in trades), 4),
            "win_rate": win_rate,
        })


# ---------------------------------------------------------------------------
# Directional Put
# ---------------------------------------------------------------------------

class DirectionalPut:
    """Buy a put when model predicts DOWN direction and large magnitude."""
    name = "directional_put"

    def __init__(self, magnitude_threshold: float = 6.0,
                 iv_rv_max: float = 1.3,
                 model_path: str = "data/models",
                 model_type: str = "gradient_boosting"):
        self.magnitude_threshold = magnitude_threshold
        self.iv_rv_max = iv_rv_max
        self.magnitude_model = self._load_model("magnitude", model_path, model_type)
        self.direction_model = self._load_model("direction", model_path, model_type)

    def _load_model(self, target, model_path, model_type):
        try:
            return joblib.load(f"{model_path}/option_model_{target}_{model_type}.joblib")
        except (FileNotFoundError, Exception):
            logger.warning("DirectionalPut: %s model unavailable", target)
            return None

    def run(self, data: DataBundle) -> StrategyResult:
        df = data.snapshots.copy()
        if df.empty:
            return StrategyResult(self.name, [])

        actual = "actual_move_pct"
        expected = "expected_move_pct"

        mask = (
            df[actual].notna()
            & df[expected].notna()
            & (df["iv30_rv30"].notna())
            & (df["iv30_rv30"] <= self.iv_rv_max)
            & (df[expected] > 0)
        )
        filtered = df.loc[mask].copy()
        if filtered.empty:
            return StrategyResult(self.name, [])

        feature_cols = [c for c in ["price","avg_volume_30d","atm_iv_near","rv30","iv30_rv30",
                         "hist_vol_3m","sigma_baseline_1y","sigma_short_leg","sigma_short_leg_fair",
                         "actual_to_fair_ratio","term_slope","term_structure_valid"]
                         if c in filtered.columns]

        trades = []
        for _, row in filtered.iterrows():
            X = row[feature_cols].to_frame().T

            pred_dir = None
            if self.direction_model is not None:
                try:
                    pred_dir = int(self.direction_model.predict(X)[0])
                except Exception:
                    pass
            if pred_dir != -1:
                continue

            pred_mag = None
            if self.magnitude_model is not None:
                try:
                    pred_mag = float(self.magnitude_model.predict(X)[0])
                except Exception:
                    pass
            if pred_mag is None or pred_mag < self.magnitude_threshold:
                continue

            actual_move = row[actual]
            premium = float(row[expected])
            pnl = -actual_move - premium

            trades.append(Trade(
                ticker=row["ticker"],
                earnings_date=row["earnings_date"] if isinstance(row["earnings_date"], date) else date.fromisoformat(str(row["earnings_date"])),
                scan_date=row["scan_date"] if isinstance(row["scan_date"], date) else date.fromisoformat(str(row["scan_date"])),
                strategy=self.name,
                side="LONG_PUT",
                entry_price=abs(premium),
                exit_price=-actual_move,
                pnl=pnl,
                pnl_pct=pnl,
                features={"direction": pred_dir, "predicted_magnitude": pred_mag},
                model_score=pred_mag,
                ml_decision="TAKE",
                notes=f"dir=DOWN mag={pred_mag:.2f} premium={premium:.2f}"
            ))

        if not trades:
            return StrategyResult(self.name, [])

        win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades)
        return StrategyResult(self.name, trades, {
            "total": len(trades),
            "taken": len(trades),
            "avg_pnl": float(np.mean([t.pnl for t in trades])),
            "total_pnl": round(sum(t.pnl for t in trades), 4),
            "win_rate": win_rate,
        })


# ---------------------------------------------------------------------------
# Volatility Risk Premium
# ---------------------------------------------------------------------------

class VolRiskPremium:
    """Structural short-vol: sell premium whenever IV/RV is extreme (>= threshold),
    regardless of model. Uses model magnitude only as a tiebreaker.
    """
    name = "vol_risk_premium"

    def __init__(self, iv_rv_min: float = 1.4, min_expected_move: float = 6.0,
                 model_path: str = "data/models",
                 model_type: str = "gradient_boosting"):
        self.iv_rv_min = iv_rv_min
        self.min_expected_move = min_expected_move
        self.model = self._load_model(model_path, model_type)

    def _load_model(self, model_path, model_type):
        try:
            return joblib.load(f"{model_path}/option_model_magnitude_{model_type}.joblib")
        except (FileNotFoundError, Exception):
            logger.warning("VolRiskPremium: magnitude model unavailable")
            return None

    def run(self, data: DataBundle) -> StrategyResult:
        df = data.snapshots.copy()
        if df.empty:
            return StrategyResult(self.name, [])

        actual = "actual_move_pct"
        expected = "expected_move_pct"

        mask = (
            df[actual].notna()
            & df[expected].notna()
            & (df["iv30_rv30"].notna())
            & (df["iv30_rv30"] >= self.iv_rv_min)
            & (df[expected] >= self.min_expected_move)
        )
        filtered = df.loc[mask].copy()

        if filtered.empty:
            return StrategyResult(self.name, [])

        feature_cols = [c for c in ["price","avg_volume_30d","atm_iv_near","rv30","iv30_rv30",
                         "hist_vol_3m","sigma_baseline_1y","sigma_short_leg","sigma_short_leg_fair",
                         "actual_to_fair_ratio","term_slope","term_structure_valid"]
                         if c in filtered.columns]

        trades = []
        for _, row in filtered.iterrows():
            X = row[feature_cols].to_frame().T
            model_score = None
            if self.model is not None:
                try:
                    model_score = float(self.model.predict(X)[0])
                except Exception:
                    pass

            actual_move = row[actual]
            premium = float(row[expected])
            pnl = premium - abs(actual_move)

            trades.append(Trade(
                ticker=row["ticker"],
                earnings_date=row["earnings_date"] if isinstance(row["earnings_date"], date) else date.fromisoformat(str(row["earnings_date"])),
                scan_date=row["scan_date"] if isinstance(row["scan_date"], date) else date.fromisoformat(str(row["scan_date"])),
                strategy=self.name,
                side="SHORT_VOL",
                entry_price=abs(premium),
                exit_price=abs(actual_move),
                pnl=pnl,
                pnl_pct=pnl,
                features={"iv_rv": row.get("iv30_rv30")},
                model_score=model_score,
                ml_decision="TAKE",
                notes=f"iv_rv={row.get('iv30_rv30'):.2f} ep={premium:.2f} am={actual_move:.2f}"
            ))

        if not trades:
            return StrategyResult(self.name, [])

        win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades)
        return StrategyResult(self.name, trades, {
            "total": len(trades),
            "taken": len(trades),
            "avg_pnl": float(np.mean([t.pnl for t in trades])),
            "total_pnl": round(sum(t.pnl for t in trades), 4),
            "win_rate": win_rate,
        })


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

POSITIONAL_STRATEGIES = {
    ShortStraddle.name: ShortStraddle,
    LongStraddle.name: LongStraddle,
    DirectionalCall.name: DirectionalCall,
    DirectionalPut.name: DirectionalPut,
    VolRiskPremium.name: VolRiskPremium,
}


def run_positional(bundle: DataBundle, strategies: Optional[List[str]] = None) -> Dict[str, StrategyResult]:
    results = {}
    selected = strategies or list(POSITIONAL_STRATEGIES.keys())
    for name in selected:
        cls = POSITIONAL_STRATEGIES[name]
        instance = cls()
        results[name] = instance.run(bundle)
    return results
