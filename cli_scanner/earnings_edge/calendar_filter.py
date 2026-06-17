"""Utilities for scoring earnings calendar-call trades with a trained filter model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalendarModelScore:
    """A model score for a single calendar-call candidate."""

    probability: float
    recommended: bool
    reason: str


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def data_quality_rejection_reasons(
    row: Mapping[str, Any],
    *,
    max_moneyness_error: float = 0.20,
    require_exit_value: bool = True,
) -> list[str]:
    """Return hard rejection reasons for corrupted/crossed calendar rows.

    These are data-quality gates, not alpha features. They catch split-adjustment
    mismatches and impossible marks before a model is trained or evaluated.
    """

    reasons: list[str] = []
    price = _to_float(row.get("price"))
    strike = _to_float(row.get("strike"))
    if price is None or price <= 0 or strike is None or strike <= 0:
        reasons.append("missing_price_or_strike")
    else:
        if abs(strike / price - 1.0) > max_moneyness_error:
            reasons.append("bad_moneyness")

    net_debit = _to_float(row.get("net_debit"))
    if net_debit is None or net_debit <= 0:
        reasons.append("non_positive_debit")

    if require_exit_value:
        exit_value = _to_float(row.get("exit_value"))
        if exit_value is None:
            reasons.append("missing_exit_value")
        elif exit_value < 0:
            reasons.append("negative_exit_value")

    return reasons


def add_calendar_entry_features(row: Mapping[str, Any]) -> dict[str, Any]:
    """Add derived pre-trade calendar-entry features to a row-like mapping."""

    features = dict(row)
    price = _to_float(features.get("price"))
    strike = _to_float(features.get("strike"))
    net_debit = _to_float(features.get("net_debit"))
    near_entry = _to_float(features.get("near_entry"))
    far_entry = _to_float(features.get("far_entry"))

    if price and strike:
        moneyness = strike / price
        features["moneyness"] = moneyness
        features["abs_moneyness_error"] = abs(moneyness - 1.0)
    else:
        features["moneyness"] = None
        features["abs_moneyness_error"] = None

    features["debit_pct_price"] = net_debit / price if price and net_debit is not None else None
    features["near_far_entry_ratio"] = near_entry / far_entry if near_entry is not None and far_entry else None

    try:
        near_expiry = pd.to_datetime(features.get("near_expiry"))
        far_expiry = pd.to_datetime(features.get("far_expiry"))
        features["entry_width_days"] = int((far_expiry - near_expiry).days)
    except Exception:
        features["entry_width_days"] = None

    return features


def score_calendar_trade(
    artifact: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    threshold: float = 0.55,
) -> CalendarModelScore:
    """Score one calendar-call candidate with a trained sklearn artifact."""

    features = list(artifact["features"])
    enriched = add_calendar_entry_features(row)
    frame = pd.DataFrame([{feature: enriched.get(feature) for feature in features}], columns=features)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    score_kind = artifact.get("score_kind", "probability")
    if score_kind == "probability":
        model_score = float(artifact["pipeline"].predict_proba(frame)[0][1])
    else:
        model_score = float(artifact["pipeline"].predict(frame)[0])
    recommended = model_score >= threshold
    comparator = ">=" if recommended else "<"
    return CalendarModelScore(
        probability=model_score,
        recommended=recommended,
        reason=f"model_score{comparator}{threshold:.2f}",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
