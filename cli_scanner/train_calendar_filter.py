#!/usr/bin/env python3
"""Train a simple filter model for earnings calendar-call trades.

This trains on pre-trade features only, with PnL/return used only as labels.
It deliberately keeps hard data-quality gates separate from the model because ML
should not be asked to learn split-adjustment/crossed-market artifacts.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DB_PATH = Path("data/earnings_ml.db")
DEFAULT_OUTPUT = Path("data/models/calendar_call_filter.joblib")

# Columns that must never be used as features (identifiers, targets, leakage).
_LEAKY_COLUMNS = frozenset({
    "id", "snapshot_id", "ticker", "earnings_date", "scan_date",
    "near_expiry", "far_expiry", "near_call_ticker", "far_call_ticker",
    "near_exit", "far_exit",
    "pnl_dollars", "return_on_debit", "exit_value",
    "model_score", "model_recommendation", "model_reason", "model_name",
    "model_scored_at", "created_at",
    "actual_move_pct", "actual_move_direction", "max_intraday_range_pct",
    "pre_earnings_close", "post_earnings_close",
    "collection_error", "mc_source", "event_source",
    "outcome_attempt_count", "outcome_fetched_at", "recommendation", "timing",
})


def discover_features(df: pd.DataFrame) -> list[str]:
    """Auto-discover numeric features from the dataframe, excluding leaky columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return sorted(
        c for c in numeric_cols
        if c not in _LEAKY_COLUMNS and c != "target" and df[c].notna().any()
    )


def load_calendar_trades(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
    select c.snapshot_id, c.ticker, c.earnings_date, c.scan_date, c.near_expiry, c.far_expiry,
           c.strike, c.near_entry, c.far_entry, c.net_debit, c.pnl_dollars,
           c.return_on_debit, c.exit_value,
           s.price, s.avg_volume_30d, s.market_cap, s.has_options, s.days_to_expiry,
           s.total_open_interest, s.atm_iv_near, s.rv30, s.iv30_rv30, s.hist_vol_3m,
           s.term_slope, s.term_structure_valid, s.expected_move_pct, s.expected_move_dollars,
           s.straddle_price, s.atm_call_delta, s.atm_put_delta, s.atm_call_iv, s.atm_put_iv,
           s.sigma_baseline_1y, s.sigma_short_leg, s.sigma_short_leg_fair,
           s.actual_to_fair_ratio, s.mc_win_rate, s.mc_quarters
    from calendar_call_trades c
    left join snapshots s on s.id = c.snapshot_id
    """
    df = pd.read_sql_query(query, conn, parse_dates=["earnings_date", "scan_date"])
    conn.close()
    if df.empty:
        return df

    df["moneyness"] = df["strike"] / df["price"]
    df["abs_moneyness_error"] = (df["moneyness"] - 1.0).abs()
    df["debit_pct_price"] = df["net_debit"] / df["price"]
    df["near_far_entry_ratio"] = df["near_entry"] / df["far_entry"].replace(0, np.nan)
    df["entry_width_days"] = (
        pd.to_datetime(df["far_expiry"]) - pd.to_datetime(df["near_expiry"])
    ).dt.days
    return df.sort_values("earnings_date").reset_index(drop=True)


def apply_data_quality_gates(df: pd.DataFrame, max_moneyness_error: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = {
        "raw_rows": int(len(df)),
        "raw_pnl": float(df["pnl_dollars"].sum()) if not df.empty else 0.0,
    }
    if df.empty:
        return df, raw

    reject_bad_moneyness = df["abs_moneyness_error"] > max_moneyness_error
    reject_non_positive_debit = df["net_debit"] <= 0
    reject_negative_exit_value = df["exit_value"] < 0
    reject = reject_bad_moneyness | reject_non_positive_debit | reject_negative_exit_value
    clean = df[~reject].copy()
    raw.update(
        {
            "rejected_rows": int(reject.sum()),
            "rejected_bad_moneyness": int(reject_bad_moneyness.sum()),
            "rejected_non_positive_debit": int(reject_non_positive_debit.sum()),
            "rejected_negative_exit_value": int(reject_negative_exit_value.sum()),
            "clean_rows": int(len(clean)),
            "clean_pnl": float(clean["pnl_dollars"].sum()),
            "clean_win_rate": float((clean["pnl_dollars"] > 0).mean()) if len(clean) else None,
        }
    )
    return clean, raw


def is_regression_target(target: str) -> bool:
    return target in {"expected_return", "expected_pnl"}


def make_target(df: pd.DataFrame, target: str, min_pnl: float, min_return: float) -> pd.Series:
    if target == "win":
        return (df["pnl_dollars"] > 0).astype(int)
    if target == "min_pnl":
        return (df["pnl_dollars"] >= min_pnl).astype(int)
    if target == "min_return":
        return (df["return_on_debit"] >= min_return).astype(int)
    if target == "expected_return":
        return df["return_on_debit"].astype(float)
    if target == "expected_pnl":
        return df["pnl_dollars"].astype(float)
    raise ValueError(f"Unsupported target: {target}")


def build_model_pipeline(
    features: list[str],
    *,
    target: str,
    model_name: str,
    random_state: int,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    pre = ColumnTransformer([("num", numeric_pipeline, features)])
    if is_regression_target(target):
        if model_name == "ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "random_forest_regressor":
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=10,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Model {model_name!r} is not valid for regression target {target!r}")
    else:
        if model_name == "logistic":
            model = LogisticRegression(max_iter=2000, class_weight="balanced")
        elif model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=10,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Model {model_name!r} is not valid for classification target {target!r}")
    return Pipeline([("preprocess", pre), ("model", model)])


def summarize_selection(name: str, df: pd.DataFrame, mask: np.ndarray) -> dict[str, Any] | None:
    selected = df[mask]
    if selected.empty:
        return None
    return {
        "name": name,
        "n": int(len(selected)),
        "coverage": float(len(selected) / len(df)),
        "pnl": float(selected["pnl_dollars"].sum()),
        "avg_pnl": float(selected["pnl_dollars"].mean()),
        "median_pnl": float(selected["pnl_dollars"].median()),
        "win_rate": float((selected["pnl_dollars"] > 0).mean()),
        "avg_return_on_debit": float(selected["return_on_debit"].mean()),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    df = load_calendar_trades(args.db)
    clean, quality = apply_data_quality_gates(df, args.max_moneyness_error)
    if args.min_iv_rv is not None:
        before = len(clean)
        clean = clean[clean["iv30_rv30"] >= args.min_iv_rv].copy()
        quality["iv_rv_filter"] = {
            "min_iv_rv": args.min_iv_rv,
            "before": before,
            "after": len(clean),
        }
    if len(clean) < args.min_rows:
        raise RuntimeError(f"Not enough clean rows: {len(clean)} < {args.min_rows}")

    clean["target"] = make_target(clean, args.target, args.min_pnl, args.min_return)
    regression = is_regression_target(args.target)
    if regression:
        if clean["target"].nunique() < 2:
            raise RuntimeError("Regression target has no variation after cleaning")
    elif clean["target"].nunique() < 2:
        raise RuntimeError("Target has only one class after cleaning")

    features = discover_features(clean)
    X = clean[features].apply(pd.to_numeric, errors="coerce")
    y = clean["target"]
    if not regression:
        y = y.astype(int)

    pipe = build_model_pipeline(
        features,
        target=args.target,
        model_name=args.model,
        random_state=args.random_state,
    )

    # Proper time holdout: train old events, test future events.
    cut = pd.Timestamp(args.holdout_start)
    train_mask = clean["earnings_date"] < cut
    test_mask = clean["earnings_date"] >= cut
    train_df = clean[train_mask]
    test_df = clean[test_mask]
    if len(train_df) < 30 or len(test_df) < 20:
        raise RuntimeError(
            f"Holdout split too small: train={len(train_df)}, test={len(test_df)}. "
            "Adjust --holdout-start."
        )

    pipe.fit(train_df[features].apply(pd.to_numeric, errors="coerce"), train_df["target"])
    if regression:
        test_score = pipe.predict(test_df[features].apply(pd.to_numeric, errors="coerce"))
        holdout_metric_name = "r2"
        holdout_metric = r2_score(test_df["target"], test_score)
        holdout_mae = mean_absolute_error(test_df["target"], test_score)
    else:
        test_score = pipe.predict_proba(test_df[features].apply(pd.to_numeric, errors="coerce"))[:, 1]
        holdout_metric_name = "auc"
        holdout_metric = roc_auc_score(test_df["target"], test_score) if test_df["target"].nunique() == 2 else None
        holdout_mae = None

    selection_masks: list[tuple[str, np.ndarray]] = [
        ("baseline_all", np.ones(len(test_df), dtype=bool)),
        ("manual_debit_le_2", (test_df["net_debit"] <= 2).to_numpy()),
        (
            "manual_debit_0.25_to_1.50",
            ((test_df["net_debit"] >= 0.25) & (test_df["net_debit"] <= 1.5)).to_numpy(),
        ),
        ("model_top_25pct", test_score >= np.quantile(test_score, 0.75)),
        ("model_top_50pct", test_score >= np.quantile(test_score, 0.50)),
    ]
    if regression:
        selection_masks.extend(
            [
                ("model_return_ge_0.10", test_score >= 0.10),
                ("model_return_ge_0.20", test_score >= 0.20),
            ]
        )
    else:
        selection_masks.extend(
            [
                ("model_p_ge_0.55", test_score >= 0.55),
                ("model_p_ge_0.60", test_score >= 0.60),
            ]
        )

    holdout_results: list[dict[str, Any]] = []
    for name, mask in selection_masks:
        summary = summarize_selection(name, test_df, mask)
        if summary:
            holdout_results.append(summary)

    # Expanding time-series CV, selecting top 50% by predicted score in each fold.
    cv_rows = []
    tscv = TimeSeriesSplit(n_splits=min(args.cv_splits, 5))
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(clean), 1):
        fold_train = clean.iloc[tr_idx]
        fold_test = clean.iloc[te_idx]
        if fold_train["target"].nunique() < 2:
            continue
        fold_pipe = build_model_pipeline(
            features,
            target=args.target,
            model_name=args.model,
            random_state=args.random_state,
        )
        fold_pipe.fit(fold_train[features].apply(pd.to_numeric, errors="coerce"), fold_train["target"])
        if regression:
            p = fold_pipe.predict(fold_test[features].apply(pd.to_numeric, errors="coerce"))
        else:
            p = fold_pipe.predict_proba(fold_test[features].apply(pd.to_numeric, errors="coerce"))[:, 1]
        cv_rows.append(summarize_selection(f"fold_{fold}_top50", fold_test, p >= np.quantile(p, 0.50)))
    cv_rows = [r for r in cv_rows if r]
    cv_summary = {
        "folds": len(cv_rows),
        "selected_n": int(sum(r["n"] for r in cv_rows)),
        "pnl": float(sum(r["pnl"] for r in cv_rows)),
        "avg_pnl": float(sum(r["pnl"] for r in cv_rows) / max(1, sum(r["n"] for r in cv_rows))),
        "avg_coverage": float(np.mean([r["coverage"] for r in cv_rows])) if cv_rows else None,
    }

    # Fit final artifact on all clean data.
    pipe.fit(X, y)
    artifact = {
        "pipeline": pipe,
        "features": features,
        "target": args.target,
        "score_kind": args.target if regression else "probability",
        "min_pnl": args.min_pnl,
        "min_return": args.min_return,
        "data_quality": quality,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(clean)),
        "positive_rate": None if regression else float(y.mean()),
        "target_mean": float(y.mean()),
        "holdout_start": args.holdout_start,
        "holdout_metric_name": holdout_metric_name,
        "holdout_metric": None if holdout_metric is None else float(holdout_metric),
        "holdout_mae": None if holdout_mae is None else float(holdout_mae),
        "holdout_auc": None if regression or holdout_metric is None else float(holdout_metric),
        "holdout_results": holdout_results,
        "time_series_cv_top50": cv_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    meta_path = args.output.with_suffix(".json")
    meta = {k: v for k, v in artifact.items() if k != "pipeline"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a calendar-call trade filter model")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest", "ridge", "random_forest_regressor"],
        default="ridge",
    )
    parser.add_argument(
        "--target",
        choices=["win", "min_pnl", "min_return", "expected_return", "expected_pnl"],
        default="expected_return",
    )
    parser.add_argument("--min-pnl", type=float, default=10.0)
    parser.add_argument("--min-return", type=float, default=0.10)
    parser.add_argument("--holdout-start", default="2025-07-01")
    parser.add_argument("--max-moneyness-error", type=float, default=0.20)
    parser.add_argument("--min-iv-rv", type=float, default=None, help="Minimum IV/RV ratio to include in training data")
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Copy trained artifact to the bot's default model path",
    )
    args = parser.parse_args()
    meta = train(args)

    if args.deploy:
        bot_default = (
            args.output.parent / "calendar_call_filter_ridge_allfeatures.joblib"
        )
        import shutil

        shutil.copy2(args.output, bot_default)
        shutil.copy2(
            args.output.with_suffix(".json"),
            bot_default.with_suffix(".json"),
        )
        meta["deployed_to"] = str(bot_default)
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
