#!/usr/bin/env python3
"""Train an earnings opportunity screening model from earnings_ml.db."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from earnings_edge.db import DEFAULT_DB_PATH, get_connection

FEATURES = [
    "price",
    "avg_volume_30d",
    "market_cap",
    "has_options",
    "days_to_expiry",
    "total_open_interest",
    "atm_iv_near",
    "rv30",
    "iv30_rv30",
    "hist_vol_3m",
    "term_slope",
    "term_structure_valid",
    "expected_move_pct",
    "expected_move_dollars",
    "straddle_price",
    "atm_call_delta",
    "atm_put_delta",
    "atm_call_iv",
    "atm_put_iv",
    "sigma_baseline_1y",
    "sigma_short_leg",
    "sigma_short_leg_fair",
    "actual_to_fair_ratio",
    "mc_win_rate",
    "mc_quarters",
]


def load_dataset(target: str, large_move_threshold: float) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM snapshots WHERE actual_move_pct IS NOT NULL", conn)
    conn.close()

    if df.empty:
        return df

    df["abs_actual_move_pct"] = df["actual_move_pct"].abs()

    if target == "beat_expected_move":
        df = df[df["expected_move_pct"].notna()].copy()
        df["target"] = (df["abs_actual_move_pct"] > df["expected_move_pct"]).astype(int)
    elif target == "large_move":
        df["target"] = (df["abs_actual_move_pct"] >= large_move_threshold).astype(int)
    elif target == "direction_up":
        df["target"] = (df["actual_move_pct"] > 0).astype(int)
    else:
        raise ValueError(f"Unknown target: {target}")

    return df


def train(target: str, model_type: str, min_rows: int, large_move_threshold: float, output: Path) -> dict:
    df = load_dataset(target, large_move_threshold)
    if len(df) < min_rows:
        raise RuntimeError(
            f"Not enough labeled rows to train: {len(df)} rows found, need >= {min_rows}. "
            "Run polygon_backfill.py or wait for daily collection/outcomes."
        )

    available = [c for c in FEATURES if c in df.columns]
    X = df[available].apply(pd.to_numeric, errors="coerce")
    y = df["target"].astype(int)

    if y.nunique() < 2:
        raise RuntimeError(f"Target has only one class ({y.iloc[0]}); need positives and negatives")

    class_counts = y.value_counts()
    tiny_dataset = len(df) < 8 or class_counts.min() < 2

    if tiny_dataset:
        # Smoke-test mode only: fit on all data and skip holdout metrics.
        X_train, y_train = X, y
        X_test = y_test = None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pre = ColumnTransformer([("num", numeric_pipeline, available)])

    if model_type == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipe = Pipeline([("preprocess", pre), ("model", clf)])
    pipe.fit(X_train, y_train)

    if tiny_dataset:
        auc = None
        report = {"note": "Tiny dataset smoke fit; no holdout metrics computed."}
    else:
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipe,
        "features": available,
        "target": target,
        "model_type": model_type,
        "trained_at": datetime.utcnow().isoformat(),
        "rows": len(df),
        "positive_rate": float(y.mean()),
        "auc": None if auc is None else float(auc),
        "classification_report": report,
    }
    joblib.dump(artifact, output)

    meta_path = output.with_suffix(".json")
    meta = {k: v for k, v in artifact.items() if k != "pipeline"}
    meta["classification_report"] = report
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta


def main() -> None:
    p = argparse.ArgumentParser(description="Train ML screener from earnings_ml.db")
    p.add_argument("--target", choices=["beat_expected_move", "large_move", "direction_up"], default="beat_expected_move")
    p.add_argument("--model", choices=["logistic", "random_forest"], default="random_forest")
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--large-move-threshold", type=float, default=5.0)
    p.add_argument("--output", type=Path, default=Path("data/models/earnings_model.joblib"))
    args = p.parse_args()

    meta = train(args.target, args.model, args.min_rows, args.large_move_threshold, args.output)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
