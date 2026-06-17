"""Diagnose why R² is negative and try better configurations."""
import json
import sqlite3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DB = "data/earnings_ml.db"

# Use the same loading logic as train_calendar_filter
from train_calendar_filter import load_calendar_trades, apply_data_quality_gates, BASE_FEATURES

df = load_calendar_trades(DB)
clean, quality = apply_data_quality_gates(df, 0.20)
print(f"Clean rows: {len(clean)}")
print(f"Target stats: mean={clean['return_on_debit'].mean():.3f} std={clean['return_on_debit'].std():.3f} min={clean['return_on_debit'].min():.3f} max={clean['return_on_debit'].max():.3f}")

# Time split
cut = pd.Timestamp("2025-07-01")
train_df = clean[clean["earnings_date"] < cut]
test_df = clean[clean["earnings_date"] >= cut]
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

features_all = [c for c in BASE_FEATURES if c in clean.columns and clean[c].notna().any()]
features_core = [c for c in features_all if c in [
    "price", "avg_volume_30d", "atm_iv_near", "rv30", "iv30_rv30",
    "hist_vol_3m", "term_slope", "expected_move_pct",
    "net_debit", "moneyness", "debit_pct_price", "near_far_entry_ratio", "entry_width_days",
    "actual_to_fair_ratio", "sigma_short_leg_fair"
]]
features_minimal = [c for c in features_all if c in [
    "net_debit", "debit_pct_price", "moneyness", "atm_iv_near", "iv30_rv30",
    "actual_to_fair_ratio", "term_slope"
]]

configs = [
    ("ridge_all_29feat_a1", Ridge(1.0), features_all),
    ("ridge_all_29feat_a10", Ridge(10.0), features_all),
    ("ridge_all_29feat_a100", Ridge(100.0), features_all),
    ("ridge_core_15feat_a1", Ridge(1.0), features_core),
    ("ridge_core_15feat_a10", Ridge(10.0), features_core),
    ("ridge_minimal_7feat_a1", Ridge(1.0), features_minimal),
    ("ridge_minimal_7feat_a10", Ridge(10.0), features_minimal),
    ("lasso_7feat_a01", Lasso(0.01), features_minimal),
    ("rf_shallow", RandomForestRegressor(n_estimators=200, max_depth=2, min_samples_leaf=15, random_state=42), features_core),
    ("gbr_shallow", GradientBoostingRegressor(n_estimators=100, max_depth=2, min_samples_leaf=15, learning_rate=0.05, random_state=42), features_core),
]

y_train = train_df["return_on_debit"].astype(float)
y_test = test_df["return_on_debit"].astype(float)

print(f"\n{'Config':35s} {'R2':>7s} {'MAE':>7s} {'Top25_WR':>9s} {'Top25_PnL':>10s}")
print("-" * 75)

for name, model, feats in configs:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pre = ColumnTransformer([("num", numeric_pipeline, feats)])
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    
    pipe.fit(train_df[feats].apply(pd.to_numeric, errors="coerce"), y_train)
    pred_test = pipe.predict(test_df[feats].apply(pd.to_numeric, errors="coerce"))
    
    r2 = r2_score(y_test, pred_test)
    mae = mean_absolute_error(y_test, pred_test)
    
    # Top 25% selection
    top25_mask = pred_test >= np.quantile(pred_test, 0.75)
    top25_pnl = test_df.loc[top25_mask, "pnl_dollars"]
    top25_wr = (top25_pnl > 0).mean() if len(top25_pnl) > 0 else 0
    top25_pnl_sum = top25_pnl.sum() if len(top25_pnl) > 0 else 0
    
    print(f"{name:35s} {r2:7.3f} {mae:7.3f} {top25_wr:9.1%} ${top25_pnl_sum:9.0f}")

# Also try classification approach
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

y_train_cls = (train_df["pnl_dollars"] > 0).astype(int)
y_test_cls = (test_df["pnl_dollars"] > 0).astype(int)

print(f"\n{'Classification':35s} {'AUC':>7s} {'Top25_WR':>9s} {'Top25_PnL':>10s}")
print("-" * 65)

for name, model, feats in [
    ("logistic_core", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1), features_core),
    ("rf_cls_shallow", RandomForestClassifier(n_estimators=200, max_depth=2, min_samples_leaf=15, class_weight="balanced_subsample", random_state=42), features_core),
]:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    pre = ColumnTransformer([("num", numeric_pipeline, feats)])
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    
    pipe.fit(train_df[feats].apply(pd.to_numeric, errors="coerce"), y_train_cls)
    pred_proba = pipe.predict_proba(test_df[feats].apply(pd.to_numeric, errors="coerce"))[:, 1]
    
    auc = roc_auc_score(y_test_cls, pred_proba) if y_test_cls.nunique() == 2 else 0
    top25_mask = pred_proba >= np.quantile(pred_proba, 0.75)
    top25_pnl = test_df.loc[top25_mask, "pnl_dollars"]
    top25_wr = (top25_pnl > 0).mean() if len(top25_pnl) > 0 else 0
    top25_pnl_sum = top25_pnl.sum() if len(top25_pnl) > 0 else 0
    
    print(f"{name:35s} {auc:7.3f} {top25_wr:9.1%} ${top25_pnl_sum:9.0f}")
