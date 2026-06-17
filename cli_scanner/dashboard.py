"""
EarningsEdge — Trading Terminal Dashboard
Dark, dense, professional trading-desk UI.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score

# ─── PAGE CONFIG ────────────────────────────────────────────────────────
st.set_page_config(page_title="EarningsEdge", layout="wide", initial_sidebar_state="collapsed")

DB_PATH = "data/earnings_ml.db"

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-tertiary: #1a1a26;
    --bg-card: #141420;
    --border: #232336;
    --text-primary: #e8e8ed;
    --text-secondary: #8888a0;
    --text-muted: #555570;
    --accent: #6366f1;
    --accent-dim: #4f46e5;
    --green: #22c55e;
    --green-dim: #16a34a;
    --red: #ef4444;
    --red-dim: #dc2626;
    --yellow: #eab308;
}

/* Global */
.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', -apple-system, sans-serif;
}

/* Header */
[data-testid="stHeader"] {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-secondary);
    border-radius: 6px;
    padding: 3px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 13px;
    padding: 8px 20px;
    border-radius: 4px;
    letter-spacing: 0.02em;
}

.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    background: var(--bg-tertiary) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 20px;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'Inter', sans-serif;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 24px !important;
    color: var(--text-primary) !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 24px;
    letter-spacing: 0.02em;
    transition: all 0.15s ease;
}

.stButton > button:hover {
    background: var(--accent-dim);
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
}

/* Selectbox & Slider */
.stSelectbox, .stSlider {
    font-size: 13px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-weight: 500;
    font-size: 13px;
}

/* Dividers */
hr {
    border-color: var(--border) !important;
    margin: 8px 0 !important;
}

/* Section headers */
h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

h1 { font-size: 20px !important; }
h2 { font-size: 15px !important; color: var(--text-secondary) !important; text-transform: uppercase; letter-spacing: 0.06em !important; }
h3 { font-size: 13px !important; color: var(--text-secondary) !important; text-transform: uppercase; letter-spacing: 0.06em !important; }

/* Status dot */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.status-green { background: var(--green); box-shadow: 0 0 6px rgba(34,197,94,0.5); }
.status-yellow { background: var(--yellow); box-shadow: 0 0 6px rgba(234,179,8,0.5); }
.status-red { background: var(--red); box-shadow: 0 0 6px rgba(239,68,68,0.5); }

/* Top bar */
.top-bar {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 20px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
}

.top-bar .title {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 16px;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.top-bar .status {
    display: flex;
    gap: 16px;
    align-items: center;
}

/* PnL coloring */
.pnl-pos { color: var(--green) !important; }
.pnl-neg { color: var(--red) !important; }

/* Compact info box */
.info-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* Make chart backgrounds match */
.js-plotly-plot .plotly {
    background: var(--bg-card) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ────────────────────────────────────────────────────────────
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


TRADE_FEATURES = [
    "price", "avg_volume_30d", "has_options", "days_to_expiry",
    "atm_iv_near", "rv30", "iv30_rv30", "hist_vol_3m", "term_slope",
    "term_structure_valid", "expected_move_pct", "expected_move_dollars",
    "straddle_price", "atm_call_delta", "atm_put_delta", "atm_call_iv",
    "atm_put_iv", "sigma_baseline_1y", "sigma_short_leg",
    "sigma_short_leg_fair", "actual_to_fair_ratio", "mc_win_rate",
    "mc_quarters", "strike", "near_entry", "far_entry", "net_debit",
    "moneyness", "abs_moneyness_error", "debit_pct_price",
    "near_far_entry_ratio", "entry_width_days",
]

MODEL_MAP = {
    "Ridge": lambda **kw: Ridge(alpha=kw.get("alpha", 1.0)),
    "Random Forest": lambda **kw: RandomForestRegressor(
        n_estimators=kw.get("n_est", 200), max_depth=kw.get("max_d", 6),
        min_samples_leaf=5, random_state=42),
    "Gradient Boosting": lambda **kw: GradientBoostingRegressor(
        n_estimators=kw.get("n_est", 200), max_depth=kw.get("max_d", 4),
        learning_rate=0.05, random_state=42),
    "SVR (RBF)": lambda **kw: SVR(C=kw.get("C", 1.0), epsilon=0.1),
}


def load_calendar_trades() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("""
            SELECT c.ticker, c.earnings_date, c.scan_date, c.near_expiry, c.far_expiry,
                   c.strike, c.near_entry, c.far_entry, c.net_debit, c.exit_value,
                   c.pnl_dollars, c.return_on_debit, c.model_score, c.model_recommendation,
                   s.price, s.avg_volume_30d, s.market_cap, s.has_options, s.days_to_expiry,
                   s.total_open_interest, s.atm_iv_near, s.rv30, s.iv30_rv30, s.hist_vol_3m,
                   s.term_slope, s.term_structure_valid, s.expected_move_pct,
                   s.expected_move_dollars, s.straddle_price, s.atm_call_delta, s.atm_put_delta,
                   s.atm_call_iv, s.atm_put_iv, s.sigma_baseline_1y, s.sigma_short_leg,
                   s.sigma_short_leg_fair, s.actual_to_fair_ratio, s.mc_win_rate, s.mc_quarters
            FROM calendar_call_trades c
            JOIN snapshots s ON c.snapshot_id = s.id
            WHERE c.return_on_debit IS NOT NULL
            ORDER BY c.scan_date
        """, conn)
    if df.empty:
        return df
    df["moneyness"] = df["strike"] / df["price"] - 1.0
    df["abs_moneyness_error"] = df["moneyness"].abs()
    df["debit_pct_price"] = df["net_debit"] / df["price"]
    df["near_far_entry_ratio"] = df["near_entry"] / df["far_entry"].replace(0, np.nan)
    df["entry_width_days"] = (
        pd.to_datetime(df["far_expiry"]) - pd.to_datetime(df["near_expiry"])
    ).dt.days
    return df


def train_and_backtest(df, model_cls_fn, threshold, model_kwargs):
    available = [f for f in TRADE_FEATURES if f in df.columns]
    X = df[available].copy()
    y = df["return_on_debit"].copy()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([("num", num_pipe, available)])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model_cls_fn(**model_kwargs)),
    ])

    split_idx = int(len(df) * 0.6)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    df_test = df.iloc[split_idx:].copy()

    model.fit(X_train, y_train)

    df_all = df.copy()
    df_all["predicted_return"] = model.predict(X)
    df_test["predicted_return"] = model.predict(X_test)

    df_test["TAKE"] = df_test["predicted_return"] >= threshold
    df_all["TAKE"] = df_all["predicted_return"] >= threshold

    taken = df_test[df_test["TAKE"]]
    metrics = {
        "n_test": len(df_test),
        "n_taken": len(taken),
        "coverage": len(taken) / len(df_test) if len(df_test) > 0 else 0,
        "win_rate": (taken["pnl_dollars"] > 0).mean() if len(taken) > 0 else 0,
        "total_pnl": taken["pnl_dollars"].sum() if len(taken) > 0 else 0,
        "avg_pnl": taken["pnl_dollars"].mean() if len(taken) > 0 else 0,
        "avg_return": taken["return_on_debit"].mean() if len(taken) > 0 else 0,
        "mae": mean_absolute_error(y_test, model.predict(X_test)),
        "r2": r2_score(y_test, model.predict(X_test)),
        "baseline_pnl": df_test["pnl_dollars"].sum(),
        "baseline_win": (df_test["pnl_dollars"] > 0).mean(),
    }
    return model, df_all, df_test, metrics, available


# ─── HEADER BAR ─────────────────────────────────────────────────────────
with get_conn() as conn:
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM snapshots")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM snapshots WHERE outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'")
    labeled = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM snapshots WHERE outcome_fetched_at IS NULL")
    pending = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM calendar_call_trades")
    trades = c.fetchone()[0]

pipeline_health = "green" if pending < 500 else ("yellow" if pending < 1000 else "red")

st.markdown(f"""
<div class="top-bar">
    <span class="title">EarningsEdge</span>
    <div class="status">
        <span><span class="status-dot status-{pipeline_health}"></span>Pipeline</span>
        <span>{total:,} snapshots</span>
        <span>{labeled:,} labeled</span>
        <span>{pending:,} pending</span>
        <span>{trades:,} trades</span>
        <span style="color:var(--text-muted)">Last updated: {datetime.now().strftime('%H:%M:%S')}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ───────────────────────────────────────────────────────────────
tab_backtest, tab_health, tab_live, tab_queue = st.tabs([
    "Backtest", "Pipeline", "Live Scans", "Queue"
])

# ─── TAB 1: BACKTEST ───────────────────────────────────────────────────
with tab_backtest:
    df_trades = load_calendar_trades()

    if df_trades.empty:
        st.warning("No calendar-call trades found.")
    else:
        # ── Controls ──
        with st.container():
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            with c1:
                model_name = st.selectbox("Model", list(MODEL_MAP.keys()), label_visibility="collapsed")
            with c2:
                threshold = st.slider("Threshold", -0.5, 0.5, 0.05, 0.01, label_visibility="collapsed",
                                      help="Predicted return ≥ this → TAKE")
            with c3:
                model_kwargs = {}
                if model_name == "Ridge":
                    model_kwargs["alpha"] = st.number_input("α", 0.01, 10.0, 1.0, 0.1, label_visibility="collapsed")
                elif model_name in ("Random Forest", "Gradient Boosting"):
                    model_kwargs["n_est"] = st.number_input("Trees", 50, 500, 200, 50, label_visibility="collapsed")
                    model_kwargs["max_d"] = st.number_input("Depth", 2, 15, 6, 1, label_visibility="collapsed")
                elif model_name == "SVR (RBF)":
                    model_kwargs["C"] = st.number_input("C", 0.1, 10.0, 1.0, 0.1, label_visibility="collapsed")
            with c4:
                run_bt = st.button("Run Backtest", type="primary", use_container_width=True)

        st.markdown(f"""
        <div style="font-size:12px; color:var(--text-muted); margin-bottom:12px; font-family:'JetBrains Mono',monospace;">
            {len(df_trades)} trades · {df_trades['scan_date'].min()} → {df_trades['scan_date'].max()} ·
            Model: <span style="color:var(--accent)">{model_name}</span> ·
            Threshold: <span style="color:var(--accent)">{threshold:+.2f}</span>
        </div>
        """, unsafe_allow_html=True)

        if run_bt:
            with st.spinner("Training..."):
                model, df_all, df_test, metrics, features = train_and_backtest(
                    df_trades, MODEL_MAP[model_name], threshold, model_kwargs
                )

            # ── Metrics row ──
            pnl_color = "var(--green)" if metrics["total_pnl"] > 0 else "var(--red)"
            ret_color = "var(--green)" if metrics["avg_return"] > 0 else "var(--red)"
            delta_pnl = metrics["total_pnl"] - metrics["baseline_pnl"]
            delta_wr = metrics["win_rate"] - metrics["baseline_win"]

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("TRADES TAKEN", f"{metrics['n_taken']}/{metrics['n_test']}",
                      f"{metrics['coverage']:.0%} coverage")
            m2.metric("WIN RATE", f"{metrics['win_rate']:.1%}",
                      f"{delta_wr:+.1%} vs baseline")
            m3.metric("TOTAL PNL", f"${metrics['total_pnl']:,.0f}",
                      f"{delta_pnl:+,.0f} vs baseline")
            m4.metric("AVG RETURN", f"{metrics['avg_return']:.1%}")
            m5.metric("AVG PNL/TRADE", f"${metrics['avg_pnl']:,.0f}")
            m6.metric("HOLDOUT R²", f"{metrics['r2']:.3f}",
                      f"MAE {metrics['mae']:.3f}")

            # ── Charts ──
            chart_c1, chart_c2 = st.columns([2, 1])

            with chart_c1:
                # Cumulative PnL
                df_taken = df_all[df_all["TAKE"]].sort_values("scan_date").copy()
                if len(df_taken) > 0:
                    df_taken["cum_pnl"] = df_taken["pnl_dollars"].cumsum()
                    # Baseline cum pnl (all trades)
                    df_baseline = df_test.sort_values("scan_date").copy()
                    df_baseline["cum_pnl_baseline"] = df_baseline["pnl_dollars"].cumsum()

                    fig = go.Figure()

                    # All-trades baseline
                    fig.add_trace(go.Scatter(
                        x=df_baseline["scan_date"], y=df_baseline["cum_pnl_baseline"],
                        mode="lines", name="All trades (baseline)",
                        line=dict(color="#555570", width=1, dash="dot"),
                    ))

                    # Model-selected
                    fig.add_trace(go.Scatter(
                        x=df_taken["scan_date"], y=df_taken["cum_pnl"],
                        mode="lines+markers", name="Model selected",
                        line=dict(color="#6366f1", width=2),
                        marker=dict(size=4, color="#6366f1"),
                        hovertemplate="<b>%{customdata[0]}</b><br>"
                                      "Date: %{x}<br>"
                                      "Cum PnL: $%{y:,.0f}<br>"
                                      "Predicted: %{customdata[1]:.1%}<br>"
                                      "Actual: %{customdata[2]:.1%}<br>"
                                      "Debit: $%{customdata[3]:.2f}<extra></extra>",
                        customdata=df_taken[["ticker", "predicted_return", "return_on_debit", "net_debit"]].values,
                    ))

                    # Holdout boundary
                    split_date = df_test["scan_date"].min()
                    fig.add_vline(x=split_date, line=dict(color="#eab308", width=1, dash="dash"),
                                  annotation_text="HOLDOUT",
                                  annotation=dict(font=dict(size=10, color="#eab308"),
                                                  bgcolor="rgba(0,0,0,0)"))

                    fig.update_layout(
                        title=dict(text="CUMULATIVE PNL", font=dict(size=13, color="#8888a0")),
                        xaxis=dict(title="", gridcolor="#1a1a26", showgrid=True,
                                   tickfont=dict(family="JetBrains Mono", size=10, color="#555570")),
                        yaxis=dict(title=dict(text="$", font=dict(size=11, color="#555570")),
                                   gridcolor="#1a1a26", showgrid=True,
                                   tickfont=dict(family="JetBrains Mono", size=10, color="#555570")),
                        plot_bgcolor="#141420",
                        paper_bgcolor="#141420",
                        font=dict(family="Inter", color="#e8e8ed"),
                        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                        height=420,
                        margin=dict(l=60, r=20, t=40, b=60),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with chart_c2:
                # Predicted vs Actual scatter
                fig2 = go.Figure()

                skip = df_all[~df_all["TAKE"]]
                take = df_all[df_all["TAKE"]]

                fig2.add_trace(go.Scatter(
                    x=skip["predicted_return"], y=skip["return_on_debit"],
                    mode="markers", name="SKIP",
                    marker=dict(color="#555570", size=5, opacity=0.5),
                    hovertemplate="<b>%{customdata[0]}</b><br>Pred: %{x:.1%}<br>Actual: %{y:.1%}<extra></extra>",
                    customdata=skip[["ticker"]].values,
                ))
                fig2.add_trace(go.Scatter(
                    x=take["predicted_return"], y=take["return_on_debit"],
                    mode="markers", name="TAKE",
                    marker=dict(color="#6366f1", size=6, opacity=0.8),
                    hovertemplate="<b>%{customdata[0]}</b><br>Pred: %{x:.1%}<br>Actual: %{y:.1%}<extra></extra>",
                    customdata=take[["ticker"]].values,
                ))

                # Threshold line
                fig2.add_vline(x=threshold, line=dict(color="#eab308", width=1, dash="dot"))
                # Zero lines
                fig2.add_hline(y=0, line=dict(color="#232336", width=1))

                fig2.update_layout(
                    title=dict(text="PREDICTED VS ACTUAL", font=dict(size=13, color="#8888a0")),
                    xaxis=dict(title=dict(text="Predicted", font=dict(size=11, color="#555570")),
                               gridcolor="#1a1a26", showgrid=True,
                               tickformat=".0%",
                               tickfont=dict(family="JetBrains Mono", size=10, color="#555570")),
                    yaxis=dict(title=dict(text="Actual", font=dict(size=11, color="#555570")),
                               gridcolor="#1a1a26", showgrid=True,
                               tickformat=".0%",
                               tickfont=dict(family="JetBrains Mono", size=10, color="#555570")),
                    plot_bgcolor="#141420",
                    paper_bgcolor="#141420",
                    font=dict(family="Inter", color="#e8e8ed"),
                    legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center",
                                font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                    height=420,
                    margin=dict(l=60, r=20, t=40, b=60),
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ── Feature importance (tree models) ──
            if model_name in ("Random Forest", "Gradient Boosting"):
                reg = model.named_steps["regressor"]
                imp = pd.DataFrame({"feature": features, "importance": reg.feature_importances_})
                imp = imp.sort_values("importance", ascending=True).tail(12)

                fig_imp = go.Figure(go.Bar(
                    x=imp["importance"], y=imp["feature"], orientation="h",
                    marker=dict(color="#6366f1", opacity=0.8),
                    hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                ))
                fig_imp.update_layout(
                    title=dict(text="FEATURE IMPORTANCE", font=dict(size=13, color="#8888a0")),
                    xaxis=dict(gridcolor="#1a1a26", showgrid=True,
                               tickfont=dict(family="JetBrains Mono", size=10, color="#555570")),
                    yaxis=dict(tickfont=dict(family="JetBrains Mono", size=10, color="#8888a0")),
                    plot_bgcolor="#141420",
                    paper_bgcolor="#141420",
                    font=dict(family="Inter", color="#e8e8ed"),
                    height=350,
                    margin=dict(l=140, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            # ── Trades table ──
            st.markdown("##### HOLDOUT TRADES")
            display_cols = ["scan_date", "ticker", "strike", "net_debit", "predicted_return",
                            "return_on_debit", "pnl_dollars", "TAKE"]
            display_cols = [c for c in display_cols if c in df_test.columns]
            df_display = df_test[display_cols].sort_values("scan_date", ascending=False).copy()
            df_display["strike"] = df_display["strike"].map("{:.0f}".format)
            df_display["net_debit"] = df_display["net_debit"].map("{:.2f}".format)
            df_display["predicted_return"] = df_display["predicted_return"].map("{:.1%}".format)
            df_display["return_on_debit"] = df_display["return_on_debit"].map("{:.1%}".format)
            df_display["pnl_dollars"] = df_display["pnl_dollars"].map("${:,.0f}".format)
            df_display.columns = [c.upper().replace("_", " ") for c in df_display.columns]
            st.dataframe(df_display, use_container_width=True, height=400, hide_index=True)


# ─── TAB 2: PIPELINE ───────────────────────────────────────────────────
with tab_health:
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM snapshots")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM snapshots WHERE outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'")
        labeled = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM snapshots WHERE outcome_fetched_at = 'unavailable'")
        unavail = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM snapshots WHERE outcome_fetched_at IS NULL")
        pending = c.fetchone()[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL SNAPSHOTS", f"{total:,}")
    c2.metric("LABELED", f"{labeled:,}")
    c3.metric("PENDING", f"{pending:,}")
    c4.metric("UNAVAILABLE", f"{unavail:,}")

    st.markdown("---")

    with get_conn() as conn:
        df_out = pd.read_sql_query("""
            SELECT actual_move_pct, expected_move_pct, ticker, earnings_date
            FROM snapshots
            WHERE actual_move_pct IS NOT NULL
              AND outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'
        """, conn)

    if not df_out.empty:
        df_out["beat"] = df_out["actual_move_pct"].abs() > df_out["expected_move_pct"]
        beat_rate = df_out["beat"].mean()

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("BEAT RATE", f"{beat_rate:.1%}")
        mc2.metric("AVG ACTUAL MOVE", f"{df_out['actual_move_pct'].abs().mean():.2f}%")
        mc3.metric("AVG EXPECTED MOVE", f"{df_out['expected_move_pct'].mean():.2f}%")

        ch1, ch2 = st.columns(2)
        with ch1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_out["actual_move_pct"], nbinsx=50,
                marker=dict(color="#6366f1", opacity=0.8),
            ))
            fig.update_layout(
                title=dict(text="ACTUAL MOVE DISTRIBUTION", font=dict(size=13, color="#8888a0")),
                xaxis=dict(title="%", gridcolor="#1a1a26", showgrid=True,
                           tickfont=dict(family="JetBrains Mono", size=10, color="#555570")),
                yaxis=dict(title="Count", gridcolor="#1a1a26", showgrid=True,
                           tickfont=dict(family="JetBrains Mono", size=10, color="#555570")),
                plot_bgcolor="#141420", paper_bgcolor="#141420",
                font=dict(family="Inter", color="#e8e8ed"),
                height=320, margin=dict(l=50, r=20, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_out["expected_move_pct"], y=df_out["actual_move_pct"],
                mode="markers",
                marker=dict(color="#6366f1", size=5, opacity=0.5),
                hovertemplate="<b>%{customdata[0]}</b><br>Expected: %{x:.1%}<br>Actual: %{y:.1%}<extra></extra>",
                customdata=df_out[["ticker"]].values,
            ))
            max_val = max(df_out["expected_move_pct"].max(), df_out["actual_move_pct"].abs().max())
            fig2.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val], mode="lines",
                line=dict(color="#ef4444", width=1, dash="dot"), showlegend=False,
            ))
            fig2.update_layout(
                title=dict(text="EXPECTED VS ACTUAL", font=dict(size=13, color="#8888a0")),
                xaxis=dict(title="Expected %", gridcolor="#1a1a26", showgrid=True,
                           tickfont=dict(family="JetBrains Mono", size=10, color="#555570")),
                yaxis=dict(title="Actual %", gridcolor="#1a1a26", showgrid=True,
                           tickfont=dict(family="JetBrains Mono", size=10, color="#555570")),
                plot_bgcolor="#141420", paper_bgcolor="#141420",
                font=dict(family="Inter", color="#e8e8ed"),
                height=320, margin=dict(l=50, r=20, t=40, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ─── TAB 3: LIVE SCANS ─────────────────────────────────────────────────
with tab_live:
    st.markdown("##### LIVE CALENDAR CANDIDATES")
    with get_conn() as conn:
        try:
            df_live = pd.read_sql_query("""
                SELECT scan_timestamp, ticker, strike, net_debit, model_expected_return,
                       model_decision, tier, sigma_short_leg_fair as fair_iv,
                       atm_iv_near as actual_iv, iv_rv_ratio,
                       term_slope as term_structure, expected_move_pct, win_rate
                FROM live_calendar_candidates
                ORDER BY scan_timestamp DESC LIMIT 100
            """, conn)
            if not df_live.empty:
                df_live.columns = [c.upper().replace("_", " ") for c in df_live.columns]
                st.dataframe(df_live, use_container_width=True, height=400, hide_index=True)
            else:
                st.info("No live candidates yet. Bot will populate this as it scans.")
        except Exception as e:
            st.warning(f"Table not found: {e}")

    st.markdown("---")
    st.markdown("##### SCANNER OUTPUTS")
    with get_conn() as conn:
        try:
            df_out = pd.read_sql_query("""
                SELECT scan_timestamp, ticker, tier, model_decision, model_expected_return,
                       atm_iv_near as actual_iv, expected_move_pct, win_rate
                FROM scanner_scan_outputs
                ORDER BY scan_timestamp DESC LIMIT 100
            """, conn)
            if not df_out.empty:
                df_out.columns = [c.upper().replace("_", " ") for c in df_out.columns]
                st.dataframe(df_out, use_container_width=True, height=400, hide_index=True)
        except Exception:
            pass


# ─── TAB 4: QUEUE MANAGEMENT ───────────────────────────────────────────
with tab_queue:
    st.markdown("##### OUTCOME QUEUE MANAGEMENT")

    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM snapshots WHERE outcome_fetched_at IS NULL")
        pending = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM snapshots WHERE outcome_fetched_at = 'unavailable'")
        unavailable = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM snapshots WHERE outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'")
        labeled = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM snapshots")
        total = c.fetchone()[0]

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("PENDING", f"{pending:,}")
    q2.metric("UNAVAILABLE", f"{unavailable:,}")
    q3.metric("LABELED", f"{labeled:,}")
    q4.metric("TOTAL", f"{total:,}")

    st.markdown("---")

    # ── Filters ──
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        queue_view = st.radio("View", ["Pending", "Unavailable", "Labeled", "All"],
                              horizontal=True, label_visibility="collapsed")
    with fc2:
        ticker_filter = st.text_input("Ticker filter", placeholder="e.g. AAPL", label_visibility="collapsed")
    with fc3:
        limit = st.number_input("Limit", 50, 2000, 200, 50, label_visibility="collapsed")

    # ── Query ──
    status_map = {
        "Pending": "outcome_fetched_at IS NULL",
        "Unavailable": "outcome_fetched_at = 'unavailable'",
        "Labeled": "outcome_fetched_at IS NOT NULL AND outcome_fetched_at != 'unavailable'",
        "All": "1=1",
    }
    where = status_map[queue_view]
    ticker_clause = f"AND ticker LIKE '%{ticker_filter}%'" if ticker_filter else ""

    with get_conn() as conn:
        df_queue = pd.read_sql_query(f"""
            SELECT id, ticker, earnings_date, scan_date, timing, price,
                   atm_iv_near, expected_move_pct, outcome_fetched_at,
                   outcome_attempt_count, actual_move_pct, actual_move_direction,
                   pre_earnings_close, post_earnings_close
            FROM snapshots
            WHERE {where} {ticker_clause}
            ORDER BY earnings_date DESC
            LIMIT {limit}
        """, conn)

    if not df_queue.empty:
        # Format for display
        df_display = df_queue.copy()
        df_display["price"] = df_display["price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "—")
        df_display["atm_iv_near"] = df_display["atm_iv_near"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        df_display["expected_move_pct"] = df_display["expected_move_pct"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        df_display["actual_move_pct"] = df_display["actual_move_pct"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")
        df_display.columns = [c.upper().replace("_", " ") for c in df_display.columns]

        st.dataframe(df_display, use_container_width=True, height=400, hide_index=True)

        # ── Reset unavailable ──
        if queue_view == "Unavailable" and len(df_queue) > 0:
            st.markdown("---")
            st.markdown("##### RESET UNAVAILABLE")

            rc1, rc2 = st.columns([3, 1])
            with rc1:
                reset_ids = st.multiselect(
                    "Select snapshots to reset back to Pending",
                    options=df_queue["id"].tolist(),
                    format_func=lambda x: f"ID {x} — {df_queue[df_queue['id']==x]['ticker'].values[0]} ({df_queue[df_queue['id']==x]['earnings_date'].values[0]})"
                )
            with rc2:
                reset_all = st.button("Reset All Visible", type="secondary", use_container_width=True)

            if reset_all:
                reset_ids = df_queue["id"].tolist()

            if reset_ids and not reset_all:
                if st.button(f"Reset {len(reset_ids)} to Pending", type="primary"):
                    with get_conn() as conn:
                        conn.executemany(
                            "UPDATE snapshots SET outcome_fetched_at = NULL, outcome_attempt_count = 0 WHERE id = ?",
                            [(sid,) for sid in reset_ids]
                        )
                        conn.commit()
                    st.success(f"Reset {len(reset_ids)} snapshots to Pending. Refresh to see changes.")
                    st.rerun()

            elif reset_all:
                with get_conn() as conn:
                    conn.execute(
                        f"UPDATE snapshots SET outcome_fetched_at = NULL, outcome_attempt_count = 0 WHERE {where} {ticker_clause}"
                    )
                    conn.commit()
                st.success(f"Reset all {len(reset_ids)} visible snapshots to Pending. Refresh to see changes.")
                st.rerun()

    else:
        st.info(f"No {queue_view.lower()} snapshots found.")

    # ── Calendar trades backfill status ──
    st.markdown("---")
    st.markdown("##### CALENDAR TRADES BACKFILL")

    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM calendar_call_trades")
        trades_total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM calendar_call_trades WHERE exit_value IS NOT NULL")
        trades_with_exit = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM calendar_call_trades WHERE exit_value IS NULL")
        trades_no_exit = c.fetchone()[0]
        c.execute("SELECT MIN(scan_date), MAX(scan_date) FROM calendar_call_trades")
        date_range = c.fetchone()

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("TOTAL TRADES", f"{trades_total:,}")
    tc2.metric("WITH EXIT", f"{trades_with_exit:,}")
    tc3.metric("NO EXIT", f"{trades_no_exit:,}")

    if date_range[0]:
        st.markdown(f"""
        <div style="font-size:12px; color:var(--text-muted); font-family:'JetBrains Mono',monospace;">
            Date range: {date_range[0]} → {date_range[1]}
        </div>
        """, unsafe_allow_html=True)
