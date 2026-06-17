"""SQLite storage for ML feature snapshots and post-earnings outcomes."""

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

from .config import get_logger

logger = get_logger("db")

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "earnings_ml.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    earnings_date   TEXT NOT NULL,       -- YYYY-MM-DD
    scan_date       TEXT NOT NULL,       -- when we collected this
    timing          TEXT,                -- Pre Market, Post Market, During Market, Manual

    -- Basic price / volume
    price                           REAL,
    avg_volume_30d                  REAL,
    market_cap                      REAL,

    -- Options availability
    has_options                     INTEGER,  -- 0/1
    nearest_expiry                  TEXT,     -- YYYY-MM-DD
    days_to_expiry                  INTEGER,
    total_open_interest             INTEGER,

    -- Volatility
    atm_iv_near                     REAL,     -- ATM IV nearest expiry
    rv30                            REAL,     -- Yang-Zhang 30d realized vol
    iv30_rv30                       REAL,     -- IV/RV ratio
    hist_vol_3m                     REAL,     -- Historical vol 3-month

    -- Term structure
    term_slope                      REAL,
    term_structure_valid            INTEGER,  -- 0/1

    -- Expected move
    expected_move_pct               REAL,
    expected_move_dollars           REAL,
    straddle_price                  REAL,

    -- ATM greeks
    atm_call_delta                  REAL,
    atm_put_delta                   REAL,
    atm_call_iv                     REAL,
    atm_put_iv                      REAL,

    -- Short-leg fair value
    sigma_baseline_1y               REAL,
    sigma_short_leg                 REAL,
    sigma_short_leg_fair            REAL,
    actual_to_fair_ratio            REAL,

    -- Analyzer recommendation
    recommendation                  TEXT,     -- BUY, SELL, HOLD

    -- Market Chameleon (optional — slow)
    mc_win_rate                     REAL,
    mc_quarters                     INTEGER,

    -- What went wrong (if feature extraction partially failed)
    collection_error                TEXT,

    -- ── Outcome columns (filled by outcomes.py) ──
    pre_earnings_close              REAL,
    post_earnings_close             REAL,
    actual_move_pct                 REAL,
    actual_move_direction           TEXT,     -- UP, DOWN, FLAT
    max_intraday_range_pct          REAL,     -- (high-low)/pre_close
    outcome_fetched_at              TEXT,
    outcome_attempt_count           INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_snap_ticker_date ON snapshots(ticker, earnings_date);
CREATE INDEX IF NOT EXISTS idx_snap_outcome     ON snapshots(outcome_fetched_at);

CREATE TABLE IF NOT EXISTS live_calendar_candidates (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_timestamp          TEXT NOT NULL,
    ticker                  TEXT NOT NULL,
    earnings_date           TEXT,
    -- Scanner context
    tier                    INTEGER,
    passed                  INTEGER,
    near_miss               INTEGER DEFAULT 0,
    scanner_reason          TEXT,
    display_status          TEXT,               -- displayed, near_miss, not_displayed
    -- Price / volume
    price                   REAL,
    volume                  REAL,
    market_cap              REAL,
    -- Options / expiry
    strike                  REAL,
    near_expiry             TEXT,
    far_expiry              TEXT,
    days_to_expiry          INTEGER,
    total_open_interest     INTEGER,
    -- Leg quotes
    near_bid                REAL,
    near_ask                REAL,
    far_bid                 REAL,
    far_ask                 REAL,
    near_entry              REAL,
    far_entry               REAL,
    -- Debit quotes
    net_debit               REAL,
    net_debit_bid           REAL,
    net_debit_mid           REAL,
    net_debit_ask           REAL,
    -- IV / volatility metrics
    atm_iv_near             REAL,
    sigma_baseline_1y       REAL,
    sigma_short_leg         REAL,
    sigma_short_leg_fair    REAL,
    actual_to_fair_ratio    REAL,
    iv_rv_ratio             REAL,
    hist_vol_3m             REAL,
    -- Term structure
    term_slope              REAL,
    term_structure_valid    INTEGER,
    -- Expected move
    expected_move_pct       REAL,
    expected_move_dollars   REAL,
    straddle_price          REAL,
    -- ATM greeks
    atm_call_delta          REAL,
    atm_put_delta           REAL,
    atm_call_iv             REAL,
    atm_put_iv              REAL,
    -- Win rate
    win_rate                REAL,
    win_quarters            INTEGER,
    -- Model results
    model_expected_return   REAL,
    model_decision          TEXT,
    model_rejection_reasons TEXT,
    selected_by_bot         INTEGER DEFAULT 0,
    features_json           TEXT,
    -- Outcome (filled later)
    exit_value              REAL,
    pnl_dollars             REAL,
    return_on_debit         REAL,
    outcome_fetched_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_live_calendar_scan ON live_calendar_candidates(scan_timestamp);
CREATE INDEX IF NOT EXISTS idx_live_calendar_ticker_date ON live_calendar_candidates(ticker, earnings_date);
CREATE INDEX IF NOT EXISTS idx_live_calendar_outcome ON live_calendar_candidates(outcome_fetched_at);

CREATE TABLE IF NOT EXISTS scanner_scan_outputs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_timestamp          TEXT NOT NULL,
    ticker                  TEXT NOT NULL,
    earnings_date           TEXT,
    -- Scanner context
    tier                    INTEGER,
    passed                  INTEGER,
    near_miss               INTEGER DEFAULT 0,
    scanner_reason          TEXT,
    display_status          TEXT,               -- displayed, near_miss, not_displayed
    -- Price / volume
    price                   REAL,
    volume                  REAL,
    market_cap              REAL,
    -- Options / expiry
    strike                  REAL,
    near_expiry             TEXT,
    far_expiry              TEXT,
    days_to_expiry          INTEGER,
    total_open_interest     INTEGER,
    -- Debit quotes
    net_debit               REAL,
    net_debit_bid           REAL,
    net_debit_mid           REAL,
    net_debit_ask           REAL,
    -- IV / volatility metrics
    atm_iv_near             REAL,
    sigma_baseline_1y       REAL,
    sigma_short_leg         REAL,
    sigma_short_leg_fair    REAL,
    actual_to_fair_ratio    REAL,
    iv_rv_ratio             REAL,
    -- Term structure
    term_slope              REAL,
    term_structure_valid    INTEGER,
    -- Expected move
    expected_move_pct       REAL,
    expected_move_dollars   REAL,
    -- Win rate
    win_rate                REAL,
    win_quarters            INTEGER,
    -- Model results
    model_expected_return   REAL,
    model_decision          TEXT,
    model_rejection_reasons TEXT,
    selected_by_bot         INTEGER DEFAULT 0,
    features_json           TEXT,
    -- Outcome (filled later)
    exit_value              REAL,
    pnl_dollars             REAL,
    return_on_debit         REAL,
    outcome_fetched_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_scan_output_scan ON scanner_scan_outputs(scan_timestamp);
CREATE INDEX IF NOT EXISTS idx_scan_output_ticker_date ON scanner_scan_outputs(ticker, earnings_date);

CREATE TABLE IF NOT EXISTS scan_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_timestamp  TEXT NOT NULL,
    scanner_name    TEXT NOT NULL,
    trigger_type    TEXT NOT NULL,  -- 'scheduled', 'manual', 'cron'
    candidate_count INTEGER DEFAULT 0,
    tier1_count     INTEGER DEFAULT 0,
    tier2_count     INTEGER DEFAULT 0,
    take_count      INTEGER DEFAULT 0,
    duration_secs   REAL,
    success         INTEGER DEFAULT 0,
    error_message   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
"""


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a connection to the ML database, creating it if needed.

    Enables WAL mode for concurrent read/write and sets a busy timeout
    to avoid ``database is locked`` errors during scans.
    """
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    # Performance: WAL mode allows concurrent readers during writes
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")  # 30s
    conn.execute("PRAGMA synchronous=NORMAL")   # WAL-safe, faster
    conn.executescript(_SCHEMA)
    _migrate_snapshots(conn)
    _migrate_live_calendar_candidates(conn)
    return conn


def _migrate_snapshots(conn: sqlite3.Connection) -> None:
    """Add missing columns to snapshots for existing databases."""
    existing = {r['name'] for r in conn.execute('pragma table_info(snapshots)')}
    migrations = {
        'outcome_attempt_count': 'INTEGER DEFAULT 0',
        'data_source': 'TEXT DEFAULT "unknown"',
    }
    for col, col_type in migrations.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE snapshots ADD COLUMN {col} {col_type}")
    conn.commit()


def _migrate_live_calendar_candidates(conn: sqlite3.Connection) -> None:
    """Add missing columns to live_calendar_candidates for existing databases."""
    existing = {r['name'] for r in conn.execute('pragma table_info(live_calendar_candidates)')}
    needed = {
        'tier': 'INTEGER',
        'passed': 'INTEGER',
        'near_miss': 'INTEGER DEFAULT 0',
        'scanner_reason': 'TEXT',
        'display_status': 'TEXT',
        'volume': 'REAL',
        'market_cap': 'REAL',
        'days_to_expiry': 'INTEGER',
        'total_open_interest': 'INTEGER',
        'atm_iv_near': 'REAL',
        'sigma_baseline_1y': 'REAL',
        'sigma_short_leg': 'REAL',
        'sigma_short_leg_fair': 'REAL',
        'actual_to_fair_ratio': 'REAL',
        'iv_rv_ratio': 'REAL',
        'hist_vol_3m': 'REAL',
        'term_slope': 'REAL',
        'term_structure_valid': 'INTEGER',
        'expected_move_pct': 'REAL',
        'expected_move_dollars': 'REAL',
        'straddle_price': 'REAL',
        'atm_call_delta': 'REAL',
        'atm_put_delta': 'REAL',
        'atm_call_iv': 'REAL',
        'atm_put_iv': 'REAL',
        'win_rate': 'REAL',
        'win_quarters': 'INTEGER',
    }
    for col, col_type in needed.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE live_calendar_candidates ADD COLUMN {col} {col_type}")
    conn.commit()


def insert_snapshot(conn: sqlite3.Connection, row: dict) -> int:
    """Insert a single feature snapshot and return the row id."""
    cols = [
        "ticker", "earnings_date", "scan_date", "timing",
        "price", "avg_volume_30d", "market_cap",
        "has_options", "nearest_expiry", "days_to_expiry", "total_open_interest",
        "atm_iv_near", "rv30", "iv30_rv30", "hist_vol_3m",
        "term_slope", "term_structure_valid",
        "expected_move_pct", "expected_move_dollars", "straddle_price",
        "atm_call_delta", "atm_put_delta", "atm_call_iv", "atm_put_iv",
        "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair", "actual_to_fair_ratio",
        "recommendation",
        "mc_win_rate", "mc_quarters",
        "collection_error",
        "data_source",
    ]
    placeholders = ", ".join(f":{c}" for c in cols)
    sql = f"INSERT INTO snapshots ({', '.join(cols)}) VALUES ({placeholders})"
    cur = conn.execute(sql, {c: row.get(c) for c in cols})
    conn.commit()
    return cur.lastrowid or 0


def fetch_pending_outcomes(conn: sqlite3.Connection, min_age_days: int = 2) -> List[sqlite3.Row]:
    """Return snapshots where earnings_date is past and outcome not yet fetched."""
    cutoff = (date.today()).isoformat()
    from datetime import timedelta
    age_cutoff = (date.today() - timedelta(days=min_age_days)).isoformat()

    rows = conn.execute(
        "SELECT * FROM snapshots "
        "WHERE outcome_fetched_at IS NULL "
        "  AND earnings_date <= ? "
        "  AND earnings_date <= ? "
        "ORDER BY earnings_date",
        (cutoff, age_cutoff),
    ).fetchall()
    return rows


def insert_live_calendar_candidate(conn: sqlite3.Connection, row: dict) -> int:
    """Insert a live call-calendar candidate quote/model snapshot."""
    cols = [
        "scan_timestamp", "ticker", "earnings_date",
        "tier", "passed", "near_miss", "scanner_reason", "display_status",
        "price", "volume", "market_cap",
        "strike", "near_expiry", "far_expiry",
        "days_to_expiry", "total_open_interest",
        "near_bid", "near_ask", "far_bid", "far_ask",
        "near_entry", "far_entry",
        "net_debit", "net_debit_bid", "net_debit_mid", "net_debit_ask",
        "atm_iv_near", "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair",
        "actual_to_fair_ratio", "iv_rv_ratio", "hist_vol_3m",
        "term_slope", "term_structure_valid",
        "expected_move_pct", "expected_move_dollars", "straddle_price",
        "atm_call_delta", "atm_put_delta", "atm_call_iv", "atm_put_iv",
        "win_rate", "win_quarters",
        "model_expected_return", "model_decision", "model_rejection_reasons",
        "selected_by_bot", "features_json",
        "exit_value", "pnl_dollars", "return_on_debit", "outcome_fetched_at",
    ]
    placeholders = ", ".join(f":{c}" for c in cols)
    sql = f"INSERT INTO live_calendar_candidates ({', '.join(cols)}) VALUES ({placeholders})"
    cur = conn.execute(sql, {c: row.get(c) for c in cols})
    conn.commit()
    return cur.lastrowid or 0


def insert_scanner_output(conn: sqlite3.Connection, row: dict) -> int:
    """Insert a scanner output row for backtest/audit purposes."""
    cols = [
        "scan_timestamp", "ticker", "earnings_date",
        "tier", "passed", "near_miss", "scanner_reason", "display_status",
        "price", "volume", "market_cap",
        "strike", "near_expiry", "far_expiry",
        "days_to_expiry", "total_open_interest",
        "net_debit", "net_debit_bid", "net_debit_mid", "net_debit_ask",
        "atm_iv_near", "sigma_baseline_1y", "sigma_short_leg", "sigma_short_leg_fair",
        "actual_to_fair_ratio", "iv_rv_ratio",
        "term_slope", "term_structure_valid",
        "expected_move_pct", "expected_move_dollars",
        "win_rate", "win_quarters",
        "model_expected_return", "model_decision", "model_rejection_reasons",
        "selected_by_bot", "features_json",
        "exit_value", "pnl_dollars", "return_on_debit", "outcome_fetched_at",
    ]
    placeholders = ", ".join(f":{c}" for c in cols)
    sql = f"INSERT INTO scanner_scan_outputs ({', '.join(cols)}) VALUES ({placeholders})"
    cur = conn.execute(sql, {c: row.get(c) for c in cols})
    conn.commit()
    return cur.lastrowid or 0


def update_outcome(conn: sqlite3.Connection, snapshot_id: int, outcome: dict) -> None:
    """Write outcome data back to a snapshot row."""
    conn.execute(
        """UPDATE snapshots SET
            pre_earnings_close = :pre_earnings_close,
            post_earnings_close = :post_earnings_close,
            actual_move_pct = :actual_move_pct,
            actual_move_direction = :actual_move_direction,
            max_intraday_range_pct = :max_intraday_range_pct,
            outcome_fetched_at = :outcome_fetched_at
        WHERE id = :id""",
        {**outcome, "id": snapshot_id},
    )
    conn.commit()


def insert_scan_run(conn: sqlite3.Connection, row: dict) -> int:
    """Insert a scan-run audit row and return its id."""
    cols = [
        "scan_timestamp", "scanner_name", "trigger_type",
        "candidate_count", "tier1_count", "tier2_count", "take_count",
        "duration_secs", "success", "error_message",
    ]
    placeholders = ", ".join(f":{c}" for c in cols)
    sql = f"INSERT INTO scan_runs ({', '.join(cols)}) VALUES ({placeholders})"
    cur = conn.execute(sql, {c: row.get(c) for c in cols})
    conn.commit()
    return cur.lastrowid or 0
