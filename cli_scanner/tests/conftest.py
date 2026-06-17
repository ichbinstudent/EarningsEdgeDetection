"""Shared test fixtures for EarningsEdge test suite.

Note: This project uses unittest, not pytest. This file exists so that if
pytest is installed in the future, the fixtures are immediately available.
Until then, the fixtures are importable helpers.
"""

import tempfile
from pathlib import Path
from datetime import date


def make_temp_db() -> Path:
    """Provide a temporary SQLite database path."""
    d = tempfile.mkdtemp()
    return Path(d) / "test.db"


def make_sample_candidate():
    """Return a sample EarningsCandidate for testing."""
    from earnings_edge.models import EarningsCandidate
    return EarningsCandidate(
        ticker="AAPL", timing="Post Market",
        earnings_date=date(2026, 6, 17), source="finnhub",
    )


def make_sample_metrics():
    """Return a sample ValidationMetrics for testing."""
    from earnings_edge.models import ValidationMetrics
    return ValidationMetrics(
        price=150.0,
        volume=5_000_000,
        days_to_expiry=7,
        open_interest=10_000,
        term_structure=-0.008,
        iv_rv_ratio=1.35,
        win_rate=65.0,
        win_quarters=8,
        expected_move_dollars=3.50,
        expected_move_pct=2.33,
    )
