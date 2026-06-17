"""Centralized configuration via dataclass with env-var overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FilterThresholds:
    """Scanner filter gate values — adjustable per-run via env."""
    min_price: float = 3.0
    min_volume: float = 1_500_000
    near_miss_volume: float = 1_000_000
    max_days_to_expiry: int = 30
    min_open_interest: int = 2000
    max_atm_delta: float = 0.57
    min_expected_move: float = 0.90

    # IV/RV defaults (overridden at runtime by SPY adjustment)
    iv_rv_pass: float = 1.25
    iv_rv_near_miss: float = 1.0

    # Term structure
    term_structure_hard_limit: float = -0.004
    term_structure_tier2: float = -0.006


@dataclass(frozen=True)
class Settings:
    """Application-wide settings loaded from environment."""

    # API keys
    polygon_api_key: str = field(default_factory=lambda: os.environ.get("POLYGON_API_KEY", ""))
    finnhub_api_key: str = field(default_factory=lambda: os.environ.get("FINNHUB_API_KEY", ""))
    telegram_bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))

    # Rate limits
    polygon_rate_sleep: float = field(default_factory=lambda: float(os.environ.get("POLYGON_RATE_SLEEP", "13")))
    yfinance_sleep: float = 0.3
    yfinance_batch_size: int = 8
    yfinance_batch_sleep: float = 5.0

    # Model
    calendar_model_path: Path = field(
        default_factory=lambda: Path(os.environ.get(
            "EARNINGS_CALENDAR_MODEL",
            "data/models/calendar_call_filter_ridge_allfeatures.joblib",
        ))
    )
    calendar_model_threshold: float = field(
        default_factory=lambda: float(os.environ.get("EARNINGS_CALENDAR_MODEL_THRESHOLD", "0.20"))
    )

    # Scanner schedule (cron: Berlin TZ)
    scanner_schedule: str = "15 21 * * 1-5"

    # Database
    db_path: Path = field(default_factory=lambda: Path("data/earnings_ml.db"))

    # Filters
    filters: FilterThresholds = field(default_factory=FilterThresholds)

    def validate(self) -> list[str]:
        """Return list of configuration errors (empty = OK)."""
        errors = []
        if not self.polygon_api_key:
            errors.append("POLYGON_API_KEY not set")
        if not self.telegram_bot_token:
            errors.append("TELEGRAM_BOT_TOKEN not set")
        return errors


# Singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
