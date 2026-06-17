"""Typed data models for data flowing between modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


# ── Earnings calendar ─────────────────────────────────────────────────

@dataclass
class EarningsCandidate:
    """A single ticker from the earnings calendar."""
    ticker: str
    timing: str  # "Pre Market", "Post Market", "During Market", "Unknown"
    earnings_date: Optional[date] = None
    source: str = "unknown"  # "investing", "finnhub", "dolthub", "merge"


# ── Options analysis ──────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Output of ``OptionsAnalyzer.compute_recommendation()``."""
    ticker: str
    current_price: float
    recommendation: str  # "BUY", "SELL", "HOLD"
    iv30_rv30: float
    term_slope: float
    term_structure_valid: bool
    term_structure_tier2: bool
    expected_move: str  # "X.XX%" or "N/A"
    avg_volume_pass: bool

    # Optional — populated when data is available
    sigma_baseline_1y: Optional[float] = None
    sigma_short_leg_fair: Optional[float] = None
    sigma_short_leg: Optional[float] = None
    actual_to_fair_ratio: Optional[float] = None
    atm_call_delta: Optional[float] = None
    atm_put_delta: Optional[float] = None
    atm_iv_near: Optional[float] = None
    atm_call_iv: Optional[float] = None
    atm_put_iv: Optional[float] = None
    rv30: Optional[float] = None
    hist_vol_3m: Optional[float] = None

    # Error state
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def fail(cls, ticker: str, msg: str) -> AnalysisResult:
        """Convenience constructor for error results."""
        return cls(
            ticker=ticker, current_price=0, recommendation="",
            iv30_rv30=0, term_slope=0, term_structure_valid=False,
            term_structure_tier2=False, expected_move="N/A",
            avg_volume_pass=False, error=msg,
        )


# ── Market Chameleon ──────────────────────────────────────────────────

@dataclass
class WinRateData:
    """Historical win-rate scraped from Market Chameleon."""
    win_rate: float = 0.0
    quarters: int = 0


# ── Stock validation ──────────────────────────────────────────────────

@dataclass
class ValidationMetrics:
    """Numeric metrics collected during stock validation."""
    price: float = 0.0
    volume: float = 0.0
    days_to_expiry: int = 0
    open_interest: int = 0
    term_structure: float = 0.0
    iv_rv_ratio: float = 0.0
    win_rate: float = 0.0
    win_quarters: int = 0
    expected_move_dollars: float = 0.0
    expected_move_pct: float = 0.0
    sigma_baseline_1y: Optional[float] = None
    sigma_short_leg: Optional[float] = None
    sigma_short_leg_fair: Optional[float] = None
    actual_to_fair_ratio: Optional[float] = None
    atm_call_delta: Optional[float] = None
    atm_put_delta: Optional[float] = None
    tier: int = 0


@dataclass
class ValidationResult:
    """Output of ``StockValidator.validate()``."""
    passed: bool
    tier: int
    near_miss: bool
    reason: str
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)


# ── Iron fly ──────────────────────────────────────────────────────────

@dataclass
class IronFlyResult:
    """Iron fly strike calculation output."""
    short_call_strike: float = 0.0
    short_put_strike: float = 0.0
    long_call_strike: float = 0.0
    long_put_strike: float = 0.0
    short_call_premium: float = 0.0
    short_put_premium: float = 0.0
    long_call_premium: float = 0.0
    long_put_premium: float = 0.0
    total_credit: float = 0.0
    total_debit: float = 0.0
    net_credit: float = 0.0
    put_wing_width: float = 0.0
    call_wing_width: float = 0.0
    max_profit: float = 0.0
    max_risk: float = 0.0
    upper_breakeven: float = 0.0
    lower_breakeven: float = 0.0
    risk_reward_ratio: float = 0.0
    expiration: str = ""

    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ── Orchestrator outputs ──────────────────────────────────────────────

@dataclass
class NearMiss:
    """A ticker that nearly passed validation."""
    ticker: str
    reason: str


@dataclass
class TickerReport:
    """Full analysis for a single ticker — validation + SPY context."""
    ticker: str
    passed: bool
    tier: int
    near_miss: bool
    reason: str
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    spy_iv_rv: float = 0.0
    iv_rv_pass_threshold: float = 0.0
    iv_rv_near_miss_threshold: float = 0.0
    earnings_date: Optional[date] = None

    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ScanResult:
    """Full earnings scan output."""
    tier1: List[str] = field(default_factory=list)
    tier2: List[str] = field(default_factory=list)
    near_misses: List[NearMiss] = field(default_factory=list)
    reports: Dict[str, TickerReport] = field(default_factory=dict)
