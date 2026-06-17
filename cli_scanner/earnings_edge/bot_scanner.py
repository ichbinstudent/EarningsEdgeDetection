"""
Earnings Calendar Scanner — adapts earnings_edge.scanner.EarningsScanner
to the BaseScanner interface for the Telegram bot.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import joblib
import numpy as np
import yfinance as yf

from earnings_edge.base import BaseScanner
from earnings_edge.calendar_spread import select_calendar_expiries
from earnings_edge.calendar_filter import (
    CalendarModelScore,
    data_quality_rejection_reasons,
    score_calendar_trade,
)
from earnings_edge.config import get_logger, session
from earnings_edge.db import get_connection, insert_live_calendar_candidate, insert_scanner_output
from earnings_edge.models import TickerReport, ValidationMetrics
from earnings_edge.scanner import EarningsScanner

logger = get_logger("earnings_bot_scanner")

DEFAULT_CALENDAR_MODEL = Path("data/models/calendar_call_filter_ridge_allfeatures.joblib")
DEFAULT_MODEL_THRESHOLD = 0.20


@dataclass(frozen=True)
class LiveCalendarQuote:
    """Pre-trade call-calendar leg marks used for live ML scoring."""

    strike: float
    near_entry: float
    far_entry: float
    net_debit: float
    near_expiry: str
    far_expiry: str
    near_bid: float
    near_ask: float
    far_bid: float
    far_ask: float
    net_debit_bid: float
    net_debit_mid: float
    net_debit_ask: float

    def asdict(self) -> dict[str, Any]:
        return {
            "strike": self.strike,
            "near_entry": self.near_entry,
            "far_entry": self.far_entry,
            "net_debit": self.net_debit,
            "near_expiry": self.near_expiry,
            "far_expiry": self.far_expiry,
            "near_bid": self.near_bid,
            "near_ask": self.near_ask,
            "far_bid": self.far_bid,
            "far_ask": self.far_ask,
            "net_debit_bid": self.net_debit_bid,
            "net_debit_mid": self.net_debit_mid,
            "net_debit_ask": self.net_debit_ask,
        }


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _option_bid_ask(row: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    if bid is not None and ask is not None and bid >= 0 and ask > 0 and ask >= bid:
        return bid, ask
    return None


def _option_mid(row: Mapping[str, Any]) -> Optional[float]:
    quote = _option_bid_ask(row)
    if quote is not None:
        bid, ask = quote
        return (bid + ask) / 2.0
    last = _safe_float(row.get("lastPrice"))
    return last if last is not None and last > 0 else None


def select_live_calendar_call_quote(
    ticker: str,
    price: float,
    earnings_date: Optional[date],
) -> Optional[LiveCalendarQuote]:
    """Select the live near/far ATM call calendar spread for bot ML scoring.

    The backtest uses Polygon EOD closes. The bot has live yfinance chains, so this
    uses bid/ask mids as a display/ranking estimate rather than an executable quote.
    """

    try:
        yt = yf.Ticker(ticker, session=session)
        today = datetime.now().date()
        anchor = earnings_date or today
        expiries = [
            exp
            for exp in sorted(yt.options)
            if datetime.strptime(exp, "%Y-%m-%d").date() >= anchor
        ]
        if len(expiries) < 2:
            return None
        near_expiry_date, far_expiry_date = select_calendar_expiries(
            datetime.strptime(exp, "%Y-%m-%d").date() for exp in expiries
        )
        near_expiry = near_expiry_date.isoformat()
        far_expiry = far_expiry_date.isoformat()
        near_calls = yt.option_chain(near_expiry).calls
        far_calls = yt.option_chain(far_expiry).calls
        if near_calls.empty or far_calls.empty:
            return None

        near_idx = (near_calls["strike"] - price).abs().idxmin()
        strike = float(near_calls.loc[near_idx, "strike"])
        far_idx = (far_calls["strike"] - strike).abs().idxmin()
        near_row = near_calls.loc[near_idx]
        far_row = far_calls.loc[far_idx]
        near_quote = _option_bid_ask(near_row)
        far_quote = _option_bid_ask(far_row)
        if near_quote is None or far_quote is None:
            return None
        near_bid, near_ask = near_quote
        far_bid, far_ask = far_quote
        near_entry = (near_bid + near_ask) / 2.0
        far_entry = (far_bid + far_ask) / 2.0
        net_debit_bid = far_bid - near_ask
        net_debit_mid = far_entry - near_entry
        net_debit_ask = far_ask - near_bid
        net_debit = net_debit_ask
        if net_debit <= 0:
            return None
        return LiveCalendarQuote(
            strike=strike,
            near_entry=float(near_entry),
            far_entry=float(far_entry),
            net_debit=float(net_debit),
            near_expiry=near_expiry,
            far_expiry=far_expiry,
            near_bid=float(near_bid),
            near_ask=float(near_ask),
            far_bid=float(far_bid),
            far_ask=float(far_ask),
            net_debit_bid=float(net_debit_bid),
            net_debit_mid=float(net_debit_mid),
            net_debit_ask=float(net_debit_ask),
        )
    except Exception as exc:
        logger.info("Could not build live calendar quote for %s: %s", ticker, exc)
        return None


def build_calendar_model_feature_row(
    report: TickerReport,
    quote: Mapping[str, Any],
) -> dict[str, Any]:
    """Map scanner metrics + live call-calendar marks to model feature names."""

    m: ValidationMetrics = report.metrics
    atm_iv = _safe_float(m.sigma_short_leg)
    iv_rv = _safe_float(m.iv_rv_ratio)
    rv_estimate = atm_iv / iv_rv if atm_iv is not None and iv_rv not in (None, 0) else None
    expected_move_dollars = _safe_float(m.expected_move_dollars)
    row = {
        "ticker": report.ticker,
        "earnings_date": report.earnings_date.isoformat() if report.earnings_date else None,
        "price": _safe_float(m.price),
        "avg_volume_30d": _safe_float(m.volume),
        "market_cap": None,
        "has_options": 1,
        "days_to_expiry": int(m.days_to_expiry or 0),
        "total_open_interest": int(m.open_interest or 0),
        "atm_iv_near": atm_iv,
        "rv30": rv_estimate,
        "iv30_rv30": iv_rv,
        "hist_vol_3m": rv_estimate,
        "term_slope": _safe_float(m.term_structure),
        "term_structure_valid": 1 if (m.term_structure or 0) <= -0.004 else 0,
        "expected_move_pct": _safe_float(m.expected_move_pct),
        "expected_move_dollars": expected_move_dollars,
        "straddle_price": expected_move_dollars,
        "atm_call_delta": _safe_float(m.atm_call_delta),
        "atm_put_delta": _safe_float(m.atm_put_delta),
        "atm_call_iv": atm_iv,
        "atm_put_iv": atm_iv,
        "sigma_baseline_1y": _safe_float(m.sigma_baseline_1y),
        "sigma_short_leg": atm_iv,
        "sigma_short_leg_fair": _safe_float(m.sigma_short_leg_fair),
        "actual_to_fair_ratio": _safe_float(m.actual_to_fair_ratio),
    }
    row.update(dict(quote))
    if "net_debit" not in row or row["net_debit"] is None:
        far_entry = _safe_float(row.get("far_entry"))
        near_entry = _safe_float(row.get("near_entry"))
        row["net_debit"] = far_entry - near_entry if far_entry is not None and near_entry is not None else None
    return row


def format_model_prediction(score: CalendarModelScore, row: Mapping[str, Any]) -> str:
    """Human-readable Telegram line for an expected-return model score."""

    decision = "TAKE" if score.recommended else "SKIP"
    debit = _safe_float(row.get("net_debit")) or 0.0
    debit_bid = _safe_float(row.get("net_debit_bid"))
    debit_mid = _safe_float(row.get("net_debit_mid"))
    debit_ask = _safe_float(row.get("net_debit_ask"))
    strike = _safe_float(row.get("strike")) or 0.0
    if debit_bid is not None and debit_mid is not None and debit_ask is not None:
        debit_text = f"debit ask ${debit_ask:.2f}, mid ${debit_mid:.2f}, bid ${debit_bid:.2f}"
    else:
        debit_text = f"debit ${debit:.2f}"
    return (
        f"• ML Exp Return: {score.probability:+.1%} → {decision} "
        f"({debit_text}, strike ${strike:.2f})"
    )


def load_calendar_model(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        logger.warning("Calendar model not found: %s", path)
        return None
    artifact = joblib.load(path)
    if "pipeline" not in artifact or "features" not in artifact:
        raise RuntimeError(f"Invalid calendar model artifact: {path}")
    return artifact


class EarningsCalendarScanner(BaseScanner):
    """
    US earnings-based options scanner.
    Runs daily at 16:15 EST (21:15 UTC) on weekdays.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_threshold: Optional[float] = None,
        db_path: Optional[Path] = None,
    ):
        super().__init__("Earnings Calendar")
        self._scanner = EarningsScanner()
        self._db_path = db_path
        configured_path = model_path or Path(os.environ.get("EARNINGS_CALENDAR_MODEL", DEFAULT_CALENDAR_MODEL))
        self._model_path = configured_path
        self._model_threshold = (
            model_threshold
            if model_threshold is not None
            else float(os.environ.get("EARNINGS_CALENDAR_MODEL_THRESHOLD", DEFAULT_MODEL_THRESHOLD))
        )
        try:
            self._calendar_model = load_calendar_model(configured_path)
        except Exception:
            logger.exception("Failed to load calendar model")
            self._calendar_model = None

    @property
    def schedule(self) -> str:
        return "15 21 * * 1-5"

    def scan(self) -> Dict[str, Any]:
        """Delegate to ScanService for clean separation of concerns."""
        from earnings_edge.services.scan_service import ScanService
        svc = ScanService(scanner=self)
        return svc.run_scan(trigger="bot")

    def _build_enriched_row(
        self,
        feature_row: Mapping[str, Any],
        *,
        scan_timestamp: str,
        score: Optional[CalendarModelScore],
        rejection_reasons: list[str],
        selected_by_bot: bool,
        report: TickerReport,
        display_status: str,
    ) -> dict[str, Any]:
        """Build a row dict with all scanner metrics, quote data, and model results."""
        m = report.metrics
        model_decision = None
        model_expected_return = None
        if score is not None:
            model_expected_return = score.probability
            model_decision = "TAKE" if score.recommended else "SKIP"
        elif rejection_reasons:
            model_decision = "SKIP"
        return {
            "scan_timestamp": scan_timestamp,
            "ticker": feature_row.get("ticker"),
            "earnings_date": feature_row.get("earnings_date"),
            # Scanner context
            "tier": report.tier,
            "passed": 1 if report.passed else 0,
            "near_miss": 1 if report.near_miss else 0,
            "scanner_reason": report.reason,
            "display_status": display_status,
            # Price / volume
            "price": _safe_float(feature_row.get("price")),
            "volume": _safe_float(m.volume),
            "market_cap": _safe_float(feature_row.get("market_cap")),
            # Options / expiry
            "strike": _safe_float(feature_row.get("strike")),
            "near_expiry": feature_row.get("near_expiry"),
            "far_expiry": feature_row.get("far_expiry"),
            "days_to_expiry": int(m.days_to_expiry or 0),
            "total_open_interest": int(m.open_interest or 0),
            # Leg quotes
            "near_bid": _safe_float(feature_row.get("near_bid")),
            "near_ask": _safe_float(feature_row.get("near_ask")),
            "far_bid": _safe_float(feature_row.get("far_bid")),
            "far_ask": _safe_float(feature_row.get("far_ask")),
            "near_entry": _safe_float(feature_row.get("near_entry")),
            "far_entry": _safe_float(feature_row.get("far_entry")),
            # Debit quotes
            "net_debit": _safe_float(feature_row.get("net_debit")),
            "net_debit_bid": _safe_float(feature_row.get("net_debit_bid")),
            "net_debit_mid": _safe_float(feature_row.get("net_debit_mid")),
            "net_debit_ask": _safe_float(feature_row.get("net_debit_ask")),
            # IV / volatility metrics
            "atm_iv_near": _safe_float(feature_row.get("atm_iv_near")),
            "sigma_baseline_1y": _safe_float(feature_row.get("sigma_baseline_1y")),
            "sigma_short_leg": _safe_float(feature_row.get("sigma_short_leg")),
            "sigma_short_leg_fair": _safe_float(feature_row.get("sigma_short_leg_fair")),
            "actual_to_fair_ratio": _safe_float(feature_row.get("actual_to_fair_ratio")),
            "iv_rv_ratio": _safe_float(feature_row.get("iv30_rv30")),
            "hist_vol_3m": _safe_float(feature_row.get("hist_vol_3m")),
            # Term structure
            "term_slope": _safe_float(feature_row.get("term_slope")),
            "term_structure_valid": _safe_float(feature_row.get("term_structure_valid")),
            # Expected move
            "expected_move_pct": _safe_float(feature_row.get("expected_move_pct")),
            "expected_move_dollars": _safe_float(feature_row.get("expected_move_dollars")),
            "straddle_price": _safe_float(feature_row.get("straddle_price")),
            # ATM greeks
            "atm_call_delta": _safe_float(feature_row.get("atm_call_delta")),
            "atm_put_delta": _safe_float(feature_row.get("atm_put_delta")),
            "atm_call_iv": _safe_float(feature_row.get("atm_call_iv")),
            "atm_put_iv": _safe_float(feature_row.get("atm_put_iv")),
            # Win rate
            "win_rate": _safe_float(m.win_rate),
            "win_quarters": int(m.win_quarters or 0),
            # Model results
            "model_expected_return": model_expected_return,
            "model_decision": model_decision,
            "model_rejection_reasons": ",".join(rejection_reasons) if rejection_reasons else None,
            "selected_by_bot": 1 if selected_by_bot else 0,
            "features_json": json.dumps(dict(feature_row), sort_keys=True, default=str),
        }

    def _persist_live_calendar_candidate(
        self,
        feature_row: Mapping[str, Any],
        *,
        scan_timestamp: str,
        score: Optional[CalendarModelScore],
        rejection_reasons: list[str],
        selected_by_bot: bool,
        report: TickerReport,
        display_status: str,
    ) -> None:
        """Store a live calendar quote/model snapshot for later labeling and retraining."""
        try:
            conn = get_connection(self._db_path)
            row = self._build_enriched_row(
                feature_row,
                scan_timestamp=scan_timestamp,
                score=score,
                rejection_reasons=rejection_reasons,
                selected_by_bot=selected_by_bot,
                report=report,
                display_status=display_status,
            )
            insert_live_calendar_candidate(conn, row)
            conn.close()
        except Exception:
            logger.exception("Failed to persist live calendar candidate for %s", feature_row.get("ticker"))

    def _persist_scanner_output(
        self,
        feature_row: Optional[Mapping[str, Any]],
        *,
        scan_timestamp: str,
        score: Optional[CalendarModelScore],
        rejection_reasons: list[str],
        selected_by_bot: bool,
        report: TickerReport,
        display_status: str,
    ) -> None:
        """Store a scanner output row for backtest/audit (every candidate, including near misses)."""
        try:
            conn = get_connection(self._db_path)
            row = self._build_enriched_row(
                feature_row or {"ticker": report.ticker, "earnings_date": report.earnings_date.isoformat() if report.earnings_date else None},
                scan_timestamp=scan_timestamp,
                score=score,
                rejection_reasons=rejection_reasons,
                selected_by_bot=selected_by_bot,
                report=report,
                display_status=display_status,
            )
            insert_scanner_output(conn, row)
            conn.close()
        except Exception:
            logger.exception("Failed to persist scanner output for %s", report.ticker)

    def _score_report(
        self,
        report: TickerReport,
        *,
        scan_timestamp: str,
        selected_by_bot: bool = True,
    ) -> tuple[Optional[str], Optional[float]]:
        selected_tickers = getattr(self, '_selected_tickers', set())
        display_status = "displayed" if selected_by_bot else "not_displayed"
        if not report.metrics or report.metrics.price <= 0:
            # Still persist a scanner output row even when no price
            self._persist_scanner_output(
                None,
                scan_timestamp=scan_timestamp,
                score=None,
                rejection_reasons=["no_price"],
                selected_by_bot=selected_by_bot,
                report=report,
                display_status=display_status,
            )
            return None, None
        quote = select_live_calendar_call_quote(report.ticker, report.metrics.price, report.earnings_date)
        if quote is None:
            self._persist_scanner_output(
                None,
                scan_timestamp=scan_timestamp,
                score=None,
                rejection_reasons=["no_live_quote"],
                selected_by_bot=selected_by_bot,
                report=report,
                display_status=display_status,
            )
            return "• ML Exp Return: unavailable (no live calendar quote)", None
        feature_row = build_calendar_model_feature_row(report, quote.asdict())
        reasons = data_quality_rejection_reasons(feature_row, require_exit_value=False)
        if reasons:
            self._persist_live_calendar_candidate(
                feature_row,
                scan_timestamp=scan_timestamp,
                score=None,
                rejection_reasons=reasons,
                selected_by_bot=selected_by_bot,
                report=report,
                display_status=display_status,
            )
            self._persist_scanner_output(
                feature_row,
                scan_timestamp=scan_timestamp,
                score=None,
                rejection_reasons=reasons,
                selected_by_bot=selected_by_bot,
                report=report,
                display_status=display_status,
            )
            return f"• ML Exp Return: SKIP ({','.join(reasons)})", None
        if self._calendar_model is None:
            self._persist_live_calendar_candidate(
                feature_row,
                scan_timestamp=scan_timestamp,
                score=None,
                rejection_reasons=["model_unavailable"],
                selected_by_bot=selected_by_bot,
                report=report,
                display_status=display_status,
            )
            self._persist_scanner_output(
                feature_row,
                scan_timestamp=scan_timestamp,
                score=None,
                rejection_reasons=["model_unavailable"],
                selected_by_bot=selected_by_bot,
                report=report,
                display_status=display_status,
            )
            return None, None
        score = score_calendar_trade(self._calendar_model, feature_row, threshold=self._model_threshold)
        self._persist_live_calendar_candidate(
            feature_row,
            scan_timestamp=scan_timestamp,
            score=score,
            rejection_reasons=[],
            selected_by_bot=selected_by_bot,
            report=report,
            display_status=display_status,
        )
        self._persist_scanner_output(
            feature_row,
            scan_timestamp=scan_timestamp,
            score=score,
            rejection_reasons=[],
            selected_by_bot=selected_by_bot,
            report=report,
            display_status=display_status,
        )
        return format_model_prediction(score, feature_row), score.probability
