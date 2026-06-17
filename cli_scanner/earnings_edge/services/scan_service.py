"""Service layer for the earnings scan pipeline.

Wraps :class:`earnings_edge.bot_scanner.EarningsCalendarScanner` and exposes a
single :meth:`ScanService.run_scan` entry point that performs the
scan → score → persist → format pipeline, logs an audit row to the
``scan_runs`` table, and returns a structured result ready for delivery
(Telegram embed, CLI, cron, etc.).

This is the Phase 4 extraction of the body of
``EarningsCalendarScanner.scan()`` into a thin, injectable service. The
heavy per-ticker scoring/persistence logic stays on the scanner (so we do not
duplicate it); the service owns orchestration, embed construction, stats
accounting and audit logging.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from earnings_edge.bot_scanner import (
    EarningsCalendarScanner,
    build_calendar_model_feature_row,
    format_model_prediction,
    select_live_calendar_call_quote,
)
from earnings_edge.db import get_connection, insert_scan_run
from earnings_edge.models import ScanResult, TickerReport

logger = logging.getLogger("earnings_edge.services.scan")

EMBED_COLOR = 3066993  # dark green — trading-terminal aesthetic
SCANNER_NAME = "Earnings Calendar"

__all__ = [
    "ScanService",
    # Re-exported scanner helpers the pipeline conceptually depends on. They
    # live on bot_scanner and are exercised transitively via the wrapped
    # scanner; they are re-exported here so service consumers have a single
    # import surface.
    "EarningsCalendarScanner",
    "select_live_calendar_call_quote",
    "build_calendar_model_feature_row",
    "format_model_prediction",
]


class ScanService:
    """Orchestrates a full earnings scan with audit logging.

    Parameters
    ----------
    scanner:
        Optional pre-configured :class:`EarningsCalendarScanner` (or any object
        exposing the same ``scan_earnings`` / ``_score_report`` /
        ``_persist_scanner_output`` surface). If omitted a default one is
        constructed. Injecting a scanner or fake makes the service trivially
        unit-testable without network access.
    db_path:
        Optional override for the SQLite database path used both by the scanner
        and for ``scan_runs`` audit rows.
    """

    def __init__(
        self,
        scanner: Optional[EarningsCalendarScanner] = None,
        db_path: Optional[Any] = None,
    ) -> None:
        self._db_path = db_path
        # ``scanner`` is an EarningsCalendarScanner (or fake). It bundles the
        # raw earnings scanner (``scanner._scanner.scan_earnings``) and the
        # per-ticker score/persist methods (``_score_report``,
        # ``_persist_scanner_output``). Holding the whole object keeps a single
        # injection point for tests.
        self._bot_scanner = scanner or EarningsCalendarScanner(db_path=db_path)

    # -- public API -------------------------------------------------------

    def run_scan(
        self,
        trigger: str = "bot",
        *,
        workers: int = 8,
        use_finnhub: bool = True,
    ) -> dict[str, Any]:
        """Execute a full scan and return a structured result.

        On success returns::

            {"success": True, "embed": {...}, "scan_run_id": int, "stats": {...}}

        On failure ``success`` is ``False`` and an ``error`` key is present, but
        a ``scan_run_id`` is still returned (the failed attempt is logged with
        ``success=0``).
        """
        started = time.monotonic()
        scan_timestamp = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        stats: dict[str, Any] = {
            "candidate_count": 0,
            "tier1_count": 0,
            "tier2_count": 0,
            "take_count": 0,
            "duration_secs": 0.0,
        }
        try:
            result = self._bot_scanner._scanner.scan_earnings(
                workers=workers, use_finnhub=use_finnhub
            )
            fields, take_summary = self._process_result(result, scan_timestamp)
            embed = self._build_embed(fields)
            stats["candidate_count"] = len(result.reports)
            stats["tier1_count"] = len(result.tier1)
            stats["tier2_count"] = len(result.tier2)
            stats["take_count"] = len(take_summary)
            stats["duration_secs"] = round(time.monotonic() - started, 3)
            scan_run_id = self._log_scan_run(
                scan_timestamp, trigger, stats, success=True
            )
            return {
                "success": True,
                "embed": embed,
                "scan_run_id": scan_run_id,
                "stats": stats,
            }
        except Exception as exc:  # noqa: BLE001 — surface a stable result shape
            logger.exception("Earnings scan failed")
            stats["duration_secs"] = round(time.monotonic() - started, 3)
            scan_run_id = self._log_scan_run(
                scan_timestamp, trigger, stats, success=False, error=str(exc)
            )
            return {
                "success": False,
                "error": str(exc),
                "scan_run_id": scan_run_id,
                "stats": stats,
            }

    # -- pipeline (extracted from EarningsCalendarScanner.scan) -----------

    def _process_result(
        self,
        result: ScanResult,
        scan_timestamp: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Score/persist every report and build the ordered embed fields.

        Per-ticker scoring and persistence is delegated to the wrapped scanner
        so the ML/DB logic is not duplicated. Returns the field list and a
        take-summary list (TAKE trades only).
        """
        fields: list[dict[str, Any]] = []
        take_summary: list[str] = []
        scored_reports: list[
            tuple[TickerReport, Optional[str], Optional[float]]
        ] = []

        selected_tickers = set(result.tier1 + result.tier2)
        near_miss_tickers = {nm.ticker for nm in result.near_misses}

        for ticker, report in result.reports.items():
            if report is None or report.metrics is None:
                continue
            if ticker in selected_tickers:
                model_line, model_score = self._bot_scanner._score_report(
                    report, scan_timestamp=scan_timestamp, selected_by_bot=True
                )
                scored_reports.append((report, model_line, model_score))
            elif ticker in near_miss_tickers:
                # Persist near misses to scanner_scan_outputs only (no model score).
                self._bot_scanner._persist_scanner_output(
                    None,
                    scan_timestamp=scan_timestamp,
                    score=None,
                    rejection_reasons=[],
                    selected_by_bot=False,
                    report=report,
                    display_status="near_miss",
                )
            else:
                # Other non-selected reports: score + persist as not_displayed.
                self._bot_scanner._score_report(
                    report, scan_timestamp=scan_timestamp, selected_by_bot=False
                )

        # Selected candidates, ordered by model score then actual→fair ratio.
        for report, model_line, _model_score in sorted(
            scored_reports,
            key=lambda item: (
                item[2] if item[2] is not None else -999.0,
                item[0].metrics.actual_to_fair_ratio or 0,
            ),
            reverse=True,
        ):
            m = report.metrics
            tier_label = "T1" if report.tier == 1 else "T2"
            lines = [
                f"• Price: ${m.price:.2f}",
                f"• Volume: {m.volume:,.0f}",
                f"• Winrate: {m.win_rate:.1f}% over {m.win_quarters} quarters",
                f"• IV/RV Ratio: {m.iv_rv_ratio:.2f}",
                f"• Term Structure: {m.term_structure:.3f}",
                f"• Tier: {tier_label}",
                f"• 1Y ATM IV (Baseline): {m.sigma_baseline_1y:.4f}"
                if m.sigma_baseline_1y is not None
                else "• 1Y ATM IV (Baseline): N/A",
                f"• Fair IV (Short Leg): {m.sigma_short_leg_fair:.4f}"
                if m.sigma_short_leg_fair is not None
                else "• Fair IV (Short Leg): N/A",
                f"• Actual IV (Short Leg): {m.sigma_short_leg:.4f}"
                if m.sigma_short_leg is not None
                else "• Actual IV (Short Leg): N/A",
                f"• Actual→Fair Ratio: {m.actual_to_fair_ratio:.2f}%"
                if m.actual_to_fair_ratio is not None
                else "• Actual→Fair Ratio: N/A",
            ]
            if model_line:
                lines.append(model_line)
                if "→ TAKE" in model_line:
                    take_summary.append(
                        f"• {report.ticker} ({tier_label}) — {model_line}"
                    )
            fields.append(
                {
                    "name": f"{report.ticker} ({tier_label})",
                    "value": "\n".join(lines),
                    "inline": False,
                }
            )

        # Near misses
        for nm in result.near_misses:
            report = result.reports.get(nm.ticker)
            if report is None or report.metrics is None:
                continue
            m = report.metrics
            lines = [
                f"• Failed: {nm.reason}",
                f"• Price: ${m.price:.2f}",
                f"• Volume: {m.volume:,.0f}",
                f"• IV/RV Ratio: {m.iv_rv_ratio:.2f}",
                f"• Term Structure: {m.term_structure:.3f}",
                f"• Actual→Fair Ratio: {m.actual_to_fair_ratio:.2f}%"
                if m.actual_to_fair_ratio is not None
                else "• Actual→Fair Ratio: N/A",
            ]
            fields.append(
                {
                    "name": f"Near Miss — {nm.ticker}",
                    "value": "\n".join(lines),
                    "inline": False,
                }
            )

        if take_summary:
            fields.insert(
                0,
                {
                    "name": "Summary — TAKE trades only",
                    "value": "\n".join(take_summary),
                    "inline": False,
                },
            )

        if not fields:
            fields.append(
                {"name": "No recommendations", "value": "None found today", "inline": False}
            )

        return fields, take_summary

    def _build_embed(self, fields: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "title": "Earnings Scanner Results",
            "color": EMBED_COLOR,
            "fields": fields,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _log_scan_run(
        self,
        scan_timestamp: str,
        trigger: str,
        stats: dict[str, Any],
        *,
        success: bool,
        error: Optional[str] = None,
    ) -> int:
        """Insert a scan_runs audit row; return its id (0 on failure)."""
        row = {
            "scan_timestamp": scan_timestamp,
            "scanner_name": SCANNER_NAME,
            "trigger_type": trigger,
            "candidate_count": stats.get("candidate_count", 0),
            "tier1_count": stats.get("tier1_count", 0),
            "tier2_count": stats.get("tier2_count", 0),
            "take_count": stats.get("take_count", 0),
            "duration_secs": stats.get("duration_secs", 0.0),
            "success": 1 if success else 0,
            "error_message": error,
        }
        try:
            conn = get_connection(self._db_path)
            try:
                return insert_scan_run(conn, row)
            finally:
                conn.close()
        except Exception:
            logger.exception("Failed to log scan_run audit row")
            return 0
