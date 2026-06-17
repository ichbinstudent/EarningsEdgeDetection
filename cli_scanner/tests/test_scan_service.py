"""Unit tests for :mod:`earnings_edge.services.scan_service`.

Uses a fake :class:`EarningsCalendarScanner`-like object and a real temporary
SQLite database so the audit-logging path is exercised end to end without any
network access.
"""

import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Optional

from earnings_edge.models import (
    NearMiss,
    ScanResult,
    TickerReport,
    ValidationMetrics,
)
from earnings_edge.services.scan_service import ScanService


def _metrics(
    *,
    price: float = 100.0,
    volume: float = 2_000_000,
    win_rate: float = 75.0,
    win_quarters: int = 8,
    iv_rv_ratio: float = 1.4,
    term_structure: float = -0.01,
    sigma_baseline_1y: Optional[float] = 0.35,
    sigma_short_leg: Optional[float] = 0.40,
    sigma_short_leg_fair: Optional[float] = 0.45,
    actual_to_fair_ratio: Optional[float] = 88.9,
) -> ValidationMetrics:
    return ValidationMetrics(
        price=price,
        volume=volume,
        win_rate=win_rate,
        win_quarters=win_quarters,
        iv_rv_ratio=iv_rv_ratio,
        term_structure=term_structure,
        sigma_baseline_1y=sigma_baseline_1y,
        sigma_short_leg=sigma_short_leg,
        sigma_short_leg_fair=sigma_short_leg_fair,
        actual_to_fair_ratio=actual_to_fair_ratio,
    )


def _report(
    ticker: str,
    *,
    tier: int = 1,
    passed: bool = True,
    near_miss: bool = False,
    reason: str = "ok",
    metrics: Optional[ValidationMetrics] = None,
    earnings_date: Optional[date] = None,
) -> TickerReport:
    return TickerReport(
        ticker=ticker,
        passed=passed,
        tier=tier,
        near_miss=near_miss,
        reason=reason,
        metrics=metrics or _metrics(),
        earnings_date=earnings_date or date(2026, 6, 17),
    )


class FakeInnerScanner:
    """Stand-in for the raw EarningsScanner (``EarningsCalendarScanner._scanner``)."""

    def __init__(self, result: ScanResult, raise_exc: Optional[Exception] = None):
        self._result = result
        self._raise = raise_exc
        self.scan_calls: list[dict] = []

    def scan_earnings(self, workers: int = 8, use_finnhub: bool = True) -> ScanResult:
        self.scan_calls.append({"workers": workers, "use_finnhub": use_finnhub})
        if self._raise is not None:
            raise self._raise
        return self._result


class FakeCalendarScanner:
    """Fake :class:`EarningsCalendarScanner` mirroring its public surface.

    Exposes ``_scanner`` (a :class:`FakeInnerScanner`) plus ``_score_report``
    and ``_persist_scanner_output`` so the service can drive the full pipeline
    deterministically.
    """

    def __init__(self, result: ScanResult, raise_exc: Optional[Exception] = None):
        self._scanner = FakeInnerScanner(result, raise_exc=raise_exc)
        self.score_calls: list[tuple] = []
        self.persist_calls: list[tuple] = []

    def _score_report(
        self,
        report: TickerReport,
        *,
        scan_timestamp: str,
        selected_by_bot: bool = True,
    ) -> tuple[Optional[str], Optional[float]]:
        self.score_calls.append((report.ticker, selected_by_bot))
        # AAPL always TAKEs; other selected tickers SKIP; non-selected → None.
        if report.ticker == "AAPL":
            return (
                "• ML Exp Return: +25.0% → TAKE (debit $1.50, strike $100.00)",
                0.25,
            )
        if selected_by_bot:
            return (
                "• ML Exp Return: +5.0% → SKIP (debit $2.00, strike $50.00)",
                0.05,
            )
        return None, None

    def _persist_scanner_output(
        self,
        feature_row,
        *,
        scan_timestamp: str,
        score,
        rejection_reasons,
        selected_by_bot: bool,
        report: TickerReport,
        display_status: str,
    ) -> None:
        self.persist_calls.append((report.ticker, display_status))


def _build_result() -> ScanResult:
    reports = {
        "AAPL": _report("AAPL", tier=1, passed=True),
        "MSFT": _report("MSFT", tier=2, passed=True),
        "NVDA": _report(
            "NVDA", tier=0, passed=False, near_miss=True, reason="volume too low"
        ),
        "ORCL": _report("ORCL", tier=0, passed=False),  # non-selected filler
    }
    return ScanResult(
        tier1=["AAPL"],
        tier2=["MSFT"],
        near_misses=[NearMiss(ticker="NVDA", reason="volume too low")],
        reports=reports,
    )


class TestScanService(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        self.result = _build_result()
        self.fake = FakeCalendarScanner(self.result)
        self.service = ScanService(scanner=self.fake, db_path=self.db_path)

    # -- success path -----------------------------------------------------

    def test_run_scan_returns_expected_structure(self):
        out = self.service.run_scan(trigger="bot")
        self.assertTrue(out["success"])
        self.assertIn("embed", out)
        self.assertIsInstance(out["scan_run_id"], int)
        self.assertGreater(out["scan_run_id"], 0)
        self.assertIn("stats", out)

    def test_embed_has_title_color_and_timestamp(self):
        out = self.service.run_scan()
        embed = out["embed"]
        self.assertEqual(embed["title"], "Earnings Scanner Results")
        self.assertEqual(embed["color"], 3066993)
        self.assertIn("timestamp", embed)
        self.assertIn("fields", embed)

    def test_stats_counts_match_scan_result(self):
        out = self.service.run_scan()
        stats = out["stats"]
        # 4 reports in the fake result.
        self.assertEqual(stats["candidate_count"], 4)
        self.assertEqual(stats["tier1_count"], 1)
        self.assertEqual(stats["tier2_count"], 1)
        # Only AAPL produces a TAKE.
        self.assertEqual(stats["take_count"], 1)
        self.assertGreaterEqual(stats["duration_secs"], 0.0)

    def test_scan_earns_called_with_default_params(self):
        self.service.run_scan(trigger="cron")
        self.assertEqual(self.fake._scanner.scan_calls[0]["workers"], 8)
        self.assertTrue(self.fake._scanner.scan_calls[0]["use_finnhub"])

    def test_take_summary_inserted_at_top_of_fields(self):
        out = self.service.run_scan()
        fields = out["embed"]["fields"]
        self.assertEqual(fields[0]["name"], "Summary — TAKE trades only")
        self.assertIn("AAPL", fields[0]["value"])
        self.assertIn("→ TAKE", fields[0]["value"])

    def test_selected_and_near_miss_reports_routed_correctly(self):
        self.service.run_scan()
        scored = {t for t, _ in self.fake.score_calls}
        # AAPL + MSFT (selected) and ORCL (non-selected) all get scored.
        self.assertEqual(scored, {"AAPL", "MSFT", "ORCL"})
        # NVDA is a near miss → persisted as near_miss, not scored.
        persisted = {t for t, status in self.fake.persist_calls}
        self.assertIn("NVDA", persisted)
        nvda_status = [s for t, s in self.fake.persist_calls if t == "NVDA"][0]
        self.assertEqual(nvda_status, "near_miss")

    def test_near_miss_field_present(self):
        out = self.service.run_scan()
        names = [f["name"] for f in out["embed"]["fields"]]
        self.assertIn("Near Miss — NVDA", names)

    def test_scan_run_logged_to_db_with_trigger(self):
        out = self.service.run_scan(trigger="manual")
        from earnings_edge.db import get_connection

        conn = get_connection(self.db_path)
        row = conn.execute(
            "SELECT * FROM scan_runs WHERE id = ?", (out["scan_run_id"],)
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row["trigger_type"], "manual")
        self.assertEqual(row["scanner_name"], "Earnings Calendar")
        self.assertEqual(row["success"], 1)
        self.assertEqual(row["candidate_count"], 4)
        self.assertEqual(row["take_count"], 1)
        self.assertIsNone(row["error_message"])

    # -- failure path -----------------------------------------------------

    def test_run_scan_handles_exception_and_logs_failure(self):
        fake = FakeCalendarScanner(self.result, raise_exc=RuntimeError("boom"))
        service = ScanService(scanner=fake, db_path=self.db_path)
        out = service.run_scan(trigger="bot")

        self.assertFalse(out["success"])
        self.assertIn("boom", out["error"])
        # The failed attempt is still logged.
        self.assertIsInstance(out["scan_run_id"], int)
        self.assertGreater(out["scan_run_id"], 0)

        from earnings_edge.db import get_connection

        conn = get_connection(self.db_path)
        row = conn.execute(
            "SELECT * FROM scan_runs WHERE id = ?", (out["scan_run_id"],)
        ).fetchone()
        conn.close()
        self.assertEqual(row["success"], 0)
        self.assertIn("boom", row["error_message"])

    # -- empty result -----------------------------------------------------

    def test_empty_result_yields_no_recommendations_field(self):
        fake = FakeCalendarScanner(ScanResult())
        service = ScanService(scanner=fake, db_path=self.db_path)
        out = service.run_scan()
        self.assertTrue(out["success"])
        fields = out["embed"]["fields"]
        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0]["name"], "No recommendations")
        self.assertEqual(out["stats"]["candidate_count"], 0)
        self.assertEqual(out["stats"]["take_count"], 0)


if __name__ == "__main__":
    unittest.main()
