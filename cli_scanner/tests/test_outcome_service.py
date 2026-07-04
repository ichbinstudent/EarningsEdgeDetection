"""Unit tests for :mod:`earnings_edge.services.outcome_service`.

Uses a fake Polygon client and a real temporary SQLite database so the full
outcome-tracking pipeline is exercised without any network access.
"""

import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from earnings_edge.db import (
    get_connection,
    insert_snapshot,
)
from earnings_edge.services.outcome_service import OutcomeService


class FakePolygon:
    """Stand-in for :class:`PolygonClient` — returns canned bars."""

    def __init__(self, bars: Optional[list[dict]] = None):
        self.bars = bars if bars is not None else []
        self.calls: list[tuple] = []

    def get_daily_bars(self, ticker: str, from_date: str, to_date: str) -> list[dict]:
        self.calls.append((ticker, from_date, to_date))
        return self.bars


def _bar(day: date, c: float, h: float, l: float) -> dict:
    """Build a Polygon-style agg bar dict for a given calendar day."""
    ts = int(datetime(day.year, day.month, day.day).timestamp() * 1000)
    return {"t": ts, "c": c, "h": h, "l": l}


class TestOutcomeFromBars(unittest.TestCase):
    """Tests for the pure :meth:`outcome_from_bars` transformation."""

    def test_up_move_classification(self):
        ed = date(2026, 6, 10)
        bars = [
            _bar(ed - timedelta(days=1), c=100.0, h=101.0, l=99.0),  # pre
            _bar(ed, c=110.0, h=112.0, l=108.0),  # earnings day (post)
            _bar(ed + timedelta(days=1), c=111.0, h=113.0, l=109.0),
        ]
        out = OutcomeService.outcome_from_bars(bars, ed)
        self.assertIsNotNone(out)
        self.assertEqual(out["pre_earnings_close"], 100.0)
        self.assertEqual(out["post_earnings_close"], 110.0)
        self.assertAlmostEqual(out["actual_move_pct"], 10.0, places=2)
        self.assertEqual(out["actual_move_direction"], "UP")
        # Max range across earnings day + 2 following: day1 (112-108)=4%, day2
        # (113-109)=4% → both 4.0%.
        self.assertAlmostEqual(out["max_intraday_range_pct"], 4.0, places=2)

    def test_down_move_classification(self):
        ed = date(2026, 6, 10)
        bars = [
            _bar(ed - timedelta(days=1), c=100.0, h=101.0, l=99.0),
            _bar(ed, c=94.0, h=96.0, l=92.0),
        ]
        out = OutcomeService.outcome_from_bars(bars, ed)
        self.assertIsNotNone(out)
        self.assertEqual(out["actual_move_direction"], "DOWN")
        self.assertAlmostEqual(out["actual_move_pct"], -6.0, places=2)

    def test_flat_classification(self):
        ed = date(2026, 6, 10)
        bars = [
            _bar(ed - timedelta(days=1), c=100.0, h=101.0, l=99.0),
            _bar(ed, c=100.2, h=101.0, l=99.0),  # +0.2% → FLAT
        ]
        out = OutcomeService.outcome_from_bars(bars, ed)
        self.assertEqual(out["actual_move_direction"], "FLAT")

    def test_insufficient_bars_returns_none(self):
        self.assertIsNone(OutcomeService.outcome_from_bars([], date(2026, 6, 10)))
        self.assertIsNone(
            OutcomeService.outcome_from_bars([_bar(date(2026, 6, 10), 1, 2, 0)], date(2026, 6, 10))
        )

    def test_zero_pre_close_returns_none(self):
        ed = date(2026, 6, 10)
        bars = [
            _bar(ed - timedelta(days=1), c=0.0, h=1.0, l=0.0),
            _bar(ed, c=10.0, h=11.0, l=9.0),
        ]
        self.assertIsNone(OutcomeService.outcome_from_bars(bars, ed))


class TestComputeOutcome(unittest.TestCase):
    def test_uses_polygon_client_for_bars(self):
        ed = date(2026, 6, 10)
        bars = [
            _bar(ed - timedelta(days=1), c=100.0, h=101.0, l=99.0),
            _bar(ed, c=115.0, h=116.0, l=113.0),
        ]
        fake = FakePolygon(bars)
        service = OutcomeService(polygon_client=fake)

        out = service.compute_outcome("AAPL", "2026-06-10")
        self.assertIsNotNone(out)
        # The client was queried with a wide window around the earnings date.
        self.assertEqual(len(fake.calls), 1)
        ticker, from_d, to_d = fake.calls[0]
        self.assertEqual(ticker, "AAPL")
        self.assertEqual(from_d, "2026-06-03")  # ed - 7 days
        self.assertEqual(to_d, "2026-06-13")  # ed + 3 days

    def test_returns_none_when_no_bars(self):
        service = OutcomeService(polygon_client=FakePolygon([]))
        self.assertIsNone(service.compute_outcome("AAPL", "2026-06-10"))


class TestRunOutcomes(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        self.earnings_date = (date.today() - timedelta(days=10)).isoformat()

    def _insert_snapshot(self, ticker: str = "AAPL") -> int:
        conn = get_connection(self.db_path)
        try:
            return insert_snapshot(
                conn,
                {
                    "ticker": ticker,
                    "earnings_date": self.earnings_date,
                    "scan_date": self.earnings_date,
                    "timing": "Post Market",
                },
            )
        finally:
            conn.close()

    def test_updates_snapshot_when_bars_available(self):
        self._insert_snapshot("AAPL")
        ed = datetime.strptime(self.earnings_date, "%Y-%m-%d").date()
        bars = [
            _bar(ed - timedelta(days=1), c=100.0, h=101.0, l=99.0),
            _bar(ed, c=110.0, h=112.0, l=108.0),
        ]
        service = OutcomeService(polygon_client=FakePolygon(bars), db_path=self.db_path)

        stats = service.run_outcomes(min_age_days=2)
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(stats["processed"], 1)

        conn = get_connection(self.db_path)
        row = conn.execute(
            "SELECT actual_move_pct, actual_move_direction, outcome_fetched_at "
            "FROM snapshots WHERE ticker = ?",
            ("AAPL",),
        ).fetchone()
        conn.close()
        self.assertAlmostEqual(row["actual_move_pct"], 10.0, places=1)
        self.assertEqual(row["actual_move_direction"], "UP")
        self.assertNotEqual(row["outcome_fetched_at"], "unavailable")
        self.assertIsNotNone(row["outcome_fetched_at"])

    def test_marks_unavailable_after_max_retries(self):
        self._insert_snapshot("AAPL")
        # No bars → always fails.
        service = OutcomeService(polygon_client=FakePolygon([]), db_path=self.db_path)

        # Each run processes the pending snapshot once. After ``max_retries``
        # consecutive no-data runs (2 here) it is marked permanently
        # unavailable and drops out of the pending set.
        first = service.run_outcomes(min_age_days=2, max_retries=2)
        self.assertEqual(first["failed"], 1)

        conn = get_connection(self.db_path)
        mid = conn.execute(
            "SELECT outcome_fetched_at, outcome_attempt_count "
            "FROM snapshots WHERE ticker = ?",
            ("AAPL",),
        ).fetchone()
        conn.close()
        # After a single run it is only bumped, not yet unavailable.
        self.assertIsNone(mid["outcome_fetched_at"])
        self.assertEqual(mid["outcome_attempt_count"], 1)

        second = service.run_outcomes(min_age_days=2, max_retries=2)
        self.assertEqual(second["failed"], 1)

        conn = get_connection(self.db_path)
        row = conn.execute(
            "SELECT outcome_fetched_at, outcome_attempt_count "
            "FROM snapshots WHERE ticker = ?",
            ("AAPL",),
        ).fetchone()
        conn.close()
        self.assertEqual(row["outcome_fetched_at"], "unavailable")
        self.assertEqual(row["outcome_attempt_count"], 2)

        # A third run finds no pending snapshots.
        third = service.run_outcomes(min_age_days=2, max_retries=2)
        self.assertEqual(third["processed"], 0)

    def test_bumps_attempt_count_below_retries(self):
        self._insert_snapshot("AAPL")
        service = OutcomeService(polygon_client=FakePolygon([]), db_path=self.db_path)

        # max_retries=5 → a single failure should bump to 1 but not mark unavailable.
        service.run_outcomes(min_age_days=2, max_retries=5)
        conn = get_connection(self.db_path)
        row = conn.execute(
            "SELECT outcome_fetched_at, outcome_attempt_count FROM snapshots WHERE ticker = ?",
            ("AAPL",),
        ).fetchone()
        conn.close()
        self.assertIsNone(row["outcome_fetched_at"])
        self.assertEqual(row["outcome_attempt_count"], 1)

    def test_limit_caps_processing(self):
        for t in ("AAPL", "MSFT", "NVDA"):
            self._insert_snapshot(t)
        ed = datetime.strptime(self.earnings_date, "%Y-%m-%d").date()
        bars = [
            _bar(ed - timedelta(days=1), c=100.0, h=101.0, l=99.0),
            _bar(ed, c=110.0, h=112.0, l=108.0),
        ]
        service = OutcomeService(polygon_client=FakePolygon(bars), db_path=self.db_path)

        stats = service.run_outcomes(min_age_days=2, limit=2)
        self.assertEqual(stats["processed"], 2)
        self.assertEqual(stats["updated"], 2)


class TestRunLiveCandidateOutcomes(unittest.TestCase):
    """Ensures :meth:`OutcomeService.run_live_candidate_outcomes` works end-to-end."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"
        # live_calendar_candidates row needs scan_timestamp + earnings_date + ticker
        self.ed = (date.today() - timedelta(days=10)).isoformat()

    def _insert_candidate(self, ticker: str = "AAPL") -> int:
        conn = get_connection(self.db_path)
        try:
            cur = conn.execute(
                "INSERT INTO live_calendar_candidates "
                "(scan_timestamp, ticker, earnings_date) "
                "VALUES (?, ?, ?)",
                ("2026-06-30T19:15:00Z", ticker, self.ed),
            )
            conn.commit()
            return cur.lastrowid or 0
        finally:
            conn.close()

    def test_updates_live_candidate_when_bars_available(self):
        cid = self._insert_candidate("AAPL")
        ed = datetime.strptime(self.ed, "%Y-%m-%d").date()
        bars = [
            _bar(ed - timedelta(days=1), c=100.0, h=101.0, l=99.0),
            _bar(ed, c=110.0, h=112.0, l=108.0),
        ]
        service = OutcomeService(polygon_client=FakePolygon(bars), db_path=self.db_path)

        stats = service.run_live_candidate_outcomes(min_age_days=2)
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(stats["failed"], 0)

        conn = get_connection(self.db_path)
        row = conn.execute(
            "SELECT actual_move_pct, actual_move_direction "
            "FROM live_calendar_candidates WHERE id = ?",
            (cid,),
        ).fetchone()
        conn.close()
        self.assertAlmostEqual(row["actual_move_pct"], 10.0, places=1)
        self.assertEqual(row["actual_move_direction"], "UP")


if __name__ == "__main__":
    unittest.main()
