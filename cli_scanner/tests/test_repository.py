"""Tests for the database repository layer."""

import tempfile
import unittest
from pathlib import Path

from earnings_edge.models import EarningsCandidate
from datetime import date


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"

    def test_wal_mode_enabled(self):
        from earnings_edge.db import get_connection
        conn = get_connection(self.db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        self.assertEqual(mode.lower(), "wal")

    def test_connection_sets_busy_timeout(self):
        from earnings_edge.db import get_connection
        conn = get_connection(self.db_path)
        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        conn.close()
        self.assertGreater(timeout, 0)

    def test_scan_runs_table_exists(self):
        from earnings_edge.db import get_connection
        conn = get_connection(self.db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        conn.close()
        self.assertIn("scan_runs", tables)

    def test_insert_scan_run(self):
        from earnings_edge.db import get_connection, insert_scan_run
        conn = get_connection(self.db_path)
        row_id = insert_scan_run(conn, {
            "scan_timestamp": "2026-06-17T12:00:00Z",
            "scanner_name": "Earnings Calendar",
            "trigger_type": "test",
            "candidate_count": 50,
            "tier1_count": 3,
            "tier2_count": 2,
            "take_count": 1,
            "duration_secs": 42.5,
            "success": 1,
        })
        row = conn.execute("SELECT * FROM scan_runs WHERE id = ?", (row_id,)).fetchone()
        conn.close()
        self.assertEqual(row["scanner_name"], "Earnings Calendar")
        self.assertEqual(row["candidate_count"], 50)

    def test_data_source_column_exists_on_snapshots(self):
        from earnings_edge.db import get_connection
        conn = get_connection(self.db_path)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(snapshots)")}
        conn.close()
        self.assertIn("data_source", cols)


class TestEarningsCandidateSource(unittest.TestCase):
    def test_candidate_carries_source(self):
        c = EarningsCandidate(ticker="AAPL", timing="Post Market", source="finnhub")
        self.assertEqual(c.source, "finnhub")

    def test_candidate_default_source(self):
        c = EarningsCandidate(ticker="AAPL", timing="Post Market")
        self.assertEqual(c.source, "unknown")


if __name__ == "__main__":
    unittest.main()
