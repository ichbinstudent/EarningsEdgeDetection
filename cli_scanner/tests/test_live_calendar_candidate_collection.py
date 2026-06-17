from datetime import date
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from earnings_edge.bot_scanner import EarningsCalendarScanner, LiveCalendarQuote
from earnings_edge.db import get_connection, insert_live_calendar_candidate
from earnings_edge.models import NearMiss, ScanResult, TickerReport, ValidationMetrics


class FixedRegressionPipeline:
    def predict(self, frame):
        return [0.42]


class FakeScanner:
    def scan_earnings(self, workers=8, use_finnhub=True):
        return ScanResult(
            tier1=["AAA"],
            tier2=["BBB"],
            near_misses=[NearMiss(ticker="CCC", reason="IV/RV too low")],
            reports={
                "AAA": TickerReport(
                    ticker="AAA",
                    passed=True,
                    tier=1,
                    near_miss=False,
                    reason="Tier 1 Trade",
                    earnings_date=date(2026, 6, 1),
                    metrics=ValidationMetrics(
                        price=100.0,
                        volume=1_000_000,
                        days_to_expiry=3,
                        open_interest=12_000,
                        term_structure=-0.01,
                        iv_rv_ratio=1.3,
                        expected_move_dollars=4.0,
                        expected_move_pct=4.0,
                        sigma_short_leg=0.80,
                        sigma_short_leg_fair=0.65,
                        actual_to_fair_ratio=23.0,
                    ),
                ),
                "BBB": TickerReport(
                    ticker="BBB",
                    passed=True,
                    tier=2,
                    near_miss=False,
                    reason="Tier 2 Trade",
                    earnings_date=date(2026, 6, 2),
                    metrics=ValidationMetrics(
                        price=50.0,
                        volume=800_000,
                        days_to_expiry=4,
                        open_interest=6_000,
                        term_structure=-0.006,
                        iv_rv_ratio=1.2,
                        expected_move_dollars=2.5,
                        expected_move_pct=5.0,
                        sigma_short_leg=0.70,
                        sigma_short_leg_fair=0.60,
                        actual_to_fair_ratio=16.0,
                    ),
                ),
                "CCC": TickerReport(
                    ticker="CCC",
                    passed=False,
                    tier=0,
                    near_miss=True,
                    reason="IV/RV too low",
                    earnings_date=date(2026, 6, 3),
                    metrics=ValidationMetrics(
                        price=25.0,
                        volume=500_000,
                        days_to_expiry=5,
                        open_interest=3_000,
                        term_structure=-0.004,
                        iv_rv_ratio=1.0,
                        expected_move_dollars=1.0,
                        expected_move_pct=4.0,
                        sigma_short_leg=0.50,
                        sigma_short_leg_fair=0.50,
                        actual_to_fair_ratio=0.0,
                    ),
                ),
            },
        )


def quote_for(ticker, price, earnings_date):
    return LiveCalendarQuote(
        strike=price,
        near_entry=1.0,
        far_entry=1.5,
        net_debit=0.8,
        near_expiry="2026-06-05",
        far_expiry="2026-07-02",
        near_bid=0.9,
        near_ask=1.1,
        far_bid=1.2,
        far_ask=1.7,
        net_debit_bid=0.1,
        net_debit_mid=0.5,
        net_debit_ask=0.8,
    )


class LiveCalendarCandidateCollectionTests(unittest.TestCase):
    def test_insert_live_calendar_candidate_creates_table_and_persists_quote_score_and_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(Path(tmp) / "earnings_ml.db")
            row_id = insert_live_calendar_candidate(
                conn,
                {
                    "scan_timestamp": "2026-05-26T15:00:00Z",
                    "ticker": "KSS",
                    "earnings_date": "2026-05-26",
                    "price": 13.0,
                    "strike": 13.0,
                    "near_expiry": "2026-05-29",
                    "far_expiry": "2026-06-26",
                    "near_bid": 0.93,
                    "near_ask": 0.99,
                    "far_bid": 0.85,
                    "far_ask": 1.73,
                    "net_debit_bid": -0.14,
                    "net_debit_mid": 0.33,
                    "net_debit_ask": 0.80,
                    "net_debit": 0.80,
                    "model_expected_return": 0.42,
                    "model_decision": "TAKE",
                    "model_rejection_reasons": None,
                    "selected_by_bot": 1,
                    "features_json": '{"price":13.0}',
                },
            )

            stored = conn.execute("SELECT * FROM live_calendar_candidates WHERE id = ?", (row_id,)).fetchone()
            self.assertEqual(stored["ticker"], "KSS")
            self.assertEqual(stored["far_expiry"], "2026-06-26")
            self.assertEqual(stored["net_debit"], 0.80)
            self.assertEqual(stored["net_debit_ask"], 0.80)
            self.assertEqual(stored["model_decision"], "TAKE")
            self.assertEqual(stored["features_json"], '{"price":13.0}')
            conn.close()

    def test_bot_scan_persists_all_candidates_not_only_displayed_winners(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "earnings_ml.db"
            scanner = EarningsCalendarScanner(model_path=Path("missing.joblib"), model_threshold=0.20, db_path=db_path)
            scanner._scanner = FakeScanner()
            scanner._calendar_model = {
                "pipeline": FixedRegressionPipeline(),
                "features": ["price", "net_debit", "debit_pct_price"],
                "score_kind": "expected_return",
            }

            with patch("earnings_edge.bot_scanner.select_live_calendar_call_quote", side_effect=quote_for):
                result = scanner.scan()

            self.assertTrue(result["success"])
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM live_calendar_candidates ORDER BY ticker").fetchall()
            self.assertEqual([row["ticker"] for row in rows], ["AAA", "BBB"])
            self.assertEqual([row["model_decision"] for row in rows], ["TAKE", "TAKE"])
            self.assertEqual([row["selected_by_bot"] for row in rows], [1, 1])
            self.assertTrue(all(row["features_json"] for row in rows))
            conn.close()

    def test_bot_scan_stores_all_scanner_output_rows_including_near_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "earnings_ml.db"
            scanner = EarningsCalendarScanner(model_path=Path("missing.joblib"), model_threshold=0.20, db_path=db_path)
            scanner._scanner = FakeScanner()
            scanner._calendar_model = {
                "pipeline": FixedRegressionPipeline(),
                "features": ["price", "net_debit", "debit_pct_price"],
                "score_kind": "expected_return",
            }

            with patch("earnings_edge.bot_scanner.select_live_calendar_call_quote", side_effect=quote_for):
                result = scanner.scan()

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM scanner_scan_outputs ORDER BY ticker").fetchall()
            self.assertEqual([row["ticker"] for row in rows], ["AAA", "BBB", "CCC"])
            self.assertEqual([row["display_status"] for row in rows], ["displayed", "displayed", "near_miss"])
            self.assertEqual([row["model_decision"] for row in rows], ["TAKE", "TAKE", None])
            self.assertEqual([row["model_expected_return"] for row in rows], [0.42, 0.42, None])
            self.assertEqual([row["selected_by_bot"] for row in rows], [1, 1, 0])
            self.assertTrue(all(row["features_json"] for row in rows))
            conn.close()


if __name__ == "__main__":
    unittest.main()
