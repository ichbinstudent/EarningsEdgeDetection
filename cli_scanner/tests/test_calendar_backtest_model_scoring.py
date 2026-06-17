import sqlite3
import unittest

from calendar_call_backtest import ensure_schema, score_existing_trades


class FixedPipeline:
    def predict_proba(self, frame):
        return [[0.2, 0.8] for _ in range(len(frame))]


class CalendarBacktestModelScoringTests(unittest.TestCase):
    def test_ensure_schema_adds_model_score_columns(self):
        conn = sqlite3.connect(":memory:")
        try:
            ensure_schema(conn)
            columns = {row[1] for row in conn.execute("pragma table_info(calendar_call_trades)")}
        finally:
            conn.close()

        self.assertIn("model_score", columns)
        self.assertIn("model_recommendation", columns)
        self.assertIn("model_reason", columns)
        self.assertIn("model_name", columns)
        self.assertIn("model_scored_at", columns)

    def test_score_existing_trades_updates_clean_rows_and_rejects_bad_rows(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            ensure_schema(conn)
            conn.execute(
                """
                create table snapshots (
                    id integer primary key,
                    price real,
                    avg_volume_30d real,
                    has_options integer,
                    days_to_expiry integer,
                    atm_iv_near real,
                    rv30 real,
                    iv30_rv30 real,
                    hist_vol_3m real,
                    term_slope real,
                    term_structure_valid integer,
                    expected_move_pct real,
                    expected_move_dollars real,
                    straddle_price real,
                    atm_call_delta real,
                    atm_put_delta real,
                    atm_call_iv real,
                    atm_put_iv real,
                    sigma_short_leg real
                )
                """
            )
            conn.execute(
                "insert into snapshots (id, price, avg_volume_30d, has_options, days_to_expiry, atm_iv_near) values (1, 100, 1000000, 1, 2, 0.8)"
            )
            conn.execute(
                "insert into snapshots (id, price, avg_volume_30d, has_options, days_to_expiry, atm_iv_near) values (2, 100, 1000000, 1, 2, 0.8)"
            )
            base = {
                "ticker": "AAA",
                "earnings_date": "2025-01-15",
                "scan_date": "2025-01-14",
                "near_expiry": "2025-01-17",
                "far_expiry": "2025-01-24",
                "near_call_ticker": "O:AAA250117C00100000",
                "far_call_ticker": "O:AAA250124C00100000",
                "near_entry": 2.0,
                "far_entry": 3.0,
                "near_exit": 0.5,
                "far_exit": 2.0,
                "net_debit": 1.0,
                "exit_value": 1.5,
                "pnl_dollars": 50.0,
                "return_on_debit": 0.5,
            }
            conn.execute(
                """
                insert into calendar_call_trades (
                    snapshot_id, ticker, earnings_date, scan_date, near_expiry, far_expiry, strike,
                    near_call_ticker, far_call_ticker, near_entry, far_entry, near_exit, far_exit,
                    net_debit, exit_value, pnl_dollars, return_on_debit
                ) values (:snapshot_id, :ticker, :earnings_date, :scan_date, :near_expiry, :far_expiry, :strike,
                    :near_call_ticker, :far_call_ticker, :near_entry, :far_entry, :near_exit, :far_exit,
                    :net_debit, :exit_value, :pnl_dollars, :return_on_debit)
                """,
                {**base, "snapshot_id": 1, "strike": 100.0},
            )
            conn.execute(
                """
                insert into calendar_call_trades (
                    snapshot_id, ticker, earnings_date, scan_date, near_expiry, far_expiry, strike,
                    near_call_ticker, far_call_ticker, near_entry, far_entry, near_exit, far_exit,
                    net_debit, exit_value, pnl_dollars, return_on_debit
                ) values (:snapshot_id, :ticker, :earnings_date, :scan_date, :near_expiry, :far_expiry, :strike,
                    :near_call_ticker, :far_call_ticker, :near_entry, :far_entry, :near_exit, :far_exit,
                    :net_debit, :exit_value, :pnl_dollars, :return_on_debit)
                """,
                {**base, "snapshot_id": 2, "strike": 400.0},
            )
            artifact = {
                "pipeline": FixedPipeline(),
                "features": ["price", "strike", "net_debit", "moneyness"],
                "target": "min_pnl",
            }

            summary = score_existing_trades(conn, artifact, model_name="unit", threshold=0.55)

            self.assertEqual(summary, {"scored": 1, "rejected": 1})
            rows = conn.execute(
                "select snapshot_id, model_score, model_recommendation, model_reason from calendar_call_trades order by snapshot_id"
            ).fetchall()
            self.assertAlmostEqual(rows[0]["model_score"], 0.8)
            self.assertEqual(rows[0]["model_recommendation"], 1)
            self.assertEqual(rows[0]["model_reason"], "model_score>=0.55")
            self.assertIsNone(rows[1]["model_score"])
            self.assertEqual(rows[1]["model_recommendation"], 0)
            self.assertEqual(rows[1]["model_reason"], "bad_moneyness")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
