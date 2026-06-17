from datetime import date
import unittest

from earnings_edge.bot_scanner import LiveCalendarQuote, build_calendar_model_feature_row, format_model_prediction
from earnings_edge.calendar_filter import score_calendar_trade
from earnings_edge.models import TickerReport, ValidationMetrics


class FixedRegressionPipeline:
    def __init__(self):
        self.seen_columns = None

    def predict(self, frame):
        self.seen_columns = list(frame.columns)
        return [0.42]


class BotScannerModelTests(unittest.TestCase):
    def test_build_calendar_model_feature_row_maps_report_metrics_and_live_calendar_quote(self):
        report = TickerReport(
            ticker="ABC",
            passed=True,
            tier=1,
            near_miss=False,
            reason="Tier 1 Trade",
            earnings_date=date(2025, 1, 15),
            metrics=ValidationMetrics(
                price=100.0,
                volume=2_500_000,
                days_to_expiry=3,
                open_interest=12_000,
                term_structure=-0.007,
                iv_rv_ratio=1.25,
                expected_move_dollars=4.0,
                expected_move_pct=4.0,
                sigma_short_leg=0.80,
                sigma_short_leg_fair=0.60,
                atm_call_delta=0.52,
                atm_put_delta=-0.48,
            ),
        )
        quote = {
            "strike": 100.0,
            "near_entry": 1.20,
            "far_entry": 2.00,
            "net_debit": 1.05,
            "net_debit_mid": 0.80,
            "net_debit_ask": 1.05,
            "near_expiry": "2025-01-17",
            "far_expiry": "2025-01-24",
        }

        row = build_calendar_model_feature_row(report, quote)

        self.assertEqual(row["ticker"], "ABC")
        self.assertEqual(row["price"], 100.0)
        self.assertEqual(row["avg_volume_30d"], 2_500_000)
        self.assertEqual(row["has_options"], 1)
        self.assertEqual(row["total_open_interest"], 12_000)
        self.assertEqual(row["term_structure_valid"], 1)
        self.assertEqual(row["expected_move_pct"], 4.0)
        self.assertEqual(row["straddle_price"], 4.0)
        self.assertEqual(row["atm_iv_near"], 0.80)
        self.assertAlmostEqual(row["rv30"], 0.64)
        self.assertEqual(row["net_debit"], 1.05)
        self.assertEqual(row["net_debit_mid"], 0.80)
        self.assertEqual(row["near_expiry"], "2025-01-17")

    def test_expected_return_model_score_formats_as_take_line(self):
        artifact = {
            "pipeline": FixedRegressionPipeline(),
            "features": ["price", "net_debit", "debit_pct_price"],
            "target": "expected_return",
            "score_kind": "expected_return",
        }
        row = {
            "price": 100.0,
            "strike": 100.0,
            "near_entry": 1.20,
            "far_entry": 2.00,
            "net_debit": 1.05,
            "net_debit_bid": 0.55,
            "net_debit_mid": 0.80,
            "net_debit_ask": 1.05,
            "near_expiry": "2025-01-17",
            "far_expiry": "2025-01-24",
        }

        score = score_calendar_trade(artifact, row, threshold=0.20)
        line = format_model_prediction(score, row)

        self.assertEqual(artifact["pipeline"].seen_columns, ["price", "net_debit", "debit_pct_price"])
        self.assertEqual(line, "• ML Exp Return: +42.0% → TAKE (debit ask $1.05, mid $0.80, bid $0.55, strike $100.00)")

    def test_live_calendar_quote_uses_executable_ask_debit_for_model_features(self):
        quote = LiveCalendarQuote(
            strike=13.0,
            near_entry=0.985,
            far_entry=1.29,
            net_debit=0.83,
            near_expiry="2026-05-29",
            far_expiry="2026-06-26",
            near_bid=0.90,
            near_ask=1.07,
            far_bid=0.85,
            far_ask=1.73,
            net_debit_bid=-0.22,
            net_debit_mid=0.305,
            net_debit_ask=0.83,
        )

        row = build_calendar_model_feature_row(
            TickerReport(
                ticker="KSS",
                passed=True,
                tier=1,
                near_miss=False,
                reason="Tier 1 Trade",
                earnings_date=date(2026, 5, 26),
                metrics=ValidationMetrics(price=13.0),
            ),
            quote.asdict(),
        )

        self.assertEqual(row["net_debit"], 0.83)
        self.assertEqual(row["net_debit_ask"], 0.83)
        self.assertAlmostEqual(row["net_debit_mid"], 0.305)


if __name__ == "__main__":
    unittest.main()
