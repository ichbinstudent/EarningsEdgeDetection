import unittest

from earnings_edge.calendar_filter import (
    add_calendar_entry_features,
    data_quality_rejection_reasons,
    score_calendar_trade,
)


class FakePipeline:
    def __init__(self):
        self.seen_columns = None

    def predict_proba(self, frame):
        self.seen_columns = list(frame.columns)
        return [[0.3, 0.7]]


class FakeRegressionPipeline:
    def __init__(self):
        self.seen_columns = None

    def predict(self, frame):
        self.seen_columns = list(frame.columns)
        return [0.18]


class CalendarFilterTests(unittest.TestCase):
    def test_data_quality_rejects_split_mismatch_and_bad_marks(self):
        row = {
            "price": 123.856,
            "strike": 400.0,
            "net_debit": -0.1,
            "exit_value": -1.0,
        }

        reasons = data_quality_rejection_reasons(row, max_moneyness_error=0.20)

        self.assertEqual(
            reasons,
            ["bad_moneyness", "non_positive_debit", "negative_exit_value"],
        )

    def test_add_calendar_entry_features_uses_only_pre_trade_values(self):
        row = {
            "price": 100.0,
            "strike": 105.0,
            "near_entry": 2.0,
            "far_entry": 3.5,
            "net_debit": 1.5,
            "near_expiry": "2025-01-17",
            "far_expiry": "2025-01-24",
        }

        features = add_calendar_entry_features(row)

        self.assertAlmostEqual(features["moneyness"], 1.05)
        self.assertAlmostEqual(features["abs_moneyness_error"], 0.05)
        self.assertAlmostEqual(features["debit_pct_price"], 0.015)
        self.assertAlmostEqual(features["near_far_entry_ratio"], 2.0 / 3.5)
        self.assertEqual(features["entry_width_days"], 7)

    def test_score_calendar_trade_aligns_artifact_features_and_threshold(self):
        pipeline = FakePipeline()
        artifact = {
            "pipeline": pipeline,
            "features": ["price", "net_debit", "moneyness"],
            "target": "min_pnl",
            "score_kind": "probability",
            "min_pnl": 10.0,
        }

        score = score_calendar_trade(
            artifact,
            {"price": 100, "net_debit": 1.25, "moneyness": 1.0, "ignored": 999},
            threshold=0.55,
        )

        self.assertEqual(pipeline.seen_columns, ["price", "net_debit", "moneyness"])
        self.assertAlmostEqual(score.probability, 0.7)
        self.assertTrue(score.recommended)
        self.assertEqual(score.reason, "model_score>=0.55")

    def test_score_calendar_trade_supports_expected_return_regression_artifacts(self):
        pipeline = FakeRegressionPipeline()
        artifact = {
            "pipeline": pipeline,
            "features": ["price", "net_debit", "debit_pct_price"],
            "target": "expected_return",
            "score_kind": "expected_return",
        }

        score = score_calendar_trade(
            artifact,
            {"price": 100, "net_debit": 1.25, "ignored": 999},
            threshold=0.10,
        )

        self.assertEqual(pipeline.seen_columns, ["price", "net_debit", "debit_pct_price"])
        self.assertAlmostEqual(score.probability, 0.18)
        self.assertTrue(score.recommended)
        self.assertEqual(score.reason, "model_score>=0.10")


if __name__ == "__main__":
    unittest.main()
