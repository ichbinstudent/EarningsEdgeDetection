import unittest

import pandas as pd

from train_calendar_filter import make_target, is_regression_target, build_model_pipeline


class CalendarFilterTrainingTests(unittest.TestCase):
    def test_expected_return_target_uses_continuous_return_on_debit(self):
        df = pd.DataFrame({"return_on_debit": [0.25, -0.10, 1.50]})

        target = make_target(df, "expected_return", min_pnl=10.0, min_return=0.10)

        self.assertEqual(target.tolist(), [0.25, -0.10, 1.50])

    def test_expected_return_uses_regression_model_with_predict_not_classifier(self):
        pipe = build_model_pipeline(["net_debit", "debit_pct_price"], target="expected_return", model_name="ridge", random_state=42)

        self.assertTrue(is_regression_target("expected_return"))
        self.assertTrue(hasattr(pipe.named_steps["model"], "predict"))
        self.assertFalse(hasattr(pipe.named_steps["model"], "predict_proba"))


if __name__ == "__main__":
    unittest.main()
