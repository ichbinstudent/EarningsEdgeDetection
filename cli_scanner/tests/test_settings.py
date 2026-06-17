import os
import unittest


class TestSettings(unittest.TestCase):
    def test_settings_loads_from_env(self):
        os.environ["POLYGON_API_KEY"] = "test_key"
        os.environ["POLYGON_RATE_SLEEP"] = "5.0"
        # Force re-read
        import importlib
        import earnings_edge.settings
        importlib.reload(earnings_edge.settings)
        from earnings_edge.settings import Settings
        s = Settings()
        self.assertEqual(s.polygon_api_key, "test_key")
        self.assertEqual(s.polygon_rate_sleep, 5.0)

    def test_filter_thresholds_have_defaults(self):
        from earnings_edge.settings import Settings, FilterThresholds
        s = Settings()
        self.assertEqual(s.filters.min_price, 3.0)
        self.assertEqual(s.filters.min_volume, 1_500_000)
        self.assertEqual(s.filters.term_structure_hard_limit, -0.004)

    def test_model_path_configurable(self):
        from earnings_edge.settings import Settings
        s = Settings()
        self.assertIn("calendar_call_filter", str(s.calendar_model_path))

    def test_validate_returns_errors_for_missing_keys(self):
        from earnings_edge.settings import Settings
        s = Settings(polygon_api_key="", telegram_bot_token="")
        errors = s.validate()
        self.assertTrue(any("POLYGON_API_KEY" in e for e in errors))
        self.assertTrue(any("TELEGRAM_BOT_TOKEN" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
