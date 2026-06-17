from datetime import date
import unittest

from earnings_edge.calendar_spread import select_calendar_expiries


class CalendarSpreadSelectionTests(unittest.TestCase):
    def test_select_calendar_expiries_prefers_month_out_far_leg_when_weeklies_exist(self):
        expiries = [
            date(2026, 5, 29),
            date(2026, 6, 5),
            date(2026, 6, 12),
            date(2026, 6, 18),
            date(2026, 6, 26),
        ]

        near_expiry, far_expiry = select_calendar_expiries(expiries)

        self.assertEqual(near_expiry, date(2026, 5, 29))
        self.assertEqual(far_expiry, date(2026, 6, 26))

    def test_select_calendar_expiries_falls_back_to_next_expiry_if_no_month_out_exists(self):
        expiries = [
            date(2026, 5, 29),
            date(2026, 6, 5),
            date(2026, 6, 12),
        ]

        near_expiry, far_expiry = select_calendar_expiries(expiries)

        self.assertEqual(near_expiry, date(2026, 5, 29))
        self.assertEqual(far_expiry, date(2026, 6, 5))


if __name__ == "__main__":
    unittest.main()
