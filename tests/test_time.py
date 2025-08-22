import unittest
from datetime import datetime

import pytz

from src.utils.time import get_time_str


class TestGetCurrentTimeStr(unittest.TestCase):
    """Test cases for the get_time_str function."""

    def setUp(self):
        """Set up test fixtures with various timezones for testing."""
        self.utc_tz = pytz.UTC
        self.et_tz = pytz.timezone("US/Eastern")
        self.pt_tz = pytz.timezone("US/Pacific")
        self.london_tz = pytz.timezone("Europe/London")

    def test_basic_conversion_utc_to_et(self):
        """Test basic conversion from UTC to Eastern Time."""
        # January 15, 2024, 3:30 PM UTC (10:30 AM ET)
        utc_time = datetime(2024, 1, 15, 15, 30, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        self.assertEqual(result, "january-15-10am-et")

    def test_midnight_conversion(self):
        """Test midnight conversion (12:00 AM)."""
        # January 15, 2024, 5:00 AM UTC (12:00 AM ET)
        utc_time = datetime(2024, 1, 15, 5, 0, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        self.assertEqual(result, "january-15-12am-et")

    def test_noon_conversion(self):
        """Test noon conversion (12:00 PM)."""
        # January 15, 2024, 5:00 PM UTC (12:00 PM ET)
        utc_time = datetime(2024, 1, 15, 17, 0, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        self.assertEqual(result, "january-15-12pm-et")

    def test_single_digit_hour(self):
        """Test single digit hours (1-9) don't have leading zeros."""
        # January 15, 2024, 6:00 AM UTC (1:00 AM ET)
        utc_time = datetime(2024, 1, 15, 6, 0, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        self.assertEqual(result, "january-15-1am-et")

    def test_double_digit_hour(self):
        """Test double digit hours display correctly."""
        # January 15, 2024, 3:00 PM UTC (10:00 AM ET)
        utc_time = datetime(2024, 1, 15, 15, 0, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        self.assertEqual(result, "january-15-10am-et")

    def test_all_months(self):
        """Test that all month names are correctly formatted in lowercase."""
        expected_months = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]

        for month, expected_name in enumerate(expected_months, 1):
            # Create a date in each month - adjust UTC time based on DST
            # DST in 2024: March through October (November 15 is already EST)
            if month in [3, 4, 5, 6, 7, 8, 9, 10]:  # DST months (March-October)
                # During DST, ET is UTC-4, so 15:00 UTC = 11:00 AM ET
                utc_time = datetime(2024, month, 15, 15, 0, 0, tzinfo=self.utc_tz)
                expected = f"{expected_name}-15-11am-et"
            else:  # Standard time months (November, December, January, February)
                # During standard time, ET is UTC-5, so 15:00 UTC = 10:00 AM ET
                utc_time = datetime(2024, month, 15, 15, 0, 0, tzinfo=self.utc_tz)
                expected = f"{expected_name}-15-10am-et"

            result = get_time_str(utc_time)
            self.assertEqual(result, expected, f"Month {month} test failed")

    def test_day_zero_padding(self):
        """Test that days are zero-padded correctly."""
        # Test single digit day
        utc_time = datetime(2024, 1, 5, 15, 0, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        self.assertEqual(result, "january-05-10am-et")

        # Test double digit day
        utc_time = datetime(2024, 1, 25, 15, 0, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        self.assertEqual(result, "january-25-10am-et")

    def test_dst_transition_spring(self):
        """Test behavior during spring DST transition."""
        # March 10, 2024, 7:00 AM UTC during DST transition
        utc_time = datetime(2024, 3, 10, 7, 0, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        # Should be 3:00 AM EDT (Eastern Daylight Time)
        self.assertEqual(result, "march-10-3am-et")

    def test_dst_transition_fall(self):
        """Test behavior during fall DST transition."""
        # November 3, 2024, 6:00 AM UTC during DST transition
        utc_time = datetime(2024, 11, 3, 6, 0, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        # Should be 1:00 AM EST (Eastern Standard Time)
        self.assertEqual(result, "november-03-1am-et")

    def test_different_input_timezones(self):
        """Test that function works with different input timezones."""
        # Same moment in time, different input timezones
        base_utc = datetime(2024, 1, 15, 15, 0, 0, tzinfo=self.utc_tz)

        # Convert to different timezones
        pt_time = base_utc.astimezone(self.pt_tz)
        london_time = base_utc.astimezone(self.london_tz)
        et_time = base_utc.astimezone(self.et_tz)

        # All should produce the same result
        expected = "january-15-10am-et"
        self.assertEqual(get_time_str(base_utc), expected)
        self.assertEqual(get_time_str(pt_time), expected)
        self.assertEqual(get_time_str(london_time), expected)
        self.assertEqual(get_time_str(et_time), expected)

    def test_pm_hours(self):
        """Test PM hours are handled correctly."""
        test_cases = [
            (datetime(2024, 1, 15, 18, 0, 0, tzinfo=self.utc_tz), "january-15-1pm-et"),  # 1 PM ET
            (datetime(2024, 1, 15, 19, 0, 0, tzinfo=self.utc_tz), "january-15-2pm-et"),  # 2 PM ET
            (datetime(2024, 1, 15, 22, 0, 0, tzinfo=self.utc_tz), "january-15-5pm-et"),  # 5 PM ET
            (
                datetime(2024, 1, 15, 4, 0, 0, tzinfo=self.utc_tz),
                "january-14-11pm-et",
            ),  # 11 PM ET (crosses day boundary)
        ]

        for utc_time, expected in test_cases:
            with self.subTest(utc_time=utc_time):
                result = get_time_str(utc_time)
                self.assertEqual(result, expected)

    def test_edge_case_year_boundary(self):
        """Test behavior at year boundaries."""
        # December 31, 2023, 11:59 PM UTC -> December 31, 2023, 6:59 PM ET
        utc_time = datetime(2023, 12, 31, 23, 59, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        self.assertEqual(result, "december-31-6pm-et")

        # January 1, 2024, 12:00 AM UTC -> December 31, 2023, 7:00 PM ET
        utc_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        self.assertEqual(result, "december-31-7pm-et")

    def test_leap_year_february(self):
        """Test February 29th in leap year."""
        # February 29, 2024, 3:00 PM UTC (10:00 AM ET)
        utc_time = datetime(2024, 2, 29, 15, 0, 0, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)
        self.assertEqual(result, "february-29-10am-et")

    def test_format_consistency(self):
        """Test that the format is always consistent."""
        utc_time = datetime(2024, 6, 15, 14, 30, 45, tzinfo=self.utc_tz)
        result = get_time_str(utc_time)

        # Check format: month-DD-Hhh{am|pm}-et
        parts = result.split("-")
        self.assertEqual(len(parts), 4, "Should have 4 parts separated by dashes")
        self.assertTrue(parts[0].islower(), "Month should be lowercase")
        self.assertTrue(parts[1].isdigit(), "Day should be numeric")
        self.assertEqual(len(parts[1]), 2, "Day should be zero-padded to 2 digits")
        self.assertTrue(parts[2].endswith(("am", "pm")), "Hour should end with am or pm")
        self.assertEqual(parts[3], "et", "Should end with 'et'")

    def test_minute_component_ignored(self):
        """Test that minute and second components are ignored - only hour matters."""
        # Test that different minutes/seconds don't affect the output
        base_time = datetime(2024, 1, 15, 15, 0, 0, tzinfo=self.utc_tz)  # 10:00 AM ET
        time_with_minutes = datetime(2024, 1, 15, 15, 30, 45, tzinfo=self.utc_tz)  # 10:30:45 AM ET
        time_with_seconds = datetime(2024, 1, 15, 15, 59, 59, tzinfo=self.utc_tz)  # 10:59:59 AM ET

        expected = "january-15-10am-et"

        self.assertEqual(get_time_str(base_time), expected)
        self.assertEqual(get_time_str(time_with_minutes), expected)
        self.assertEqual(get_time_str(time_with_seconds), expected)


if __name__ == "__main__":
    unittest.main()
