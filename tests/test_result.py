from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

from pvlearn.result import ForecastResult

BERLIN = ZoneInfo("Europe/Berlin")


def make_result(
    start: datetime,
    periods: int,
    interval_minutes: int = 60,
    wh_per_period: int = 100,
    zone: str = "Europe/Berlin",
) -> ForecastResult:
    interval = timedelta(minutes=interval_minutes)
    return ForecastResult(
        interval_minutes=interval_minutes,
        timezone=zone,
        energy_period={
            start + interval * i: wh_per_period * (i + 1) for i in range(periods)
        },
    )


def at(moment: datetime):
    return patch.object(ForecastResult, "_now", staticmethod(lambda: moment))


class TestDailyAggregates:
    def test_energy_today_sums_the_current_date_only(self):
        start = datetime(2024, 6, 15, 0, 0, tzinfo=BERLIN)
        result = make_result(start, 48)

        with at(datetime(2024, 6, 15, 0, 30, tzinfo=BERLIN)):
            assert result.energy_today == sum(range(100, 100 * 25, 100))

    def test_energy_tomorrow_sums_the_next_date(self):
        start = datetime(2024, 6, 15, 0, 0, tzinfo=BERLIN)
        result = make_result(start, 48)

        with at(datetime(2024, 6, 15, 12, 0, tzinfo=BERLIN)):
            assert result.energy_tomorrow == sum(range(2500, 100 * 49, 100))

    def test_energy_today_ignores_periods_before_today(self):
        """A series does not have to start at midnight of the current day."""
        start = datetime(2024, 6, 14, 22, 0, tzinfo=BERLIN)
        result = make_result(start, 4)

        with at(datetime(2024, 6, 15, 0, 30, tzinfo=BERLIN)):
            assert result.energy_today == 300 + 400

    def test_energy_today_remaining_starts_at_the_current_hour(self):
        start = datetime(2024, 6, 15, 0, 0, tzinfo=BERLIN)
        result = make_result(start, 24)

        with at(datetime(2024, 6, 15, 20, 45, tzinfo=BERLIN)):
            assert result.energy_today_remaining == sum(range(2100, 2500, 100))


class TestHourlyAggregates:
    def test_energy_current_hour(self):
        start = datetime(2024, 6, 15, 0, 0, tzinfo=BERLIN)
        result = make_result(start, 24)

        with at(datetime(2024, 6, 15, 5, 20, tzinfo=BERLIN)):
            assert result.energy_current_hour == 600

    def test_energy_next_hour(self):
        start = datetime(2024, 6, 15, 0, 0, tzinfo=BERLIN)
        result = make_result(start, 24)

        with at(datetime(2024, 6, 15, 5, 20, tzinfo=BERLIN)):
            assert result.energy_next_hour == 700

    def test_energy_next_hour_rolls_into_tomorrow_at_hour_23(self):
        start = datetime(2024, 6, 15, 0, 0, tzinfo=BERLIN)
        result = make_result(start, 48)

        with at(datetime(2024, 6, 15, 23, 10, tzinfo=BERLIN)):
            assert result.energy_next_hour == 2500

    def test_missing_period_is_zero_rather_than_an_error(self):
        start = datetime(2024, 6, 15, 12, 0, tzinfo=BERLIN)
        result = make_result(start, 4)

        with at(datetime(2024, 6, 15, 3, 0, tzinfo=BERLIN)):
            assert result.energy_current_hour == 0
            assert result.energy_next_hour == 0


class TestIntervalIndependence:
    def test_hourly_figures_sum_sub_hour_intervals(self):
        """Nothing assumes one row equals one hour."""
        start = datetime(2024, 6, 15, 0, 0, tzinfo=BERLIN)
        result = make_result(start, 96, interval_minutes=15, wh_per_period=25)

        with at(datetime(2024, 6, 15, 1, 5, tzinfo=BERLIN)):
            assert result.energy_current_hour == 25 * (5 + 6 + 7 + 8)
            assert result.energy_next_hour == 25 * (9 + 10 + 11 + 12)
            assert result.energy_today == 25 * sum(range(1, 97))


class TestTimezoneHandling:
    def test_aggregates_in_the_plants_zone_not_the_processs(self):
        """A service holds brains in several zones; none of them is local."""
        start = datetime(2024, 6, 14, 12, 0, tzinfo=timezone.utc)
        result = make_result(start, 24, zone="Pacific/Auckland")

        # 12:00 UTC on the 14th is already 00:00 on the 15th in Auckland, so
        # the whole series belongs to that day there.
        with at(datetime(2024, 6, 15, 3, 0, tzinfo=timezone.utc)):
            assert result.energy_today == sum(range(100, 100 * 25, 100))

    def test_utc_keys_are_converted_before_grouping(self):
        start = datetime(2024, 6, 14, 23, 0, tzinfo=timezone.utc)
        result = make_result(start, 2)

        # 23:00 UTC is 01:00 in Berlin on the 15th, so both periods are today.
        with at(datetime(2024, 6, 15, 12, 0, tzinfo=BERLIN)):
            assert result.energy_today == 300

    def test_naive_periods_are_read_as_local_wall_clock(self):
        start = datetime(2024, 6, 15, 0, 0)
        result = make_result(start, 24)

        with at(datetime(2024, 6, 15, 5, 20, tzinfo=BERLIN)):
            assert result.energy_current_hour == 600

    def test_empty_period_aggregates_to_zero(self):
        result = ForecastResult(
            interval_minutes=60, timezone="Europe/Berlin", energy_period={}
        )

        assert result.energy_today == 0
        assert result.energy_current_hour == 0

    def test_survives_the_dst_transition(self):
        """The night the clock goes back has 25 hours in Berlin.

        Keyed in UTC on purpose: two local timestamps of 02:00 that hour are
        the same wall clock and would collapse into one dictionary entry.
        """
        start = datetime(2024, 10, 26, 22, 0, tzinfo=timezone.utc)
        result = make_result(start, 25, wh_per_period=100)

        with at(datetime(2024, 10, 27, 12, 0, tzinfo=BERLIN)):
            assert result.energy_today == sum(range(100, 100 * 26, 100))

    def test_the_repeated_hour_is_counted_once_at_a_time(self):
        """02:00 happens twice; only the 60 minutes we are in count."""
        start = datetime(2024, 10, 26, 22, 0, tzinfo=timezone.utc)
        result = make_result(start, 25, wh_per_period=100)

        with at(datetime(2024, 10, 27, 0, 30, tzinfo=timezone.utc)):  # first 02:30
            assert result.energy_current_hour == 300
            assert result.energy_next_hour == 400

        with at(datetime(2024, 10, 27, 1, 30, tzinfo=timezone.utc)):  # second 02:30
            assert result.energy_current_hour == 400
            assert result.energy_next_hour == 500

    def test_next_hour_survives_the_spring_forward_gap(self):
        """02:00 does not exist that night; the next hour is 03:00."""
        start = datetime(2024, 3, 30, 23, 0, tzinfo=timezone.utc)
        result = make_result(start, 6)

        with at(datetime(2024, 3, 31, 0, 30, tzinfo=timezone.utc)):  # 01:30 CET
            assert result.energy_current_hour == 200
            assert result.energy_next_hour == 300

    def test_now_reflects_the_real_clock(self):
        assert ForecastResult._now().tzinfo is not None
