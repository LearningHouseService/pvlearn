from datetime import datetime, timedelta
from unittest.mock import patch

from pvlearn.result import ForecastResult


def make_energy_period(start: datetime, hours: int, wh_per_hour: int = 100):
    return {start + timedelta(hours=i): wh_per_hour * (i + 1) for i in range(hours)}


def test_energy_today_sums_first_24_hours():
    start = datetime(2024, 6, 15, 0, 0)
    result = ForecastResult(energy_period=make_energy_period(start, 48))

    with patch.object(ForecastResult, "_current_hour", return_value=0):
        assert result.energy_today == sum(range(100, 100 * 25, 100))


def test_energy_tomorrow_sums_second_24_hours():
    start = datetime(2024, 6, 15, 0, 0)
    period = make_energy_period(start, 48)
    result = ForecastResult(energy_period=period)

    expected = sum(list(period.values())[24:])
    assert result.energy_tomorrow == expected


def test_energy_current_hour_uses_current_hour_index():
    start = datetime(2024, 6, 15, 0, 0)
    period = make_energy_period(start, 24)
    result = ForecastResult(energy_period=period)

    with patch.object(ForecastResult, "_current_hour", return_value=5):
        assert result.energy_current_hour == list(period.values())[5]


def test_energy_next_hour_rolls_into_tomorrow_at_hour_23():
    start = datetime(2024, 6, 15, 0, 0)
    period = make_energy_period(start, 48)
    result = ForecastResult(energy_period=period)

    with patch.object(ForecastResult, "_current_hour", return_value=23):
        assert result.energy_next_hour == list(period.values())[24]


def test_energy_today_remaining_sums_from_current_hour():
    start = datetime(2024, 6, 15, 0, 0)
    period = make_energy_period(start, 24)
    result = ForecastResult(energy_period=period)

    with patch.object(ForecastResult, "_current_hour", return_value=20):
        assert result.energy_today_remaining == sum(list(period.values())[20:])


def test_energy_next_hour_uses_next_hour_index_when_not_hour_23():
    start = datetime(2024, 6, 15, 0, 0)
    period = make_energy_period(start, 24)
    result = ForecastResult(energy_period=period)

    with patch.object(ForecastResult, "_current_hour", return_value=5):
        assert result.energy_next_hour == list(period.values())[6]


def test_current_hour_reflects_the_real_clock():
    assert 0 <= ForecastResult._current_hour() < 24
