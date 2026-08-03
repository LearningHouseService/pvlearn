"""Guards on the frozen reference dataset.

Every phase of the roadmap is verified by reproducing forecasts from this dataset,
so it has to stay exactly what it is. These tests do not check pvlearn behaviour;
they fail loudly if the fixture is ever silently replaced with a different export
that no longer covers the cases later tests depend on.
"""

import pandas as pd

EXPECTED_ROWS = 9580

EXPECTED_COLUMNS = {
    "_time",
    "clouds",
    "dew_point",
    "energy",
    "feels_like",
    "humidity",
    "pop",
    "power",
    "pressure",
    "rain",
    "snow",
    "temp",
    "uvi",
    "visibility",
    "weather_id",
    "weather_main",
    "wind_deg",
    "wind_gust",
    "wind_speed",
}

BERLIN = "Europe/Berlin"

# Europe/Berlin transitions. On the first date the local hour 02:00 occurs twice,
# on the second it does not occur at all.
DST_END_DATE = "2025-10-26"
DST_START_DATE = "2026-03-29"


class TestReferenceDatasetShape:
    def test_row_count_is_frozen(self, reference_dataset: pd.DataFrame):
        assert len(reference_dataset) == EXPECTED_ROWS

    def test_columns_are_frozen(self, reference_dataset: pd.DataFrame):
        assert set(reference_dataset.columns) == EXPECTED_COLUMNS

    def test_metadata_matches_data(
        self, reference_dataset: pd.DataFrame, reference_metadata: dict
    ):
        assert reference_metadata["rows"] == len(reference_dataset)
        assert set(reference_metadata["columns"]) == set(reference_dataset.columns)

    def test_timestamps_are_utc_and_unique(self, reference_dataset: pd.DataFrame):
        times = reference_dataset["_time"]
        assert str(times.dt.tz) == "UTC"
        assert not bool(times.duplicated().any())

    def test_target_column_has_no_gaps(self, reference_dataset: pd.DataFrame):
        assert not bool(reference_dataset["energy"].isna().any())


class TestReferenceDatasetCoverage:
    """The properties later tests rely on the dataset actually containing."""

    def test_spans_a_full_annual_cycle(self, reference_dataset: pd.DataFrame):
        local = reference_dataset["_time"].dt.tz_convert(BERLIN)
        assert local.dt.month.nunique() == 12

    def test_covers_both_utc_offsets(self, reference_dataset: pd.DataFrame):
        local = reference_dataset["_time"].dt.tz_convert(BERLIN)
        assert set(local.dt.strftime("%z")) == {"+0100", "+0200"}

    def test_ambiguous_local_hour_is_present(self, reference_dataset: pd.DataFrame):
        """The repeated 02:00 on the autumn transition.

        Two rows share a local wall-clock hour here and are distinguishable only
        by their UTC timestamp. Anything grouping on local time collides on them.
        """
        local = reference_dataset["_time"].dt.tz_convert(BERLIN)
        on_date = local[local.dt.strftime("%Y-%m-%d") == DST_END_DATE]
        repeated = on_date.dt.strftime("%Y-%m-%d %H")

        assert bool(repeated.duplicated().any())

    def test_skipped_local_hour_is_absent(self, reference_dataset: pd.DataFrame):
        """02:00 does not exist on the spring transition."""
        local = reference_dataset["_time"].dt.tz_convert(BERLIN)
        on_date = local[local.dt.strftime("%Y-%m-%d") == DST_START_DATE]

        assert 2 not in set(on_date.dt.hour)

    def test_contains_winter_weather(self, reference_dataset: pd.DataFrame):
        assert bool((reference_dataset["snow"] > 0).any())
        assert "Snow" in set(reference_dataset["weather_main"])

    def test_optional_column_has_missing_values(self, reference_dataset: pd.DataFrame):
        """Visibility is patchy, which is what exercises tolerance of gaps."""
        assert bool(reference_dataset["visibility"].isna().any())


class TestReferenceLocation:
    def test_location_travels_with_the_dataset(self, reference_location: dict):
        assert reference_location["timezone"] == BERLIN
        assert 47 < reference_location["latitude"] < 56
        assert 5 < reference_location["longitude"] < 16
