"""Tests for pvlearn.encoders, ported from solaredge2mqtt's forecast module."""

import pickle
from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from numpy import isclose
from pandas import DataFrame, Series
from sklearn import clone

from pvlearn.encoders import (
    BaseEncoder,
    CategoricalEncoder,
    CyclicalEncoder,
    SunEncoder,
    TimeEncoder,
)


class TestBaseEncoder:
    def test_base_encoder_init(self):
        encoder = BaseEncoder()

        assert encoder.features is None
        assert encoder._feature_names_out == []

    def test_base_encoder_fit(self):
        encoder = BaseEncoder()
        df = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        result = encoder.fit(df)

        assert result is encoder
        assert encoder.features == ["col1", "col2"]

    def test_base_encoder_transform_not_fitted_raises(self):
        encoder = BaseEncoder()
        df = DataFrame({"col1": [1, 2, 3]})

        with pytest.raises(AttributeError, match="is not been fitted yet"):
            encoder._transform(df)

    def test_base_encoder_transform_missing_features_raises(self):
        encoder = BaseEncoder()
        df_fit = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        df_transform = DataFrame({"col1": [1, 2, 3]})
        encoder.fit(df_fit)

        with pytest.raises(AttributeError, match="are not in the vector"):
            encoder._transform(df_transform)

    def test_base_encoder_transform_extra_features_raises(self):
        encoder = BaseEncoder()
        df_fit = DataFrame({"col1": [1, 2, 3]})
        df_transform = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        encoder.fit(df_fit)

        with pytest.raises(AttributeError, match="are not in the vector"):
            encoder._transform(df_transform)

    def test_base_encoder_get_feature_names_out(self):
        encoder = BaseEncoder()
        encoder._feature_names_out = ["feature1", "feature2"]

        result = encoder.get_feature_names_out()

        assert result == ["feature1", "feature2"]

    def test_base_encoder_save_feature_names_out(self):
        encoder = BaseEncoder()
        df = DataFrame({"col1": [1, 2], "col2": [3, 4]})

        result = encoder._save_feature_names_out(df)

        assert encoder._feature_names_out == ["col1", "col2"]
        assert result is df

    def test_base_encoder_as_series_raises_for_dataframe_column(self):
        df = DataFrame([[1, 2], [3, 4]], columns=["dup", "dup"])

        with pytest.raises(TypeError, match="unexpectedly resolved to DataFrame"):
            BaseEncoder._as_series(df, "dup")


class TestCategoricalEncoder:
    def test_categorical_encoder_transform(self):
        encoder = CategoricalEncoder()
        df = DataFrame({"category": ["a", "b", "a", "c"]})
        encoder.fit(df)

        result = encoder.transform(df)

        assert result["category"].dtype.name == "category"
        assert encoder._feature_names_out == ["category"]

    def test_categorical_encoder_multiple_columns(self):
        encoder = CategoricalEncoder()
        df = DataFrame(
            {
                "cat1": ["a", "b", "a"],
                "cat2": ["x", "y", "z"],
            }
        )
        encoder.fit(df)

        result = encoder.transform(df)

        assert result["cat1"].dtype.name == "category"
        assert result["cat2"].dtype.name == "category"


class TestCyclicalEncoder:
    def test_cyclical_encoder_init(self):
        encoder = CyclicalEncoder(**{"hour": 24, "month": 12})

        assert encoder.cycle_lengths == {"hour": 24, "month": 12}

    def test_cyclical_encoder_transform(self):
        encoder = CyclicalEncoder(**{"hour": 24})
        df = DataFrame({"hour": [0, 6, 12, 18]})
        encoder.fit(df)

        result = encoder.transform(df)

        assert "hour_cos" in result.columns
        assert "hour_sin" in result.columns
        assert "hour" not in result.columns

    def test_cyclical_encoder_transform_not_fitted_raises(self):
        encoder = CyclicalEncoder(**{"hour": 24})
        df = DataFrame({"hour": [0, 6]})

        with pytest.raises(AttributeError, match="is not been fitted yet"):
            encoder.transform(df)

    def test_cyclical_encoder_values_at_boundaries(self):
        encoder = CyclicalEncoder(**{"hour": 24})
        df = DataFrame({"hour": [0, 12, 24]})
        encoder.fit(df)

        result = encoder.transform(df)

        assert isclose(result["hour_cos"].iloc[0], 1.0)
        assert isclose(result["hour_sin"].iloc[0], 0.0)

        assert isclose(result["hour_cos"].iloc[1], -1.0)
        assert isclose(result["hour_sin"].iloc[1], 0.0, atol=1e-10)

    def test_cyclical_encoder_unknown_feature_raises(self):
        encoder = CyclicalEncoder(**{"known_feature": 24})
        df = DataFrame({"unknown_feature": [1, 2, 3]})
        encoder.fit(df)

        with pytest.raises(ValueError, match="Unknown cyclical feature"):
            encoder.transform(df)

    def test_cyclical_encoder_get_params(self):
        encoder = CyclicalEncoder(**{"hour": 24, "month": 12})

        params = encoder.get_params()

        assert params == {"hour": 24, "month": 12}

    def test_cyclical_encoder_transform_cycle_columns_static(self):
        df = DataFrame({"value": [0, 6, 12, 18]})

        result = CyclicalEncoder.transform_cycle_columns(
            df,
            "test",
            cast(Series, df["value"]),
            24,
        )

        assert "test_cos" in result.columns
        assert "test_sin" in result.columns
        assert isclose(result["test_cos"].iloc[0], 1.0)
        assert isclose(result["test_sin"].iloc[0], 0.0)


class TestTimeEncoder:
    def test_time_encoder_transform(self):
        encoder = TimeEncoder()
        df = DataFrame(
            {
                "timestamp": [
                    datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 15, 6, 0, tzinfo=timezone.utc),
                ]
            }
        )
        encoder.fit(df)

        result = encoder.transform(df)

        assert "timestamp_minute_of_day_cos" in result.columns
        assert "timestamp_minute_of_day_sin" in result.columns
        assert "timestamp_month_cos" in result.columns
        assert "timestamp_month_sin" in result.columns
        assert "timestamp_dst" in result.columns
        assert "timestamp_day_of_year_cos" in result.columns
        assert "timestamp_day_of_year_sin" in result.columns

        assert "timestamp" not in result.columns

    def test_time_encoder_has_no_season_feature(self):
        """Dropped with `ephem`; day_of_year carries the same information."""
        encoder = TimeEncoder()
        df = DataFrame(
            {"timestamp": [datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)]}
        )
        encoder.fit(df)

        result = encoder.transform(df)

        assert not any("season" in column for column in result.columns)

    def test_time_encoder_transform_not_fitted_raises(self):
        encoder = TimeEncoder()
        df = DataFrame(
            {"timestamp": [datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)]}
        )

        with pytest.raises(AttributeError, match="is not been fitted yet"):
            encoder.transform(df)

    def test_time_of_day_distinguishes_intervals_within_an_hour(self):
        """The point of minutes since midnight over the plain hour (3.2)."""
        encoder = TimeEncoder()
        df = DataFrame(
            {
                "timestamp": [
                    datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
                    datetime(2024, 6, 15, 12, 15, tzinfo=timezone.utc),
                    datetime(2024, 6, 15, 12, 30, tzinfo=timezone.utc),
                ]
            }
        )
        encoder.fit(df)

        result = encoder.transform(df)

        assert result["timestamp_minute_of_day_sin"].nunique() == 3

    def test_minute_of_day_wraps_around_midnight(self):
        encoder = TimeEncoder()
        df = DataFrame(
            {
                "timestamp": [
                    datetime(2024, 6, 15, 0, 0, tzinfo=timezone.utc),
                    datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
                ]
            }
        )
        encoder.fit(df)

        result = encoder.transform(df)

        assert isclose(result["timestamp_minute_of_day_cos"].iloc[0], 1.0)
        assert isclose(result["timestamp_minute_of_day_sin"].iloc[0], 0.0)
        assert isclose(result["timestamp_minute_of_day_cos"].iloc[1], -1.0)

    def test_dst_flag_follows_the_timestamps_own_zone(self):
        berlin = ZoneInfo("Europe/Berlin")
        encoder = TimeEncoder()
        df = DataFrame(
            {
                "timestamp": [
                    datetime(2024, 7, 15, 12, 0, tzinfo=berlin),
                    datetime(2024, 1, 15, 12, 0, tzinfo=berlin),
                ]
            }
        )
        encoder.fit(df)

        result = encoder.transform(df)

        assert result["timestamp_dst"].tolist() == [True, False]


def make_sun_encoder(interval_minutes: int = 60) -> SunEncoder:
    return SunEncoder(52.52, 13.405, "Europe/Berlin", interval_minutes)


class TestSunEncoder:
    def test_sun_encoder_init(self):
        encoder = make_sun_encoder()

        assert encoder.latitude == 52.52
        assert encoder.longitude == 13.405
        assert encoder.timezone == "Europe/Berlin"
        assert encoder.interval_minutes == 60
        assert encoder._location is not None

    def test_sun_encoder_is_clonable_and_picklable(self):
        """sklearn.clone and joblib both need primitive constructor arguments."""
        encoder = make_sun_encoder()

        cloned = cast(SunEncoder, clone(encoder))
        restored: SunEncoder = pickle.loads(pickle.dumps(encoder))

        assert cloned.get_params() == encoder.get_params()
        assert restored.get_params() == encoder.get_params()

    def test_sun_encoder_transform(self):
        encoder = make_sun_encoder()

        df = DataFrame(
            {
                "time": [
                    datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
                    datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc),
                ]
            }
        )
        encoder.fit(df)

        result = encoder.transform(df)

        assert "time_elevation" in result.columns
        assert "time_azimuth_cos" in result.columns
        assert "time_azimuth_sin" in result.columns
        assert "time_daylight" in result.columns
        assert "time_delta_sunrise" in result.columns
        assert "time_delta_sunset" in result.columns

        assert "time" not in result.columns

    def test_sun_encoder_daylight_info(self):
        encoder = make_sun_encoder()

        time = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        result = encoder.daylight_info(time)

        assert isinstance(result, Series)
        assert len(result) == 3
        # In June at Berlin (52.52), daylight should be around 16-17 hours
        assert result.iloc[0] > 15

    def test_sun_encoder_elevation_at_noon(self):
        encoder = make_sun_encoder()

        df = DataFrame({"time": [datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)]})
        encoder.fit(df)

        result = encoder.transform(df)

        assert result["time_elevation"].iloc[0] > 0

    def test_sun_position_is_evaluated_at_the_middle_of_the_interval(self):
        """A timestamp labels a span, and the sun moves across it."""
        start = datetime(2024, 6, 15, 6, 0, tzinfo=timezone.utc)

        elevations = {}
        for interval in (60, 30):
            encoder = make_sun_encoder(interval)
            df = DataFrame({"time": [start]})
            encoder.fit(df)
            elevations[interval] = encoder.transform(df)["time_elevation"].iloc[0]

        centered = make_sun_encoder(0)
        shifted = DataFrame({"time": [start + timedelta(minutes=30)]})
        centered.fit(shifted)

        # Morning sun: the later the evaluation, the higher it stands.
        assert elevations[60] > elevations[30]
        assert isclose(
            elevations[60], centered.transform(shifted)["time_elevation"].iloc[0]
        )

    def test_sun_features_follow_the_encoders_location_not_the_process(self):
        """Two brains, two hemispheres, one process (multi-tenancy)."""
        df = DataFrame({"time": [datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)]})

        berlin = make_sun_encoder()
        sydney = SunEncoder(-33.87, 151.21, "Australia/Sydney", 60)
        berlin.fit(df)
        sydney.fit(df)

        berlin_result = berlin.transform(df.copy())
        sydney_result = sydney.transform(df.copy())

        assert berlin_result["time_daylight"].iloc[0] > 15
        assert sydney_result["time_daylight"].iloc[0] < 11

    def test_daylight_info_is_unaffected_by_how_an_instant_is_expressed(self):
        """Same moment, different zone: identical sun, DST or not."""
        berlin = ZoneInfo("Europe/Berlin")
        encoder = make_sun_encoder()

        for local in (
            datetime(2024, 10, 26, 12, 0, tzinfo=berlin),  # CEST, before the switch
            datetime(2024, 10, 28, 12, 0, tzinfo=berlin),  # CET, after it
        ):
            local_info = encoder.daylight_info(local)
            utc_info = encoder.daylight_info(local.astimezone(timezone.utc))

            assert isclose(local_info.iloc[1], utc_info.iloc[1])
            assert isclose(local_info.iloc[2], utc_info.iloc[2])

    def test_daylight_shrinks_across_the_dst_weekend(self):
        """The 25-hour day is a clock artefact, not an astronomical one."""
        berlin = ZoneInfo("Europe/Berlin")
        encoder = make_sun_encoder()

        before = encoder.daylight_info(datetime(2024, 10, 26, 12, 0, tzinfo=berlin))
        after = encoder.daylight_info(datetime(2024, 10, 28, 12, 0, tzinfo=berlin))

        assert 0 < before.iloc[0] - after.iloc[0] < 0.2

    def test_sun_encoder_transform_with_missing_features_raises(self):
        encoder = make_sun_encoder()
        df = DataFrame({"time": [datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)]})

        with patch.object(SunEncoder, "_transform", return_value=df.copy()):
            with pytest.raises(AttributeError, match="is not been fitted yet"):
                encoder.transform(df)

    def test_sun_encoder_transform_raises_for_dataframe_azimuth(self):
        encoder = make_sun_encoder()
        df = DataFrame(
            {
                "time": [
                    datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc),
                    datetime(2024, 6, 15, 13, 0, tzinfo=timezone.utc),
                ]
            }
        )
        encoder.fit(df)

        fake_time_series = MagicMock()
        fake_time_series.apply.side_effect = [
            Series([10.0, 11.0]),
            DataFrame({"bad": [1, 2]}),
        ]

        with patch.object(
            SunEncoder,
            "_as_series",
            side_effect=[cast(Series, df["time"]), fake_time_series],
        ):
            with pytest.raises(TypeError, match="Computed azimuth"):
                encoder.transform(df)
