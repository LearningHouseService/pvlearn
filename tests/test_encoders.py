"""Tests for pvlearn.encoders, ported from solaredge2mqtt's forecast module."""

from datetime import datetime, timezone
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from numpy import isclose
from pandas import DataFrame, Series

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
    def test_time_encoder_init(self):
        encoder = TimeEncoder()

        assert encoder.season_starts == {}

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

        assert "timestamp_hour_cos" in result.columns
        assert "timestamp_hour_sin" in result.columns
        assert "timestamp_month_cos" in result.columns
        assert "timestamp_month_sin" in result.columns
        assert "timestamp_dst" in result.columns
        assert "timestamp_season" in result.columns
        assert "timestamp_day_of_year_cos" in result.columns
        assert "timestamp_day_of_year_sin" in result.columns

        assert "timestamp" not in result.columns

    def test_time_encoder_transform_not_fitted_raises(self):
        encoder = TimeEncoder()
        df = DataFrame(
            {"timestamp": [datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)]}
        )

        with pytest.raises(AttributeError, match="is not been fitted yet"):
            encoder.transform(df)

    def test_time_encoder_map_season_spring(self):
        encoder = TimeEncoder()
        date = datetime(2024, 4, 15, 12, 0, tzinfo=timezone.utc)

        season = encoder._map_season(date)

        assert season == "spring"

    def test_time_encoder_map_season_summer(self):
        encoder = TimeEncoder()
        date = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)

        season = encoder._map_season(date)

        assert season == "summer"

    def test_time_encoder_map_season_autumn(self):
        encoder = TimeEncoder()
        date = datetime(2024, 10, 15, 12, 0, tzinfo=timezone.utc)

        season = encoder._map_season(date)

        assert season == "autumn"

    def test_time_encoder_map_season_winter(self):
        encoder = TimeEncoder()
        date = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)

        season = encoder._map_season(date)

        assert season == "winter"

    def test_time_encoder_map_season_late_december_is_winter(self):
        encoder = TimeEncoder()
        date = datetime(2024, 12, 30, 12, 0, tzinfo=timezone.utc)

        season = encoder._map_season(date)

        assert season == "winter"

    def test_time_encoder_caches_season_starts(self):
        encoder = TimeEncoder()
        date1 = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        date2 = datetime(2024, 7, 15, 12, 0, tzinfo=timezone.utc)

        encoder._map_season(date1)
        encoder._map_season(date2)

        assert 2024 in encoder.season_starts
        assert len(encoder.season_starts) == 1


class TestSunEncoder:
    def test_sun_encoder_init(self):
        encoder = SunEncoder(52.52, 13.405, "Europe/Berlin")

        assert encoder.latitude == 52.52
        assert encoder.longitude == 13.405
        assert encoder.timezone == "Europe/Berlin"
        assert encoder._location is not None

    def test_sun_encoder_transform(self):
        encoder = SunEncoder(52.52, 13.405, "Europe/Berlin")

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
        encoder = SunEncoder(52.52, 13.405, "Europe/Berlin")

        time = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        result = encoder.daylight_info(time)

        assert isinstance(result, Series)
        assert len(result) == 3
        # In June at Berlin (52.52), daylight should be around 16-17 hours
        assert result.iloc[0] > 15

    def test_sun_encoder_elevation_at_noon(self):
        encoder = SunEncoder(52.52, 13.405, "Europe/Berlin")

        df = DataFrame({"time": [datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)]})
        encoder.fit(df)

        result = encoder.transform(df)

        assert result["time_elevation"].iloc[0] > 0

    def test_sun_encoder_transform_with_missing_features_raises(self):
        encoder = SunEncoder(52.52, 13.405, "Europe/Berlin")
        df = DataFrame({"time": [datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)]})

        with patch.object(SunEncoder, "_transform", return_value=df.copy()):
            with pytest.raises(AttributeError, match="is not been fitted yet"):
                encoder.transform(df)

    def test_sun_encoder_transform_raises_for_dataframe_azimuth(self):
        encoder = SunEncoder(52.52, 13.405, "Europe/Berlin")
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
