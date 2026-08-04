"""The canonical schema's tolerance for providers that deliver less.

Every weather feature is optional. A provider without an UV index, or one that
skips irradiance, has to yield a smaller feature set — never an exception, and
never a column the model has not seen.
"""

from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import cast

import pandas as pd
import pytest
from pandas import DataFrame

from pvlearn.config import ForecasterConfig
from pvlearn.forecaster import Forecaster
from pvlearn.location import Location
from pvlearn.schema import (
    CATEGORICAL_FEATURES,
    CYCLICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_FEATURE,
    TIME_FEATURE,
)

LOCAL_TZ = "Europe/Berlin"

ALL_WEATHER_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, *CYCLICAL_FEATURES]

ROWS = 60


def make_forecaster() -> Forecaster:
    return Forecaster(
        Location(latitude=52.52, longitude=13.405, timezone=LOCAL_TZ),
        ForecasterConfig(interval_minutes=60, weather_provider="test"),
    )


def make_data(features: list[str]) -> DataFrame:
    columns: dict[str, list] = {
        TIME_FEATURE: [
            datetime.now(timezone.utc) + timedelta(hours=i) for i in range(ROWS)
        ],
        TARGET_FEATURE: [100.0 + i for i in range(ROWS)],
    }
    for feature in features:
        columns[feature] = (
            [800] * ROWS if feature in CATEGORICAL_FEATURES else [1.0 * ROWS] * ROWS
        )

    data = DataFrame(columns)
    data[TIME_FEATURE] = data[TIME_FEATURE].astype(
        pd.DatetimeTZDtype(unit="ns", tz=LOCAL_TZ)
    )
    return data


def transform(features: list[str]) -> DataFrame:
    forecaster = make_forecaster()
    data = make_data(features)
    preprocessor = forecaster._prepare_preprocessor(data.columns.to_list())

    return cast(DataFrame, preprocessor.fit_transform(data))


@pytest.mark.parametrize(
    "present",
    [
        pytest.param(ALL_WEATHER_FEATURES, id="all"),
        pytest.param([], id="none"),
        *(
            pytest.param([feature], id=f"only-{feature}")
            for feature in ALL_WEATHER_FEATURES
        ),
        *(
            pytest.param(
                [item for item in ALL_WEATHER_FEATURES if item != feature],
                id=f"without-{feature}",
            )
            for feature in ALL_WEATHER_FEATURES
        ),
    ],
)
def test_any_subset_of_weather_features_transforms(present: list[str]):
    transformed = transform(present)

    assert len(transformed) == ROWS


def test_feature_set_shrinks_with_the_provider():
    full = transform(ALL_WEATHER_FEATURES)
    reduced = transform(
        [feature for feature in ALL_WEATHER_FEATURES if feature != "ghi"]
    )

    assert set(reduced.columns) < set(full.columns)


def test_time_and_sun_features_survive_a_provider_without_any_weather():
    transformed = transform([])

    assert "sun__time_elevation" in transformed.columns
    assert "time__time_minute_of_day_sin" in transformed.columns


@pytest.mark.parametrize("pair", list(combinations(ALL_WEATHER_FEATURES, 2))[:20])
def test_no_feature_pair_collides_in_the_output(pair: tuple[str, str]):
    """Two features must never map onto the same transformed column."""
    transformed = transform(list(pair))

    assert len(set(transformed.columns)) == len(transformed.columns)


def test_unknown_columns_are_dropped_rather_than_passed_through():
    forecaster = make_forecaster()
    data = make_data(ALL_WEATHER_FEATURES)
    data["weather_main"] = "Clear"

    preprocessor = forecaster._prepare_preprocessor(data.columns.to_list())
    transformed = cast(DataFrame, preprocessor.fit_transform(data))

    assert not any("weather_main" in column for column in transformed.columns)
