import logging
from datetime import datetime, timedelta
from typing import Any, Self

from astral import LocationInfo
from astral.sun import azimuth, elevation, sun
from numpy import cos, pi, sin
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class BaseEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.features: list[str] | None = None
        self._feature_names_out: list[str] = []

    def fit(self, x_vector: DataFrame, *_: Any) -> Self:
        self.features = x_vector.columns.tolist()
        return self

    def _transform(self, x_vector: DataFrame) -> DataFrame:
        if self.features is None:
            raise AttributeError(f"Encoder {self.__class__} is not been fitted yet.")

        if not all(feature in self.features for feature in x_vector.columns):
            raise AttributeError(f"Columns {x_vector.columns} are not in the vector")

        if not all(feature in x_vector.columns for feature in self.features):
            raise AttributeError(f"Columns {self.features} are not in the vector")

        return x_vector

    def _save_feature_names_out(self, x_vector: DataFrame) -> DataFrame:
        self._feature_names_out = x_vector.columns.to_list()
        return x_vector

    def get_feature_names_out(self, *_) -> list[str]:
        return self._feature_names_out

    @staticmethod
    def _as_series(x_vector: DataFrame, feature: str) -> Series:
        column = x_vector[feature]
        if isinstance(column, DataFrame):
            raise TypeError(f"Column {feature} unexpectedly resolved to DataFrame")

        return column


class CategoricalEncoder(BaseEncoder):
    def transform(self, x_vector: DataFrame) -> DataFrame:
        x_vector = self._transform(x_vector).astype("category")
        return self._save_feature_names_out(x_vector)


class CyclicalEncoder(BaseEncoder):
    def __init__(self, **cycle_lengths: int) -> None:
        super().__init__()
        self.cycle_lengths: dict[str, int] = cycle_lengths

    def transform(self, x_vector: DataFrame) -> DataFrame:
        if self.features is None:
            raise AttributeError(f"Encoder {self.__class__} is not been fitted yet.")

        x_vector = self._transform(x_vector)
        for feature in self.features:
            cycle = self.cycle_lengths.get(feature, None)
            if not cycle:
                raise ValueError(f"Unknown cyclical feature {feature}")

            x_vector = self.transform_cycle_columns(
                x_vector,
                feature,
                self._as_series(x_vector, feature),
                cycle,
            )
            x_vector.drop(feature, axis=1, inplace=True)

        return self._save_feature_names_out(x_vector)

    def get_params(self, deep: bool = True) -> dict[str, int]:
        _ = deep
        return self.cycle_lengths

    @staticmethod
    def transform_cycle_columns(
        x_vector: DataFrame,
        prefix: str,
        cycle_vector: Series,
        cycle_length: float,
    ) -> DataFrame:
        x_vector[f"{prefix}_cos"] = cos(2 * pi * cycle_vector / cycle_length)
        x_vector[f"{prefix}_sin"] = sin(2 * pi * cycle_vector / cycle_length)

        return x_vector


class TimeEncoder(BaseEncoder):
    """Calendar and clock features derived from an interval's timestamp.

    Time of day is encoded as minutes since midnight rather than the hour: on
    an hourly grid both are the same feature, but for any finer interval every
    slot within an hour would share one value and the feature would carry no
    information at all.

    There is no season feature. It used to come from `ephem`'s equinox and
    solstice dates, which is the only thing that dependency was needed for,
    while `day_of_year_sin/cos` already carries the same annual position
    continuously instead of in four steps.
    """

    MINUTES_PER_DAY = 24 * 60
    DAYS_PER_YEAR = 365.25
    MONTHS_PER_YEAR = 12

    def transform(self, x_vector: DataFrame) -> DataFrame:
        if self.features is None:
            raise AttributeError(f"Encoder {self.__class__} is not been fitted yet.")

        x_vector = self._transform(x_vector)
        for feature in self.features:
            feature_series = self._as_series(x_vector, feature)
            x_vector = CyclicalEncoder.transform_cycle_columns(
                x_vector,
                f"{feature}_minute_of_day",
                feature_series.dt.hour * 60 + feature_series.dt.minute,
                self.MINUTES_PER_DAY,
            )

            x_vector = CyclicalEncoder.transform_cycle_columns(
                x_vector,
                f"{feature}_month",
                feature_series.dt.month,
                self.MONTHS_PER_YEAR,
            )

            x_vector[f"{feature}_dst"] = feature_series.apply(
                lambda x: x.dst() != timedelta(0)
            ).astype("category")

            x_vector = CyclicalEncoder.transform_cycle_columns(
                x_vector,
                f"{feature}_day_of_year",
                feature_series.dt.dayofyear,
                self.DAYS_PER_YEAR,
            )

            x_vector.drop(feature, axis=1, inplace=True)

        logger.debug("%s", x_vector.head(30))
        return self._save_feature_names_out(x_vector)


class SunEncoder(BaseEncoder):
    """Sun position and daylight features for an interval.

    Positions are evaluated at the middle of the interval, not at its start:
    the timestamp labels a span of `interval_minutes`, and the sun moves
    noticeably across it.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        timezone: str,
        interval_minutes: int,
    ) -> None:
        super().__init__()
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.interval_minutes = interval_minutes
        self._location = LocationInfo(
            "name",
            "region",
            timezone=timezone,
            latitude=latitude,
            longitude=longitude,
        )

    def transform(self, x_vector: DataFrame) -> DataFrame:
        x_vector = self._transform(x_vector)

        if self.features is None:
            raise AttributeError(f"Encoder {self.__class__} is not been fitted yet.")

        interval_center = timedelta(minutes=self.interval_minutes / 2)

        for feature in self.features:
            time_key = f"{feature}_time"
            feature_series = self._as_series(x_vector, feature)

            x_vector[time_key] = feature_series.apply(lambda x: x + interval_center)
            time_series = self._as_series(x_vector, time_key)

            x_vector[f"{feature}_elevation"] = time_series.apply(
                lambda x: elevation(self._location.observer, x)
            )

            azimuth_series = time_series.apply(
                lambda x: azimuth(self._location.observer, x),
            )
            if isinstance(azimuth_series, DataFrame):
                raise TypeError("Computed azimuth is unexpectedly a DataFrame")

            x_vector = CyclicalEncoder.transform_cycle_columns(
                x_vector,
                f"{feature}_azimuth",
                azimuth_series,
                360,
            )

            x_vector[
                [
                    f"{feature}_daylight",
                    f"{feature}_delta_sunrise",
                    f"{feature}_delta_sunset",
                ]
            ] = time_series.apply(self.daylight_info)

            x_vector.drop([feature, time_key], axis=1, inplace=True)

        return self._save_feature_names_out(x_vector)

    def daylight_info(self, row_time: datetime) -> Series:
        s = sun(self._location.observer, row_time)
        daylight = (s["sunset"] - s["sunrise"]).total_seconds() / 3600

        delta_sunrise = (row_time - s["sunrise"]).total_seconds() / 3600
        delta_sunset = (s["sunset"] - row_time).total_seconds() / 3600

        return Series([daylight, delta_sunrise, delta_sunset])
