from pydantic import BaseModel, Field, field_validator

#: Interval lengths the library trains and predicts on today. The field exists
#: so no code path silently assumes an hour; finer resolutions need measurement
#: data and a provider that delivers them natively.
SUPPORTED_INTERVAL_MINUTES: tuple[int, ...] = (60,)


class ForecasterConfig(BaseModel):
    """Training-time configuration for a single `Forecaster`.

    Deliberately narrower than solaredge2mqtt's `ForecastSettings`: `retain` and
    `enable` are MQTT/service concerns and stay with the caller. `cachingdir` is
    accepted as-is and never defaulted or prepared here, since the library does
    not touch the filesystem except through paths it is explicitly given.

    There is no `weather_provider` field: which provider a row came from is a
    per-row fact of the training data, not a training-run setting, and lives
    as the `weather_provider` categorical feature in `pvlearn.schema` instead.
    `interval_minutes` defaults to 60, the only interval
    `SUPPORTED_INTERVAL_MINUTES` currently allows.
    """

    interval_minutes: int = Field(default=60)
    hyperparametertuning: bool = Field(default=False)
    cachingdir: str | None = Field(default=None)
    cache_size_limit_mb: int = Field(default=512, ge=1)

    @field_validator("interval_minutes")
    @classmethod
    def ensure_supported_interval(cls, value: int) -> int:
        if value not in SUPPORTED_INTERVAL_MINUTES:
            supported = ", ".join(str(item) for item in SUPPORTED_INTERVAL_MINUTES)
            raise ValueError(
                f"Unsupported forecast interval {value} minutes, supported: {supported}"
            )

        return value

    @property
    def is_caching_enabled(self) -> bool:
        return self.cachingdir is not None
