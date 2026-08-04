from pydantic import BaseModel, Field, field_validator

#: Interval lengths the library trains and predicts on today. The field exists
#: so no code path silently assumes an hour; finer resolutions need measurement
#: data and a provider that delivers them natively (chapter 3.5).
SUPPORTED_INTERVAL_MINUTES: tuple[int, ...] = (60,)


class ForecasterConfig(BaseModel):
    """Training-time configuration for a single `Forecaster`.

    Deliberately narrower than solaredge2mqtt's `ForecastSettings`: `retain` and
    `enable` are MQTT/service concerns and stay with the caller. `cachingdir` is
    accepted as-is and never defaulted or prepared here, since the library does
    not touch the filesystem except through paths it is explicitly given.

    `interval_minutes` and `weather_provider` have no defaults: both end up in
    the model metadata and a wrong guess for either produces a model that is
    silently trained on inputs it will never see again.
    """

    interval_minutes: int
    weather_provider: str = Field(min_length=1)
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
