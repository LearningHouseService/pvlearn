from pydantic import BaseModel, Field


class ForecasterConfig(BaseModel):
    """Training-time configuration for a single `Forecaster`.

    Deliberately narrower than solaredge2mqtt's `ForecastSettings`: `retain` and
    `enable` are MQTT/service concerns and stay with the caller. `cachingdir` is
    accepted as-is and never defaulted or prepared here, since the library does
    not touch the filesystem except through paths it is explicitly given.
    """

    hyperparametertuning: bool = Field(default=False)
    cachingdir: str | None = Field(default=None)
    cache_size_limit_mb: int = Field(default=512, ge=1)

    @property
    def is_caching_enabled(self) -> bool:
        return self.cachingdir is not None
