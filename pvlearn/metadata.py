import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from pvlearn import __version__
from pvlearn.config import ForecasterConfig
from pvlearn.exceptions import SchemaMismatchError
from pvlearn.location import Location

#: Leading release segment of a PEP 440 version — `0.3.0` out of
#: `0.3.0.post3.dev1+g527ccef`.
_RELEASE_SEGMENT = re.compile(r"^\d+(?:\.\d+)*")


def release_version(version: str = __version__) -> str:
    """The release part of a version, without pre/post/dev/local segments.

    See ADR 0003 for why the release alone decides model compatibility.
    """
    match = _RELEASE_SEGMENT.match(version)
    if match is None:
        raise ValueError(f"Version {version!r} does not start with a release segment")

    return match.group()


class ModelMetrics(BaseModel):
    """Quality of a model on a holdout, in the unit of the target (Wh)."""

    mae: float
    rmse: float
    r2: float


class ModelMetadata(BaseModel):
    """Sidecar describing what a persisted model was trained on and with.

    Loading a model whose metadata disagrees with the current configuration is
    a hard error, never a best-effort load: a model trained on a different
    feature set, interval, or location keeps predicting plausible numbers, so
    the failure is silent and effectively undebuggable. The weather provider
    is deliberately not part of this: it is the `weather_provider` categorical
    feature in `pvlearn.schema` now, a per-row fact the model can learn from
    rather than a setting the whole model is pinned to.

    The pvlearn release is the single version this compares — see ADR 0003.
    """

    pvlearn_version: str
    interval_minutes: int
    location: Location
    trained_at: datetime
    training_rows: int
    selected_features: list[str]
    metrics: ModelMetrics

    @classmethod
    def create(
        cls,
        location: Location,
        config: ForecasterConfig,
        training_rows: int,
        selected_features: list[str],
        metrics: ModelMetrics,
        trained_at: datetime | None = None,
    ) -> "ModelMetadata":
        return cls(
            pvlearn_version=__version__,
            interval_minutes=config.interval_minutes,
            location=location,
            trained_at=trained_at or datetime.now().astimezone(),
            training_rows=training_rows,
            selected_features=selected_features,
            metrics=metrics,
        )

    def raise_on_mismatch(self, location: Location, config: ForecasterConfig) -> None:
        """Reject the model unless it was trained under the given setup.

        Only the release segment of `pvlearn_version` takes part, so a model
        survives the dev builds between two releases — see ADR 0003.
        """
        mismatches = [
            self._compare(
                "pvlearn_version", release_version(), transform=release_version
            ),
            self._compare("interval_minutes", config.interval_minutes),
            self._compare("location", location),
        ]

        reasons = [reason for reason in mismatches if reason is not None]
        if reasons:
            raise SchemaMismatchError(
                "Persisted model does not match the current configuration "
                f"and has to be retrained: {'; '.join(reasons)}"
            )

    def _compare(
        self,
        attribute: str,
        expected: object,
        transform: Callable[[Any], Any] = lambda value: value,
    ) -> str | None:
        persisted = getattr(self, attribute)
        if transform(persisted) == expected:
            return None

        return f"{attribute} is {persisted!r}, expected {expected!r}"
