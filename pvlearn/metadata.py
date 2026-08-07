from datetime import datetime

import sklearn
from pydantic import BaseModel

from pvlearn import __version__
from pvlearn.config import ForecasterConfig
from pvlearn.exceptions import SchemaMismatchError
from pvlearn.location import Location
from pvlearn.schema import FEATURE_SCHEMA_VERSION

#: Version of how the training pipeline is built — preprocessing steps, the
#: feature-selection rule and the estimator. Distinct from
#: `FEATURE_SCHEMA_VERSION`, which versions the feature vocabulary in
#: `pvlearn.schema`: the same columns can be assembled into a different model.
#: Bump it in the same commit as any change to `Forecaster._prepare_model_pipeline`
#: or to `PFISelector`.
PIPELINE_VERSION = 2


def sklearn_minor_version() -> str:
    """The `major.minor` part of the installed scikit-learn version.

    Patch releases do not change model behaviour, minor ones can — and a model
    unpickled across a minor bump either warns or predicts differently, both of
    which are worse than retraining.
    """
    major, minor = sklearn.__version__.split(".")[:2]
    return f"{major}.{minor}"


class ModelMetrics(BaseModel):
    """Quality of a model on a holdout, in the unit of the target (Wh)."""

    mae: float
    rmse: float
    r2: float


class ModelMetadata(BaseModel):
    """Sidecar describing what a persisted model was trained on and with.

    Loading a model whose metadata disagrees with the current configuration is
    a hard error, never a best-effort load: a model trained on a different
    feature set, provider, interval, or location keeps predicting plausible
    numbers, so the failure is silent and effectively undebuggable.
    """

    pvlearn_version: str
    feature_schema_version: int
    #: Defaulted so that sidecars written before this field existed are read as
    #: version 1 — which is what they are — and rejected with a precise reason
    #: rather than as unreadable metadata.
    pipeline_version: int = 1
    sklearn_version: str
    weather_provider: str
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
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            pipeline_version=PIPELINE_VERSION,
            sklearn_version=sklearn_minor_version(),
            weather_provider=config.weather_provider,
            interval_minutes=config.interval_minutes,
            location=location,
            trained_at=trained_at or datetime.now().astimezone(),
            training_rows=training_rows,
            selected_features=selected_features,
            metrics=metrics,
        )

    def raise_on_mismatch(self, location: Location, config: ForecasterConfig) -> None:
        """Reject the model unless it was trained under the given setup.

        `pvlearn_version` deliberately does not take part: not every release
        changes how features are built, and the parts that do are covered by
        `feature_schema_version` and `pipeline_version`.
        """
        mismatches = [
            self._compare("feature_schema_version", FEATURE_SCHEMA_VERSION),
            self._compare("pipeline_version", PIPELINE_VERSION),
            self._compare("sklearn_version", sklearn_minor_version()),
            self._compare("weather_provider", config.weather_provider),
            self._compare("interval_minutes", config.interval_minutes),
            self._compare("location", location),
        ]

        reasons = [reason for reason in mismatches if reason is not None]
        if reasons:
            raise SchemaMismatchError(
                "Persisted model does not match the current configuration "
                f"and has to be retrained: {'; '.join(reasons)}"
            )

    def _compare(self, attribute: str, expected: object) -> str | None:
        persisted = getattr(self, attribute)
        if persisted == expected:
            return None

        return f"{attribute} is {persisted!r}, expected {expected!r}"
