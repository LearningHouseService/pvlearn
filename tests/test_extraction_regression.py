"""Phase 1b's acceptance test: the consolidated model against the baseline.

The Umsetzungsplan requires that forecast quality on the reference dataset is
not worse than the frozen baseline, which was produced by solaredge2mqtt's
original two-model code. What is compared is predictive quality, not the raw
numbers.

Bit-for-bit comparison was tried in Phase 1a and does not hold up: predictions
reproduce exactly on the machine that trains them, but diverge across different
hardware even with `random_state=42` fixed everywhere and identical library
versions. `HistGradientBoostingRegressor`'s greedy split search is sensitive to
CPU-microarchitecture-dependent floating point rounding (SIMD reduction order),
which a razor-edge split threshold can flip into a visibly different tree.
Confirmed this is not a thread-count or Python-version effect (same machine,
`n_jobs=1` and `OMP_NUM_THREADS=1..8`, Python 3.12 and 3.13 - all reproduce the
baseline bit-for-bit; only a different physical CPU, i.e. CI, diverges).

Phase 1b adds a second source of divergence on top: the feature set itself
changed. Time of day is now minutes since midnight, the season feature is gone,
the columns arrive under canonical names, and `PFISelector` cuts at an absolute
importance instead of the 75th percentile of the candidates.

The tolerance covers hardware noise, not model changes. When the canonical
schema first shifted the selector's percentile cut, quality dropped 7.67% and
stayed inside this tolerance - the cause was isolated and fixed rather than
absorbed (`docs/adr/0001-feature-selection-threshold.md`). A future deviation
inside the tolerance deserves the same treatment: check whether it comes from
the machine or from the model.
"""

import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pvlearn.config import ForecasterConfig
from pvlearn.forecaster import Forecaster
from pvlearn.location import Location

pytestmark = pytest.mark.slow

# How far MAE/R² may drift from the frozen baseline before it stops looking
# like hardware noise plus the schema change and starts looking like a
# regression. 5% was tried in Phase 1a and undershot: CI's energy MAE landed
# 5.26% off, comfortably within what a different tree from CPU-rounding-flipped
# splits explains, so this is calibrated up rather than down.
RELATIVE_MAE_TOLERANCE = 0.10
ABSOLUTE_R2_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def retrained(
    canonical_reference_dataset: pd.DataFrame,
    reference_location: dict,
    baseline_metadata: dict,
) -> tuple[Forecaster, dict[str, float]]:
    location = Location(**reference_location)
    data = canonical_reference_dataset.copy()
    data["time"] = data["_time"].dt.tz_convert(location.timezone)

    holdout = baseline_metadata["holdout_rows"]
    train_data = data.iloc[:-holdout]
    holdout_data = data.iloc[-holdout:]

    config = ForecasterConfig(
        interval_minutes=60,
        hyperparametertuning=False,
    )

    forecaster = Forecaster(location, config)
    forecaster.train(train_data)

    assert forecaster.model_pipeline is not None
    predicted = forecaster.model_pipeline.predict(holdout_data)
    published = pd.Series(predicted).apply(forecaster.prepare_value)

    actual = holdout_data["energy"].to_numpy()
    comparable = published.to_numpy()

    metrics = {
        "mae": float(mean_absolute_error(actual, comparable)),
        "rmse": float(mean_squared_error(actual, comparable) ** 0.5),
        "r2": float(r2_score(actual, comparable)),
    }

    return forecaster, metrics


@pytest.fixture(scope="module")
def baseline_energy_metrics(baseline_metadata: dict) -> dict:
    return baseline_metadata["metrics"]["energy"]


class TestConsolidatedModelMatchesBaseline:
    def test_mae_is_not_worse_than_the_baseline(
        self, retrained: tuple[Forecaster, dict], baseline_energy_metrics: dict
    ):
        _, metrics = retrained

        assert metrics["mae"] == pytest.approx(
            baseline_energy_metrics["mae"], rel=RELATIVE_MAE_TOLERANCE
        )

    def test_r2_is_not_worse_than_the_baseline(
        self, retrained: tuple[Forecaster, dict], baseline_energy_metrics: dict
    ):
        _, metrics = retrained

        assert metrics["r2"] == pytest.approx(
            baseline_energy_metrics["r2"], abs=ABSOLUTE_R2_TOLERANCE
        )

    def test_sun_elevation_is_still_a_selected_feature(
        self, retrained: tuple[Forecaster, dict]
    ):
        """Losing the sun features to a timezone or encoder bug wouldn't
        necessarily move MAE/R² enough to fail the tolerance checks above."""
        forecaster, _ = retrained
        assert forecaster.metadata is not None

        assert "sun__time_elevation" in forecaster.metadata.selected_features

    def test_recorded_metrics_agree_with_the_holdout_they_are_measured_on(
        self, retrained: tuple[Forecaster, dict]
    ):
        """The metadata metrics come from an internal TimeSeriesSplit holdout,
        the numbers above from the frozen one. Different data, so they will not
        be equal - but an order of magnitude apart would mean the internal
        holdout is measuring something else entirely."""
        forecaster, metrics = retrained
        assert forecaster.metadata is not None

        recorded = forecaster.metadata.metrics
        assert 0.5 < recorded.mae / metrics["mae"] < 2.0
        assert recorded.r2 > 0.7
