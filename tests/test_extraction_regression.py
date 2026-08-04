"""Phase 1a's actual acceptance test.

The Umsetzungsplan requires that `pvlearn`'s extracted code reproduces the
frozen baseline. This retrains both `Forecaster` models on the reference
dataset the same way `scripts/freeze_baseline_forecast.py` trained
solaredge2mqtt's original `Forecaster`, and compares the resulting predictive
quality - not the raw numbers.

Bit-for-bit comparison was tried first and does not hold up: predictions
reproduce exactly on the machine that trains them, but diverge across
different hardware even with `random_state=42` fixed everywhere and identical
library versions. `HistGradientBoostingRegressor`'s greedy split search is
sensitive to CPU-microarchitecture-dependent floating point rounding (SIMD
reduction order), which a razor-edge split threshold can flip into a visibly
different tree. Confirmed this is not a thread-count or Python-version effect
(same machine, `n_jobs=1` and `OMP_NUM_THREADS=1..8`, Python 3.12 and 3.13 -
all reproduce the baseline bit-for-bit; only a different physical CPU, i.e.
CI, diverges). So the extraction is verified against overall predictive
quality (MAE, R²) instead: a real bug would move these outside of noise,
where hardware-driven divergence stays well within it.
"""

import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pvlearn.config import ForecasterConfig
from pvlearn.forecaster import Forecaster, ForecasterType
from pvlearn.location import Location

pytestmark = pytest.mark.slow

# How far MAE/R² may drift from the frozen baseline before it stops looking
# like hardware noise and starts looking like an extraction bug.
RELATIVE_MAE_TOLERANCE = 0.05
ABSOLUTE_R2_TOLERANCE = 0.05

WH_PER_KWH = 1000


@pytest.fixture(scope="module")
def retrained_metrics(
    reference_dataset: pd.DataFrame,
    reference_location: dict,
    baseline_metadata: dict,
) -> dict[str, dict[str, object]]:
    location = Location(**reference_location)
    data = reference_dataset.copy()
    data["time"] = data["_time"].dt.tz_convert(location.timezone)

    holdout = baseline_metadata["holdout_rows"]
    train_data = data.iloc[:-holdout]
    holdout_data = data.iloc[-holdout:]

    config = ForecasterConfig(hyperparametertuning=False)

    metrics: dict[str, dict[str, object]] = {}
    for typed in ForecasterType:
        forecaster = Forecaster(typed, location, config)
        forecaster.train(train_data)

        assert forecaster.model_pipeline is not None
        predicted = forecaster.model_pipeline.predict(holdout_data)
        published = pd.Series(predicted).apply(typed.prepare_value)

        # Energy is published in kWh but recorded in Wh; compare in Wh, same
        # as scripts/freeze_baseline_forecast.py did for the baseline.
        scale = WH_PER_KWH if typed.target_column == "energy" else 1
        actual = holdout_data[typed.target_column].to_numpy()
        comparable = published.to_numpy() * scale

        metrics[typed.target_column] = {
            "mae": float(mean_absolute_error(actual, comparable)),
            "rmse": float(mean_squared_error(actual, comparable) ** 0.5),
            "r2": float(r2_score(actual, comparable)),
            "selected_features": forecaster.model_pipeline.named_steps[
                "feature_selector"
            ].important_features_,
        }

    return metrics


class TestExtractionReproducesBaseline:
    @pytest.mark.parametrize("target", ["energy", "power"])
    def test_mae_matches_baseline_within_tolerance(
        self, target: str, retrained_metrics: dict, baseline_metadata: dict
    ):
        retrained_mae = retrained_metrics[target]["mae"]
        baseline_mae = baseline_metadata["metrics"][target]["mae"]

        assert retrained_mae == pytest.approx(baseline_mae, rel=RELATIVE_MAE_TOLERANCE)

    @pytest.mark.parametrize("target", ["energy", "power"])
    def test_r2_matches_baseline_within_tolerance(
        self, target: str, retrained_metrics: dict, baseline_metadata: dict
    ):
        retrained_r2 = retrained_metrics[target]["r2"]
        baseline_r2 = baseline_metadata["metrics"][target]["r2"]

        assert retrained_r2 == pytest.approx(baseline_r2, abs=ABSOLUTE_R2_TOLERANCE)

    @pytest.mark.parametrize("target", ["energy", "power"])
    def test_sun_elevation_is_still_a_selected_feature(
        self, target: str, retrained_metrics: dict
    ):
        """Losing the sun features to a timezone or encoder bug wouldn't
        necessarily move MAE/R² enough to fail the tolerance checks above."""
        assert "sun__time_elevation" in retrained_metrics[target]["selected_features"]
