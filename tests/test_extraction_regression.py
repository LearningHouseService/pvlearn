"""Phase 1a's actual acceptance test.

The Umsetzungsplan requires that `pvlearn`'s extracted code reproduces the
frozen baseline exactly. This retrains both `Forecaster` models on the
reference dataset the same way `scripts/freeze_baseline_forecast.py` trained
solaredge2mqtt's original `Forecaster`, and compares predictions bit for bit.

If this drifts, the extraction changed behavior somewhere - a schema mismatch,
a dropped column, a different encoder parameterization - and that is exactly
what this test exists to catch before it reaches Phase 1b, where functional
changes make such regressions impossible to attribute.
"""

import pandas as pd
import pytest

from pvlearn.config import ForecasterConfig
from pvlearn.forecaster import Forecaster, ForecasterType
from pvlearn.location import Location

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def retrained_predictions(
    reference_dataset: pd.DataFrame,
    reference_location: dict,
    baseline_metadata: dict,
) -> pd.DataFrame:
    location = Location(**reference_location)
    data = reference_dataset.copy()
    data["time"] = data["_time"].dt.tz_convert(location.timezone)

    holdout = baseline_metadata["holdout_rows"]
    train_data = data.iloc[:-holdout]
    holdout_data = data.iloc[-holdout:]

    config = ForecasterConfig(hyperparametertuning=False)

    predictions = pd.DataFrame({"_time": holdout_data["_time"].to_numpy()})
    for typed in ForecasterType:
        forecaster = Forecaster(typed, location, config)
        forecaster.train(train_data)

        assert forecaster.model_pipeline is not None
        predicted = forecaster.model_pipeline.predict(holdout_data)
        published = pd.Series(predicted).apply(typed.prepare_value)

        predictions[f"{typed.target_column}_raw"] = predicted
        predictions[f"{typed.target_column}_published"] = published.to_numpy()

    return predictions


class TestExtractionReproducesBaseline:
    def test_energy_raw_predictions_are_bit_identical(
        self, retrained_predictions: pd.DataFrame, baseline_forecast: pd.DataFrame
    ):
        pd.testing.assert_series_equal(
            retrained_predictions["energy_raw"],
            baseline_forecast["energy_raw"],
            check_names=False,
        )

    def test_energy_published_predictions_are_bit_identical(
        self, retrained_predictions: pd.DataFrame, baseline_forecast: pd.DataFrame
    ):
        pd.testing.assert_series_equal(
            retrained_predictions["energy_published"],
            baseline_forecast["energy_published"],
            check_names=False,
        )

    def test_power_raw_predictions_are_bit_identical(
        self, retrained_predictions: pd.DataFrame, baseline_forecast: pd.DataFrame
    ):
        pd.testing.assert_series_equal(
            retrained_predictions["power_raw"],
            baseline_forecast["power_raw"],
            check_names=False,
        )

    def test_power_published_predictions_are_bit_identical(
        self, retrained_predictions: pd.DataFrame, baseline_forecast: pd.DataFrame
    ):
        pd.testing.assert_series_equal(
            retrained_predictions["power_published"],
            baseline_forecast["power_published"],
            check_names=False,
        )
