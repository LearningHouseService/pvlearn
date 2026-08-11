"""Guards on the frozen baseline forecasts.

These are the predictions the current solaredge2mqtt implementation produces on the
reference dataset. Phase 1a is accepted only when the extracted pvlearn code
reproduces them, so this file's job is to keep the target honest: it pins what the
baseline is and under which library versions it was produced.

The comparison against pvlearn's own output arrives with the extracted code — there
is nothing to compare against yet.
"""

import pandas as pd

EXPECTED_HOLDOUT_ROWS = 720

EXPECTED_COLUMNS = {
    "_time",
    "energy_raw",
    "energy_published",
    "power_raw",
    "power_published",
}

# The version the frozen artifact was produced under. Pinning it here keeps the
# artifact and the pin in pyproject.toml from drifting apart unnoticed; it says
# nothing about whether another version would reproduce the baseline, which has
# never been measured.
BASELINE_SKLEARN_VERSION = "1.9.0"

WH_PER_KWH = 1000


class TestBaselineArtifact:
    def test_row_count_matches_holdout(self, baseline_forecast: pd.DataFrame):
        assert len(baseline_forecast) == EXPECTED_HOLDOUT_ROWS

    def test_columns_are_frozen(self, baseline_forecast: pd.DataFrame):
        assert set(baseline_forecast.columns) == EXPECTED_COLUMNS

    def test_metadata_matches_predictions(
        self, baseline_forecast: pd.DataFrame, baseline_metadata: dict
    ):
        assert baseline_metadata["holdout_rows"] == len(baseline_forecast)
        assert baseline_metadata["training_rows"] > 0

    def test_records_the_versions_it_depends_on(self, baseline_metadata: dict):
        assert baseline_metadata["versions"]["scikit-learn"] == BASELINE_SKLEARN_VERSION

    def test_was_produced_without_hyperparameter_tuning(self, baseline_metadata: dict):
        """The tuning search would pick different hyperparameters per data run."""
        assert baseline_metadata["hyperparameter_tuning"] is False

    def test_holdout_follows_the_training_data(
        self, baseline_forecast: pd.DataFrame, reference_dataset: pd.DataFrame
    ):
        """The holdout is the tail of the dataset, not a random sample of it."""
        expected_tail = reference_dataset["_time"].tail(EXPECTED_HOLDOUT_ROWS)
        assert baseline_forecast["_time"].tolist() == expected_tail.tolist()


class TestBaselinePredictions:
    def test_predictions_are_finite(self, baseline_forecast: pd.DataFrame):
        for column in EXPECTED_COLUMNS - {"_time"}:
            assert bool(baseline_forecast[column].notna().all())

    def test_published_values_are_never_negative(self, baseline_forecast: pd.DataFrame):
        """ForecasterType.prepare_value clamps predictions at zero."""
        assert bool((baseline_forecast["energy_published"] >= 0).all())
        assert bool((baseline_forecast["power_published"] >= 0).all())

    def test_energy_is_published_in_kilowatt_hours(
        self, baseline_forecast: pd.DataFrame
    ):
        """prepare_value divides energy by 1000, unlike power.

        This unit change is easy to lose in the extraction and would show up as a
        thousandfold forecast error rather than as a crash.
        """
        published_peak = baseline_forecast["energy_published"].max()
        raw_peak = baseline_forecast["energy_raw"].max()

        assert published_peak == round(raw_peak / WH_PER_KWH, 3)

    def test_power_is_published_as_whole_watts(self, baseline_forecast: pd.DataFrame):
        published = baseline_forecast["power_published"]
        assert bool((published == published.round()).all())


class TestBaselineQuality:
    """Sanity bounds on the metrics, not acceptance criteria.

    Phase 1b compares pvlearn's MAE against the recorded baseline value. These
    tests only catch a baseline that is obviously broken rather than merely worse.
    """

    def test_energy_model_explains_most_variance(self, baseline_metadata: dict):
        assert baseline_metadata["metrics"]["energy"]["r2"] > 0.8

    def test_energy_and_power_metrics_agree(self, baseline_metadata: dict):
        """At hourly resolution mean power in W equals energy in Wh.

        The power model was dropped on exactly this basis, so the two MAEs
        should stay within a few percent of one another.
        """
        energy_mae = baseline_metadata["metrics"]["energy"]["mae"]
        power_mae = baseline_metadata["metrics"]["power"]["mae"]

        assert abs(energy_mae - power_mae) / energy_mae < 0.05

    def test_sun_elevation_is_a_selected_feature(self, baseline_metadata: dict):
        """Losing the sun features to a timezone bug would not raise on its own."""
        for target in ("energy", "power"):
            selected = baseline_metadata["metrics"][target]["selected_features"]
            assert "sun__time_elevation" in selected
