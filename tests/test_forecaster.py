"""Tests for pvlearn.forecaster.

Only the parts that moved to pvlearn are covered here: `Forecaster` and
`PFISelector`. Everything event-bus/InfluxDB/MQTT-shaped stayed behind in
solaredge2mqtt's `ForecastService` and is out of scope.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from pvlearn.config import ForecasterConfig
from pvlearn.exceptions import (
    InsufficientDataError,
    ModelNotTrainedError,
    SchemaMismatchError,
)
from pvlearn.forecaster import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    Forecaster,
    PFISelector,
)
from pvlearn.location import Location
from pvlearn.metadata import ModelMetrics

LOCAL_TZ = "Europe/Berlin"


def make_location(latitude: float = 52.52, longitude: float = 13.405) -> Location:
    return Location(latitude=latitude, longitude=longitude, timezone=LOCAL_TZ)


def make_config(**overrides) -> ForecasterConfig:
    return ForecasterConfig(
        **{"interval_minutes": 60, "weather_provider": "openweathermap", **overrides}
    )


def make_training_data(rows: int = 100, weather: bool = True) -> DataFrame:
    columns: dict[str, list] = {
        "time": [datetime.now(timezone.utc) + timedelta(hours=i) for i in range(rows)],
        "energy": [100.0 + i for i in range(rows)],
    }

    if weather:
        columns.update(
            {
                "cloud_cover": [50] * rows,
                "temperature": [25.0] * rows,
                "relative_humidity": [50] * rows,
                "surface_pressure": [1013] * rows,
                "wind_speed": [5.0] * rows,
                "wind_direction": [180] * rows,
                "uv_index": [5.0] * rows,
                "precipitation_probability": [0.1] * rows,
                "condition_code": [800] * rows,
            }
        )

    data = DataFrame(columns)
    data["time"] = data["time"].astype(pd.DatetimeTZDtype(unit="ns", tz=LOCAL_TZ))
    return data


def make_metrics() -> ModelMetrics:
    return ModelMetrics(mae=1.0, rmse=2.0, r2=0.9)


def make_mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.named_steps = {
        "preprocessor": MagicMock(
            get_feature_names_out=MagicMock(return_value=["f1"]),
        ),
        "feature_selector": MagicMock(important_features_=["f1"]),
    }
    return pipeline


class TestPrepareValue:
    def test_clamps_negative_to_zero(self):
        assert Forecaster.prepare_value(-5) == 0

    def test_rounds_to_whole_watt_hours(self):
        assert Forecaster.prepare_value(1234.6) == 1235
        assert isinstance(Forecaster.prepare_value(1234.6), int)

    def test_does_not_convert_to_kilowatt_hours(self):
        """Phase 1b publishes Wh; kWh was the old power/energy split's doing."""
        assert Forecaster.prepare_value(1234) == 1234


class TestForecasterInit:
    def test_forecaster_init(self):
        location = make_location()
        config = make_config()

        forecaster = Forecaster(location, config)

        assert forecaster.location == location
        assert forecaster.interval_minutes == 60
        assert forecaster.enable_hyperparameter_tuning is False
        assert forecaster.model_pipeline is None
        assert forecaster.metadata is None

    def test_forecaster_init_with_hyperparameter_tuning(self):
        forecaster = Forecaster(make_location(), make_config(hyperparametertuning=True))

        assert forecaster.enable_hyperparameter_tuning is True

    def test_forecaster_is_trained_false_initially(self):
        forecaster = Forecaster(make_location(), make_config())

        assert forecaster.is_trained is False

    def test_forecaster_is_trained_true_after_training(self):
        forecaster = Forecaster(make_location(), make_config())
        forecaster.model_pipeline = MagicMock()

        assert forecaster.is_trained is True

    def test_forecaster_no_memory_when_caching_disabled(self):
        forecaster = Forecaster(make_location(), make_config())

        assert forecaster.memory is None

    def test_minimum_training_rows_follows_the_interval(self):
        forecaster = Forecaster(make_location(), make_config())

        assert forecaster.minimum_training_rows == 60


class TestForecasterTrain:
    def test_train_raises_with_insufficient_data(self):
        forecaster = Forecaster(make_location(), make_config())

        with pytest.raises(InsufficientDataError, match="at least 60 hours"):
            forecaster.train(make_training_data(rows=30))

    def test_train_creates_pipeline(self):
        forecaster = Forecaster(make_location(), make_config())

        forecaster.train(make_training_data())

        assert forecaster.model_pipeline is not None
        assert forecaster.is_trained is True

    def test_train_records_metadata(self):
        forecaster = Forecaster(make_location(), make_config())
        data = make_training_data()

        forecaster.train(data)

        assert forecaster.metadata is not None
        assert forecaster.metadata.training_rows == len(data)
        assert forecaster.metadata.weather_provider == "openweathermap"
        assert forecaster.metadata.interval_minutes == 60
        assert forecaster.metadata.location == forecaster.location
        assert forecaster.metadata.selected_features

    def test_train_records_holdout_metrics(self):
        forecaster = Forecaster(make_location(), make_config())

        forecaster.train(make_training_data())

        assert forecaster.metadata is not None
        metrics = forecaster.metadata.metrics
        assert metrics.mae >= 0
        assert metrics.rmse >= metrics.mae

    def test_train_uses_hyperparameter_tuning_when_enabled(self):
        forecaster = Forecaster(make_location(), make_config(hyperparametertuning=True))
        tuned_pipeline = make_mock_pipeline()

        with (
            patch.object(
                forecaster, "_prepare_model_pipeline", return_value=MagicMock()
            ),
            patch.object(
                forecaster, "_hyperparametertuning", return_value=tuned_pipeline
            ) as mock_tune,
            patch.object(forecaster, "_evaluate", return_value=make_metrics()),
        ):
            forecaster.train(make_training_data(rows=70, weather=False))

        mock_tune.assert_called_once()
        tuned_pipeline.fit.assert_called_once()

    def test_train_calls_cleanup_cache(self):
        forecaster = Forecaster(make_location(), make_config())

        with (
            patch.object(
                forecaster, "_prepare_model_pipeline", return_value=make_mock_pipeline()
            ),
            patch.object(forecaster, "_evaluate", return_value=make_metrics()),
            patch.object(forecaster, "_cleanup_cache") as mock_cleanup,
        ):
            forecaster.train(make_training_data(rows=70, weather=False))

        mock_cleanup.assert_called_once()

    def test_evaluate_scores_a_copy_of_the_pipeline(self):
        """The kept model must see the holdout too, so evaluation clones."""
        forecaster = Forecaster(make_location(), make_config())
        data = make_training_data()
        pipeline = forecaster._prepare_model_pipeline(data.columns.to_list())
        split = forecaster._holdout_split(data)

        metrics = forecaster._evaluate(
            data, cast(Series, data["energy"]), split, pipeline
        )

        assert metrics.mae >= 0
        with pytest.raises(NotFittedError):
            pipeline.predict(data)

    def test_train_sorts_the_data_chronologically(self):
        """TimeSeriesSplit cuts by position, so unsorted rows would make the
        holdout a random sample and the recorded metrics meaningless."""
        forecaster = Forecaster(make_location(), make_config())
        shuffled = make_training_data().sample(frac=1, random_state=7)

        captured: dict[str, DataFrame] = {}
        real_split = forecaster._holdout_split

        def capture(data: DataFrame):
            captured["data"] = data
            return real_split(data)

        with patch.object(forecaster, "_holdout_split", side_effect=capture):
            forecaster.train(shuffled)

        assert captured["data"]["time"].is_monotonic_increasing

    def test_tuning_never_sees_the_holdout(self):
        """Otherwise the recorded metrics are scored on data that helped pick
        the hyperparameters, and they flatter the model."""
        forecaster = Forecaster(make_location(), make_config(hyperparametertuning=True))
        data = make_training_data()

        with (
            patch.object(
                forecaster,
                "_hyperparametertuning",
                side_effect=lambda tuning_data, _y, pipeline: pipeline,
            ) as mock_tune,
            patch.object(forecaster, "_evaluate", return_value=make_metrics()),
        ):
            forecaster.train(data)

        tuning_data = mock_tune.call_args.args[0]
        _, test_index = forecaster._holdout_split(data)
        holdout_times = set(data.iloc[test_index]["time"])

        assert holdout_times.isdisjoint(set(tuning_data["time"]))

    def test_failed_training_releases_waiters(self):
        """A predict() awaiting training must never block on a model that will
        not arrive."""
        forecaster = Forecaster(make_location(), make_config())

        with patch.object(forecaster, "_evaluate", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                forecaster.train(make_training_data())

        assert forecaster.training_completed.is_set()

    def test_failed_training_keeps_the_previous_model(self):
        forecaster = Forecaster(make_location(), make_config())
        data = make_training_data()
        forecaster.train(data)
        working_pipeline = forecaster.model_pipeline
        working_metadata = forecaster.metadata

        with patch.object(forecaster, "_evaluate", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                forecaster.train(data)

        assert forecaster.model_pipeline is working_pipeline
        assert forecaster.metadata is working_metadata

    def test_cleanup_cache_calls_reduce_size_when_memory_available(self):
        forecaster = Forecaster(make_location(), make_config())
        forecaster.memory = MagicMock()

        forecaster._cleanup_cache()

        forecaster.memory.reduce_size.assert_called_once_with(
            bytes_limit=forecaster.cache_size_limit_bytes
        )

    def test_cleanup_cache_noop_without_memory(self):
        forecaster = Forecaster(make_location(), make_config())

        forecaster._cleanup_cache()  # must not raise

    def test_hyperparametertuning_returns_cloned_best_estimator(self):
        """Exercises the real HalvingGridSearchCV path with a minimal pipeline.

        A full preprocessor would make this as expensive as real training;
        param_grid only touches the "model" step, so a bare estimator is enough.
        """
        forecaster = Forecaster(make_location(), make_config())
        data = DataFrame({"x": range(30)})
        y_vector = Series(range(30), dtype=float)
        pipeline = Pipeline(
            steps=[("model", HistGradientBoostingRegressor(random_state=42))]
        )

        tuned = forecaster._hyperparametertuning(data, y_vector, pipeline)

        assert isinstance(tuned, Pipeline)
        assert tuned is not pipeline


class TestForecasterPredict:
    async def test_predict_raises_when_not_trained(self):
        forecaster = Forecaster(make_location(), make_config())

        data = DataFrame({"time": [datetime.now()], "energy": [100.0]})

        with pytest.raises(ModelNotTrainedError, match="not been trained"):
            await forecaster.predict(data)

    async def test_predict_returns_predictions(self):
        forecaster = Forecaster(make_location(), make_config())

        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [100.0, 150.0]
        forecaster.model_pipeline = mock_pipeline
        forecaster.training_completed.set()

        data = DataFrame(
            {"time": [datetime.now(), datetime.now() + timedelta(hours=1)]}
        )

        result = await forecaster.predict(data)

        assert result["energy"].tolist() == [100, 150]
        mock_pipeline.predict.assert_called_once()

    async def test_predict_handles_tuple_predictions(self):
        forecaster = Forecaster(make_location(), make_config())
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = ([100.0, -200.0],)
        forecaster.model_pipeline = mock_pipeline
        forecaster.training_completed.set()

        data = DataFrame(
            {"time": [datetime.now(), datetime.now() + timedelta(hours=1)]}
        )

        result = await forecaster.predict(data)

        assert result["energy"].tolist() == [100, 0]

    async def test_predict_keeps_the_input_index(self):
        forecaster = Forecaster(make_location(), make_config())
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [100.0, 150.0]
        forecaster.model_pipeline = mock_pipeline
        forecaster.training_completed.set()

        data = DataFrame(
            {"time": [datetime.now(), datetime.now() + timedelta(hours=1)]},
            index=[7, 8],
        )

        result = await forecaster.predict(data)

        assert result["energy"].tolist() == [100, 150]


class TestForecasterPersistence:
    def test_save_raises_without_a_trained_model(self, tmp_path):
        forecaster = Forecaster(make_location(), make_config())

        with pytest.raises(ModelNotTrainedError, match="no trained model"):
            forecaster.save(tmp_path)

    def test_save_writes_model_and_metadata(self, tmp_path):
        forecaster = Forecaster(make_location(), make_config())
        forecaster.train(make_training_data())

        forecaster.save(tmp_path / "brain")

        assert (tmp_path / "brain" / MODEL_FILENAME).is_file()
        assert (tmp_path / "brain" / METADATA_FILENAME).is_file()

    def test_load_restores_a_usable_forecaster(self, tmp_path):
        location = make_location()
        config = make_config()
        original = Forecaster(location, config)
        original.train(make_training_data())
        original.save(tmp_path)

        restored = Forecaster.load(tmp_path, location, config)

        assert restored.is_trained is True
        assert restored.training_completed.is_set()
        assert restored.metadata == original.metadata

    async def test_loaded_forecaster_predicts_the_same_values(self, tmp_path):
        location = make_location()
        config = make_config()
        data = make_training_data()
        original = Forecaster(location, config)
        original.train(data)
        original.save(tmp_path)

        restored = Forecaster.load(tmp_path, location, config)

        expected = await original.predict(data)
        actual = await restored.predict(data)
        assert actual["energy"].tolist() == expected["energy"].tolist()

    def test_load_raises_when_nothing_is_persisted(self, tmp_path):
        with pytest.raises(ModelNotTrainedError, match="No persisted model"):
            Forecaster.load(tmp_path, make_location(), make_config())

    def test_load_raises_when_only_the_model_is_persisted(self, tmp_path):
        forecaster = Forecaster(make_location(), make_config())
        forecaster.train(make_training_data())
        forecaster.save(tmp_path)
        (tmp_path / METADATA_FILENAME).unlink()

        with pytest.raises(ModelNotTrainedError, match="No persisted model"):
            Forecaster.load(tmp_path, make_location(), make_config())

    def test_load_rejects_a_model_trained_at_another_location(self, tmp_path):
        forecaster = Forecaster(make_location(), make_config())
        forecaster.train(make_training_data())
        forecaster.save(tmp_path)

        with pytest.raises(SchemaMismatchError, match="location"):
            Forecaster.load(tmp_path, make_location(latitude=48.1), make_config())

    def test_load_rejects_a_model_trained_on_another_provider(self, tmp_path):
        forecaster = Forecaster(make_location(), make_config())
        forecaster.train(make_training_data())
        forecaster.save(tmp_path)

        with pytest.raises(SchemaMismatchError, match="weather_provider"):
            Forecaster.load(
                tmp_path,
                make_location(),
                make_config(weather_provider="open-meteo"),
            )

    def test_load_rejects_unreadable_metadata_as_a_mismatch(self, tmp_path):
        """A caller catching the documented errors to retrain must not crash on
        a sidecar written by another version."""
        forecaster = Forecaster(make_location(), make_config())
        forecaster.train(make_training_data())
        forecaster.save(tmp_path)
        (tmp_path / METADATA_FILENAME).write_text("{ truncated")

        with pytest.raises(SchemaMismatchError, match="unreadable"):
            Forecaster.load(tmp_path, make_location(), make_config())

    def test_load_rejects_metadata_missing_required_fields(self, tmp_path):
        forecaster = Forecaster(make_location(), make_config())
        forecaster.train(make_training_data())
        forecaster.save(tmp_path)
        (tmp_path / METADATA_FILENAME).write_text(
            json.dumps({"pvlearn_version": "9.9"})
        )

        with pytest.raises(SchemaMismatchError, match="unreadable"):
            Forecaster.load(tmp_path, make_location(), make_config())

    def test_load_rejects_a_corrupt_model_file(self, tmp_path):
        forecaster = Forecaster(make_location(), make_config())
        forecaster.train(make_training_data())
        forecaster.save(tmp_path)
        (tmp_path / MODEL_FILENAME).write_bytes(b"not a pickle")

        with pytest.raises(SchemaMismatchError, match="could not be loaded"):
            Forecaster.load(tmp_path, make_location(), make_config())

    def test_save_leaves_the_previous_pair_intact_when_it_fails(self, tmp_path):
        """Half a save must not produce a model described by stale metadata."""
        forecaster = Forecaster(make_location(), make_config())
        forecaster.train(make_training_data())
        forecaster.save(tmp_path)
        first = Forecaster.load(tmp_path, make_location(), make_config())

        forecaster.train(make_training_data(rows=120))
        with patch("pvlearn.forecaster.dump", side_effect=OSError("no space left")):
            with pytest.raises(OSError):
                forecaster.save(tmp_path)

        restored = Forecaster.load(tmp_path, make_location(), make_config())
        assert restored.metadata == first.metadata
        assert not list(tmp_path.glob("*.tmp"))

    def test_load_rejects_an_outdated_feature_schema(self, tmp_path):
        forecaster = Forecaster(make_location(), make_config())
        forecaster.train(make_training_data())
        forecaster.save(tmp_path)

        metadata_path = tmp_path / METADATA_FILENAME
        metadata = json.loads(metadata_path.read_text())
        metadata["feature_schema_version"] = 0
        metadata_path.write_text(json.dumps(metadata))

        with pytest.raises(SchemaMismatchError, match="feature_schema_version"):
            Forecaster.load(tmp_path, make_location(), make_config())

    def test_load_rejects_a_model_from_an_older_pipeline(self, tmp_path):
        """A sidecar written before `pipeline_version` existed reads as 1 and
        has to be retrained, not loaded."""
        forecaster = Forecaster(make_location(), make_config())
        forecaster.train(make_training_data())
        forecaster.save(tmp_path)

        metadata_path = tmp_path / METADATA_FILENAME
        metadata = json.loads(metadata_path.read_text())
        del metadata["pipeline_version"]
        metadata_path.write_text(json.dumps(metadata))

        with pytest.raises(SchemaMismatchError, match="pipeline_version is 1"):
            Forecaster.load(tmp_path, make_location(), make_config())


class TestForecasterExtractUsedColumns:
    def test_extract_used_columns_with_list(self):
        features = ["col1", "col2", "col3"]
        columns = ["col1", "col3", "col4"]

        result = Forecaster._extract_used_columns(features, columns)

        assert result == ["col1", "col3"]

    def test_extract_used_columns_with_dict(self):
        features = {"col1": 24, "col2": 12}
        columns = ["col1", "col3"]

        result = Forecaster._extract_used_columns(features, columns)

        assert result == ["col1"]

    def test_extract_used_columns_no_match(self):
        features = ["col1", "col2"]
        columns = ["col3", "col4"]

        result = Forecaster._extract_used_columns(features, columns)

        assert result == []


class TestPFISelector:
    def test_pfi_selector_init(self):
        mock_estimator = MagicMock()

        selector = PFISelector(estimator=mock_estimator, n_repeats=5)

        assert selector.estimator == mock_estimator
        assert selector.n_repeats == 5
        assert selector.n_std == 1.0

    def test_pfi_selector_transform_not_fitted_raises(self):
        mock_estimator = MagicMock()
        selector = PFISelector(estimator=mock_estimator)

        data = DataFrame({"col1": [1, 2, 3]})

        with pytest.raises(RuntimeError, match="not been fitted"):
            selector.transform(data)

    def test_pfi_selector_get_support_not_fitted_raises(self):
        mock_estimator = MagicMock()
        selector = PFISelector(estimator=mock_estimator)

        with pytest.raises(RuntimeError, match="not been fitted"):
            selector.get_support()

    def test_pfi_selector_transform_returns_selected_features(self):
        mock_estimator = MagicMock()
        selector = PFISelector(estimator=mock_estimator)
        selector.important_features_ = ["col1", "col3"]

        data = DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [7, 8, 9],
            }
        )

        result = selector.transform(data)

        assert list(result.columns) == ["col1", "col3"]
        assert "col2" not in result.columns

    def test_pfi_selector_get_support_returns_indices(self):
        selector = PFISelector(estimator=MagicMock())
        selector.important_indices_ = [True, False, True]

        assert selector.get_support() == [True, False, True]

    def test_pfi_selector_keeps_every_feature_with_positive_importance(self):
        """Absolute criterion, not a quantile of the candidates (ADR 0001)."""
        selector = PFISelector(estimator=HistGradientBoostingRegressor(random_state=42))
        rows = 200
        data = DataFrame(
            {
                "signal": [float(i % 24) for i in range(rows)],
                "noise": [0.0] * rows,
            }
        )
        target = Series([float(i % 24) * 10 for i in range(rows)])

        selector.fit(data, target)

        assert "signal" in (selector.important_features_ or [])
        assert "noise" not in (selector.important_features_ or [])

    def test_pfi_selector_selection_does_not_depend_on_candidate_count(self):
        """Adding a useless column must not evict a useful one."""
        rows = 200
        signal = [float(i % 24) for i in range(rows)]
        target = Series([value * 10 for value in signal])
        data = DataFrame({"signal": signal, "second": [v * 2 for v in signal]})

        def select(frame: DataFrame) -> list[str]:
            selector = PFISelector(
                estimator=HistGradientBoostingRegressor(random_state=42)
            )
            selector.fit(frame, target)
            return selector.important_features_ or []

        without_extra = select(data)
        with_extra = select(data.assign(padding=[0.0] * rows))

        assert set(without_extra) <= set(with_extra)

    def test_pfi_selector_keeps_all_features_when_none_helps(self):
        """An uninformative importance estimate is not evidence against them."""
        selector = PFISelector(estimator=MagicMock())
        data = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        with patch(
            "pvlearn.forecaster.permutation_importance",
            return_value={
                "importances_mean": np.array([-0.1, -0.2]),
                "importances_std": np.array([0.01, 0.01]),
            },
        ):
            selector.fit(data, Series([1.0, 2.0, 3.0]))

        assert selector.important_features_ == ["a", "b"]

    def test_pfi_selector_drops_features_below_the_permutation_noise(self):
        """A mean importance smaller than the spread of its own repeats says
        nothing about the feature (ADR 0002)."""
        selector = PFISelector(estimator=MagicMock(), n_std=1.0)
        data = DataFrame({"solid": [1, 2, 3], "noisy": [4, 5, 6]})

        with patch(
            "pvlearn.forecaster.permutation_importance",
            return_value={
                "importances_mean": np.array([0.5, 0.05]),
                "importances_std": np.array([0.1, 0.2]),
            },
        ):
            selector.fit(data, Series([1.0, 2.0, 3.0]))

        assert selector.important_features_ == ["solid"]

    def test_pfi_selector_measures_importance_on_the_most_recent_rows(self):
        """A shuffled holdout puts a row's neighbours on both sides of the
        split, and autocorrelated hours then flatter every importance."""
        selector = PFISelector(estimator=MagicMock())
        rows = 100
        data = DataFrame({"a": list(range(rows))})
        target = Series([float(index) for index in range(rows)])

        with patch(
            "pvlearn.forecaster.permutation_importance",
            return_value={
                "importances_mean": np.array([1.0]),
                "importances_std": np.array([0.0]),
            },
        ) as importance:
            selector.fit(data, target)

        held_out = importance.call_args.args[1]
        assert held_out["a"].to_list() == list(range(90, 100))

    def test_pfi_selector_keeps_cyclical_pairs_together(self):
        """A cos without its sin cannot tell morning from afternoon."""
        selector = PFISelector(estimator=MagicMock())
        data = DataFrame(
            {"angle_sin": [1, 2, 3], "angle_cos": [4, 5, 6], "other": [7, 8, 9]}
        )

        with patch(
            "pvlearn.forecaster.permutation_importance",
            return_value={
                "importances_mean": np.array([0.5, -0.1, 0.5]),
                "importances_std": np.array([0.0, 0.0, 0.0]),
            },
        ):
            selector.fit(data, Series([1.0, 2.0, 3.0]))

        assert selector.important_features_ == ["angle_sin", "angle_cos", "other"]

    def test_pfi_selector_drops_a_cyclical_pair_when_neither_half_helps(self):
        selector = PFISelector(estimator=MagicMock())
        data = DataFrame(
            {"angle_sin": [1, 2, 3], "angle_cos": [4, 5, 6], "other": [7, 8, 9]}
        )

        with patch(
            "pvlearn.forecaster.permutation_importance",
            return_value={
                "importances_mean": np.array([-0.2, -0.1, 0.5]),
                "importances_std": np.array([0.0, 0.0, 0.0]),
            },
        ):
            selector.fit(data, Series([1.0, 2.0, 3.0]))

        assert selector.important_features_ == ["other"]

    def test_pfi_selector_holdout_leaves_both_sides_non_empty(self):
        """Frames too short for a tenth still have to split somewhere."""
        assert PFISelector._holdout_start(3) == 2
        assert PFISelector._holdout_start(2) == 1
        assert PFISelector._holdout_start(100) == 90

    def test_pfi_selector_splits_a_plain_array_target(self):
        """`Forecaster` passes a Series, but a Pipeline caller may hand the
        step a bare array."""
        train, test = PFISelector._split_target(np.arange(10), 8)

        assert train.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
        assert test.tolist() == [8, 9]


class TestForecasterPreparePreprocessor:
    def test_prepare_preprocessor_returns_column_transformer(self):
        forecaster = Forecaster(make_location(), make_config())

        columns = ["time", "cloud_cover", "temperature", "condition_code"]
        preprocessor = forecaster._prepare_preprocessor(columns)

        assert isinstance(preprocessor, ColumnTransformer)

    def test_sun_encoder_gets_the_configured_interval(self):
        forecaster = Forecaster(make_location(), make_config())

        preprocessor = forecaster._prepare_preprocessor(["time"])

        sun_encoder = dict(
            (name, transformer) for name, transformer, _ in preprocessor.transformers
        )["sun"]
        assert sun_encoder.interval_minutes == 60


class TestForecasterPrepareModelPipeline:
    def test_prepare_model_pipeline_returns_pipeline(self):
        forecaster = Forecaster(make_location(), make_config())

        pipeline = forecaster._prepare_model_pipeline(["time", "cloud_cover"])

        assert isinstance(pipeline, Pipeline)

    def test_prepare_model_pipeline_has_steps(self):
        forecaster = Forecaster(make_location(), make_config())

        pipeline = forecaster._prepare_model_pipeline(["time", "cloud_cover"])

        step_names = [name for name, _ in pipeline.steps]
        assert "preprocessor" in step_names
        assert "feature_selector" in step_names
        assert "model" in step_names
