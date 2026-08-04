"""Tests for pvlearn.forecaster, ported from solaredge2mqtt's ForecastService.

Only the parts that moved to pvlearn are covered here: `Forecaster`,
`PFISelector`, and `ForecasterType`. Everything event-bus/InfluxDB/MQTT-shaped
stayed behind in solaredge2mqtt's `ForecastService` and is out of scope.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from pvlearn.config import ForecasterConfig
from pvlearn.exceptions import InsufficientDataError, ModelNotTrainedError
from pvlearn.forecaster import Forecaster, ForecasterType, PFISelector
from pvlearn.location import Location

LOCAL_TZ = "Europe/Berlin"


def make_location(latitude: float = 52.52, longitude: float = 13.405) -> Location:
    return Location(latitude=latitude, longitude=longitude, timezone=LOCAL_TZ)


class TestForecasterType:
    def test_target_column_matches_value(self):
        assert ForecasterType.ENERGY.target_column == "energy"
        assert ForecasterType.POWER.target_column == "power"

    def test_prepare_value_clamps_negative_to_zero(self):
        assert ForecasterType.ENERGY.prepare_value(-5) == 0
        assert ForecasterType.POWER.prepare_value(-5) == 0

    def test_prepare_value_energy_divides_by_thousand(self):
        assert ForecasterType.ENERGY.prepare_value(1234) == 1.234

    def test_prepare_value_power_rounds_to_int(self):
        assert ForecasterType.POWER.prepare_value(1234.6) == 1235
        assert isinstance(ForecasterType.POWER.prepare_value(1234.6), int)


class TestForecasterInit:
    def test_forecaster_init(self):
        location = make_location()
        config = ForecasterConfig()

        forecaster = Forecaster(ForecasterType.ENERGY, location, config)

        assert forecaster.typed == ForecasterType.ENERGY
        assert forecaster.location == location
        assert forecaster.enable_hyperparameter_tuning is False
        assert forecaster.model_pipeline is None

    def test_forecaster_init_with_hyperparameter_tuning(self):
        location = make_location()
        config = ForecasterConfig(hyperparametertuning=True)

        forecaster = Forecaster(ForecasterType.POWER, location, config)

        assert forecaster.enable_hyperparameter_tuning is True

    def test_forecaster_is_trained_false_initially(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )

        assert forecaster.is_trained is False

    def test_forecaster_is_trained_true_after_training(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )
        forecaster.model_pipeline = MagicMock()

        assert forecaster.is_trained is True

    def test_forecaster_no_memory_when_caching_disabled(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )

        assert forecaster.memory is None


class TestForecasterTrain:
    def test_train_raises_with_insufficient_data(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )

        data = DataFrame(
            {
                "time": [datetime.now() for _ in range(30)],
                "energy": [100.0] * 30,
                "clouds": [50] * 30,
            }
        )

        with pytest.raises(InsufficientDataError, match="at least 60 hours"):
            forecaster.train(data)

    def test_train_creates_pipeline(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )

        data = DataFrame(
            {
                "time": [
                    datetime.now(timezone.utc) + timedelta(hours=i) for i in range(100)
                ],
                "energy": [100.0 + i for i in range(100)],
                "power": [1000 + i * 10 for i in range(100)],
                "clouds": [50] * 100,
                "temp": [25.0] * 100,
                "humidity": [50] * 100,
                "pressure": [1013] * 100,
                "wind_speed": [5.0] * 100,
                "wind_deg": [180] * 100,
                "uvi": [5.0] * 100,
                "pop": [0.1] * 100,
                "weather_id": [800] * 100,
                "weather_main": ["Clear"] * 100,
            }
        )
        data["time"] = data["time"].astype(pd.DatetimeTZDtype(unit="ns", tz=LOCAL_TZ))

        forecaster.train(data)

        assert forecaster.model_pipeline is not None
        assert forecaster.is_trained is True

    def test_train_uses_hyperparameter_tuning_when_enabled(self):
        config = ForecasterConfig(hyperparametertuning=True)
        forecaster = Forecaster(ForecasterType.ENERGY, make_location(), config)

        data = DataFrame(
            {
                "time": [
                    datetime.now(timezone.utc) + timedelta(hours=i) for i in range(70)
                ],
                "energy": [100.0 + i for i in range(70)],
            }
        )
        data["time"] = data["time"].astype(pd.DatetimeTZDtype(unit="ns", tz=LOCAL_TZ))

        prepared_pipeline = MagicMock()
        tuned_pipeline = MagicMock()
        tuned_pipeline.named_steps = {
            "preprocessor": MagicMock(
                get_feature_names_out=MagicMock(return_value=["f1"])
            ),
            "feature_selector": MagicMock(important_features_=["f1"]),
        }

        with (
            patch.object(
                forecaster, "_prepare_model_pipeline", return_value=prepared_pipeline
            ),
            patch.object(
                forecaster, "_hyperparametertuning", return_value=tuned_pipeline
            ) as mock_tune,
        ):
            forecaster.train(data)

        mock_tune.assert_called_once()
        tuned_pipeline.fit.assert_called_once()

    def test_train_calls_cleanup_cache(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )

        data = DataFrame(
            {
                "time": [
                    datetime.now(timezone.utc) + timedelta(hours=i) for i in range(70)
                ],
                "energy": [100.0 + i for i in range(70)],
            }
        )
        data["time"] = data["time"].astype(pd.DatetimeTZDtype(unit="ns", tz=LOCAL_TZ))

        prepared_pipeline = MagicMock()
        prepared_pipeline.named_steps = {
            "preprocessor": MagicMock(
                get_feature_names_out=MagicMock(return_value=["f1"])
            ),
            "feature_selector": MagicMock(important_features_=["f1"]),
        }

        with (
            patch.object(
                forecaster, "_prepare_model_pipeline", return_value=prepared_pipeline
            ),
            patch.object(forecaster, "_cleanup_cache") as mock_cleanup,
        ):
            forecaster.train(data)

        mock_cleanup.assert_called_once()

    def test_cleanup_cache_calls_reduce_size_when_memory_available(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )
        forecaster.memory = MagicMock()

        forecaster._cleanup_cache()

        forecaster.memory.reduce_size.assert_called_once_with(
            bytes_limit=forecaster.cache_size_limit_bytes
        )

    def test_cleanup_cache_noop_without_memory(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )

        forecaster._cleanup_cache()  # must not raise

    def test_hyperparametertuning_returns_cloned_best_estimator(self):
        """Exercises the real GridSearchCV path with a minimal pipeline.

        A full preprocessor would make this as expensive as real training;
        param_grid only touches the "model" step, so a bare estimator is enough.
        """
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )
        data = DataFrame({"x": range(30)})
        y_vector = Series(range(30), dtype=float)
        pipeline = Pipeline(
            steps=[("model", HistGradientBoostingRegressor(random_state=42))]
        )

        tuned = forecaster._hyperparametertuning(data, y_vector, pipeline)

        assert isinstance(tuned, Pipeline)
        assert tuned is not pipeline


class TestForecasterPredict:
    @pytest.mark.asyncio
    async def test_predict_raises_when_not_trained(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )

        data = DataFrame({"time": [datetime.now()], "energy": [100.0]})

        with pytest.raises(ModelNotTrainedError, match="not been trained"):
            await forecaster.predict(data)

    @pytest.mark.asyncio
    async def test_predict_returns_predictions(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )

        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = [100.0, 150.0]
        forecaster.model_pipeline = mock_pipeline
        forecaster.training_completed.set()

        data = DataFrame(
            {"time": [datetime.now(), datetime.now() + timedelta(hours=1)]}
        )

        result = await forecaster.predict(data)

        assert "energy" in result.columns
        mock_pipeline.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_predict_handles_tuple_predictions(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = ([100.0, 200.0],)
        forecaster.model_pipeline = mock_pipeline
        forecaster.training_completed.set()

        data = DataFrame(
            {"time": [datetime.now(), datetime.now() + timedelta(hours=1)]}
        )

        result = await forecaster.predict(data)

        assert result["energy"].tolist() == [0.1, 0.2]


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


class TestForecasterPreparePreprocessor:
    def test_prepare_preprocessor_returns_column_transformer(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )

        columns = ["time", "clouds", "temp", "weather_id", "wind_deg"]
        preprocessor = forecaster._prepare_preprocessor(columns)

        assert isinstance(preprocessor, ColumnTransformer)


class TestForecasterPrepareModelPipeline:
    def test_prepare_model_pipeline_returns_pipeline(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )

        columns = ["time", "clouds", "temp"]
        pipeline = forecaster._prepare_model_pipeline(columns)

        assert isinstance(pipeline, Pipeline)

    def test_prepare_model_pipeline_has_steps(self):
        forecaster = Forecaster(
            ForecasterType.ENERGY, make_location(), ForecasterConfig()
        )

        columns = ["time", "clouds", "temp"]
        pipeline = forecaster._prepare_model_pipeline(columns)

        step_names = [name for name, _ in pipeline.steps]
        assert "preprocessor" in step_names
        assert "feature_selector" in step_names
        assert "model" in step_names
