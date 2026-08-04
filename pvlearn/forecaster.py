import logging
import time
from asyncio import Event
from enum import Enum
from typing import cast

from joblib import Memory
from numpy import percentile
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline

from pvlearn.config import ForecasterConfig
from pvlearn.encoders import (
    CategoricalEncoder,
    CyclicalEncoder,
    SunEncoder,
    TimeEncoder,
)
from pvlearn.exceptions import InsufficientDataError, ModelNotTrainedError
from pvlearn.location import Location

logger = logging.getLogger(__name__)


class ForecasterType(Enum):
    ENERGY = "energy"
    POWER = "power"

    @property
    def target_column(self) -> str:
        return self.value

    def prepare_value(self, value: float | int) -> float | int:
        if value <= 0:
            prepared = 0
        elif self.target_column == "energy":
            prepared = round(value / 1000, 3)
        else:
            prepared = int(round(value))

        return prepared


class Forecaster:
    NUMERIC_FEATURES: list[str] = [
        "clouds",
        "dew_point",
        "feels_like",
        "humidity",
        "pop",
        "pressure",
        "rain",
        "temp",
        "uvi",
        "visibility",
        "wind_speed",
        "wind_gust",
    ]
    CATEGORICAL_FEATURES: list[str] = ["weather_id", "weather_main"]
    CYCLICAL_FEATURES: dict[str, int] = {
        "wind_deg": 360,
    }
    TARGET_FEATURES: dict[str, str] = {
        ForecasterType.ENERGY.target_column: "energy",
        ForecasterType.POWER.target_column: "power",
    }

    def __init__(
        self,
        typed: ForecasterType,
        location: Location,
        config: ForecasterConfig,
    ) -> None:
        self.typed: ForecasterType = typed
        self.location = location
        self.enable_hyperparameter_tuning = config.hyperparametertuning
        self.model_pipeline: Pipeline | None = None
        self.training_completed: Event = Event()
        self.cache_size_limit_bytes = config.cache_size_limit_mb * 1024 * 1024

        self.memory: Memory | None = (
            Memory(config.cachingdir, verbose=0) if config.is_caching_enabled else None
        )

    def train(self, data: DataFrame) -> None:
        data_count = len(data)
        logger.info(
            "Training model %s with %d hours of data points", self.typed, data_count
        )

        if data_count < 60:
            raise InsufficientDataError(
                "Forecast needs at least 60 hours of data at least to start training",
            )

        self.training_completed.clear()
        start_time = time.time()
        y_vector = cast(Series, data[self.typed.target_column])

        pipeline = self._prepare_model_pipeline(data.columns.to_list())

        if self.enable_hyperparameter_tuning:
            self.model_pipeline = self._hyperparametertuning(data, y_vector, pipeline)
        else:
            self.model_pipeline = pipeline

        self.model_pipeline.fit(data, y_vector)
        self._cleanup_cache()

        execution_time = time.time() - start_time
        self.training_completed.set()

        transformed_features = self.model_pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
        logger.debug(
            "Transformed features (%d): %s",
            len(transformed_features),
            ", ".join(transformed_features),
        )

        selected_features = self.model_pipeline.named_steps[
            "feature_selector"
        ].important_features_
        logger.info(
            "Selected features (%d): %s",
            len(selected_features),
            ", ".join(selected_features),
        )

        logger.info("Training execution time: %.2f seconds", execution_time)

    def _prepare_model_pipeline(
        self, x_vector_columns: list[str], n_repeats: int = 10
    ) -> Pipeline:
        base_estimator = HistGradientBoostingRegressor(
            random_state=42,
            categorical_features="from_dtype",
            learning_rate=0.1,
        )

        return Pipeline(
            steps=[
                ("preprocessor", self._prepare_preprocessor(x_vector_columns)),
                (
                    "feature_selector",
                    PFISelector(estimator=clone(base_estimator), n_repeats=n_repeats),
                ),
                ("model", clone(base_estimator)),
            ],
            memory=self.memory,
        )

    def _prepare_preprocessor(self, x_vector_columns: list[str]) -> ColumnTransformer:
        ct = ColumnTransformer(
            transformers=[
                (
                    "cyc",
                    CyclicalEncoder(**self.CYCLICAL_FEATURES),
                    self._extract_used_columns(
                        self.CYCLICAL_FEATURES, x_vector_columns
                    ),
                ),
                (
                    "num",
                    "passthrough",
                    self._extract_used_columns(self.NUMERIC_FEATURES, x_vector_columns),
                ),
                (
                    "time",
                    TimeEncoder(),
                    ["time"],
                ),
                (
                    "sun",
                    SunEncoder(
                        self.location.latitude,
                        self.location.longitude,
                        self.location.timezone,
                    ),
                    ["time"],
                ),
                (
                    "cat",
                    CategoricalEncoder(),
                    self._extract_used_columns(
                        self.CATEGORICAL_FEATURES, x_vector_columns
                    ),
                ),
            ],
            remainder="drop",
        )
        ct.set_output(transform="pandas")
        return ct

    def _hyperparametertuning(
        self,
        data: DataFrame,
        y_vector: Series,
        pipeline: Pipeline,
    ) -> Pipeline:
        param_grid = {
            "model__max_iter": [100, 200, 300],
            "model__max_depth": [None, 5, 10],
            "model__learning_rate": [0.01, 0.1],
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=TimeSeriesSplit(n_splits=2),
            scoring="neg_mean_squared_error",
        )
        grid_search.fit(data, y_vector)

        logger.info("Training with best parameters: %s", grid_search.best_params_)
        logger.info("Training with best score: %s", grid_search.best_score_)

        return cast(Pipeline, clone(grid_search.best_estimator_))

    def _cleanup_cache(self) -> None:
        if self.memory is None:
            return

        self.memory.reduce_size(bytes_limit=self.cache_size_limit_bytes)

    @staticmethod
    def _extract_used_columns(
        typed_features: list[str] | dict[str, int], x_vector_columns: list[str]
    ) -> list[str]:
        return [col for col in typed_features if col in x_vector_columns]

    @property
    def is_trained(self) -> bool:
        return self.model_pipeline is not None

    async def predict(self, new_data: DataFrame) -> DataFrame:
        if self.model_pipeline is None:
            raise ModelNotTrainedError("The model has not been trained yet.")

        data_for_prediction = new_data.copy()

        await self.training_completed.wait()

        predictions = self.model_pipeline.predict(data_for_prediction)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        data_for_prediction[self.typed.target_column] = predictions
        data_for_prediction[self.typed.target_column] = data_for_prediction[
            self.typed.target_column
        ].apply(self.typed.prepare_value)

        return data_for_prediction


class PFISelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_repeats=10):
        self.estimator = estimator
        self.n_repeats = n_repeats
        self.important_features_: list[str] | None = None
        self.important_indices_: list[bool] | None = None

    def fit(self, x_vector: DataFrame, y_vector=None) -> "PFISelector":
        x_train, x_test, y_train, y_test = train_test_split(
            x_vector, y_vector, test_size=0.1, random_state=42
        )
        self.estimator_ = self.estimator.fit(x_train, y_train)
        results = permutation_importance(
            self.estimator_,
            x_test,
            y_test,
            n_repeats=self.n_repeats,
            random_state=42,
            n_jobs=-1,
        )

        self.feature_importances_ = cast(
            NDArray,
            results["importances_mean"],
        )

        threshold_value = percentile(self.feature_importances_, 75)

        selected = self.feature_importances_ > threshold_value
        if not selected.any():
            selected = self.feature_importances_ >= threshold_value

        important_indices = cast(list[bool], selected.tolist())
        self.important_indices_ = important_indices
        self.important_features_ = [
            col
            for col, keep in zip(x_vector.columns.to_list(), important_indices)
            if keep
        ]
        return self

    def transform(self, x_vector: DataFrame) -> DataFrame:
        if self.important_features_ is None:
            raise RuntimeError("PFISelector has not been fitted yet.")
        return cast(DataFrame, x_vector.loc[:, self.important_features_])

    def get_support(self, *_) -> list[bool]:
        if self.important_indices_ is None:
            raise RuntimeError("PFISelector has not been fitted yet.")
        return self.important_indices_
