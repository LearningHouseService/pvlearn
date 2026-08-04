import json
import logging
import time
from asyncio import Event
from math import ceil
from pathlib import Path
from typing import cast

from joblib import Memory, dump, load
from numpy import percentile
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
from pvlearn.metadata import ModelMetadata, ModelMetrics
from pvlearn.schema import (
    CATEGORICAL_FEATURES,
    CYCLICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_FEATURE,
    TIME_FEATURE,
)

logger = logging.getLogger(__name__)

MODEL_FILENAME = "trained.pkl"
METADATA_FILENAME = "metadata.json"

MINIMUM_TRAINING_HOURS = 60
MINUTES_PER_HOUR = 60


class Forecaster:
    """Trains and predicts the energy produced per interval, in Wh.

    There is exactly one model. The former power model trained on identical
    features and, at hourly resolution, produced the same number in a different
    unit — on the reference dataset the two were less than a percent apart
    (chapter 3.3 of the Umsetzungsplan).
    """

    #: Splits used to carve a holdout out of the training data for the metrics
    #: in the model metadata. The last split's test fold is the most recent
    #: data, which is what a forecast is judged on.
    METRICS_SPLITS = 5

    def __init__(
        self,
        location: Location,
        config: ForecasterConfig,
    ) -> None:
        self.location = location
        self.config = config
        self.enable_hyperparameter_tuning = config.hyperparametertuning
        self.interval_minutes = config.interval_minutes
        self.model_pipeline: Pipeline | None = None
        self.metadata: ModelMetadata | None = None
        self.training_completed: Event = Event()
        self.cache_size_limit_bytes = config.cache_size_limit_mb * 1024 * 1024

        self.memory: Memory | None = (
            Memory(config.cachingdir, verbose=0) if config.is_caching_enabled else None
        )

    @property
    def minimum_training_rows(self) -> int:
        return ceil(MINIMUM_TRAINING_HOURS * MINUTES_PER_HOUR / self.interval_minutes)

    def train(self, data: DataFrame) -> None:
        data_count = len(data)
        logger.info(
            "Training energy model with %d intervals of %d minutes",
            data_count,
            self.interval_minutes,
        )

        if data_count < self.minimum_training_rows:
            raise InsufficientDataError(
                f"Forecast needs at least {MINIMUM_TRAINING_HOURS} hours of data "
                f"({self.minimum_training_rows} intervals) to start training",
            )

        self.training_completed.clear()
        start_time = time.time()
        y_vector = cast(Series, data[TARGET_FEATURE])

        pipeline = self._prepare_model_pipeline(data.columns.to_list())

        if self.enable_hyperparameter_tuning:
            pipeline = self._hyperparametertuning(data, y_vector, pipeline)

        metrics = self._evaluate(data, y_vector, pipeline)

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

        self.metadata = ModelMetadata.create(
            location=self.location,
            config=self.config,
            training_rows=data_count,
            selected_features=list(selected_features),
            metrics=metrics,
        )

        logger.info(
            "Holdout metrics: MAE %.2f Wh, RMSE %.2f Wh, R² %.4f",
            metrics.mae,
            metrics.rmse,
            metrics.r2,
        )
        logger.info("Training execution time: %.2f seconds", execution_time)

    def _evaluate(
        self, data: DataFrame, y_vector: Series, pipeline: Pipeline
    ) -> ModelMetrics:
        """Score a copy of the pipeline on the most recent time-series split.

        A copy, because the model that is kept has to see all of the data —
        including the holdout, which is the freshest and therefore the most
        relevant part of it.
        """
        splitter = TimeSeriesSplit(n_splits=self.METRICS_SPLITS)
        train_index, test_index = list(splitter.split(data))[-1]

        evaluation_pipeline = cast(Pipeline, clone(pipeline))
        evaluation_pipeline.fit(data.iloc[train_index], y_vector.iloc[train_index])

        predicted = self._as_published(
            evaluation_pipeline.predict(data.iloc[test_index])
        )
        actual = y_vector.iloc[test_index].to_numpy()

        return ModelMetrics(
            mae=float(mean_absolute_error(actual, predicted)),
            rmse=float(mean_squared_error(actual, predicted) ** 0.5),
            r2=float(r2_score(actual, predicted)),
        )

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
                    CyclicalEncoder(**CYCLICAL_FEATURES),
                    self._extract_used_columns(CYCLICAL_FEATURES, x_vector_columns),
                ),
                (
                    "num",
                    "passthrough",
                    self._extract_used_columns(NUMERIC_FEATURES, x_vector_columns),
                ),
                (
                    "time",
                    TimeEncoder(),
                    [TIME_FEATURE],
                ),
                (
                    "sun",
                    SunEncoder(
                        self.location.latitude,
                        self.location.longitude,
                        self.location.timezone,
                        self.interval_minutes,
                    ),
                    [TIME_FEATURE],
                ),
                (
                    "cat",
                    CategoricalEncoder(),
                    self._extract_used_columns(CATEGORICAL_FEATURES, x_vector_columns),
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

    @staticmethod
    def prepare_value(value: float | int) -> int:
        """Round a raw prediction to the published Wh value, clamped at zero.

        A negative production is not a forecast, it is the model extrapolating
        past the edge of its training data at night.
        """
        if value <= 0:
            return 0

        return int(round(value))

    def _as_published(self, predictions: NDArray) -> NDArray:
        return Series(predictions).apply(self.prepare_value).to_numpy()

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

        data_for_prediction[TARGET_FEATURE] = Series(
            predictions, index=data_for_prediction.index
        ).apply(self.prepare_value)

        return data_for_prediction

    def save(self, directory: Path) -> None:
        """Persist model and metadata into an existing directory.

        The only filesystem access in the library, and only ever below a path
        the caller passed in.
        """
        if self.model_pipeline is None or self.metadata is None:
            raise ModelNotTrainedError("There is no trained model to save.")

        directory.mkdir(parents=True, exist_ok=True)
        dump(self.model_pipeline, directory / MODEL_FILENAME)
        (directory / METADATA_FILENAME).write_text(
            self.metadata.model_dump_json(indent=2)
        )

        logger.info("Saved trained model to %s", directory)

    @classmethod
    def load(
        cls, directory: Path, location: Location, config: ForecasterConfig
    ) -> "Forecaster":
        """Restore a model, or refuse to.

        Raises `ModelNotTrainedError` when there is nothing to load and
        `SchemaMismatchError` when what is there was trained under a different
        setup. Both mean the same thing to a caller — train from scratch — but
        only one of them is worth a warning in the log.
        """
        model_path = directory / MODEL_FILENAME
        metadata_path = directory / METADATA_FILENAME

        if not model_path.is_file() or not metadata_path.is_file():
            raise ModelNotTrainedError(f"No persisted model found in {directory}")

        metadata = ModelMetadata.model_validate(
            json.loads(metadata_path.read_text()),
        )
        metadata.raise_on_mismatch(location, config)

        forecaster = cls(location, config)
        forecaster.model_pipeline = cast(Pipeline, load(model_path))
        forecaster.metadata = metadata
        forecaster.training_completed.set()

        logger.info(
            "Loaded model trained at %s from %s", metadata.trained_at, directory
        )

        return forecaster


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
