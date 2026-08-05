from datetime import datetime

import pytest
import sklearn

from pvlearn import __version__
from pvlearn.config import ForecasterConfig
from pvlearn.exceptions import SchemaMismatchError
from pvlearn.location import Location
from pvlearn.metadata import ModelMetadata, ModelMetrics, sklearn_minor_version
from pvlearn.schema import FEATURE_SCHEMA_VERSION


def make_location(**overrides) -> Location:
    return Location(
        **{
            "latitude": 52.52,
            "longitude": 13.405,
            "timezone": "Europe/Berlin",
            **overrides,
        }
    )


def make_config(**overrides) -> ForecasterConfig:
    return ForecasterConfig(
        **{"interval_minutes": 60, "weather_provider": "open-meteo", **overrides}
    )


def make_metadata(**overrides) -> ModelMetadata:
    metadata = ModelMetadata.create(
        location=make_location(),
        config=make_config(),
        training_rows=1440,
        selected_features=["sun__time_elevation"],
        metrics=ModelMetrics(mae=1.0, rmse=2.0, r2=0.9),
    )

    if not overrides:
        return metadata

    return metadata.model_copy(update=overrides)


def test_sklearn_minor_version_drops_the_patch_level():
    major, minor = sklearn.__version__.split(".")[:2]

    assert sklearn_minor_version() == f"{major}.{minor}"


class TestCreate:
    def test_stamps_the_current_environment(self):
        metadata = make_metadata()

        assert metadata.pvlearn_version == __version__
        assert metadata.feature_schema_version == FEATURE_SCHEMA_VERSION
        assert metadata.sklearn_version == sklearn_minor_version()
        assert metadata.weather_provider == "open-meteo"
        assert metadata.interval_minutes == 60
        assert metadata.training_rows == 1440

    def test_trained_at_defaults_to_an_aware_timestamp(self):
        assert make_metadata().trained_at.tzinfo is not None

    def test_trained_at_can_be_supplied(self):
        trained_at = datetime(2026, 8, 4, 12, 0).astimezone()

        metadata = ModelMetadata.create(
            location=make_location(),
            config=make_config(),
            training_rows=1,
            selected_features=[],
            metrics=ModelMetrics(mae=0.0, rmse=0.0, r2=0.0),
            trained_at=trained_at,
        )

        assert metadata.trained_at == trained_at


class TestRaiseOnMismatch:
    def test_accepts_the_setup_it_was_created_from(self):
        make_metadata().raise_on_mismatch(make_location(), make_config())

    def test_rejects_a_different_feature_schema(self):
        metadata = make_metadata(feature_schema_version=FEATURE_SCHEMA_VERSION + 1)

        with pytest.raises(SchemaMismatchError, match="feature_schema_version"):
            metadata.raise_on_mismatch(make_location(), make_config())

    def test_rejects_a_different_sklearn_minor_version(self):
        metadata = make_metadata(sklearn_version="0.1")

        with pytest.raises(SchemaMismatchError, match="sklearn_version"):
            metadata.raise_on_mismatch(make_location(), make_config())

    def test_rejects_a_different_provider(self):
        with pytest.raises(SchemaMismatchError, match="weather_provider"):
            make_metadata().raise_on_mismatch(
                make_location(), make_config(weather_provider="openweathermap")
            )

    def test_rejects_a_different_interval(self):
        metadata = make_metadata(interval_minutes=15)

        with pytest.raises(SchemaMismatchError, match="interval_minutes"):
            metadata.raise_on_mismatch(make_location(), make_config())

    def test_rejects_a_different_location(self):
        with pytest.raises(SchemaMismatchError, match="location"):
            make_metadata().raise_on_mismatch(
                make_location(latitude=48.14), make_config()
            )

    def test_rejects_a_different_timezone_at_the_same_coordinates(self):
        """Same spot, wrong clock: every time feature would be shifted."""
        with pytest.raises(SchemaMismatchError, match="location"):
            make_metadata().raise_on_mismatch(
                make_location(timezone="UTC"), make_config()
            )

    def test_reports_every_mismatch_at_once(self):
        metadata = make_metadata(interval_minutes=15, sklearn_version="0.1")

        with pytest.raises(SchemaMismatchError) as error:
            metadata.raise_on_mismatch(
                make_location(), make_config(weather_provider="openweathermap")
            )

        message = str(error.value)
        assert "interval_minutes" in message
        assert "sklearn_version" in message
        assert "weather_provider" in message

    def test_ignores_the_pvlearn_version(self):
        """Not every release changes how features are built."""
        metadata = make_metadata(pvlearn_version="0.0.1")

        metadata.raise_on_mismatch(make_location(), make_config())


class TestSerialization:
    def test_round_trips_through_json(self):
        metadata = make_metadata()

        restored = ModelMetadata.model_validate_json(metadata.model_dump_json())

        assert restored == metadata
