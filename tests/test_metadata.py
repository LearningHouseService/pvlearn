from datetime import datetime

import pytest

from pvlearn import __version__
from pvlearn.config import ForecasterConfig
from pvlearn.exceptions import SchemaMismatchError
from pvlearn.location import Location
from pvlearn.metadata import ModelMetadata, ModelMetrics, release_version


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
    return ForecasterConfig(**{"interval_minutes": 60, **overrides})


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


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.3.0", "0.3.0"),
        ("0.3.0.post3+g527ccef", "0.3.0"),
        ("0.3.0.post3.dev1+g527ccef.d20260810", "0.3.0"),
        ("1.0.0rc1", "1.0.0"),
        ("1.0.0.dev0", "1.0.0"),
        ("2", "2"),
    ],
)
def test_release_version_keeps_only_the_release_segment(version: str, expected: str):
    assert release_version(version) == expected


def test_release_version_rejects_a_version_without_a_release_segment():
    with pytest.raises(ValueError, match="does not start with a release segment"):
        release_version("unknown")


class TestCreate:
    def test_stamps_the_current_environment(self):
        metadata = make_metadata()

        assert metadata.pvlearn_version == __version__
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

    def test_rejects_a_model_from_another_release(self):
        metadata = make_metadata(pvlearn_version="0.0.1")

        with pytest.raises(SchemaMismatchError, match="pvlearn_version"):
            metadata.raise_on_mismatch(make_location(), make_config())

    def test_accepts_a_dev_build_of_the_same_release(self):
        """Commits between two releases must not invalidate a model."""
        metadata = make_metadata(
            pvlearn_version=f"{release_version()}.post3.dev1+g527ccef"
        )

        metadata.raise_on_mismatch(make_location(), make_config())

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
        metadata = make_metadata(interval_minutes=15, pvlearn_version="0.0.1")

        with pytest.raises(SchemaMismatchError) as error:
            metadata.raise_on_mismatch(make_location(), make_config())

        message = str(error.value)
        assert "interval_minutes" in message
        assert "pvlearn_version" in message


class TestSerialization:
    def test_round_trips_through_json(self):
        metadata = make_metadata()

        restored = ModelMetadata.model_validate_json(metadata.model_dump_json())

        assert restored == metadata

    def test_a_sidecar_from_before_adr_0003_is_a_mismatch_not_unreadable(self):
        """The dropped version fields must degrade into a precise reason."""
        sidecar = make_metadata(pvlearn_version="0.3.0").model_dump(mode="json")
        sidecar |= {
            "feature_schema_version": 2,
            "pipeline_version": 2,
            "sklearn_version": "1.9",
        }

        restored = ModelMetadata.model_validate(sidecar)

        with pytest.raises(SchemaMismatchError, match="pvlearn_version is '0.3.0'"):
            restored.raise_on_mismatch(make_location(), make_config())
