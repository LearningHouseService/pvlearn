import pytest
from pydantic import ValidationError

from pvlearn.config import ForecasterConfig


def make_config(**overrides) -> ForecasterConfig:
    return ForecasterConfig(**{"interval_minutes": 60, **overrides})


def test_forecaster_config_defaults():
    config = make_config()

    assert config.hyperparametertuning is False
    assert config.cachingdir is None
    assert config.cache_size_limit_mb == 512
    assert config.is_caching_enabled is False


def test_forecaster_config_caching_enabled_when_dir_set(tmp_path):
    config = make_config(cachingdir=str(tmp_path))

    assert config.is_caching_enabled is True


def test_forecaster_config_rejects_non_positive_cache_limit():
    with pytest.raises(ValidationError):
        make_config(cache_size_limit_mb=0)


def test_forecaster_config_rejects_unsupported_interval():
    """Finer intervals need measurement data that does not exist yet (3.5)."""
    with pytest.raises(ValidationError, match="Unsupported forecast interval"):
        make_config(interval_minutes=15)
