import pytest
from pydantic import ValidationError

from pvlearn.config import ForecasterConfig


def test_forecaster_config_defaults():
    config = ForecasterConfig()

    assert config.hyperparametertuning is False
    assert config.cachingdir is None
    assert config.cache_size_limit_mb == 512
    assert config.is_caching_enabled is False


def test_forecaster_config_caching_enabled_when_dir_set(tmp_path):
    config = ForecasterConfig(cachingdir=str(tmp_path))

    assert config.is_caching_enabled is True


def test_forecaster_config_rejects_non_positive_cache_limit():
    with pytest.raises(ValidationError):
        ForecasterConfig(cache_size_limit_mb=0)
