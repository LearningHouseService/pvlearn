import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REFERENCE_DATASET = FIXTURES_DIR / "reference_dataset.parquet"
REFERENCE_METADATA = FIXTURES_DIR / "reference_dataset.json"
BASELINE_FORECAST = FIXTURES_DIR / "baseline_forecast.parquet"
BASELINE_METADATA = FIXTURES_DIR / "baseline_forecast.json"


@pytest.fixture(scope="session")
def reference_metadata() -> dict:
    """Export metadata describing the frozen reference dataset."""
    return json.loads(REFERENCE_METADATA.read_text())


@pytest.fixture(scope="session")
def reference_location(reference_metadata: dict) -> dict:
    """Location the reference dataset was recorded at.

    SunEncoder needs this to reproduce its features, so it travels with the
    dataset rather than being hard-coded here.
    """
    return reference_metadata["location"]


@pytest.fixture(scope="session")
def reference_dataset() -> pd.DataFrame:
    """The frozen training dataset every phase is regression-tested against.

    Hourly rows of the OpenWeatherMap forecast that was valid for that hour,
    joined with the energy actually produced in it. `_time` is UTC; converting
    it to local time is what any consumer has to do explicitly.
    """
    data = pd.read_parquet(REFERENCE_DATASET)
    return data.sort_values("_time").reset_index(drop=True)


@pytest.fixture(scope="session")
def baseline_metadata() -> dict:
    """Metrics and library versions the frozen baseline was produced under."""
    return json.loads(BASELINE_METADATA.read_text())


@pytest.fixture(scope="session")
def baseline_forecast() -> pd.DataFrame:
    """Forecasts the current solaredge2mqtt code produces on the reference dataset.

    Phase 1a is accepted when the extracted pvlearn code reproduces these. Columns
    ending in `_raw` are the pipeline output, `_published` the values after
    ForecasterType.prepare_value, which clamps at zero and converts energy to kWh.
    """
    data = pd.read_parquet(BASELINE_FORECAST)
    return data.sort_values("_time").reset_index(drop=True)
