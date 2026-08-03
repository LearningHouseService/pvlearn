import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REFERENCE_DATASET = FIXTURES_DIR / "reference_dataset.parquet"
REFERENCE_METADATA = FIXTURES_DIR / "reference_dataset.json"


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
