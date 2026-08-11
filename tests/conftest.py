import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REFERENCE_DATASET = FIXTURES_DIR / "reference_dataset.parquet"
REFERENCE_METADATA = FIXTURES_DIR / "reference_dataset.json"
BASELINE_FORECAST = FIXTURES_DIR / "baseline_forecast.parquet"
BASELINE_METADATA = FIXTURES_DIR / "baseline_forecast.json"

#: OpenWeatherMap One Call field names to the canonical schema in `pvlearn.schema`.
#: The real adapter lives in the consuming application (solaredge2mqtt today,
#: learninghouse in the future); this one only exists to feed the frozen
#: reference export, which was recorded before the schema was canonical, to the
#: current code.
OWM_TO_CANONICAL: dict[str, str] = {
    "clouds": "cloud_cover",
    "temp": "temperature",
    "feels_like": "apparent_temperature",
    "dew_point": "dew_point",
    "humidity": "relative_humidity",
    "pressure": "surface_pressure",
    "rain": "precipitation",
    "pop": "precipitation_probability",
    "uvi": "uv_index",
    "visibility": "visibility",
    "wind_speed": "wind_speed",
    "wind_gust": "wind_gust",
    "wind_deg": "wind_direction",
    "weather_id": "condition_code",
}


def to_canonical(data: pd.DataFrame) -> pd.DataFrame:
    """Rename a raw OpenWeatherMap export onto the canonical feature names.

    `weather_main` and `snow` are dropped: the first is a coarse summary of
    `weather_id` and carries no own information, the second has no canonical
    field. Condition codes are passed through unmapped — translating them to
    WMO is the provider adapter's job, and the model treats the column as
    opaque categories either way.
    """
    keep = [*OWM_TO_CANONICAL, "_time", "energy"]

    canonical = data[[column for column in data.columns if column in keep]].copy()
    canonical.columns = [
        OWM_TO_CANONICAL.get(str(column), str(column)) for column in canonical.columns
    ]

    return cast(pd.DataFrame, canonical)


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
    joined with the energy actually produced in it, under the raw provider
    field names it was exported with. `_time` is UTC; converting it to local
    time is what any consumer has to do explicitly.
    """
    data = pd.read_parquet(REFERENCE_DATASET)
    return data.sort_values("_time").reset_index(drop=True)


@pytest.fixture(scope="session")
def canonical_reference_dataset(reference_dataset: pd.DataFrame) -> pd.DataFrame:
    """The reference dataset as the current schema expects to receive it."""
    return to_canonical(reference_dataset)


@pytest.fixture(scope="session")
def baseline_metadata() -> dict:
    """Metrics and library versions the frozen baseline was produced under."""
    return json.loads(BASELINE_METADATA.read_text())


@pytest.fixture(scope="session")
def baseline_forecast() -> pd.DataFrame:
    """Forecasts the original solaredge2mqtt code produces on the reference dataset.

    Columns ending in `_raw` are the pipeline output, `_published` the values
    after the original `ForecasterType.prepare_value`, which clamped at zero
    and converted energy to kWh. pvlearn publishes Wh since Phase 1b, so the
    energy column has to be scaled when comparing against it.
    """
    data = pd.read_parquet(BASELINE_FORECAST)
    return data.sort_values("_time").reset_index(drop=True)
