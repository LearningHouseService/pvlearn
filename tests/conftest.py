import pytest


@pytest.fixture
def reference_location() -> dict:
    return {"latitude": 49.45, "longitude": 11.08, "timezone": "Europe/Berlin"}
