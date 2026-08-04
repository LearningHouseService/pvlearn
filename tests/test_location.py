import pytest
from pydantic import ValidationError

from pvlearn.location import Location


def test_location_requires_all_fields():
    location = Location(latitude=52.52, longitude=13.405, timezone="Europe/Berlin")

    assert location.latitude == 52.52
    assert location.longitude == 13.405
    assert location.timezone == "Europe/Berlin"


def test_location_rejects_missing_timezone():
    with pytest.raises(ValidationError):
        Location(latitude=52.52, longitude=13.405)  # pyright: ignore[reportCallIssue]
