import pytest

from pvlearn.exceptions import (
    InsufficientDataError,
    ModelNotTrainedError,
    PVLearnError,
    SchemaMismatchError,
)


@pytest.mark.parametrize(
    "error_type",
    [InsufficientDataError, ModelNotTrainedError, SchemaMismatchError],
)
def test_all_pvlearn_errors_derive_from_pvlearn_error(error_type):
    assert issubclass(error_type, PVLearnError)


def test_pvlearn_error_message_is_preserved():
    error = InsufficientDataError("not enough rows")

    assert str(error) == "not enough rows"
