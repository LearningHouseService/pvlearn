class PVLearnError(Exception):
    """Base class for all pvlearn errors."""


class InsufficientDataError(PVLearnError):
    """Raised when there is not enough data to train or predict."""


class ModelNotTrainedError(PVLearnError):
    """Raised when a prediction is requested from an untrained model."""


class SchemaMismatchError(PVLearnError):
    """Raised when persisted model metadata does not match the current config."""
