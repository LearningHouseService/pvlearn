from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pvlearn")
except PackageNotFoundError:  # pragma: no cover - only hit when not installed
    __version__ = "0.0.0.unknown"
