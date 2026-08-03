try:
    from importlib.metadata import version

    __version__ = version("pvlearn")
except Exception:  # pragma: no cover
    __version__ = "0.0.0.unknown"
