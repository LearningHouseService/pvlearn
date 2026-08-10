"""The canonical feature schema every brain and every model is trained against.

Provider-specific field names never reach the library: an adapter maps whatever
the weather service returns onto the names below. All weather features are
optional — a provider that has no UV index simply yields a smaller feature set,
never an error.

Changing anything here invalidates every persisted model, which is what
`FEATURE_SCHEMA_VERSION` exists for. Bump it in the same commit as the change.
"""

FEATURE_SCHEMA_VERSION = 2

#: Column holding the timezone-aware timestamp of an interval's start.
TIME_FEATURE = "time"

#: Column holding the energy produced during the interval, in Wh.
TARGET_FEATURE = "energy"

NUMERIC_FEATURES: list[str] = [
    "cloud_cover",
    "temperature",
    "apparent_temperature",
    "dew_point",
    "relative_humidity",
    "surface_pressure",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "visibility",
    "wind_speed",
    "wind_gust",
    "ghi",
    "dni",
    "dhi",
]

#: Unlike the rest, "weather_provider" is not read off the provider's own
#: payload — the adapter stamps its own name onto every row. That lets a
#: forecast that switches providers keep the old rows instead of needing a
#: retrain: the model just sees a new category value.
CATEGORICAL_FEATURES: list[str] = ["condition_code", "weather_provider"]

CYCLICAL_FEATURES: dict[str, int] = {"wind_direction": 360}
