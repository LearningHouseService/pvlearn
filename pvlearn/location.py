from pydantic import BaseModel


class Location(BaseModel):
    """A PV plant's location, explicit and serializable.

    Passed around instead of read from process-global state (system timezone,
    environment) so a single process can serve brains in different timezones.
    """

    latitude: float
    longitude: float
    timezone: str
