from datetime import datetime

from pydantic import BaseModel


class ForecastResult(BaseModel):
    """The energy-period aggregation logic, without any HA/MQTT decoration.

    solaredge2mqtt's `Forecast(Component)` inherits from this and adds the
    battery-charge fields, Home Assistant discovery decoration, and
    `power_period` — none of which belong in an I/O-free library. This class
    exists so the aggregation logic is written once, since every consumer
    (service, HA integration, solaredge2mqtt) needs the same today/current
    hour/next hour/tomorrow breakdown.
    """

    energy_period: dict[datetime, int]

    @property
    def energy_today(self) -> int:
        return sum(self._energy_today)

    @property
    def energy_today_remaining(self) -> int:
        return sum(self._energy_today[self._current_hour() :])

    @property
    def energy_current_hour(self) -> int:
        return self._energy_today[self._current_hour()]

    @property
    def energy_next_hour(self) -> int:
        if self._current_hour() == 23:
            return self._energy_tomorrow[0]

        return self._energy_today[self._current_hour() + 1]

    @property
    def energy_tomorrow(self) -> int:
        return sum(self._energy_tomorrow)

    @property
    def _energy_today(self) -> list[int]:
        return [*self.energy_period.values()][:24]

    @property
    def _energy_tomorrow(self) -> list[int]:
        return [*self.energy_period.values()][24:]

    @staticmethod
    def _current_hour() -> int:
        return datetime.now().hour
