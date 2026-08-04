from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import BaseModel


class ForecastResult(BaseModel):
    """The energy-period aggregation logic, without any HA/MQTT decoration.

    solaredge2mqtt's `Forecast(Component)` inherits from this and adds the
    battery-charge fields and Home Assistant discovery decoration — neither of
    which belongs in an I/O-free library. This class exists so the aggregation
    logic is written once, since every consumer (service, HA integration,
    solaredge2mqtt) needs the same today/current hour/next hour/tomorrow
    breakdown.

    Aggregation works on the timestamps, not on positions: a row is not
    implicitly an hour and the series does not implicitly start at midnight.
    With `interval_minutes` below 60 the hourly figures sum the slots that fall
    into the respective clock hour.

    `timezone` is what "today" means — the plant's, never the process's. Keys
    may arrive in any zone (UTC is the safe choice, since a local-time key
    cannot distinguish the two 02:00s of a DST fallback night); naive keys are
    read as wall clock in `timezone`.
    """

    interval_minutes: int
    timezone: str
    energy_period: dict[datetime, int]

    @property
    def energy_today(self) -> int:
        return self._sum_day(self._now_local().date())

    @property
    def energy_today_remaining(self) -> int:
        now = self._now_local()
        hour_start = self._hour_start(now)
        return sum(
            energy
            for period, energy in self._local_periods()
            if period.date() == now.date() and period >= hour_start
        )

    @property
    def energy_current_hour(self) -> int:
        return self._sum_hour(self._hour_start(self._now_local()))

    @property
    def energy_next_hour(self) -> int:
        return self._sum_hour(self._hour_start(self._now_local()) + timedelta(hours=1))

    @property
    def energy_tomorrow(self) -> int:
        return self._sum_day(self._now_local().date() + timedelta(days=1))

    def _sum_day(self, day: date) -> int:
        return sum(
            energy for period, energy in self._local_periods() if period.date() == day
        )

    def _sum_hour(self, hour_start: datetime) -> int:
        hour_end = hour_start + timedelta(hours=1)
        return sum(
            energy
            for period, energy in self._local_periods()
            if hour_start <= period < hour_end
        )

    def _local_periods(self) -> list[tuple[datetime, int]]:
        return [
            (self._to_local(period), energy)
            for period, energy in self.energy_period.items()
        ]

    def _to_local(self, period: datetime) -> datetime:
        zone = ZoneInfo(self.timezone)

        if period.tzinfo is None:
            return period.replace(tzinfo=zone)

        return period.astimezone(zone)

    @staticmethod
    def _hour_start(moment: datetime) -> datetime:
        return moment.replace(minute=0, second=0, microsecond=0)

    def _now_local(self) -> datetime:
        return self._now().astimezone(ZoneInfo(self.timezone))

    @staticmethod
    def _now() -> datetime:
        return datetime.now().astimezone()
