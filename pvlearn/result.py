from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import BaseModel


@dataclass(frozen=True)
class _Period:
    """One forecast interval seen both ways.

    `local` answers which calendar day it belongs to, `instant` answers what
    came before or after it. On a DST boundary those two questions have
    different answers, which is why both are kept.
    """

    local: datetime
    instant: datetime

    @classmethod
    def of(cls, period: datetime, zone: ZoneInfo) -> "_Period":
        if period.tzinfo is None:
            period = period.replace(tzinfo=zone)

        return cls(local=period.astimezone(zone), instant=period.astimezone(UTC))


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
    Below 60 minutes per interval the hourly figures sum every slot that falls
    into the hour.

    `timezone` is what "today" means — the plant's, never the process's. Keys
    may arrive in any zone (UTC is the safe choice, since a local-time key
    cannot distinguish the two 02:00s of a DST fallback night); naive keys are
    read as wall clock in `timezone`.

    Days are calendar days in that zone, so a DST fallback day is 25 hours
    long. The hourly figures instead cover 60 real minutes from the start of
    the current clock hour: on a fallback night that is the first 02:00 and
    then the second, never both at once, and across a spring-forward gap the
    next hour is the one that actually follows.

    `interval_minutes` does not drive any of that — the timestamps do. It is
    carried because the interval belongs in the forecast response and a
    consumer should not have to infer it from key spacing.
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
        hour_start = self._instant(self._hour_start(now))
        return sum(
            energy
            for period, energy in self._periods()
            if period.local.date() == now.date() and period.instant >= hour_start
        )

    @property
    def energy_current_hour(self) -> int:
        return self._sum_hour(self._instant(self._hour_start(self._now_local())))

    @property
    def energy_next_hour(self) -> int:
        # An hour later in real time, not on the wall clock: adding to the
        # local timestamp would land in the non-existent hour of a
        # spring-forward night and match nothing.
        return self._sum_hour(
            self._instant(self._hour_start(self._now_local())) + timedelta(hours=1)
        )

    @property
    def energy_tomorrow(self) -> int:
        return self._sum_day(self._now_local().date() + timedelta(days=1))

    def _sum_day(self, day: date) -> int:
        return sum(
            energy for period, energy in self._periods() if period.local.date() == day
        )

    def _sum_hour(self, hour_start: datetime) -> int:
        hour_end = hour_start + timedelta(hours=1)
        return sum(
            energy
            for period, energy in self._periods()
            if hour_start <= period.instant < hour_end
        )

    def _periods(self) -> list[tuple["_Period", int]]:
        zone = ZoneInfo(self.timezone)
        return [
            (_Period.of(period, zone), energy)
            for period, energy in self.energy_period.items()
        ]

    @staticmethod
    def _hour_start(moment: datetime) -> datetime:
        return moment.replace(minute=0, second=0, microsecond=0)

    @staticmethod
    def _instant(moment: datetime) -> datetime:
        return moment.astimezone(UTC)

    def _now_local(self) -> datetime:
        return self._now().astimezone(ZoneInfo(self.timezone))

    @staticmethod
    def _now() -> datetime:
        return datetime.now().astimezone()
