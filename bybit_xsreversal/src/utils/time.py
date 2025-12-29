from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def parse_hhmm(s: str) -> time:
    parts = s.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time: {s}")
    hh = int(parts[0])
    mm = int(parts[1])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"Invalid HH:MM time: {s}")
    return time(hour=hh, minute=mm, tzinfo=UTC)


def utc_day_start(dt: datetime) -> datetime:
    dt = dt.astimezone(UTC)
    return datetime(dt.year, dt.month, dt.day, tzinfo=UTC)


def last_complete_daily_close(now: datetime, close_delay_seconds: int) -> datetime:
    """
    Returns the timestamp of the most recent *completed* UTC daily candle close.

    Bybit daily candle boundaries are UTC. A candle for day D closes at D+1 00:00:00 UTC.
    If we're too close to midnight (within delay), we still consider yesterday's candle last complete.
    """
    now = now.astimezone(UTC)
    today_start = utc_day_start(now)
    midnight = today_start
    delay = timedelta(seconds=max(0, close_delay_seconds))
    if now < midnight + delay:
        # still within safety delay after 00:00:00 UTC
        return midnight - timedelta(days=1)
    return midnight


def next_run_time(now: datetime, run_hhmm_utc: str) -> datetime:
    t = parse_hhmm(run_hhmm_utc)
    now = now.astimezone(UTC)
    today = utc_day_start(now)
    candidate = datetime.combine(today.date(), t, tzinfo=UTC)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


@dataclass(frozen=True)
class DateRange:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("DateRange must be timezone-aware.")
        if self.end < self.start:
            raise ValueError("DateRange.end must be >= start.")


