"""Simulated time: an integer count of minutes since Monday 00:00.

DISCIPLINE (enforced, not conventional):
- Sim time has ONE owner (World._clock), ONE reader surface (World.clock,
  read-only property) and ONE mutation point (World.advance_clock_to, which
  guards monotonicity). Nothing an agent or NPC ever sees is derived from
  the system clock.
- The system clock exists in exactly one function below: wall_now(), for
  TELEMETRY ONLY (API latency measurement, run-dir names, log stamps).
  A test greps the codebase to keep it that way.
"""


def wall_now():
    """TELEMETRY ONLY. Never feeds sim semantics, scheduling, world state,
    or anything an agent/NPC observes. The single permitted system-clock
    call site in this codebase."""
    import time
    return time.time()

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MIN_PER_DAY = 24 * 60

WORK_START = 9 * 60        # 09:00
WORK_END = 17 * 60 + 30    # 17:30


def fmt(t):
    """545 -> 'Mon 09:05'"""
    day = DAY_NAMES[(t // MIN_PER_DAY) % 7]
    h, m = divmod(t % MIN_PER_DAY, 60)
    return "%s %02d:%02d" % (day, h, m)


def is_workday(t):
    return (t // MIN_PER_DAY) % 7 < 5


def in_working_hours(t):
    return is_workday(t) and WORK_START <= t % MIN_PER_DAY < WORK_END


def next_work_start(t):
    """Earliest work-start time strictly relevant to t (today if before 09:00, else next workday)."""
    day = t // MIN_PER_DAY
    if is_workday(t) and t % MIN_PER_DAY < WORK_START:
        return day * MIN_PER_DAY + WORK_START
    d = day + 1
    while d % 7 >= 5:
        d += 1
    return d * MIN_PER_DAY + WORK_START


def working_minutes_between(a, b):
    """Work-calendar minutes between sim-times a and b (one person)."""
    total = 0
    for day in range(a // MIN_PER_DAY, b // MIN_PER_DAY + 1):
        if day % 7 >= 5:
            continue  # weekend
        ws = day * MIN_PER_DAY + WORK_START
        we = day * MIN_PER_DAY + WORK_END
        total += max(0, min(b, we) - max(a, ws))
    return total
