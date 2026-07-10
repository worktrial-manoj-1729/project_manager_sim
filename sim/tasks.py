"""Project/task world model: analytic work-calendar scheduling.

Task progress is a PURE FUNCTION of (task list, logged mutations, sim time) —
no ticks. Tasks arrive three ways: seeded in the scenario (arrival = sim
start), injected on a schedule by external beats (OOD arrivals), or added by
the agent for tracking. Every arrival is a logged `task_added` mutation, so
rewind/replay reconstruct the board at any time.

Scheduling semantics (deterministic, PREEMPTIVE, work-conserving):
- Each assignee works one task at a time, only during working hours
  (Mon-Fri 09:00-17:30).
- A task is READY once it has arrived, somebody holds it
  (max(arrival, assigned_at)), and its blockers are done.
- At every instant each person works their highest-priority ready task
  (urgent -> priority -> creation order, with the PM's timestamped
  `order_events` applied forward-only). When that changes — a higher-priority
  task arrives or unblocks, or a reprioritization fires — they SWITCH; the
  paused task keeps its accrued work and resumes exactly where it stopped.
- Tasks without an assignee or effort are "tracking" items: visible on the
  board, never scheduled.
"""

from .sim_time import MIN_PER_DAY, WORK_END, WORK_START


def _day_segments(day, busy):
    """Available working segments on one day, minus busy (meeting) intervals."""
    if day % 7 >= 5:
        return []
    segs = [(day * MIN_PER_DAY + WORK_START, day * MIN_PER_DAY + WORK_END)]
    for bs, be in busy:
        nxt = []
        for s, e in segs:
            if be <= s or bs >= e:
                nxt.append((s, e))
                continue
            if bs > s:
                nxt.append((s, bs))
            if be < e:
                nxt.append((be, e))
        segs = nxt
    return segs


def work_minutes_between(a, b, busy=()):
    """Available working minutes (Mon-Fri 09:00-17:30, minus busy) in [a, b)."""
    if b <= a:
        return 0
    total = 0
    day = a // MIN_PER_DAY
    while day * MIN_PER_DAY < b:
        for s, e in _day_segments(day, busy):
            total += max(0, min(b, e) - max(a, s))
        day += 1
    return total


def add_work_minutes(t, minutes, busy=()):
    """Earliest time u >= t with work_minutes_between(t, u, busy) == minutes."""
    remaining = minutes
    cur = t
    day = cur // MIN_PER_DAY
    while True:
        for s, e in _day_segments(day, busy):
            start = max(cur, s)
            if start < e:
                if remaining <= e - start:
                    return start + remaining
                remaining -= e - start
        day += 1
        cur = day * MIN_PER_DAY


_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def _order_at(t, clock):
    """The scheduling order (urgent-rank, priority-rank) in effect for task t
    at time `clock`. Authored `urgent`/`priority` are the defaults; the PM's
    reprioritizations are TIMESTAMPED events (`order_events`, appended by the
    reprioritize tool) applied FORWARD-ONLY — an event at time T changes the
    order only for clock >= T, never retroactively. Scoring weight always comes
    from the authored `priority`, so the PM reorders work but cannot relabel a
    task to mint value."""
    urgent, prio = t.get("urgent"), t.get("priority")
    for ev in t.get("order_events") or []:
        if ev.get("at", 0) <= clock:
            if "order_urgent" in ev:
                urgent = ev["order_urgent"]
            if "order_priority" in ev:
                prio = ev["order_priority"]
    return (0 if urgent else 1, _PRIORITY_RANK.get(prio, 2))


def _earliest(t, sim_start):
    """Work can't start before the task EXISTS (arrival) or before somebody
    actually HOLDS it (assigned_at, stamped by assignment mutations). Without
    the latter, assignment is retroactive — picking a task up at 16:00 would
    be credited work since its 11:30 arrival, and acting early earns nothing."""
    return max(t.get("arrival", sim_start), t.get("assigned_at") or 0)


def _primary(t):
    return t["assignees"][0] if t.get("assignees") else None


def _schedulable(t):
    # Agent-CREATED tasks are tracking-only: they never consume worker
    # capacity. Symmetric with scoring (agent tasks carry zero weight) — an
    # agent can neither mint value NOR destroy it by inventing fake work that
    # eats an engineer's week. Real work reaches the board as seed tasks or
    # FILED external tickets (source stays "external"); only invented
    # source=="agent" items are excluded.
    return (t.get("assignees") and t.get("effort_hours") is not None
            and t.get("source") != "agent")


def _task_multiplier(t, skills):
    """Speed factor for task t under its CURRENT primary assignee: geometric
    mean of that person's skill factors over the task's matching tags (frozen
    config; default 1.0 when either side is absent). >1 = the specialist works
    it faster. HIDDEN from the PM — inferred only from how fast work actually
    completes. So `m` calendar-minutes of work = `m` effort-minutes done."""
    if not skills:
        return 1.0
    sk = skills.get(_primary(t)) or {}
    tags = [tag for tag in (t.get("tags") or []) if tag in sk]
    if not tags:
        return 1.0
    prod = 1.0
    for tag in tags:
        prod *= sk[tag]
    return prod ** (1.0 / len(tags))


def compute_schedule(tasks, sim_start, busy_by_assignee=None, skills=None):
    """Fully PREEMPTIVE, event-sourced scheduler. At every instant each person
    works their highest-priority READY task; the moment that changes — a
    higher-priority task arrives or unblocks, or the PM reprioritizes at a
    timestamp — they SWITCH, and the paused task keeps its accrued work and
    resumes later from exactly where it stopped (work is conserved).

    Returns, per schedulable task:
        {start, done_at, segments: [(seg_start, seg_end, person), ...]}
    where progress is the FOLD of the segments — so `task_view` can ask for
    true progress at any past/future instant. Deterministic and replay-safe:
    the only inputs are the task list (incl. timestamped `order_events`),
    sim_start, and busy intervals.

    busy_by_assignee: {person: [(start, end), ...]} — meetings/interruptions
    that consume that person's working time.
    """
    busy_by_assignee = busy_by_assignee or {}
    creation = {t["id"]: i for i, t in enumerate(tasks)}
    sched = {t["id"]: t for t in tasks if _schedulable(t)}
    persons = {}
    for tid, t in sched.items():
        persons.setdefault(_primary(t), []).append(tid)

    # remaining is CALENDAR-minutes = effort-minutes / skill-multiplier, so a
    # faster specialist finishes sooner. mult is kept per task so progress
    # readouts can convert calendar-work back to true effort-hours.
    mult = {tid: _task_multiplier(sched[tid], skills) for tid in sched}
    remaining = {tid: max(0.0, round((sched[tid]["effort_hours"]
                                      - sched[tid].get("done_hours", 0))
                                     * 60.0 / mult[tid]))
                 for tid in sched}
    segments = {tid: [] for tid in sched}
    done_at, started = {}, {}
    # tasks already complete at seed (done_hours == effort) are done the moment
    # they exist — set done_at up front so blockers referencing them clear.
    for tid in sched:
        if remaining[tid] <= 0:
            done_at[tid] = _earliest(sched[tid], sim_start)

    # decision times we know statically: when each task becomes holdable
    # (arrival/assignment) and every reprioritization timestamp. Completions
    # are discovered dynamically as the sim advances.
    statics = set()
    for tid, t in sched.items():
        statics.add(_earliest(t, sim_start))
        for ev in t.get("order_events") or []:
            statics.add(ev.get("at", sim_start))

    def ready(tid, clock):
        t = sched[tid]
        if remaining[tid] <= 0 or _earliest(t, sim_start) > clock:
            return False
        for b in t.get("blocked_by", []):
            if b in sched and (b not in done_at or done_at[b] > clock):
                return False  # a scheduled blocker isn't finished by `clock`
        return True

    clock, INF, guard = sim_start, float("inf"), 0
    while any(r > 0 for r in remaining.values()) and guard < 100000:
        guard += 1
        # who works what at `clock`: each person's highest-priority ready task
        chosen = {}
        for pers, tids in persons.items():
            avail = [tid for tid in tids if ready(tid, clock)]
            chosen[pers] = (min(avail, key=lambda x: (_order_at(sched[x], clock),
                                                      creation[x]))
                            if avail else None)
        # next decision point: earliest completion of a chosen task, or the
        # next static event after `clock`
        nxt, comp = INF, {}
        for pers, tid in chosen.items():
            if tid is None:
                continue
            c = add_work_minutes(clock, remaining[tid], busy_by_assignee.get(pers, ()))
            comp[pers] = c
            nxt = min(nxt, c)
        for s in statics:
            if s > clock:
                nxt = min(nxt, s)
        if nxt == INF:
            break  # everyone idle and no future event — remaining tasks stuck
        # accrue each chosen task's work over [clock, nxt); switch happens at nxt
        for pers, tid in chosen.items():
            if tid is None:
                continue
            w = work_minutes_between(clock, nxt, busy_by_assignee.get(pers, ()))
            if w <= 0:
                continue  # [clock, nxt) had no working minutes for this person
            segments[tid].append((clock, nxt, pers))
            started.setdefault(tid, clock)
            remaining[tid] -= w
            if remaining[tid] <= 1e-6:
                remaining[tid] = 0
                done_at[tid] = comp.get(pers, nxt)
        clock = nxt

    return {tid: {"start": started.get(tid), "done_at": done_at.get(tid),
                  "segments": segments[tid], "mult": mult[tid]}
            for tid in sched}


def _accrued_hours(seg_row, sim_start_seed, now, busy_by_assignee):
    """True EFFORT-hours done on a task by `now`: seed progress + the fold of
    its work segments up to `now`, converted from calendar-minutes back to
    effort via the task's skill multiplier (m calendar-min = m effort-min)."""
    acc = 0.0
    for (s, e, pers) in seg_row["segments"]:
        if s >= now:
            break
        acc += work_minutes_between(s, min(e, now), busy_by_assignee.get(pers, ()))
    return sim_start_seed + acc * seg_row.get("mult", 1.0) / 60.0


def task_view(tasks, sim_start, now, busy_by_assignee=None, skills=None):
    """Ground-truth task states at `now`. Tasks not yet arrived are omitted."""
    busy_by_assignee = busy_by_assignee or {}
    sched = compute_schedule(tasks, sim_start, busy_by_assignee, skills)
    out = []
    for t in tasks:
        if t.get("arrival", sim_start) > now:
            continue  # doesn't exist yet at this (possibly rewound) time
        row = {
            "id": t["id"],
            "title": t["title"],
            "assignees": t.get("assignees", []),
            "effort_hours": t.get("effort_hours"),
            "blocked_by": t.get("blocked_by", []),
            "reported": t.get("reported", ""),
            "source": t.get("source", "seed"),
            "priority": t.get("priority"),
            "urgent": bool(t.get("urgent")),
            # progress that predates the week — utilization must not credit it
            "seed_done_hours": t.get("done_hours", 0),
        }
        if not _schedulable(t):
            row.update(status="tracking", pct=None, true_done_hours=None,
                       projected_done=None)
        elif t["id"] not in sched:
            row.update(status="stuck", pct=0,
                       true_done_hours=t.get("done_hours", 0), projected_done=None)
        else:
            s = sched[t["id"]]
            effort, base = t["effort_hours"], t.get("done_hours", 0)
            done_at = s["done_at"]
            true_done = _accrued_hours(s, base, now, busy_by_assignee)
            if done_at is not None and now >= done_at:
                status, true_done = "done", effort
            else:
                # coarse board column (no % / ETA leaked): is a work segment
                # live at `now` (in_progress), waiting on a blocker (blocked),
                # or ready-but-behind / not-yet-begun (queued)?
                active = any(s0 <= now < e0 for (s0, e0, _) in s["segments"])
                blocked = any(b in sched and (sched[b]["done_at"] is None
                                              or now < sched[b]["done_at"])
                              for b in t.get("blocked_by", []))
                status = ("in_progress" if active
                          else "blocked" if blocked else "queued")
            row.update(
                status=status,
                true_done_hours=round(true_done, 1),
                pct=int(round(100.0 * true_done / effort)) if effort else 100,
                projected_done=done_at,
            )
        out.append(row)
    return out


def task_done(tasks, sim_start, now, task_id, busy_by_assignee=None, skills=None):
    sched = compute_schedule(tasks, sim_start, busy_by_assignee or {}, skills)
    s = sched.get(task_id)
    return s is not None and s["done_at"] is not None and now >= s["done_at"]
