"""Project/task world model: analytic work-calendar scheduling.

Task progress is a PURE FUNCTION of (task list, logged mutations, sim time) —
no ticks. Tasks arrive three ways: seeded in the scenario (arrival = sim
start), injected on a schedule by external beats (OOD arrivals), or added by
the agent for tracking. Every arrival is a logged `task_added` mutation, so
rewind/replay reconstruct the board at any time.

Scheduling semantics (deterministic, non-preemptive, work-conserving):
- Each assignee works one task at a time, only during working hours
  (Mon-Fri 09:00-17:30).
- A task can start only after: its assignee is free, it has ARRIVED, and all
  its blockers are done.
- When an assignee frees up, they pick the highest-priority ready task:
  urgent tasks first, then creation order. Urgent arrivals jump the queue but
  do NOT interrupt in-flight work.
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


def _work_order_key(t, creation_idx):
    """How an assignee sorts their ready queue: urgent first, then priority,
    then creation order. The PM can REORDER via `order_urgent` / `order_priority`
    (set by the reprioritize tool) — these override the defaults for SCHEDULING
    only. Scoring weight still comes from the authored `priority`, so the PM
    reorders work but cannot relabel a task to mint value."""
    urgent = t.get("order_urgent")
    if urgent is None:
        urgent = t.get("urgent")
    prio = t.get("order_priority") or t.get("priority")
    return (0 if urgent else 1, _PRIORITY_RANK.get(prio, 2), creation_idx)


def _earliest(t, sim_start):
    """Work can't start before the task EXISTS (arrival) or before somebody
    actually HOLDS it (assigned_at, stamped by assignment mutations). Without
    the latter, assignment is retroactive — picking a task up at 16:00 would
    be credited work since its 11:30 arrival, and acting early earns nothing."""
    return max(t.get("arrival", sim_start), t.get("assigned_at") or 0)


def _primary(t):
    return t["assignees"][0] if t.get("assignees") else None


def _schedulable(t):
    return t.get("assignees") and t.get("effort_hours") is not None


def compute_schedule(tasks, sim_start, busy_by_assignee=None):
    """Per schedulable task: {start, done_at}. Greedy event-driven scheduler.

    busy_by_assignee: {person: [(start, end), ...]} — meeting intervals that
    consume that person's working time (meetings have a real capacity cost).
    """
    busy_by_assignee = busy_by_assignee or {}
    order = {t["id"]: i for i, t in enumerate(tasks)}
    sched = {t["id"]: t for t in tasks if _schedulable(t)}
    info, done_at, free = {}, {}, {}
    for t in sched.values():
        free.setdefault(_primary(t), sim_start)
    unfinished = set(sched)
    waiting = set()  # assignees that can't progress until someone else finishes
    guard = 0

    while unfinished and guard < 10000:
        guard += 1
        active = {_primary(sched[tid]) for tid in unfinished}
        pickable = [a for a in active if a not in waiting]
        if not pickable:
            break  # dependency cycle — remaining tasks are stuck
        a = min(pickable, key=lambda x: (free[x], str(x)))
        t_now = free[a]

        def ready(tid):
            t = sched[tid]
            if _primary(t) != a or _earliest(t, sim_start) > t_now:
                return False
            for b in t.get("blocked_by", []):
                if b in done_at:
                    if done_at[b] > t_now:
                        return False
                elif b in sched:
                    return False  # blocker not finished yet
            return True

        avail = [tid for tid in unfinished if ready(tid)]
        if avail:
            # urgent first, then priority, then creation order — people
            # sort their own queue sensibly even with no PM in the world;
            # the PM's reprioritize overrides (order_*) apply here and
            # ONLY here (scoring weight stays authored)
            tid = min(avail, key=lambda x: _work_order_key(sched[x], order[x]))
            t = sched[tid]
            remaining = int(round(max(0, t["effort_hours"] - t.get("done_hours", 0)) * 60))
            end = add_work_minutes(t_now, remaining, busy_by_assignee.get(a, ()))
            info[tid] = {"start": t_now, "done_at": end}
            done_at[tid] = end
            free[a] = end
            unfinished.discard(tid)
            waiting.clear()  # a completion may enable someone else
        else:
            # advance a's clock to the next enabling moment (arrival/blocker)
            nxts = []
            for tid in unfinished:
                t = sched[tid]
                if _primary(t) != a:
                    continue
                if _earliest(t, sim_start) > t_now:
                    nxts.append(_earliest(t, sim_start))
                for b in t.get("blocked_by", []):
                    if b in done_at and done_at[b] > t_now:
                        nxts.append(done_at[b])
            if nxts:
                free[a] = min(nxts)
            else:
                waiting.add(a)  # blocked on another assignee's unfinished work
    return info


def task_view(tasks, sim_start, now, busy_by_assignee=None):
    """Ground-truth task states at `now`. Tasks not yet arrived are omitted."""
    busy_by_assignee = busy_by_assignee or {}
    sched = compute_schedule(tasks, sim_start, busy_by_assignee)
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
            if now >= s["done_at"]:
                status, true_done = "done", effort
            elif now < s["start"]:
                blocked = any(
                    b in sched and now < sched[b]["done_at"]
                    for b in t.get("blocked_by", [])
                )
                status, true_done = ("blocked" if blocked else "queued"), base
            else:
                status = "in_progress"
                true_done = base + work_minutes_between(
                    s["start"], now, busy_by_assignee.get(_primary(t), ())) / 60.0
            row.update(
                status=status,
                true_done_hours=round(true_done, 1),
                pct=int(round(100.0 * true_done / effort)) if effort else 100,
                projected_done=s["done_at"],
            )
        out.append(row)
    return out


def task_done(tasks, sim_start, now, task_id, busy_by_assignee=None):
    sched = compute_schedule(tasks, sim_start, busy_by_assignee or {})
    return task_id in sched and now >= sched[task_id]["done_at"]
