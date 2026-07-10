"""Scenario validation + world-mutation bounds.

Two layers of control on the levers:

1. LOAD-TIME: `validate_scenario` rejects a scenario whose config exceeds
   bounds (too many tasks, unreasonable efforts, dangling references) —
   authored or generated, every scenario passes through the same gate.
2. RUNTIME: `check_task_bounds` / meeting bounds guard the mutation paths
   (external arrivals, agent-added tasks, meetings), so no actor — including
   a future LLM-driven scenario generator — can push the world outside the
   authored envelope.

The LLM itself has NO mutation channel: it produces message/transcript text
only. These bounds exist so that the things which DO mutate (config, agent
tools) stay reasonable too.
"""

DEFAULT_BOUNDS = {
    "max_tasks": 50,               # total tasks ever on the board
    "max_task_effort_hours": 80,   # one task can't be a month of work
    "min_task_effort_hours": 0.5,
    "max_external_arrivals": 10,   # OOD injections per scenario
    "max_beats": 20,
    "max_meeting_minutes": 240,    # no all-day meetings
    "max_horizon_days": 14,        # events must land inside the run window
    "max_questions_per_task": 3,   # a task can't be an endless blocking series
}


def bounds_of(scenario):
    b = dict(DEFAULT_BOUNDS)
    b.update(scenario.get("bounds", {}))
    return b


def check_task_bounds(task, existing_count, bounds):
    """Runtime gate for every task addition (seed/external/agent)."""
    if existing_count >= bounds["max_tasks"]:
        return "task limit reached (%d)" % bounds["max_tasks"]
    eh = task.get("effort_hours")
    if eh is not None and not (bounds["min_task_effort_hours"] <= eh
                               <= bounds["max_task_effort_hours"]):
        return ("effort_hours %.1f outside [%s, %s]"
                % (eh, bounds["min_task_effort_hours"],
                   bounds["max_task_effort_hours"]))
    return None


def validate_scenario(scenario):
    """Return a list of violations (empty = valid). Called at engine init."""
    errors = []
    b = bounds_of(scenario)
    start = scenario.get("start_time", 545)
    horizon = start + b["max_horizon_days"] * 1440
    npc_ids = {n["id"] for n in scenario.get("npcs", [])}
    worker_ids = {n["id"] for n in scenario.get("npcs", []) if n.get("worker", True)}

    if not npc_ids:
        # LIVENESS: NPC heartbeats are what keep the event queue non-empty.
        errors.append("scenario has no NPCs — the event queue would starve")
    if not worker_ids:
        # OPT and the baseline both need someone who actually does work; an
        # all-stakeholder cast has an empty labor pool (no winnable value).
        errors.append("scenario has no workers — every NPC is worker:false")

    tasks = (scenario.get("project") or {}).get("tasks", [])
    arrivals = scenario.get("task_arrivals", [])
    beats = scenario.get("beats", [])

    if len(tasks) + len(arrivals) > b["max_tasks"]:
        errors.append("too many tasks: %d seed + %d arrivals > max_tasks=%d"
                      % (len(tasks), len(arrivals), b["max_tasks"]))
    if len(arrivals) > b["max_external_arrivals"]:
        errors.append("too many external arrivals: %d > %d"
                      % (len(arrivals), b["max_external_arrivals"]))
    if len(beats) > b["max_beats"]:
        errors.append("too many beats: %d > %d" % (len(beats), b["max_beats"]))

    seen_ids = set()
    all_specs = [(t, "project.tasks") for t in tasks] + \
                [(a["task"], "task_arrivals") for a in arrivals]
    known_ids = {t["id"] for t, _ in all_specs}
    for t, where in all_specs:
        if t["id"] in seen_ids:
            errors.append("%s: duplicate task id %r" % (where, t["id"]))
        seen_ids.add(t["id"])
        msg = check_task_bounds(t, 0, b)
        if msg:
            errors.append("%s[%s]: %s" % (where, t["id"], msg))
        # a task MAY have several assignees: when more than one works it at once
        # it runs at the damped parallel_rate (a meeting lifts that to a swarm).
        # More hands is sublinear and costs their own work, so it's a real
        # tradeoff, not a free win. assignees[0] remains the accountable owner
        # (belief holder / question owner default).
        for a in t.get("assignees", []):
            if a not in npc_ids:
                errors.append("%s[%s]: unknown assignee %r" % (where, t["id"], a))
            elif a not in worker_ids:
                errors.append("%s[%s]: assigned to %r, a stakeholder "
                              "(worker:false) — not on the delivery team"
                              % (where, t["id"], a))
        for dep in t.get("blocked_by", []):
            if dep not in known_ids:
                errors.append("%s[%s]: unknown blocker %r" % (where, t["id"], dep))
        for j, bel in enumerate(t.get("belief") or []):
            if j == 0 and "at" in bel:
                errors.append("%s[%s]: belief[0] is the initial picture — no 'at'"
                              % (where, t["id"]))
            if j > 0 and not (start <= bel.get("at", -1) <= horizon):
                errors.append("%s[%s]: belief[%d].at outside run window"
                              % (where, t["id"], j))
            held = bel.get("held_by")
            if held is not None and held not in worker_ids:
                errors.append("%s[%s]: belief[%d].held_by=%r must be a worker "
                              "(the person carrying the stale estimate)"
                              % (where, t["id"], j, held))
        # blocking questions: a gating question suspends the task until the PM
        # replies to its OWNER. Must open in-window, be owned by the task's sole
        # worker, and carry an id (completability is checked in the fairness gate).
        qs = t.get("questions") or []
        if len(qs) > b["max_questions_per_task"]:
            errors.append("%s[%s]: %d questions > max_questions_per_task=%d"
                          % (where, t["id"], len(qs), b["max_questions_per_task"]))
        owner = (t.get("assignees") or [None])[0]
        for j, q in enumerate(qs):
            if "id" not in q:
                errors.append("%s[%s]: questions[%d] needs an id" % (where, t["id"], j))
            if not (start <= q.get("at", -1) <= horizon):
                errors.append("%s[%s]: questions[%d].at outside run window"
                              % (where, t["id"], j))
            held = q.get("held_by", owner)
            if held != owner:
                errors.append("%s[%s]: questions[%d].held_by=%r is not the task's "
                              "owner %r — only the doer can be blocked on it"
                              % (where, t["id"], j, held, owner))
            if owner is not None and owner not in worker_ids:
                errors.append("%s[%s]: a blocking question needs a worker owner, "
                              "got %r" % (where, t["id"], owner))

    for i, arr in enumerate(arrivals):
        if not (start <= arr["at"] <= horizon):
            errors.append("task_arrivals[%d]: at=%d outside run window" % (i, arr["at"]))
        if arr.get("npc") and arr["npc"] not in npc_ids:
            errors.append("task_arrivals[%d]: unknown npc %r" % (i, arr["npc"]))
        if arr.get("via") not in (None, "chat", "email"):
            errors.append("task_arrivals[%d]: via must be chat|email, got %r"
                          % (i, arr["via"]))
        # Real work always has an OWNER — an arrival must land already assigned
        # to a worker (no unowned backlog). A complete no-PM baseline: the
        # default holder works it, badly if they're the wrong/swamped choice;
        # the PM earns credit by REARRANGING to a better owner, not by merely
        # filing something nobody was on.
        owners = arr["task"].get("assignees") or []
        if not owners:
            errors.append("task_arrivals[%d]: task %r has no owner — arrivals "
                          "must land assigned to a default holder"
                          % (i, arr["task"]["id"]))
        for o in owners:
            osp = next((n for n in scenario.get("npcs", []) if n["id"] == o), None)
            if osp is None:
                errors.append("task_arrivals[%d]: owner %r unknown" % (i, o))
            elif not osp.get("worker", True):
                errors.append("task_arrivals[%d]: owner %r is a stakeholder — "
                              "not on the delivery team" % (i, o))
        if arr.get("announce") and not arr.get("npc"):
            errors.append("task_arrivals[%d]: announce needs an npc messenger" % i)
        if not arr.get("announce"):
            # unfiled + unannounced = a task nobody could ever learn about
            errors.append("task_arrivals[%d]: arrivals reach the PM only via "
                          "announce (chat/email) — silent arrivals are "
                          "undiscoverable now that filing is required" % i)

    # FAIRNESS IS A RUBRIC TOO: a scored ask must be physically completable
    # from the moment the PM could first KNOW it (chat: instant; email: the
    # next batch tick). Anything tighter scores luck, not skill — rejected
    # at authoring time, same as any other invalid scenario.
    grade_horizon = None
    try:
        from .rubric import load_rubric
        grade_horizon = load_rubric(scenario).get("horizon")
    except SystemExit:
        pass  # no rubric resolvable here — the fingerprint still reports
    if grade_horizon:
        from .sim_time import working_minutes_between
        batch = scenario.get("email_batch_minutes", 30)
        for i, arr in enumerate(arrivals):
            delivery = (arr["at"] if arr.get("via", "chat") == "chat"
                        else (arr["at"] // batch + 1) * batch)
            need = arr["task"].get("effort_hours", 0) * 60
            window = working_minutes_between(delivery, grade_horizon)
            if need and window < need:
                errors.append(
                    "task_arrivals[%d]: UNFAIR — %r needs %d work-min but only "
                    "%d remain after the PM can first know of it (%s %s)"
                    % (i, arr["task"]["id"], need, window,
                       arr.get("via", "chat"), "delivery"))
        # A blocking question is fair BY CONSTRUCTION: answered the instant it
        # opens it is identical to no question at all (empty [at, at) block), so
        # a prompt reply always recovers the no-question outcome — the winnable
        # value is exactly the stall the PM saves. The only trap is a question
        # opening in dead time (no working minutes left to matter), which the
        # in-window check above already forbids. So no extra effort-fits gate.
    for i, beat in enumerate(beats):
        if not (start <= beat["at"] <= horizon):
            errors.append("beats[%d]: at=%d outside run window" % (i, beat["at"]))
        if beat.get("npc") not in npc_ids:
            errors.append("beats[%d]: unknown npc %r" % (i, beat.get("npc")))
        arms = beat.get("arms")
        if arms is not None:
            if not arms or any("intent" not in a for a in arms):
                errors.append("beats[%d]: arms must be non-empty, each with an intent" % i)
        elif "intent" not in beat:
            errors.append("beats[%d]: needs an intent or arms" % i)
    return errors
