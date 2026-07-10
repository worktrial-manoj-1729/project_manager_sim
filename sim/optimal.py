"""OPT_ideal: the deterministic ceiling on task_value for a scenario.

    python -m sim.optimal scenarios/demo.json

The IDEALIZED relaxation — every friction at its best value:
  - communication free & instantaneous (full information at t=0)
  - skill multiplier 1x for everyone (any person can do any task)
  - no meetings / interruption overhead (busy = {})
  - decisions cost nothing

What remains are the HARD constraints: arrivals (a task can't start before
it lands), precedence, effort hours, one-task-at-a-time per person, working
calendars, and the horizon.

OPT_ideal is a pure function of the scenario file alone — no NPCs, no LLM,
no agent, no engine version. It's a true upper bound on any achievable
score, computable at scenario-generation time, and a stable denominator for
normalized comparison ("captured X% of ideal") across environment versions.

Computation: exact search over worker assignments (workers^n for the
schedulable tasks) x candidate orderings, each evaluated with the SAME
compute_schedule the world runs (frictions zeroed). Milliseconds at our
scale; scales as an admissible bound via the weight-descending heuristic if
instances ever grow past exhaustive range.
"""

import itertools
import json
import sys

from .rubric import DEFAULT_PRIORITY_WEIGHTS, load_rubric, task_value
from .tasks import task_view


def _all_tasks(scenario):
    start = scenario.get("start_time", 545)
    tasks = [dict(t, arrival=start, source="seed")
             for t in (scenario.get("project") or {}).get("tasks", [])]
    for arr in scenario.get("task_arrivals", []):
        tasks.append(dict(arr["task"], arrival=arr["at"], source="external"))
    return tasks


def worker_ids(scenario):
    """The labor pool OPT may assign to. Stakeholders you REPORT to (a VP, an
    exec) carry `worker: false` and are EXCLUDED — treating them as fungible
    labor inflates the ceiling with work no intended policy should do, and
    deflates every normalized score against it."""
    return [n["id"] for n in scenario["npcs"] if n.get("worker", True)]


def _skills_of(scenario):
    return {n["id"]: n["skills"] for n in scenario["npcs"] if n.get("skills")}


def opt_ideal(scenario, rubric=None, max_exhaustive=200000):
    rubric = rubric or load_rubric(scenario)
    tv_cfg = rubric.get("task_value", {})
    weights = tv_cfg.get("priority_weights", DEFAULT_PRIORITY_WEIGHTS)
    horizon = rubric["horizon"]
    start = scenario.get("start_time", 545)
    workers = worker_ids(scenario)
    skills = _skills_of(scenario)

    tasks = _all_tasks(scenario)
    fixed = [t for t in tasks if not t.get("effort_hours")]
    free = [t for t in tasks if t.get("effort_hours")]

    def evaluate(assignment, ordering):
        # frictionless (busy={}) but with REAL skills — OPT's search over
        # assignments naturally routes each task to its fastest eligible
        # worker, so it stays a true upper bound (the agent's assignment is
        # one point in this same search space).
        candidate = [dict(t, assignees=[a]) for t, a in zip(ordering, assignment)]
        rows = task_view(candidate + fixed, start, horizon,
                         busy_by_assignee={}, skills=skills)
        return task_value(rows, tv_cfg, horizon, start, workers=workers)

    # three ceilings tracked in ONE search pass: each metric gets its own
    # maximum (the completion-optimal and efficiency-optimal assignments
    # may differ)

    # candidate orderings: authored order + weight-descending (urgent-first
    # engine semantics make ordering matter for contended workers)
    orderings = [list(free)]
    by_weight = sorted(free, key=lambda t: -weights.get(t.get("priority") or "P2", 2))
    if [t["id"] for t in by_weight] != [t["id"] for t in free]:
        orderings.append(by_weight)

    n_combos = len(workers) ** len(free) * len(orderings)
    if n_combos > max_exhaustive:
        raise SystemExit("instance too large for exhaustive OPT (%d combos); "
                         "add a relaxation bound" % n_combos)

    best = {"completion": (-1.0, None), "efficiency": (-1.0, None),
            "done_weight_rate": (-1.0, None), "combined": (-1.0, None, None)}
    for ordering in orderings:
        for assignment in itertools.product(workers, repeat=len(free)):
            v = evaluate(assignment, ordering)
            named = dict(zip((t["id"] for t in ordering), assignment))
            for metric in ("completion", "efficiency", "done_weight_rate"):
                if (v[metric] or 0) > best[metric][0]:
                    best[metric] = (v[metric], named)
            if v["combined"] > best["combined"][0]:
                best["combined"] = (v["combined"], named, v)
    v_best = best["combined"][2]
    return {
        "opt_completion": best["completion"][0],
        "opt_efficiency": best["efficiency"][0],
        "opt_done_weight_rate": best["done_weight_rate"][0],
        "opt_combined": best["combined"][0],
        "assignment": best["combined"][1],
        "per_task": v_best["per_task"],
        # fairness OF the combined-optimal schedule — the reference point:
        # how balanced the week is under perfect play (chains can force
        # imbalance even on OPT, so agents are compared to this, not to 1.0)
        "opt_workload_fairness": v_best["workload_fairness"],
        "opt_utilization": v_best["utilization"],
        "horizon": horizon,
        "combos_searched": n_combos,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/demo.json"
    with open(path) as f:
        scenario = json.load(f)
    result = opt_ideal(scenario)
    print("OPT completion=%.3f  efficiency=%.3f  combined=%.3f  (searched %d combos)"
          % (result["opt_completion"], result["opt_efficiency"],
             result["opt_combined"], result["combos_searched"]))
    print("optimal assignment:", json.dumps(result["assignment"], indent=2))
    for t in result["per_task"]:
        print("  %-24s %s w=%d progress=%.0f%% done=%s completion=%.2f efficiency=%.2f"
              % (t["id"], t["priority"], t["weight"], t["progress"] * 100,
                 t["done"], t["completion"], t["efficiency"]))


if __name__ == "__main__":
    main()
