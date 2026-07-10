"""Difficulty fingerprint: how hard is this task, computed from config alone.

    python -m sim.difficulty scenarios/demo.json

Hardness is a VECTOR, not a number — and raw event counts are a poor proxy
(heartbeats dominate the queue and add zero difficulty). What matters:

  interrupt_load        decision-forcing events (arrivals + beats), not noise
  unowned_arrivals      tasks that land without an owner — forced decisions
  winnable              OPT − baseline: how much a PM can even matter here
  slack_minutes         horizon − last completion under PERFECT play
  assignment_divergence how many tasks the optimum moves vs the authored setup
  capacity_utilization  remaining effort / total person-hours to horizon —
                        the pressure knob: ~0.3 is roomy, ~0.8 is tight
  opt_done_weight_rate  <1.0 means even PERFECT play must sacrifice work —
                        the scenario forces triage, not just scheduling

Everything is deterministic and needs zero engine runs (the null baseline is
itself analytic: no agent actions → authored assignments + arrivals, no
meetings — a pure task_view fold). The generator's validity/difficulty gates
read this fingerprint; empirical hardness (probe-ladder degradation) remains
the ground truth it must be calibrated against.
"""

import json
import math
import sys

from .optimal import _all_tasks, _skills_of, opt_ideal, worker_ids
from .rubric import load_rubric, task_value
from .sim_time import fmt, working_minutes_between
from .tasks import compute_schedule, physics_of, task_view


def baseline_value(scenario, rubric):
    """Null-agent outcome, analytically: authored assignments plus the org
    fallback pickups (a no-PM team still self-organizes, badly), no actions."""
    start = scenario.get("start_time", 545)
    horizon = rubric["horizon"]
    fb = {a["task"]["id"]: a["fallback"]
          for a in scenario.get("task_arrivals", []) if a.get("fallback")}
    tasks = [dict(t, assignees=[fb[t["id"]]["npc"]],
                  assigned_at=fb[t["id"]]["at"])
             if t["id"] in fb and not t.get("assignees") else t
             for t in _all_tasks(scenario)]
    # answers={} (default): the no-PM baseline never answers a blocking
    # question, so a gated task stalls from its open-time onward — that stall is
    # exactly the value a proactive PM recovers by replying.
    rows = task_view(tasks, start, horizon, busy_by_assignee={},
                     skills=_skills_of(scenario), answers={},
                     physics=physics_of(scenario))
    return task_value(rows, rubric.get("task_value", {}), horizon, start)


def fingerprint(scenario, rubric=None):
    rubric = rubric or load_rubric(scenario)
    horizon = rubric["horizon"]
    start = scenario.get("start_time", 545)

    arrivals = scenario.get("task_arrivals", [])
    beats = scenario.get("beats", [])
    base = baseline_value(scenario, rubric)
    opt = opt_ideal(scenario, rubric)

    # slack under perfect play: latest completion in the OPT schedule
    tasks = _all_tasks(scenario)
    assigned = [dict(t, assignees=[opt["assignment"][t["id"]]])
                if t["id"] in opt["assignment"] else t for t in tasks]
    sched = compute_schedule(assigned, start, skills=_skills_of(scenario))
    last_done = max((s["done_at"] for s in sched.values()
                     if s["done_at"] is not None), default=start)

    authored_owner = {t["id"]: (t.get("assignees") or [None])[0] for t in tasks}
    remaining = {t["id"]: t.get("effort_hours", 0) - t.get("done_hours", 0)
                 for t in tasks}
    # only tasks with real work left — a finished task's "owner" is noise
    divergence = sum(1 for tid, who in opt["assignment"].items()
                     if remaining.get(tid, 0) > 0 and authored_owner.get(tid) != who)

    # capacity pressure: how full is the week under the ideal relaxation?
    # only the LABOR pool counts — stakeholders (worker:false) don't do tasks.
    workers = len(worker_ids(scenario))
    avail_h = workers * working_minutes_between(start, horizon) / 60.0
    remaining_h = sum(max(0.0, r) for r in remaining.values())

    # -- FAIRNESS: a scored ask must be completable from the moment the PM
    # could first KNOW it. Ratio = one person's work-calendar window from
    # delivery (chat: instant; email: next batch tick) to horizon, over the
    # effort needed. <1.0 is unfair (physically impossible after discovery,
    # scoring luck); 1.0-1.3 is a tight, legitimate squeeze.
    batch = scenario.get("email_batch_minutes", 30)
    fairness = []
    for a in arrivals:
        delivery = (a["at"] if a.get("via", "chat") == "chat"
                    else (a["at"] // batch + 1) * batch)
        need = a["task"].get("effort_hours", 0) * 60
        if need:
            window = working_minutes_between(delivery, horizon)
            fairness.append({"id": a["task"]["id"], "kind": "arrival",
                             "reaction_ratio": round(window / need, 2)})
    # confessions: after a belief correction reveals the real state, enough
    # calendar must remain to absorb the revealed remaining work. Measured under
    # REASONABLE play — a sensible PM answers a blocking question promptly, so
    # the progress-so-far here assumes instant answers (else an unanswered
    # question would stall the task and make an otherwise-reactable slip look
    # unfair, which is a floor-of-play artifact, not a real trap).
    instant_answers = {t["id"]: {q["id"]: q["at"] for q in t.get("questions") or []
                                 if q.get("gates")}
                       for t in _all_tasks(scenario) if t.get("questions")}
    for t in tasks:
        for bel in (t.get("belief") or []):
            if "at" not in bel:
                continue
            rows_at = task_view(_all_tasks(scenario), start, bel["at"],
                                busy_by_assignee={}, skills=_skills_of(scenario),
                                answers=instant_answers,
                                physics=physics_of(scenario))
            row = next((r for r in rows_at if r["id"] == t["id"]), None)
            left = (t["effort_hours"] - (row["true_done_hours"] if row else 0))
            if left <= 0:
                continue
            window = working_minutes_between(bel["at"], horizon)
            fairness.append({"id": t["id"], "kind": "confession",
                             "proactive": bool(bel.get("proactive_ping")),
                             "reaction_ratio": round(window / (left * 60), 2)})
    min_reaction = min((f["reaction_ratio"] for f in fairness), default=None)

    # -- trajectory space: how many OUTCOME-DISTINGUISHABLE plans exist -----
    # The raw action space is unbounded (any text, any minute, any channel),
    # but the score is a projection: it only sees WHO ends up owning each
    # task (workers^n_live) x WHEN each ask starts being worked (a minute in
    # [first knowable, horizon]) x the noise load imposed on each person.
    # log10 of the who x when product estimates the score-relevant decision
    # space; the band [baseline, OPT] is what that whole space projects onto.
    n_live = sum(1 for t in tasks
                 if t.get("effort_hours")
                 and t["effort_hours"] - t.get("done_hours", 0) > 0)
    log_who = n_live * math.log10(max(2, workers))
    log_when = 0.0
    for a in arrivals:
        delivery = (a["at"] if a.get("via", "chat") == "chat"
                    else (a["at"] // batch + 1) * batch)
        log_when += math.log10(max(2, horizon - delivery))
    log10_classes = round(log_who + log_when, 1)

    days = max(1.0, (horizon - start) / 1440.0)
    return {
        "interrupt_load": len(arrivals) + len(beats),
        "interrupts_per_day": round((len(arrivals) + len(beats)) / days, 2),
        "unowned_arrivals": sum(1 for a in arrivals
                                if not a["task"].get("assignees")),
        "winnable_combined": round(opt["opt_combined"] - base["combined"], 3),
        "winnable_completion": round(opt["opt_completion"] - base["completion"], 3),
        "slack_minutes_at_opt": horizon - last_done,
        "opt_finishes_at": fmt(last_done),
        "assignment_divergence": divergence,
        "n_scheduled_tasks": len(opt["assignment"]),
        "capacity_utilization": round(remaining_h / avail_h, 2) if avail_h else None,
        "effort_hours_remaining": round(remaining_h, 1),
        "capacity_hours_available": round(avail_h, 1),
        "opt_done_weight_rate": opt["opt_done_weight_rate"],
        "forced_triage": opt["opt_done_weight_rate"] < 0.999,
        "min_reaction_ratio": min_reaction,
        "fair": min_reaction is None or min_reaction >= 1.0,
        "fairness": fairness,
        "log10_trajectory_classes": log10_classes,
        # raw band anchors, for stamping into the scenario file
        "baseline_combined": base["combined"],
        "opt_combined": opt["opt_combined"],
        "baseline_raw": {k: base[k] for k in
                         ("combined", "completion", "efficiency",
                          "done_weight_rate")},
        "opt_raw": {"combined": opt["opt_combined"],
                    "completion": opt["opt_completion"],
                    "efficiency": opt["opt_efficiency"],
                    "done_weight_rate": opt["opt_done_weight_rate"]},
    }


def stamp(path):
    """Embed the band anchors into the scenario file itself: the no-PM
    baseline and OPT_max every run of this task is graded against, plus the
    trajectory-space estimate. DERIVED data — re-stamp after any edit (the
    values are recomputable in milliseconds; the stamp is documentation)."""
    with open(path) as f:
        scenario = json.load(f)
    scenario.pop("band", None)  # never fingerprint a stale stamp
    fp = fingerprint(scenario)
    scenario["band"] = {
        "no_pm_baseline": fp["baseline_raw"],
        "opt_max": fp["opt_raw"],
        "winnable_combined": fp["winnable_combined"],
        "log10_trajectory_classes": fp["log10_trajectory_classes"],
        "capacity_utilization": fp["capacity_utilization"],
        "forced_triage": fp["forced_triage"],
        "min_reaction_ratio": fp["min_reaction_ratio"],
        "_": "derived by `python -m sim.difficulty --stamp` — re-stamp after edits",
    }
    with open(path, "w") as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)
    return fp


def main():
    args = [a for a in sys.argv[1:] if a != "--stamp"]
    if "--stamp" in sys.argv[1:]:
        for path in args or ["scenarios/demo.json"]:
            fp = stamp(path)
            print("stamped %-28s band=[%.2f, %.2f] winnable=%.3f "
                  "log10(classes)=%.1f"
                  % (path, fp["baseline_combined"], fp["opt_combined"],
                     fp["winnable_combined"], fp["log10_trajectory_classes"]))
        return
    path = args[0] if args else "scenarios/demo.json"
    with open(path) as f:
        scenario = json.load(f)
    fp = fingerprint(scenario)
    print("difficulty fingerprint: %s" % path)
    for k, v in fp.items():
        print("  %-24s %s" % (k, v))
    if fp["winnable_combined"] < 0.5:
        print("\nWARNING: near-degenerate — almost nothing winnable beyond baseline")


if __name__ == "__main__":
    main()
