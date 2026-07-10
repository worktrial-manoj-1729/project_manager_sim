"""Task-intent audit: WHICH decisions carry the winnable band?

    python -m sim.intent scenarios/demo.json

A task is authored with an INTENT — the decisions a winning PM must make
(act the moment information lands, allocate better than the org volunteer,
keep noise off people's focus). The trajectory space is huge, but a
well-authored task collapses it onto the [baseline, OPT] band where a
trajectory's position equals which intended mechanisms it captured.

This audit VERIFIES the intent causally, with zero LLM calls: it plays a
family of deterministic policies that differ in exactly one decision and
measures each one's score. share(mechanism) = score(oracle) − score(without
that mechanism): the slice of the band that decision owns.

Reading the report:
  - an intended mechanism with share ~0 is NOT taught by this task
    (physics lets you skip it for free — re-author)
  - one mechanism owning nearly everything = a one-bit task: sparse
    learning signal, saturates after a single insight
  - several comparable shares = dense signal across skill levels

Policies (all friction-respecting; acting only on knowable information):
  oracle       seed moves Monday; file+assign each arrival to its OPT owner
               at the earliest instant it is knowable (chat announce / email
               batch tick)
  late         same allocations, but only AFTER the org fallback has fired —
               isolates the TIMING mechanism (preempt the volunteer)
  no-spread    same timing, but every arrival goes to its fallback volunteer —
               isolates the ALLOCATION mechanism (owner choice)
  noisy        oracle + 3 chat pings to each assignee per decision —
               isolates CHANNEL DISCIPLINE (the serialized focus tax)
"""

import json
import shutil
import sys

from .engine import Engine
from .eval import StubClient, evaluate
from .optimal import opt_ideal
from .tools import call_tool


def _knowable_at(arr, scenario):
    """Earliest instant the PM can know this arrival exists."""
    if arr.get("via", "chat") == "email":
        batch = scenario.get("email_batch_minutes", 30)
        return (arr["at"] // batch + 1) * batch + 1
    return arr["at"] + 1


def _run_policy(scenario, name, opt, timing, owner, noisy=False):
    """One deterministic trajectory. timing: 'early'|'late'. owner:
    'opt'|'fallback'."""
    tag = (scenario.get("project") or {}).get("id", "x")
    run_dir = "runs/intent-%s-%s" % (tag, name)
    # a fresh dir ALWAYS: reusing one appends to the old event log and the
    # evaluation replays a franken-history (this once made oracle "beat" OPT)
    shutil.rmtree(run_dir, ignore_errors=True)
    eng = Engine(scenario, client=StubClient(), verbose=False,
                 run_dir=run_dir)
    w = eng.world

    def goto(t):
        while w.clock < t:
            call_tool(eng, "advance_time", {"minutes": t - w.clock})

    def ping(npc):
        for i in range(3):
            call_tool(eng, "send_chat",
                      {"npc": npc, "text": "quick check-in %d — how is it "
                                           "going?" % i})

    # seed reassignments OPT wants: possible Monday, no information needed
    for t in scenario["project"]["tasks"]:
        want = opt["assignment"].get(t["id"])
        have = (t.get("assignees") or [None])[0]
        if want and have and want != have:
            call_tool(eng, "assign_task", {"task_id": t["id"], "npc": want})
            if noisy:
                ping(want)

    plan = []
    for arr in scenario.get("task_arrivals", []):
        when = (_knowable_at(arr, scenario) if timing == "early"
                else arr["fallback"]["at"] + 5)
        who = (opt["assignment"].get(arr["task"]["id"], arr["fallback"]["npc"])
               if owner == "opt" else arr["fallback"]["npc"])
        plan.append((when, arr["task"]["id"], arr["task"]["title"], who))
    for when, tid, title, who in sorted(plan):
        goto(when)
        call_tool(eng, "add_task", {"title": title, "id": tid})
        call_tool(eng, "assign_task", {"task_id": tid, "npc": who})
        if noisy:
            ping(who)

    horizon = None
    try:
        from .rubric import load_rubric
        horizon = load_rubric(scenario)["horizon"]
    except SystemExit:
        horizon = w.clock + 7 * 1440
    goto(horizon)
    return evaluate(run_dir)


def audit(scenario):
    opt = opt_ideal(scenario)
    runs = {
        "oracle":    _run_policy(scenario, "oracle", opt, "early", "opt"),
        "late":      _run_policy(scenario, "late", opt, "late", "opt"),
        "no-spread": _run_policy(scenario, "no-spread", opt, "early", "fallback"),
        "noisy":     _run_policy(scenario, "noisy", opt, "early", "opt",
                                 noisy=True),
    }
    top = runs["oracle"]["score"]
    mechanisms = {
        "timing (preempt the fallback)":  top - runs["late"]["score"],
        "allocation (owner choice)":      top - runs["no-spread"]["score"],
        "channel discipline (focus tax)": top - runs["noisy"]["score"],
    }
    return {"scores": {k: v["score"] for k, v in runs.items()},
            "odds": {k: v.get("score_odds") for k, v in runs.items()},
            "oracle_score": top,
            # 1 - oracle = the relaxation gap: band no feasible policy reaches
            "relaxation_gap": round(1 - top, 3),
            "mechanism_shares": {k: round(v, 3)
                                 for k, v in mechanisms.items()}}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/demo.json"
    with open(path) as f:
        scenario = json.load(f)
    r = audit(scenario)
    print("intent audit: %s" % path)
    print("  policy scores:  %s" % "  ".join(
        "%s=%.3f" % (k, v) for k, v in r["scores"].items()))
    print("  relaxation gap: %.1f%% of the band is unreachable by any policy"
          % (r["relaxation_gap"] * 100))
    print("  mechanism shares of the band (oracle minus ablation):")
    for k, v in sorted(r["mechanism_shares"].items(), key=lambda x: -x[1]):
        flag = "   <- NOT TAUGHT by this task" if v < 0.02 else ""
        print("    %-34s %6.3f%s" % (k, v, flag))


if __name__ == "__main__":
    main()
