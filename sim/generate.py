"""Difficulty-tuned task generation: seeded remix of a template scenario.

    python -m sim.generate scenarios/demo.json --tier hard --n 5 --seed 7 \
        --out scenarios/gen

This is where randomness LIVES (DESIGN.md §9): every draw is keyed by
(template, tier, instance-seed) and MATERIALIZED into the emitted config, so
each instance is exactly reproducible and every gate can see what it got.
No LLM anywhere: the template supplies the fiction (personas, announce
texts, herrings), the generator perturbs the PHYSICS, and the gates keep
only instances that are valid, fair, and inside the requested difficulty
band. Rejected samples are just resampled — authoring bugs cannot ship.

Pipeline per instance:   sample -> validate -> fingerprint -> gate -> stamp

What varies (instance latents — the in-context learnables):
  - task efforts and pre-done fractions (tier-scaled jitter)
  - arrival times (work-hours jitter) and TICKET IDS (anti-memorization:
    filing must come from reading your inbox, never from weights)
  - the fallback volunteer (uniform over workers) and pickup latency
    (urgent asks get grabbed in hours; paperwork waits for the panic —
    the gap between those is where PM value lives, so the tiers tune it)
  - belief error sizes and confession times
  - the NPC latency seed
What never varies (in-weights physics): channel taxes, batch grids, working
hours — the environment constants a trained policy is supposed to absorb.
"""

import argparse
import copy
import json
import os
import random

from .difficulty import fingerprint, stamp
from .optimal import worker_ids
from .validate import validate_scenario

GENERATOR_VERSION = 1

# tier = target difficulty band, enforced by the gate, shaped by the knobs.
# Under the owned-arrivals contract (real work always lands on a default
# holder; the PM's lever is REARRANGING) the band exists only when defaults
# are WRONG: the load-bearing knobs are pileup (probability every arrival
# lands on the same volunteer) and the effort multipliers that overflow that
# person's week. The gate rejects anything degenerate, so mis-set knobs cost
# attempts, never bad instances.
TIERS = {
    "easy":   {"util": (0.0, 0.55), "min_winnable": 0.2, "no_triage": True,
               "effort_mult": (0.60, 0.95), "arr_mult": (0.8, 1.3),
               "pileup": 0.6},
    "medium": {"util": (0.40, 0.85), "min_winnable": 0.5, "no_triage": False,
               "effort_mult": (0.90, 1.30), "arr_mult": (1.0, 1.6),
               "pileup": 0.85},
    "hard":   {"util": (0.70, 2.00), "min_winnable": 1.0, "no_triage": False,
               "effort_mult": (1.10, 1.60), "arr_mult": (1.2, 2.0),
               "pileup": 1.0},
}


def _r5(x):
    """Efforts live on a half-hour grid."""
    return max(0.5, round(x * 2) / 2.0)


def _snap_work(t, start):
    """Push a sampled instant into working hours (Mon–Fri 09:05–17:00)."""
    t = max(int(t), start + 30)
    while True:
        day, m = divmod(t, 1440)
        if day % 7 >= 5:
            t = (day + (7 - day % 7)) * 1440 + 545
        elif m < 545:
            t = day * 1440 + 545
        elif m > 1020:
            t = (day + 1) * 1440 + 545
        else:
            return t


def sample(template, tier, inst_seed):
    """One candidate instance: a deep remix of the template's physics.
    Deterministic — the rng is keyed by (template project, tier, seed)."""
    knobs = TIERS[tier]
    s = copy.deepcopy(template)
    name = (s.get("project") or {}).get("id", "task")
    rng = random.Random("%s:%s:%d" % (name, tier, inst_seed))
    start = s.get("start_time", 545)
    horizon = (s.get("evaluation") or {}).get("horizon") or \
        (s.get("project") or {}).get("due", start + 5 * 1440)
    workers = worker_ids(s)

    s["seed"] = inst_seed          # per-instance NPC latency streams
    s.pop("band", None)            # never inherit a stale stamp

    # seed tasks: scale effort, keep the done-fraction (the narrative's
    # beliefs stay approximately coherent), jitter belief pcts and times
    for t in (s.get("project") or {}).get("tasks", []):
        if not t.get("effort_hours"):
            continue
        frac = (t.get("done_hours", 0) or 0) / t["effort_hours"]
        t["effort_hours"] = min(80.0, _r5(
            t["effort_hours"] * rng.uniform(*knobs["effort_mult"])))
        t["done_hours"] = min(t["effort_hours"], _r5(t["effort_hours"] * frac))
        for bel in t.get("belief") or []:
            if "pct" in bel:
                bel["pct"] = int(min(95, max(5, bel["pct"] + rng.randint(-8, 8))))
            if "at" in bel:
                bel["at"] = _snap_work(bel["at"] + rng.randint(-240, 240), start)

    # arrivals: jitter timing, rescale effort, RANDOMIZE the ticket id, and
    # sample the DEFAULT OWNER — with probability `pileup` every arrival
    # lands on the same volunteer (the classic org failure the PM must fix)
    volunteer = rng.choice(workers)
    pile = rng.random() < knobs["pileup"]
    for arr in s.get("task_arrivals", []):
        arr["at"] = _snap_work(arr["at"] + rng.randint(-180, 300), start)
        arr.pop("fallback", None)   # owned-arrivals contract: no org pickup
        task = arr["task"]
        task["effort_hours"] = min(80.0, _r5(
            task["effort_hours"] * rng.uniform(*knobs["arr_mult"])))
        stem = task["id"].rsplit("-", 1)[0][:24]
        task["id"] = "%s-%04x" % (stem, rng.randrange(16 ** 4))
        task["assignees"] = [volunteer if pile else rng.choice(workers)]

    s["generated"] = {"template": name, "tier": tier, "seed": inst_seed,
                      "generator_version": GENERATOR_VERSION}
    return s


def gate(scenario, tier):
    """(ok, fingerprint, reasons) — validity, fairness, and the tier band."""
    knobs = TIERS[tier]
    problems = validate_scenario(scenario)
    if problems:
        return False, None, ["invalid: %s" % problems[0]]
    fp = fingerprint(scenario)
    reasons = []
    if not fp["fair"]:
        reasons.append("unfair: min_reaction_ratio %s" % fp["min_reaction_ratio"])
    lo, hi = knobs["util"]
    if not (lo <= (fp["capacity_utilization"] or 0) <= hi):
        reasons.append("utilization %s outside [%s, %s]"
                       % (fp["capacity_utilization"], lo, hi))
    if fp["winnable_combined"] < knobs["min_winnable"]:
        reasons.append("winnable %.3f < %.1f — degenerate for this tier"
                       % (fp["winnable_combined"], knobs["min_winnable"]))
    if knobs["no_triage"] and fp["forced_triage"]:
        reasons.append("forced triage in an easy instance")
    return not reasons, fp, reasons


def generate(template_path, tier, n, base_seed, out_dir, max_attempts=60,
             verbose=True):
    with open(template_path) as f:
        template = json.load(f)
    tname = os.path.splitext(os.path.basename(template_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for i in range(n):
        for attempt in range(max_attempts):
            inst_seed = base_seed * 100000 + i * 1000 + attempt
            cand = sample(template, tier, inst_seed)
            ok, fp, reasons = gate(cand, tier)
            if not ok:
                continue
            path = os.path.join(out_dir, "%s-%s-%d.json" % (tname, tier, inst_seed))
            with open(path, "w") as f:
                json.dump(cand, f, indent=2, ensure_ascii=False)
            stamp(path)   # embed the band anchors the instance was gated on
            made.append(path)
            if verbose:
                print("%-44s util %-5s winnable %-6s triage %-5s reaction %s"
                      % (os.path.basename(path), fp["capacity_utilization"],
                         fp["winnable_combined"], fp["forced_triage"],
                         fp["min_reaction_ratio"]))
            break
        else:
            if verbose:
                print("instance %d: no valid sample in %d attempts "
                      "(template may not reach tier %r)" % (i, max_attempts, tier))
    return made


def main():
    p = argparse.ArgumentParser()
    p.add_argument("template", nargs="?", default="scenarios/demo.json")
    p.add_argument("--tier", choices=sorted(TIERS), default="medium")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="scenarios/gen")
    args = p.parse_args()
    made = generate(args.template, args.tier, args.n, args.seed, args.out)
    print("generated %d/%d instances -> %s" % (len(made), args.n, args.out))


if __name__ == "__main__":
    main()
