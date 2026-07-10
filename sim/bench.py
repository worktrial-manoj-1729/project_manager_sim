"""Max@k bench aggregator: read all scored runs, group by model, report the
distribution of each normalized metric — mean, max@k, min, std — against the
baseline (0) and OPT (1).

    python -m sim.bench                 # aggregate everything under runs/
    python -m sim.bench runs            # same

A task is a good GRPO target when, per capable tier: Max@k clears baseline on
EVERY metric (achievable), the mean is climbable (0<mean<1), and there is
real variance (a spread to learn from). Single runs are noisy — read the
distribution, never one rollout.

This only aggregates; produce the runs with parallel harness invocations
(collision-safe run dirs), e.g.:
    for m in ...; do for k in 1 2 3; do python -m sim.harness ... --model $m & done; done; wait
"""

import glob
import json
import os
import sys

METRICS = ["completion", "efficiency", "done_weight_rate", "combined"]


def _load(runs_root):
    by_model = {}
    for meta_path in glob.glob(os.path.join(runs_root, "*", "meta.json")):
        d = os.path.dirname(meta_path)
        sc_path = os.path.join(d, "scorecard.json")
        if not os.path.exists(sc_path):
            continue
        try:
            meta = json.load(open(meta_path))
            sc = json.load(open(sc_path))
        except (ValueError, OSError):
            continue
        model = meta.get("agent_model", "?")
        row = {m: (sc.get(m) or {}).get("normalized") for m in METRICS}
        if any(v is not None for v in row.values()):
            by_model.setdefault(model, []).append(row)
    return by_model


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {"n": n, "mean": mean, "max": max(vals), "min": min(vals),
            "std": var ** 0.5}


def main():
    runs_root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    by_model = _load(runs_root)
    if not by_model:
        print("no scored runs under %s/" % runs_root)
        sys.exit(1)

    order = ["scripted", "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8"]
    models = sorted(by_model, key=lambda m: order.index(m) if m in order else 99)

    print("Max@k over %s/  (normalized: 0 = do-nothing baseline, 1 = OPT ideal)\n"
          % runs_root)
    print("  %-16s %-3s %-24s %-24s" % ("model", "k", "metric", "mean [min .. MAX@k] std"))
    all_good = {}
    for model in models:
        rows = by_model[model]
        k = len(rows)
        for i, metric in enumerate(METRICS):
            st = _stats([r[metric] for r in rows])
            if st is None:
                continue
            flag = ""
            if metric == "combined":
                all_good[model] = None  # placeholder
            beats = "✓" if st["max"] > 0 else "✗ (never clears baseline)"
            label = model.replace("claude-", "") if i == 0 else ""
            kk = str(k) if i == 0 else ""
            print("  %-16s %-3s %-18s %+.3f [%+.3f .. %+.3f] σ=%.3f  %s"
                  % (label, kk, metric, st["mean"], st["min"], st["max"],
                     st["std"], beats))
        # does Max@k clear baseline on EVERY metric?
        maxes = {m: _stats([r[m] for r in rows]) for m in METRICS}
        every = all(s and s["max"] > 0 for s in maxes.values())
        print("  %-16s -> Max@%d clears baseline on ALL metrics: %s\n"
              % ("", k, "YES ✓" if every else "no"))


if __name__ == "__main__":
    main()
