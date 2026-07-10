# Workload fairness

*Did the plan spread work across the team, or ride one person while others
idle?* — a rubric over the *shape* of the schedule, not just its outputs.

## Definition

Over the **worker** pool (stakeholders excluded), per person `a`:

```
utilizationₐ = (hours a worked within the graded window) / (hours available)
workload_fairness = 1 − σ(utilization)
```

where `σ` is the population standard deviation of utilization across workers
(`sim/rubric.py::task_value`, active when `workers` is passed). Available hours
= working-calendar minutes from `t0` to `H` (Mon–Fri 09:00–17:30). Only hours
worked *inside* the window count — pre-week progress (`seed_done_hours`) is
subtracted, so you are not credited for work that predates the episode.

## Why variance, not the max–min gap

A max–min spread only sees the two extreme people. Standard deviation sees
**everyone's** imbalance — three people at 90/50/10 is flagged as unfair even
though a fourth might sit at the mean. Fairness = `1 − σ` so higher is more
balanced; `1.0` is perfectly even, lower is more skewed.

## Reported raw against OPT, not normalized to 1.0

Fairness is **not** normalized into `[0,1]` like the outcome metrics. Perfect
balance is often *impossible*: a dependency chain (migration → backfill, both
serial on one person) forces that person hot while others wait. So we report
the raw triplet `baseline → agent → OPT` and use **OPT's own fairness as the
reference** — "how balanced was the week under perfect play" — rather than
comparing to an unreachable `1.0`.

```
FAIRNESS  1−σ(util):  baseline B_f → agent A_f → OPT O_f
          util:  sarah XX%  dave YY%
```

## Interpretation for the talk

- Fairness is a **pure consequence of assignment decisions** — no behavioral
  rule, no penalty. Dumping everything on Sarah while Dave idles shows up as
  high `σ` in the schedule itself.
- It is diagnostic, not (currently) part of the RL reward scalar `K`. It
  explains *how* a policy reached its completion/efficiency — a useful readout
  when two policies score the same `K` by different means (one balanced, one
  by overloading a single engineer).
- Because stakeholders are `worker: false`, they never appear in the
  utilization denominator — a VP sitting at 0% work is not "unfair," they were
  never labor.
