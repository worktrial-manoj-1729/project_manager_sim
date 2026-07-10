# Scoring & difficulty docs

Slide-ready technical notes on how a run is graded and how a scenario's
difficulty is measured. Every quantity below is **deterministic**: a pure
function of the scenario file and the agent's logged action trajectory. No LLM
touches the score.

## The one-paragraph version

A run is graded by a small set of **outcome metrics** computed on the true task
state at a fixed horizon (Friday 17:00). Each metric is reported on a **band**
`[B, OPT]`: `B` is the no-PM baseline (a self-organizing team that never gets a
PM), `OPT` is the frictionless optimum (a pure function of the scenario). The
headline reward is the agent's position in that band,
`normalized = (agent − B) / (OPT − B)` — 0 means "no better than having no PM,"
1 means "reached the theoretical ceiling." The band is also the difficulty
knob: if `OPT − B` is small, or if the whole band collapses onto one trivial
action, there is nothing to learn.

## Read in this order

1. [scoring-band.md](scoring-band.md) — **the `[B, OPT]` band**: baseline, OPT,
   normalization, the RL reward, and the odds re-expansion. *Start here.*
2. [completion.md](completion.md) — did the work get done.
3. [efficiency.md](efficiency.md) — how early it got done.
4. [combined.md](combined.md) — the single scalar `K` and the reward contract.
5. [done-weight-rate.md](done-weight-rate.md) — the crisp "what fraction shipped."
6. [fairness.md](fairness.md) — workload balance across the team.
7. [difficulty-fingerprint.md](difficulty-fingerprint.md) — capacity pressure,
   forced triage, fairness gate, trajectory space.

## Notation used throughout

| symbol | meaning |
|---|---|
| `H` | grading horizon (sim-minutes; demo/crunch: 6780 = Fri 17:00) |
| `t0` | sim start (545 = Mon 09:05) |
| `S = H − t0` | wall span (6235 min) |
| `wᵢ` | priority weight of task `i` (P0=8, P1=4, P2=2, P3=1) |
| `Eᵢ` | effort hours of task `i` |
| `hᵢ` | true work-hours accrued on `i` by `H` |
| `pᵢ = min(1, hᵢ/Eᵢ)` | fractional progress |
| `dᵢ ∈ {0,1}` | task `i` done by `H` |
| `cᵢ` | completion time of task `i` (if done) |
| `α = 0.5` | done-bonus weight (`task_value.alpha`) |
| `γ = 0.5` | efficiency weight in the combined scalar (`task_value.gamma`) |

Only **authored** tasks (seed + external arrivals) carry weight; agent-created
tasks are weight 0 — you cannot mint value by filing busywork.

## Two reference scenarios

The docs use these two as worked examples (numbers from
`python -m sim.difficulty scenarios/<x>.json`):

| | `demo.json` | `crunch.json` |
|---|---|---|
| baseline combined `B` | 31.32 | 24.03 |
| OPT combined | 32.07 | 28.85 |
| **winnable** `OPT − B` | **0.75** | **4.82** |
| capacity utilization | 0.64 | 0.79 |
| forced triage | no | **yes** |
| completion band width | **0.0** | **6.0** |

`demo` is a *walkthrough* scenario (roomy, mostly an efficiency story);
`crunch` is the *training* scenario (over-subscribed, real triage, a wide
completion band). See [difficulty-fingerprint.md](difficulty-fingerprint.md)
for why the completion band width is the make-or-break number.
