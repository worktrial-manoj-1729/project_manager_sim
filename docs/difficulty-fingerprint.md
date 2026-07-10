# Difficulty fingerprint

*How hard — and how trainable — is a scenario, computed from the config alone?*

Difficulty is a **vector**, not a scalar. Raw event counts are a poor proxy
(heartbeats dominate the queue and add zero difficulty). Every field below is
deterministic and needs **zero engine runs** — the no-PM baseline is itself
analytic (`python -m sim.difficulty scenarios/<x>.json`). These fields are the
validity/difficulty gates a scenario generator will read; empirical hardness
(the probe-ladder score) remains the ground truth they are calibrated against.

## The fields

| field | meaning | trainable range |
|---|---|---|
| `winnable_combined` | `OPT − B` on the combined scalar — how much a PM can move at all | `> ~0.5`, bigger better |
| `winnable_completion` | `OPT − B` on completion — **completion headroom** | **`> 0`** (else efficiency-only) |
| `capacity_utilization` | remaining effort / worker-hours to horizon | ~0.3 roomy, ~0.8 tight |
| `opt_done_weight_rate` | weighted fraction even OPT can ship | `< 1` ⇒ triage forced |
| `forced_triage` | `opt_done_weight_rate < 1` | **True** for a real triage task |
| `slack_minutes_at_opt` | horizon − last completion under OPT | small / negative = tight |
| `assignment_divergence` | tasks OPT moves off their authored owner | `> 0` = allocation matters |
| `min_reaction_ratio` / `fair` | fairness gate (below) | `fair == True` required |
| `log10_trajectory_classes` | log₁₀ of outcome-distinguishable plans | bigger = richer decision space |

Plus the raw band anchors (`baseline_raw`, `opt_raw`) — the `[B, OPT]` per
metric, stampable into the scenario via `python -m sim.difficulty --stamp`.

## Capacity utilization — the pressure knob

```
capacity_utilization = Σ remaining_effort_hours / (n_workers · work_hours(t0→H))
```

Only the **worker** pool counts (stakeholders excluded). Below ~0.5 the week is
roomy and everything gets done either way → the band collapses. Around 0.8 the
week binds and allocation decisions start to cost real completion.

## Forced triage — the difference between scheduling and choosing

If `opt_done_weight_rate < 1`, **even perfect play must sacrifice work**. That
turns the task from "schedule everything sensibly" into "choose what slips" —
the decision we actually want to train. `forced_triage` is the boolean gate.

## The fairness / reaction-ratio gate (`fair`)

A *scored* ask must be physically completable from the moment the PM could
first **know** it. For each arrival and each belief-confession:

```
reaction_ratio = work_calendar_window(first_knowable → H) / effort_needed
```

`first_knowable` is the announce time (chat: instant; email: next batch tick).
`reaction_ratio < 1` means the ask is impossible after discovery — scoring luck,
not skill — so `validate_scenario` **rejects** it. `1.0–1.3` is a tight, fair
squeeze. `fair = (min_reaction_ratio ≥ 1)`. This is what keeps a hard scenario
*hard-but-fair* rather than rigged.

## Trajectory space — why the band isn't a cliff

The raw action space is unbounded (any text, any minute, any channel), but the
**score is a projection**: it only sees *who* ends up owning each task
(`workers^n_live`) × *when* each ask starts being worked (a minute in
`[first_knowable, H]`) × the noise load per person. `log10_trajectory_classes =
log₁₀(who × when)` estimates the score-relevant decision space that the band
`[B, OPT]` is the image of. A wide band spanned by *many* distinguishable plans
is a ramp (rollouts spread → GRPO gradient); a wide band reachable by *one*
action is a cliff (rollouts snap to endpoints → no gradient).

## Worked comparison

```
                        demo.json     crunch.json
winnable_combined         0.75          4.82
winnable_completion       0.0           6.0        ← the decisive difference
capacity_utilization      0.64          0.79
opt_done_weight_rate      1.00          0.789
forced_triage             False         True
slack_minutes_at_opt      1645          −5
min_reaction_ratio        3.88          1.72   (both fair)
log10_trajectory_classes  12.2          12.2
```

Both have the same trajectory-space size, yet `demo` is a poor training target
and `crunch` is a good one. The tell is `winnable_completion`: `demo`'s is **0**
— the self-organizing team finishes exactly what OPT finishes, so the only
signal is efficiency (band width 0.75, low variance). `crunch` opens a 6-point
**completion** band by (1) over-subscribing the week (`capacity 0.79`,
`forced_triage True`) and (2) delaying the org's fallback pickups so a proactive
PM lands two P1s the no-PM org misses — while staying fair
(`min_reaction 1.72 ≥ 1`). That is the recipe for turning a cliff into a ramp.

## Suggested gate (one-liner)

```
trainable = (forced_triage
             and fair
             and winnable_completion > 0
             and winnable_combined  > 0.5)
```

Necessary conditions for a scenario to carry a usable GRPO signal; the
probe-ladder (null < scripted < weak-LLM < strong-LLM must *separate*) is the
empirical confirmation.
