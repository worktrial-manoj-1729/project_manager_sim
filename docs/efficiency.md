# Efficiency

*How early did the work get done?* — the timing metric that makes noise
self-punishing.

## Definition

For each authored task `i`, done by the horizon at completion time `cᵢ`:

```
EFFICIENCY = Σᵢ  wᵢ · dᵢ · (H − cᵢ) / S            S = H − t0
```

(`sim/rubric.py::task_value`). Only **finished** tasks contribute (`dᵢ`), and
each contributes its weight scaled by how much of the week was still left when
it landed. A P0 finished Wednesday noon is worth far more than the same P0
finished Friday 16:59; an unfinished task contributes 0.

## Why it exists

Completion alone is indifferent between "done early" and "done at the buzzer."
Efficiency is what prices **friction** without any explicit penalty term:

- **Meetings** consume every attendee's working block → their tasks finish later
  → efficiency drops.
- **Chat interruptions** cost the recipient ~20 serialized focus-minutes each,
  in working hours → later completions → lower efficiency.
- **Late allocation** (filing an arrival hours after it landed) pushes `cᵢ`
  right → lower efficiency.

So spamming, over-meeting, and dawdling are all disincentivized *physically* —
the capacity they burn shows up as later `cᵢ`. There are **no keyword checks,
no message-count penalties, no LLM judge**; the schedule does the accounting.

## Empirical bite

A controlled check (identical assignments, one clean, one with 40 spam pings)
separates cleanly on efficiency: ~99% efficiency clean vs ~32% under spam, even
where completion is similar. Efficiency is the metric that notices you wasted
the team's attention.

## Worked example

| | `demo.json` | `crunch.json` |
|---|---|---|
| efficiency `[B, OPT]` | `[17.95, 20.20]` | `[13.53, 15.97]` |

In `demo` (zero completion band) efficiency is essentially the *only* winnable
signal — which is exactly why `demo` is a thin training target: the band is
narrow (2.25) and earliness is a lower-variance quantity than completion.

## Relationship to completion

Efficiency is gated by completion (`dᵢ` appears in both). You cannot trade
completion for efficiency: an unfinished task scores 0 on *both*. Efficiency
only re-ranks *among* the things you finished, rewarding finishing the
heavy ones sooner.
