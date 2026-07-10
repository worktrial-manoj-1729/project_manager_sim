# Completion

*Did the work get done?* — the primary outcome metric.

## Definition

For each authored task `i` with priority weight `wᵢ`, fractional progress
`pᵢ = min(1, hᵢ/Eᵢ)`, and done flag `dᵢ`:

```
COMPLETION = Σᵢ  wᵢ · (pᵢ + α·dᵢ) / (1 + α)          α = 0.5
```

(`sim/rubric.py::task_value`). Two design choices are load-bearing:

- **Dense, not binary.** A half-finished P0 is worth more than an untouched one:
  the `pᵢ` term gives partial credit for real hours accrued, so the gradient is
  smooth rather than a step at 100%.
- **A finishing bonus.** The `α·dᵢ` term rewards actually *shipping*. With
  `α = 0.5`, a fully-done task scores `wᵢ·(1 + 0.5)/1.5 = wᵢ`; a task at 99%
  but unfinished scores `wᵢ·0.99/1.5 ≈ 0.66·wᵢ`. Crossing the finish line is
  worth ~50% more than the last sliver of progress — this discourages leaving
  everything at 90%.

`pᵢ` and `dᵢ` are read from **ground truth** (the analytic scheduler), never
from what anyone *reports*. Gaming the tracker does nothing.

## Weights

Priority weights `{P0:8, P1:4, P2:2, P3:1}` (geometric, factor 2) make a P0
worth 4× a P2. This is what prices Dave's rate-limiting herring: it is a P2
(weight 2), so spending scarce capacity on it instead of a P1 (weight 4) is a
measurable loss.

## Worked example (`crunch.json`)

Completion band: `[B, OPT] = [29.29, 35.29]`, width **6.0**. The 6.0 is exactly
two P1s (questionnaire + readiness, weight 4 each, done bonus included:
`4·1.5/1.5 = 4` each, minus their partial baseline credit) that the no-PM org
picks up too late to finish but a proactive PM lands. In `demo.json` the band
is `[38.0, 38.0]` — width 0, because the self-organizing team finishes
everything on its own.

## What moves it

- **Filing + assigning unowned arrivals** before the org's late fallback → their
  `dᵢ` flips 0→1.
- **Reallocating under scarcity** so high-weight work finishes instead of
  low-weight work (the triage decision).
- **Protecting capacity**: every chat received in work hours costs the recipient
  ~20 serialized focus-minutes; enough interruptions push a completion past the
  horizon and `dᵢ` drops back to 0. Completion self-punishes noise with no
  explicit penalty term.

## Anti-hacking

- Agent-created tasks are skipped (`source == "agent"`) → cannot inflate the sum.
- Tracking-only items (no `effort_hours`) carry no value.
- Progress is the scheduler's truth; reported/belief status is irrelevant to score.
