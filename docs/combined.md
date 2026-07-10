# Combined & the reward contract

*The single scalar `K` used for ranking and RL, and how it becomes a reward.*

## Definition

```
COMBINED  K = (COMPLETION + γ·EFFICIENCY) / (1 + γ)         γ = 0.5
```

(`sim/rubric.py::task_value`). Completion is the dominant term; efficiency is a
`γ`-weighted tie-breaker among policies that finish similar work. With
`γ = 0.5`, efficiency carries `0.5/1.5 = ⅓` of the weight — enough to price
friction and earliness, not so much that a fast-but-incomplete policy outranks a
complete one.

## Why one scalar

GRPO needs a single per-rollout number to compute group-relative advantages.
`K` is that number. It is:

- **monotone in doing the right work** (completion dominates),
- **sensitive to friction** (efficiency term),
- **deterministic** (pure function of true task state at `H`),
- **bounded above by OPT** (see [scoring-band.md](scoring-band.md)).

## From `K` to reward

`K` is an absolute quantity; the reward is its position in the band `[B, OPT]`:

```
winnable = OPT_K − B_K
score    = (K_agent − B_K) / winnable        # None iff winnable ≤ 0.001 (degenerate)
reward   = score if score is not None else (K_agent − B_K)     # ALWAYS numeric
```

Contract for a training loop:

| field | meaning | guarantee |
|---|---|---|
| `reward` | the RL scalar | **always a float** |
| `score` | normalized `K` in band | float, or `None` if degenerate |
| `degenerate` | `score is None` | bool flag for the batch filter |
| `score_odds` | `score / (1 − score)` | top-end re-expansion (may be `None`) |
| `score_raw_delta` | `K_agent − B_K` | unnormalized fallback |

Degenerate scenarios (nothing winnable) should be dropped by the difficulty
gate before they reach training; `reward` stays numeric regardless so nothing
downstream crashes on a `None`.

## Worked example

| | `demo.json` | `crunch.json` |
|---|---|---|
| `B_K` | 31.32 | 24.03 |
| `OPT_K` | 32.07 | 28.85 |
| `winnable` | 0.75 | 4.82 |
| reward at OPT | 1.00 | 1.00 |
| reward at baseline | 0.00 | 0.00 |

Same reward endpoints, but `crunch`'s band is ~6× wider and (critically) built
from completion headroom, so intermediate policies land *between* 0 and 1
instead of snapping to an endpoint. That spread is what carries the training
signal.

## Tuning knobs

- `α` (done bonus, default 0.5) — how much shipping beats near-shipping.
- `γ` (efficiency weight, default 0.5) — how much timing matters vs raw completion.
- `priority_weights` — the value ratio across P0/P1/P2/P3.

All live in the rubric file (`rubrics/demo.json`), versioned and copied into
each run dir so historical runs stay re-gradeable.
