# Done-weight rate

*What weighted fraction of the week's work actually shipped?* — the crisp,
human-readable headline (not the RL reward).

## Definition

```
done_weight_rate = Σ_{i: dᵢ}  wᵢ  /  Σᵢ  wᵢ
```

(`sim/rubric.py::task_value`). The weight of all **finished** authored tasks
over the weight of **all** authored tasks. A single number in `[0, 1]` that
reads like a completion percentage but is priority-weighted — shipping a P0 and
dropping a P2 scores far higher than the reverse.

## Why it's separate from completion

`COMPLETION` gives partial credit (a task at 60% contributes 0.6·something).
`done_weight_rate` is strictly **binary per task** — it answers "did it ship,
yes or no," weighted by importance. That makes it the natural slide number:
"the agent shipped 79% of the week's weighted work" is instantly legible in a
way that "completion 30.2 of 35.3" is not.

It is reported on the same `[B, OPT]` band as the other metrics, so you can also
say "of the shippable-beyond-baseline work, the agent shipped X%."

## The triage signal

`done_weight_rate` is where **forced triage** becomes visible. When a scenario
is over-subscribed, even `OPT` cannot ship everything → `OPT_done_weight_rate <
1`. That gap is the definition of `forced_triage` in the difficulty fingerprint.

| | `demo.json` | `crunch.json` |
|---|---|---|
| `B` done-weight-rate | 1.00 | 0.579 |
| `OPT` done-weight-rate | 1.00 | 0.789 |
| forced triage? | no (`OPT` ships all) | **yes** (`OPT` ships 79%) |

In `crunch`, even perfect play ships only 79% of weighted work — one P0
(`data-backfill`, a hard dependency-chained task) is sacrificed by *everyone*.
That is the intended lesson: under real scarcity a PM chooses what slips; the
metric proves the choice was unavoidable, not a failure, by pricing the agent
against `OPT`, not against 1.0.
