# The `[B, OPT]` band

*How every metric is anchored, why the band is also the difficulty knob, and
how the RL reward is derived from it.*

## 1. Motivation

An absolute score ("the agent finished 30 weighted points of work") is
uninterpretable: 30 might be everything winnable or a fraction of it, and much
of it might happen with no PM at all. We need a scale on which **0 means the
agent added nothing** and **1 means it did everything achievable**. That scale
is the band `[B, OPT]`, computed per scenario, per metric.

## 2. The two anchors

Both anchors are deterministic functions of the scenario file — no engine run,
no LLM, no agent.

**`B` — the no-PM baseline (floor).** The same world with no project manager,
graded at the horizon. Crucially this is *not* a strawman "do nothing" world:
a real team self-organizes even without a PM. We model that with an **org
fallback** — every unowned external ask is eventually picked up by its natural
owner at an authored time `fallback.at` (`sim/difficulty.py::baseline_value`,
mirrored at runtime by the `org_pickup` event). The baseline therefore
*already captures* everything a team stumbles into on its own. The PM earns
credit only for coordinating **better** than that: reallocating under
scarcity, and acting *early enough* that work lands before the org's late,
clumsy pickup.

**`OPT` — the frictionless optimum (ceiling).** The best achievable value under
an idealized relaxation of the same scenario (`sim/optimal.py`):

- communication is free and instantaneous (full information at `t0`);
- the skill multiplier is the *fastest eligible worker's*, clamped `≥ 1`;
- no meeting / interruption overhead;
- decisions cost nothing.

What remains are the **hard constraints** — arrivals, precedence, effort,
one-task-at-a-time per person, working calendars, the horizon. `OPT` is
computed by exact search over worker→task assignments × orderings, each scored
with the *same* scheduler the world runs (frictions zeroed). Because it relaxes
only frictions, **no real policy can beat it** → `OPT` is a true upper bound.

> **Worker pool.** Only NPCs with `worker ≠ false` are assignable. Stakeholders
> you report to (a VP) are `worker: false`: OPT never uses them as labor and
> the agent cannot either (`agent_assign_task` rejects it). Otherwise OPT would
> count work no intended policy should do, deflating every normalized score.

> **Skill-clamp (upper-bound safety).** `OPT` uses `Eᵢ / max(1, best
> multiplier)`. If a real worker had a skill factor > 1, a naive `1×` ideal
> could be *beaten* by reality (normalized > 1). Clamping the ideal to the
> fastest worker keeps `OPT` a valid ceiling the moment skills are authored.

## 3. Normalization — the reading

For any metric `M` with agent value `M_a`, baseline `B`, optimum `O`:

```
available = O − B          # how much a PM can even move this metric
delta     = M_a − B        # how much this agent actually moved it
normalized = delta / available          (∈ (−∞, 1], undefined if available≈0)
regret    = available − delta
```

- `normalized = 0` → no better than no PM.
- `normalized = 1` → reached the ceiling.
- `normalized < 0` → the agent made things **worse than its own absence**
  (e.g. burned capacity in meetings, or reassigned work to a slower path).

## 4. The RL reward (`sim/eval.py`)

The headline scalar is the **combined** metric, normalized:

```
winnable = OPT_combined − B_combined
score    = (agent_combined − B_combined) / winnable      # None if winnable ≈ 0
reward   = score          if score is not None
           else (agent_combined − B_combined)            # always numeric
```

`score` is deliberately `None` for a **degenerate** scenario (nothing winnable)
so tooling can flag it — but `reward` is **always a number**, so a training loop
is never handed a `None`. Degenerate instances should be rejected by the
difficulty gate *before* training; the raw delta is a safe fallback if one slips
through. The result also carries `"degenerate": bool`.

### 4.1 Odds re-expansion (top-end resolution)

`OPT` is a *frictionless* relaxation no real policy reaches, so `normalized`
saturates below 1 and **compresses** differences among strong policies (0.90 vs
0.97 looks tiny but is a large real gap in remaining regret). We also report the
**odds form**:

```
score_odds = delta / (OPT − agent) = N / (1 − N)      where N = normalized
```

`score_odds` is "gain captured per unit of regret still on the table." It is
monotone in `normalized` (so it never reorders policies) but stretches the top
end back out — useful when ranking rollouts that all sit near the ceiling.

## 5. The band *is* the difficulty knob

Two scenarios can share the plumbing yet differ entirely in trainability, and
the band tells you which:

| | `demo.json` | `crunch.json` |
|---|---|---|
| `B` combined | 31.32 | 24.03 |
| `OPT` combined | 32.07 | 28.85 |
| **winnable** | **0.75** | **4.82** |
| completion band `[B, OPT]` | `[38.0, 38.0]` → **width 0** | `[29.3, 35.3]` → **width 6.0** |

`demo` has a **zero-width completion band**: the self-organizing team finishes
exactly what OPT finishes, so the only thing left to win is *earliness*
(efficiency) — a thin, low-variance signal. `crunch` opens a 6-point completion
band by (a) over-subscribing the week so triage is forced and (b) delaying the
org's fallback pickups so a proactive PM lands two P1s the no-PM org misses.

**For GRPO this is the whole game.** Group-relative advantages need rollouts to
*spread* across `[B, OPT]`. A wide band that is only reachable through a graded
sequence of decisions produces spread; a zero-width or single-action band
produces a cliff (every rollout at 0 or 1 → advantage ≈ 0 → no gradient). The
band width is necessary; the *shape* between the anchors (capacity pressure,
forced triage, distributed value) is what makes it sufficient — see
[difficulty-fingerprint.md](difficulty-fingerprint.md).

## 6. Properties (why this resists gaming)

- **Causality gate.** Credit requires beating `B`. Anything that happens without
  a PM earns 0 — no points for "surfacing" work the org would have grabbed anyway.
- **Bounded above.** `agent ≤ OPT` by construction → `normalized ≤ 1`.
- **Activity is not rewarded.** Only true task state at `H` enters the score;
  messages, meetings, tracker edits earn nothing directly. They matter only
  through their *physical* consequences (capacity, timing) on the metrics.
- **Replayable forever.** Both anchors and the agent value are pure functions of
  logged state, recomputable from the run dir with zero model calls.
