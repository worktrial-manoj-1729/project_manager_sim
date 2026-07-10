# project_manager_sim

A discrete-event simulation of a project manager's week at a small SaaS
company (see `Problem.txt`) — built as an evaluation and RL-training
environment: an agent PM discovers blockers, resolves conflicts, allocates
people, and is graded **only on outcomes**.

## The one-paragraph pitch

The world is a deterministic discrete-event simulation: LLMs voice the
coworkers but can never mutate state ("rules decide, LLM narrates"). Every
task instance carries a deterministically computable band
**[no-PM baseline, OPT_ideal]** — the floor is an *honest* baseline where
the org self-organizes without a PM (volunteers pick up unowned work,
people order their own queues sensibly), the ceiling is the frictionless
optimum from exhaustive search — and the agent's score is its normalized
position in that band: **0 = added nothing a functioning unmanaged team
wouldn't produce, 1 = perfect coordination, negative = made things worse**.
Everything is replayable byte-for-byte from the event log.

## What the environment tests (and how agents can learn it)

- **Channel economics** — chat is instant but costs the recipient 20
  serialized focus-minutes; email is delayed (batch delivery) but tax-free;
  meetings block every attendee's calendar. None of this is in any tool
  description: agents learn it from evidence (reply latencies, completions
  slipping), from testimony (NPCs *feel* interruptions and complain in
  persona), or in-weights (the reward gradient across rollouts).
- **Epistemics** — three layers of knowing (truth / belief / tracker),
  honest-but-wrong estimates that mature on authored schedules, red
  herrings, and a tracker that only auto-updates on completions. The
  interrogation hack is dead: asking everyone on Monday yields honest
  *wrong* answers.
- **Execution** — external asks arrive as chat pings (push, wakes the
  agent mid-`advance_time`) or email (batched push); they are not on the
  board until the PM files the ticket, and can't be assigned before that.
  If the PM does nothing, the org's fallback volunteer eventually grabs the
  work — usually the wrong person, late.
- **Fairness, twice** — scenarios are *gated* for fairness (every scored
  ask must be completable after the PM could first know it), and the PM is
  *scored* on fairness (1 − σ of per-person hours-worked/hours-available).

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY
```

```sh
# dashboard hub: every run, live feeds, benchmark Pareto (per-task curves,
# model-family trajectories)
.venv/bin/python -m sim.server --hub 8742          # open http://127.0.0.1:8742

# run an agent
.venv/bin/python -m sim.harness scenarios/demo.json --probe scripted     # zero agent tokens
.venv/bin/python -m sim.harness scenarios/demo_2.json --probe llm --model claude-sonnet-5

# grade a run (deterministic, reproducible forever)
.venv/bin/python -m sim.eval runs/run-XXXXXXXX-XXXXXX

# replay any run with zero LLM calls
.venv/bin/python -m sim.replay runs/run-XXXXXXXX-XXXXXX [until_seq]
```

## Task authoring & calibration toolchain

```sh
.venv/bin/python -m sim.optimal scenarios/demo.json        # the OPT ceiling + assignment
.venv/bin/python -m sim.difficulty scenarios/demo.json     # hardness vector + fairness gate
.venv/bin/python -m sim.difficulty --stamp scenarios/*.json # embed band metadata into configs
.venv/bin/python -m sim.intent scenarios/demo_2.json       # WHICH decisions carry the band
```

- `sim.difficulty` prints the hardness **vector** (capacity utilization,
  winnable band, forced triage, assignment divergence, reaction ratios,
  log₁₀ trajectory classes) — all from config, no engine runs, no LLM.
- `sim.validate` (run at every engine init) **rejects unfair tasks**: any
  ask that can't physically be completed after the PM could first know of
  it scores luck, not skill.
- `sim.intent` plays deterministic policy ablations (oracle / late /
  no-spread / noisy) and reports each mechanism's **share of the band** — a
  mechanism with share ≈ 0 is not taught by that task.

## The scenario ladder

| scenario | character | band width | teaches (measured shares) |
|---|---|---|---|
| `demo_1` | roomy floor | 0.35 | sanity / saturation check |
| `demo` | timing task | 0.75 | preempt the fallback (0.999), channel discipline (0.15) |
| `demo_2` | delivery task, forced triage | 5.14 | timing (0.70), allocation (0.37), channel discipline (0.29) |

Names are deliberately neutral — hardness is *determined* by measured score
degradation, never asserted by a label.

## Invariants (the short version of DESIGN.md)

1. **LLMs produce text only** — no mutation channel, no physics, no scoring.
2. **The world is a deterministic function of (config + seed, actions).**
   Randomness must be keyed by (seed, decision id), never stream-ordered;
   draws that can be materialized pre-run are stamped into the config.
3. **Outcome-only rubrics** — no keyword checks, no behavior rules, no LLM
   judging. Noise punishes itself through capacity physics.
4. **Sim time never touches the wall clock** — `wall_now()` is the single
   permitted system-clock call site, telemetry only.
5. **Liveness** — no agent action and no NPC brain failure can kill the
   sim; unknown recipients bounce, LLM errors degrade to placeholder text.

## Layout

```
sim/
  sim_time.py    # minutes since Mon 00:00; work calendar; the only wall-clock exile
  events.py      # (sim_time, seq) min-heap — time only advances on pop
  world.py       # single clock owner; append-only log; tracker/belief/truth views
  tasks.py       # the scheduler: pure fold of (tasks, assignments, busy intervals)
  signals.py     # push/pull delivery semantics for every signal, one table
  npc.py         # pure appended-context agents: experience stream + one voice call
  engine.py      # pop → advance → dispatch; arrivals, fallbacks, beats, meetings
  tools.py       # the agent-facing tool surface (neutral descriptions, no economics)
  validate.py    # load-time scenario gate incl. the fairness gate
  rubric.py      # task_value: completion, efficiency, done-rate, workload fairness
  optimal.py     # OPT_ideal: exhaustive frictionless ceiling (labor pool = workers)
  eval.py        # band-normalized scorecard; odds form; always-numeric reward
  difficulty.py  # hardness vector + band stamping + trajectory-space estimate
  intent.py      # mechanism-share audit: which decisions carry the band
  harness.py     # scripted + LLM probes; cost/turn accounting; meta stamping
  replay.py      # world_at(seq) = fold(scenario, events[:seq+1])
  server.py      # observer-only web UI: engine / --watch / --hub modes
web/index.html   # live feeds, rewind, trajectory lanes, benchmark Pareto
scenarios/       # task configs with stamped band metadata
rubrics/         # scoring configs, copied into each run dir
```

Every run dir is self-contained and re-gradeable forever: `scenario.json`,
`rubric.json`, `events.jsonl` (source of truth), `llm.jsonl` (full traces),
`meta.json`, `scorecard.json`.

Full design rationale: `DESIGN.md`.
