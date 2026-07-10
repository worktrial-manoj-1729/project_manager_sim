# Design: systems, state, and sim-time flow

This document pins down the semantics the problem statement asks to be legible:
*what advances synchronously with an agent action, what advances asynchronously
in the background, how state is owned and mutated, and how time flows.*

## 1. The three classes of state change

Everything that happens in the sim falls into exactly one of three classes:

| Class | Examples | When it executes | Clock behavior |
|---|---|---|---|
| **Synchronous** (agent-caused) | `say`; later: edit task, book meeting | Immediately, inside the agent's call frame | `clock += 1` (fixed action cost); the world mutates *at that instant* |
| **Asynchronous** (queue-driven) | NPC replies, heartbeats, scenario beats; later: meeting start/end | Only when the agent yields time (`wait`/`run`) and the heap pops | Clock **jumps** to `event.time`; between pops, nothing executes |
| **Derived** (analytic) | Task progress, availability, projected completion dates | **Never** — pure functions of `(scenario config, clock, logged mutations)` | No events at all; evaluating at a later clock simply yields a later value |

The **derived** class is the load-bearing trick. Task hours "accrue overnight"
without anything running overnight: progress is

```
progress(t) = done_hours + work_minutes_between(task_start, min(t, done_at)) / 60
```

computed on demand from the work calendar (Mon–Fri 09:00–17:30, one task at a
time per assignee, blocked tasks accrue nothing). Because nothing executes,
there is no sync/async question to answer for it, the async event set stays
tiny, and rewind/replay get it for free — ask for the task board at any past
time and the same function answers.

## 2. The two mutation frames

World state is owned by a single `World` object and mutates in **exactly two
stack frames**:

1. `Engine.agent_say(...)` — and future agent actions (synchronous class)
2. `Engine._dispatch(event)` — the queue pop handler (asynchronous class)

Both funnel every mutation through `world.record(...)`, which appends to the
in-memory log **and** flushes to `runs/<id>/events.jsonl`. Consequences:

- The log **is** the complete mutation history: `state = fold(scenario, events)`.
- Replay (`sim/replay.py`) reconstructs any past state with zero LLM calls,
  because the only nondeterminism (LLM reply text) is captured in the log.
- The UI's rewind slider is just "render the fold of a log prefix."

## 3. The two places time advances

1. **Agent action cost**: `clock += 1` sim-minute per action. Prevents "do 500
   things at 09:00:00" and makes agent activity occupy real (sim) time.
2. **Queue pop**: `clock = event.time`. Discrete-event style — time jumps to
   the next scheduled thing, never crawls.

Nothing else moves the clock. In particular, **wall-clock never leaks in**:
when an `npc_respond` event pops, the LLM call happens *inside that frozen
instant*. Sarah "replies at 09:42" whether inference took 2 or 30 real seconds.
The agent can think indefinitely between actions; the world waits.

## 4. Control flow: who advances time

The agent (or the operator driving it) **owns** time advancement. The sim never
advances on its own — there is no background thread, no timer. `wait` pops one
event; `run` pops until a message for the agent arrives. This is what makes the
loop testable: a scripted sequence of agent actions produces a deterministic
event timeline (seeded per-NPC RNGs, seq-numbered heap ties).

## 5. Event kinds (current)

| Kind | Class | Scheduled by | On fire |
|---|---|---|---|
| `npc_respond` | async | latency rule at message delivery (seeded RNG + working hours) | one LLM call → reply message |
| `npc_wakeup` | async | self-rescheduling heartbeat (~2h, working hours only) | reschedules itself; later: initiative hook |
| `beat` | async | scenario config (`beats[]`) | condition = deterministic world-state IF (task done? — existence checks, never keywords) deciding *whether/what substance* fires; **intent = motivation, never phrasing and never unverified facts**. An intent may assert only facts the condition just verified or the NPC's injected views contain; the LLM supplies human behavior from persona + visible history. Engine-if and NPC-eyes read the same fold → narration can't contradict the decision. |
| `task_arrival` | async | scenario config (`task_arrivals[]`) — OOD tasks caused by externalities | `task_added` mutation; optionally an NPC announces it (LLM ping) |

Beats and task arrivals are how **authored pressure and externality** live in
the same queue as organic behavior: the scenario file is data (personas +
tasks + beats + arrivals + seed), not code, which is how this scales to many
scenarios without prompt spaghetti.

### Task origins

Tasks enter the world three ways, all landing in one dynamic list where every
addition is a logged `task_added` mutation (so rewind/replay reconstruct the
board at any time):

1. **Seed** — `project.tasks` in the scenario; arrival = sim start.
2. **External (OOD)** — `task_arrivals[]`: scheduled, externally-caused work
   (an incident, a legal request) that lands mid-week, optionally announced
   by an NPC. These are out-of-distribution on purpose: the agent's plan has
   to survive them.
3. **Agent** — the agent adds tasks for tracking (a synchronous action).

The scheduler is a deterministic greedy: per assignee, one task at a time,
working hours only; a task starts when its assignee is free AND it has
arrived AND its blockers are done; when free, the assignee picks urgent-first
then creation order. Non-preemptive: an urgent arrival jumps the queue but
never interrupts in-flight work. Tasks with no assignee/effort are
"tracking" items — visible, never scheduled.

## 5a. The tool surface (`sim/tools.py`)

One registry — name + description + JSON schema + handler — drives the web
UI/REST API today and becomes the Claude tool list for the agent harness
verbatim. All handlers are synchronous agent actions (+1 sim-min cost):

| Tool | Consequence |
|---|---|
| `send_chat` / `send_email` | message delivery; reply scheduled by latency rule (email is hours, chat is minutes; meetings defer replies) |
| `add_task` / `assign_task` / `update_tracker_note` | logged task mutations — reassignment genuinely changes `compute_schedule` |
| `schedule_meeting` | books attendees; **costs their working capacity** (task projections slip); at meeting end one LLM call writes a transcript, which becomes shared context for every attendee — the NPC↔NPC information channel |
| `write_doc` | shared docs enter recipients' context |
| `view_tasks` / `view_inbox` | read-only observations |
| `advance_time` / `wait_for_reply` | the agent's **yield primitives** (below) |

**Why time-yielding is a tool:** the agent owns time, so it needs a way to
express "I choose to do nothing until X" — waiting instead of meddling is a
real PM decision we want to measure. It is not an escape hatch: beats,
arrivals, and escalations fire *during* the jump, and evaluation grades the
world at a fixed horizon, so skipping time forfeits agency rather than
dodging judgment. `wait_for_reply` gives up at a ~1.5-workday horizon so
waiting can never silently burn the week. (Alternative considered:
harness-driven turns, tau-bench style — rejected because pacing judgment is
part of the skill under test.)

**Meetings and secrecy:** the transcript prompt lists each attendee's persona
and private knowledge but instructs that characters only *say aloud* what
they would realistically reveal in front of the specific people in the room —
private knowledge steers behavior without being narrated.

## 5b. Operator time controls

The sim never advances itself, but the *operator* can advance it:
`advance_until(target)` fires every event up to the target, then jumps the
clock there — even through empty space (nights, weekends). The UI's
+1h/+4h/+1day buttons and Play mode (repeated advances at a fixed sim-speed)
are just this call in a loop; engine semantics are unchanged, so a full
7-day run is a button press. The jump is logged (`clock_advanced`) so replay
reproduces the final clock.

## 6. NPC execution model

**The NPC contract** (the whole model in three lines):

| Capability | Mechanism | Bound |
|---|---|---|
| **They do their work** | the deterministic scheduler accrues their task hours through the work calendar — no LLM involved | capacity is physics: meetings, interruptions, PTO all subtract |
| **They read** | engine-injected derived views (own live tasks, timestamped 1:1 history, attended transcripts, shared docs, live meeting minutes) — reads are unconditional, never optional tool calls | each NPC sees only ITS slice; no global board, no others' workloads |
| **They speak, and only speak** | replies (latency rules), pings (beat arms), meeting turns (speak-or-PASS agent decision) — the LLM chooses words, and in meetings *whether* to say them | speech NEVER mutates world state; every write action belongs to the PM's tools alone |

Consequences: information asymmetry is structural (the optimization can only
happen in the PM), determinism survives (text is the only sampled thing, and
it's captured in the log), and the benchmark measures the agent — never the
NPC model.

An NPC is not a process. It is persona (static) + knowledge (config) + rules +
one LLM call:

- **WHEN** an NPC acts is decided by deterministic rules (latency sampling from
  its seeded RNG stream, working-hours availability, heartbeat cadence).
- **WHAT** it says is decided by a single LLM call *at event-fire time*, seeing
  the persona, private knowledge, chat history — and a **live read-only
  self-view** injected by the engine: its own tasks' true progress and
  projections, its meetings, docs, transcripts. NPCs get "read tools" by
  prompt injection, never by tool-calling loops — the reads are derived
  views computed deterministically, the LLM stays a single call, and there
  is still no write channel. (This fixes the stale-knowledge failure where
  Sarah kept claiming 55% on Tuesday while ground truth had her at 94%.)

Between events an NPC literally does not exist — no state can drift. All of an
NPC's future behavior is visible in the heap (`pending` in the UI).

**Reasoning stays in the PM — by information asymmetry, not prompt rules.**
NPCs see only their own slice (their tasks, their meetings, their thread);
they emit local facts and human texture ("I'm on the incident till Wed —
if I take T, something slips; your call"). They cannot answer about others'
workloads (not in their views), cannot compute global schedules (no global
board), and cannot execute anything (no write channel). The PM alone holds
global observability (`view_tasks`) and allocation authority, so the
optimization can only happen there — outsourcing it to an NPC means
consulting someone strictly less informed than yourself, voiced by a cheap
model (NPC brains default to Haiku via `SIM_NPC_MODEL`; the evaluated
agent's model is a separate knob). This is eval integrity: the benchmark
measures the PM's reasoning, never the NPCs'.

## 7. The determinism boundary

Every subsystem is explicitly deterministic or LLM-based. **The LLM produces
text only — it has no channel to mutate the world.** All world mutations come
from scenario config, engine rules, or agent tools.

**The randomness invariant.** The world must be a deterministic function of
(config + seed, action sequence). Randomness is allowed anywhere provided:
(a) every draw is keyed by (seed, stable decision id) — never by
shared-stream order, where one extra agent action shifts every later draw
and reintroduces rollout variance in a determinism costume; and (b) any draw
that CAN be materialized before the run IS stamped into the config, so the
gates (fairness, band, intent audit) can see and reject what they need to.
Endogenous draws — weights that depend on the trajectory, e.g. a fallback
volunteer chosen ∝ available hours at pickup time — stay lazy but keyed;
they are legitimate dynamics, and any lever they hand the agent (shaping
loads to shape who volunteers) should earn its own intent-audit line before
being trusted.

| Subsystem | Deterministic? | Notes |
|---|---|---|
| Clock, event queue, ordering | ✅ | seq-tiebroken heap |
| Response latencies, heartbeats | ✅ | per-NPC seeded RNG streams |
| Task scheduling & progress | ✅ | pure function (work calendar + busy intervals) |
| External task arrivals | ✅ | **authored config**, fixed times/efforts — not LLM-generated |
| Beats (fire/skip decision) | ✅ | condition checked against world truth |
| Meeting capacity cost | ✅ | interval subtraction |
| NPC message/ping text | 🤖 LLM | text only; captured in log → replay is LLM-free |
| Meeting transcript text | 🤖 LLM | text only; same capture |

**Bounds on the levers** (`sim/validate.py`): even the deterministic mutation
paths are gated so no scenario author, procedural generator, or agent can
push the world outside a reasonable envelope — validated at load
(`validate_scenario`: task counts, effort ranges, dangling references, events
inside the run window) and enforced at runtime (`check_task_bounds` on every
task addition; meeting-duration cap). Defaults: ≤50 tasks, effort 0.5–80h,
≤10 external arrivals, ≤4h meetings — overridable per scenario via a
`bounds` block. If a future version ever lets an LLM propose world changes
(e.g. generated arrivals), those proposals must pass the same gates.

## 7b. Liveness invariants (no deadlock, no staleness)

Tested in-repo (broken-brain clients, 14-day advances, zero-NPC configs):

1. **The queue is never empty.** Every `npc_wakeup` reschedules itself at a
   strictly-future time; scenarios must have ≥1 NPC (validated at load) —
   the heartbeat chain is the sim's pilot light.
2. **No zero-delay cycles.** Every scheduled event lands strictly after its
   scheduling instant; `advance_until` always reaches its target (a high
   runaway cap logs a warning instead of silently stopping short).
3. **100% causal clock.** Invariant: *no undispatched event exists at or
   before the clock, at any observable instant.* Queue pops advance time in
   event order by construction; action-cost advances (+1 min) immediately
   drain any events that became due, so the clock never passes over a
   pending event — if the agent's 3rd action of a burst crosses a reply's
   due-time, the reply fires right then and later actions happen in a world
   where it exists. All scheduled events are strictly future (drain
   terminates). A monotonicity guard in `step()` remains as
   belt-and-suspenders and logs loudly if it ever engages.
4. **LLM failures never stall the world.** Any NPC-brain exception —
   API errors, thinking-only/truncated responses, bugs — degrades to
   placeholder text; the event still completes and the queue keeps moving.
   Agent-side API failures end the agent's week gracefully: the run
   advances to horizon and is still scored.
5. **Waiting is bounded.** `wait_for_reply` gives up at a ~1.5-workday
   horizon with `replied: false` rather than silently consuming the week.

## 8. Observer / driver separation

The web UI is **read-only observability**: clock, rewind, feed, task board,
timeline, trajectory. It never mutates the sim. Driving happens through the
CLI, the tool API (`POST /api/tool`), or the agent harness — one writer,
many watchers.

## 9. Evaluation (`sim/eval.py`, rubrics in `rubrics/`)

**v2 — two outcome metrics, and nothing else.** Over **authored tasks only**
(agent-created tasks carry zero weight):

    COMPLETION = Σ w · (progress + α·done)/(1+α)        did the work get done
    EFFICIENCY = Σ w · done · (horizon − done_at)/span   how early it got done
    COMBINED   = (completion + γ·efficiency)/(1+γ)       single scalar for RL

    each reported as a ladder: null baseline ≤ agent ≤ OPT_ideal

The efficiency metric makes noise self-punishing through physics: meetings
and chat interruptions (10 focus-min tax per chat received) consume capacity
→ completions land later → efficiency drops. Verified: identical assignments
with 40 spam pings capture 50% completion / 32% efficiency vs 100%/99% clean
— **no checks, no penalties, no keywords, no LLM anywhere in scoring**; the
rubric (`rubrics/demo.json`) contains only the two metrics' parameters and
the horizon.

The ladder per run: **baseline ≤ agent ≤ OPT_ideal** where `OPT_ideal`
(`sim/optimal.py`) is the frictionless optimum — communication free, skills
1×, no interruptions; only arrivals, precedence, effort, serial capacity,
calendars remain. A pure function of the scenario file (computed by exact
search reusing `compute_schedule`), it is a stable ceiling for normalization
(`% captured`), regret, and generator validity gating (reject instances
where OPT ≈ baseline: nothing to win).

Rubrics are separate versioned artifacts in `rubrics/`, referenced by the
scenario and copied into each run dir (runs stay re-gradeable forever).
Informational behavior (discovery, stakeholder communication, tracker
hygiene) is deliberately **unscored** — communication is instrumental, and
its value flows through allocation quality and capacity into the two
metrics. Behavioral readouts live in `navigation.json` (tool mix, channel
mix, contact patterns), which is telemetry, never reward.

### The original check-based design (v1, superseded)

**Score the world, not the agent.** Ground truth lives in the scenario's
`evaluation` block: weighted checks (world-state predicates) plus penalties,
graded at a fixed horizon (`python -m sim.eval runs/<id>`).

**Delta over a null-agent baseline.** The same scenario is re-run with an
agent that does nothing (stub LLM — NPC text never mutates world state, so
the baseline is deterministic and free; it's persisted under
`runs/<id>/baseline/` for inspection). Scoring per check:

| agent | baseline | points |
|---|---|---|
| pass | fail | **+weight** (agent-caused improvement) |
| pass | pass | 0 — *happens anyway; not yours* |
| fail | fail | 0 |
| fail | pass | **−weight** (agent-caused regression) |

The fixed horizon means time-skipping can't dodge grading, and the baseline
delta is the causality gate: credit requires the outcome to NOT occur
without the agent.

**Check types (all deterministic):** `task_done_by`, `task_has_owner`,
`task_updated_by_agent` (tracker hygiene), `agent_told` (stakeholder
informed in time), `discovered` (the agent surfaced a hidden fact).
Message-content checks use keyword lists declared in the config —
inspectable and replay-stable. **No LLM judges anywhere in evaluation**;
if one is ever added it would decide only "does this text convey fact F,"
never whether an outcome happened (see §7 determinism boundary).

**Anti-reward-hacking:** activity earns nothing (only world-state checks
score); message spam and over-budget meeting time are penalized; meetings
already cost real capacity in-sim; gaming the tracker does nothing because
checks read ground truth, not reported status. Verified behavior: a
do-nothing week scores 0.0; a run that discovers the slip, warns the
stakeholder, assigns the unowned P1 + questionnaire, and corrects the
tracker scores the full 12/12 available beyond baseline.

## 9b. Fairness is a rubric too

Two senses, both deterministic, both measurable from config/schedule alone:

**Fairness OF the scenario (authoring gate).** A scored ask must be
physically completable from the moment the PM could first KNOW it (chat:
instant; email: the next batch tick). Anything tighter scores luck, not
skill — poison for GRPO advantages. Enforced twice: `validate.py` rejects
unfair arrivals outright, and the difficulty fingerprint reports
`reaction_ratio` per arrival/confession (work-calendar window from delivery
to horizon ÷ effort revealed; ≥1.0 required, 1.0–1.3 is a legitimate tight
squeeze). The ladder: demo_1 4.75 / demo 3.88 / demo_2 1.56 — the tightest
variant is tight but
never impossible.

**Fairness OF the PM (outcome metric).** Per-person utilization = hours
worked *within the graded window* ÷ hours available;
`workload_fairness = 1 − σ(utilization)` (population std — variance sees the
whole team's imbalance, not just the extreme pair). A pure outcome of
assignment decisions: two strategies with identical task scores separate
when one rides a single engineer (demo: pile-everything-on-Dave 0.69 vs
spread 0.77). Reported as a raw baseline → agent → OPT triplet, NOT
normalized: dependency chains can force imbalance on perfect play too, so
OPT is the reference, not 1.0. Kept out of the combined scalar by default —
it's a second axis for the Pareto view (task value × team health), and can
be folded into `combined` later via a rubric weight if training should
optimize it directly.

## 10. Epistemics: three layers of knowing

| Layer | Who has it | Mechanism |
|---|---|---|
| **Truth** | engine + eval only | the scheduler; nobody in the sim reads it — not the PM, not even the task's holder |
| **Belief(t)** | each holder, about their own work | honest-but-possibly-wrong; **matures on an authored schedule** (`belief` entries → `belief_update` events, optionally with a proactive confession ping). Red herrings are the same thing about the org: confidently-held false knowledge (authored in `herrings` + persona lines) |
| **Tracker** | everyone | the record — lags, inherits stale beliefs; auto-updates only on **completions**, the one public reliable signal |

Consequences, all deterministic (zero LLM on the information path):
- **The interrogation hack is dead**: asking everyone Monday yields honest
  *wrong* answers — there is no perfect truth to extract, only a process to
  run (slack against noisy estimates, re-checks, fast reaction to authored
  belief corrections).
- **Exploration is a first-class skill**: skill/velocity factors and real
  progress are inferable only from observed completions (≈2 data points per
  person) blended with persona-colored self-reports. Channel economics (the
  email batching factor K, meeting costs, tracker lag) are deliberately
  ABSENT from tool descriptions — discovered, never documented.
- **Noise resistance is measurable**: chasing an authored herring (Dave's
  "rate-limiting is critical", Priya's "questionnaire is an hour") burns
  capacity the score prices; the herring list makes the distractors
  auditable per scenario.
- **NPCs are pure appended-context agents**: no tools of any kind; every
  stimulus (messages, room lines, invites, docs, belief realizations,
  completions) is PUSHED into their experience stream when it happens; one
  LLM call per utterance voices it. The stream is a fold of the log —
  replay-safe, no hidden state.
- **Physics is FELT, so testimony about it is honest**: the same 20
  focus-minutes the scheduler charges for an in-hours chat lands in the
  holder's experience stream ("that ping broke your concentration"). Whether
  they *voice* it is persona-gated — Sarah endures three pings politely,
  a fourth gets a terse "one thread, please" — so the fastest in-episode
  route to learning channel economics is the human one: listen when people
  tell you you're the problem. (How agents learn the taxes, in full: 1.
  evidence — email K read directly off reply timestamps, tax inferred from
  completions slipping vs calibrated velocity; 2. testimony — this
  mechanism; 3. in-weights — the reward gradient across GRPO rollouts,
  which needs no explicit knowledge at all.)

### The information ledger

Every quantity in the environment is classified by how it can be known:

| Class | Examples | Access | Cost |
|---|---|---|---|
| Hidden truth | true progress, projections, skill factors | inference only | — |
| Exposed truth | completions, assignments, deadlines, calendars | tracker/events | free |
| Testimony | beliefs, self-assessments, org folklore | asking | latency + tax + honestly-wrong risk |
| Learnable in-trajectory | velocities (from completions), testimony reliability, email K | probe + observe | time & actions |

Two learning channels, deliberately separate: **in-weights** (environment
physics — channel economics, working hours; constant across tasks, absent
from tool descriptions, learned across episodes) and **in-context**
(instance latents — this week's wrong beliefs, this team's speeds; varied
by the generator, must be inferred fresh each trajectory from evidence).
The split trains adaptivity, not memorization.

### Push vs pull: delivery semantics of every signal

Every interpersonal signal is classified on BOTH sides (`sim/signals.py`,
one static table; `World.record` stamps each logged event with its
`delivery` so runs are self-describing):

- **sender push** — a deliberate act, emitted now (type a chat, speak in a
  room). **sender pull** — state exposed passively; others come read it.
- **recipient push** — lands in the recipient's experience unasked, and can
  interrupt (a chat in working hours costs ~20 serialized focus-minutes).
  **recipient pull** — the recipient must come get it: on their own schedule
  (email drains at the next wakeup batch) or by checking a surface
  (the tracker board via `view_tasks`).

| Signal | Sender | Recipient | Physics |
|---|---|---|---|
| chat message | push | **push** | instant; interrupts — an NPC pays the 20-min serialized focus tax; the PM is *woken mid-`advance_time`* |
| email | push | **push (batched)** | delivered by event on the recipient's batch cadence: NPCs at their next wakeup (the learnable K), the PM at the next `email_batch_minutes` grid tick. Delayed, zero tax |
| meeting invite | push | push | lands on every attendee's calendar/stream |
| room line (`talk_in_meeting`) | push | push | heard live by all attendees (wakes the PM if they're in the room); consumes the meeting slot |
| minutes / transcript | push (auto) | push | broadcast to attendees at meeting end |
| doc share | push | push | notification into each `shared_with` stream |
| task added to board | push | **pull** | the board shows only *filed* tasks; anyone must check it to see them |
| task board edit | push | pull | exception: an *assignment* additionally pushes a notify to the assignee |
| task completion | push (auto) | push | the one public, reliable broadcast — delivered with the PM's next tool result, but never wakes them (only people wake you) |
| holder belief | — (pull) | — | never emitted on its own; surfaced only when asked (testimony) or at an authored confession ping (which is a push chat) |

**The PM's side, concretely — everything is push, because this is a DES.**
A human PM "always refreshing email" is a background process; our agent has
no background thread, and in a discrete-event simulation nothing happens
except by an event popping off the heap. So there is no poll tool anywhere:
push-class signals ride along with every tool result (`notifications`), and
a chat ping, a delivered email batch, or someone speaking in your meeting
interrupts `advance_time` early, handing control back at that instant.
Email's nature survives as *when its event fires*: an `email_delivery`
event on a fixed batch grid (`email_batch_minutes`, default 30) rather than
instantly — delayed but tax-free. Board broadcasts (completions) are
delivered with the next result but never break sleep — only people and mail
do. `view_inbox` is pure scrollback of what has been delivered; an email in
flight is invisible.

### Channel frictions, with their real-life justification

Every friction is a claim about how offices actually work — if a number
can't be defended by the real-world row, it shouldn't be in the physics.

**Chat**
| friction | value | real life |
|---|---|---|
| NPC reply latency | 5–45 min (work hours; else next morning) | people see Slack between tasks, not instantly; overnight pings get answered at 9am |
| focus tax on recipient | 20 min per message, **serialized** | an interruption costs ~20 min of refocus on deep work (attention-research classic); three pings are three broken flows, not one |
| delivery to PM | instant push, wakes mid-sleep | a direct ping buzzes your phone — humans are interrupt-driven for chat |
| unknown recipient | error: not in directory | you can't DM someone who isn't in the workspace |

**Email**
| friction | value | real life |
|---|---|---|
| focus tax | zero, both directions | email is async by social contract — nobody drops their editor because mail arrived |
| NPC reads | at next wakeup (~90–150 min) | people process inboxes in batches between work blocks |
| PM receives | next 30-min grid tick, then push | "always refreshing my email" is a background habit; in a DES that habit is a scheduled delivery event |
| unknown address | bounces (reported in result) | mail to a nonexistent address bounces; it never crashes the office |

**Meetings**
| friction | value | real life |
|---|---|---|
| capacity block | every attendee, full duration | a 1-hour meeting with 4 people costs the org 4 person-hours of work — the most expensive artifact in any company |
| `talk_in_meeting` | zero queue latency, ~3 min of slot per exchange | everyone is already in the room: answers are immediate, but the talking itself consumes the shared hour |
| reply deferral | chats answered after the meeting ends | people don't answer DMs mid-meeting (or you wish they didn't) |
| minutes broadcast | verbatim, to all attendees at end | the meeting's one durable output is its record — same context for everyone who was there |

**Universal**: every PM action costs 1 sim-minute (typing and deciding take
real time), and due events drain on every advance (the world doesn't pause
because you're busy).

**External work arrives as communication, never as board state.** An arrival
(`task_arrivals[i]`) creates the *need* (truth) but writes nothing to the
tracker. The ask reaches the PM via `announce` — chat (instant push) or
email (batched push, lands a grid-tick later) — carrying a deterministic ticket reference
(`(ticket: id)`, appended by rules, not the LLM voice). The PM must file it
(`add_task` with that id) before it appears on the board or can be assigned.
Filing mints nothing: only the authored ticket carries rubric weight, an
unfiled ticket sits at zero progress forever, and made-up tasks stay
zero-weight — so the outcome metrics price discovery, reading your inbox,
and record hygiene without any behavioral checks.

**The org fallback: the baseline is not a strawman.** A no-PM team still
functions — badly. Each arrival carries an authored
`fallback: {npc, at}`: if the ask is still unowned at that time, its
natural volunteer just grabs it (assigned, filed, tracker-noted; source
`org`). Same physics in EVERY run, so the null baseline includes the org's
self-organization and the PM earns credit only for what coordination adds:
picking better owners, acting before the fallback, spreading load the
volunteer won't. Authoring rule: the org reacts fast to urgent incidents
(hours) and slowly to cross-team paperwork (days) — that latency gap is
where PM value lives. This forced a physics fix: `assigned_at` is stamped
on every assignment mutation and the scheduler starts work at
max(arrival, assigned_at) — assignment used to be retroactive (a 16:00
pickup was credited work since the 11:30 arrival), which made acting early
worthless. Reassignment persists progress-so-far into `done_hours` at the
handoff instant, so prior work is kept but never re-credited. Measured
gradient on the demo: lazy 0.000, Friday-scramble reassign 0.042, prompt
file+assign+spread 0.926.

### Skill multipliers (decided, not yet implemented)

Per-(person × task) speed multipliers are realistic — Sarah is faster on DB
work than on frontend — but they must NOT be a runtime LLM call
(`multiplier = LLM(skills, task)`). Two reasons, both fatal for GRPO:
(1) group-relative advantages compare K rollouts of the same task T; if the
physics differ between rollouts, the advantage estimate is noise, not
signal; (2) anything an LLM computes from text is promptable — an agent
that words its messages or task notes the right way is optimizing the
judge, not the schedule.

The setting that keeps both realism and determinism: **LLM authors,
rules execute.** Tasks carry tags (`["backend", "db"]`), people carry a
skill map (`{"db": 1.3, "frontend": 0.7}`), and the runtime multiplier is a
pure lookup (e.g. geometric mean over matching tags, default 1.0) — frozen
into the scenario config at *generation time*, where an LLM may freely
propose skills/tags from personas. Constant across all rollouts of T,
replayable, OPT-compatible (the ideal relaxation pins multipliers at 1×),
and hidden: the PM learns who is fast at what only from observed
completions, like everything else.

## 10b. Truth vs. tracker (superseded by §10; kept for history)

Ground-truth task state (the analytic function) is world state. What people
*report* is separate data (`reported` on each task; NPC knowledge). Sarah's
tracker entry says 70%; the truth is 55%. The discovery loop — noticing the
gap and asking the right person the right question — is the core gameplay,
and the future evaluator scores outcomes against truth, not against reports.
