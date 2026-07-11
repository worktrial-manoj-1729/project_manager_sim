# Auto-loop design knowledge — how to build a task with real training signal

This is distilled, MEASURED knowledge (numbers below are things that were
actually probed in this environment). You are editing a scenario to hit a
winnable-band threshold WITHOUT producing a low-signal task. Read it, apply it,
and append what you learn under "## Iteration log".

## 0. What "good" means here (GRPO)
The reward is normalized `(agent - baseline)/(OPT - baseline)` and GRPO's
advantage is `(r - group_mean)/group_std` over N rollouts of the SAME task. So a
good task needs, at once:
  (a) WITHIN-GROUP VARIANCE — if all rollouts score alike, std->0, no gradient.
  (b) that variance must be DECISION-driven, not RNG (determinism guarantees it).
  (c) a GRADED, hack-proof reward — differences reflect how much better the work
      was, nothing else.
Calibration: a good task sits at the target model's competence EDGE — its
rollouts scatter across the MIDDLE of the band, not piled at 0 or 1. (Measured:
bigweek separates haiku 0.61 from sonnet/opus ~0.85 WITH headroom = good; mobile
at 0.89-0.97 for strong models = too easy, little signal for them.)

## 1. The five necessary conditions (each earned by a measurement)
1. A WINNABLE BAND EXISTS — the no-PM org must genuinely FAIL somewhere.
   Removing the only gate made baseline ~= OPT and winnable collapsed to 1.3
   (org does 89% unaided) -> nothing to learn. The band only opens where the
   unmanaged team fails: a blocking gate (work stalls), an infeasible
   mis-assignment (owner can't finish, a faster specialist has slack), or
   oversubscription (org burns time on the wrong priority).
2. GRADED, NOT ONE-BIT — this is the #1 anti-signal trap. One big gate carried
   0.944 of the band: answering it is OBVIOUS, so every capable model does it and
   clusters at the ceiling -> zero within-group variance -> no gradient despite a
   "hard" task. Spread the band across several load-bearing decisions so no
   single mechanism owns more than ~0.85.
3. CAPACITY PRESSURE makes mistakes COST. With roomy capacity (util 0.82) the
   traps (leave incident on the bottleneck, chase the herring, misroute the
   specialist) swung the score a total of 0.05 — slack absorbed every mistake.
   Oversubscribe until even OPT must sacrifice: forced_triage true,
   capacity_utilization ~0.9-1.1.
4. EVERY DECISION NEEDS A REAL WINDOW. Late gates were decorative: gates opening
   at t=1400/2200/3100 on small tasks contributed EXACTLY 0.000 because the owner
   finished the task before the gate bit. A gate creates band only if it opens
   WHILE the task still has substantial work left.
5. DETERMINISM so the variance is signal not noise (NPCs temp 0, seeded draws,
   keep `seed`). Never add an LLM/keyword to physics or scoring.

## 2. Failure modes (measured — do not reproduce)
- One dominant obvious lever (a lone gate at 0.94 share) -> clusters at ceiling.
- No baseline-failure -> winnable ~1 -> nothing to train on.
- A late or small gate -> decorative (0.000 contribution).
- Roomy capacity -> traps absorbed (0.05 swing) -> no signal.
- add_helper with a LOW-skill helper scored -0.9 (freezes the helper's own P1 while
  pooling barely speeds the target) — a trap, not a lever.
- Reassigning to an ALREADY-LOADED specialist nets ~0 (queue congestion eats the
  1.9x skill gain). Skill-routing pays ONLY if the specialist has real slack.
- WINNABLE-GAMING: inflating effort hours grows `winnable` with no added depth.
  Forbidden — the band must stay SPREAD and the depth must be real.

## 3. Lever -> effect (what actually moves the band, and how)
- Blocking gate on a big, still-in-progress CRITICAL-PATH task: opens a large
  winnable band, but is one-bit if it is the only lever. Use it, then SPREAD.
- Skill routing: a task mis-assigned to a slow/wrong-skilled owner while a much
  faster specialist (skill ~1.9 vs 1.0) has SLACK -> reassigning is a big, real
  win. Needs the skill gap AND the slack, or it is ~0.
- Oversubscription: the multiplier — it turns otherwise-decorative traps into
  real losses (a burned specialist now drops a P1). This is how you SPREAD a
  one-bit band without adding another gate.
- Dependency chain + HIDDEN bottleneck (a critical task reported ~80% done that
  is really ~25%, confessed in a later proactive ping): depth. Variance lives
  over TRAJECTORIES — an early misread cascades. This is how you get STRONG
  models to scatter.
- Belief lie + loud herring: DOWNSIDE variance (a misled model does WORSE).
  IMPORTANT: this is invisible to the analytic tools (they score with truth), so
  it will NOT show up in the mechanism shares — it only appears in real LLM
  repeats. Add it for depth, but don't expect it to move `winnable`.
- Arrival on a busy/wrong owner: a reassignment + timing lever; only bites under
  capacity pressure.

## 4. How to SPREAD a one-bit band (the usual fix)
If the audit shows one mechanism (usually "answering blockers") owning > ~0.85:
  - Add a genuinely INFEASIBLE mis-assignment with an IDLE specialist (skill lever).
  - Oversubscribe so triage carries weight (forced_triage true).
  - Or add a SECOND independent gate on a different chain/owner.
  - Balance efforts so the independent / downstream P1s rival the gated chain's
    weight — if the gated chain dwarfs everything, it re-dominates.

## 5. Tensions (resolved)
- Graded-and-analyzable (independent levers) vs DEEP (coupled): depth needs
  coupling, but coupling makes the analytic combo-probe blind. Get structure from
  the analytic band; confirm depth-variance with LLM repeats.
- Winnable requires a baseline-failure, and the cleanest one (a gate) is also the
  biggest one-bit trap. Resolve with SEVERAL moderate baseline-failures + capacity
  pressure, not one giant gate.

## 6. Operational knobs and values that worked
- TIME: minutes from Mon 00:00; workday 09:00-17:00 (480 min/day), nights/weekends
  accrue 0. Mon 09:00 = 545, Fri 17:00 = 6780. All `at` in [545, 6780].
- PRIORITY weights: P0=8 P1=4 P2=2 P3=1 (authored priority = the weight).
- SKILLS: per-tag multipliers; a strong specialist ~1.9, default 1.0. A ~1.9 skill
  SPREAD is what makes routing load-bearing.
- POOLING: parallel cap 1.5, meeting cap 2.5; fades 1.0, 0.6, 0.36 (Brooks); per-
  person meeting cap 180 min/day; meetings can't overlap OOO. answer_batch 120;
  chat focus tax 10.
- Difficulty is SEVEN dials: capacity pressure, default-owner wrongness,
  dependency depth, arrival timing/channel, belief error & confession time, skill
  spread, forced triage. Pull any axis to harden the week there.
- Reference bands: crunch winnable ~7, mobile ~10, bigweek ~32 (6 engineers + 2
  leads, ~21 tasks, two gated chains, big skill spread — the model of a wide AND
  spread band).

## 7. The acceptance gate (what the loop measures — target these)
- valid (sim/validate.py): 0 errors.
- winnable_combined >= T.
- max mechanism-share <= ~0.85 (sim/intent.py audit): NOT one-bit.
- forced_triage True (capacity_utilization ~0.9-1.1).
- fair True: every scored task completable from first-knowable.
- log10_trajectory_classes high = many outcome-distinguishable plans.
- (Ground truth for variance — LLM repeats std — is NOT in the analytic gate;
  the analytic band is necessary-not-sufficient.)

## Iteration log
