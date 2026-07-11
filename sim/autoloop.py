"""Auto-loop task generator — evolve a seed scenario until its MEASURED band
clears a threshold, using headless Claude to author edits and the deterministic
sim tools to measure and GATE.

    python -m sim.autoloop --task0 scenarios/crunch.json --target 20 \
        --out scenarios/generated.json \
        --directive "grow into a 5-engineer infra week, add depth" \
        --knowledge docs/autoloop_knowledge.md --max-iters 8

Design contract (this is what keeps it honest — it mirrors the project's rules):

  * The LLM only AUTHORS CONFIG, offline. Each iteration it edits the scenario
    JSON (and may append a note to the knowledge file). It NEVER sits on the
    physics / information / scoring path — same rule as the NPC narrators.
  * ALL measurement is deterministic sim code (validate / difficulty / intent).
    Difficulty is measured, never asserted by the model.
  * Acceptance is a GATE, not the model's opinion. A candidate is accepted only
    when, measured by the tools, it is:
        valid  AND  fair  AND  winnable >= T
        AND    NOT one-bit   (no single mechanism owns > --dominance-cap of the band)
        AND    forced_triage (oversubscribed, unless --no-triage)
    The one-bit and triage gates are what stop the loop from "winning" by
    inflating effort into a single dominant blocking gate — the exact failure
    mode measured while hand-building scenarios (a lone gate carried 0.94 of the
    band and every capable model just answered it -> no training signal).

The analytic band the loop optimizes is CHEAP and blind to belief/persuasion
variance, so it is a necessary-not-sufficient check. Use --verify-llm N to run
N real rollouts on the accepted task and print the score spread (the ground
truth for within-model variance) — off by default because it costs LLM calls.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

from .difficulty import fingerprint, stamp
from .intent import audit
from .rubric import load_rubric
from .validate import validate_scenario

KNOWLEDGE_SEED = r"""# Auto-loop design knowledge — how to build a task with real training signal

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
"""


def measure(path):
    """Deterministic read of a scenario's quality. Never calls an LLM."""
    with open(path) as f:
        sc = json.load(f)          # raises on broken JSON -> caller reports it
    errs = validate_scenario(sc)
    m = {"valid": not errs, "errors": errs}
    if errs:                        # invalid: band tools would crash on it
        return m
    rub = load_rubric(sc)
    fp = fingerprint(sc, rub)
    m.update(winnable=round(fp["winnable_combined"], 3),
             baseline=round(fp["baseline_combined"], 3),
             opt=round(fp["opt_combined"], 3),
             forced_triage=fp["forced_triage"], fair=fp["fair"],
             capacity_util=round(fp["capacity_utilization"], 3),
             opt_done_rate=round(fp["opt_done_weight_rate"], 3),
             log10_classes=fp["log10_trajectory_classes"])
    a = audit(sc)
    if a.get("degenerate"):
        m.update(degenerate=True, shares={}, max_share=0.0, oracle=None)
    else:
        sh = a["mechanism_shares"]
        m.update(degenerate=False, shares=sh, oracle=a["oracle_score"],
                 max_share=round(max(sh.values()), 3),
                 top_mechanism=max(sh, key=sh.get))
    return m


def accepted(m, T, cap, require_triage):
    return (m.get("valid") and m.get("fair") and not m.get("degenerate")
            and m.get("winnable", 0) >= T
            and m.get("max_share", 1.0) <= cap
            and (m.get("forced_triage") or not require_triage))


def gaps(m, T, cap, require_triage):
    """Human-readable list of what still fails the gate — fed to the author."""
    g = []
    if not m.get("valid"):
        return ["FIX VALIDITY FIRST: " + " | ".join(m["errors"])]
    if m.get("degenerate"):
        g.append("BAND IS EMPTY (OPT == baseline) — nothing to win. Add a "
                 "mechanism the no-PM org fails at (a blocking gate on a big, "
                 "still-in-progress task; an infeasible mis-assignment; or "
                 "oversubscription).")
    if m.get("winnable", 0) < T:
        g.append("winnable=%s, need >= %s — widen the band (more/bigger "
                 "baseline-failures)." % (m.get("winnable"), T))
    if m.get("max_share", 1.0) > cap:
        g.append("ONE-BIT RISK: mechanism '%s' owns %.2f of the band (> cap "
                 "%.2f). Spread it — add an independent lever (skill-routing, "
                 "triage, a second gate) so no single decision dominates."
                 % (m.get("top_mechanism"), m.get("max_share"), cap))
    if require_triage and not m.get("forced_triage"):
        g.append("NOT oversubscribed (capacity_util=%s, opt_done_rate=%s) — add "
                 "work so even OPT must sacrifice (forced_triage true)."
                 % (m.get("capacity_util"), m.get("opt_done_rate")))
    if not m.get("fair"):
        g.append("UNFAIR: a scored task isn't completable from first-knowable — "
                 "fix its effort/arrival timing.")
    return g


ENV_CHEATSHEET = """ENVIRONMENT — READ THESE REPO FILES FIRST (they are ground truth, not my summary):
  - DESIGN.md : the spec. Skim §2 (state & time), §6 (the scheduler & the
    org-fallback baseline), §8 (three belief layers), §10 (the band/scoring),
    §11 (task authoring & difficulty).
  - sim/validate.py : the EXACT rules your edited file must pass. Your edit is
    REJECTED if any fail (events in-window; a question's held_by = the task's
    sole worker owner; <=3 questions/task; belief held_by is a worker; arrivals
    completable; at least one worker; etc.).
  - scenarios/bigweek.json : a rich, valid EXEMPLAR — two gated dependency
    chains, belief slips, herrings, owned arrivals, per-tag skills, full
    per-person queues. Copy its shapes; it is the gold standard for structure.

KEY FACTS (getting these wrong = failed validation = a wasted iteration):
  - TIME is integer minutes from Monday 00:00. Work happens 09:00-17:00 only;
    nights & weekends accrue ZERO. Monday 09:00 = 545, Friday 17:00 (horizon)
    = 6780. Every `at` (questions/beliefs/arrivals/beats) must be in [545, 6780].
  - PRIORITY sets BOTH scheduling order AND score weight: P0=8 P1=4 P2=2 P3=1.
    (Authored priority is the weight; the PM may reorder but cannot relabel to
    mint value — so don't expect relabeling to change difficulty.)
  - SKILLS are per-tag speed multipliers on an NPC, e.g. "skills": {"backend": 1.9};
    a task routes by its `tags`; a worker with no matching skill works at 1.0x.
    Skills are HIDDEN from the PM — never reveal them in any NPC text.
  - BLOCKING GATE: a task's "questions": [{"id":..., "at":..., "gates": true,
    "held_by": <the task's SOLE worker owner>, "ping":"..."}] stalls that task
    from `at` until the PM replies to held_by. It only creates band if it opens
    WHILE the task still has substantial work left (not a near-done task).
  - BELIEF: "belief": [{"remaining_frac": <0..1 = fraction of effort LEFT>,
    "held_by": <worker>, "note":...}, optionally {"at":..., "remaining_frac":...,
    "proactive_ping": true, "note":...}] — the reported estimate and its later
    slip. Display-only; scoring always uses TRUTH (effort_hours/done_hours).
  - NPCs with "worker": false are STAKEHOLDERS — they CANNOT be assigned tasks.
  - ARRIVALS: "task_arrivals": [{"at":..., "npc":..., "via":"chat|email"?,
    "announce":..., "task": {...}}] land mid-week, usually on a busy/wrong owner.
  - Keep the top-level `seed`; add nothing stochastic; the headcount in the
    `company` string must match the worker + stakeholder cast size."""


def author_prompt(out_abs, know_abs, T, cap, require_triage, m, directive, it):
    return f"""You are tuning a Project-Manager-simulation SCENARIO so its
MEASURED difficulty band clears a threshold. You author CONFIG only — you never
change the engine or scoring (an LLM must never sit on the physics/scoring path;
you are the offline config author).

Iteration {it}. GOAL: edit the scenario so, measured by the deterministic tools,
it is valid AND fair AND winnable (OPT - baseline, COMBINED) >= {T} AND no single
mechanism owns more than {cap} of the band{"" if require_triage else ""} \
{"AND it is oversubscribed (forced_triage true)" if require_triage else "(forced_triage optional)"}.

{ENV_CHEATSHEET}

1. READ, in this order: DESIGN.md (§6, §10, §11), sim/validate.py,
   scenarios/bigweek.json (exemplar), then {out_abs} (the file you edit) and
   {know_abs} (design principles + this loop's past iterations — LEARN from it).

2. CURRENT measurement of {os.path.basename(out_abs)} (deterministic tools, not opinion):
   winnable={m.get('winnable')}  (baseline={m.get('baseline')} opt={m.get('opt')})
   mechanism shares of the band: {json.dumps(m.get('shares', {}))}
   forced_triage={m.get('forced_triage')}  capacity_util={m.get('capacity_util')}  fair={m.get('fair')}

   STILL FAILING THE GATE:
   {chr(10).join("   - " + x for x in gaps(m, T, cap, require_triage))}

3. USER DIRECTIVE for this scenario: {directive or "(none — just hit the gate)"}

4. EDIT {out_abs} to close those gaps, using the levers above (add NPCs with
   hidden skills / private knowledge / persona; add or resize tasks; add
   dependency chains; add a blocking gate that opens while work remains; tune
   efforts & priorities; add arrivals on a busy owner; add belief slips +
   herrings). WHY each edit moves the band (from the knowledge file): a bigger
   baseline-failure widens winnable; a second INDEPENDENT lever (skill-routing,
   triage, another gate) SPREADS a one-bit band; more work than capacity makes
   forced_triage true. Keep it a coherent, realistic week.

   HARD RULES (break these and you fail the gate or the point):
   - Do NOT inflate effort merely to grow the band — depth must be real (coupled
     decisions on a dependency chain + hidden latents), and the band must stay
     SPREAD (no single mechanism > {cap}).
   - Company headcount must match the cast; every scored task completable from
     first-knowable; skills stay hidden; keep `seed` and determinism.

5. Append ONE line to {know_abs} under "## Iteration log": what you changed and
   your hypothesis for how it moves the band. Then stop.

Make ONE coherent set of edits this turn. Do NOT run measurement tools yourself —
the harness re-measures deterministically after you finish and feeds you the result."""


def call_author(prompt, model, timeout):
    claude = shutil.which("claude")
    if not claude:
        return None, "`claude` CLI not on PATH — headless authoring needs Claude Code"
    cmd = [claude, "-p", prompt, "--allowedTools", "Read,Edit,Write",
           "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "author timed out after %ds" % timeout
    if proc.returncode != 0:
        return None, "claude -p failed: %s" % (proc.stderr or "no output")[-300:]
    return proc.stdout.strip(), None


def verify_llm(path, model, n):
    """OPTIONAL ground-truth check: run n real rollouts, report the score spread
    (within-model variance the analytic band cannot see)."""
    from .eval import evaluate
    import glob
    import statistics as st
    scores = []
    for i in range(n):
        before = set(glob.glob("runs/run-*"))
        subprocess.run([sys.executable, "-m", "sim.harness", path,
                        "--probe", "llm", "--model", model],
                       capture_output=True, text=True, timeout=1800)
        new = sorted(set(glob.glob("runs/run-*")) - before)
        if not new:
            continue
        try:
            scores.append(evaluate(new[-1])["score"])
        except Exception:
            pass
    if not scores:
        return "  (no rollouts scored)"
    return ("  n=%d  min=%.3f max=%.3f range=%.3f mean=%.3f std=%.3f\n  scores=%s"
            % (len(scores), min(scores), max(scores), max(scores) - min(scores),
               st.mean(scores), st.pstdev(scores),
               ", ".join("%.2f" % s for s in sorted(scores))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task0", required=True, help="seed scenario to evolve")
    ap.add_argument("--target", type=float, required=True,
                    help="threshold T: accept when winnable (OPT-baseline) >= T")
    ap.add_argument("--out", required=True, help="working/output scenario file")
    ap.add_argument("--knowledge", default="docs/autoloop_knowledge.md",
                    help="living design-knowledge file the author reads & appends")
    ap.add_argument("--directive", default="", help="what you want this task to become")
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--dominance-cap", type=float, default=0.85,
                    help="reject if any one mechanism owns > this share (one-bit guard)")
    ap.add_argument("--no-triage", action="store_true",
                    help="don't require forced_triage (oversubscription)")
    ap.add_argument("--model", default=os.environ.get("PM_SIM_AUTHOR_MODEL"),
                    help="model for the headless author (default: Claude Code's)")
    ap.add_argument("--author-timeout", type=int, default=900)
    ap.add_argument("--resume", action="store_true",
                    help="continue evolving --out instead of copying --task0 over it")
    ap.add_argument("--verify-llm", type=int, default=0, metavar="N",
                    help="after acceptance, run N real rollouts and report score spread")
    ap.add_argument("--verify-model", default="claude-sonnet-5")
    args = ap.parse_args()

    require_triage = not args.no_triage
    if not (args.resume and os.path.exists(args.out)):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        shutil.copyfile(args.task0, args.out)
        print("seeded %s from %s" % (args.out, args.task0))
    if not os.path.exists(args.knowledge):
        os.makedirs(os.path.dirname(args.knowledge) or ".", exist_ok=True)
        with open(args.knowledge, "w") as f:
            f.write(KNOWLEDGE_SEED)

    out_abs = os.path.abspath(args.out)
    know_abs = os.path.abspath(args.knowledge)
    best = None   # (winnable, snapshot_path) among VALID candidates seen

    for it in range(1, args.max_iters + 1):
        try:
            m = measure(args.out)
        except (ValueError, KeyError) as e:
            m = {"valid": False, "errors": ["scenario unreadable/malformed: %s" % e]}
        tag = "ACCEPT" if accepted(m, args.target, args.dominance_cap, require_triage) else "..."
        print("\n[iter %d] %s  winnable=%s  max_share=%s  triage=%s  fair=%s  valid=%s"
              % (it, tag, m.get("winnable"), m.get("max_share"),
                 m.get("forced_triage"), m.get("fair"), m.get("valid")))
        if m.get("valid") and not m.get("degenerate") and (
                best is None or m.get("winnable", 0) > best[0]):
            snap = args.out + ".best"
            shutil.copyfile(args.out, snap)
            best = (m.get("winnable", 0), snap)

        if accepted(m, args.target, args.dominance_cap, require_triage):
            stamp(args.out)
            print("\n== DONE at iter %d ==" % it)
            print("   %s  band=[%s, %s]  winnable=%s"
                  % (args.out, m.get("baseline"), m.get("opt"), m.get("winnable")))
            print("   shares: %s" % json.dumps(m.get("shares", {})))
            if args.verify_llm:
                print("\n== LLM verification (%d rollouts, %s) =="
                      % (args.verify_llm, args.verify_model))
                print(verify_llm(args.out, args.verify_model, args.verify_llm))
            return

        for g in gaps(m, args.target, args.dominance_cap, require_triage):
            print("   gap: %s" % g)
        out, err = call_author(
            author_prompt(out_abs, know_abs, args.target, args.dominance_cap,
                          require_triage, m, args.directive, it),
            args.model, args.author_timeout)
        if err:
            print("   author error: %s — stopping." % err)
            break
        print("   author: %s" % (out or "(edited files, no summary)")[:400])

    print("\n== stopped without clearing the gate ==")
    if best:
        shutil.copyfile(best[1], args.out)
        stamp(args.out)
        print("   kept best valid candidate: winnable=%s -> %s" % (best[0], args.out))
    else:
        print("   no valid candidate produced.")


if __name__ == "__main__":
    main()
