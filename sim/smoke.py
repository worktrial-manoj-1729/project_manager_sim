"""Deterministic trajectory smoke tests — NO LLM / API (uses the eval StubClient).

    python -m sim.smoke [scenarios/crunch.json]

These pin the DES trajectory invariants we care about (and just fixed), so
they can't silently regress. NPC text is a stub — irrelevant to trajectory
STRUCTURE, which is what we assert. Exits non-zero if any check fails.

Checks:
  1. causality      — no undispatched event ever sits at/before the clock
  2. stops-at-push  — an interruptible advance halts AT the first push instant
  3. no-push-skip   — driving to horizon delivers EVERY push (none fired into
                      the void — the bug that let the PM sleep through Wed/Thu)
  4. determinism    — same seed -> byte-identical event trajectory on replay
  5. wait-for-reply — the yield primitive returns when a message for the PM lands
"""

import json
import sys

from .engine import Engine
from .eval import StubClient
from .rubric import load_rubric
from .sim_time import fmt


def _engine(scenario):
    return Engine(scenario, client=StubClient(), verbose=False)


def _horizon(scenario):
    return load_rubric(scenario)["horizon"]


def _waking_pushes(log):
    """The (time, sender) of every push that should WAKE the PM — chat to the
    agent + delivered email. (Completions are delivered but don't wake.)"""
    out = []
    for e in log:
        if (e["kind"] == "message" and e.get("recipient") == "agent"
                and e.get("via", "chat") == "chat"):
            out.append((e["t"], e["sender"]))
        elif e["kind"] == "email_delivered":
            out.append((e["t"], e["sender"]))
    return out


def _queue_clean(eng):
    """The causality invariant: nothing undispatched at/before the clock."""
    nxt = eng.queue.peek()
    return nxt is None or nxt.time > eng.world.clock


def check_causality(scenario):
    """Drive a mixed trajectory (actions + advances); after every operation no
    pending event may sit at/before the clock."""
    eng = _engine(scenario)
    horizon = _horizon(scenario)
    npc = next(iter(eng.world.npcs))
    assert _queue_clean(eng), "dirty queue at init"
    # a burst of synchronous actions, each must drain due events
    for i in range(5):
        eng.agent_say(npc, "ping %d" % i)
        if not _queue_clean(eng):
            return False, "action %d left an event at/before clock %s" % (i, eng.world.now())
    # interruptible advances toward horizon
    steps = 0
    while eng.world.clock < horizon and steps < 500:
        eng.advance_until(horizon, interruptible=True)
        eng.drain_agent_push()
        steps += 1
        if not _queue_clean(eng):
            return False, "advance left an event at/before clock %s" % eng.world.now()
    return True, "invariant held across actions + full drive to %s" % fmt(horizon)


def check_stops_at_push(scenario):
    """An interruptible advance from t0 must halt AT the first push instant,
    not somewhere past it."""
    truth = _waking_pushes(_drive_full(scenario))
    if not truth:
        return False, "scenario has no PM-directed pushes to test"
    first_t = min(t for t, _ in truth)
    eng = _engine(scenario)
    horizon = _horizon(scenario)
    eng.advance_until(horizon, interruptible=True)
    woke = eng.drain_agent_push()
    if eng.world.clock != first_t:
        return False, ("stopped at %s, but first push is at %s"
                       % (fmt(eng.world.clock), fmt(first_t)))
    if not any("chat_from" in w or "email_from" in w for w in woke):
        return False, "halted but delivered no waking push"
    return True, "halted exactly at first push %s (%d pushes total)" % (fmt(first_t), len(truth))


def _drive_full(scenario):
    """Full non-interruptible drive to horizon -> the ground-truth log."""
    eng = _engine(scenario)
    eng.advance_until(_horizon(scenario), max_events=5000)
    return eng.world.log


def check_no_push_skipped(scenario):
    """Driving to horizon with the interruptible loop must surface EVERY push
    that a full drive produces — none skipped, none duplicated (exact
    multiset), so a repeated (time, sender) can't hide a skip."""
    from collections import Counter
    truth = Counter(_waking_pushes(_drive_full(scenario)))
    eng = _engine(scenario)
    horizon = _horizon(scenario)
    delivered, stops = Counter(), 0
    while eng.world.clock < horizon and stops < 1000:
        eng.advance_until(horizon, interruptible=True)
        got = eng.drain_agent_push()
        stops += 1
        for w in got:
            if "chat_from" in w:
                delivered[(_unfmt(scenario, w["time"]), w["chat_from"])] += 1
            elif "email_from" in w:
                delivered[(_unfmt(scenario, w["time"]), w["email_from"])] += 1
    if delivered != truth:
        return False, ("push multiset mismatch — missing=%s extra=%s"
                       % (sorted((truth - delivered).elements()),
                          sorted((delivered - truth).elements())))
    return True, "all %d pushes delivered across the week (exact, none skipped)" % sum(truth.values())


def _unfmt(scenario, timestr):
    """Recover sim-minutes from a 'Day HH:MM' stamp (for set comparison)."""
    from .sim_time import DAY_NAMES
    day, hm = timestr.split(" ")
    h, m = hm.split(":")
    return DAY_NAMES.index(day) * 1440 + int(h) * 60 + int(m)


def check_determinism(scenario):
    """Same seed + stub -> identical event trajectory (kind, time, text)."""
    def sig(log):
        return [(e["seq"], e["t"], e["kind"], e.get("text", ""),
                 e.get("task_id", ""), e.get("sender", "")) for e in log]
    a = sig(_drive_full(scenario))
    b = sig(_drive_full(scenario))
    if a != b:
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
        return False, "trajectories diverge at seq %s" % i
    return True, "%d events identical across two runs" % len(a)


def check_wait_for_reply(scenario):
    """wait_for_reply must return once a message for the PM lands."""
    eng = _engine(scenario)
    npc = next(iter(eng.world.npcs))
    eng.agent_say(npc, "are you there?")
    msg = eng.run_until_reply()
    if msg is None:
        return False, "no reply surfaced within the wait horizon"
    if msg.get("recipient") != "agent":
        return False, "returned a non-PM message"
    return True, "reply from %s surfaced at %s" % (msg.get("sender"), fmt(msg["t"]))


def check_harness_drives_to_horizon(scenario):
    """The actual run_llm loop: a no-op agent (stub returns text, never tools)
    must still be driven all the way to Friday — woken on every push, never
    hanging, never advanced past work. Guards the turn-budget removal + the
    interruptible drive directly."""
    from .harness import run_llm
    eng = _engine(scenario)
    horizon = _horizon(scenario)
    run_llm(eng)   # StubClient agent: yields every turn -> pure drive loop
    if eng.world.clock < horizon:
        return False, "run_llm returned at %s, short of horizon %s" % (
            fmt(eng.world.clock), fmt(horizon))
    if not _queue_clean(eng):
        return False, "run_llm left the queue dirty"
    return True, "no-op agent driven to %s, queue clean" % fmt(eng.world.clock)


CHECKS = [
    ("causality", check_causality),
    ("stops-at-push", check_stops_at_push),
    ("no-push-skipped", check_no_push_skipped),
    ("determinism", check_determinism),
    ("wait-for-reply", check_wait_for_reply),
    ("harness-drive", check_harness_drives_to_horizon),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/crunch.json"
    with open(path) as f:
        scenario = json.load(f)
    print("trajectory smoke: %s\n" % path)
    failed = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn(scenario)
        except Exception as e:
            ok, detail = False, "EXCEPTION %s: %s" % (type(e).__name__, e)
        print("  [%s] %-16s %s" % ("PASS" if ok else "FAIL", name, detail))
        failed += not ok
    print("\n%s (%d/%d passed)"
          % ("ALL PASS" if not failed else "FAILURES", len(CHECKS) - failed, len(CHECKS)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
