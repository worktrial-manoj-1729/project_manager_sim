"""The HARD discrete-event-simulation guarantees, checked as properties
under seeded random interleavings — not hand-picked examples.

Invariants asserted after EVERY fuzzed action:
  CAUSALITY   no undispatched event sits at or before the clock
  MONOTONE    the clock never decreases; log times are non-decreasing
  LIVENESS    no tool call ever raises; the heap never starves (heartbeats)

And at the horizon of every fuzzed week:
  REPLAY      fold(events) == the live world, byte for byte
  CEILING     score <= 1.0 for any play WITHOUT task-labeled working
              sessions (the fuzzer books none) — sessions are the one
              mechanic that can legitimately exceed the solo optimum
  TERMINATION advance always reaches the horizon

Seeds are FIXED: the fuzz is deterministic, so a failure here is a real,
reproducible counterexample — never flake.
"""

import random
import unittest

from sim.eval import evaluate
from sim.replay import load_run, replay
from sim.tools import call_tool

from tests.fixtures import HORIZON, goto, make_engine

NPCS = ["ana", "bo", "vp", "ghost"]          # includes invalid targets
TASKS = ["core", "side", "incident", "form", "nope"]


def fuzz_action(rng):
    """One random-but-valid-shaped tool call (invalid targets included on
    purpose — errors must come back as dicts, never as exceptions)."""
    roll = rng.random()
    if roll < 0.22:
        return "send_chat", {"npc": rng.choice(NPCS), "text": "fuzz ping"}
    if roll < 0.34:
        return "send_email", {"to": rng.sample(NPCS, rng.randint(1, 2)),
                              "subject": "fuzz", "body": "fuzz body"}
    if roll < 0.46:
        return "add_task", {"title": "Fuzz item %d" % rng.randint(0, 9),
                            **({"id": rng.choice(TASKS)} if rng.random() < .5 else {})}
    if roll < 0.60:
        return "assign_task", {"task_id": rng.choice(TASKS),
                               "npc": rng.choice(NPCS)}
    if roll < 0.68:
        return "schedule_meeting", {
            "attendees": rng.sample(["agent", "ana", "bo"], rng.randint(1, 3)),
            "start_in_minutes": rng.randint(1, 600),
            "duration_minutes": rng.randint(15, 240),
            "topic": "fuzz sync"}
    if roll < 0.74:
        return "talk_in_meeting", {"text": "fuzz remark"}
    if roll < 0.80:
        return "update_tracker_note", {"task_id": rng.choice(TASKS),
                                       "note": "fuzz note"}
    if roll < 0.86:
        return "view_tasks", {}
    if roll < 0.92:
        return "wait_for_reply", {}
    return "advance_time", {"minutes": rng.randint(1, 700)}


class TestDESProperties(unittest.TestCase):
    def _assert_invariants(self, eng, prev_clock, prev_log_len):
        clock = eng.world.clock
        # MONOTONE clock
        self.assertGreaterEqual(clock, prev_clock)
        # CAUSALITY: nothing due-or-past may still sit in the heap
        nxt = eng.queue.peek()
        if nxt is not None:
            self.assertGreater(nxt.time, clock,
                               "undispatched event at/before the clock")
        # MONOTONE log: append-only, times never go backwards
        log = eng.world.log
        self.assertGreaterEqual(len(log), prev_log_len)
        for a, b in zip(log[prev_log_len and prev_log_len - 1:],
                        log[prev_log_len:]):
            self.assertLessEqual(a["t"], b["t"])
        return clock, len(log)

    def _fuzz_week(self, seed):
        rng = random.Random(seed)
        eng = make_engine()
        clock, loglen = eng.world.clock, len(eng.world.log)
        for _ in range(120):
            if eng.world.clock >= HORIZON:
                break
            name, args = fuzz_action(rng)
            try:
                call_tool(eng, name, args)   # LIVENESS: must never raise
            except Exception as e:           # noqa: BLE001 — the assertion
                self.fail("tool %s%r raised %r" % (name, args, e))
            clock, loglen = self._assert_invariants(eng, clock, loglen)
        # LIVENESS: heartbeats keep the heap alive forever
        self.assertGreater(len(eng.queue), 0, "event heap starved")
        # TERMINATION: the week always ends
        goto(eng, HORIZON)
        self.assertEqual(eng.world.clock, HORIZON)
        return eng

    def test_fuzzed_weeks_hold_all_invariants(self):
        for seed in (11, 23, 47):
            eng = self._fuzz_week(seed)
            # REPLAY: the fold reconstructs the live world exactly
            world2 = replay(*load_run(eng.run_dir))
            self.assertEqual(eng.world.messages, world2.messages)
            self.assertEqual(eng.world.log, world2.log)
            self.assertEqual(eng.world.tracker_view(), world2.tracker_view())
            # CEILING (solo policy space): without task-labeled sessions,
            # nothing beats OPT — collaboration is the sole legitimate
            # way past it, and this fuzzer never books one
            res = evaluate(eng.run_dir)
            if res["score"] is not None:
                self.assertLessEqual(res["score"], 1.0,
                                     "seed %d beat the OPT ceiling" % seed)

    def test_advance_granularity_is_irrelevant(self):
        """A DES must not care HOW time is advanced: one jump to the horizon
        and a thousand small hops land in the identical world."""
        a, b = make_engine(), make_engine()
        a.advance_until(HORIZON, max_events=5000)
        step = 7
        while b.world.clock < HORIZON:
            b.advance_until(min(HORIZON, b.world.clock + step), max_events=5000)
        self.assertEqual(a.world.messages, b.world.messages)
        self.assertEqual(a.world.completed_announced, b.world.completed_announced)
        self.assertEqual(a.world.tasks_view(at=HORIZON),
                         b.world.tasks_view(at=HORIZON))

    def test_empty_queue_cannot_deadlock(self):
        """Liveness even with a drained heap: advancing jumps the void."""
        eng = make_engine()
        while len(eng.queue):
            eng.queue.pop()                  # sabotage: drain the heap
        eng.advance_until(HORIZON, max_events=10)
        self.assertEqual(eng.world.clock, HORIZON)


if __name__ == "__main__":
    unittest.main()
