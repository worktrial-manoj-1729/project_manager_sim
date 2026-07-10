"""Determinism invariants: the clock, the queue, and twin-run equality."""

import unittest

from sim.events import EventQueue
from sim.sim_time import (fmt, in_working_hours, is_workday, next_work_start,
                          working_minutes_between)
from sim.tools import call_tool
from sim.world import World

from tests.fixtures import HORIZON, goto, make_engine, scenario


class TestSimTime(unittest.TestCase):
    def test_fmt_and_working_hours(self):
        self.assertEqual(fmt(545), "Mon 09:05")
        self.assertTrue(in_working_hours(545))
        self.assertFalse(in_working_hours(400))          # before 09:00
        self.assertFalse(in_working_hours(5 * 1440 + 600))  # Saturday
        self.assertFalse(is_workday(6 * 1440))           # Sunday

    def test_next_work_start_skips_weekend(self):
        fri_evening = 4 * 1440 + 18 * 60
        self.assertEqual(next_work_start(fri_evening), 7 * 1440 + 540)  # Mon 09:00

    def test_working_minutes_full_week(self):
        # Mon 00:00 .. Sun 24:00 = 5 workdays x 8.5h
        self.assertEqual(working_minutes_between(0, 7 * 1440), 5 * 510)


class TestQueueAndClock(unittest.TestCase):
    def test_seq_tiebreak_preserves_insertion_order(self):
        q = EventQueue()
        q.push(10, "b", {})
        q.push(10, "a", {})
        q.push(5, "c", {})
        self.assertEqual([q.pop().kind for _ in range(3)], ["c", "b", "a"])

    def test_clock_is_monotonic(self):
        w = World(scenario())
        w.advance_clock_to(600)
        with self.assertRaises(ValueError):
            w.advance_clock_to(599)


class TestTwinRuns(unittest.TestCase):
    def test_same_actions_same_event_log(self):
        """The determinism contract: same config + same actions -> identical
        event logs, byte for byte (stub LLM; seeded per-NPC rng)."""
        def week(eng):
            call_tool(eng, "send_chat", {"npc": "ana", "text": "status?"})
            goto(eng, 2100)
            call_tool(eng, "add_task", {"title": "Incident", "id": "incident"})
            call_tool(eng, "assign_task", {"task_id": "incident", "npc": "bo"})
            call_tool(eng, "send_email", {"to": ["ana", "bo"],
                                          "subject": "plan", "body": "see board"})
            goto(eng, HORIZON)
            return eng.world.log

        a, b = week(make_engine()), week(make_engine())
        self.assertEqual(a, b)

    def test_felt_interruption_lands_in_stream(self):
        eng = make_engine()
        call_tool(eng, "send_chat", {"npc": "ana", "text": "ping 1"})
        call_tool(eng, "send_chat", {"npc": "ana", "text": "ping 2"})
        ctx = "\n".join(m["content"] for m in eng.world.npcs["ana"].context
                        if m["role"] == "user")
        self.assertEqual(ctx.count("broke your concentration"), 2)


if __name__ == "__main__":
    unittest.main()
