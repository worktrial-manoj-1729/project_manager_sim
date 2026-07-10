"""Scheduler physics (pure functions): ordering, assigned_at, the focus tax."""

import unittest

from sim.tasks import task_view
from sim.world import Message, World

from tests.fixtures import scenario

START = 545


def _view(tasks, now, busy=None):
    return {r["id"]: r for r in task_view(tasks, START, now, busy or {})}


class TestOrdering(unittest.TestCase):
    def test_priority_beats_creation_order(self):
        """Regression: personal queues were (urgent, creation order) only —
        a P2 created first would run before a P1."""
        tasks = [
            {"id": "p2", "title": "p2", "assignees": ["bo"], "effort_hours": 4,
             "priority": "P2", "arrival": START},
            {"id": "p1", "title": "p1", "assignees": ["bo"], "effort_hours": 4,
             "priority": "P1", "arrival": START},
        ]
        v = _view(tasks, START + 1)
        self.assertLess(v["p1"]["projected_done"], v["p2"]["projected_done"])

    def test_urgent_beats_priority(self):
        tasks = [
            {"id": "p1", "title": "p1", "assignees": ["bo"], "effort_hours": 4,
             "priority": "P1", "arrival": START},
            {"id": "p2u", "title": "p2u", "assignees": ["bo"], "effort_hours": 4,
             "priority": "P2", "urgent": True, "arrival": START},
        ]
        v = _view(tasks, START + 1)
        self.assertLess(v["p2u"]["projected_done"], v["p1"]["projected_done"])

    def test_blocker_serializes(self):
        tasks = [
            {"id": "a", "title": "a", "assignees": ["ana"], "effort_hours": 4,
             "priority": "P0", "arrival": START},
            {"id": "b", "title": "b", "assignees": ["bo"], "effort_hours": 2,
             "priority": "P0", "blocked_by": ["a"], "arrival": START},
        ]
        v = _view(tasks, START + 1)
        self.assertGreaterEqual(v["b"]["projected_done"], v["a"]["projected_done"])


class TestReprioritize(unittest.TestCase):
    def test_order_override_changes_scheduling_only(self):
        """The PM can reorder a queue (order_priority/order_urgent) but the
        authored priority keeps the rubric weight — no value minting."""
        tasks = [
            {"id": "p1", "title": "p1", "assignees": ["bo"], "effort_hours": 4,
             "priority": "P1", "arrival": START},
            {"id": "p2", "title": "p2", "assignees": ["bo"], "effort_hours": 4,
             "priority": "P2", "order_priority": "P0", "arrival": START},
        ]
        v = _view(tasks, START + 1)
        self.assertLess(v["p2"]["projected_done"], v["p1"]["projected_done"])
        self.assertEqual(v["p2"]["priority"], "P2")  # scoring weight untouched

    def test_order_urgent_override(self):
        tasks = [
            {"id": "u", "title": "u", "assignees": ["bo"], "effort_hours": 4,
             "priority": "P2", "urgent": True, "order_urgent": False,
             "arrival": START},
            {"id": "n", "title": "n", "assignees": ["bo"], "effort_hours": 4,
             "priority": "P2", "arrival": START},
        ]
        v = _view(tasks, START + 1)  # de-urgented: creation order decides
        self.assertLess(v["u"]["projected_done"], v["n"]["projected_done"])
        tasks[0]["order_urgent"] = True
        tasks[1]["order_urgent"] = False
        v = _view(tasks, START + 1)
        self.assertLess(v["u"]["projected_done"], v["n"]["projected_done"])


class TestAssignedAt(unittest.TestCase):
    def test_no_retroactive_credit(self):
        """Regression: assignment used to be retroactive — a 14:05 pickup was
        credited work since the 09:05 arrival, making acting early worthless."""
        base = {"id": "x", "title": "x", "assignees": ["ana"],
                "effort_hours": 6, "priority": "P1", "arrival": START}
        now = START + 360  # Mon 15:05
        early = _view([dict(base)], now)["x"]
        late = _view([dict(base, assigned_at=START + 300)], now)["x"]  # 14:05
        self.assertGreater(early["true_done_hours"], late["true_done_hours"])
        self.assertLess(early["projected_done"], late["projected_done"])
        # late pickup works exactly (now - assigned_at) minutes, no more
        self.assertAlmostEqual(late["true_done_hours"], 1.0, places=1)


class TestFocusTax(unittest.TestCase):
    def _world(self, costs=None):
        scn = scenario()
        if costs is not None:
            scn["costs"] = costs
        return World(scn)

    def test_chats_serialize(self):
        w = self._world()
        w.messages.append(Message(0, 600, "agent", "ana", "a", "chat"))
        w.messages.append(Message(1, 601, "agent", "ana", "b", "chat"))
        self.assertEqual(w.busy_by_assignee()["ana"], [(600, 620), (620, 640)])

    def test_email_agent_and_offhours_exempt(self):
        w = self._world()
        w.messages.append(Message(0, 600, "agent", "ana", "a", "email"))
        w.messages.append(Message(1, 600, "ana", "agent", "b", "chat"))
        w.messages.append(Message(2, 400, "agent", "ana", "c", "chat"))  # 06:40
        self.assertEqual(w.busy_by_assignee(), {})

    def test_tax_is_a_config_knob(self):
        w = self._world(costs={"chat_interrupt_minutes": 0})
        w.messages.append(Message(0, 600, "agent", "ana", "a", "chat"))
        self.assertEqual(w.busy_by_assignee(), {})


if __name__ == "__main__":
    unittest.main()
