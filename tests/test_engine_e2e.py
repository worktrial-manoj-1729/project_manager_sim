"""End-to-end engine flows on the stub LLM: arrivals as communication,
filing, the org fallback, batched email, wake semantics, liveness, replay."""

import unittest

from sim.replay import load_run, replay
from sim.tools import call_tool

from tests.fixtures import HORIZON, goto, make_engine, unwrap


def advance_collect(eng, target):
    """Advance to an absolute time, collecting every push notification."""
    notifs = []
    while eng.world.clock < target:
        r = call_tool(eng, "advance_time",
                      {"minutes": target - eng.world.clock})
        _, n = unwrap(r)
        notifs += n
    return notifs


class TestArrivalAndFiling(unittest.TestCase):
    def test_chat_announce_wakes_with_ticket_ref(self):
        eng = make_engine()
        notifs = advance_collect(eng, 2100)  # incident announced at 2040
        self.assertTrue(any(n.get("chat_from") == "bo"
                            and "(ticket: incident)" in n.get("text", "")
                            for n in notifs))
        # the proactive belief confession (t=2000) is a push too
        self.assertTrue(any(n.get("chat_from") == "ana" for n in notifs))
        self.assertEqual(eng.world.find_task("core")["belief_pct"], 40)

    def test_unfiled_hidden_and_unassignable_until_filed(self):
        eng = make_engine()
        goto(eng, 2100)
        board = {t["id"] for t in eng.world.tracker_view()}
        self.assertNotIn("incident", board)
        r, _ = unwrap(call_tool(eng, "assign_task",
                                {"task_id": "incident", "npc": "bo"}))
        self.assertIn("error", r)
        r, _ = unwrap(call_tool(eng, "add_task",
                                {"title": "Incident", "id": "incident"}))
        self.assertEqual(r.get("filed"), "incident")
        r, _ = unwrap(call_tool(eng, "assign_task",
                                {"task_id": "incident", "npc": "bo"}))
        self.assertEqual(r.get("assigned"), "incident")
        # tracker-shaped ack only — never the truth dict
        self.assertNotIn("belief", r)
        self.assertNotIn("effort_hours", r)

    def test_preempting_skips_the_fallback(self):
        eng = make_engine()
        goto(eng, 2100)
        call_tool(eng, "add_task", {"title": "Incident", "id": "incident"})
        call_tool(eng, "assign_task", {"task_id": "incident", "npc": "ana"})
        goto(eng, 2300)  # past fallback.at=2200
        self.assertEqual(eng.world.find_task("incident")["assignees"], ["ana"])
        org = [e for e in eng.world.log if e["kind"] == "task_updated"
               and e.get("source") == "org"]
        self.assertEqual(org, [])

    def test_fallback_fires_when_unowned(self):
        eng = make_engine()
        goto(eng, 2300)
        t = eng.world.find_task("incident")
        self.assertEqual(t["assignees"], ["bo"])
        self.assertTrue(t["filed"])
        self.assertEqual(t["assigned_at"], 2200)  # picked up AT the fallback


class TestChannels(unittest.TestCase):
    def test_email_arrival_delivers_on_batch_grid(self):
        eng = make_engine()
        notifs = advance_collect(eng, 3600)  # email arrival 3540 -> tick 3570
        mail = [n for n in notifs if "email_from" in n]
        self.assertTrue(any("(ticket: form)" in n["text"] for n in mail))
        delivered = [e for e in eng.world.log if e["kind"] == "email_delivered"]
        self.assertEqual(delivered[0]["t"], 3570)

    def test_completions_deliver_but_never_wake(self):
        eng = make_engine()  # bo's side task (6h) completes Mon ~15:05
        r, notifs = unwrap(call_tool(eng, "advance_time", {"minutes": 655}))
        self.assertNotIn("interrupted", r)  # a board broadcast is not a person
        self.assertTrue(any("tracker" in n for n in notifs))

    def test_chat_wakes_mid_advance(self):
        eng = make_engine()
        r, _ = unwrap(call_tool(eng, "advance_time", {"minutes": 3000}))
        self.assertTrue(r.get("interrupted"))       # woken by a person
        self.assertLess(eng.world.clock, 545 + 3000)


class TestLiveness(unittest.TestCase):
    def test_unknown_recipients_bounce_not_crash(self):
        eng = make_engine()
        n_msgs = len(eng.world.messages)
        r, _ = unwrap(call_tool(eng, "send_email",
                                {"to": ["ghost@nowhere"], "subject": "s",
                                 "body": "b"}))
        self.assertEqual(r, {"sent": False, "bounced": ["ghost@nowhere"]})
        r, _ = unwrap(call_tool(eng, "send_chat", {"npc": "ghost", "text": "x"}))
        self.assertIn("error", r)
        self.assertEqual(len(eng.world.messages), n_msgs)  # nothing delivered

    def test_reprioritize_tool(self):
        eng = make_engine()
        r, _ = unwrap(call_tool(eng, "reprioritize",
                                {"task_id": "side", "priority": "P0"}))
        self.assertEqual(r.get("order_priority"), "P0")
        self.assertEqual(eng.world.find_task("side")["priority"], "P2")
        r, _ = unwrap(call_tool(eng, "reprioritize",
                                {"task_id": "side", "priority": "P9"}))
        self.assertIn("error", r)
        r, _ = unwrap(call_tool(eng, "reprioritize", {"task_id": "incident"}))
        self.assertIn("error", r)  # unfiled ticket isn't on the board

    def test_stakeholders_take_no_tickets(self):
        eng = make_engine()
        r, _ = unwrap(call_tool(eng, "assign_task",
                                {"task_id": "side", "npc": "vp"}))
        self.assertIn("stakeholder", r.get("error", ""))


class TestReplay(unittest.TestCase):
    def test_replay_reconstructs_the_live_world(self):
        """Regression: replay once dropped Message.via, so the evaluator taxed
        email runs as chat. The fold must reproduce the live world exactly."""
        eng = make_engine()
        goto(eng, 2100)
        call_tool(eng, "add_task", {"title": "Incident", "id": "incident"})
        call_tool(eng, "assign_task", {"task_id": "incident", "npc": "bo"})
        call_tool(eng, "send_email", {"to": ["ana"], "subject": "s", "body": "b"})
        goto(eng, HORIZON)

        world2 = replay(*load_run(eng.run_dir))
        self.assertEqual(eng.world.messages, world2.messages)  # incl. via
        self.assertEqual(eng.world.tracker_view(), world2.tracker_view())
        self.assertEqual([e for e in eng.world.log],
                         [e for e in world2.log])
        # every interpersonal signal is stamped with delivery semantics
        for e in world2.log:
            if e["kind"] == "message":
                self.assertIn("delivery", e)


if __name__ == "__main__":
    unittest.main()
