"""Shared test fixture: a small, self-contained scenario.

Deliberately NOT scenarios/demo.json — tuning the shipped scenarios must
never break the test suite. Everything runs on sim.eval.StubClient: zero
LLM calls, fully deterministic.

Cast: two workers (ana, bo) + one stakeholder (vp, worker:false).
Seeds: core (P0, ana, 20h/8 done, belief 60% -> confession at Tue 09:20),
       side (P2, bo, 6h).
Arrivals: incident (P1 urgent, chat Tue 10:00, fallback bo Tue 12:40),
          form (P1, email Wed 11:00 -> batch tick 11:30, fallback bo Thu 11:20).
Horizon: Fri 17:00 (6780).
"""

import copy
import tempfile

from sim.engine import Engine
from sim.eval import StubClient
from sim.tools import call_tool

HORIZON = 6780

_SCENARIO = {
    "seed": 7,
    "company": "Testco",
    "agent_name": "Alex",
    "start_time": 545,
    "evaluation": {"horizon": HORIZON,
                   "task_value": {"alpha": 0.5, "gamma": 0.5}},
    "project": {
        "id": "proj", "name": "Test project", "priority": "P0", "due": HORIZON,
        "tasks": [
            {"id": "core", "title": "Core work", "assignees": ["ana"],
             "effort_hours": 20, "done_hours": 8, "priority": "P0",
             "belief": [{"pct": 60, "note": "on track"},
                        {"at": 2000, "pct": 40,
                         "note": "worse than I thought",
                         "proactive_ping": True}]},
            {"id": "side", "title": "Side work", "assignees": ["bo"],
             "effort_hours": 6, "done_hours": 0, "priority": "P2"},
        ],
    },
    "task_arrivals": [
        {"at": 2040, "npc": "bo",
         "announce": "Tell Alex an incident just landed and needs an owner.",
         "task": {"id": "incident", "title": "Incident", "effort_hours": 4,
                  "priority": "P1", "urgent": True},
         "fallback": {"npc": "bo", "at": 2200}},
        {"at": 3540, "npc": "bo", "via": "email",
         "announce": "Email Alex about the compliance form due this week.",
         "task": {"id": "form", "title": "Compliance form", "effort_hours": 2,
                  "priority": "P1"},
         "fallback": {"npc": "bo", "at": 5000}},
    ],
    "beats": [],
    "npcs": [
        {"id": "ana", "name": "Ana", "role": "Engineer",
         "persona": "terse", "knowledge": ["You own the core work."]},
        {"id": "bo", "name": "Bo", "role": "Engineer",
         "persona": "cheery", "knowledge": ["You own the side work."]},
        {"id": "vp", "name": "Vee", "role": "VP (stakeholder)", "worker": False,
         "persona": "exec", "knowledge": ["You are a stakeholder."]},
    ],
}


def scenario():
    return copy.deepcopy(_SCENARIO)


def make_engine(scn=None, run_dir=None):
    """Engine on the stub LLM in a throwaway run dir."""
    run_dir = run_dir or tempfile.mkdtemp(prefix="pmsim-test-")
    return Engine(scn or scenario(), client=StubClient(), verbose=False,
                  run_dir=run_dir)


def unwrap(result):
    """call_tool results may carry push notifications: {'result', 'notifications'}."""
    if isinstance(result, dict) and "notifications" in result:
        return result["result"], result["notifications"]
    return result, []


def goto(engine, t):
    """Advance THROUGH interrupts to an absolute sim time."""
    while engine.world.clock < t:
        call_tool(engine, "advance_time", {"minutes": t - engine.world.clock})
