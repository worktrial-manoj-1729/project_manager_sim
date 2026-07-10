"""Tiny REPL for driving the simulation.

Usage: python -m sim.cli [scenarios/demo.json]
"""

import json
import os
import sys

from .engine import Engine
from .sim_time import fmt

HELP = """commands:
  say <npc> <message...>   send a chat message to an NPC
  wait                     advance sim time to the next event
  run                      advance until someone messages you back
  advance <minutes>        advance sim time by N minutes, firing events on the way
  addtask <title...>       track a new task on the board
  email <npc> <text...>    email someone (slower replies than chat)
  assign <task> <npc>      reassign a task (changes the real schedule)
  note <task> <text...>    correct a task's tracker note
  meet <ids,csv> <in_min> <dur_min> <topic...>   book a meeting (transcript at end)
  inbox                    show messages you've received
  people                   list NPCs
  tasks                    show the project task board (truth vs tracker)
  time                     show current sim time
  log                      show the raw event log
  quit                     exit
"""


def load_env(path=".env"):
    """Minimal .env loader (no dependency): sets vars not already in the environment."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def main():
    load_env()
    scenario_path = sys.argv[1] if len(sys.argv) > 1 else "scenarios/demo.json"
    with open(scenario_path) as f:
        scenario = json.load(f)

    engine = Engine(scenario)
    world = engine.world

    print("=== project_manager_sim v0 ===")
    print("scenario: %s | you are %s | time: %s"
          % (scenario_path, scenario.get("agent_name", "the PM"), world.now()))
    print("recording to %s (replay: python -m sim.replay %s)"
          % (engine.run_dir, engine.run_dir))
    print("people: " + ", ".join(
        "%s (%s, %s)" % (n.id, n.name, n.role) for n in world.npcs.values()))
    print("type 'help' for commands.\n")

    while True:
        try:
            raw = input("[%s] > " % world.now()).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        parts = raw.split(None, 2)
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "help":
            print(HELP)
        elif cmd == "time":
            print(world.now())
        elif cmd == "people":
            for n in world.npcs.values():
                print("  %s — %s, %s" % (n.id, n.name, n.role))
        elif cmd == "tasks":
            proj = world.project
            if not proj:
                print("  (no project loaded)")
            else:
                print("  %s [%s] due %s" % (proj["name"], proj.get("priority", "-"),
                                            fmt(proj["due"]) if proj.get("due") else "?"))
                for t in world.tasks_view():
                    print("    %-14s %-11s %3d%% (%.1f/%dh) -> done %s | tracker says: %s"
                          % (t["id"], t["status"], t["pct"], t["true_done_hours"],
                             t["effort_hours"], fmt(t["projected_done"]), t["reported"]))
        elif cmd == "inbox":
            msgs = world.agent_inbox()
            if not msgs:
                print("  (empty)")
            for m in msgs:
                print("  [%s] %s: %s" % (fmt(m.time), world.npcs[m.sender].name, m.text))
        elif cmd == "log":
            for entry in world.log:
                print("  [%s] %s" % (fmt(entry["t"]),
                                     {k: v for k, v in entry.items() if k != "t"}))
        elif cmd == "wait":
            if engine.step() is None:
                print("  (nothing scheduled)")
        elif cmd == "run":
            if engine.run_until_reply() is None:
                print("  (no reply arrived — is anything pending?)")
        elif cmd == "advance":
            try:
                minutes = int(parts[1])
            except (IndexError, ValueError):
                print("  usage: advance <minutes>")
                continue
            engine.advance_until(world.clock + minutes)
        elif cmd == "addtask":
            title = raw.split(None, 1)[1] if len(parts) > 1 else ""
            if not title:
                print("  usage: addtask <title...>")
            else:
                engine.agent_add_task({"id": title.lower().replace(" ", "-")[:40],
                                       "title": title})
        elif cmd == "email":
            if len(parts) < 3 or parts[1] not in world.npcs:
                print("  usage: email <npc> <text...>")
            else:
                engine.agent_say(parts[1], parts[2], via="email")
        elif cmd == "assign":
            if len(parts) < 3:
                print("  usage: assign <task_id> <npc>")
            else:
                result = engine.agent_assign_task(parts[1], parts[2])
                if isinstance(result, dict) and "error" in result:
                    print("  " + result["error"])
        elif cmd == "note":
            if len(parts) < 3:
                print("  usage: note <task_id> <text...>")
            else:
                result = engine.agent_update_note(parts[1], parts[2])
                if isinstance(result, dict) and "error" in result:
                    print("  " + result["error"])
        elif cmd == "meet":
            bits = raw.split(None, 4)
            if len(bits) < 5:
                print("  usage: meet <ids,csv> <in_min> <dur_min> <topic...>")
            else:
                try:
                    result = engine.agent_schedule_meeting(
                        bits[1].split(","), world.clock + int(bits[2]),
                        int(bits[3]), bits[4])
                    if isinstance(result, dict) and "error" in result:
                        print("  " + result["error"])
                except ValueError:
                    print("  usage: meet <ids,csv> <in_min> <dur_min> <topic...>")
        elif cmd == "say":
            if len(parts) < 3:
                print("  usage: say <npc> <message...>")
            elif parts[1] not in world.npcs:
                print("  unknown npc %r — try: %s" % (parts[1], ", ".join(world.npcs)))
            else:
                engine.agent_say(parts[1], parts[2])
        else:
            print("  unknown command %r — type 'help'" % cmd)


if __name__ == "__main__":
    main()
