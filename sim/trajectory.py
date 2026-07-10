"""Render a run's transcript as a readable, annotated trajectory.

    python -m sim.trajectory runs/run-XXXXXXXX-XXXXXX

Folds transcript.jsonl into a human-readable timeline: what the PM thought,
what it did (tool calls with the key args inline), what came back (replies /
completions / errors), and — from the event log + scorecard — whether it was
actually LEARNING the world (discovering the migration slip, rerouting off the
overloaded owner, resisting the herrings, swarming the bottleneck). Read-only.
"""

import json
import os
import sys

from .sim_time import fmt


def _load(run_dir):
    tp = os.path.join(run_dir, "transcript.jsonl")
    rows = [json.loads(l) for l in open(tp)] if os.path.exists(tp) else []
    sc = json.load(open(os.path.join(run_dir, "scenario.json")))
    meta = {}
    mp = os.path.join(run_dir, "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    card = {}
    cp = os.path.join(run_dir, "scorecard.json")
    if os.path.exists(cp):
        card = json.load(open(cp))
    return rows, sc, meta, card


def _arg_summary(name, a):
    """The one or two args that matter for reading the trajectory."""
    a = a or {}
    keys = ("npc", "task_id", "task", "id", "title", "minutes", "t",
            "attendees", "meeting_id", "priority", "urgent")
    bits = []
    for k in keys:
        if k in a and a[k] not in (None, "", []):
            v = a[k]
            if isinstance(v, list):
                v = ",".join(map(str, v))
            bits.append("%s=%s" % (k, str(v)[:40]))
    return " ".join(bits)


def render(run_dir):
    rows, sc, meta, card = _load(run_dir)
    names = {n["id"]: n["name"] for n in sc["npcs"]}
    names["agent"] = sc.get("agent_name", "PM")

    print("=" * 78)
    print("TRAJECTORY  %s" % run_dir)
    print("  scenario=%s  probe=%s  model=%s"
          % (meta.get("task", "?"), meta.get("probe", "?"),
             meta.get("agent_model", "?")))
    print("=" * 78)

    for e in rows:
        role = e["role"]
        t = e.get("sim_t_fmt", "")
        if role == "system":
            continue
        if role == "user":
            txt = (e.get("content") or "")
            if txt.startswith("It is") and "over to you" in txt:
                print("\n[%s] >> week starts" % t)
            else:
                # a push handed to the PM mid-week
                snippet = txt.replace("\n", " ")
                print("\n[%s] << PUSH to PM: %s" % (t, snippet[:200]))
        elif role == "assistant":
            think = (e.get("thinking") or "").strip().replace("\n", " ")
            say = (e.get("content") or "").strip().replace("\n", " ")
            if think:
                print("[%s]   (thinks) %s" % (t, think[:220]))
            if say:
                print("[%s]   (says)   %s" % (t, say[:200]))
            for tc in e.get("tool_calls") or []:
                print("[%s]   -> %s(%s)"
                      % (t, tc["name"], _arg_summary(tc["name"], tc["arguments"])))
        elif role == "tool":
            c = e.get("content")
            note = _tool_note(e.get("name"), c, names)
            if note:
                print("[%s]      = %s" % (t, note))

    _learning_readout(sc, card, run_dir)


def _tool_note(name, content, names):
    """Surface only the trajectory-relevant part of a tool result."""
    if not isinstance(content, dict):
        return None
    if "error" in content:
        return "ERROR: %s" % content["error"]
    body = content.get("result", content)
    notes = content.get("notifications") or []
    out = []
    for n in notes:
        who = names.get(n.get("chat_from") or n.get("email_from"), "?")
        if "chat_from" in n or "email_from" in n:
            out.append("REPLY %s: %s" % (who, (n.get("text") or "").replace("\n", " ")[:160]))
        elif "completed" in n or "task_completed" in n:
            out.append("DONE: %s" % (n.get("task_id") or n.get("completed")))
    if name == "wait_for_reply" and isinstance(body, dict) and not out:
        return None
    return " | ".join(out) if out else None


def _learning_readout(sc, card, run_dir):
    """Did the PM learn the hidden world? Cross the event log against the
    scenario's planted latents (slip, herrings, owned arrivals, swarm)."""
    print("\n" + "=" * 78)
    print("LEARNING READOUT")
    print("=" * 78)
    events = []
    ep = os.path.join(run_dir, "events.jsonl")
    if os.path.exists(ep):
        events = [json.loads(l) for l in open(ep)]

    def agent_muts(kind):
        return [e for e in events if e.get("kind") == kind and e.get("source") == "agent"]

    reassigns = [e for e in events if e.get("kind") == "task_updated"
                 and e.get("source") == "agent"
                 and "assignees" in (e.get("changes") or {})]
    meetings = [e for e in events if e.get("kind") == "meeting_scheduled"]
    swarms = [m for m in meetings if m.get("task")]
    cancels = [e for e in events if e.get("kind") == "meeting_cancelled"]

    print("  reassignments (rerouting owners):  %d" % len(reassigns))
    for e in reassigns:
        print("     %s -> %s" % (e["task_id"], (e["changes"]["assignees"])))
    print("  swarm meetings booked:             %d" % len(swarms))
    for m in swarms:
        print("     '%s' on %s with %s" % (m.get("topic"), m.get("task"), m.get("attendees")))
    print("  meetings cancelled:                %d" % len(cancels))

    # blocking questions: did the PM notice the stall and answer it (fast)?
    from .world import World
    w = World(sc)
    for t in w.tasks:
        for q in t.get("questions") or []:
            if not q.get("gates"):
                continue
            ans = None
            for e in events:
                if (e.get("kind") == "message" and e.get("sender") == "agent"
                        and e.get("recipient") == (t.get("assignees") or [None])[0]
                        and e.get("t", 0) >= q["at"]):
                    ans = (e["t"], e.get("via", "chat"))
                    break
            from .sim_time import fmt as _fmt
            opened = _fmt(q["at"])
            if ans:
                print("  question '%s' on %s: opened %s -> ANSWERED %s via %s (%d min)"
                      % (q["id"], t["id"], opened, _fmt(ans[0]), ans[1], ans[0] - q["at"]))
            else:
                print("  question '%s' on %s: opened %s -> NEVER ANSWERED (task stalled)"
                      % (q["id"], t["id"], opened))

    # herrings: did the PM pour scarce capacity into the trap task?
    for h in sc.get("herrings", []):
        print("  herring [%s]: claim=%r" % (h["id"], h.get("claim")))

    if card:
        c = card.get("combined") or {}
        print("\n  SCORE: reward=%s  normalized=%s"
              % (card.get("reward"), c.get("normalized")))
        print("     agent=%s  in band [B=%s, OPT=%s]"
              % (c.get("agent"), c.get("baseline"), c.get("opt_ideal")))
        per = card.get("per_task_agent") or []
        for p in per:
            print("     %-22s done=%s  pct=%s  weight=%s"
                  % (p.get("id"), p.get("done"), p.get("pct"), p.get("weight")))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    render(sys.argv[1])


if __name__ == "__main__":
    main()
