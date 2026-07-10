"""World state. Single owner of the clock, messages, and the append-only event log."""

from dataclasses import dataclass

from .signals import delivery
from .sim_time import fmt


@dataclass
class Message:
    id: int
    time: int
    sender: str      # "agent" or an npc id
    recipient: str
    text: str
    via: str = "chat"   # "chat" | "email"


class World:
    def __init__(self, scenario, store=None):
        self.scenario = scenario
        self.start_time = scenario.get("start_time", 9 * 60 + 5)
        self._clock = self.start_time
        self.messages = []
        self.log = []          # append-only event log (dicts)
        self.npcs = {}         # id -> NPC
        self.store = store     # optional RunStore: persists every log entry
        self.project = scenario.get("project")
        # Dynamic task list: seed tasks + logged task_added mutations
        # (external OOD arrivals and agent-tracked tasks).
        # THREE LAYERS OF KNOWING (DESIGN.md §10): truth (engine/eval only),
        # holder belief (belief_pct/belief_note, matures via belief_update
        # events), tracker (reported — what's recorded). The initial belief
        # entry (no "at") applies at load.
        self.tasks = []
        for t in (self.project or {}).get("tasks", []):
            t = dict(t, arrival=self.start_time, source="seed")
            beliefs = t.get("belief") or []
            if beliefs and "at" not in beliefs[0]:
                t["belief_pct"] = beliefs[0].get("pct")
                t["belief_note"] = beliefs[0].get("note", "")
            self.tasks.append(t)
        self.completed_announced = set()  # completions made public so far
        self.meetings = []     # {id, start, end, attendees, topic, agenda}
        self.transcripts = []  # {meeting_id, t, attendees, topic, text}
        self.docs = []         # {id, title, content, shared_with, t}

    def add_task(self, spec, source):
        """Append a task (external arrival or agent-tracked) — a logged mutation."""
        spec = dict(spec)
        spec.setdefault("arrival", self.clock)
        spec["source"] = source
        base = spec.get("id") or "task"
        existing = {t["id"] for t in self.tasks}
        tid, n = base, 2
        while tid in existing:
            tid = "%s-%d" % (base, n)
            n += 1
        spec["id"] = tid
        self.tasks.append(spec)
        self.record("task_added", task=spec, source=source)
        return spec

    def update_task(self, task_id, changes, source):
        """Mutate a task (assignment, tracker note, urgency) — logged."""
        for t in self.tasks:
            if t["id"] == task_id:
                t.update(changes)
                self.record("task_updated", task_id=task_id,
                            changes=changes, source=source)
                return t
        return None

    def find_task(self, task_id):
        return next((t for t in self.tasks if t["id"] == task_id), None)

    # -- meetings / transcripts / docs --------------------------------------

    def add_meeting(self, start, end, attendees, topic, agenda=""):
        m = {"id": "mtg-%d" % len(self.meetings), "start": start, "end": end,
             "attendees": attendees, "topic": topic, "agenda": agenda}
        self.meetings.append(m)
        self.record("meeting_scheduled", **m)
        return m

    def active_meeting_with(self, who):
        """The meeting `who` is sitting in right now, if any."""
        for m in self.meetings:
            if who in m["attendees"] and m["start"] <= self.clock < m["end"]:
                return m
        return None

    def add_room_line(self, meeting, speaker, text):
        """One utterance in a live meeting — heard by every attendee, logged,
        and broadcast into every other attendee's experience stream."""
        meeting.setdefault("minutes", []).append(
            {"t": self.clock, "speaker": speaker, "text": text})
        self.record("room_line", meeting_id=meeting["id"], speaker=speaker,
                    text=text, topic=meeting["topic"])
        disp = (self.scenario.get("agent_name", "the PM") if speaker == "agent"
                else self.npcs[speaker].name)
        for a in meeting["attendees"]:
            if a != speaker and a in self.npcs:
                self.npcs[a].notify(self.clock, "in '%s', %s said: %s"
                                    % (meeting["topic"], disp, text))

    def add_transcript(self, meeting, text):
        tr = {"meeting_id": meeting["id"], "t": self.clock,
              "attendees": meeting["attendees"], "topic": meeting["topic"],
              "text": text}
        self.transcripts.append(tr)
        self.record("transcript", **tr)
        return tr

    def add_doc(self, title, content, shared_with):
        d = {"id": "doc-%d" % len(self.docs), "title": title,
             "content": content, "shared_with": list(shared_with), "t": self.clock}
        self.docs.append(d)
        self.record("doc_added", **d)
        return d

    def busy_by_assignee(self):
        """Time that doesn't go to tasks, per person — all derived, replayable:
        - meetings consume their full block for every attendee
        - each CHAT message received in working hours costs ~20 focus-minutes
          (refocus cost), and interruptions SERIALIZE — three pings in three
          minutes are three full distractions, not one. Email is exempt: it
          waits for the wakeup batch; that's its advantage.
        """
        from .sim_time import in_working_hours
        busy = {}
        for m in self.meetings:
            for a in m["attendees"]:
                busy.setdefault(a, []).append((m["start"], m["end"]))
        last_end = {}
        for msg in self.messages:
            if (msg.via == "chat" and msg.recipient != "agent"
                    and in_working_hours(msg.time)):
                start = max(msg.time, last_end.get(msg.recipient, 0))
                busy.setdefault(msg.recipient, []).append((start, start + 20))
                last_end[msg.recipient] = start + 20
        return busy

    def meetings_for(self, npc_id):
        return [m for m in self.meetings if npc_id in m["attendees"]]

    def transcripts_for(self, npc_id):
        """Transcripts of meetings this NPC attended — its cross-thread memory."""
        return [tr for tr in self.transcripts if npc_id in tr["attendees"]]

    def docs_for(self, npc_id):
        return [d for d in self.docs if npc_id in d["shared_with"]]

    def defer_for_meetings(self, npc_id, due, rng):
        """If a reply lands inside one of the NPC's meetings, push it after."""
        moved = True
        while moved:
            moved = False
            for m in self.meetings_for(npc_id):
                if m["start"] <= due < m["end"]:
                    due = m["end"] + rng.randint(2, 10)
                    moved = True
        return due

    # -- derived task state ---------------------------------------------------

    def tracker_view(self):
        """What the PM's board shows: the RECORD, never the truth. Structure
        (assignments, deps, priority) + reported notes + announced completions.
        No true progress, no scheduler projections — those are earned by
        asking people (beliefs) and observing completions (facts)."""
        out = []
        for t in self.tasks:
            if t.get("arrival", self.start_time) > self.clock:
                continue
            if t.get("filed") is False:
                continue  # announced to the PM but never filed: not on the board
            done = t["id"] in self.completed_announced
            out.append({
                "id": t["id"],
                "title": t["title"],
                "assignees": t.get("assignees", []),
                "priority": t.get("priority"),
                "urgent": bool(t.get("urgent")),
                "blocked_by": t.get("blocked_by", []),
                "source": t.get("source", "seed"),
                "status": "done" if done else
                          ("unowned" if not t.get("assignees") else "open"),
                "tracker_says": t.get("reported", ""),
            })
        return out

    def belief_view(self, npc_id):
        """A holder's honest current picture of THEIR OWN tasks — belief(t),
        which can be wrong; never the scheduler's truth."""
        out = []
        for t in self.tasks:
            if npc_id not in t.get("assignees", []):
                continue
            if t.get("arrival", self.start_time) > self.clock:
                continue
            if t["id"] in self.completed_announced:
                out.append({"task": t["title"], "your_view": "done"})
            elif t.get("belief_pct") is not None:
                out.append({"task": t["title"],
                            "your_view": "~%s%% — %s" % (t["belief_pct"],
                                                         t.get("belief_note", ""))})
            else:
                # no authored belief: the holder tracks this one accurately
                row = next((r for r in self.tasks_view() if r["id"] == t["id"]), None)
                pct = row["pct"] if row and row.get("pct") is not None else "?"
                out.append({"task": t["title"], "your_view": "~%s%% done" % pct})
        return out

    def tasks_view(self, at=None):
        """Ground-truth task states — DERIVED state: a pure function of time."""
        if not self.tasks:
            return []
        from .tasks import task_view
        return task_view(self.tasks, self.start_time,
                         self.clock if at is None else at, self.busy_by_assignee())

    def task_done(self, task_id, at=None):
        if not self.tasks:
            return False
        from .tasks import task_done
        return task_done(self.tasks, self.start_time,
                         self.clock if at is None else at, task_id,
                         self.busy_by_assignee())

    # -- the sim clock: one owner, one mutation point, monotonic ------------

    @property
    def clock(self):
        """Read-only everywhere. Sim minutes since Mon 00:00 — never wall time."""
        return self._clock

    def advance_clock_to(self, t):
        """THE single mutation point for simulated time. Monotonic by
        construction — going backwards raises instead of corrupting causality."""
        if t < self._clock:
            raise ValueError("sim clock must be monotonic: %d < %d"
                             % (t, self._clock))
        self._clock = t

    # -- event log ---------------------------------------------------------

    def record(self, kind, **data):
        entry = {"seq": len(self.log), "t": self.clock, "kind": kind}
        entry.update(data)
        d = delivery(kind, data.get("via"))
        if d is not None:
            entry["delivery"] = d  # push/pull semantics, per side (signals.py)
        self.log.append(entry)
        if self.store is not None:
            self.store.append(entry)
        return entry

    # -- messages ----------------------------------------------------------

    def send_message(self, sender, recipient, text, via="chat"):
        msg = Message(len(self.messages), self.clock, sender, recipient, text, via)
        self.messages.append(msg)
        # text is captured in the log: replay never needs the LLM again
        self.record("message", sender=sender, recipient=recipient,
                    msg_id=msg.id, text=text, via=via)
        return msg

    def chat_history(self, npc_id):
        """All messages between the agent and one NPC, in order."""
        pair = {("agent", npc_id), (npc_id, "agent")}
        return [m for m in self.messages if (m.sender, m.recipient) in pair]

    def agent_inbox(self):
        return [m for m in self.messages if m.recipient == "agent"]

    # -- display -----------------------------------------------------------

    def now(self):
        return fmt(self.clock)
