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
    via: str = "chat"     # "chat" | "email"
    group: int = None     # shared id when ONE send has many recipients


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
            if beliefs:
                # a belief belongs to a PERSON (held_by) — their stale estimate,
                # pinned so it doesn't drift to whoever the PM reassigns it to.
                # Authored as a fraction of total effort (`remaining_frac`,
                # scale-invariant), surfaced as REMAINING work-hours; `pct` (%
                # done) stays for older scenarios.
                from .tasks import belief_hours_left
                t["belief_holder"] = beliefs[0].get("held_by") or \
                    (t.get("assignees") or [None])[0]
                if "at" not in beliefs[0]:
                    t["belief_pct"] = beliefs[0].get("pct")
                    t["belief_remaining"] = belief_hours_left(
                        beliefs[0], t.get("effort_hours"))
                    t["belief_note"] = beliefs[0].get("note", "")
            self.tasks.append(t)
        self.completed_announced = set()  # completions made public so far
        # meetings already on the calendar at week start (authored). A meeting
        # labelled to a task swarms it (boost); an unlabelled / wrong-room one
        # is pure overhead the PM should cancel. `deletable`-style flags aren't
        # needed — value is emergent from who's in the room vs the task.
        self.meetings = []     # {id, start, end, attendees, topic, agenda, task}
        for i, mm in enumerate(scenario.get("meetings", [])):
            self.meetings.append({
                "id": mm.get("id", "seed-mtg-%d" % i),
                "start": mm["start"], "end": mm["end"],
                "attendees": mm["attendees"], "topic": mm.get("topic", "meeting"),
                "agenda": mm.get("agenda", ""), "task": mm.get("task")})
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

    def add_meeting(self, start, end, attendees, topic, agenda="", task=None):
        m = {"id": "mtg-%d" % len(self.meetings), "start": start, "end": end,
             "attendees": attendees, "topic": topic, "agenda": agenda,
             "task": task}
        self.meetings.append(m)
        self.record("meeting_scheduled", **m)
        return m

    def cancel_meeting(self, meeting_id):
        """Mark a (future) meeting cancelled — logged so replay reconstructs it.
        Frees the attendees' block and voids the meeting's swarm deposit."""
        for m in self.meetings:
            if m["id"] == meeting_id:
                m["cancelled"] = True
                self.record("meeting_cancelled", meeting_id=meeting_id)
                return m
        return None

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
        - each CHAT message received by a WORKER in working hours costs a
          refocus tax (`costs.chat_interrupt_minutes`, default 20 min), and
          interruptions SERIALIZE — three pings in three minutes are three full
          distractions, not one. Email is exempt: it waits for the wakeup
          batch; that's its advantage. Setting the tax to 0 makes chat free.

        The tax applies ONLY to workers (people with scored tasks): a
        stakeholder you report to (worker:false) owns no work, so interrupting
        them consumes no schedulable capacity — messaging the VP is free, and
        the model says so rather than charging a tax it would then ignore."""
        from .sim_time import in_working_hours
        tax = self.scenario.get("costs", {}).get("chat_interrupt_minutes", 20)
        workers = {n["id"] for n in self.scenario.get("npcs", [])
                   if n.get("worker", True)}
        busy = {}
        for m in self.meetings:
            if m.get("cancelled"):
                continue  # a cancelled meeting never happens — frees the block
            for a in m["attendees"]:
                busy.setdefault(a, []).append((m["start"], m["end"]))
        last_end = {}
        for msg in self.messages:
            if (msg.via == "chat" and msg.recipient in workers and tax
                    and in_working_hours(msg.time)):
                start = max(msg.time, last_end.get(msg.recipient, 0))
                busy.setdefault(msg.recipient, []).append((start, start + tax))
                last_end[msg.recipient] = start + tax
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
        which can be wrong; never the scheduler's truth. The stale estimate
        belongs to a PERSON (belief_holder): only they carry the blind spot, so
        a fresh owner the PM reassigns to just tracks reality. Estimates are in
        REMAINING work-hours where authored (decision-relevant, owner-invariant),
        falling back to a % for older scenarios."""
        out = []
        for t in self.tasks:
            if npc_id not in t.get("assignees", []):
                continue
            if t.get("arrival", self.start_time) > self.clock:
                continue
            if t["id"] in self.completed_announced:
                out.append({"task": t["title"], "your_view": "done"})
                continue
            holds = npc_id == t.get("belief_holder")
            if holds and t.get("belief_remaining") is not None:
                view = "~%.0fh of work left — %s" % (t["belief_remaining"],
                                                     t.get("belief_note", ""))
            elif holds and t.get("belief_pct") is not None:
                view = "~%s%% — %s" % (t["belief_pct"], t.get("belief_note", ""))
            else:
                # not the blind-spot holder (or no belief): track accurately,
                # reported as remaining hours to keep the picture in one unit
                row = next((r for r in self.tasks_view() if r["id"] == t["id"]), None)
                td = row["true_done_hours"] if row and row.get("true_done_hours") \
                    is not None else t.get("done_hours", 0)
                left = max(0.0, t.get("effort_hours", 0) - td)
                view = "~%.0fh of work left" % left
            out.append({"task": t["title"], "your_view": view})
        return out

    def skills(self):
        """Per-person skill maps ({person: {tag: factor}}) from the scenario —
        frozen config the SCHEDULER uses to speed up specialists. Hidden from
        the PM (never surfaced on the tracker); inferred from completions."""
        return {n["id"]: n["skills"] for n in self.scenario.get("npcs", [])
                if n.get("skills")}

    def meeting_deposits(self):
        """{task_id: [(meeting_end, effort_minutes)]} — a meeting labelled to a
        task, whose OWNER attends, banks a chunk of collaborative work into that
        task at its end: work-minutes x swarm_rate(attendees). Cancelled
        meetings bank nothing. This is the meeting-as-swarm mechanism, derived
        (replayable) from the meeting list + hidden skills."""
        from .tasks import physics_of, swarm_rate, work_minutes_between
        sk = self.skills()
        phys = physics_of(self.scenario)
        out = {}
        for m in self.meetings:
            tid = m.get("task")
            if not tid or m.get("cancelled"):
                continue
            t = self.find_task(tid)
            if t is None:
                continue
            owner = (t.get("assignees") or [None])[0]
            if owner not in m["attendees"]:
                continue  # the doer wasn't in the room — no work happens on T
            dur = work_minutes_between(m["start"], m["end"])
            dep = dur * swarm_rate(t, m["attendees"], sk, phys)
            if dep > 0:
                out.setdefault(tid, []).append((m["end"], dep))
        return out

    def question_answers(self):
        """{task_id: {question_id: answered_at}} — when the PM's reply to each
        BLOCKING question was DELIVERED to its owner. Derived (replayable) from
        the message log + config, like meeting_deposits(): the FIRST agent->owner
        message at/after a question's open-time answers it, and the delivery time
        is the channel latency — chat lands instantly, email waits for the next
        answer-batch tick (the async cost the PM is trading against the chat
        focus-tax). A question with no reply yet stays unanswered (stalled)."""
        batch = self.scenario.get("answer_batch_minutes", 120)
        out = {}
        for t in self.tasks:
            owner = (t.get("assignees") or [None])[0]
            for q in t.get("questions") or []:
                if not q.get("gates"):
                    continue
                reply = next((m for m in self.messages
                              if m.sender == "agent" and m.recipient == owner
                              and m.time >= q["at"]), None)
                if reply is None:
                    continue
                delivered = (reply.time if reply.via == "chat"
                             else (reply.time // batch + 1) * batch)
                out.setdefault(t["id"], {})[q["id"]] = delivered
        return out

    def tasks_view(self, at=None):
        """Ground-truth task states — DERIVED state: a pure function of time."""
        if not self.tasks:
            return []
        from .tasks import physics_of, task_view
        return task_view(self.tasks, self.start_time,
                         self.clock if at is None else at,
                         self.busy_by_assignee(), self.skills(),
                         self.meeting_deposits(), answers=self.question_answers(),
                         physics=physics_of(self.scenario))

    def task_done(self, task_id, at=None):
        if not self.tasks:
            return False
        from .tasks import physics_of, task_done
        return task_done(self.tasks, self.start_time,
                         self.clock if at is None else at, task_id,
                         self.busy_by_assignee(), self.skills(),
                         self.meeting_deposits(), answers=self.question_answers(),
                         physics=physics_of(self.scenario))

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

    def send_message(self, sender, recipient, text, via="chat", group=None):
        """`group`: one send, many recipients — every copy carries the same
        explicit group id in the log, so tooling never infers batches from
        timestamps or text matching."""
        msg = Message(len(self.messages), self.clock, sender, recipient, text,
                      via, group)
        self.messages.append(msg)
        # text is captured in the log: replay never needs the LLM again
        self.record("message", sender=sender, recipient=recipient,
                    msg_id=msg.id, text=text, via=via, group=group)
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
