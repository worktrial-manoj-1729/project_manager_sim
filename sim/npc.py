"""NPCs: pure appended-context agents — no tools, everything pushed.

THE NPC CONTRACT (DESIGN.md §6): they do their work (scheduler physics),
they experience (everything that happens TO them is pushed into their
stream when it happens), they speak — and speech never mutates world state.

Formulation:
- Each NPC holds a persistent CONTEXT: an append-only stream of everything
  it experienced — messages received, meeting lines heard, assignments,
  docs shared, its own utterances. Change awareness is automatic: events
  arrive in order, so "the meeting moved" is simply two entries.
- Every entry derives from logged world events, so the context is a fold of
  the log: reconstructible, replay-safe, no hidden state that the log
  doesn't explain.
- On its turn to speak (WHEN is still decided by deterministic rules), one
  LLM call over the stream produces text. No tools at all — everything the
  NPC can know is PUSHED into its stream when it happens (beliefs mature via
  authored realizations, invites/docs/completions/room lines arrive live).
- WHEN rules are unchanged and deterministic: response latency, heartbeats,
  beat conditions, meeting turn offers.
"""

import json
import logging
import os
import random
import re

import anthropic

from .sim_time import fmt, in_working_hours, next_work_start, wall_now

# NPCs are deliberately CHEAP — persona voice + local facts, Haiku-class.
# Override with SIM_NPC_MODEL (e.g. claude-sonnet-5) to A/B.
MODEL = os.environ.get("SIM_NPC_MODEL", "claude-haiku-4-5")
logger = logging.getLogger("sim.npc")



def _gen_kwargs():
    """Generation settings for NPC voices — determinism first.

    temperature=0: the same experience stream yields (near-)identical text,
    removing sampling variance across GRPO rollouts of the same task. (Not a
    bit-exact guarantee — API serving isn't perfectly deterministic — but it
    removes the dominant noise term.)

    The API requires temperature=1 when extended thinking is on, so a
    thinking-capable SIM_NPC_MODEL override trades determinism for depth —
    for training runs, keep NPCs on the non-thinking default."""
    if MODEL.startswith("claude-haiku"):
        return {"temperature": 0}
    return {"thinking": {"type": "adaptive"}}


def _sanitize(text):
    """Strip leaked '[Mon 09:42]'-style prefixes models imitate from context."""
    return re.sub(r"^\[[^\]]{0,40}\]\s*", "", text.strip()).strip() or text.strip()


class NPC:
    def __init__(self, spec, seed):
        self.id = spec["id"]
        self.name = spec["name"]
        self.role = spec["role"]
        self.persona = spec.get("persona", "")
        self.knowledge = list(spec.get("knowledge", []))
        # Per-NPC seeded RNG stream: same seed -> identical latencies on replay.
        self.rng = random.Random("%s:%s" % (seed, self.id))
        # THE EXPERIENCE STREAM: persistent, append-only agent context.
        # Every entry derives from a logged event (fold of the log).
        self.context = []
        self.unanswered_emails = 0  # answered in batch at the next wakeup

    # -- WHEN: pure rules, no LLM (unchanged) --------------------------------

    def response_time(self, t, via="chat"):
        """Absolute sim-time at which this NPC answers a CHAT message received
        at t. (Email is not scheduled at all — it waits, silently, for the
        recipient's next wakeup: batching IS what email is.)"""
        if in_working_hours(t):
            return t + self.rng.randint(5, 45)
        return next_work_start(t) + self.rng.randint(5, 40)

    def next_wakeup(self, t):
        """Heartbeat: the NPC checks in roughly every 2 hours during work hours."""
        nxt = t + self.rng.randint(90, 150)
        if in_working_hours(nxt):
            return nxt
        return next_work_start(nxt) + self.rng.randint(0, 20)

    # -- the experience stream ------------------------------------------------

    def notify(self, t, text):
        """Something happened to this NPC — append it to their experience.
        Cheap (no LLM); they'll see it whenever they next speak."""
        entry = "[%s] %s" % (fmt(t), text)
        # merge consecutive user entries to keep the context tidy
        if self.context and self.context[-1]["role"] == "user" \
                and isinstance(self.context[-1]["content"], str):
            self.context[-1]["content"] += "\n" + entry
        else:
            self.context.append({"role": "user", "content": entry})

    # -- the agent loop: read tools, then speak --------------------------------

    def _system(self, world):
        return (
            "You are {name}, {role} at {company}. This is your working life; "
            "behave as this person would.\n"
            "Persona: {persona}\n\n"
            "Private knowledge (background — share only what you'd "
            "realistically say):\n{knowledge}\n\n"
            "Your context is everything you have experienced, in order, each "
            "entry stamped [time] — messages, meetings, realizations about "
            "your own work. What you know is what happened to you; never "
            "assume facts your context contradicts.\n"
            "You talk to {agent}, the new project manager, by chat/email/"
            "meetings. Style: concise, natural, in character. NEVER include "
            "bracketed timestamps in what you say. You cannot take actions in "
            "the world — you can only speak."
        ).format(name=self.name, role=self.role,
                 company=world.scenario.get("company", "the company"),
                 persona=self.persona,
                 knowledge="\n".join("- " + k for k in self.knowledge) or "-",
                 agent=world.scenario.get("agent_name", "the PM"))

    def act(self, client, world, instruction):
        """One speaking turn: instruction enters the experience stream, one
        LLM call speaks. No tools — everything the NPC knows was pushed into
        its context when it happened. Returns the utterance (None on PASS)."""
        self.notify(world.clock, "((%s))" % instruction)
        t0 = wall_now()
        response = client.messages.create(
            model=MODEL, max_tokens=1024,
            system=self._system(world),
            messages=self.context,
            **_gen_kwargs())
        _log_llm(world, self.id, "act", response, t0,
                 "(experience stream: %d entries)" % len(self.context), None)
        text = _sanitize(next(
            (b.text for b in response.content if b.type == "text"),
            "(got distracted — no reply)"))
        self.context.append({"role": "assistant", "content": text})
        if text.strip().upper().rstrip(".") == "PASS":
            return None
        return text

    # -- speak paths (all funnel through act) ----------------------------------

    def reply(self, client, world, via="chat"):
        style = ("a thorough email reply" if via == "email"
                 else "a chat reply, 1-3 sentences")
        return self.act(client, world,
                        "It is %s. Reply now to the latest message(s) above — "
                        "write %s. Output only your message."
                        % (fmt(world.clock), style))

    def ping(self, client, world, intent):
        return self.act(client, world,
                        "It is %s. You've decided to reach out. Motivation: %s "
                        "React naturally to everything in your context — never "
                        "assume facts it contradicts. Output only your message."
                        % (fmt(world.clock), intent))

    def room_turn(self, client, world, meeting):
        return self.act(client, world,
                        "It is %s, you are in the meeting '%s' (agenda: %s). "
                        "It's your turn: speak 1-3 natural sentences to the "
                        "room, or output exactly PASS if you have nothing to "
                        "add right now."
                        % (fmt(world.clock), meeting["topic"],
                           meeting.get("agenda") or "none"))


def _log_llm(world, who, kind, response, t0, system, messages):
    """Write the LLM trace (llm.jsonl and sim.log)."""
    latency_ms = int((wall_now() - t0) * 1000)
    text = next((b.text for b in response.content if b.type == "text"), "")
    logger.debug(
        "llm %s kind=%s sim_t=%s latency=%dms in=%d out=%d stop=%s",
        who, kind, fmt(world.clock), latency_ms,
        response.usage.input_tokens, response.usage.output_tokens,
        response.stop_reason,
    )
    if world.store is not None:
        world.store.append_llm({
            "wall_ts": wall_now(),
            "sim_t": world.clock,
            "sim_t_fmt": fmt(world.clock),
            "npc": who,
            "kind": kind,
            "model": MODEL,
            "latency_ms": latency_ms,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "system": system if isinstance(system, str) else None,
            "response_text": text,
        })
    return text


def generate_transcript(client, world, meeting):
    """NPC-only meetings (nobody live in the room): lazy one-shot synthesis.
    PM-attended meetings never use this — their transcript is the verbatim
    minutes of the real exchange."""
    agent_name = world.scenario.get("agent_name", "the PM")
    company = world.scenario.get("company", "a small SaaS company")

    sections = []
    for aid in meeting["attendees"]:
        if aid == "agent":
            continue
        npc = world.npcs[aid]
        recent = world.chat_history(aid)[-8:]
        recent_txt = "\n".join(
            "  %s: %s" % (agent_name if m.sender == "agent" else npc.name, m.text)
            for m in recent) or "  (no prior chats)"
        sections.append(
            "%s — %s\nPersona: %s\nPrivate knowledge (they choose what to say "
            "aloud):\n%s\nRecent 1:1 chat with %s:\n%s"
            % (npc.name, npc.role, npc.persona,
               "\n".join("- " + k for k in npc.knowledge), agent_name, recent_txt))

    system = (
        "You write realistic workplace meeting transcripts for a simulation.\n"
        "Company: {company}. Simulated time: {now}.\n"
        "Meeting topic: {topic}\nAgenda: {agenda}\n\n"
        "Attendees:\n\n{sections}\n\n"
        "Rules: each character only says what they would realistically reveal "
        "IN FRONT OF the other attendees — private knowledge is not narrated, "
        "only spoken lines. Personas drive candor. Keep it tight: 8-16 short "
        "lines, format 'Name: line'. No stage directions, no summary."
    ).format(company=company, now=fmt(world.clock), topic=meeting["topic"],
             agenda=meeting.get("agenda") or "(none given)",
             sections="\n\n".join(sections))
    messages = [{"role": "user", "content": "Write the transcript now."}]

    t0 = wall_now()
    response = client.messages.create(
        model=MODEL, max_tokens=2048, system=system, messages=messages,
        **_gen_kwargs())
    return _log_llm(world, meeting["id"], "transcript", response, t0,
                    system, messages)
