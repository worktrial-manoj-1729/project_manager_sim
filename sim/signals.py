"""Delivery semantics: for every interpersonal signal, whether it is PUSH
or PULL on each side. One table, consulted at the single log-mutation point
(World.record), so every recorded signal carries its own semantics.

Definitions (the four cells):

  sender PUSH     the sender emits the signal as a deliberate act, now
                  (types a chat, sends an email, speaks in a room)
  sender PULL     the sender exposes state passively; they never initiate —
                  others come and read it (a board entry, a belief when asked)

  recipient PUSH  the signal lands in the recipient's experience without them
                  asking. Push can INTERRUPT: a chat received in working
                  hours costs an NPC ~20 serialized focus-minutes, and a chat
                  or delivered email wakes the PM even mid-advance_time.
  recipient PUSH-BATCHED  delivered by an event, but on the recipient's batch
                  cadence rather than instantly (email). In a DES this IS how
                  "checking your inbox" exists: as a scheduled event — the
                  agent has no background thread, so nothing is ever pollable.
  recipient PULL  the recipient must come read a surface (the tracker board
                  via view_tasks); the surface never comes to them

The push/pull split is the physics behind channel economics, and it is
NEVER disclosed in tool descriptions — the agent learns it by exploration:
chat is instant but taxes the receiver, email arrives a batch-tick late but
is tax-free, and the board only speaks when read. All values are static
config — no LLM, no randomness — so the stamp is deterministic and
replay-stable.
"""

# kind (message kinds split by via) -> {sender, recipient}
DELIVERY = {
    # -- direct channels -----------------------------------------------------
    # chat: instant delivery, interrupts the receiver — an NPC pays the
    # serialized focus tax; the PM is woken even mid-advance_time
    "message/chat":      {"sender": "push", "recipient": "push"},
    # email: BATCHED push. In a DES nothing reaches anyone except by event,
    # so "the recipient checks their inbox" is itself an event: NPCs get the
    # batch at their next wakeup; the PM gets an email_delivery event on a
    # fixed batch grid (email_batch_minutes). Delayed, zero focus tax.
    "message/email":     {"sender": "push", "recipient": "push-batched"},

    # -- meetings ------------------------------------------------------------
    # invite lands on every attendee's calendar/stream
    "meeting_scheduled": {"sender": "push", "recipient": "push"},
    # a live utterance is heard by everyone in the room, immediately
    "room_line":         {"sender": "push", "recipient": "push"},
    # minutes are broadcast to attendees when the meeting ends
    "transcript":        {"sender": "push", "recipient": "push"},

    # -- artifacts -----------------------------------------------------------
    # sharing notifies each shared_with target's stream
    "doc_added":         {"sender": "push", "recipient": "push"},

    # -- the tracker board (the pull surface) --------------------------------
    # a task written to the board is found only by whoever checks it.
    # External arrivals never write the board directly: the ask reaches the
    # PM as a chat/email announcement carrying a ticket ref, and the PM must
    # file it (add_task) before it exists on the tracker or can be assigned
    "task_added":        {"sender": "push", "recipient": "pull"},
    # board edits are pull too — EXCEPT assignment, which additionally
    # pushes a notify into the assignee's stream (recorded separately)
    "task_updated":      {"sender": "push", "recipient": "pull"},
    # completions are announced publicly the moment they happen (delivered
    # to the PM with the next tool result, but a board broadcast never
    # WAKES anyone — only people do)
    "task_completed":    {"sender": "push", "recipient": "push"},
}


def delivery(kind, via=None):
    """Delivery stamp for a signal kind, or None for engine-internal events
    (scheduled, clock_advanced, npc_wakeup, ...) which signal nobody."""
    if kind == "message":
        return DELIVERY.get("message/%s" % (via or "chat"))
    return DELIVERY.get(kind)
