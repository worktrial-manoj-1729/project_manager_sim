"""The event queue: a min-heap keyed on (sim_time, seq).

Time only advances when an event is popped. The seq counter breaks ties
deterministically, so identical runs replay identically.
"""

import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class Event:
    time: int
    seq: int
    kind: str = field(compare=False)
    payload: dict = field(compare=False)


class EventQueue:
    def __init__(self):
        self._heap = []
        self._seq = 0

    def push(self, time, kind, payload=None):
        e = Event(time, self._seq, kind, payload or {})
        self._seq += 1
        heapq.heappush(self._heap, e)
        return e

    def pop(self):
        return heapq.heappop(self._heap)

    def peek(self):
        return self._heap[0] if self._heap else None

    def __len__(self):
        return len(self._heap)
