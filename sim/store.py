"""Run persistence: an append-only events.jsonl per run (tau-bench style).

The scenario copy + events.jsonl make a run fully self-contained: replay is
fold(scenario, events) with zero LLM calls, so recorded runs are shareable,
deterministic, and free to re-inspect.
"""

import datetime
import json
import logging
import os

from .sim_time import wall_now


def new_run_dir(base="runs"):
    run_id = ("run-" + datetime.datetime.fromtimestamp(wall_now())
              .strftime("%Y%m%d-%H%M%S"))  # TELEMETRY: naming only
    path = os.path.join(base, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def setup_logging(run_dir):
    """Per-run structured log at runs/<id>/sim.log (DEBUG and up)."""
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger("sim")
    logger.setLevel(logging.DEBUG)
    marker = os.path.abspath(run_dir)
    if not any(getattr(h, "_sim_run", None) == marker for h in logger.handlers):
        fh = logging.FileHandler(os.path.join(run_dir, "sim.log"))
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"))
        fh._sim_run = marker
        logger.addHandler(fh)
    return logger


class RunStore:
    def __init__(self, run_dir, scenario=None):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        if scenario is not None:
            with open(os.path.join(run_dir, "scenario.json"), "w") as f:
                json.dump(scenario, f, indent=2)
            # copy the rubric too: runs stay self-contained and re-gradeable
            # even if rubrics/ changes later
            ref = scenario.get("rubric")
            if ref and os.path.exists(ref):
                with open(ref) as src, \
                     open(os.path.join(run_dir, "rubric.json"), "w") as dst:
                    dst.write(src.read())
        self.path = os.path.join(run_dir, "events.jsonl")
        self._f = open(self.path, "a")
        self._llm_f = open(os.path.join(run_dir, "llm.jsonl"), "a")

    def append(self, entry):
        self._f.write(json.dumps(entry) + "\n")
        self._f.flush()  # crash-safe: every event hits disk immediately

    def append_llm(self, entry):
        """Full LLM call trace: request, response, usage, latency."""
        self._llm_f.write(json.dumps(entry) + "\n")
        self._llm_f.flush()

    def close(self):
        self._f.close()
        self._llm_f.close()
