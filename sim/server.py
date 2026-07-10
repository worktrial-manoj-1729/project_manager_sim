"""Local web server for driving and visualizing the sim.

    python -m sim.server [scenarios/demo.json] [port]          # own engine
    python -m sim.server --watch runs/run-XXXXXXXX-XXXXXX [port]  # observe ANY
        run (live agent runs in other processes, finished runs) by tailing its
        events.jsonl — strictly read-only; mutation endpoints are disabled.

API:
  GET  /                    -> web/index.html
  GET  /api/meta            -> scenario info (agent, npcs, run_dir)
  GET  /api/events?since=N  -> log entries with seq >= N, plus clock + pending
  POST /api/say             -> {"npc": id, "text": str}   (agent action)
  POST /api/step            -> advance to next event
  POST /api/run             -> advance until an NPC replies (blocking; LLM call)
"""

import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .cli import load_env
from .engine import Engine
from .sim_time import fmt
from .tools import call_tool, schemas

WEB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")

ENGINE = None
WATCHER = None
HUB = None
LOCK = threading.Lock()  # serializes mutations; reads are lock-free (append-only log)


# agent-cost pricing, $/MTok (in, out) — TELEMETRY for the Pareto axis
PRICES = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
    # OpenAI models: fill in when arms activate
}


def load_transcript(run_dir):
    """The agent conversation for a run, OpenAI chat format. Prefers the
    untruncated transcript.jsonl the harness writes; for runs that predate
    it, reconstructs assistant turns from llm.jsonl (tool OUTPUTS were not
    recorded back then and are marked as such)."""
    if not run_dir:
        return []
    tp = os.path.join(run_dir, "transcript.jsonl")
    if os.path.exists(tp):
        out = []
        with open(tp) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    lp = os.path.join(run_dir, "llm.jsonl")
    out = []
    if os.path.exists(lp):
        with open(lp) as f:
            for line in f:
                if '"agent_turn"' not in line:
                    continue
                e = json.loads(line)
                out.append({"role": "assistant", "sim_t": e.get("sim_t"),
                            "sim_t_fmt": e.get("sim_t_fmt"),
                            "content": e.get("response_text", ""),
                            "tool_calls": [
                                {"name": c["name"], "arguments": c["input"]}
                                for c in e.get("tool_calls", [])]})
                for c in e.get("tool_calls", []):
                    out.append({"role": "tool", "name": c["name"],
                                "sim_t": e.get("sim_t"),
                                "content": "(output not recorded — run "
                                           "predates transcript.jsonl)"})
    return out


class RunHub:
    """One dashboard, every run: lists runs/ with labels (agent model, live
    status, score, cost) and serves any of them read-only via ?run=<id>."""

    def __init__(self, base="runs"):
        self.base = base
        self._watchers = {}
        self._cache = {}  # rid -> (mtime_key, record)

    def _agent_cost(self, d):
        """Agent-side tokens -> $ from llm.jsonl (kind=agent_turn only). Prices
        PROMPT-CACHE tokens correctly: a cache READ bills 0.1x input, a cache
        WRITE 1.25x — so a cached run's cost isn't undercounted just because its
        `input_tokens` (uncached-only) went to ~nothing. `tin` reported is the
        effective input volume (uncached + read + write) for the tokens column."""
        lp = os.path.join(d, "llm.jsonl")
        if not os.path.exists(lp):
            return None, 0, 0, 0
        tin = tout = turns = 0
        cost_in = 0.0
        model = None
        with open(lp) as f:
            for line in f:
                if '"agent_turn"' not in line:
                    continue
                e = json.loads(line)
                turns += 1
                model = e.get("model")
                u = e["usage"]
                raw = u["input_tokens"]
                rd = u.get("cache_read_input_tokens", 0) or 0
                wr = u.get("cache_creation_input_tokens", 0) or 0
                pin, _ = PRICES.get((model or "").split(" ")[0], (0.0, 0.0))
                cost_in += (raw + wr * 1.25 + rd * 0.1) / 1e6 * pin
                tin += raw + rd + wr
                tout += u["output_tokens"]
        if turns == 0:
            return 0.0, 0, 0, 0  # scripted/null: zero agent cost
        _, pout = PRICES.get((model or "").split(" ")[0], (0.0, 0.0))
        return round(cost_in + tout / 1e6 * pout, 4), tin, tout, turns

    def list_runs(self):
        import glob
        import time as _t  # mtime freshness = liveness heuristic (telemetry)
        out = []
        for d in sorted(glob.glob(os.path.join(self.base, "run-*")), reverse=True):
            if not os.path.exists(os.path.join(d, "scenario.json")):
                continue
            # only harness/bench runs (meta.json) — interactive engine runs
            # and API-test debris don't belong on the dashboard
            if not os.path.exists(os.path.join(d, "meta.json")):
                continue
            rid = os.path.basename(d)
            ep = os.path.join(d, "events.jsonl")
            sp = os.path.join(d, "scorecard.json")
            key = tuple(os.path.getmtime(p) if os.path.exists(p) else 0
                        for p in (ep, sp, os.path.join(d, "llm.jsonl"),
                                  os.path.join(d, "meta.json")))
            cached = self._cache.get(rid)
            if cached and cached[0] == key:
                rec = dict(cached[1])
            else:
                label, score, completion, efficiency = "", None, None, None
                probe, task, done_rate, fairness, band = "", "", None, None, None
                mp = os.path.join(d, "meta.json")
                if os.path.exists(mp):
                    with open(mp) as f:
                        m = json.load(f)
                    label = m.get("agent_model") or m.get("probe", "")
                    probe = m.get("probe", "")
                    task = m.get("task", "")
                if not task:
                    # older runs: task id lives in the copied scenario
                    try:
                        with open(os.path.join(d, "scenario.json")) as f:
                            task = (json.load(f).get("project") or {}).get("id", "")
                    except (OSError, ValueError):
                        task = ""
                # RE-SCORE from the immutable events.jsonl with CURRENT code —
                # never trust the cached scorecard.json (written by whatever code
                # ran the rollout; a re-stamp or scoring change makes it stale,
                # the exact false-signal trap sim.bench also guards against).
                # Gate on scorecard.json EXISTING = the run FINISHED (don't score
                # a partial in-progress run — that keeps it flagged `live`); the
                # mtime cache means this fresh re-score happens once per run.
                if os.path.exists(sp):
                    try:
                        from .eval import evaluate
                        sc = evaluate(d)
                        score = sc.get("score")
                        if isinstance(sc.get("completion"), dict):
                            completion = sc["completion"].get("normalized")
                            efficiency = sc["efficiency"].get("normalized")
                        if isinstance(sc.get("done_weight_rate"), dict):
                            done_rate = sc["done_weight_rate"].get("agent")
                        if isinstance(sc.get("workload_fairness"), dict):
                            fairness = sc["workload_fairness"].get("agent")
                        if isinstance(sc.get("combined"), dict):
                            band = sc["combined"].get("available")
                    except Exception as ex:
                        # a live run scoring mid-flight is normal; a FINISHED
                        # run failing is not — say so, and never cache it
                        print("hub: could not score %s: %r" % (rid, ex))
                        score = None
                n = 0
                if os.path.exists(ep):
                    with open(ep) as f:
                        n = sum(1 for _ in f)
                cost, tin, tout, turns = self._agent_cost(d)
                rec = {"id": rid, "label": label or "(interactive/other)",
                       "probe": probe, "task": task, "band": band,
                       "events": n, "score": score, "completion": completion,
                       "efficiency": efficiency, "done_rate": done_rate,
                       "fairness": fairness, "cost_usd": cost,
                       "tokens_in": tin, "tokens_out": tout, "agent_turns": turns}
                if score is not None:
                    # never cache a failed/incomplete scoring: the key's
                    # files stop changing once a run ends, so a cached None
                    # would be served forever
                    self._cache[rid] = (key, rec)
            rec["live"] = (os.path.exists(ep)
                           and (_t.time() - os.path.getmtime(ep)) < 90
                           and rec["score"] is None)
            out.append(rec)
        return out

    def list_tasks(self):
        """Runs grouped by scenario/task: band, per-model aggregates (LLM
        probes with a score), and the raw run records (newest first)."""
        groups = {}
        for rec in self.list_runs():  # already newest-first
            groups.setdefault(rec["task"] or "unknown", []).append(rec)

        def mean(recs, field):
            vals = [r[field] for r in recs if r[field] is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        tasks = []
        for name, recs in groups.items():
            by_label = {}
            for r in recs:
                if r["probe"] == "llm" and r["score"] is not None:
                    by_label.setdefault(r["label"], []).append(r)
            models = {}
            for label, rs in by_label.items():
                models[label] = {
                    "n": len(rs),
                    "mean_score": mean(rs, "score"),
                    "mean_cost_usd": mean(rs, "cost_usd"),
                    "mean_efficiency": mean(rs, "efficiency"),
                    "mean_done_rate": mean(rs, "done_rate"),
                    "mean_fairness": mean(rs, "fairness"),
                    "mean_turns": mean(rs, "agent_turns"),
                    "mean_tokens_out": mean(rs, "tokens_out"),
                    "runs": [r["id"] for r in rs],
                }
            bands = [r["band"] for r in recs if r["band"] is not None]
            tasks.append({"task": name, "band": max(bands) if bands else None,
                          "models": models, "runs": recs})
        tasks.sort(key=lambda t: (t["band"] is None,
                                  t["band"] if t["band"] is not None else 0))
        return tasks

    def watcher(self, rid):
        rid = os.path.basename(rid or "")
        if rid not in self._watchers:
            path = os.path.join(self.base, rid)
            if not os.path.isdir(path):
                return None
            self._watchers[rid] = RunWatcher(path)
        return self._watchers[rid]


class RunWatcher:
    """Read-only observer over a run directory: tails events.jsonl and
    reconstructs derived state by replay — works on live runs owned by other
    processes (harness/bench/CLI) and on finished runs alike."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        with open(os.path.join(run_dir, "scenario.json")) as f:
            self.scenario = json.load(f)

    def events(self):
        out = []
        try:
            with open(os.path.join(self.run_dir, "events.jsonl")) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except FileNotFoundError:
            pass
        return out

    def clock(self, events):
        start = self.scenario.get("start_time", 545)
        return max([e["t"] for e in events] +
                   [e["to"] for e in events if e["kind"] == "clock_advanced"] +
                   [start])

    def state(self, since=0):
        events = self.events()
        clock = self.clock(events)
        # pending approximated from the log: scheduled entries not yet due
        pending = sorted(
            ({"fire_at": e["fire_at"], "event": e["event"],
              "payload": {"npc": e.get("npc", "")}}
             for e in events if e["kind"] == "scheduled" and e["fire_at"] > clock),
            key=lambda x: x["fire_at"])
        return {"clock": clock, "now": fmt(clock), "seq": len(events),
                "events": events[since:], "pending": pending}

    def world_at(self, at=None):
        from .replay import replay
        return replay(self.scenario, self.events())


def state_payload(since=0, src=None):
    if src is not None:
        return src.state(since)
    world = ENGINE.world
    return {
        "clock": world.clock,
        "now": fmt(world.clock),
        "seq": len(world.log),
        "events": world.log[since:],
        "pending": ENGINE.pending(),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep stdout clean

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _src(self, query):
        """Resolve the read source: hub run (?run=id) > single watcher > None
        (own engine)."""
        if HUB is not None:
            rid = parse_qs(query).get("run", [""])[0]
            if not rid:
                runs = HUB.list_runs()
                rid = runs[0]["id"] if runs else ""
            return HUB.watcher(rid)
        return WATCHER

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            with open(os.path.join(WEB_ROOT, "index.html"), "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/runs":
            if HUB is None:
                return self._json({"error": "not a hub"}, 404)
            self._json({"runs": HUB.list_runs()})
        elif (url.path == "/api/tasks" and HUB is not None
              and "run" not in parse_qs(url.query)):
            # hub aggregate view; ?run=<id> falls through to the per-run board
            self._json({"tasks": HUB.list_tasks()})
        elif url.path == "/api/problem":
            # the AUTHORED problem for a run, as graph data: every task the
            # week will throw at the PM (seed + scheduled arrivals), with
            # dependencies, default owners, priorities, and arrival times.
            # Interventions/outcomes are overlaid client-side from the event
            # log the page already has.
            src = self._src(url.query)
            scenario = (src.scenario if src is not None
                        else ENGINE.world.scenario if ENGINE is not None
                        else None)
            if scenario is None:
                return self._json({"tasks": []})
            start = scenario.get("start_time", 545)
            out = []
            for t in (scenario.get("project") or {}).get("tasks", []):
                out.append({"id": t["id"], "title": t["title"],
                            "priority": t.get("priority"),
                            "effort_hours": t.get("effort_hours"),
                            "done_hours": t.get("done_hours", 0),
                            "blocked_by": t.get("blocked_by", []),
                            "owner": (t.get("assignees") or [None])[0],
                            "urgent": bool(t.get("urgent")),
                            "arrival": start, "source": "seed"})
            for arr in scenario.get("task_arrivals", []):
                t = arr["task"]
                out.append({"id": t["id"], "title": t["title"],
                            "priority": t.get("priority"),
                            "effort_hours": t.get("effort_hours"),
                            "done_hours": 0,
                            "blocked_by": t.get("blocked_by", []),
                            "owner": (t.get("assignees") or [None])[0],
                            "urgent": bool(t.get("urgent")),
                            "arrival": arr["at"], "source": "external",
                            "via": arr.get("via", "chat")})
            from .optimal import worker_ids
            from .sim_time import working_minutes_between
            due = (scenario.get("project") or {}).get("due")
            workers = worker_ids(scenario)
            self._json({"start": start, "due": due, "tasks": out,
                        "workers": len(workers),
                        "capacity_hours": (round(len(workers) *
                            working_minutes_between(start, due) / 60.0, 1)
                            if due else None)})
        elif url.path == "/api/transcript":
            # the agent's whole conversation, OpenAI chat format, untruncated
            src = self._src(url.query)
            run_dir = (src.run_dir if src is not None
                       else ENGINE.run_dir if ENGINE is not None else None)
            self._json({"messages": load_transcript(run_dir)})
        elif url.path == "/api/meta":
            src = self._src(url.query)
            if src is not None:
                scenario, run_dir = src.scenario, src.run_dir + " (watch)"
                npcs = [{"id": n["id"], "name": n["name"], "role": n["role"]}
                        for n in scenario["npcs"]]
            else:
                scenario, run_dir = ENGINE.world.scenario, ENGINE.run_dir
                npcs = [{"id": n.id, "name": n.name, "role": n.role}
                        for n in ENGINE.world.npcs.values()]
            self._json({
                "company": scenario.get("company", ""),
                "agent_name": scenario.get("agent_name", "PM"),
                "start_time": scenario.get("start_time", 545),
                "run_dir": run_dir,
                "npcs": npcs,
            })
        elif url.path == "/api/events":
            since = int(parse_qs(url.query).get("since", ["0"])[0])
            self._json(state_payload(since, self._src(url.query)))
        elif url.path == "/api/tasks":
            q = parse_qs(url.query)
            at = int(q["at"][0]) if "at" in q else None
            src = self._src(url.query)
            world = src.world_at() if src is not None else ENGINE.world
            project = world.project or {}
            tasks = world.tasks_view(at)
            for t in tasks:
                t["projected_done_fmt"] = (
                    fmt(t["projected_done"]) if t["projected_done"] is not None else None)
            self._json({
                "project": {
                    "name": project.get("name", "(no project)"),
                    "priority": project.get("priority", ""),
                    "due": project.get("due"),
                    "due_fmt": fmt(project["due"]) if project.get("due") else "",
                },
                "tasks": tasks,
            })
        elif url.path == "/api/tools":
            self._json({"tools": schemas()})
        elif url.path == "/api/docs":
            src = self._src(url.query)
            world = src.world_at() if src is not None else ENGINE.world
            self._json({"docs": world.docs, "transcripts": world.transcripts})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if WATCHER is not None or HUB is not None:
            return self._json({"error": "watch/hub mode is read-only"}, 403)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        url = urlparse(self.path)

        if url.path == "/api/say":
            npc, text = body.get("npc"), (body.get("text") or "").strip()
            if npc not in ENGINE.world.npcs or not text:
                return self._json({"error": "need valid npc + text"}, 400)
            with LOCK:
                due = ENGINE.agent_say(npc, text)
            return self._json({"ok": True, "reply_due": due, "reply_due_fmt": fmt(due)})

        if url.path == "/api/step":
            with LOCK:
                event = ENGINE.step()
            return self._json({"ok": True, "stepped": event is not None})

        if url.path == "/api/run":
            with LOCK:
                event = ENGINE.run_until_reply()
            return self._json({"ok": True, "replied": event is not None})

        if url.path == "/api/advance":
            with LOCK:
                target = int(body["to"]) if "to" in body \
                    else ENGINE.world.clock + int(body.get("minutes", 60))
                fired = ENGINE.advance_until(target)
            return self._json({"ok": True, "fired": fired,
                               "now": fmt(ENGINE.world.clock)})

        if url.path == "/api/tool":
            name, args = body.get("name"), body.get("args") or {}
            with LOCK:
                result = call_tool(ENGINE, name, args)
            code = 400 if isinstance(result, dict) and "error" in result else 200
            return self._json({"result": result}, code)

        if url.path == "/api/task":
            title = (body.get("title") or "").strip()
            if not title:
                return self._json({"error": "need title"}, 400)
            spec = {"id": re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40],
                    "title": title}
            if body.get("assignee"):
                spec["assignees"] = [body["assignee"]]
            if body.get("effort_hours"):
                spec["effort_hours"] = float(body["effort_hours"])
            with LOCK:
                task = ENGINE.agent_add_task(spec)
            return self._json({"ok": True, "task": task})

        return self._json({"error": "not found"}, 404)


def main():
    global ENGINE, WATCHER, HUB
    load_env()
    args = sys.argv[1:]
    if args and args[0] == "--hub":
        port = int(args[1]) if len(args) > 1 else 8742
        HUB = RunHub()
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        print("sim hub (all runs, read-only): http://127.0.0.1:%d" % port)
        server.serve_forever()
        return
    if args and args[0] == "--watch":
        run_dir = args[1]
        port = int(args[2]) if len(args) > 2 else 8741
        WATCHER = RunWatcher(run_dir)
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        print("sim observer (WATCH, read-only): http://127.0.0.1:%d  -> %s"
              % (port, run_dir))
        server.serve_forever()
        return
    scenario_path = args[0] if args else "scenarios/demo.json"
    port = int(args[1]) if len(args) > 1 else 8741
    with open(scenario_path) as f:
        scenario = json.load(f)
    ENGINE = Engine(scenario, verbose=False)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("sim server: http://127.0.0.1:%d  (run dir: %s)" % (port, ENGINE.run_dir))
    server.serve_forever()


if __name__ == "__main__":
    main()
