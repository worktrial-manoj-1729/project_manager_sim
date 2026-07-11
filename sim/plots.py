"""Render the benchmark + difficulty plots as dependency-free SVG into plots/.

    python -m sim.plots            # reads runs/ + scenarios/, writes plots/*.svg

No matplotlib / numpy — the SVG is generated from strings so it version-controls
cleanly, renders on GitHub, and matches the deck's inline-SVG aesthetic. Every
number comes from real data: benchmark scores from each run's scorecard.json,
bands from each scenario's stamped `band` block. Re-run after new rollouts or a
re-stamp to refresh.
"""

import glob
import json
import os
import statistics as st

BG = "#0d0d0d"
INK = "#f2f1ec"
DIM = "#a5a39b"
BORDER = "rgba(255,255,255,.14)"
MODEL_COLOR = {"claude-haiku-4-5": "#eda100", "claude-sonnet-5": "#2fd39a",
               "claude-opus-4-8": "#9085e9"}
MODEL_SHORT = {"claude-haiku-4-5": "haiku", "claude-sonnet-5": "sonnet",
               "claude-opus-4-8": "opus"}
MODEL_ORDER = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8"]
BENCH_TASKS = ["crunch", "mobile", "bigweek"]
BAND_SCENARIOS = ["crunch", "mobile", "spread", "selfbuild", "bigweek", "megaweek"]

FONT = ('font-family="system-ui,-apple-system,Segoe UI,sans-serif"')


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" {FONT}>'
            f'<rect width="{w}" height="{h}" fill="{BG}"/>'
            f'<text x="28" y="34" fill="{INK}" font-size="19" font-weight="700">{_esc(title)}</text>'
            f'{body}</svg>')


def _txt(x, y, s, fill=INK, size=12, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{_esc(s)}</text>')


def _line(x1, y1, x2, y2, stroke=BORDER, w=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{d}/>')


# --------------------------------------------------------------------------- #
def load_bench():
    """{task: {model: [scores]}} from every scored run (current runs/ only)."""
    agg = {}
    for d in glob.glob("runs/run-*"):
        mp, sp = os.path.join(d, "meta.json"), os.path.join(d, "scorecard.json")
        if not (os.path.exists(mp) and os.path.exists(sp)):
            continue
        m = json.load(open(mp))
        task, model = m.get("task"), m.get("agent_model")
        if task not in BENCH_TASKS or model not in MODEL_COLOR:
            continue
        score = json.load(open(sp)).get("score")
        if score is None:
            continue
        agg.setdefault(task, {}).setdefault(model, []).append(score)
    return agg


def load_bands():
    rows = []
    for name in BAND_SCENARIOS:
        p = "scenarios/%s.json" % name
        if not os.path.exists(p):
            continue
        b = (json.load(open(p)) or {}).get("band") or {}
        base = (b.get("no_pm_baseline") or {}).get("combined")
        opt = (b.get("opt_max") or {}).get("combined")
        if base is None or opt is None:
            continue
        rows.append({"name": name, "base": base, "opt": opt,
                     "win": b.get("winnable_combined", opt - base),
                     "log10": b.get("log10_trajectory_classes"),
                     "util": b.get("capacity_utilization"),
                     "triage": b.get("forced_triage")})
    return rows


# --------------------------------------------------------------------------- #
def plot_benchmark(agg):
    """Per (scenario, model): the individual run scores as dots + a mean tick,
    on a shared 0..max axis with baseline(0) and OPT(1) reference lines."""
    L, R, T = 210, 60, 66
    rows = [(t, m) for t in BENCH_TASKS for m in MODEL_ORDER if agg.get(t, {}).get(m)]
    rh = 40
    W, H = 940, T + rh * len(rows) + 40
    all_s = [s for t in agg.values() for xs in t.values() for s in xs]
    xmax = max(1.05, (max(all_s) if all_s else 1) + 0.1)
    xmin = min(0.0, (min(all_s) if all_s else 0) - 0.05)
    x0, x1 = L, W - R

    def sx(v):
        return x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)

    body = [_txt(28, 52, "Benchmark — normalized score per model  (3 rollouts each · dot = run · │ = mean)",
                 DIM, 12.5)]
    # reference lines at 0 (baseline) and 1 (OPT)
    for v, lab in ((0.0, "baseline = 0"), (1.0, "OPT = 1")):
        if xmin <= v <= xmax:
            body.append(_line(sx(v), T - 8, sx(v), H - 30, DIM, 1, "3 3"))
            body.append(_txt(sx(v), H - 14, lab, DIM, 11, "middle"))
    last_task = None
    for i, (task, model) in enumerate(rows):
        y = T + rh * i + rh / 2
        if task != last_task:
            body.append(_txt(28, y - rh / 2 + 16, task.upper(), INK, 13, "start", "700"))
            last_task = task
        c = MODEL_COLOR[model]
        xs = agg[task][model]
        body.append(_txt(L - 12, y + 4, MODEL_SHORT[model], c, 12.5, "end", "600"))
        body.append(_line(x0, y, x1, y, BORDER, 1))
        for s in xs:
            body.append(f'<circle cx="{sx(s):.1f}" cy="{y:.1f}" r="4.5" fill="{c}" '
                        f'fill-opacity="0.85"/>')
        mean = st.mean(xs)
        body.append(_line(sx(mean), y - 11, sx(mean), y + 11, c, 2.5))
        rng = "%.2f–%.2f" % (min(xs), max(xs))
        body.append(_txt(x1 + 8, y + 4, "mean %.2f  (%s)" % (mean, rng), DIM, 11))
    return _svg(W, H, "".join(body), "PM-sim benchmark — model separation")


def plot_bands(rows):
    """Horizontal [baseline, OPT] band per scenario (raw COMBINED); the bar's
    WIDTH is the winnable band the PM can move within."""
    rows = sorted(rows, key=lambda r: r["win"])
    L, R, T = 150, 40, 74
    rh = 46
    W, H = 940, T + rh * len(rows) + 34
    x0, x1 = L, W - R
    vmax = max(r["opt"] for r in rows) * 1.02
    sx = lambda v: x0 + v / vmax * (x1 - x0)
    body = [_txt(28, 52, "Difficulty band per scenario — [no-PM baseline , frictionless OPT] on COMBINED; "
                         "bar width = winnable", DIM, 12.5)]
    for gx in (0, vmax / 4, vmax / 2, 3 * vmax / 4, vmax):
        body.append(_line(sx(gx), T - 6, sx(gx), H - 22, BORDER, 1))
        body.append(_txt(sx(gx), H - 8, "%.0f" % gx, DIM, 10.5, "middle"))
    for i, r in enumerate(rows):
        y = T + rh * i + rh / 2
        col = "#3987e5" if r["name"] != "megaweek" else "#e34948"
        body.append(_txt(L - 12, y + 4, r["name"], INK, 13, "end", "600"))
        body.append(f'<rect x="{sx(r["base"]):.1f}" y="{y-9:.1f}" '
                    f'width="{sx(r["opt"])-sx(r["base"]):.1f}" height="18" rx="4" '
                    f'fill="{col}" fill-opacity="0.28" stroke="{col}" stroke-width="1.5"/>')
        body.append(f'<circle cx="{sx(r["base"]):.1f}" cy="{y:.1f}" r="4" fill="{DIM}"/>')
        body.append(f'<circle cx="{sx(r["opt"]):.1f}" cy="{y:.1f}" r="4" fill="{col}"/>')
        extra = "win %.0f" % r["win"]
        if r.get("log10") is not None:
            extra += " · 10^%.0f plans" % r["log10"]
        if r.get("util") is not None:
            extra += " · load %.2f%s" % (r["util"], " · TRIAGE" if r.get("triage") else "")
        body.append(_txt(sx(r["opt"]) + 10, y + 4, extra, DIM, 10.5))
    return _svg(W, H, "".join(body), "PM-sim difficulty bands")


def main():
    os.makedirs("plots", exist_ok=True)
    agg = load_bench()
    bands = load_bands()
    out = []
    if agg:
        open("plots/benchmark_scores.svg", "w").write(plot_benchmark(agg))
        out.append("plots/benchmark_scores.svg")
    if bands:
        open("plots/scenario_bands.svg", "w").write(plot_bands(bands))
        out.append("plots/scenario_bands.svg")
    # a tiny README so the folder is self-describing
    with open("plots/README.md", "w") as f:
        f.write("# Plots\n\nGenerated by `python -m sim.plots` from real run data "
                "(`runs/*/scorecard.json`) and stamped scenario bands "
                "(`scenarios/*.json`). Dependency-free SVG — re-run to refresh.\n\n"
                + "".join("- `%s`\n" % os.path.basename(o) for o in out))
    print("wrote:", ", ".join(out) or "(no data found)")


if __name__ == "__main__":
    main()
