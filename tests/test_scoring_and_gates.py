"""Scoring anchors and authoring gates: the band, the ceiling, the fairness
gate, the fingerprint, and the anti-minting guard."""

import json
import os
import tempfile
import unittest

from sim.difficulty import fingerprint, stamp
from sim.eval import evaluate
from sim.tools import call_tool
from sim.validate import validate_scenario

from tests.fixtures import HORIZON, goto, make_engine, scenario


class TestBandAnchors(unittest.TestCase):
    def test_lazy_pm_scores_exactly_zero(self):
        """The baseline IS the do-nothing world — including the org fallback
        picking up both arrivals. A PM who adds nothing scores 0, not credit
        for work that happens anyway."""
        eng = make_engine()
        goto(eng, HORIZON)
        res = evaluate(eng.run_dir)
        self.assertAlmostEqual(res["score"], 0.0, places=3)
        self.assertIsInstance(res["reward"], float)
        # unfiled-or-not, every authored task stays in the denominators
        ids = {t["id"] for t in res["per_task_agent"]}
        self.assertLessEqual({"core", "side", "incident", "form"}, ids)

    def test_active_pm_lands_in_the_band(self):
        eng = make_engine()
        goto(eng, 2050)
        call_tool(eng, "add_task", {"title": "Incident", "id": "incident"})
        call_tool(eng, "assign_task", {"task_id": "incident", "npc": "bo"})
        goto(eng, 3580)  # form email delivered at 3570
        call_tool(eng, "add_task", {"title": "Compliance form", "id": "form"})
        call_tool(eng, "assign_task", {"task_id": "form", "npc": "ana"})
        # a made-up task must never mint value (weight 0), only cost capacity
        call_tool(eng, "add_task", {"title": "Fake thing", "effort_hours": 3})
        goto(eng, HORIZON)
        res = evaluate(eng.run_dir)
        self.assertGreaterEqual(res["score"], 0.0)   # preempting can't hurt
        self.assertLessEqual(res["score"], 1.0)      # OPT is a true ceiling
        ids = {t["id"] for t in res["per_task_agent"]}
        self.assertNotIn("fake-thing", ids)
        # odds is the same measurement reshaped: odds = N/(1-N). `score` is
        # rounded to 3dp while odds is computed unrounded, and the transform
        # amplifies rounding by 1/(1-N)^2 near the ceiling — tolerate that.
        if res["score_odds"] is not None and res["score"] < 1:
            slack = 0.001 / (1 - res["score"]) ** 2 + 0.01
            self.assertAlmostEqual(res["score_odds"],
                                   res["score"] / (1 - res["score"]),
                                   delta=slack)


class TestGates(unittest.TestCase):
    def test_valid_fixture_passes(self):
        self.assertEqual(validate_scenario(scenario()), [])

    def test_missing_fallback_rejected(self):
        s = scenario()
        del s["task_arrivals"][0]["fallback"]
        self.assertTrue(any("fallback" in e for e in validate_scenario(s)))

    def test_stakeholder_fallback_rejected(self):
        s = scenario()
        s["task_arrivals"][0]["fallback"]["npc"] = "vp"
        self.assertTrue(any("stakeholder" in e for e in validate_scenario(s)))

    def test_silent_arrival_rejected(self):
        s = scenario()
        del s["task_arrivals"][0]["announce"]
        self.assertTrue(any("announce" in e for e in validate_scenario(s)))

    def test_unfair_arrival_rejected(self):
        """Fairness is a rubric too: an ask that cannot be completed after
        the PM could first know it scores luck, not skill."""
        s = scenario()
        s["task_arrivals"].append({
            "at": 6600, "npc": "bo", "announce": "Impossible Friday ask.",
            "task": {"id": "impossible", "title": "x", "effort_hours": 8,
                     "priority": "P0"},
            "fallback": {"npc": "bo", "at": 6700}})
        self.assertTrue(any("UNFAIR" in e for e in validate_scenario(s)))

    def test_bad_via_rejected(self):
        s = scenario()
        s["task_arrivals"][0]["via"] = "fax"
        self.assertTrue(any("via" in e for e in validate_scenario(s)))


class TestFingerprintAndStamp(unittest.TestCase):
    def test_fingerprint_vector(self):
        fp = fingerprint(scenario())
        for key in ("winnable_combined", "capacity_utilization",
                    "opt_done_weight_rate", "min_reaction_ratio", "fair",
                    "log10_trajectory_classes"):
            self.assertIn(key, fp)
        self.assertTrue(fp["fair"])
        self.assertGreaterEqual(fp["winnable_combined"], 0.0)  # OPT >= baseline

    def test_stamp_embeds_band_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scn.json")
            with open(path, "w") as f:
                json.dump(scenario(), f)
            stamp(path)
            with open(path) as f:
                band = json.load(f)["band"]
            self.assertIn("no_pm_baseline", band)
            self.assertIn("opt_max", band)
            self.assertGreaterEqual(band["opt_max"]["combined"],
                                    band["no_pm_baseline"]["combined"])


if __name__ == "__main__":
    unittest.main()
