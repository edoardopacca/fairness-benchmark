"""Scientific invariants and adapters, runnable with unittest (no extra test dependency)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .data import SCENARIOS, effect, make_data
from .metrics import dependence, group_statistics, pearson
from .penalties import penalty
from .run import jobs, train
from .upstream import ROOT, gedi, kde, mary, provenance


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_reproducible_disjoint_splits(self):
        sizes = {"train": 100, "validation": 30, "audit": 30, "test": 40}
        a, info_a = make_data(4, "quadratic", sizes)
        b, info_b = make_data(4, "quadratic", sizes)
        self.assertEqual(info_a, info_b)
        self.assertEqual(len(set(np.concatenate([p["ids"] for p in a.values()]))), 200)
        np.testing.assert_allclose(a["train"]["x"].mean(0), 0, atol=1e-6)
        for name in sizes:
            for key in a[name]:
                np.testing.assert_array_equal(a[name][key], b[name][key])
        _, other = make_data(5, "quadratic", sizes)
        self.assertNotEqual(info_a["sha256"], other["sha256"])

    def test_scenario_normalization(self):
        s = np.linspace(-1, 1, 100001)
        for name in SCENARIOS:
            v = effect(s, name)
            self.assertLess(abs(v.mean()), 1e-4)
            self.assertAlmostEqual(v.std(), 1, places=4)

    def test_nonlinear_audit_detects_pearson_blind_spot(self):
        s = np.linspace(-1, 1, 500)
        p = s**2
        self.assertLess(abs(pearson(s, p)), 1e-10)
        self.assertGreater(dependence(s, p)["dcor"], .45)
        self.assertGreater(dependence(s, p)["hsic_normalized"], .2)
        perm = np.random.default_rng(4).permutation(p)
        self.assertLess(dependence(s, perm)["dcor"], .2)

    def test_gedi_matches_independent_least_squares_and_gradient(self):
        torch.manual_seed(2)
        s = torch.linspace(-1, 1, 48, dtype=torch.double)
        p = (.2 + .3 * s + .7 * s**2 - .1 * s**3).requires_grad_()
        value = gedi(s, p, degree=3, use_torch=True)
        self.assertAlmostEqual(float(value), 1.1, places=7)
        self.assertTrue(torch.autograd.gradcheck(lambda t: gedi(s, t, 3, use_torch=True), (p,)))

    def test_mary_exact_upstream_and_gradient(self):
        torch.manual_seed(7)
        s = torch.randn(20, dtype=torch.double)
        p = torch.randn(20, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.equal(penalty("mary_chi2", p, s), mary.chi_2(p, s, kde)))
        self.assertTrue(torch.autograd.gradcheck(lambda t: penalty("mary_chi2", t, s), (p,), atol=1e-4))
        chi = mary.chi_2(p, s, kde)
        hgr = mary.hgr(p, s, kde, damping=0)
        self.assertGreaterEqual(float(chi) + 1e-8, float(hgr**2))

    def test_constant_predictions_and_boundary_coverage(self):
        s = torch.linspace(-1, 1, 32)
        for method in ("baseline", "pearson", "mary_chi2", "gedi"):
            p = torch.ones(32, requires_grad=True)
            value = penalty(method, p, s)
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(p.grad).all())
        s = np.array([-1., -.5, 0., .5, 1.])
        stats, rows = group_statistics(s, np.array([0., 0., .8, 1., 1.]), [-1, 0, 1])
        self.assertEqual(sum(r["n"] for r in rows), 5)
        self.assertEqual([r["n"] for r in rows], [2, 3])
        self.assertEqual(stats["dp_gap"], 1)

    def test_grid_count(self):
        config = json.loads((ROOT / "experiments/configs/pilot.json").read_text())
        self.assertEqual(len(list(jobs(config))), 520)

    def test_zero_lambda_pairing_and_no_test_leakage(self):
        config = json.loads((ROOT / "experiments/configs/smoke.json").read_text())
        config.update(epochs=2, permutations=0)
        base = {"scenario": "quadratic", "seed": 8, "batch_size": 64, "lambda": 0., "method": "baseline"}
        with tempfile.TemporaryDirectory() as tmp:
            reference = None
            for i, method in enumerate(config["methods"]):
                dest = Path(tmp) / str(i)
                dest.mkdir()
                train({**base, "method": method}, config, dest)
                values = np.load(dest / "predictions.npz")["test_p"]
                if reference is None:
                    reference = values
                else:
                    np.testing.assert_array_equal(reference, values)
            original = make_data
            def changed(*args):
                parts, info = original(*args)
                parts["test"]["x"] *= 100
                parts["test"]["y"] = 1 - parts["test"]["y"]
                return parts, info
            dest = Path(tmp) / "perturbed"
            dest.mkdir()
            with patch("experiments.run.make_data", changed):
                train(base, config, dest)
            old = torch.load(Path(tmp) / "0/checkpoint.pt", weights_only=False)["model"]
            new = torch.load(dest / "checkpoint.pt", weights_only=False)["model"]
            for key in old:
                self.assertTrue(torch.equal(old[key], new[key]))

    def test_upstream_unchanged(self):
        self.assertTrue(all(not v["status"] for v in provenance().values()))


if __name__ == "__main__":
    unittest.main()
