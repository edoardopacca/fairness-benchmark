import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from ..upstream import gedi
from .core import group_at_threshold, jobs, make_data, regularizer
from .run import choose, train_one

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())


class ThresholdExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_grid_and_new_seeds(self):
        self.assertEqual(len(list(jobs(CONFIG))), 2190)
        self.assertEqual(len(CONFIG["seeds"]), 10)
        self.assertTrue(set(CONFIG["seeds"]).isdisjoint(range(5)))

    def test_oracle_branches_and_means(self):
        s = np.linspace(-1,1,10001)
        q = .6+.2*s
        p0, p1 = .5-.4*q, .9-.4*q
        np.testing.assert_allclose((1-q)*p0+q*p1, .5, atol=2e-16)
        self.assertTrue((p0<.5).all() and (p1>.5).all())

    def test_exact_gedi_and_soft_gradient(self):
        # Avoid exactly zero L1 coefficients: abs is nondifferentiable there.
        generator = torch.Generator().manual_seed(28)
        s = 2*torch.rand(40, generator=generator, dtype=torch.double)-1
        p = (.25+.5*torch.rand(40, generator=generator, dtype=torch.double)).requires_grad_()
        for t in CONFIG["temperatures"]:
            got = regularizer("threshold_gedi", p, s, t, CONFIG)
            expected = gedi(s, ((p-.5)/t).sigmoid(), 3, use_torch=True)
            self.assertTrue(torch.equal(got, expected))
            self.assertTrue(torch.autograd.gradcheck(lambda v: regularizer("threshold_gedi",v,s,t,CONFIG),(p,)))
        self.assertTrue(torch.equal(regularizer("gedi",p,s,0,CONFIG), gedi(s,p,3,use_torch=True)))
        with self.assertRaises(ValueError):
            regularizer("threshold_gedi",p,s,0,CONFIG)

    def test_splits_and_no_oracle_input(self):
        sizes = {"train":128, "validation":64, "audit":64, "test":128}
        parts, info = make_data(100,"controlled",sizes)
        self.assertEqual(parts["train"]["x"].shape[1],4)
        self.assertEqual(len(np.unique(np.concatenate([p["ids"] for p in parts.values()]))),384)
        _, repeated = make_data(100,"controlled",sizes)
        self.assertEqual(info,repeated)

    def test_thresholds_and_variance(self):
        s = np.array([-.9,-.8,.8,.9])
        p = np.array([.4,.5,.6,.7])
        for threshold in (.4,.5,.6):
            stats, rows = group_at_threshold(s,p,[-1,0,1],threshold)
            expected = [np.mean(p[:2]>=threshold),np.mean(p[2:]>=threshold)]
            np.testing.assert_allclose([r["positive_rate"] for r in rows],expected)
            self.assertAlmostEqual(stats["weighted_variance"], np.var(expected))
            self.assertAlmostEqual(stats["score_gap"], .2)

    def test_selection_has_no_test_or_audit_dependency(self):
        rows=[]
        for method, lam, acc, gap in [("baseline",0,.8,.4),("gedi",.1,.79,.2),("gedi",1,.7,.05),
                                     ("threshold_gedi",.1,.79,.15),("mary_chi2",.1,.78,.18)]:
            for seed in CONFIG["seeds"]:
                rows.append(dict(scenario="controlled",method=method,temperature=.05 if method=="threshold_gedi" else 0.,
                                 **{"lambda":lam},seed=seed,validation_accuracy=acc,validation_decision_gap=gap,
                                 validation_degenerate=False,test_accuracy=.1,audit_gap=.9))
        c={**CONFIG,"scenarios":["controlled"]}
        a,_=choose(pd.DataFrame(rows),c)
        altered=pd.DataFrame(rows); altered.test_accuracy=1.; altered.audit_gap=0.
        b,_=choose(altered,c)
        pd.testing.assert_frame_equal(a,b)
        selected=a[(a.method=="gedi")&(a.policy=="performance_matched")].iloc[0]
        self.assertEqual(selected["lambda"],.1)

    def test_training_does_not_use_audit_test_or_oracle_targets(self):
        config={**CONFIG,"sizes":{"train":128,"validation":64,"audit":64,"test":128},"epochs":2,"workers":1,"eval_max_n":64}
        job={"scenario":"controlled","seed":100,"method":"threshold_gedi","lambda":.1,"temperature":.05}
        original=make_data
        def modified(*args):
            parts,info=original(*args)
            for name in ("test","audit"):
                parts[name]["x"]*=100
                parts[name]["y"]=1-parts[name]["y"]
            parts["train"]["oracle"][:]=0
            return parts,info
        with tempfile.TemporaryDirectory() as tmp:
            train_one(job,config,Path(tmp)/"a")
            with patch("experiments.threshold_fairness.run.make_data",modified):
                train_one(job,config,Path(tmp)/"b")
            a=torch.load(next((Path(tmp)/"a").glob("runs/*/checkpoint.pt")),weights_only=False)["model"]
            b=torch.load(next((Path(tmp)/"b").glob("runs/*/checkpoint.pt")),weights_only=False)["model"]
            for key in a:
                self.assertTrue(torch.equal(a[key],b[key]))


if __name__ == "__main__":
    unittest.main()
