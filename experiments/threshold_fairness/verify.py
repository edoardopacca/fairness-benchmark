"""Replay evidence without retraining; verify selections, split pairing and audit statistics."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..upstream import ROOT, fairgroups_class
from .core import jobs, make_data, model, run_id, group_at_threshold, validation_metrics, predefined
from .run import identity, choose


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("out",type=Path,nargs="?",default=ROOT/"results/threshold_fairness")
    args=parser.parse_args(); out=args.out
    manifest=json.loads((out/"manifest.json").read_text()); config=manifest["identity"]["config"]
    assert identity(config)==manifest["identity"]
    seal=json.loads((out/"selection_seal.json").read_text())
    for name,key in [("validation_selection.csv","selection_sha256"),("validation_results.csv","validation_results_sha256")]:
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==seal[key]
    validation=pd.read_csv(out/"validation_results.csv")
    recomputed,_=choose(validation,config)
    recorded=pd.read_csv(out/"validation_selection.csv")
    pd.testing.assert_frame_equal(recomputed,recorded,check_dtype=False,atol=1e-12,rtol=1e-12)
    all_jobs=list(jobs(config)); assert len(validation)==len(all_jobs)
    results=pd.read_csv(out/"per_seed_results.csv"); assert len(results)==len(all_jobs)
    assert not list((out/"runs").glob("*/error.txt"))
    torch.set_num_threads(1)
    cache={}; replayed=0; partition_replays=set()
    for job in all_jobs:
        dest=out/"runs"/run_id(job)
        key=(job["scenario"],job["seed"])
        if key not in cache:
            cache[key]=make_data(job["seed"],job["scenario"],config["sizes"])
            ids=np.concatenate([p["ids"] for p in cache[key][0].values()])
            assert len(np.unique(ids))==sum(config["sizes"].values())
        parts,info=cache[key]
        state=torch.load(dest/"checkpoint.pt",map_location="cpu",weights_only=False)
        assert state["data"]==info
        net=model(config); net.load_state_dict(state["model"])
        with torch.no_grad():
            test=net(torch.from_numpy(parts["test"]["x"])).sigmoid().numpy().flatten()
            audit=net(torch.from_numpy(parts["audit"]["x"])).sigmoid().numpy().flatten()
            val=net(torch.from_numpy(parts["validation"]["x"])).sigmoid().numpy().flatten()
        with np.load(dest/"test_audit_predictions.npz") as saved:
            np.testing.assert_array_equal(test,saved["test_p"])
            np.testing.assert_array_equal(audit,saved["audit_p"])
            for split in ("test","audit"):
                for field in ("ids","s","y"):
                    np.testing.assert_array_equal(parts[split][field],saved[f"{split}_{field}"])
        with np.load(dest/"validation_predictions.npz") as saved:
            np.testing.assert_array_equal(val,saved["p"])
        metrics=validation_metrics(parts["test"],test,predefined(parts["train"]["s"],config),config)
        result=json.loads((dest/"evaluation.json").read_text())
        for name,value in metrics.items():
            np.testing.assert_allclose(value,result[f"test_{name}"],atol=1e-12)
        decisions=pd.read_csv(dest/"decision_metrics.csv")
        learned=json.loads((dest/"fairgroups_partitions.json").read_text())
        replay_key=(job["scenario"],job["method"],job["temperature"])
        for threshold in config["evaluation_thresholds"]:
            bounds={**predefined(parts["train"]["s"],config),"fairgroups":learned[str(threshold)]}
            if replay_key not in partition_replays:
                np.random.seed(job["seed"]+20000+round(threshold*100))
                with open(os.devnull,"w") as log, contextlib.redirect_stderr(log):
                    fg=fairgroups_class()(n_groups=config["groups"],grid_size=config["audit_grid"])
                    fg.fit(parts["audit"]["s"],(audit>=threshold).astype(int))
                np.testing.assert_array_equal(fg.partition[1:-1],bounds["fairgroups"][1:-1])
            for partition,cuts in bounds.items():
                got,_=group_at_threshold(parts["test"]["s"],test,cuts,threshold)
                row=decisions[(decisions.split=="test")&(decisions.partition==partition)&(decisions.threshold==threshold)].iloc[0]
                for name,value in got.items():
                    np.testing.assert_allclose(value,row[name],atol=1e-12)
        partition_replays.add(replay_key); replayed+=1
        if replayed%300==0: print(f"Verified {replayed}/{len(all_jobs)} checkpoints",flush=True)
    inference=pd.read_csv(out/"statistical_audit.csv")
    family=json.loads((out/"audit_family.json").read_text())
    assert len(inference)==family["family_size"]
    np.testing.assert_allclose(inference.p_bonferroni,np.minimum(inference.permutation_p*len(inference),1.),atol=1e-12)
    # Selection seal predates every test evaluation output.
    seal_time=(out/"selection_seal.json").stat().st_mtime
    assert all(seal_time <= (out/"runs"/run_id(j)/"evaluation.json").stat().st_mtime for j in all_jobs)
    proof={"all_checks_passed":True,"completed_runs":len(all_jobs),"paired_datasets":len(cache),
           "exact_checkpoints_replayed":replayed,"audit_partition_cases_replayed":len(partition_replays)*3,
           "validation_selections_reconstructed":len(recorded),"corrected_audit_tests":len(inference),
           "selection_sealed_before_test_scores":True,"upstream_unchanged":True,
           "verifier_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/"verification.json").write_text(json.dumps(proof,indent=2)+"\n")
    print(json.dumps(proof,indent=2))


if __name__=="__main__": main()
