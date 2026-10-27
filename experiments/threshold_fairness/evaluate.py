"""Evaluate a sealed grid; FairGroups is used exclusively here, after training/selection."""
import tempfile
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fairness-benchmark-matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import torch
from ..metrics import basic
from ..posthoc.audit import permutation_p
from ..run import write_json
from ..upstream import ROOT, fairgroups_class
from .core import group_at_threshold, jobs, make_data, model, predefined, run_id, validation_metrics
from .run import identity


def selected_jobs(selected, config, main_only=False):
    frame = selected[selected.status == "selected"]
    if main_only:
        frame = frame[frame.policy != "temperature_ablation"]
    keys = ["scenario", "method", "lambda", "temperature"]
    configurations = {tuple(r) for r in frame[keys].itertuples(index=False, name=None)}
    return {run_id(j) for j in jobs(config) if tuple(j[k] for k in keys) in configurations}


def evaluate_one(job, config, root, inferential):
    torch.set_num_threads(config["threads_per_worker"])
    dest = Path(root)/"runs"/run_id(job)
    if (dest/"evaluation.json").exists():
        return json.loads((dest/"evaluation.json").read_text())
    checkpoint = torch.load(dest/"checkpoint.pt", weights_only=False, map_location="cpu")
    parts, info = make_data(job["seed"], job["scenario"], config["sizes"])
    assert info == checkpoint["data"]
    net = model(config)
    net.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        predictions = {name: net(torch.from_numpy(parts[name]["x"])).sigmoid().numpy().flatten()
                       for name in ("test", "audit")}
    boundaries = predefined(parts["train"]["s"], config)
    score_values = validation_metrics(parts["test"], predictions["test"], boundaries, config)
    result = {**job, "run_id": run_id(job), **{f"test_{k}": v for k, v in score_values.items()}}
    records, group_rows, profiles, learned = [], [], [], {}
    for threshold in config["evaluation_thresholds"]:
        np.random.seed(job["seed"]+20000+round(threshold*100))
        with (dest/f"audit_{threshold:g}.log").open("w") as log, contextlib.redirect_stderr(log):
            fg = fairgroups_class()(n_groups=config["groups"], grid_size=config["audit_grid"])
            fg.fit(parts["audit"]["s"], (predictions["audit"] >= threshold).astype(int))
        cuts = [-1., *[float(v) for v in fg.partition[1:-1]], 1.]
        learned[str(threshold)] = cuts
        for partition, bounds in {**boundaries, "fairgroups": cuts}.items():
            for split in ("audit", "test"):
                p, s = predictions[split], parts[split]["s"]
                stats, rows = group_at_threshold(s, p, bounds, threshold)
                record = {**job, "run_id": run_id(job), "threshold": threshold, "partition": partition,
                          "split": split, **stats, "accuracy": float(((p>=threshold)==parts[split]["y"]).mean()),
                          "prevalence": float((p>=threshold).mean()),
                          "constant_decision": bool((p>=threshold).all() or not (p>=threshold).any()),
                          "near_degenerate": bool((p>=threshold).mean()<config["near_degenerate_prevalence"] or
                                                  (p>=threshold).mean()>1-config["near_degenerate_prevalence"])}
                if split == "test" and inferential:
                    groups = np.searchsorted(bounds[1:-1], s, side="right")
                    counts = np.bincount(groups, minlength=config["groups"])
                    positives = np.bincount(groups, weights=(p>=threshold), minlength=config["groups"]).astype(int)
                    record["permutation_p"] = permutation_p(counts, positives,
                        permutations=config["audit_permutations"], seed=812)
                records.append(record)
                group_rows.extend({"threshold": threshold, "partition": partition, "split": split, **r} for r in rows)
        _, rows = group_at_threshold(parts["test"]["s"], predictions["test"], np.linspace(-1,1,21), threshold)
        profiles.extend({"threshold": threshold, **r} for r in rows)
    pd.DataFrame(records).to_csv(dest/"decision_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(dest/"groups.csv", index=False)
    pd.DataFrame(profiles).to_csv(dest/"conditional.csv", index=False)
    np.savez_compressed(dest/"test_audit_predictions.npz", **{
        f"{split}_{key}": value for split in ("test", "audit")
        for key, value in {"p": predictions[split], "s": parts[split]["s"], "y": parts[split]["y"], "ids": parts[split]["ids"]}.items()})
    write_json(dest/"fairgroups_partitions.json", learned)
    write_json(dest/"evaluation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out", type=Path, nargs="?", default=ROOT/"results/threshold_fairness")
    args = parser.parse_args()
    out = args.out
    manifest = json.loads((out/"manifest.json").read_text())
    config = manifest["identity"]["config"]
    assert manifest["identity"] == identity(config), "Code/upstream/config changed"
    seal = json.loads((out/"selection_seal.json").read_text())
    assert hashlib.sha256((out/"validation_selection.csv").read_bytes()).hexdigest() == seal["selection_sha256"]
    assert hashlib.sha256((out/"validation_results.csv").read_bytes()).hexdigest() == seal["validation_results_sha256"]
    selected = pd.read_csv(out/"validation_selection.csv")
    inferential = selected_jobs(selected, config, main_only=True)
    all_jobs = list(jobs(config))
    family_size = len(inferential)*len(config["evaluation_thresholds"])*3
    write_json(out/"audit_family.json", {"run_ids": sorted(inferential), "family_size": family_size,
               "permutations": config["audit_permutations"], "correction": "Bonferroni over all selected runs, thresholds, partitions",
               "selection_sha256": seal["selection_sha256"]})
    pending = [j for j in all_jobs if not (out/"runs"/run_id(j)/"evaluation.json").exists()]
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        futures = [pool.submit(evaluate_one, j, config, str(out), run_id(j) in inferential) for j in pending]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if n%60 == 0 or n == len(pending):
                print(f"Evaluated {n}/{len(pending)}, {time.monotonic()-start:.1f}s", flush=True)
    frame = pd.DataFrame([json.loads((out/"runs"/run_id(j)/"evaluation.json").read_text()) for j in all_jobs])
    train = pd.read_csv(out/"validation_results.csv")
    frame.merge(train, on=["scenario","seed","method","lambda","temperature","run_id"], validate="one_to_one").to_csv(out/"per_seed_results.csv", index=False)
    decisions = pd.concat([pd.read_csv(out/"runs"/run_id(j)/"decision_metrics.csv") for j in all_jobs], ignore_index=True)
    decisions.to_csv(out/"decision_results.csv", index=False)
    inference = decisions[decisions.split.eq("test") & decisions.run_id.isin(inferential)].copy()
    assert len(inference) == family_size and inference.permutation_p.notna().all()
    inference["p_bonferroni"] = np.minimum(inference.permutation_p*family_size, 1.)
    inference.to_csv(out/"statistical_audit.csv", index=False)
    print(f"Saved {len(frame)} learned-model runs, {family_size} inferential audit tests")


if __name__ == "__main__":
    main()
