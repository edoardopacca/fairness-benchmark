"""Train and seal validation selections. No audit fitting or test prediction in this module."""
import tempfile
import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fairness-benchmark-matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from torch import nn
from ..upstream import ROOT, provenance
from ..run import write_json
from .core import jobs, make_data, model, predefined, regularizer, run_id, validation_metrics

SOURCE_FILES = ["experiments/data.py", "experiments/metrics.py", "experiments/penalties.py",
                "experiments/upstream.py", "experiments/run.py", "experiments/posthoc/audit.py",
                "experiments/threshold_fairness/core.py", "experiments/threshold_fairness/run.py",
                "experiments/threshold_fairness/evaluate.py", "experiments/threshold_fairness/oracle.py"]


def hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def identity(config):
    return {"config": config, "source_files_sha256": hashes(), "upstream": provenance()}


def train_one(job, config, root):
    torch.set_num_threads(config["threads_per_worker"])
    torch.use_deterministic_algorithms(True)
    dest = Path(root)/"runs"/run_id(job)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/"training.json").exists():
        return json.loads((dest/"training.json").read_text())
    start = time.monotonic()
    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    parts, info = make_data(job["seed"], job["scenario"], config["sizes"])
    net = model(config)
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    gen = torch.Generator().manual_seed(job["seed"]+10000)
    x, y, s = [torch.from_numpy(parts["train"][key]) for key in ("x", "y", "s")]
    history, norms, clipped, reg_norms = [], [], 0, []
    for epoch in range(1, config["epochs"]+1):
        total = np.zeros(5)
        batches = list(torch.randperm(len(x), generator=gen).split(config["batch_size"]))
        if len(batches[-1]) == 1:
            batches[-2] = torch.cat([batches[-2], batches[-1]])
            batches.pop()
        last_reg_norm = 0.
        for index, ids in enumerate(batches):
            logits = net(x[ids]).flatten()
            p = logits.sigmoid()
            bce = nn.functional.binary_cross_entropy_with_logits(logits, y[ids])
            reg = regularizer(job["method"], p, s[ids], job["temperature"], config)
            weight = job["lambda"] if epoch > config["warmup_epochs"] else 0.
            loss = bce+weight*reg
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite objective at epoch {epoch}")
            if index == len(batches)-1 and job["method"] != "baseline":
                grads = torch.autograd.grad(reg, net.parameters(), retain_graph=True)
                last_reg_norm = float(torch.sqrt(sum(g.square().sum() for g in grads)))
                reg_norms.append(last_reg_norm)
            optimizer.zero_grad()
            loss.backward()
            norm = float(nn.utils.clip_grad_norm_(net.parameters(), config["grad_clip"], error_if_nonfinite=True))
            norms.append(norm)
            clipped += norm > config["grad_clip"]
            optimizer.step()
            derivative, saturated = 0., 0.
            if job["method"] == "threshold_gedi":
                soft = ((p.detach()-.5)/job["temperature"]).sigmoid()
                d = soft*(1-soft)/job["temperature"]
                derivative, saturated = float(d.mean()), float((d<1e-3).float().mean())
            total += np.array([float(bce.detach()), float(reg.detach()), float(loss.detach()), derivative, saturated])*len(ids)
        row = dict(zip(["bce", "penalty", "objective", "soft_derivative_mean", "soft_saturated_fraction"], total/len(x)))
        row.update(epoch=epoch, last_batch_regularizer_grad_norm=last_reg_norm)
        history.append(row)
    boundaries = predefined(parts["train"]["s"], config)
    with torch.no_grad():
        p = net(torch.from_numpy(parts["validation"]["x"])).sigmoid().numpy().flatten()
    metrics = validation_metrics(parts["validation"], p, boundaries, config)
    result = {**job, "run_id": run_id(job), "status": "ok", "data_sha256": info["sha256"],
              **{f"validation_{k}": v for k, v in metrics.items()},
              "updates": len(norms), "grad_clip_fraction": clipped/len(norms),
              "grad_norm_max": max(norms), "grad_norm_p95": float(np.quantile(norms, .95)),
              "regularizer_grad_norm_max": max(reg_norms, default=0.),
              "soft_saturated_fraction": float(np.mean([r["soft_saturated_fraction"] for r in history])),
              "seconds": time.monotonic()-start}
    # No trained test/audit scores are generated here. Oracle values are not training targets.
    torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(), "job": job,
                "config": config, "data": info}, dest/"checkpoint.pt")
    np.savez_compressed(dest/"validation_predictions.npz", p=p, s=parts["validation"]["s"],
                        y=parts["validation"]["y"], ids=parts["validation"]["ids"])
    pd.DataFrame(history).to_csv(dest/"history.csv", index=False)
    write_json(dest/"predefined_partitions.json", boundaries)
    write_json(dest/"training.json", result)
    return result


def choose(frame, config):
    keys = ["scenario", "method", "lambda", "temperature"]
    candidates = frame.groupby(keys).agg(accuracy=("validation_accuracy", "mean"),
                                         gap=("validation_decision_gap", "mean"),
                                         degenerate=("validation_degenerate", "max"),
                                         seeds=("seed", "nunique")).reset_index()
    candidates = candidates[candidates.seeds == len(config["seeds"])]
    selections = []
    for scenario in config["scenarios"]:
        base = candidates[(candidates.scenario == scenario)&(candidates.method == "baseline")].iloc[0]
        for method in ("baseline", "gedi", "threshold_gedi", "mary_chi2"):
            group = candidates[(candidates.scenario == scenario)&(candidates.method == method)]
            for policy in ("performance_matched", "fairness_target"):
                eligible = group[~group.degenerate]
                if method == "baseline":
                    eligible = group  # Always show the reference, even if target not met.
                elif policy == "performance_matched":
                    eligible = eligible[eligible.accuracy >= base.accuracy-config["accuracy_tolerance"]]
                else:
                    eligible = eligible[eligible.gap <= config["fairness_target"]]
                if not len(eligible):
                    selections.append({"scenario": scenario, "method": method, "policy": policy, "status": "unavailable"})
                    continue
                order = (["gap", "accuracy", "lambda", "temperature"] if policy == "performance_matched"
                         else ["accuracy", "gap", "lambda", "temperature"])
                ascending = ([True, False, True, False] if policy == "performance_matched" else [False, True, True, False])
                best = eligible.sort_values(order, ascending=ascending).iloc[0].to_dict()
                selections.append({**best, "policy": policy, "status": "selected"})
        # H3: each temperature gets its own performance-matched lambda, still validation only.
        for temp in config["temperatures"]:
            group = candidates[(candidates.scenario == scenario)&(candidates.method == "threshold_gedi")&
                               (candidates.temperature == temp)&(~candidates.degenerate)&
                               (candidates.accuracy >= base.accuracy-config["accuracy_tolerance"])]
            if len(group):
                best = group.sort_values(["gap", "accuracy", "lambda"], ascending=[True, False, True]).iloc[0].to_dict()
                selections.append({**best, "policy": "temperature_ablation", "status": "selected"})
    return pd.DataFrame(selections), candidates


def initialize(out, config):
    out.mkdir(parents=True, exist_ok=True)
    current = identity(config)
    if any(v["status"] for v in current["upstream"].values()):
        raise RuntimeError("Upstream repositories must stay clean")
    if (out/"manifest.json").exists():
        if json.loads((out/"manifest.json").read_text())["identity"] != current:
            raise ValueError("Config or training/evaluation sources changed: use a new output directory")
    else:
        write_json(out/"manifest.json", {"identity": current, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "command": sys.argv, "platform": platform.platform(), "python": sys.version,
                   "versions": {m.__name__: m.__version__ for m in (torch, np, scipy, pd, sklearn)},
                   "device": "cpu", "workers": config["workers"]})
        write_json(out/"config.json", config)
        for relative in SOURCE_FILES:
            target = out/"source"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, target)
        for name in ("PROTOCOL.md", "HYPOTHESIS.md"):
            shutil.copy2(Path(__file__).parent/name, out/name)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--out", type=Path, default=ROOT/"results/threshold_fairness")
    parser.add_argument("--stage", choices=["train", "select"], default="train")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    initialize(args.out, config)
    all_jobs = list(jobs(config))
    if args.stage == "train":
        pending = [j for j in all_jobs if not (args.out/"runs"/run_id(j)/"training.json").exists()]
        start, count = time.monotonic(), 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=config["workers"]) as pool:
            futures = {pool.submit(train_one, job, config, str(args.out)): job for job in pending}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception:
                    import traceback
                    dest = args.out/"runs"/run_id(job)
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest/"error.txt").write_text(traceback.format_exc())
                    raise
                count += 1
                if count % 30 == 0 or count == len(pending):
                    print(f"Trained {count}/{len(pending)} pending, {time.monotonic()-start:.1f}s; latest {result['run_id']}", flush=True)
    records = [json.loads((args.out/"runs"/run_id(job)/"training.json").read_text()) for job in all_jobs]
    frame = pd.DataFrame(records)
    frame.to_csv(args.out/"validation_results.csv", index=False)
    selected, candidates = choose(frame, config)
    selected.to_csv(args.out/"validation_selection.csv", index=False)
    candidates.to_csv(args.out/"validation_candidates.csv", index=False)
    seal = {"selection_sha256": hashlib.sha256((args.out/"validation_selection.csv").read_bytes()).hexdigest(),
            "validation_results_sha256": hashlib.sha256((args.out/"validation_results.csv").read_bytes()).hexdigest(),
            "created_utc": datetime.now(timezone.utc).isoformat(), "trained_runs": len(frame),
            "test_evaluated": False}
    if (args.out/"selection_seal.json").exists():
        old = json.loads((args.out/"selection_seal.json").read_text())
        assert seal["selection_sha256"] == old["selection_sha256"]
    else:
        write_json(args.out/"selection_seal.json", seal)
    print(f"Sealed selections after {len(frame)} runs; no test scores inspected.", flush=True)


if __name__ == "__main__":
    main()
