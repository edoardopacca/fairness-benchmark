"""Run a fixed, paired grid; persist every experiment before aggregating.

Usage: python -m experiments.run --config experiments/configs/smoke.json --out results/smoke
"""
import tempfile
import argparse
import contextlib
import hashlib
import itertools
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fairness-benchmark-matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from torch import nn

from .data import make_data
from .metrics import basic, dependence, group_statistics
from .penalties import NeuralHGR, penalty
from .upstream import ROOT, fairgroups_class, provenance


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def source_hash():
    h = hashlib.sha256()
    for path in sorted((ROOT / "experiments").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def jobs(config):
    for scenario, seed, batch, method in itertools.product(
            config["scenarios"], config["seeds"], config["batch_sizes"], config["methods"]):
        for lam in ([0.0] if method == "baseline" else config["lambdas"]):
            yield {"scenario": scenario, "seed": seed, "batch_size": batch,
                   "method": method, "lambda": lam}


def run_id(job):
    return f'{job["scenario"]}_s{job["seed"]}_b{job["batch_size"]}_{job["method"]}_l{job["lambda"]:g}'


def predict(model, part):
    with torch.no_grad():
        return model(torch.from_numpy(part["x"])).sigmoid().numpy().reshape(-1)


def train(job, config, dest):
    started = time.monotonic()
    seed = job["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    parts, data_info = make_data(seed, job["scenario"], config["sizes"])
    layers, dim = [], 4
    for width in config["hidden"]:
        layers.extend([nn.Linear(dim, width), nn.ReLU()])
        dim = width
    layers.append(nn.Linear(dim, 1))
    model = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # A dedicated generator guarantees identical minibatches despite adversary RNG consumption.
    generator = torch.Generator().manual_seed(seed + 10000)
    adv = NeuralHGR(config["adversary_lr"]) if job["method"] == "neural_hgr" else None
    x = torch.from_numpy(parts["train"]["x"])
    y = torch.from_numpy(parts["train"]["y"])
    s = torch.from_numpy(parts["train"]["s"])
    history, clipped, updates, max_grad = [], 0, 0, 0.0
    for epoch in range(1, config["epochs"] + 1):
        totals = np.zeros(3)
        order = torch.randperm(len(x), generator=generator)
        chunks = list(order.split(job["batch_size"]))
        if len(chunks[-1]) == 1:  # Avoid undefined sample variances; no discarded observation.
            chunks[-2] = torch.cat([chunks[-2], chunks[-1]])
            chunks.pop()
        for ids in chunks:
            p = model(x[ids]).reshape(-1).sigmoid()
            warm = epoch <= config["warmup_epochs"]
            if adv is not None:
                adv.maximize(p, s[ids], config["adversary_steps"])
                reg = adv(p, s[ids])
            else:
                reg = penalty(job["method"], p, s[ids], config["gedi_degree"])
            logits = model(x[ids]).reshape(-1)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, y[ids])
            loss = bce + (0.0 if warm else job["lambda"]) * reg
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss, epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            norm = float(grad_norm)
            clipped += norm > config["grad_clip"]
            updates += 1
            max_grad = max(max_grad, norm)
            optimizer.step()
            totals += np.array([float(bce.detach()), float(reg.detach()), float(loss.detach())]) * len(ids)
        row = {"epoch": epoch, **dict(zip(["train_bce", "train_penalty", "train_objective"], totals / len(x)))}
        if epoch == 1 or epoch % config["monitor_every"] == 0 or epoch == config["epochs"]:
            for name in ("train", "validation"):
                pp = predict(model, parts[name])
                row.update({f"{name}_{k}": v for k, v in dependence(parts[name]["s"], pp, max_n=config["eval_max_n"]).items()})
                row[f"{name}_accuracy"] = float(((pp >= .5) == parts[name]["y"]).mean())
        history.append(row)
    predictions = {name: predict(model, part) for name, part in parts.items()}
    result = {**job, "run_id": run_id(job), "status": "ok", "data": data_info,
              "grad_clip_fraction": clipped / updates, "max_grad_norm": max_grad, "updates": updates}
    # Validation metrics are the only inputs to lambda selection.
    for name in ("train", "validation", "test"):
        values = basic(parts[name]["s"], parts[name]["y"], predictions[name],
                       max_n=config["eval_max_n"],
                       permutations=config["permutations"] if name == "test" else 0)
        result.update({f"{name}_{k}": v for k, v in values.items()})

    audit = parts["audit"]
    np.random.seed(seed + 20000)
    with (dest / "audit.log").open("w") as log, contextlib.redirect_stderr(log):
        fg = fairgroups_class()(n_groups=config["audit_groups"], grid_size=config["audit_grid"])
        fg.fit(audit["s"], (predictions["audit"] >= .5).astype(int))
    # Extend finite endpoint bounds to the known synthetic support. Internal cuts unchanged.
    adaptive = [-1.0, *[float(v) for v in fg.partition[1:-1]], 1.0]
    quantile = np.quantile(parts["train"]["s"], np.linspace(0, 1, config["audit_groups"] + 1)).tolist()
    quantile[0], quantile[-1] = -1.0, 1.0
    partitions = {"uniform": np.linspace(-1, 1, config["audit_groups"] + 1).tolist(),
                  "quantile": quantile, "fairgroups": adaptive}
    group_rows = []
    for name, boundaries in partitions.items():
        for split in ("audit", "test"):
            stats, rows = group_statistics(parts[split]["s"], predictions[split], boundaries)
            result.update({f"{split}_{name}_{key}": value for key, value in stats.items()})
            group_rows.extend({"partition": name, "split": split, **row} for row in rows)
    # Full conditional profiles, independent of outcome-driven grouping.
    _, profile = group_statistics(parts["test"]["s"], predictions["test"], np.linspace(-1, 1, 21).tolist())
    pd.DataFrame(profile).to_csv(dest / "conditional.csv", index=False)
    pd.DataFrame(group_rows).to_csv(dest / "groups.csv", index=False)
    pd.DataFrame(history).to_csv(dest / "history.csv", index=False)
    np.savez_compressed(dest / "predictions.npz", **{
        f"{name}_{key}": value for name, part in parts.items()
        for key, value in {"s": part["s"], "y": part["y"], "ids": part["ids"], "p": predictions[name]}.items()})
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "job": job,
                  "config": config, "data": data_info}
    if adv is not None:
        checkpoint.update(adversary=adv.state_dict(), adversary_optimizer=adv.optimizer.state_dict())
    torch.save(checkpoint, dest / "checkpoint.pt")
    write_json(dest / "partitions.json", partitions)
    result["seconds"] = time.monotonic() - started
    write_json(dest / "result.json", result)
    return result


def aggregate(out, config):
    records = []
    for job in jobs(config):
        path = out / "runs" / run_id(job) / "result.json"
        if path.exists():
            r = json.loads(path.read_text())
            records.append({k: v for k, v in r.items() if k != "data"})
    frame = pd.DataFrame(records)
    frame.to_csv(out / "results.csv", index=False)
    ok = frame[frame.status == "ok"].copy()
    keys = ["scenario", "method", "batch_size", "lambda"]
    columns = [c for c in ok if c.startswith(("test_", "validation_", "train_")) or c == "grad_clip_fraction"]
    summary = ok.groupby(keys)[columns].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / "summary.csv", index=False)
    candidates = ok.groupby(keys).agg(accuracy=("validation_accuracy", "mean"),
                                      dcor=("validation_dcor", "mean"), seeds=("seed", "nunique")).reset_index()
    # Incomplete methods cannot win via survivor bias.
    candidates = candidates[candidates.seeds == len(config["seeds"])]
    selection = []
    for _, group in candidates.groupby(["scenario", "method", "batch_size"]):
        eligible = group[group.dcor <= config["selection_dcor_limit"]]
        ranked = (eligible.sort_values(["accuracy", "dcor", "lambda"], ascending=[False, True, True])
                  if len(eligible) else group.sort_values(["dcor", "accuracy", "lambda"], ascending=[True, False, True]))
        best = ranked.iloc[0].to_dict()
        best["constraint_met"] = bool(len(eligible))
        selection.append(best)
    selected = pd.DataFrame(selection)
    selected.to_csv(out / "validation_selection.csv", index=False)
    if len(selected):
        ok.merge(selected[keys], on=keys).to_csv(out / "selected_test_results.csv", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--job-index", type=int, help="Run one zero-based grid item, e.g. in Slurm arrays")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if any(v < 2 for v in config["sizes"].values()) or min(config["batch_sizes"]) < 2:
        raise ValueError("Every split and minibatch needs at least two observations")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    prov = provenance()
    if any(r["status"] for r in prov.values()):
        raise RuntimeError("Upstream trees must be clean; preserve upstream repositories unchanged")
    identity = {"config": config, "source_sha256": source_hash(), "upstream": prov}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Config, source or upstream changed: choose a NEW results directory")
        if not (args.resume or args.aggregate_only):
            raise ValueError("Directory already initialized: use --resume or choose a new directory")
    else:
        manifest = {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "command": sys.argv, "python": sys.version, "platform": platform.platform(),
                    "versions": {m.__name__: m.__version__ for m in (torch, np, scipy, sklearn, pd)},
                    "device": "cpu", "torch_threads": torch.get_num_threads()}
        write_json(manifest_path, manifest)
    all_jobs = list(jobs(config))
    if not args.aggregate_only:
        chosen = enumerate(all_jobs) if args.job_index is None else [(args.job_index, all_jobs[args.job_index])]
        for index, job in chosen:
            dest = out / "runs" / run_id(job)
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "result.json").exists() and args.resume:
                continue
            try:
                result = train(job, config, dest)
                print(f'[{index+1}/{len(all_jobs)}] {run_id(job)}: acc={result["test_accuracy"]:.3f}, dcor={result["test_dcor"]:.3f}, {result["seconds"]:.1f}s', flush=True)
            except Exception as exc:
                import traceback
                (dest / "error.txt").write_text(traceback.format_exc())
                write_json(dest / "result.json", {**job, "run_id": run_id(job), "status": "failed", "error": str(exc)})
                print(f"FAILED {run_id(job)}: {exc}", flush=True)
        if args.job_index is not None:
            return  # Aggregate only after all array jobs finish; no shared CSV write races.
    frame = aggregate(out, config)
    print(f"Saved {len(frame)} / {len(all_jobs)} records to {out}", flush=True)
    if len(frame) != len(all_jobs) or (frame.status != "ok").any():
        raise SystemExit("Grid incomplete or contains failures. Inspect results.csv and error.txt.")


if __name__ == "__main__":
    main()
