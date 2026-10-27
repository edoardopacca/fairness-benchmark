"""Extension adapters; original benchmark modules and repositories stay unchanged."""
import hashlib
import itertools
import numpy as np
import torch
from torch import nn
from ..data import make_data as existing_data
from ..metrics import basic, group_statistics
from ..penalties import penalty as existing_penalty
from ..upstream import gedi


def make_data(seed, scenario, sizes):
    if scenario != "controlled":
        return existing_data(seed, scenario, sizes)
    rng = np.random.default_rng(seed)
    n = sum(sizes.values())
    s = rng.uniform(-1, 1, n)
    q = .6+.2*s
    b = (rng.uniform(size=n) < q).astype(float)
    z = rng.normal(size=(n, 2))
    oracle = .5+.4*(b-q)
    y = (rng.uniform(size=n) < oracle).astype("float32")
    x = np.column_stack([s, b, z]).astype("float32")
    order, offset, parts = rng.permutation(n), 0, {}
    for name, size in sizes.items():
        ids = order[offset:offset+size]
        parts[name] = {"x": x[ids], "s": s[ids].astype("float32"), "y": y[ids], "ids": ids,
                       "oracle": oracle[ids].astype("float32")}
        offset += size
    mean, std = parts["train"]["x"].mean(0), parts["train"]["x"].std(0).clip(1e-6)
    digest = hashlib.sha256()
    for part in parts.values():
        part["x"] = (part["x"]-mean)/std
        for value in part.values():
            digest.update(value.tobytes())
    return parts, {"mean": mean.tolist(), "std": std.tolist(), "sha256": digest.hexdigest()}


def model(config):
    layers, dim = [], 4
    for width in config["hidden"]:
        layers += [nn.Linear(dim, width), nn.ReLU()]
        dim = width
    return nn.Sequential(*layers, nn.Linear(dim, 1))


def regularizer(method, p, s, temperature, config):
    if method == "threshold_gedi":
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        soft = torch.sigmoid((p-config["threshold"])/temperature)
        return gedi(s, soft, degree=config["gedi_degree"], use_torch=True)
    return existing_penalty(method, p, s, config["gedi_degree"])


def jobs(config):
    for scenario, seed in itertools.product(config["scenarios"], config["seeds"]):
        yield {"scenario": scenario, "seed": seed, "method": "baseline", "lambda": 0., "temperature": 0.}
        for method in ("gedi", "mary_chi2", "threshold_gedi"):
            for t in (config["temperatures"] if method == "threshold_gedi" else [0.]):
                for lam in config["lambdas"]:
                    yield {"scenario": scenario, "seed": seed, "method": method, "lambda": lam, "temperature": t}


def run_id(job):
    return f'{job["scenario"]}_s{job["seed"]}_{job["method"]}_l{job["lambda"]:g}_T{job["temperature"]:g}'


def predefined(train_s, config):
    uniform = np.linspace(-1, 1, config["groups"]+1)
    quantile = np.quantile(train_s, np.linspace(0, 1, config["groups"]+1))
    quantile[0], quantile[-1] = -1, 1
    return {"uniform": uniform.tolist(), "quantile": quantile.tolist()}


def group_at_threshold(s, p, boundaries, threshold):
    # Reuse original groups/Wilson CIs; hard decisions use the exact requested comparison.
    stats, rows = group_statistics(s, (p >= threshold).astype(float), boundaries)
    groups = np.searchsorted(boundaries[1:-1], s, side="right")
    for row in rows:
        row["score_mean"] = float(p[groups == row["group"]].mean()) if row["n"] else float("nan")
    means = [r["score_mean"] for r in rows if r["n"]]
    stats["score_gap"] = float(max(means)-min(means))
    stats["weighted_variance"] = stats["weighted_phi_std"]**2
    return stats, rows


def validation_metrics(part, p, boundaries, config):
    values = basic(part["s"], part["y"], p, max_n=config["eval_max_n"])
    values["gedi_score"] = float(gedi(part["s"].astype(float), p.astype(float), degree=config["gedi_degree"]))
    for name, cuts in boundaries.items():
        stats, _ = group_at_threshold(part["s"], p, cuts, .5)
        values.update({f"{name}_{key}": value for key, value in stats.items()})
    values["decision_gap"] = max(values["uniform_dp_gap"], values["quantile_dp_gap"])
    values["near_constant"] = bool(p.std() < config["near_constant_std"])
    prevalence = float((p>=.5).mean())
    values["near_degenerate"] = bool(prevalence < config["near_degenerate_prevalence"] or prevalence > 1-config["near_degenerate_prevalence"])
    values["constant_decision"] = bool(prevalence == 0 or prevalence == 1)
    values["degenerate"] = values["near_constant"] or values["near_degenerate"]
    if "oracle" in part:
        values["oracle_score_mse"] = float(np.mean((p-part["oracle"])**2))
    return values
