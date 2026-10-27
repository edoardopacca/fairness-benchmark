"""Paired synthetic tasks. No preprocessing statistic is fitted outside training."""
import hashlib
import numpy as np

SCENARIOS = ("linear", "quadratic", "sinusoidal", "nonmonotonic")


def effect(s, scenario):
    if scenario == "linear":
        return np.sqrt(3) * s
    if scenario == "quadratic":
        return np.sqrt(45) / 2 * (s ** 2 - 1 / 3)
    if scenario == "sinusoidal":
        return np.sqrt(2) * np.sin(2 * np.pi * s)
    if scenario == "nonmonotonic":
        # Even, localized central bump; distinct from the polynomial and periodic cases.
        # Population moments under Uniform(-1,1), evaluated by fixed quadrature.
        grid = np.linspace(-1, 1, 100001)
        g = np.exp(-0.5 * (grid / 0.22) ** 2)
        return (np.exp(-0.5 * (s / 0.22) ** 2) - g.mean()) / g.std()
    raise ValueError(scenario)


def make_data(seed, scenario, sizes):
    rng = np.random.default_rng(seed)
    n = sum(sizes.values())
    s = rng.uniform(-1, 1, n)
    z = rng.normal(size=(n, 3))
    proxy = effect(s, scenario) + 0.35 * rng.normal(size=n)
    x = np.column_stack([z, proxy]).astype("float32")  # S is not a model input.
    logit = 1.3 * z[:, 0] - 0.7 * z[:, 1] + 1.8 * effect(s, scenario)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype("float32")
    order = rng.permutation(n)
    parts, offset = {}, 0
    for name, size in sizes.items():
        ids = order[offset:offset + size]
        parts[name] = {"x": x[ids], "s": s[ids].astype("float32"), "y": y[ids], "ids": ids}
        offset += size
    mean, std = parts["train"]["x"].mean(0), parts["train"]["x"].std(0).clip(1e-6)
    for part in parts.values():
        part["x"] = (part["x"] - mean) / std
    digest = hashlib.sha256()
    for part in parts.values():
        for value in part.values():
            digest.update(value.tobytes())
    return parts, {"mean": mean.tolist(), "std": std.tolist(), "sha256": digest.hexdigest()}
