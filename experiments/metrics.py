"""Independent held-out dependence and classification metrics (never training losses)."""
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, auc, precision_recall_curve


def pearson(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def centered(a):
    return a - a.mean(0)[None, :] - a.mean(1)[:, None] + a.mean()


def rbf(x):
    d2 = (x[:, None] - x[None, :]) ** 2
    positive = d2[d2 > 0]
    scale = np.median(positive) if len(positive) else 1.0
    return centered(np.exp(-d2 / (2 * max(scale, 1e-12))))


def dependence(s, p, max_n=384, permutations=0):
    # The fixed subset and permutation RNG are shared across compared models.
    ids = np.random.default_rng(173).permutation(len(s))[:max_n]
    s, p = np.asarray(s[ids], dtype=float), np.asarray(p[ids], dtype=float)
    a, b = centered(np.abs(s[:, None] - s)), centered(np.abs(p[:, None] - p))
    den = np.sqrt(np.mean(a * a) * np.mean(b * b))
    dcor = np.sqrt(max(np.mean(a * b), 0) / den) if den > 1e-24 else 0.0
    k, l = rbf(s), rbf(p)
    norm = np.sqrt(np.sum(k * k) * np.sum(l * l))
    hsic = float(np.sum(k * l) / norm) if norm > 1e-24 else 0.0
    out = {"dcor": float(dcor), "hsic_normalized": hsic, "dependence_n": len(ids)}
    if permutations:
        rng, count = np.random.default_rng(991), 0
        for _ in range(permutations):
            perm = rng.permutation(len(p))
            null = np.sum(k * l[perm][:, perm]) / norm if norm > 1e-24 else 0
            count += null >= hsic - 1e-14
        out["hsic_permutation_p"] = (count + 1) / (permutations + 1)
    return out


def basic(s, y, p, **kwargs):
    precision, recall, _ = precision_recall_curve(y, p)
    return {"accuracy": float(accuracy_score(y, p >= .5)),
            "pr_auc": float(auc(recall, precision)),
            "average_precision": float(average_precision_score(y, p)),
            "pearson": pearson(s, p), "abs_pearson": abs(pearson(s, p)),
            "score_std": float(p.std()), "positive_rate": float((p >= .5).mean()),
            **dependence(s, p, **kwargs)}


def group_statistics(s, p, boundaries):
    # Half-open bins, tails included. This explicitly fixes upstream predict's tail bug.
    groups = np.searchsorted(np.asarray(boundaries)[1:-1], s, side="right")
    rows = []
    global_rate = (p >= .5).mean()
    for g in range(len(boundaries) - 1):
        mask, n = groups == g, int((groups == g).sum())
        rate = float((p[mask] >= .5).mean()) if n else float("nan")
        if n:
            z = 1.959963984540054
            c = (rate + z*z/(2*n))/(1 + z*z/n)
            half = z * np.sqrt(rate*(1-rate)/n + z*z/(4*n*n))/(1 + z*z/n)
        else:
            c, half = float("nan"), float("nan")
        rows.append({"group": g, "left": boundaries[g], "right": boundaries[g+1],
                     "n": n, "positive_rate": rate,
                     "score_mean": float(p[mask].mean()) if n else float("nan"),
                     "phi": rate - global_rate, "rate_ci_low": c-half, "rate_ci_high": c+half})
    occupied = [r for r in rows if r["n"]]
    rates, scores = [r["positive_rate"] for r in occupied], [r["score_mean"] for r in occupied]
    stats = {"dp_gap": float(max(rates)-min(rates)), "score_gap": float(max(scores)-min(scores)),
             "min_group_n": min(r["n"] for r in rows),
             "empty_groups": sum(r["n"] == 0 for r in rows),
             "weighted_phi_std": float(np.sqrt(sum(r["n"]*r["phi"]**2 for r in occupied)/len(s)))}
    return stats, rows
