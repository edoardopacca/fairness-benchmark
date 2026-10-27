"""Exploratory fixed-partition permutation audit, separate from the training protocol.

Run: python -m experiments.posthoc.audit results/pilot_v1
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def permutation_p(counts, positives, permutations=9999, seed=812):
    """Exact conditional randomization null, sampled by multivariate hypergeometric.

    Under independence, condition on group sizes and total predicted positives;
    distributing the positives uniformly is equivalent to permuting binary outputs.
    Statistic: weighted variance of positive rates. No chi-square approximation.
    """
    counts, positives = np.asarray(counts, dtype=int), np.asarray(positives, dtype=int)
    keep = counts > 0
    counts, positives = counts[keep], positives[keep]
    total, k = counts.sum(), positives.sum()
    rate = k / total
    observed = np.sum((positives/counts - rate)**2 * counts) / total
    if k == 0 or k == total:
        return 1.0
    draws = np.random.default_rng(seed).multivariate_hypergeometric(counts, k, size=permutations)
    null = ((draws/counts - rate)**2 * counts).sum(1) / total
    return (1 + np.sum(null >= observed - 1e-14)) / (permutations + 1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = args.out
    selected = pd.read_csv(out / "selected_test_results.csv")
    rows = []
    for row in selected.itertuples():
        path = out / "runs" / row.run_id
        with np.load(path / "predictions.npz") as arrays:
            s, positive = arrays["test_s"], arrays["test_p"] >= .5
        boundaries = json.loads((path / "partitions.json").read_text())["fairgroups"]
        groups = np.searchsorted(boundaries[1:-1], s, side="right")
        counts = np.bincount(groups, minlength=len(boundaries)-1)
        positives = np.bincount(groups, weights=positive, minlength=len(boundaries)-1).astype(int)
        pvalue = permutation_p(counts, positives)
        rows.append({"run_id": row.run_id, "method": row.method, "scenario": row.scenario,
                     "batch_size": row.batch_size, "seed": row.seed,
                     "p_unadjusted": pvalue, "p_bonferroni": min(pvalue * len(selected), 1),
                     "dp_gap": row.test_fairgroups_dp_gap, "min_group_n": int(counts.min()),
                     "test_dcor": row.test_dcor, "test_hsic_normalized": row.test_hsic_normalized})
    result = pd.DataFrame(rows)
    result.to_csv(out / "fairgroups_permutation_audit.csv", index=False)
    gaps = selected.groupby("method")[["test_uniform_dp_gap", "test_quantile_dp_gap", "test_fairgroups_dp_gap"]].agg(["mean", "std"])
    gaps.columns = ["_".join(c) for c in gaps.columns]
    gaps.to_csv(out / "partition_comparison.csv")
    generalization = selected.copy()
    generalization["validation_minus_train_dcor"] = selected.validation_dcor - selected.train_dcor
    generalization["test_minus_train_dcor"] = selected.test_dcor - selected.train_dcor
    generalization.groupby("method")[["validation_minus_train_dcor", "test_minus_train_dcor"]].agg(["mean", "std", "max"]).to_csv(out / "fairness_generalization.csv")
    record = {"analysis": "exploratory_posthoc_fixed_partition_audit", "permutations": 9999,
              "statistic": "weighted variance of predicted positive rates",
              "multiplicity": f"Bonferroni over {len(selected)} validation-selected runs",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "training_source_sha256": json.loads((out / "manifest.json").read_text())["identity"]["source_sha256"],
              "tested_runs": len(result), "significant_bonferroni_005": int((result.p_bonferroni <= .05).sum())}
    (out / "posthoc_audit_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print("Low global estimate (dCor < 0.1), local gap > 0.15:")
    print(result[(result.test_dcor < .1) & (result.dp_gap > .15)].to_string(index=False))


if __name__ == "__main__":
    main()
