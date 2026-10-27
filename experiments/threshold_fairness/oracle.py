"""Analytic and Monte Carlo checks, kept separate from learned predictions."""
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/threshold_fairness/oracle"))
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(-1, 1, 10001)
    q = .6 + .2 * grid
    p0, p1 = .5 - .4*q, .9 - .4*q
    conditional_mean = (1-q)*p0 + q*p1
    conditional_rate = (1-q)*(p0 >= .5) + q*(p1 >= .5)
    np.testing.assert_allclose(conditional_mean, .5, atol=2e-16)
    np.testing.assert_allclose(conditional_rate, q, atol=2e-16)
    rng = np.random.default_rng(27012026)
    s = rng.uniform(-1, 1, 1_000_000)
    qq = .6 + .2*s
    b = rng.uniform(size=len(s)) < qq
    p = .5 + .4*(b-qq)
    y = rng.uniform(size=len(s)) < p
    np.testing.assert_array_equal(p >= .5, b)
    bins = np.linspace(-1, 1, 11)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (s >= lo) & (s < hi)
        n = int(m.sum())
        row = {"left": lo, "right": hi, "n": n, "mean_s": float(s[m].mean()),
               "mean_oracle_score": float(p[m].mean()), "mean_y": float(y[m].mean()),
               "positive_decision_rate": float((p[m] >= .5).mean()),
               "expected_decision_rate": float(qq[m].mean())}
        assert abs(row["mean_oracle_score"]-.5) < 6*.3/np.sqrt(n)
        assert abs(row["mean_y"]-.5) < 6*.5/np.sqrt(n)
        assert abs(row["positive_decision_rate"]-row["expected_decision_rate"]) < 6*.5/np.sqrt(n)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "monte_carlo_bins.csv", index=False)
    result = {"kind": "analytic_oracle_not_learned_predictions", "monte_carlo_n": len(s),
              "max_analytic_mean_error": float(abs(conditional_mean-.5).max()),
              "max_analytic_rate_error": float(abs(conditional_rate-q).max()),
              "monte_carlo_score_mean": float(p.mean()), "monte_carlo_y_mean": float(y.mean()),
              "monte_carlo_positive_rate": float(b.mean()),
              "monte_carlo_oracle_accuracy": float(((p>=.5)==y).mean()),
              "analytic_oracle_accuracy": .5+.8*(.24-.04/3),
              "all_checks_passed": True}
    (out / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
