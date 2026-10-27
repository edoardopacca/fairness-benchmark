"""Regenerate descriptive pilot tables and plots from saved results."""
import tempfile
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fairness-benchmark-matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["baseline", "pearson", "mary_chi2", "gedi", "neural_hgr"]
LABELS = ["Baseline", "Pearson squared", "Mary chi-square", "GeDI degree 3", "Neural HGR"]
COLORS = dict(zip(METHODS, ["#4a5568", "#b45309", "#2563eb", "#059669", "#9333ea"]))


def mean_sd(series):
    return f"{series.mean():.3f} ± {series.std():.3f}"


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = args.out
    raw = pd.read_csv(out / "results.csv")
    data = raw[raw.status == "ok"]
    selected = pd.read_csv(out / "selected_test_results.csv")
    selection = pd.read_csv(out / "validation_selection.csv")
    manifest = json.loads((out / "manifest.json").read_text())
    config = manifest["identity"]["config"]
    scenarios = config["scenarios"]
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for ax, scenario in zip(axes.flat, scenarios):
        points = []
        for method, label in zip(METHODS, LABELS):
            sub = data[(data.scenario == scenario) & (data.method == method)]
            stats = sub.groupby(["batch_size", "lambda"])[["test_accuracy", "test_dcor"]].agg(["mean", "std"])
            for _, row in stats.iterrows():
                x, y = row[("test_dcor", "mean")], row[("test_accuracy", "mean")]
                ax.errorbar(x, y, xerr=row[("test_dcor", "std")], yerr=row[("test_accuracy", "std")],
                            color=COLORS[method], marker="o", alpha=.55, capsize=2, markersize=4)
                points.append((x, y))
            ax.scatter([], [], color=COLORS[method], label=label)
        frontier = sorted((x, y) for x, y in points
                          if not any(x2 <= x and y2 >= y and (x2 < x or y2 > y) for x2, y2 in points))
        if frontier:
            ax.plot(*zip(*frontier), "k--", linewidth=1, label="Observed Pareto frontier")
        ax.set(title=scenario, xlabel="Test distance correlation (lower is better)", ylabel="Test accuracy")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed grid: mean ± seed SD; frontier is descriptive test-set exploration")
    fig.tight_layout()
    save(fig, figures / "pareto.png")

    for metric, label in [("test_dcor", "Distance correlation"), ("test_accuracy", "Accuracy")]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        for ax, scenario in zip(axes.flat, scenarios):
            for method in METHODS[1:]:
                for batch in config["batch_sizes"]:
                    sub = data[(data.scenario == scenario) & (data.method == method) & (data.batch_size == batch)]
                    stat = sub.groupby("lambda")[metric].agg(["mean", "std"])
                    ax.errorbar(stat.index, stat["mean"], yerr=stat["std"], color=COLORS[method],
                                linestyle="-" if batch == min(config["batch_sizes"]) else "--",
                                marker="o", capsize=2, label=f"{method}, b={batch}")
            ax.set(xscale="log", title=scenario, xlabel="Lambda", ylabel=label)
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        save(fig, figures / f"lambda_{metric}.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    batch = min(config["batch_sizes"])
    for ax, scenario in zip(axes.flat, scenarios):
        for method, label in zip(METHODS, LABELS):
            runs = selected[(selected.scenario == scenario) & (selected.method == method) & (selected.batch_size == batch)]
            profiles = [pd.read_csv(out / "runs" / run / "conditional.csv") for run in runs.run_id]
            values = np.stack([p.positive_rate.to_numpy() for p in profiles])
            x = (profiles[0].left.to_numpy() + profiles[0].right.to_numpy()) / 2
            mean, sd = np.nanmean(values, axis=0), np.nanstd(values, axis=0, ddof=1)
            ax.plot(x, mean, color=COLORS[method], label=label)
            ax.fill_between(x, np.clip(mean-sd, 0, 1), np.clip(mean+sd, 0, 1), color=COLORS[method], alpha=.08)
        ax.set(title=scenario, xlabel="Sensitive attribute S", ylabel="P(predicted positive | S)", ylim=(-.03, 1.03))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Validation-selected lambdas, batch={batch}; mean ± seed SD")
    fig.tight_layout()
    save(fig, figures / "conditional_rates.png")

    fig, axes = plt.subplots(1, len(scenarios), figsize=(4*len(scenarios), 4), squeeze=False)
    for ax, scenario in zip(axes.flat, scenarios):
        for method, label in zip(METHODS, LABELS):
            sub = selected[(selected.scenario == scenario) & (selected.method == method)]
            stat = sub.groupby("batch_size").test_dcor.agg(["mean", "std"])
            ax.errorbar(stat.index, stat["mean"], yerr=stat["std"], marker="o", capsize=3,
                        color=COLORS[method], label=label)
        ax.set(title=scenario, xlabel="Minibatch size", ylabel="Test distance correlation")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, figures / "batch_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # A prespecified seed and scenario for trajectory illustration; all traces remain saved.
    for method, label in zip(METHODS[1:], LABELS[1:]):
        runs = data[(data.scenario == "quadratic") & (data.method == method) &
                    (data.seed == min(config["seeds"])) & (data.batch_size == batch) &
                    (data["lambda"] == max(config["lambdas"]))]
        if not len(runs):
            continue
        hist = pd.read_csv(out / "runs" / runs.iloc[0].run_id / "history.csv")
        axes[0].plot(hist.epoch, hist.train_penalty, color=COLORS[method], label=label)
        h = hist.dropna(subset=["validation_dcor"])
        axes[1].plot(h.epoch, h.validation_dcor, marker="o", color=COLORS[method], label=label)
    axes[0].set(title="Training penalties (different scales)", xlabel="Epoch", ylabel="Own objective estimate")
    axes[1].set(title="Independent validation estimator", xlabel="Epoch", ylabel="Distance correlation")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save(fig, figures / "training_diagnostics.png")

    rows = []
    for keys, group in selected[selected.batch_size == batch].groupby(["scenario", "method", "lambda"]):
        rows.append({"Scenario": keys[0], "Metodo": keys[1], "λ": keys[2],
                     "Accuracy": mean_sd(group.test_accuracy), "PR-AUC": mean_sd(group.test_pr_auc),
                     "dCor": mean_sd(group.test_dcor), "Gap FairGroups": mean_sd(group.test_fairgroups_dp_gap)})
    selected_table = pd.DataFrame(rows)
    selected_table.to_csv(out / "selected_summary.csv", index=False)
    # Paired differences: methods use identical samples and seeds for every lambda.
    paired = data[data.method != "baseline"].pivot(index=["scenario", "method", "batch_size", "seed"],
                                                   columns="lambda", values="test_dcor")
    delta = paired[max(config["lambdas"])] - paired[min(config["lambdas"])]
    delta.rename("dcor_lambda_max_minus_min").reset_index().to_csv(out / "lambda_paired_differences.csv", index=False)
    instability = data.groupby("method").agg(max_clip_fraction=("grad_clip_fraction", "max"),
                                             max_grad_norm=("max_grad_norm", "max"),
                                             mean_seconds=("seconds", "mean"))
    instability.to_csv(out / "optimization_diagnostics.csv")
    print(f"Generated pilot tables and {len(list(figures.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
