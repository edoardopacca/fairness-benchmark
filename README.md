# Continuous-attribute fairness benchmark

Small, reproducible PyTorch experiments on fairness with a continuous sensitive attribute.
The main question is whether regularizing prediction scores also improves fairness of
binary decisions after thresholding.

## Run

Python 3.11, CPU. Create the pinned Conda environment:

```bash
conda env create -f environment.yml
conda activate fairness-benchmark
python reproduce.py smoke
```

The smoke command runs 19 tests, the analytic diagnostic, and small training/evaluation
grids for both experiments. Then reproduce either full experiment:

```bash
python reproduce.py threshold   # Score vs decision fairness: 2,190 runs, 10 seeds
python reproduce.py pilot       # Initial comparison: 520 runs, 5 seeds
```

`python reproduce.py all` runs the smoke checks and both full experiments.
Alternatively install `requirements.txt` in a Python 3.11 environment.

## Experiments

- **Pilot:** baseline, Pearson, Mary chi-square, GeDI and neural HGR; four synthetic
  dependence structures, two minibatch sizes and multiple regularization strengths.
- **Threshold study:** baseline, score GeDI, threshold-aware GeDI and Mary chi-square;
  controlled mean-score parity, quadratic and nonmonotonic scenarios; thresholds 0.4/0.5/0.6.

The threshold-aware variant applies the same GeDI computation to
`sigmoid((P - 0.5) / T)` instead of `P`. It is an experimental adaptation.
**FairGroups is used only for held-out auditing**, never training or model selection.
Hyperparameters use validation only; audit groups are fitted separately and frozen for test.
These are controlled comparisons, not complete reproductions of the papers' experiments.

## Outputs and files

Outputs go to `results/pilot/` and `results/threshold_fairness/`: per-seed CSVs, aggregate
tables, plots, predictions, checkpoints, source hashes and replay verification.
Results and caches are excluded from Git. Use `--output PATH` for another output directory.
Completed runs resume automatically when sources/configurations match; use a new directory
after changing them. Do not run two coordinators on the same output directory.

- `experiments/`: data, penalties, training, evaluation, plots and tests.
- `experiments/configs/` and `experiments/threshold_fairness/*.json`: exact experiment settings.
- `third_party/`: only the required original source files, licenses, pinned commits and hashes.

No external datasets, cloned repositories, Git installation or network access are needed
after installing the Python dependencies. Numerical results are deterministic in the tested
CPU environment; small differences may occur across platforms or library builds.

Method details and deviations: [METHODS.md](experiments/METHODS.md).
Reference code and citations: [third_party/README.md](third_party/README.md).
