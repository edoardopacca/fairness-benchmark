# Reference code

Only the files used by the experiments are included. Original file contents and licenses
are preserved; `manifest.json` records original paths, repository commits and SHA256 hashes.
The loader verifies them before use. No repository download is required.

- `mary/`: `hgr.py`, `pytorch_kde.py`; Apache-2.0.
  [Code](https://github.com/criteo-research/continuous-fairness),
  [Mary et al., ICML 2019](https://proceedings.mlr.press/v97/mary19a.html).
- `gedi/`: original `src/metrics.py`; MIT. Only its two numerical static methods are
  executed from their AST, avoiding unrelated Moving Targets/Gurobi dependencies.
  [Code](https://github.com/giuluck/GeneralizedDisparateImpact),
  [Giuliani et al., ICML 2023](https://proceedings.mlr.press/v202/giuliani23a.html).
- `fairgroups/`: original partition estimation and fairness metrics; MIT. The loader
  bypasses the upstream package initializer's unused visualization imports.
  [Code](https://github.com/artefactory/fair_groups),
  [Shilova et al., ECAI 2025](https://arxiv.org/abs/2507.11247).

Neural HGR is independently implemented in `experiments/penalties.py`, inspired by
[Grari et al., IJCAI 2020](https://www.ijcai.org/proceedings/2020/313).
No code from the unlicensed HGR_NN checkout or the unused W2reg repository is included.
