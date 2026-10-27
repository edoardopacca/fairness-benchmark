# Methods and interpretation

All classifiers minimize binary cross-entropy plus a weighted fairness penalty.
Architecture, splits, initialization and minibatch order are shared within each scenario/seed.
The configurations preserve the original experiments; packaging changes concern paths,
dependency loading and command orchestration, not mathematical objectives.

| Method | Computation | Status |
| --- | --- | --- |
| Baseline | No fairness term | Shared MLP |
| Pearson | Squared batch correlation, denominator epsilon 1e-12 | Simple baseline |
| Mary | KDE chi-square between scores and S | Original numerical code; zero-variance guard at SD<1e-7 |
| GeDI | L1 norm of least-squares coefficients on centered S, S², S³ | Exact original computation; fixed-lambda training differs from the paper's dual constraints |
| Neural HGR | Alternating learned transformations maximizing normalized correlation | Independently implemented approximation inspired by Grari; five adversary steps per batch |
| Threshold GeDI | GeDI on sigmoid((P-0.5)/T) | Our experimental adaptation, same basis and solver |

Mary's chi-square upper-bounds squared HGR for the estimated distribution, not automatically
for the unknown population. GeDI controls a finite basis and depends on scale. Neural HGR
does not guarantee the inner optimum. No method is claimed to reproduce a full paper protocol.

The controlled scenario has q(S)=0.6+0.2S, B|S~Bernoulli(q),
P*=0.5+0.4(B-q), Y|S,B~Bernoulli(P*). Its mean score and mean label conditional on S
are 0.5, but its threshold-0.5 positive rate is q(S). Networks learn sampled labels;
oracle scores are evaluated separately. Constant conditional means do not imply independence.

Evaluation separates score dependence (Pearson, distance correlation, normalized HSIC,
mean-score gaps and score SD) from hard-decision accuracy, prevalence and group gaps.
PR-AUC always uses continuous scores. Global nonlinear estimators use a fixed subset;
group audits use the full held-out split. Small finite estimates do not certify independence.

FairGroups is **audit-only**: fit on independent audit predictions, freeze boundaries,
then evaluate test predictions. Uniform and training-quantile partitions provide references.
Its optimized weighted variance differs from maximum group gap. Our adapter extends the
outer groups to the known support and uses disjoint intervals; the original search is unchanged.
Small/empty groups are reported. Permutation audits correct for multiple testing.

The threshold study seals validation-only selections before generating test/audit scores.
Its primary selection minimizes predefined decision disparity among nondegenerate models
within two percentage points of baseline mean validation accuracy. Secondary selection
targets a validation gap at most 0.1. Infeasible cases stay unavailable. Test frontiers are
descriptive. Seed intervals condition on selection and omit search uncertainty; the threshold
variant searches four temperatures, so its search budget is larger.

Full threshold specification: [PROTOCOL.md](threshold_fairness/PROTOCOL.md).
Analytic diagnostic: [HYPOTHESIS.md](threshold_fairness/HYPOTHESIS.md).
