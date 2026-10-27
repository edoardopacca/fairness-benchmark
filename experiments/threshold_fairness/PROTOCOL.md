# Frozen protocol: score-to-decision fairness

Written before the new training grid and before looking at new test predictions.
The oracle verification precedes this protocol; it is a diagnostic, never learned output.

## One experiment

Three scenarios: controlled mean-parity counterexample, original quadratic, original
nonmonotonic/localized effect. The original generators remain unchanged and are imported.
Ten new independent seeds (100--109); for each scenario/seed, identical observations,
splits, model initialization, minibatch order and training budget across all methods.
Score preprocessing is fitted on training only. Controlled inputs are (S,B,Z1,Z2),
not oracle probabilities. Other scenarios preserve their original inputs.

Baseline, GeDI on scores, GeDI on soft decisions, Mary chi-square on scores.
GeDI uses the exact upstream degree-3 centered monomial least-squares/L1 computation.
Soft decisions are sigmoid((P−0.5)/T), T∈{0.02,0.05,0.1,0.2}.
The soft-decision variant is our experimental adaptation, not a new reproduced paper
algorithm or a novelty claim. Mary uses the existing direct upstream adapter unchanged.
Pearson and neural HGR are not repeated; they remain in the original pilot.
FairGroups is absent from the training and validation-selection code paths.

12 lambdas from 0.01 to 10, densely spaced between 0.03 and 1. Each scenario has
73 configurations (1 baseline + 12 score GeDI + 48 threshold GeDI + 12 Mary),
for 2190 complete runs. MLP 4→32→16→1, ReLU, sigmoid scores; BCE on logits,
Adam lr 0.003, batch 256, 40 epochs, 3-epoch common prediction-only warmup,
gradient clipping at 5. No early stopping or selected checkpoints. All methods see
identical epochs and updates. Record gradient norms, clipping, soft saturation and
regularizer gradients on the last minibatch of each epoch. These are diagnostics,
not the independent fairness evaluation.

## Validation-only selection, frozen before test evaluation

Validation uses three uniform bins on [-1,1] and three quantile bins whose internal
cuts come from training. The primary decision disparity is the larger of the two
maximum positive-rate gaps at threshold 0.5. No FairGroups partitions are used here.
Aggregate accuracy and disparity across all ten seeds for each configuration.

A configuration is nondegenerate if **every validation seed** has score SD≥0.01 and
positive prevalence in [0.01,0.99]. These operational flags do not certify utility.
Two separately labelled operating points are fixed:

1. **Performance matched (primary):** retain configurations with mean validation
   accuracy at least baseline mean accuracy minus 0.02 and nondegenerate predictions;
   minimize mean primary decision disparity. Ties: higher accuracy, smaller lambda,
   then larger temperature. If no feasible configuration exists, report unavailable;
   do not substitute a model outside the accuracy band.
2. **Fairness target (secondary):** among nondegenerate configurations with mean primary
   gap≤0.10, maximize mean accuracy. Ties: smaller gap, smaller lambda, larger T.
   If none pass, report unavailable. This target is operational, not a fairness certificate.

Select lambda/T once per scenario/method/operating point using all validation seeds.
Also select a performance-matched lambda separately at each temperature for H3.
Seal selections to JSON/CSV and record their hash before generating any test scores.
Never consult audit data for selection. Test performance need not remain within 2 pp:
report the observed test accuracy difference and uncertainty rather than rescreening it.
Test grid frontiers are explicitly descriptive; validation frontiers are also saved.
No hyperparameter or threshold will be revised based on this test.

## Evaluation and auditing

Evaluate frozen models at 0.5 primarily, and fixed sensitivity thresholds 0.4 and 0.6.
Score metrics: Pearson, independent dCor/normalized Gaussian HSIC (same fixed 384-point
subsample per split), predefined mean-score gaps, score SD, PR-AUC (continuous scores).
Decision metrics: threshold-specific accuracy/prevalence, group positive rates, maximum
gap, weighted variance of group rates. PR-AUC is never computed from hard decisions.
Flag SD<0.01, prevalence outside [0.01,0.99], and exact all-positive/all-negative outputs.

For every frozen model and threshold, fit official FairGroups only on independent
audit binary predictions (three groups, grid 16), freeze cuts, evaluate on test.
Uniform/quantile partitions use the same number of groups. Preserve the existing
tail extension and disjoint-bin adapter, and save group sizes/rates. Weighted variance
is the FairGroups objective; it is not the maximum gap and need not rank models identically.
Audit thresholds separately, without feedback to training. Save test profiles in 20 fixed bins.

For inferential auditing, use the existing conditional hypergeometric permutation test,
99999 draws, on the union of validation-selected main operating-point models (excluding
temperature-only ablation models unless also selected in the main union), all thresholds
and all three partitions. Bonferroni corrects this complete family; record family size
before inspecting p-values. Constant decisions have p=1. Group rate CI are pointwise Wilson.
Descriptive full-grid plots never imply significance for every grid point.

Report mean, sample SD, and 95% Student-t intervals across independent seeds, n=10.
Primary method contrasts use paired seed differences; exact two-sided sign-flip tests
under paired exchangeability, with Holm correction across all available primary
threshold-GeDI versus score-GeDI/Mary contrasts (three scenarios, two operating points,
accuracy and predefined decision gap). Audit significance is a separate family.
These intervals condition on validation selection and omit search uncertainty; the same
seed cohort selects configurations, so results are preliminary rather than confirmatory.

## Integrity and compute

Smoke uses all four methods/all temperatures, three scenarios, two seeds, two lambdas,
small datasets and three epochs; not evidence for H1--H5. Complete grid follows only
after numerical/gradient/leakage tests pass. Original pilot and all upstream code remain
unchanged. Use the existing Conda CPU environment and three single-thread worker
processes; estimate runtime from smoke before deciding whether HPC is necessary.
Each run saves checkpoint/history; model scores on test/audit are generated only after
the grid's validation selections have been sealed. The generator constructs all four
splits deterministically, but training/selection consult only train/validation. Config, source snapshots, per-file
hashes, upstream commits, environment and output hashes are recorded.
