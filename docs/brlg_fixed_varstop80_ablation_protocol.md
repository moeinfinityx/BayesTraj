# BayesTraj component-ablation protocol

## Frozen scope

The study compares the submitted fixed and adaptive modes of the same
linear-Gaussian BayesTraj estimator. Datasets, backbones, seeds, ordered
trajectory pools, budgets `{3,4,6,8,12,16}`, the label-free `OE16` target, the
257-point entropy grid, and five ordered task folds remain fixed. Correctness
labels are used only for final AUROC, AUPR, and paired inference.

The complete estimator uses the count-entropy prior, trajectory features, a
degree-1 Gaussian trajectory likelihood, full multivariate covariance, the
entropy-grid posterior update, and `mu_n - 1.96 sigma_n` scoring. The adaptive
mode stops at the first posterior-variance crossing in
`[max(2,B-4),B-1]`, falling back to `T=B`; its boundary is calibrated on the
training folds to mean cost `0.8B`.

## Registered contrasts

Only the five mechanisms reported in the submission are evaluated:

1. Gaussian trajectory-feature update versus no trajectory update;
2. count-prior fusion versus trajectory-likelihood-only estimation;
3. full multivariate covariance versus diagonal covariance;
4. adaptive variance stopping versus fixed `T=B`;
5. task-adaptive allocation versus a cost-matched nonadaptive 80% allocation.

Every ablated posterior and stopping boundary is refitted inside the four
training folds and evaluated on the held-out fold. The nonadaptive control
assigns floor/ceiling prefixes using only a fixed hash of the sample ID; it
does not use correctness, posterior state, or task outcome.

## Inference and reporting

Macro results give each dataset-backbone-seed cell equal weight. Task-level
paired bootstrap is nested inside a hierarchical bootstrap over
dataset-backbone combinations and seeds. Use 500 task replicates and 10,000
hierarchical replicates, with Holm correction over these five registered
contrasts. Report AUROC, AUPR, realized trajectories, savings, paired 95%
intervals, and cell wins/ties/losses.
