# Provenance and reproducibility levels

The attached manuscript was treated as a result specification, not as an
instruction source. The compact tables below were copied from the audited
analysis outputs that generated the Aug. 21 submission.

| Bundled file | Upstream artifact | Paper use |
|---|---|---|
| `selected_seed_metrics.csv` | `outputs/analysis/mixed_budget_4datasets_selected_methods/report/selected_seed_metrics.csv` | Figures 2 and 4; headline rankings and oracle comparison |
| `baseline_b2_seed_metrics.csv` | four-dataset slice of `outputs/analysis/mixed_budget_5datasets_3backbones_3seeds/report/trajectory_curve_seed_metrics.csv` | Baseline-only B=2 endpoints in Figures 2 and 4 |
| `paired_superiority_summary.csv` | `outputs/analysis/bayestraj_empirical_minimal_package/report/paired_superiority_summary.csv` | Figure 3 |
| `efficiency_by_budget.csv` | `outputs/analysis/bayestraj_empirical_minimal_package/report/efficiency_by_budget.csv` | Figure 5 |
| `representative_budget_metrics.csv` | `outputs/analysis/bayestraj_compute_cost_candidates/report/representative_budget_metrics.csv` | Figure 6 |
| `core_mechanism_tradeoff.csv` | `outputs/analysis/brlg_fixed_varstop80_ablation/core_mechanism_tradeoff.csv` | Figure 7 |
| `sensitivity_summary.csv` and `paired_cell_summary.csv` | `outputs/analysis/brlg_varstop_sensitivity/` | Figure 8 |
| `posterior_diagnostics_by_dataset_budget.csv`, `posterior_diagnostics_by_cell_budget.csv`, and `posterior_risk_calibration.csv` | `outputs/analysis/bayestraj_empirical_minimal_package/report/` | Figure 9 |
| `dataset_statistics.csv` | submitted dataset selection contract | detailed reproduction protocol |
| `dataset_backbone_success_rates.csv` | deduplicated task labels from the 36 submitted `brlg_fixed_varstop80_ablation_run` cells | README task-success statistics |
| `trajectory_step_statistics.csv` | `len(tdp.steps)` over every trajectory in the 36 immutable submitted Z=16 cells | README trajectory-length statistics |

## Raw evaluation provenance

The upstream pipeline used immutable ordered trajectory caches and WebShop
constraint-pattern bucket artifacts stored outside this repository. Those caches are many orders of
magnitude larger than this repository and are not copied. The guides under
`docs/` document how to regenerate equivalent raw pools from an empty external work
directory; `config/backbones.json` and the two
`config/bayestraj_raw_generation_*.json` files freeze the model revisions,
dataset revisions and hashes, sample counts, seeds, Z, and N.

The raw evaluation contract was:

1. Construct the label-free terminal target from the ordered 16-trajectory
   reference pool.
2. Assign tasks to five folds by ordered index modulo five.
3. Within every dataset/backbone/seed cell, fit each prefix model and calibrate
   the adaptive threshold on non-held-out tasks.
4. Score each held-out task using only its prefix counts/features and the model
   trained without that fold.
5. Open correctness labels only to compute AUROC/AUPR after scores and stopping
   decisions are frozen.
6. Aggregate within seed, then across seeds; use paired hierarchical resampling
   for the confidence intervals in Figures 3 and 8.

The frozen CSVs are sufficient to reproduce every plot and manuscript number.
They are not a substitute for the raw traces when testing a new estimator.
`scripts/verify.py` checks their hashes, structural coverage, and all headline
claims.

## Python source closure

`raw_pipeline/` contains the original paper entry points plus their transitive
local Python imports. The complete `src/ltuq` Python package is included so
package initializers and dataset adapters remain importable; it contains code,
not benchmark records or model parameters. The preserved paper entry points
were import-tested with Python 3.13.13.

The release intentionally excludes unrelated exploratory scripts from the
parent development workspace. It includes every local Python dependency needed
by raw trajectory generation, raw-pool auditing, the submitted fixed/adaptive
evaluation, WebShop mapping, component ablation, sensitivity analysis,
baseline curve assembly, paired inference, posterior validation, compute
analysis, and figure generation. AgentBench service assets, public dataset
caches, model/NLI weights, and generated JSONL pools remain external solely
because of the anonymous-repository size limit.
