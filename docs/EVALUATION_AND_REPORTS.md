# Evaluation, baselines, and report construction

This guide starts after all three backbones have complete audited Z=16 and
Z=12/N=4 pools. One command runs the full CPU/GPU post-generation pipeline:

```bash
. .venv-raw/bin/activate
export BAYESTRAJ_RAW_ROOT=/external/scratch/bayestraj-raw
export BAYESTRAJ_N4_ROOT=/external/scratch/bayestraj-z12-n4
export BAYESTRAJ_EVAL_ROOT=/external/scratch/bayestraj-evaluations
export BAYESTRAJ_WORK=/external/scratch/bayestraj-analysis
export PYTHONPATH="$PWD/raw_pipeline:$PWD/raw_pipeline/scripts"

python scripts/reproduce.py raw analyze --workers 8 --device cuda:0
```

Use `--dry-run` to print the complete execution plan. Every stage is resumable
or writes to a distinct output directory; raw trajectory pools are read-only.

## Pipeline stages

| Order | Stage | Preserved implementation | Main output |
|---:|---|---|---|
| 1 | Audit Z=16 pools | `scripts/audit_raw_generation.py` | `raw_generation_audit.json` |
| 2 | Audit Z=12/N=4 campaign | `scripts/build_z12_n4_manifest.py` | strict campaign manifest |
| 3 | Materialize 36 label-free checkpoints | `materialize_bayestraj_checkpoints.py` | per-cell count priors, trajectory features, `OE16`, fold IDs |
| 4 | Fixed/adaptive component evaluation | `evaluate_brlg_fixed_varstop80_ablation.py` | held-out task scores for 36 cells |
| 5 | Component aggregation | `aggregate_brlg_fixed_varstop80_ablation.py` | ablation tables and Figure 7 source |
| 6 | Adaptive sensitivity | `evaluate_brlg_varstop_sensitivity.py` and its aggregator | `rho×w` sensitivity results |
| 7 | WebShop mapping | `evaluate_bayestraj_webshop_constraint_pattern.py` | nine constraint-pattern cells |
| 8 | Core baselines | `evaluate_mixed_budget_core_multiseed.py` | PPL, LS, PE, SE, SD, SentSAR, MC-OE, UProp, Degree |
| 9 | Semantic baselines | `evaluate_z16_semantic_baselines_multiseed.py` | SNNE, KLE, EigV, both CoCoA variants |
| 10 | SAUP and BSE | dedicated SAUP and BSE evaluators | SAUP, BSE-Fixed, BSE-Adaptive |
| 11 | Curve registry and reports | report and empirical-package builders | all paper-facing tables and plots |

All script paths in the table are under `raw_pipeline/scripts/` unless shown
otherwise. The exact expanded commands and output filenames are in
[REPRODUCTION_PROTOCOL.md](REPRODUCTION_PROTOCOL.md#37-produce-the-36-bayestraj-evaluation-checkpoints).

## Cross-fitting and label isolation

Every dataset/backbone/seed cell is trained independently. Ordered task index
modulo five assigns folds. For each held-out fold, only the other four folds
fit the linear-Gaussian likelihood, label-free `OE16` target model, count-prior
support, and adaptive threshold. Correctness labels are opened only after
scores and stopping times have been frozen.

## Budget alignment

- Fixed methods consume the first `B` ordered trajectories.
- BayesTraj-Adaptive searches `max(2,B-4),...,B-1`, stops at the first prefix
  satisfying its variance certificate, and otherwise uses `B`.
- Adaptive points are plotted at realized mean cost.
- UProp and Degree use the separate exact `Z=12,N=4` campaign and therefore
  end at budget 12.
- All other fixed baselines use prefixes of the Z=16 pools.
- Baseline-only `B=2` points are contextual and excluded from shared numerical
  comparisons.

## External evaluation models

The semantic stage downloads `microsoft/deberta-v2-xlarge-mnli`; the SAUP
stage downloads the pinned `deepset/roberta-base-squad2` revision. These stages
use a GPU but do not call the three agent backbones or generate trajectories.
All other post-generation stages are CPU-only.

## Final verification

After raw analysis completes, copy the regenerated compact tables into a
separate comparison location rather than overwriting bundled paper inputs.
Compare numeric CSV values to `data/paper_inputs/` with floating-point
tolerance. PDF byte hashes may differ across font and Matplotlib versions.

Finally run the same public entry point used by reviewers:

```bash
python scripts/reproduce.py paper
```

It rebuilds Figures 1–9, recalculates the manuscript claims, and verifies the
release hashes and 48 frozen numerical claims.

