# BayesTraj submission reproducibility repository

This repository contains the BayesTraj implementation, the 17 baselines
reported in the submission, compact frozen result tables, and code to
reproduce every empirical figure and numerical claim. Raw datasets, model
weights, AgentBench services, and generated trajectory pools remain external
so that the release stays below 20 MB.

## One reproduction script

All reviewer-facing workflows use one entry point:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-fast.txt
python -m pip install -e . --no-deps
python scripts/reproduce.py paper
```

The `paper` command rebuilds Figures 1–9 and all numerical claims from the
bundled frozen tables, writes them under `outputs/`, and runs the integrity and
claim audit. It is deterministic, CPU-only, and normally completes in under
one minute.

The same script orchestrates reproduction from raw trajectories:

```bash
python scripts/reproduce.py raw --help
python scripts/reproduce.py raw doctor
python scripts/reproduce.py raw generate --backbone qwen35 --model qwen3.5:9b
python scripts/reproduce.py raw analyze --workers 8 --device cuda:0
```

Raw reproduction requires Python 3.13, external scratch space, the pinned
backbone currently served on the configured vLLM endpoints, and the AgentBench
services. Run `raw generate` once for each of the three served backbones. The
driver's `--dry-run` option prints every underlying command without executing
it.

Detailed instructions are kept outside this README:

- [Complete reproduction protocol](docs/REPRODUCTION_PROTOCOL.md)
- [Raw generation and artifact audit](docs/RAW_GENERATION.md)
- [Evaluation, baselines, and report construction](docs/EVALUATION_AND_REPORTS.md)
- [Provenance](PROVENANCE.md)

## Experimental contract

- Datasets: DBBench, HotpotQA, WebShop, and StrategyQA.
- Backbones: Qwen-3.5 9B, Gemma-3 12B, and GPT-OSS 20B.
- Seeds: 101, 202, and 303.
- Shared budgets: `B={3,4,6,8,12,16}`.
- Metrics: task-failure AUROC and AUPR.
- Training: five folds assigned by ordered task index modulo five, fitted
  independently within every dataset/backbone/seed cell.
- Correctness labels are used only after scores and stopping times are frozen.
- WebShop uses the submitted label-free constraint-pattern outcome mapping.

### Frozen task selection

The same task identities are reused for every backbone and seed; seeds change
trajectory generation, not task selection.

| Dataset | Frozen source and selection | Tasks per cell |
|---|---|---:|
| DBBench | Complete AgentBench `dbbench-std` registry, indices 0–299 | 300 |
| HotpotQA | First 1,000 ordered examples from pinned `distractor/validation` | 1,000 |
| WebShop | Complete AgentBench `webshop-std` registry, indices 0–199 | 200 |
| StrategyQA | Every ordered example from the pinned `test` split | 687 |

HotpotQA uses a deterministic prefix, not a random sample. DBBench and WebShop
use every task exposed by their standard AgentBench registries. Exact dataset
revisions, row hashes, and expected task counts are frozen in
`config/bayestraj_raw_generation_configuration.json`.

### Task success by dataset and backbone

Success is the complement of the frozen task-failure label. Each seed column
is the percentage of successful tasks in that dataset/backbone/seed cell; the
last column is the arithmetic mean and sample SD across seeds.

| Dataset | Backbone | Tasks/cell | Seed 101 | Seed 202 | Seed 303 | Mean ± seed SD |
|---|---|---:|---:|---:|---:|---:|
| DBBench | Qwen-3.5 9B | 300 | 63.67% | 61.33% | 62.33% | **62.44% ± 1.17%** |
| DBBench | Gemma-3 12B | 300 | 36.00% | 37.00% | 37.33% | **36.78% ± 0.69%** |
| DBBench | GPT-OSS 20B | 300 | 59.67% | 61.67% | 59.33% | **60.22% ± 1.26%** |
| HotpotQA | Qwen-3.5 9B | 1,000 | 57.00% | 57.30% | 58.90% | **57.73% ± 1.02%** |
| HotpotQA | Gemma-3 12B | 1,000 | 54.10% | 53.80% | 53.90% | **53.93% ± 0.15%** |
| HotpotQA | GPT-OSS 20B | 1,000 | 60.90% | 61.20% | 60.50% | **60.87% ± 0.35%** |
| WebShop | Qwen-3.5 9B | 200 | 18.00% | 21.00% | 18.00% | **19.00% ± 1.73%** |
| WebShop | Gemma-3 12B | 200 | 12.00% | 10.50% | 11.00% | **11.17% ± 0.76%** |
| WebShop | GPT-OSS 20B | 200 | 17.00% | 16.00% | 16.00% | **16.33% ± 0.58%** |
| StrategyQA | Qwen-3.5 9B | 687 | 89.52% | 88.94% | 88.94% | **89.13% ± 0.34%** |
| StrategyQA | Gemma-3 12B | 687 | 93.30% | 93.60% | 93.60% | **93.50% ± 0.17%** |
| StrategyQA | GPT-OSS 20B | 687 | 94.03% | 94.03% | 94.47% | **94.18% ± 0.25%** |

Exact counts are in `data/paper_inputs/dataset_backbone_success_rates.csv`.

### Recorded steps per trajectory

Statistics pool all 16 trajectories across all evaluated tasks and seeds. A
step is one stored trajectory step; SD is calculated across trajectories.

| Dataset | Backbone | Trajectories | Mean steps | Step SD |
|---|---|---:|---:|---:|
| DBBench | Qwen-3.5 9B | 14,400 | **4.76** | 3.81 |
| DBBench | Gemma-3 12B | 14,400 | **3.57** | 2.64 |
| DBBench | GPT-OSS 20B | 14,400 | **2.39** | 0.98 |
| HotpotQA | Qwen-3.5 9B | 48,000 | **2.95** | 1.67 |
| HotpotQA | Gemma-3 12B | 48,000 | **2.25** | 1.00 |
| HotpotQA | GPT-OSS 20B | 48,000 | **1.45** | 0.88 |
| WebShop | Qwen-3.5 9B | 9,600 | **10.26** | 7.08 |
| WebShop | Gemma-3 12B | 9,600 | **4.66** | 2.95 |
| WebShop | GPT-OSS 20B | 9,600 | **5.68** | 4.62 |
| StrategyQA | Qwen-3.5 9B | 32,976 | **1.30** | 0.73 |
| StrategyQA | Gemma-3 12B | 32,976 | **1.12** | 0.36 |
| StrategyQA | GPT-OSS 20B | 32,976 | **1.36** | 0.90 |

Exact values and per-seed summaries are in
`data/paper_inputs/trajectory_step_statistics.csv`.

## Complete baseline settings

All baselines use ordered prefixes of the same task-level trajectory pools.
No baseline hyperparameter was selected from correctness labels.

| Baseline(s) | Submitted setting (recommended setting from the original paper) |
|---|---|
| PPL, PE, LS | Direct stored-likelihood or lexical definitions; mean aggregation; no tuned parameter |
| SE | Semantic-equivalence threshold `0.85`; mean aggregation |
| SentSAR | Temperature `1e-3`; mean aggregation |
| SD | Probability-weighted semantic density; no tuned parameter |
| SNNE | ROUGE-L similarity and temperature `1.0` |
| KLE | DeBERTa-v2-xlarge-MNLI; unweighted NLI graph; threshold `1.5`; heat time `0.4`; normalized von Neumann entropy |
| EigV | Symmetrized NLI entailment affinity and normalized graph Laplacian |
| CoCoA-MaxProb, CoCoA-PPL | Graded NLI similarity with sequence-probability or mean-token-probability confidence |
| MC-OE | Natural-log empirical entropy over BayesTraj's dataset-specific outcome buckets |
| BSE-Fixed, BSE-Adaptive | Dirichlet `alpha=0.5`; 1,000 Monte Carlo samples; 5 integration replicates; 5-fold fitting |
| SAUP | Generic-distance form with `alpha=beta=gamma=1`; pinned RoBERTa-SQuAD semantic encoder |
| UProp | `N=4`; Gaussian-kernel `tau=1`; epsilon `1e-6`; ratio cap `10`; no intrinsic cap or transform |
| Degree | `N=4` candidates per eligible step |

## BayesTraj outcome buckets

BayesTraj first maps each sampled trajectory to a discrete, label-free outcome
bucket. For DBBench, HotpotQA, and StrategyQA, its count statistic is the
natural-log empirical entropy of those buckets. The mapping uses only the task
request, the generated trajectory, and information exposed during execution;
it never uses the reference answer, task reward, or correctness label.

| Dataset | Bucket assigned to a trajectory |
|---|---|
| DBBench | Read-only tasks use a normalized semantic final-answer bucket. State-changing tasks use the canonical replayed final database state when available, with the executed write or write sequence as the fallback. Missing, ambiguous, and failed executions retain separate status buckets. |
| HotpotQA | The final answer is lowercased, stripped, and whitespace-normalized, producing `answer:<text>`. A trajectory without an answer is assigned `no-answer:<terminal-status>`. |
| StrategyQA | Valid final answers map to `answer:yes` or `answer:no`. Missing or invalid answers map to `no-answer:<terminal-status>`. |
| WebShop | A purchase maps to a constraint-pattern signature consisting of request-token coverage (`zero`, `low`, `mid`, or `high`), binary selected-option overlap, and price status (`pass`, `fail`, or `unknown`). No-purchase trajectories retain their detailed action or terminal-failure buckets. |

For WebShop, exact product identity is too fine-grained because several
products may satisfy the same request. BayesTraj therefore applies the fixed
hierarchical functional

\[
h_{\mathrm{WS}}(q)=H_q(C)
+0.25\,q(C=\mathrm{purchase})
 H_q(\phi_{\mathrm{WS}}\mid C=\mathrm{purchase})
+q(C=\mathrm{no\mbox{-}purchase})
 H_q(F\mid C=\mathrm{no\mbox{-}purchase}),
\]

where \(C\) records purchase versus no purchase,
\(\phi_{\mathrm{WS}}\) is the constraint-pattern bucket, and \(F\) is the
detailed no-purchase bucket. The fixed weight
\(0.25=\log(2)/\log(16)\) prevents purchased-product disagreement from
dominating the binary completion uncertainty. The executable definitions are
in `raw_pipeline/src/ltuq/runners/` and the WebShop functional is in
`raw_pipeline/scripts/webshop_outcomes.py`.

## Implementation map

| Location | Contents |
|---|---|
| `src/bayestraj/core.py` | Reference BayesTraj estimator: count prior, linear-Gaussian likelihood, posterior fusion, LCB95, calibration, and stopping |
| `src/bayestraj/pipeline.py` | Orchestration used by the `raw` commands of the single reproduction script |
| `src/bayestraj/paper.py` | Frozen-table aggregation and claim computation |
| `src/bayestraj/figures.py` | Figures 1–9 |
| `raw_pipeline/src/ltuq/` | Full experiment runtime, dataset adapters, model clients, trajectory records, and baseline implementations |
| `raw_pipeline/scripts/` | Raw evaluation, cross-fitting, ablations, sensitivity, aggregation, and report generation |
| `scripts/reproduce.py` | The only public reproduction entry point |

The authoritative method registry is `config/paper.json`. This release excludes
post-submission methods, exploratory stopping rules, outcome-judge variants,
and unreported baselines.

## Dependencies and external assets

- `requirements-fast.txt`: minimal CPU environment for `paper`.
- `requirements.txt`: complete Python environment for both workflows.
- `requirements-raw.txt`: backward-compatible alias to the complete list.
- vLLM runs in the pinned container documented in the detailed protocol.
- AgentBench, dataset assets, model weights, and trajectory caches are external
  and are not bundled.

`MANIFEST.sha256` protects the release files, and `scripts/verify.py` is called
automatically by `python scripts/reproduce.py paper`.
