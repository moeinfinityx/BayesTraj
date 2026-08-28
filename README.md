# BayesTraj submission reproducibility repository

This repository regenerates every empirical figure and numerical claim in
`BayesTraj-Aug-21-submission.pdf` from compact, frozen evaluation tables. It
also contains a readable implementation of the BayesTraj posterior and
adaptive stopping rule.

The large trajectory caches are intentionally not copied here. They are
immutable experimental inputs and would violate the 20 MB release limit.
`PROVENANCE.md` documents where each bundled table came from and how the raw
evaluation was performed.

## Reviewer reference: settings and budget alignment

This first section collects every experimental setting and alignment rule
promised by the manuscript's anonymous-repository references.

### Experimental contract

- Datasets: DBBench, HotpotQA, WebShop, StrategyQA.
- Backbones: Qwen-3.5 9B, Gemma-3 12B, GPT-OSS 20B.
- Seeds: 101, 202, 303.
- Shared budgets: 3, 4, 6, 8, 12, 16. Baseline-only `B=2` points appear in
  plots when present but are excluded from quantitative comparisons.
- Predictive metrics: task-failure AUROC and AUPR.
- Curves: mean and sample standard deviation across three seed macros.
- Training: five folds assigned by ordered task index modulo five, fitted
  independently within each dataset/backbone/seed cell.
- Correctness labels are used only after held-out scores and stopping times are
  frozen.
- WebShop BayesTraj uses the label-free constraint-pattern outcome mapping.

### Dataset statistics

Each benchmark task is evaluated for three backbones and three seeds, producing
nine dataset-specific cells. Thus, the 2,700 DBBench evaluations below are
**not 2,700 distinct tasks or a 300-task subsample of 2,700**: they are the
same 300 evaluated DBBench tasks repeated across 3 backbones × 3 seeds.
`Task–backbone–seed evaluations` equals evaluated tasks per cell times nine,
and `Cached trajectories` equals those evaluations times the frozen pool size
`Z=16`.


The task identities were fixed once and reused for every backbone and seed.
Seeds control stochastic trajectory generation; they do not resample tasks.

| Dataset | Frozen source and selection | Tasks per cell |
|---|---|---:|
| DBBench | Complete AgentBench `dbbench-std` registry, ordered indices 0–299 | 300 |
| HotpotQA | First 1,000 ordered examples from the pinned `distractor/validation` split | 1,000 |
| WebShop | Complete AgentBench `webshop-std` registry, ordered indices 0–199 | 200 |
| StrategyQA | Every ordered example from the pinned `test` split | 687 |

HotpotQA is the only benchmark for which the study uses a size-limited subset:
it is a deterministic prefix, not a random sample. StrategyQA uses its complete
test split. DBBench and WebShop are service-backed AgentBench benchmarks rather
than Hugging Face datasets with conventional split names, so the study uses
every task exposed by their standard registries. The 2,700 DBBench
task–backbone–seed evaluations in the preceding table therefore mean
`300 tasks × 3 backbones × 3 seeds`, not 300 tasks selected from a pool of
2,700. Exact revisions, selected-row hashes, and expected registry sizes are
frozen in `config/bayestraj_raw_generation_configuration.json`.

### Task success by dataset and backbone

Success is the complement of the frozen task-failure label used for final
AUROC/AUPR evaluation. The three seed columns report the percentage of
successful evaluated tasks in that dataset-backbone-seed cell. The final column
is the arithmetic mean and sample standard deviation across seeds; it is not a
trajectory-level success rate and does not use any uncertainty method's
predictions.

| Dataset | Backbone | Evaluated tasks/cell | Seed 101 | Seed 202 | Seed 303 | Mean ± seed SD |
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

Exact success counts and unrounded percentages are in
`data/paper_inputs/dataset_backbone_success_rates.csv`. They are derived once
per task from the 36 submitted held-out task-score cells, after deduplicating
the repeated method and budget rows by `sample_id`.

### Recorded steps per trajectory

The following statistics pool all 16 trajectories for every evaluated task
and all three seeds within each dataset–backbone combination. A step is one
stored trajectory step (`len(tdp.steps)`). The standard deviation is the
sample SD across individual trajectories—not the SD of the three seed means.

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

Exact values, per-seed means, medians, and ranges are in
`data/paper_inputs/trajectory_step_statistics.csv`. These values come from
the immutable submitted `Z=16` pools; forced or terminal steps are counted if
they are represented as a stored step in the trajectory.


### Complete baseline settings

All baselines use ordered prefixes of the same task-level trajectory pools.
Settings not listed as tunable are the fixed definitions implemented in the
cited files; no baseline hyperparameter was selected from correctness labels.

| Baseline(s) | Submitted setting |
|---|---|
| PPL, PE, LS | Direct stored-likelihood or lexical definitions; mean aggregation across the first `B` trajectories; no tuned parameter |
| SE | Semantic-equivalence threshold `0.85`; mean aggregation |
| SentSAR | Temperature `1e-3`; mean aggregation |
| SD | Probability-weighted semantic density; no tuned parameter |
| SNNE | ROUGE-L similarity and temperature `1.0` |
| KLE | `microsoft/deberta-v2-xlarge-mnli`; unweighted NLI graph, bidirectional decision-sum threshold `1.5`, heat time `0.4`, unnormalized graph Laplacian, normalized von Neumann entropy |
| EigV | Symmetrized NLI entailment affinity and normalized graph Laplacian; no tuned parameter |
| CoCoA-MaxProb, CoCoA-PPL | Graded NLI similarity `p(entail)+0.5 p(neutral)` with sequence-probability or mean-token-probability confidence, respectively |
| MC-OE | Natural-log empirical entropy of the same dataset-specific outcome buckets as BayesTraj |
| BSE-Fixed, BSE-Adaptive | Dirichlet concentration `alpha=0.5`, 1,000 Monte Carlo samples, 5 integration replicates, and 5-fold within-cell fitting |
| SAUP | Generic-distance form with `alpha=beta=gamma=1`; `deepset/roberta-base-squad2` revision `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4`, mean-pooled last hidden state, maximum length 512 |
| UProp | `N=4` candidates per eligible step, Gaussian-kernel `tau=1`, denominator epsilon `1e-6`, ratio cap `10`, no intrinsic cap, and no intrinsic transform |
| Degree | `N=4` candidates per eligible step |

The executable definitions are in `raw_pipeline/src/ltuq/baselines/`,
`raw_pipeline/uq_baselines.py`, and the four evaluators in Track B §3.11.

### Budget-alignment protocol

Budget alignment is defined as follows:

1. A nominal budget `B` always refers to the ordered prefix of a task's frozen
   trajectory pool. Reordering or selecting favorable trajectories is
   prohibited.
2. **BayesTraj-Fixed** uses exactly `T=B` and reports `S_B`.
3. **BayesTraj-Adaptive** searches only
   `T in {max(2,B-4),...,B-1}`. Within each held-out fold, its variance
   threshold is calibrated using the other four folds so their mean stopping
   cost is closest to `0.80B`. It stops at the first certified prefix and uses
   `T=B` if none qualifies. It never exceeds the nominal budget.
4. Fixed-budget baselines use their first `B` trajectories. BSE-Adaptive uses
   its own cross-fitted adaptive rule. Every adaptive curve is plotted at its
   **realized mean trajectory count**, not at nominal `B`.
5. UProp and Degree use the separately audited `Z=12,N=4` campaign. Their
   task IDs and first-12 trajectory identities must match the `Z=16` pool;
   their reported points therefore extend through `B=12`. Other fixed
   baselines use the ordered `Z=16` pool.
6. Baseline-only `B=2` points may be displayed for context but are excluded
   from all quantitative comparisons. The paper's shared comparison budgets
   are `{3,4,6,8,12,16}`; a method contributes only where its required
   artifacts exist.
7. AUROC/AUPR are computed within each dataset-backbone-seed cell. Curves
   macro-average dataset-backbone cells within each seed and then report the
   mean and sample standard deviation across the three seeds. Paired claims
   use hierarchical resampling of dataset-backbone combinations and seeds,
   with task resampling where specified.

Correctness labels are opened only after posterior fitting, threshold
calibration, scores, and stopping decisions are frozen. The complete
implementation of this alignment is in
`raw_pipeline/scripts/evaluate_brlg_fixed_varstop80_ablation.py`,
`raw_pipeline/scripts/evaluate_mixed_budget_core_multiseed.py`, and
`raw_pipeline/scripts/build_paper_curve_table.py`.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
python scripts/reproduce.py
python scripts/verify.py
pytest -q
```

Generated figures and `results.md` are written to `outputs/`. The reproduction
command makes Figures 1--9 from the paper:

| Figure | Output |
|---:|---|
| 1 | `fig01_method_overview.pdf` |
| 2 | `fig02_macro_cost_performance.pdf` |
| 3 | `fig03_paired_improvements.pdf` |
| 4 | `fig04_dataset_backbone_auroc.pdf` |
| 5 | `fig05_adaptive_efficiency.pdf` |
| 6 | `fig06_lower_cost_gain.pdf` |
| 7 | `fig07_core_ablation.pdf` |
| 8 | `fig08_sensitivity.pdf` |
| 9 | `fig09_posterior_validation.pdf` |

## What “reproduce” means

The default path starts from frozen per-seed or paired summary tables. It is
fast, deterministic, CPU-only, and reproduces the paper-facing aggregation,
plots, and claims without rerunning LLM agents. This is the appropriate path
for checking the submitted paper.

The upstream raw experiment used ordered pools of 16 cached trajectories,
five-fold within-cell cross-fitting, four datasets, three backbones, and seeds
101/202/303. Raw caches are not required to audit the paper-facing arithmetic.
Their locations and the SHA-256 manifest are recorded in `PROVENANCE.md` and
`MANIFEST.sha256`.

## Repository layout

```text
config/paper.json          frozen protocol and method registry
config/backbones.json      pinned model IDs, revisions, served names
config/bayestraj_raw_*     pinned raw-generation datasets and budgets
data/paper_inputs/         compact tables used by the submitted figures
expected/claims.json       machine-checkable manuscript claims
src/bayestraj/core.py      posterior, LCB score, threshold calibration
src/bayestraj/paper.py     aggregation and claim computation
src/bayestraj/figures.py   all paper figures
scripts/reproduce.py       one-command reproduction
scripts/verify.py          numerical and integrity audit
scripts/audit_raw_generation.py  complete raw-pool validator
scripts/merge_jsonl_parts.py     deterministic shard merger
tests/                     unit and end-to-end smoke tests
```

## Hardware and runtime

Plot reproduction is CPU-only and normally completes in under one minute.
Track B includes the code and commands for generation from scratch. The model
weights, benchmark assets, and generated trajectory caches remain external to
this repository and require substantial GPU time and storage.

---

# Detailed step-by-step reproduction guide

There are two deliberately separate reproduction tracks. **Track A** checks
the submitted results from compact frozen tables and is what most reviewers
should run. **Track B** starts from an empty external work directory: it
downloads the pinned public data, serves the three pinned backbones, generates
all raw trajectories, audits the resulting pools, and recomputes the scores,
ablations, sensitivity results, and reports. Raw data and model weights are
not distributed in this repository.

## 1. Obtain and inspect the repository

```bash
git clone <anonymous-repository-url> BayesTraj-08EB
cd BayesTraj-08EB
du -sh .
find . -type f | sort
```

The checked-in repository must remain below 20 MB. Generated files under
`outputs/` are ignored by Git. The release contains:

- all Python source used by the submitted BayesTraj evaluation pipeline;
- the transitive LTUQ Python modules imported by those scripts;
- paper configuration and protocol documents;
- compact per-seed/result tables needed for exact paper verification;
- no benchmark examples, raw trajectories, model checkpoints, or backbone
  weights.

The Python source is split into two areas:

- `src/bayestraj/` is the clean, documented release API and fast paper
  reproduction path;
- `raw_pipeline/` preserves the original raw-cache evaluators and their local
  Python dependency closure. It is deliberately restricted to BayesTraj and
  the 17 baselines reported in the submission.

The authoritative method registry is `config/paper.json`. In particular, this
release does **not** contain post-submission methods, exploratory stopping
rules, outcome-judge variants, or baselines absent from the submitted paper.

## 2. Track A: reproduce the submitted figures and claims

### 2.1 Create a clean environment

The fast path works with Python 3.9 or newer. The versions used for the release
audit are pinned in `requirements.txt`.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2.2 Verify the frozen input tables before plotting

```bash
python scripts/verify.py
```

Expected terminal output begins with:

```text
{
  "status": "AUDIT PASSED",
  "files_checked": ...,
  "claims_checked": 48,
  "repository_bytes": ...
}
```

This command performs four independent checks:

1. SHA-256 verification of source, configuration, and compact input tables;
2. dataset/backbone/seed registry checks;
3. recomputation of 48 numerical claims against `expected/claims.json`;
4. enforcement of the 20,000,000-byte repository limit.

### 2.3 Reproduce every submission figure

```bash
python scripts/reproduce.py
```

The command creates PDF and PNG versions of Figures 1--9 under `outputs/`, plus
two machine-readable summaries:

```text
outputs/claims.json
outputs/results.md
outputs/fig01_method_overview.{pdf,png}
outputs/fig02_macro_cost_performance.{pdf,png}
outputs/fig03_paired_improvements.{pdf,png}
outputs/fig04_dataset_backbone_auroc.{pdf,png}
outputs/fig05_adaptive_efficiency.{pdf,png}
outputs/fig06_lower_cost_gain.{pdf,png}
outputs/fig07_core_ablation.{pdf,png}
outputs/fig08_sensitivity.{pdf,png}
outputs/fig09_posterior_validation.{pdf,png}
```

Figure-to-input lineage is explicit:

| Figure | Input table(s) | Aggregation |
|---:|---|---|
| 1 | none | deterministic method diagram |
| 2 | `selected_seed_metrics.csv`, `baseline_b2_seed_metrics.csv` | dataset/backbone macro within seed, then mean ± seed SD |
| 3 | `paired_superiority_summary.csv` | paired hierarchical 95% intervals over six budgets |
| 4 | same per-seed curve tables as Fig. 2 | mean ± seed SD in each dataset/backbone cell |
| 5 | `efficiency_by_budget.csv` | adaptive minus fixed; realized trajectory/step/token saving |
| 6 | `representative_budget_metrics.csv` | B=8 adaptive minus each representative baseline |
| 7 | `core_mechanism_tradeoff.csv` | full adaptive method minus each mechanism ablation |
| 8 | `sensitivity_summary.csv`, `paired_cell_summary.csv` | hierarchical cost-ratio/window comparison |
| 9 | posterior diagnostic and calibration CSVs | held-out target MSE and risk calibration |

### 2.4 Run the tests

```bash
pytest -q
```

The tests cover the 257-point entropy grid, Jeffreys-smoothed count prior,
linear-Gaussian fit, grid fusion, LCB95 score, first-certificate stopping,
threshold calibration, all headline claims, and an end-to-end nine-figure
smoke test.

### 2.5 Check the central manuscript numbers manually

```bash
cat outputs/results.md
python -m json.tool outputs/claims.json
```

The expected highlights are: Fixed ranks first in 12/12 shared budget–metric
comparisons; average gains are 4.33 AUROC and 2.74 AUPR points; Adaptive saves
19.9% of trajectories with mean losses of about 0.65 points; and the two-view
posterior reduces held-out target MSE by 4.9%.

## 3. Track B: reproduce from scratch, including raw generation

Track B begins without trajectory caches. It reconstructs the entire empirical
path: acquire benchmark inputs, serve the exact model revisions, generate the
ordered pools of 16 trajectories, validate and freeze those pools, evaluate
all methods, and rebuild the reports. The checked-in configuration fixes every
dataset, model revision, seed, sample count, and generation budget. The
preserved source uses Python 3.13 type annotations, so use Python 3.13 here.

This is a large experiment: it generates 19,683 task–backbone–seed evaluation
records and 314,928 full trajectories before per-step StrategyQA samples. Use
approximately 1 TB of external scratch space. Eight 80 GB GPUs are
recommended; process one backbone
at a time with one vLLM replica per GPU. Generated data must go outside the Git
checkout.

### 3.1 Install the generation and evaluation environment

```bash
python3.13 -m venv .venv-raw
. .venv-raw/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-raw.txt
export PYTHONPATH="$PWD/raw_pipeline:$PWD/raw_pipeline/scripts"
```

Install Podman or Docker, an NVIDIA container runtime compatible with the host
driver, and eight visible GPUs. `nvidia-smi` and `podman info` should succeed.
The requirements file covers the Python generation/evaluation path; the two
AgentBench services are installed separately in §3.4.

Create external locations and record the repository revision before running:

```bash
export BAYESTRAJ_REPO="$PWD"
export BAYESTRAJ_RAW_ROOT=/external/scratch/bayestraj-raw
export BAYESTRAJ_EVAL_ROOT=/external/scratch/bayestraj-evaluations
export BAYESTRAJ_WORK=/external/scratch/bayestraj-analysis
export HF_HOME=/external/scratch/huggingface
mkdir -p "$BAYESTRAJ_RAW_ROOT" "$BAYESTRAJ_EVAL_ROOT" \
  "$BAYESTRAJ_WORK" "$HF_HOME"
git rev-parse HEAD | tee "$BAYESTRAJ_WORK/repository_commit.txt"
sha256sum config/bayestraj_raw_generation_*.json config/backbones.json \
  | tee "$BAYESTRAJ_WORK/frozen_configuration.sha256"
```

Do not set `LTUQ_REQUIRE_FROZEN_TDP_CACHE`, do not seed a sampling cache from
another run, and do not reuse a prior JSONL file. Those actions would turn this
into cache extension rather than from-scratch reproduction.

### 3.2 Acquire the pinned datasets and benchmark environments

StrategyQA and HotpotQA are downloaded automatically from Hugging Face using
the immutable revisions in
`config/bayestraj_raw_generation_configuration.json`. The exact contract is:

| Dataset | Split and selection | Tasks | Per-step samples |
|---|---|---:|---:|
| DBBench | AgentBench `dbbench-std`, indices 0–299 | 300 | 1 |
| HotpotQA | `distractor/validation`, deterministic prefix | 1,000 | 1 |
| WebShop | AgentBench `webshop-std`, indices 0–199 | 200 | 1 |
| StrategyQA | test split, all ordered rows | 687 | 4 |

The configuration also contains hashes for the selected HotpotQA and
StrategyQA IDs and rows. The launcher refuses to proceed when those hashes do
not match.

DBBench and WebShop require a separate AgentBench checkout because their
databases, product catalog, images, and services exceed the anonymous-release
size limit:

```bash
git clone https://github.com/THUDM/AgentBench.git "$BAYESTRAJ_WORK/AgentBench"
cd "$BAYESTRAJ_WORK/AgentBench"
# Follow AgentBench's own installation instructions for dbbench-std and
# webshop-std, including its Redis/controller and task-server containers.
cd "$BAYESTRAJ_REPO"
export AGENTBENCH_CONTROLLER_URL=http://127.0.0.1:5000/api
curl -fsS "$AGENTBENCH_CONTROLLER_URL/list_workers" >/dev/null
```

Use the benchmark assets provided by AgentBench; do not copy them into this
repository. Confirm through the controller that `dbbench-std` exposes exactly
300 ordered tasks and `webshop-std` exactly 200 before generation. Because the
upstream AgentBench repository does not expose all service data as a small
Python dependency, its checkout and service images are part of the external
experimental environment, not this release.

### 3.3 Download and serve the three pinned backbones

`config/backbones.json` is authoritative. It records these exact revisions:

| Key | Hugging Face model | Revision | Served name |
|---|---|---|---|
| `qwen35` | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `qwen3.5:9b` |
| `gemma3` | `google/gemma-3-12b-it` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` | `gemma3:12b` |
| `gptoss20b` | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | `gpt-oss:20b` |

Accept any gated-model licenses first and authenticate with Hugging Face. For
each backbone in turn, launch eight independent vLLM replicas on ports
8000–8007. The following template uses the recorded vLLM image digest; replace
`BACKBONE`, `MODEL`, `REVISION`, and `SERVED` from the table:

```bash
export VLLM_IMAGE="docker.io/vllm/vllm-openai@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b"
export MODEL=Qwen/Qwen3.5-9B
export REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export SERVED=qwen3.5:9b

for gpu in 0 1 2 3 4 5 6 7; do
  port=$((8000 + gpu))
  podman run -d --rm --name "bayestraj-vllm-${gpu}" --network host \
    --device nvidia.com/gpu=all --security-opt=label=disable \
    -e CUDA_VISIBLE_DEVICES="$gpu" -e HF_HOME=/root/.cache/huggingface \
    -v "$HF_HOME:/root/.cache/huggingface" "$VLLM_IMAGE" \
    --model "$MODEL" --revision "$REVISION" --served-model-name "$SERVED" \
    --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.80 --max-num-seqs 64 --enforce-eager \
    --host 0.0.0.0 --port "$port"
done

for port in 8000 8001 8002 8003 8004 8005 8006 8007; do
  until curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; do sleep 10; done
done
```

Run §§3.4–3.5 for all three seeds before stopping these containers and
repeating with the next backbone. WebShop can require contexts longer than
8,192 tokens; restart the same model replicas with `--max-model-len 131072` for §3.5 if the
model/hardware configuration supports it. Record vLLM logs and `/v1/models`
responses beside the raw artifacts.

### 3.4 Generate DBBench, HotpotQA, and StrategyQA raw pools

The launcher validates the frozen manifest, endpoint model identity, pinned
dataset hashes, and output structure. It partitions work across ports and can
resume interrupted *shards*, but a clean reproduction starts with empty output
directories. Set the current backbone fields each time the model servers are
changed:

```bash
export BACKBONE=qwen35
export SERVED=qwen3.5:9b
export PORTS=8000,8001,8002,8003,8004,8005,8006,8007

for seed in 101 202 303; do
  python raw_pipeline/scripts/generate_raw_trajectory_pools.py \
    --run-id "bayestraj_dbbench_seed${seed}_z16_${BACKBONE}" \
    --model "$SERVED" --backbone "$BACKBONE" --ports "$PORTS" \
    --datasets dbbench --seeds "$seed" --dbbench-tasks 300 \
    --trajectory-samples 16 --per-step-samples 1 \
    --output-root "$BAYESTRAJ_RAW_ROOT/dbbench_seed${seed}_z16" \
    --log-root "$BAYESTRAJ_RAW_ROOT/dbbench_seed${seed}_z16/logs" \
    --state "$BAYESTRAJ_RAW_ROOT/dbbench_seed${seed}_z16/state_${BACKBONE}.json"

  python raw_pipeline/scripts/generate_raw_trajectory_pools.py \
    --run-id "bayestraj_hotpotqa_seed${seed}_z16_${BACKBONE}" \
    --model "$SERVED" --backbone "$BACKBONE" --ports "$PORTS" \
    --datasets hotpotqa --seeds "$seed" --hotpot-tasks 1000 --hotpot-chunk 5 \
    --trajectory-samples 16 --per-step-samples 1 \
    --output-root "$BAYESTRAJ_RAW_ROOT/hotpotqa_seed${seed}_z16" \
    --log-root "$BAYESTRAJ_RAW_ROOT/hotpotqa_seed${seed}_z16/logs" \
    --state "$BAYESTRAJ_RAW_ROOT/hotpotqa_seed${seed}_z16/state_${BACKBONE}.json"

  python raw_pipeline/scripts/generate_raw_trajectory_pools.py \
    --run-id "bayestraj_strategyqa_seed${seed}_z16_n4_${BACKBONE}" \
    --model "$SERVED" --backbone "$BACKBONE" --ports "$PORTS" \
    --datasets strategyqa --seeds "$seed" --strategyqa-tasks 687 \
    --strategyqa-chunk 5 --trajectory-samples 16 --per-step-samples 4 \
    --output-root "$BAYESTRAJ_RAW_ROOT/strategyqa_seed${seed}_z16_n4" \
    --log-root "$BAYESTRAJ_RAW_ROOT/strategyqa_seed${seed}_z16_n4/logs" \
    --state "$BAYESTRAJ_RAW_ROOT/strategyqa_seed${seed}_z16_n4/state_${BACKBONE}.json"
done
```

The runner writes one ordered merged JSONL per cell under
`<dataset>_<backbone>/`. Keep all part files, logs, sampling records, state
files, and merged files: together they are the raw artifact trail. Correctness
labels may be stored for later AUROC/AUPR evaluation, but are never supplied to
the posterior fit, threshold calibration, outcome mapping, or stop decision.

### 3.5 Generate WebShop raw pools

WebShop is interactive and is launched directly. The example below assigns 25
disjoint tasks to each of eight endpoints, preserves every raw shard, and then
creates the canonical task-index-ordered pool. Run it for every seed and the
currently served backbone:

```bash
export BACKBONE=qwen35
export SERVED=qwen3.5:9b
export MAX_TOKENS=2048            # use 1024 for gemma3

for seed in 101 202 303; do
  cell_root="$BAYESTRAJ_RAW_ROOT/webshop_seed${seed}_z16/webshop_${BACKBONE}"
  mkdir -p "$cell_root/parts" "$cell_root/sampling"
  pids=""
  for shard in 0 1 2 3 4 5 6 7; do
    offset=$((25 * shard)); port=$((8000 + shard))
    python raw_pipeline/main.py run-agentbench-webshop \
      --provider vllm --model "$SERVED" --base-url "http://127.0.0.1:${port}/v1" \
      --api-key vllm --method uprop --controller-url "$AGENTBENCH_CONTROLLER_URL" \
      --task-name webshop-std --offset "$offset" --limit 25 \
      --tdp-samples 16 --per-step-samples 1 --backbone-samples 1 \
      --next-step-samples 1 --temperature 1.0 --seed "$seed" --max-steps 20 \
      --max-tokens "$MAX_TOKENS" --parallel-requests 1 \
      --no-fair-trajectory-budget --emulate-tool-calls --disable-tracking \
      --sampling-dir "$cell_root/sampling" --restart \
      --output "$cell_root/parts/offset${offset}_limit25.jsonl" \
      >"$cell_root/parts/offset${offset}.log" 2>&1 &
    pids="$pids $!"
  done
  for pid in $pids; do wait "$pid"; done

  python scripts/merge_jsonl_parts.py \
    --expected-rows 200 \
    --output "$cell_root/bayestraj_webshop_seed${seed}_z16_${BACKBONE}_seed${seed}_pe.jsonl" \
    "$cell_root"/parts/offset*_limit25.jsonl
done
```

Although the historical filename ends in `_pe.jsonl`, each row contains the
16 UProp trajectory-distribution paths used to construct BayesTraj. The suffix
is retained because downstream manuscript scripts use this frozen convention.
For Gemma set `MAX_TOKENS=1024`; Qwen and GPT-OSS use 2048.

#### 3.5.1 Generate the supplemental exact-Z=12/N=4 baseline campaign

The paper's UProp and Degree endpoints at budget 12 require four
candidate continuations per nonterminal step. They cannot be reconstructed
from an N=1 pool, so generate a separate raw campaign rather than modifying the
BayesTraj Z=16 pools:

```bash
export BAYESTRAJ_N4_ROOT=/external/scratch/bayestraj-z12-n4
mkdir -p "$BAYESTRAJ_N4_ROOT"

# With BACKBONE, SERVED, and PORTS set as in §3.4, repeat for each served model.
for seed in 101 202 303; do
  for dataset in dbbench hotpotqa strategyqa; do
    case "$dataset" in
      dbbench)    tasks=300;  extra="--dbbench-tasks 300" ;;
      hotpotqa)   tasks=1000; extra="--hotpot-tasks 1000 --hotpot-chunk 5" ;;
      strategyqa) tasks=687;  extra="--strategyqa-tasks 687 --strategyqa-chunk 5" ;;
    esac
    root="$BAYESTRAJ_N4_ROOT/${dataset}_seed${seed}_z12_n4"
    # shellcheck disable=SC2086 -- $extra is a fixed option list above.
    python raw_pipeline/scripts/generate_raw_trajectory_pools.py \
      --run-id "bayestraj_${dataset}_seed${seed}_z12_n4_${BACKBONE}" \
      --model "$SERVED" --backbone "$BACKBONE" --ports "$PORTS" \
      --datasets "$dataset" --seeds "$seed" $extra \
      --trajectory-samples 12 --per-step-samples 4 \
      --output-root "$root" --log-root "$root/logs" \
      --state "$root/state_${BACKBONE}.json"
  done
done
```

For WebShop, repeat the eight-shard command in §3.5 with
`--tdp-samples 12 --per-step-samples 4`, write beneath
`$BAYESTRAJ_N4_ROOT/webshop_seed${seed}_z12_n4/webshop_${BACKBONE}`, and merge
to this exact filename:

```text
bayestraj_webshop_seed<seed>_z12_n4_<backbone>_seed<seed>_pe.jsonl
```

Then build the strict manifest consumed by the curve code:

```bash
python scripts/build_z12_n4_manifest.py --campaign-root "$BAYESTRAJ_N4_ROOT"
```

This campaign is intentionally separate: no raw Z=16 record is overwritten,
and its manifest identifies that the Z=12/N=4 trajectories were generated
from scratch.

### 3.6 Audit and freeze all generated artifacts

After all three backbones and seeds finish, run the global structural audit:

```bash
python scripts/audit_raw_generation.py \
  --pool-root "$BAYESTRAJ_RAW_ROOT" \
  --output "$BAYESTRAJ_RAW_ROOT/raw_generation_audit.json"
```

It requires 36/36 cells, 19,683 ordered task rows, unique task/sample IDs, and
exactly 16 trajectories in every task record, and records a SHA-256 digest for
every canonical JSONL. Preserve this audit, all logs, the three model service
receipts, and the configuration hashes. Make the raw root read-only or take an
immutable filesystem snapshot before evaluation. Every reported budget must
use an ordered prefix of this same frozen 16-trajectory pool.

### 3.7 Produce the 36 BayesTraj evaluation checkpoints

This CPU-only transformation creates the count-prior moments, ten trajectory
features, label-free `OE16` targets, and deterministic five-fold assignments
used downstream. It does not evaluate any other method:

```bash
python raw_pipeline/scripts/materialize_bayestraj_checkpoints.py \
  --pool-root "$BAYESTRAJ_RAW_ROOT" \
  --output-dir "$BAYESTRAJ_EVAL_ROOT" \
  --datasets dbbench hotpotqa webshop strategyqa \
  --backbones qwen35 gemma3 gptoss20b \
  --seeds 101 202 303 --posterior-draws 2048
```

Five-fold cross-fitting is performed independently inside each
dataset/backbone/seed cell using ordered task index modulo five. A task's own
label-free `OE16` target never fits the model that scores that task.

At this point the required inputs are:

- one evaluation checkpoint for each of the 36
  dataset/backbone/seed cells;
- the newly generated and audited ordered 16-trajectory pool for every task;
- the baseline scores generated in §3.12.

The 36 cells are the Cartesian product:

```text
dataset  = dbbench | hotpotqa | webshop | strategyqa
backbone = qwen35 | gemma3 | gptoss20b
seed     = 101 | 202 | 303
cell     = <dataset>-<backbone>-seed<seed>
```

### 3.8 Recompute the fixed/adaptive component study

Run one independent CPU job per cell. The example is sequential; a scheduler
may safely parallelize cells because every cell writes to its own directory.

```bash
export ABLATION_RUN="$BAYESTRAJ_WORK/brlg_fixed_varstop80_ablation_run"
for dataset in dbbench hotpotqa webshop strategyqa; do
  for backbone in qwen35 gemma3 gptoss20b; do
    for seed in 101 202 303; do
      cell="${dataset}-${backbone}-seed${seed}"
      python raw_pipeline/scripts/evaluate_brlg_fixed_varstop80_ablation.py \
        --cell "$cell" \
        --evaluation-root "$BAYESTRAJ_EVAL_ROOT" \
        --output-root "$ABLATION_RUN/cells" \
        --posterior-draws 2048 \
        --resume
    done
  done
done
```

Aggregate the 36 completed cells and reproduce Figure 7:

```bash
export ABLATION_REPORT="$BAYESTRAJ_WORK/brlg_fixed_varstop80_ablation"
python raw_pipeline/scripts/aggregate_brlg_fixed_varstop80_ablation.py \
  --run-root "$ABLATION_RUN" \
  --output-root "$ABLATION_REPORT" \
  --task-bootstrap-replicates 500 \
  --hierarchical-bootstrap-replicates 10000 \
  --workers 24
```

The relevant output is
`$ABLATION_REPORT/report/plots/core_mechanism_tradeoff.pdf`; its source values
are `$ABLATION_REPORT/core_mechanism_tradeoff.csv`.

### 3.9 Recompute adaptive sensitivity

```bash
export SENSITIVITY_RUN="$BAYESTRAJ_WORK/brlg_varstop_sensitivity_run"
for dataset in dbbench hotpotqa webshop strategyqa; do
  for backbone in qwen35 gemma3 gptoss20b; do
    for seed in 101 202 303; do
      cell="${dataset}-${backbone}-seed${seed}"
      python raw_pipeline/scripts/evaluate_brlg_varstop_sensitivity.py \
        --cell "$cell" \
        --evaluation-root "$BAYESTRAJ_EVAL_ROOT" \
        --output-root "$SENSITIVITY_RUN/cells" \
        --posterior-draws 2048 \
        --resume
    done
  done
done

export SENSITIVITY_REPORT="$BAYESTRAJ_WORK/brlg_varstop_sensitivity"
python raw_pipeline/scripts/aggregate_brlg_varstop_sensitivity.py \
  --run-root "$SENSITIVITY_RUN" \
  --output-root "$SENSITIVITY_REPORT" \
  --hierarchical-replicates 10000
```

This evaluates `rho ∈ {0.70,0.75,0.80,0.85,0.90}` and
`w ∈ {2,4,6,all}` while keeping every estimator component fixed. Figure 8 is
`$SENSITIVITY_REPORT/report/plots/rho_window_auroc_window_effect_one_column.pdf`.

### 3.10 Recompute the WebShop outcome mapping

Only WebShop uses the constraint-pattern hierarchy. Run its nine cells:

```bash
export WEBSHOP_RUN="$BAYESTRAJ_WORK/brlg_webshop_task_equivalent_buckets_run"
for backbone in qwen35 gemma3 gptoss20b; do
  for seed in 101 202 303; do
    cell="webshop-${backbone}-seed${seed}"
    python raw_pipeline/scripts/evaluate_bayestraj_webshop_constraint_pattern.py \
      --cell "$cell" \
      --evaluation-root "$BAYESTRAJ_EVAL_ROOT" \
      --z16-root "$BAYESTRAJ_Z16_ROOT" \
      --output-root "$WEBSHOP_RUN/cells" \
      --posterior-draws 2048 \
      --resume
  done
done
```

The mapping uses only request text, chosen options, purchase status, product
metadata, and explicit price constraints. It never reads correctness labels,
rewards, reference answers, or evaluator decisions while defining buckets.

### 3.11 Rebuild the complete baseline scores

The preserved baseline and curve code is located in:

```text
raw_pipeline/run_uq_baselines.py
raw_pipeline/uq_baselines.py
raw_pipeline/src/ltuq/baselines/
raw_pipeline/scripts/generate_selected_methods_four_dataset_report.py
```

All reported baselines are recomputed from the pools generated above. The
semantic stage downloads `microsoft/deberta-v2-xlarge-mnli` on
first use and is GPU-accelerated; it does not call any of the three agent
backbones or generate new responses.

```bash
export BASELINE_ROOT="$BAYESTRAJ_WORK/mixed_budget_sources"
mkdir -p "$BASELINE_ROOT"

python raw_pipeline/scripts/evaluate_mixed_budget_core_multiseed.py \
  --z16-root "$BAYESTRAJ_RAW_ROOT" \
  --strict-root "$BAYESTRAJ_N4_ROOT" \
  --output-dir "$BASELINE_ROOT/core"

export LTUQ_KLE_NLI_MODEL=microsoft/deberta-v2-xlarge-mnli
python raw_pipeline/scripts/evaluate_z16_semantic_baselines_multiseed.py \
  --pool-root "$BAYESTRAJ_RAW_ROOT" \
  --output-dir "$BASELINE_ROOT/semantic" \
  --device cuda:0 --batch-size 48

python raw_pipeline/scripts/evaluate_z16_saup_multiseed.py \
  --pool-root "$BAYESTRAJ_RAW_ROOT" \
  --output-dir "$BASELINE_ROOT/saup" \
  --device cuda:0 --batch-size 128

python raw_pipeline/scripts/evaluate_bse_multiseed.py \
  --pool-root "$BAYESTRAJ_RAW_ROOT" \
  --output-dir "$BASELINE_ROOT/bse" \
  --monte-carlo-samples 1000 --integration-replicates 5
```

The core step computes PPL, LS, PE, SE, SD, SentSAR, MC-OE, UProp, and Degree.
The semantic step computes SNNE, KLE, EigV, CoCoA-MaxProb, and CoCoA-PPL; the
third computes SAUP. The separate Bayesian-semantic-entropy evaluator computes
BSE-Fixed and BSE-Adaptive. Together these are exactly the 17 paper baselines.
They write per-task
scores, seed-level metrics, and provenance without new agent generation.

Merge the two BayesTraj variants and those baseline scores into the sole
paper-facing curve registry:

```bash
python raw_pipeline/scripts/build_paper_curve_table.py \
  --ablation-root "$ABLATION_RUN" \
  --core-root "$BASELINE_ROOT/core" \
  --semantic-root "$BASELINE_ROOT/semantic" \
  --saup-root "$BASELINE_ROOT/saup" \
  --bse-root "$BASELINE_ROOT/bse" \
  --output "$BAYESTRAJ_WORK/mixed_budget_complete/report"
```

### 3.12 Rebuild the selected four-dataset report

The original report builder expects its related WebShop campaign at the
standard relative location under `raw_pipeline/outputs/analysis`. Link or copy
only the small generated campaign outputs there (never the raw trajectories):

```bash
mkdir -p raw_pipeline/outputs/analysis
ln -sfn "$WEBSHOP_RUN" \
  raw_pipeline/outputs/analysis/brlg_webshop_task_equivalent_buckets_run

python raw_pipeline/scripts/generate_selected_methods_four_dataset_report.py \
  --source "$BAYESTRAJ_WORK/mixed_budget_complete/report" \
  --output "$BAYESTRAJ_WORK/mixed_budget_4datasets_selected_methods/report"
```

This step filters to the four paper datasets and 17 baselines, replaces only
the two BayesTraj WebShop rows with the constraint-pattern mapping, and computes
seed-macro and dataset/backbone tables.

### 3.13 Recompute paired inference, efficiency, and posterior validation

```bash
python raw_pipeline/scripts/generate_bayestraj_empirical_minimal_package.py \
  --selected-report "$BAYESTRAJ_WORK/mixed_budget_4datasets_selected_methods/report" \
  --ablation-run-root "$ABLATION_RUN" \
  --raw-root "$BAYESTRAJ_Z16_ROOT" \
  --ablation-scores "$ABLATION_REPORT/task_scores.jsonl.gz" \
  --webshop-root "$WEBSHOP_RUN" \
  --output "$BAYESTRAJ_WORK/bayestraj_empirical/report"
```

This stage regenerates Figure 3, Figure 5, and Figure 9 using 10,000 paired
hierarchical bootstrap replicates and 2,000 calibration replicates. It records
that no generation or GPU call occurred.

Rebuild the B=8 comparison used for Figure 6:

```bash
python raw_pipeline/scripts/generate_bayestraj_compute_cost_candidates.py \
  --metrics "$BAYESTRAJ_WORK/mixed_budget_4datasets_selected_methods/report/selected_seed_metrics.csv" \
  --efficiency "$BAYESTRAJ_WORK/bayestraj_empirical/report/efficiency_by_budget.csv" \
  --output "$BAYESTRAJ_WORK/bayestraj_compute_cost" \
  --budget 8
```

### 3.14 Compare raw recomputation with the submission package

Compare the regenerated source CSVs against `data/paper_inputs/`. Numeric CSV
values should agree to floating-point precision; PDFs may have different byte
hashes across Matplotlib/font versions even when values are identical.

At minimum, compare:

```bash
python - <<'PY'
import pandas as pd

frozen = pd.read_csv("data/paper_inputs/core_mechanism_tradeoff.csv")
fresh = pd.read_csv("raw_work/brlg_fixed_varstop80_ablation/core_mechanism_tradeoff.csv")
pd.testing.assert_frame_equal(frozen, fresh, check_exact=False, atol=1e-10, rtol=1e-10)
print("ablation values match")
PY
```

Then rerun Track A. It remains the authoritative check that all paper-facing
aggregations and rounded claims agree with the submitted manuscript.

## 4. Reproducibility cautions

- Never train on a held-out task's own `OE_16` target.
- Never use correctness labels to fit posterior parameters, define WebShop
  buckets, calibrate adaptive thresholds, or choose individual stopping times.
- Preserve trajectory order; budget `B` always means the first `B` cached
  executions.
- Do not compare adaptive nominal budget with fixed realized cost without
  reporting the actual mean trajectory count.
- Use sample SD across the three seed macros for curve ribbons, and paired
  hierarchical confidence intervals for inferential comparisons.
- Do not treat the five folds as five independent experimental seeds.
- The within-cell cross-fitting protocol does not claim cross-dataset
  generalization.
- Generated PDF byte hashes can vary with fonts and plotting libraries. Verify
  source CSVs and `outputs/claims.json`, not PDF hashes.

## 5. Cleaning and rerunning

```bash
make clean
make reproduce
make verify
make test
```

`make clean` deletes generated files only inside this repository's `outputs/`
directory. It does not touch external datasets, model weights, or raw caches.
