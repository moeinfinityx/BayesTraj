# Raw generation and artifact audit

This guide covers the GPU-dependent portion of the from-scratch reproduction.
The top-level entry point is always `scripts/reproduce.py`; the individual
programs it invokes are documented in
[REPRODUCTION_PROTOCOL.md](REPRODUCTION_PROTOCOL.md) for inspection and
troubleshooting.

## 1. Environment

Use Python 3.13 and keep all generated data outside the repository:

```bash
python3.13 -m venv .venv-raw
. .venv-raw/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
export PYTHONPATH="$PWD/raw_pipeline:$PWD/raw_pipeline/scripts"

export BAYESTRAJ_RAW_ROOT=/external/scratch/bayestraj-raw
export BAYESTRAJ_N4_ROOT=/external/scratch/bayestraj-z12-n4
export BAYESTRAJ_EVAL_ROOT=/external/scratch/bayestraj-evaluations
export BAYESTRAJ_WORK=/external/scratch/bayestraj-analysis
export HF_HOME=/external/scratch/huggingface

python scripts/reproduce.py raw doctor
```

The driver reads the pinned datasets and generation contract from
`config/bayestraj_raw_generation_configuration.json` and the exact backbone
revisions from `config/backbones.json`.

## 2. External benchmark services

StrategyQA and HotpotQA are downloaded from their pinned Hugging Face
revisions. DBBench and WebShop require the upstream AgentBench checkout and
its controller, Redis instance, task servers, database, and product assets.
Set:

```bash
export AGENTBENCH_CONTROLLER_URL=http://127.0.0.1:5000/api
```

Before generation, confirm that AgentBench exposes exactly 300 ordered
`dbbench-std` tasks and 200 ordered `webshop-std` tasks.

| Dataset | Frozen selection | Tasks |
|---|---|---:|
| DBBench | AgentBench `dbbench-std`, indices 0–299 | 300 |
| HotpotQA | deterministic first 1,000 rows of `distractor/validation` | 1,000 |
| WebShop | AgentBench `webshop-std`, indices 0–199 | 200 |
| StrategyQA | complete ordered test split | 687 |

## 3. Serve each backbone

Serve one backbone at a time with one vLLM replica per GPU on ports
8000–8007. The submitted identifiers are:

| Key | Model | Served name |
|---|---|---|
| `qwen35` | `Qwen/Qwen3.5-9B` | `qwen3.5:9b` |
| `gemma3` | `google/gemma-3-12b-it` | `gemma3:12b` |
| `gptoss20b` | `openai/gpt-oss-20b` | `gpt-oss:20b` |

The immutable revisions, container digest, and full vLLM command are in
[the complete protocol](REPRODUCTION_PROTOCOL.md#33-download-and-serve-the-three-pinned-backbones).
Record the `/v1/models` response and service logs with the generated artifacts.

## 4. Generate both frozen campaigns

With Qwen served:

```bash
python scripts/reproduce.py raw generate \
  --backbone qwen35 --model qwen3.5:9b \
  --ports 8000,8001,8002,8003,8004,8005,8006,8007
```

After switching the endpoints to Gemma:

```bash
python scripts/reproduce.py raw generate \
  --backbone gemma3 --model gemma3:12b \
  --ports 8000,8001,8002,8003,8004,8005,8006,8007 \
  --max-tokens 1024
```

After switching to GPT-OSS:

```bash
python scripts/reproduce.py raw generate \
  --backbone gptoss20b --model gpt-oss:20b \
  --ports 8000,8001,8002,8003,8004,8005,8006,8007
```

Each invocation generates all four datasets and seeds for both campaigns:

- `Z=16`: the ordered BayesTraj and fixed-baseline trajectory pools;
- `Z=12,N=4`: a separate, provenance-distinct UProp/Degree campaign.

WebShop is split into eight disjoint 25-task shards. Other datasets are
partitioned by the frozen launcher. Part files, sampling records, logs, state
files, and merged JSONL files are all retained. The commands are resumable at
the shard level; do not seed them from another run or modify completed pools.

Use `--skip-z16` or `--skip-n4` only when the corresponding campaign is already
complete. Use `--dry-run` to inspect every generated command.

## 5. Required audit conditions

The subsequent `raw analyze` command begins by auditing both roots. The Z=16
audit requires:

- 36/36 dataset/backbone/seed cells;
- 19,683 ordered task records;
- exactly 16 trajectories per task;
- unique task and trajectory identities;
- pinned StrategyQA and HotpotQA row hashes.

The Z=12/N=4 manifest additionally verifies exact `Z=12`, exactly four
candidates at every eligible step, task-set equality with Z=16, and ordered
trajectory-prefix identity. No generated record in either campaign is changed
during evaluation.

