from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Callable, cast

from ..cli_progress import RunProgressDisplay
from ..baselines import OUTCOME_ENTROPY_BASELINE
from ..config import ExperimentConfig, TaskFamily
from ..estimators import UPropEstimator
from ..executors import (
    AgentBenchControllerClient,
    AgentBenchSample,
    AgentBenchSingleTrajectoryExecutor,
)
from ..executors.agentbench import AgentBenchControllerError
from ..executors.agentbench_dbbench import AgentBenchDBBenchExecutor
from ..experiments import ExperimentTracker
from ..logging_utils import configure_logging
from ..models import ChatModelConfig, ModelGenerationError, RecordedChatModelClient, close_chat_model, create_chat_model
from ..models.base import ModelProvider
from ..results_summary import resolve_record_uncertainty, summarize_records
from ..sampling import SharedSamplingStorage, reusable_model_signature
from ..trajectory import (
    MultiStepBaselineEstimate,
    TDPCounterfactualBranch,
    TDPCounterfactualRecord,
    TDPStepRecord,
    TrajectoryDependentDecisionProcess,
)
from .logprob_resume import (
    filter_stale_records,
    probe_structured_logprob_support,
    record_has_missing_logprob_state,
)


LOGGER = logging.getLogger(__name__)

_EMULATED_TOOL_CALL_DATASET_SUFFIX = "_emulated_tool_calls"
_EMULATED_TOOL_CALL_NAME_SUFFIX = "-emulated-tool-calls"
_DBBENCH_CANONICAL_OUTCOME_BUCKET_VERSION = "dbbench-canonical-outcome-v1"
_DBBENCH_SEMANTIC_BUCKET_CANONICALIZER_MAX_TOKENS = 240

_RESUME_IGNORED_CONFIG_KEYS = {
    "api_key",
    "experiment_name",
    "run_name",
    "notes",
    "tracking_dir",
    "track_experiment",
    "tags",
    "log_level",
    "log_path",
    "show_progress",
    "parallel_requests",
    "provider",
    "base_url",
    "azure_endpoint",
    "api_version",
    "deployment_name",
    "restart",
    "shared_sampling_dir",
}

_PROFILE_ENV_VAR = "LTUQ_PROFILE_AGENTBENCH_TIMINGS"
_DISABLE_HARD_FINALIZATION_ENV = "LTUQ_DISABLE_HARD_FINALIZATION"
_OUTCOME_ENTROPY_GENERATION_FAILURE_UNCERTAINTY = 1_000_000_000.0
_OUTCOME_ENTROPY_TASK_LIMIT_UNCERTAINTY = 1_000_000_000.0
_DEGREE_SHARED_CACHE_PER_STEP_SAMPLES = 4
_DBBENCH_UNANIMOUS_TASK_LIMIT_BUCKETS = frozenset(
    {
        "no-answer:task limit reached",
    }
)
_DBBENCH_NO_ANSWER = "NO_ANSWER"
_DBBENCH_AMBIGUOUS_ANSWER = "AMBIGUOUS"
_DBBENCH_STATE_CHANGING_SQL_RE = re.compile(
    r"^\s*(?:insert|update|delete|replace|create|drop|alter|truncate)\b",
    flags=re.IGNORECASE,
)
_DBBENCH_SQL_ERROR_RE = re.compile(
    r"\b(?:error|exception|syntax error|no such table|unknown column|failed|invalid)\b",
    flags=re.IGNORECASE,
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DBBENCH_TASK_DATA_PATH = _REPO_ROOT / "packages/AgentBench/data/dbbench/standard.jsonl"
_DBBENCH_TASK_CACHE: dict[int, Mapping[str, Any]] | None = None
_DBBENCH_POSTPROCESSING_METHODS = {"uprop"}
_DBBENCH_SHARED_TDP_CACHE_CATEGORY = "tdp_trajectories"
_DBBENCH_SHARED_TDP_CACHE_VERSION = 3
_DBBENCH_RUNNER_STATE_CANONICALIZATION_VERSION = 1


def _profile_enabled() -> bool:
    value = os.getenv(_PROFILE_ENV_VAR, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _profile_log(event: str, **payload: Any) -> None:
    if not _profile_enabled():
        return
    LOGGER.info("AGENTBENCH_PROFILE %s", json.dumps({"event": event, **payload}, sort_keys=True, ensure_ascii=True))


def _agentbench_task_family(base_family: TaskFamily, *, emulate_tool_calls: bool) -> TaskFamily:
    if emulate_tool_calls:
        return cast(TaskFamily, f"{base_family}{_EMULATED_TOOL_CALL_DATASET_SUFFIX}")
    return base_family


def _agentbench_variant_name_suffix(*, emulate_tool_calls: bool) -> str:
    if emulate_tool_calls:
        return _EMULATED_TOOL_CALL_NAME_SUFFIX
    return ""


@dataclass(frozen=True)
class AgentBenchRunnerConfig:
    method: str = "uprop"
    model: str = "gpt-5.4"
    provider: ModelProvider = "openai"
    emulate_tool_calls: bool = True
    controller_url: str = "http://localhost:5020/api"
    task_name: str = "dbbench-std"
    task_index: int | str | None = None
    limit: int | None = 1
    offset: int = 0
    output_path: str | None = None
    branches: int = 4
    window: int = 3
    adaptive: bool = False
    low_branches: int = 1
    low_window: int = 1
    high_branches: int = 4
    high_window: int = 3
    top_fraction: float = 0.2
    backbone_per_step_samples: int = 4
    next_step_entropy_samples: int = 4
    tdp_samples: int = 4
    per_step_samples: int = 4
    max_steps: int = 64
    temperature: float = 0.8
    max_tokens: int = 2048
    seed: int | None = None
    parallel_requests: int = 8
    tau: float = 1.0
    uprop_ratio_epsilon: float = 1e-6
    uprop_ratio_cap: float | None = 10.0
    uprop_intrinsic_cap: float | None = None
    uprop_intrinsic_transform: str = "none"
    fair_trajectory_budget: bool = True
    base_url: str | None = None
    azure_endpoint: str | None = None
    api_version: str | None = None
    deployment_name: str | None = None
    api_key: str | None = None
    experiment_name: str | None = None
    run_name: str | None = None
    notes: str = ""
    tracking_dir: str = "outputs/experiments"
    track_experiment: bool = True
    tags: tuple[str, ...] = ()
    shared_sampling_dir: str = "outputs/shared_sampling"
    log_level: str = "INFO"
    log_path: str | None = None
    show_progress: bool | None = None
    restart: bool = False
    strict_replay: bool = True

    def _extra_valid_methods(self) -> set[str]:
        """Return task-family-specific methods accepted by this runner config."""
        return set()

    def __post_init__(self) -> None:
        if self.method != "uprop":
            raise ValueError("the paper raw-generation runner supports only 'uprop'")
        if self.uprop_ratio_epsilon <= 0.0:
            raise ValueError("uprop_ratio_epsilon must be positive")
        if self.uprop_ratio_cap is not None and self.uprop_ratio_cap <= 0.0:
            raise ValueError("uprop_ratio_cap must be positive when provided")
        if self.uprop_intrinsic_cap is not None and self.uprop_intrinsic_cap <= 0.0:
            raise ValueError("uprop_intrinsic_cap must be positive when provided")
        if self.uprop_intrinsic_transform not in {"none", "log1p"}:
            raise ValueError("uprop_intrinsic_transform must be 'none' or 'log1p'")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        ChatModelConfig(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            emulate_tool_calls=self.emulate_tool_calls,
            max_tokens=self.max_tokens,
            parallel_requests=self.parallel_requests,
            base_url=self.base_url,
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            deployment_name=self.deployment_name,
        )


def _normalize_accuracy_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _majority_final_answer(estimates: Sequence[str | None]) -> str | None:
    filtered = [str(answer).strip() for answer in estimates if isinstance(answer, str) and answer.strip()]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def _majority_bool(values: Sequence[bool | None]) -> bool | None:
    filtered = [value for value in values if isinstance(value, bool)]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def _load_dbbench_task_cache() -> dict[int, Mapping[str, Any]]:
    global _DBBENCH_TASK_CACHE
    if _DBBENCH_TASK_CACHE is not None:
        return dict(_DBBENCH_TASK_CACHE)
    tasks: dict[int, Mapping[str, Any]] = {}
    if _DBBENCH_TASK_DATA_PATH.exists():
        for index, line in enumerate(_DBBENCH_TASK_DATA_PATH.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    tasks[index] = payload
    _DBBENCH_TASK_CACHE = tasks
    return dict(tasks)


def _dbbench_task_payload(task_index: int | str | None) -> Mapping[str, Any] | None:
    if isinstance(task_index, str) and task_index.isdigit():
        task_index = int(task_index)
    if not isinstance(task_index, int):
        return None
    return _load_dbbench_task_cache().get(task_index)


def _dbbench_reference_answers(task_index: int | str | None) -> list[str]:
    payload = _dbbench_task_payload(task_index)
    labels = payload.get("label") if isinstance(payload, Mapping) else None
    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes, bytearray)):
        return [str(label).strip() for label in labels if str(label).strip()]
    if isinstance(labels, str) and labels.strip():
        return [labels.strip()]
    return []


def _dbbench_sql_strings(task_index: int | str | None) -> list[str]:
    payload = _dbbench_task_payload(task_index)
    if not isinstance(payload, Mapping):
        return []
    candidates: list[str] = []
    labels = payload.get("label")
    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes, bytearray)):
        candidates.extend(str(label).strip() for label in labels if str(label).strip())
    elif isinstance(labels, str) and labels.strip():
        candidates.append(labels.strip())
    sql = payload.get("sql")
    if isinstance(sql, Mapping):
        query = sql.get("query")
        if isinstance(query, str) and query.strip():
            candidates.append(query.strip())
    return candidates


def _dbbench_uses_state_changing_checker(task_index: int | str | None) -> bool:
    """Return whether DBBench correctness is a DB-state check, not answer text."""
    return any(_DBBENCH_STATE_CHANGING_SQL_RE.match(candidate) for candidate in _dbbench_sql_strings(task_index))


def _dbbench_task_tables(task_index: int | str | None) -> list[Mapping[str, Any]]:
    payload = _dbbench_task_payload(task_index)
    if not isinstance(payload, Mapping):
        return []
    tables = payload.get("table")
    if isinstance(tables, Mapping):
        return [tables]
    if isinstance(tables, Sequence) and not isinstance(tables, (str, bytes, bytearray)):
        return [table for table in tables if isinstance(table, Mapping)]
    return []


def _dbbench_normalize_identifier(value: Any) -> str:
    text = str(value).strip().strip("`\"")
    return re.sub(r"\s+", " ", text).lower()


def _dbbench_quote_sqlite_identifier(value: Any) -> str:
    return f'"{str(value).replace(chr(34), chr(34) * 2)}"'


def _dbbench_normalize_state_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _dbbench_state_row_key(row: Sequence[Any]) -> str:
    return json.dumps(list(row), ensure_ascii=False, separators=(",", ":"))


def _dbbench_state_hash(state: Mapping[str, Any]) -> str:
    payload = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dbbench_state_row_dict(columns: Sequence[str], row: Sequence[Any]) -> dict[str, Any]:
    return {column: row[index] if index < len(row) else None for index, column in enumerate(columns)}


def _dbbench_build_sqlite_conn_for_task(task_index: int | str | None) -> sqlite3.Connection | None:
    tables = _dbbench_task_tables(task_index)
    if not tables:
        return None
    conn = sqlite3.connect(":memory:")
    for table in tables:
        table_name = table.get("table_name")
        table_info = table.get("table_info")
        if table_name is None or not isinstance(table_info, Mapping):
            conn.close()
            return None
        columns_payload = table_info.get("columns")
        if not isinstance(columns_payload, Sequence) or isinstance(columns_payload, (str, bytes, bytearray)):
            conn.close()
            return None
        columns = [
            str(column.get("name"))
            for column in columns_payload
            if isinstance(column, Mapping) and column.get("name") is not None
        ]
        if not columns:
            conn.close()
            return None
        quoted_columns = ", ".join(f"{_dbbench_quote_sqlite_identifier(column)} TEXT" for column in columns)
        conn.execute(f"CREATE TABLE {_dbbench_quote_sqlite_identifier(table_name)} ({quoted_columns})")
        rows = table_info.get("rows")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)) and rows:
            placeholders = ", ".join(["?"] * len(columns))
            insert_sql = (
                f"INSERT INTO {_dbbench_quote_sqlite_identifier(table_name)} "
                f"({', '.join(_dbbench_quote_sqlite_identifier(column) for column in columns)}) "
                f"VALUES ({placeholders})"
            )
            for row in rows:
                if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
                    values = [str(row[index]) if index < len(row) else None for index in range(len(columns))]
                    conn.execute(insert_sql, values)
    conn.commit()
    return conn


def _dbbench_capture_sqlite_state(conn: sqlite3.Connection, task_index: int | str | None) -> dict[str, Any] | None:
    state: dict[str, Any] = {}
    for table in _dbbench_task_tables(task_index):
        table_name = table.get("table_name")
        table_info = table.get("table_info")
        if table_name is None or not isinstance(table_info, Mapping):
            return None
        columns_payload = table_info.get("columns")
        if not isinstance(columns_payload, Sequence) or isinstance(columns_payload, (str, bytes, bytearray)):
            return None
        columns = [
            str(column.get("name"))
            for column in columns_payload
            if isinstance(column, Mapping) and column.get("name") is not None
        ]
        if not columns:
            return None
        normalized_table = _dbbench_normalize_identifier(table_name)
        normalized_columns = [_dbbench_normalize_identifier(column) for column in columns]
        column_sql = ", ".join(_dbbench_quote_sqlite_identifier(column) for column in columns)
        rows = conn.execute(f"SELECT {column_sql} FROM {_dbbench_quote_sqlite_identifier(table_name)}").fetchall()
        normalized_rows = [
            [_dbbench_normalize_state_value(value) for value in row]
            for row in rows
        ]
        normalized_rows.sort(key=_dbbench_state_row_key)
        state[normalized_table] = {
            "columns": normalized_columns,
            "rows": normalized_rows,
        }
    return dict(sorted(state.items()))


def _dbbench_state_delta(before_state: Mapping[str, Any], after_state: Mapping[str, Any]) -> dict[str, Any]:
    inserted: dict[str, list[dict[str, Any]]] = {}
    deleted: dict[str, list[dict[str, Any]]] = {}
    updated_like: list[dict[str, Any]] = []
    table_names = sorted(set(before_state) | set(after_state))
    for table_name in table_names:
        before_table = before_state.get(table_name, {}) if isinstance(before_state.get(table_name), Mapping) else {}
        after_table = after_state.get(table_name, {}) if isinstance(after_state.get(table_name), Mapping) else {}
        columns = list(after_table.get("columns") or before_table.get("columns") or [])
        before_rows = [list(row) for row in before_table.get("rows", [])]
        after_rows = [list(row) for row in after_table.get("rows", [])]
        before_counts = Counter(_dbbench_state_row_key(row) for row in before_rows)
        after_counts = Counter(_dbbench_state_row_key(row) for row in after_rows)

        inserted_rows: list[list[Any]] = []
        deleted_rows: list[list[Any]] = []
        for row_key, count in sorted((after_counts - before_counts).items()):
            inserted_rows.extend([json.loads(row_key)] * count)
        for row_key, count in sorted((before_counts - after_counts).items()):
            deleted_rows.extend([json.loads(row_key)] * count)

        if inserted_rows:
            inserted[table_name] = [_dbbench_state_row_dict(columns, row) for row in inserted_rows]
        if deleted_rows:
            deleted[table_name] = [_dbbench_state_row_dict(columns, row) for row in deleted_rows]
        if inserted_rows and deleted_rows and len(inserted_rows) == len(deleted_rows):
            for old_row, new_row in zip(deleted_rows, inserted_rows):
                updated_like.append(
                    {
                        "table": table_name,
                        "old": _dbbench_state_row_dict(columns, old_row),
                        "new": _dbbench_state_row_dict(columns, new_row),
                    }
                )
    return {
        "inserted": inserted,
        "deleted": deleted,
        "updated_like": updated_like,
    }


def _dbbench_state_delta_kind(delta: Mapping[str, Any]) -> str:
    inserted = delta.get("inserted") if isinstance(delta.get("inserted"), Mapping) else {}
    deleted = delta.get("deleted") if isinstance(delta.get("deleted"), Mapping) else {}
    inserted_count = sum(len(rows) for rows in inserted.values() if isinstance(rows, list))
    deleted_count = sum(len(rows) for rows in deleted.values() if isinstance(rows, list))
    if inserted_count == 0 and deleted_count == 0:
        return "no-change"
    if inserted_count > 0 and deleted_count == 0:
        return "insert"
    if inserted_count == 0 and deleted_count > 0:
        return "delete"
    if inserted_count == deleted_count:
        return "update"
    return "mixed"


def _dbbench_state_delta_tables(delta: Mapping[str, Any]) -> list[str]:
    tables: set[str] = set()
    inserted = delta.get("inserted") if isinstance(delta.get("inserted"), Mapping) else {}
    deleted = delta.get("deleted") if isinstance(delta.get("deleted"), Mapping) else {}
    for table_name, rows in inserted.items():
        if rows:
            tables.add(str(table_name))
    for table_name, rows in deleted.items():
        if rows:
            tables.add(str(table_name))
    return sorted(tables)


def _dbbench_sqlite_replay_sql(sql: str) -> str:
    text = sql.strip()
    while text.endswith(";"):
        text = text[:-1].strip()
    text = re.sub(r"^\s*INSERT\s+IGNORE\s+INTO\b", "INSERT OR IGNORE INTO", text, flags=re.IGNORECASE)
    return text


def _dbbench_attach_replayed_state_bucket(tdp: Any, task_index: int | str | None) -> bool:
    """Attach a DBBench final-state bucket by replaying successful write SQL locally."""
    metadata = getattr(tdp, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    result = metadata.get("result")
    if isinstance(result, Mapping):
        result = dict(result)
    else:
        result = {}
    metadata["result"] = result
    metadata["dbbench_state_bucket_attempted"] = True

    if _dbbench_tdp_state_outcome_bucket(tdp) is not None:
        metadata["dbbench_state_replay"] = {"status": "already_present"}
        return True
    if str(metadata.get("status") or "").strip().lower() not in {"completed", "success"}:
        metadata["dbbench_state_replay"] = {"status": "skipped_non_completed"}
        return False

    conn = _dbbench_build_sqlite_conn_for_task(task_index)
    if conn is None:
        metadata["dbbench_state_replay"] = {"status": "missing_task_table"}
        return False
    applied_statements = 0
    skipped_observed_errors = 0
    replay_errors: list[dict[str, str]] = []
    try:
        before_state = _dbbench_capture_sqlite_state(conn, task_index)
        if before_state is None:
            metadata["dbbench_state_replay"] = {"status": "capture_failed"}
            return False
        steps = getattr(tdp, "steps", None)
        if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes, bytearray)):
            for step in steps:
                step_had_observed_error = _dbbench_step_observation_has_sql_error(step)
                for query in _dbbench_step_execute_sql_queries(step):
                    if not _DBBENCH_STATE_CHANGING_SQL_RE.match(_dbbench_sql_without_comments(query)):
                        continue
                    if step_had_observed_error:
                        skipped_observed_errors += 1
                        continue
                    try:
                        conn.execute(_dbbench_sqlite_replay_sql(query))
                        conn.commit()
                        applied_statements += 1
                    except sqlite3.Error as exc:
                        conn.rollback()
                        replay_errors.append({"query": query[:300], "error": str(exc)[:300]})
        if replay_errors:
            metadata["dbbench_state_replay"] = {
                "status": "replay_failed",
                "applied_statements": applied_statements,
                "skipped_observed_errors": skipped_observed_errors,
                "errors": replay_errors[:3],
            }
            return False
        after_state = _dbbench_capture_sqlite_state(conn, task_index)
        if after_state is None:
            metadata["dbbench_state_replay"] = {"status": "capture_failed"}
            return False
        state_hash = _dbbench_state_hash(after_state)
        delta = _dbbench_state_delta(before_state, after_state)
        change_kind = _dbbench_state_delta_kind(delta)
        changed_tables = _dbbench_state_delta_tables(delta)
        if change_kind == "no-change":
            bucket = "db-state:no-change"
        else:
            table_key = "+".join(changed_tables) if changed_tables else "unknown"
            bucket = f"db-state:{change_kind}:{table_key}:{state_hash[:12]}"
        result.update(
            {
                "dbbench_state_bucket": bucket,
                "dbbench_state_change_kind": change_kind,
                "dbbench_state_changed_tables": changed_tables,
                "dbbench_state_hash": state_hash,
                "dbbench_state_canonical_delta": delta,
                "dbbench_state_canonicalization_version": _DBBENCH_RUNNER_STATE_CANONICALIZATION_VERSION,
                "dbbench_state_canonicalization_source": "runner_sqlite_replay",
            }
        )
        metadata["dbbench_state_replay"] = {
            "status": "ok",
            "applied_statements": applied_statements,
            "skipped_observed_errors": skipped_observed_errors,
        }
        return True
    finally:
        conn.close()


def _decode_dbbench_tool_arguments(function: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Decode DBBench tool-call arguments without depending on evaluator state."""
    arguments_text = function.get("arguments")
    if isinstance(arguments_text, Mapping):
        return dict(arguments_text)
    if not isinstance(arguments_text, str):
        return None
    try:
        arguments: Any = json.loads(arguments_text)
    except json.JSONDecodeError:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return dict(arguments) if isinstance(arguments, Mapping) else None


def _dbbench_tool_calls_from_formatted_action(action: str) -> list[dict[str, Any]]:
    """Recover simple Tool: name({...}) lines from older cached TDP records."""
    tool_calls: list[dict[str, Any]] = []
    for line in action.splitlines():
        line = line.strip()
        if not line.lower().startswith("tool:"):
            continue
        call_text = line[len("tool:") :].strip()
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", call_text)
        if match is None:
            continue
        name, arguments = match.groups()
        tool_calls.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return tool_calls


def _dbbench_step_tool_calls(step: Any) -> list[Mapping[str, Any]]:
    """Return preserved tool calls for a DBBench step, falling back to action text."""
    metadata = getattr(step, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    candidates: list[Any] = []
    raw_message = metadata.get("raw_assistant_message")
    if isinstance(raw_message, Mapping):
        candidates.append(raw_message.get("tool_calls"))
    candidates.append(metadata.get("tool_calls"))

    for candidate in candidates:
        if isinstance(candidate, list) and any(isinstance(item, Mapping) for item in candidate):
            return [item for item in candidate if isinstance(item, Mapping)]

    action = getattr(step, "realized_decision", None)
    if isinstance(action, str):
        return _dbbench_tool_calls_from_formatted_action(action)
    return []


def _dbbench_step_execute_sql_queries(step: Any) -> list[str]:
    queries: list[str] = []
    for tool_call in _dbbench_step_tool_calls(step):
        function = tool_call.get("function") if isinstance(tool_call, Mapping) else None
        if not isinstance(function, Mapping):
            continue
        if str(function.get("name", "")).strip() != "execute_sql":
            continue
        arguments = _decode_dbbench_tool_arguments(function)
        if not isinstance(arguments, Mapping):
            continue
        query = arguments.get("query")
        if isinstance(query, str) and query.strip():
            queries.append(query.strip())
    return queries


def _dbbench_sql_without_comments(sql: str) -> str:
    parts: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(sql):
        char = sql[index]
        if quote is not None:
            parts.append(char)
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    parts.append(sql[index + 1])
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            parts.append(char)
            index += 1
            continue
        if char == "-" and index + 1 < len(sql) and sql[index + 1] == "-":
            index += 2
            while index < len(sql) and sql[index] not in {"\n", "\r"}:
                index += 1
            parts.append(" ")
            continue
        if char == "/" and index + 1 < len(sql) and sql[index + 1] == "*":
            index += 2
            while index + 1 < len(sql) and not (sql[index] == "*" and sql[index + 1] == "/"):
                index += 1
            index = min(index + 2, len(sql))
            parts.append(" ")
            continue
        parts.append(char)
        index += 1
    return "".join(parts)


def _dbbench_lower_unquoted_sql(sql: str) -> str:
    parts: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(sql):
        char = sql[index]
        if quote is not None:
            parts.append(char)
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    parts.append(sql[index + 1])
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            parts.append(char)
        else:
            parts.append(char.lower())
        index += 1
    return "".join(parts)


def _dbbench_normalize_sql_signature(sql: str) -> str | None:
    """Normalize SQL enough for outcome buckets while preserving string values."""
    text = _dbbench_sql_without_comments(sql).strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    while text.endswith(";"):
        text = text[:-1].strip()
    if not text:
        return None
    return _dbbench_lower_unquoted_sql(text)


def _dbbench_compact_bucket_value(value: str, *, max_length: int = 400) -> str:
    if len(value) <= max_length:
        return value
    digest = sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{value[:max_length]}...#{digest}"


def _dbbench_step_observation_has_sql_error(step: Any) -> bool:
    metadata = getattr(step, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    observation = metadata.get("observation")
    if not isinstance(observation, str):
        observation = getattr(step, "observation", None)
    return isinstance(observation, str) and _DBBENCH_SQL_ERROR_RE.search(observation) is not None


def _dbbench_tdp_write_outcome_bucket(tdp: Any) -> str:
    """Bucket legacy state-changing DBBench trajectories by their executed write SQL."""
    operations: list[str] = []
    steps = getattr(tdp, "steps", None)
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes, bytearray)):
        for step in steps:
            for query in _dbbench_step_execute_sql_queries(step):
                if not _DBBENCH_STATE_CHANGING_SQL_RE.match(_dbbench_sql_without_comments(query)):
                    continue
                signature = _dbbench_normalize_sql_signature(query)
                if signature is None:
                    continue
                prefix = "write-error" if _dbbench_step_observation_has_sql_error(step) else "write-sql"
                operations.append(f"{prefix}:{_dbbench_compact_bucket_value(signature)}")
    if operations:
        if len(operations) == 1:
            return operations[0]
        return f"write-sequence:{_dbbench_compact_bucket_value(' -> '.join(operations))}"

    metadata = getattr(tdp, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    status = metadata.get("status")
    if isinstance(status, str) and status.strip() and status.strip().lower() not in {"completed", "success"}:
        return f"no-answer:{status.strip()}"
    if _dbbench_tdp_answer(tdp) is not None:
        return "no-write:final_text"
    if isinstance(status, str) and status.strip():
        return f"no-answer:{status.strip()}"
    return "no-write:unknown"


def _dbbench_tdp_state_outcome_bucket(tdp: Any) -> str | None:
    """Return the prediction-only final-state bucket for DBBench write tasks."""
    metadata = getattr(tdp, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    result = metadata.get("result")
    result = result if isinstance(result, Mapping) else {}
    bucket = result.get("dbbench_state_bucket")
    if isinstance(bucket, str) and bucket.strip():
        return bucket.strip()
    canonicalization = result.get("dbbench_state_canonicalization")
    if isinstance(canonicalization, Mapping):
        bucket = canonicalization.get("bucket")
        if isinstance(bucket, str) and bucket.strip():
            return bucket.strip()
    return None


def _dbbench_completed_write_tdp_needs_state_bucket(tdp: Any) -> bool:
    metadata = getattr(tdp, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if metadata.get("semantic_postprocessing_applies") is not False:
        return False
    status = str(metadata.get("status") or "").strip().lower()
    if status not in {"completed", "success"}:
        return False
    return True


def _dbbench_answer_compare_key(answer: str) -> str:
    value = answer.strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" .,:;\"'`")
    value = value.replace(",", "")
    return value


def _dbbench_numeric_value(answer: str) -> float | None:
    value = _dbbench_answer_compare_key(answer)
    match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value)
    if match is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _dbbench_answers_equivalent(candidate: str | None, references: Sequence[str]) -> bool | None:
    if candidate is None:
        return None
    candidate = candidate.strip()
    if not candidate or candidate in {_DBBENCH_NO_ANSWER, _DBBENCH_AMBIGUOUS_ANSWER}:
        return False
    candidate_key = _dbbench_answer_compare_key(candidate)
    candidate_number = _dbbench_numeric_value(candidate)
    for reference in references:
        reference = reference.strip()
        if not reference:
            continue
        if candidate_key == _dbbench_answer_compare_key(reference):
            return True
        reference_number = _dbbench_numeric_value(reference)
        if candidate_number is not None and reference_number is not None and math.isclose(
            candidate_number,
            reference_number,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            return True
    if references:
        return False
    return None


def _parse_dbbench_normalizer_output(output: str) -> str | None:
    text = output.strip()
    if not text:
        return None
    payload: Any | None = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is not None:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None
    if isinstance(payload, Mapping):
        answer = payload.get("answer")
        if isinstance(answer, str):
            answer = answer.strip()
            return answer or None
    if text in {_DBBENCH_NO_ANSWER, _DBBENCH_AMBIGUOUS_ANSWER}:
        return text
    # Fall back to a plain text response only if it is a single concise line.
    if "\n" not in text and len(text) <= 200:
        return text.strip("` ")
    return None


async def _dbbench_normalize_answer_with_backbone(
    model_client: Any,
    *,
    question: str | None,
    raw_answer: str | None,
) -> tuple[str | None, dict[str, Any]]:
    if raw_answer is None or not str(raw_answer).strip():
        return None, {"method": "none", "status": "no_answer"}
    raw_answer = str(raw_answer).strip()
    inference = getattr(model_client, "ainference", None)
    if not callable(inference):
        return raw_answer, {"method": "raw_fallback", "status": "model_client_missing_ainference"}
    prompt = (
        "Rewrite the candidate answer into the concise answer format expected by the question.\n"
        "Do not solve the task again. Do not use outside knowledge. Preserve the meaning of the candidate answer.\n"
        f"If the candidate does not contain an answer, return {json.dumps({'answer': _DBBENCH_NO_ANSWER})}.\n"
        f"If the candidate contains multiple conflicting answers, return {json.dumps({'answer': _DBBENCH_AMBIGUOUS_ANSWER})}.\n"
        "Return JSON only with exactly one string field named answer.\n\n"
        f"Question:\n{question or ''}\n\n"
        f"Candidate answer:\n{raw_answer}\n"
    )
    try:
        output = await inference(
            [
                {
                    "role": "system",
                    "content": "You are a conservative DBBench answer reformatter. You only rewrite submitted answers; you never solve from scratch.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
    except Exception as exc:
        LOGGER.warning("DBBench answer normalization failed; using raw answer: %s", exc)
        return raw_answer, {"method": "raw_fallback", "status": "normalizer_failed", "error": str(exc)}
    semantic_answer = _parse_dbbench_normalizer_output(str(output))
    if semantic_answer is None:
        return raw_answer, {
            "method": "raw_fallback",
            "status": "invalid_normalizer_output",
            "raw_output": str(output)[:500],
        }
    return semantic_answer, {"method": "qwen_answer_reformatter", "status": "ok", "raw_output": str(output)[:500]}


def _dbbench_tdp_answer(tdp: Any) -> str | None:
    """Return the committed DBBench answer for one TDP, if present."""
    answer = getattr(tdp, "final_answer", None)
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    metadata = getattr(tdp, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    hard_finalization = metadata.get("hard_finalization")
    if isinstance(hard_finalization, Mapping):
        hard_answer = hard_finalization.get("answer")
        if isinstance(hard_answer, str) and hard_answer.strip():
            return hard_answer.strip()
    result = metadata.get("result")
    if isinstance(result, Mapping):
        result_answer = result.get("answer")
        if isinstance(result_answer, str) and result_answer.strip():
            return result_answer.strip()
    return None


def _dbbench_tdp_semantic_answer(tdp: Any) -> str | None:
    metadata = getattr(tdp, "metadata", None)
    if isinstance(metadata, Mapping):
        answer = metadata.get("semantic_answer")
        if isinstance(answer, str) and answer.strip():
            answer = answer.strip()
            if answer in {_DBBENCH_NO_ANSWER, _DBBENCH_AMBIGUOUS_ANSWER}:
                return None
            return answer
    return _dbbench_tdp_answer(tdp)


async def _postprocess_dbbench_tdp_answers(
    tdps: Sequence[Any],
    *,
    model_client: Any,
    task_index: int | str | None,
) -> None:
    references = _dbbench_reference_answers(task_index)
    use_semantic_answer_check = not _dbbench_uses_state_changing_checker(task_index)
    cache: dict[tuple[str, str], tuple[str | None, dict[str, Any]]] = {}
    for tdp in tdps:
        raw_answer = _dbbench_tdp_answer(tdp)
        question = getattr(tdp, "prompt", None)
        metadata = getattr(tdp, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        result = metadata.get("result")
        result = result if isinstance(result, Mapping) else {}
        official_is_correct = _normalize_accuracy_flag(result.get("is_correct"))
        metadata["reference_answers"] = references
        metadata["semantic_postprocessing_applies"] = use_semantic_answer_check
        if raw_answer is None:
            metadata["raw_answer"] = None
            metadata["semantic_answer"] = None
            metadata["postprocessed_is_correct"] = (
                False if use_semantic_answer_check and references else official_is_correct
            )
            metadata["answer_normalization"] = {"method": "none", "status": "no_answer"}
            if not use_semantic_answer_check:
                _dbbench_attach_replayed_state_bucket(tdp, task_index)
            continue
        if not use_semantic_answer_check:
            metadata["raw_answer"] = raw_answer
            metadata["semantic_answer"] = raw_answer
            metadata["postprocessed_is_correct"] = official_is_correct
            metadata["answer_normalization"] = {
                "method": "official_evaluator",
                "status": "state_changing_task",
            }
            _dbbench_attach_replayed_state_bucket(tdp, task_index)
            continue
        key = (str(question or ""), raw_answer)
        if key not in cache:
            cache[key] = await _dbbench_normalize_answer_with_backbone(
                model_client,
                question=str(question or ""),
                raw_answer=raw_answer,
            )
        semantic_answer, normalization_metadata = cache[key]
        postprocessed_is_correct = _dbbench_answers_equivalent(semantic_answer, references)
        metadata["raw_answer"] = raw_answer
        metadata["semantic_answer"] = semantic_answer
        metadata["postprocessed_is_correct"] = postprocessed_is_correct
        metadata["answer_normalization"] = normalization_metadata


def _dbbench_majority_answer(tdps: Sequence[Any], *, semantic: bool = False) -> str | None:
    """Vote over non-empty committed DBBench answers only."""
    answer_getter = _dbbench_tdp_semantic_answer if semantic else _dbbench_tdp_answer
    answers = [answer for tdp in tdps if (answer := answer_getter(tdp)) is not None]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def _dbbench_correctness_for_selected_answer(
    tdps: Sequence[Any],
    selected_answer: str | None,
    *,
    semantic: bool = False,
) -> bool | None:
    """Attach strict or postprocessed DBBench correctness to the selected answer."""
    if selected_answer is None:
        return None
    if semantic:
        values: list[bool] = []
        for tdp in tdps:
            if _dbbench_tdp_semantic_answer(tdp) != selected_answer:
                continue
            metadata = getattr(tdp, "metadata", None)
            if not isinstance(metadata, Mapping):
                continue
            correctness = _normalize_accuracy_flag(metadata.get("postprocessed_is_correct"))
            if correctness is not None:
                values.append(correctness)
        if any(values):
            return True
        if values:
            return False
        return None
    correctness_values: list[bool] = []
    for tdp in tdps:
        if _dbbench_tdp_answer(tdp) != selected_answer:
            continue
        metadata = getattr(tdp, "metadata", None)
        result = metadata.get("result") if isinstance(metadata, Mapping) else None
        if not isinstance(result, Mapping):
            continue
        correctness = _normalize_accuracy_flag(result.get("is_correct"))
        if correctness is not None:
            correctness_values.append(correctness)
    if any(correctness_values):
        return True
    if correctness_values:
        return False
    return None


def _dbbench_tdp_outcome_bucket(tdp: Any) -> str:
    """Return the prediction-only DBBench outcome bucket used for outcome entropy."""
    metadata = getattr(tdp, "metadata", None)
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if metadata.get("semantic_postprocessing_applies") is False:
        state_bucket = _dbbench_tdp_state_outcome_bucket(tdp)
        if state_bucket is not None:
            return state_bucket
        return _dbbench_tdp_write_outcome_bucket(tdp)
    normalized_answer = metadata.get("semantic_answer")
    if isinstance(normalized_answer, str) and normalized_answer.strip():
        normalized_answer = normalized_answer.strip()
        if normalized_answer == _DBBENCH_AMBIGUOUS_ANSWER:
            return "ambiguous-answer"
        if normalized_answer == _DBBENCH_NO_ANSWER:
            return "no-answer:normalizer"
        return f"answer:{normalized_answer}"
    semantic_answer = _dbbench_tdp_semantic_answer(tdp)
    if isinstance(semantic_answer, str) and semantic_answer.strip():
        return f"answer:{semantic_answer.strip()}"
    status = metadata.get("status")
    if isinstance(status, str) and status.strip():
        return f"no-answer:{status.strip()}"
    return "no-answer:unknown"


def _compute_dbbench_outcome_entropy(tdps: Sequence[Any]) -> tuple[float | None, dict[str, int]]:
    """Compute outcome entropy from DBBench answer/write/status buckets."""
    outcome_counts: dict[str, int] = {}
    for tdp in tdps:
        bucket = _dbbench_tdp_outcome_bucket(tdp)
        outcome_counts[bucket] = outcome_counts.get(bucket, 0) + 1
    total = sum(outcome_counts.values())
    if total <= 0:
        return None, outcome_counts
    entropy = 0.0
    for count in outcome_counts.values():
        probability = count / total
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return float(entropy), outcome_counts


def _dbbench_bucket_family(bucket: str) -> str:
    """Classify DBBench outcome buckets using deterministic prefixes."""
    if bucket.startswith("answer:"):
        return "answer"
    if bucket.startswith("no-answer:"):
        return "no_answer"
    if bucket.startswith("invalid-answer:"):
        return "invalid_answer"
    if bucket.startswith("ambiguous-answer"):
        return "ambiguous_answer"
    if bucket.startswith("db-state:no-change"):
        return "db_state_no_change"
    if bucket.startswith("db-state:"):
        return "db_state"
    if bucket.startswith("write-sequence:"):
        return "write_sequence"
    if bucket.startswith("write-sql:"):
        return "write_sql"
    if bucket.startswith("write-error:"):
        return "write_error"
    return "unknown"


def _dbbench_normalize_answer_bucket_value(value: str) -> str:
    """Return a conservative canonical form for an answer bucket value."""
    text = " ".join(str(value).strip().split())
    quote_chars = "\"'`“”‘’"
    while len(text) >= 2 and text[0] in quote_chars and text[-1] in quote_chars:
        text = text[1:-1].strip()
    text = text.rstrip()
    if len(text) > 1 and text[-1] in ".。":
        text = text[:-1].rstrip()

    numeric_candidate = text.replace(",", "")
    if numeric_candidate:
        try:
            number = Decimal(numeric_candidate)
        except InvalidOperation:
            number = None
        if number is not None and number.is_finite():
            normalized = number.normalize()
            if normalized == normalized.to_integral_value():
                return str(normalized.quantize(Decimal(1)))
            return format(normalized, "f").rstrip("0").rstrip(".")

    simple_categorical = {
        "higher",
        "lower",
        "yes",
        "no",
        "true",
        "false",
        "none",
        "null",
        "unknown",
        "same",
        "different",
    }
    folded = text.casefold()
    if folded in simple_categorical:
        return folded
    return text


def _dbbench_deterministic_canonical_bucket(bucket: str) -> str:
    family = _dbbench_bucket_family(bucket)
    if family != "answer":
        return bucket
    answer = bucket[len("answer:") :]
    return f"answer:{_dbbench_normalize_answer_bucket_value(answer)}"


def _dbbench_deterministic_canonical_outcome_counts(
    raw_outcome_counts: Mapping[str, int],
) -> dict[str, int]:
    canonical_counts: dict[str, int] = {}
    for bucket, count in raw_outcome_counts.items():
        canonical_bucket = _dbbench_deterministic_canonical_bucket(str(bucket))
        canonical_counts[canonical_bucket] = canonical_counts.get(canonical_bucket, 0) + int(count)
    return canonical_counts


def _dbbench_answer_candidates_from_counts(outcome_counts: Mapping[str, int]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for bucket in outcome_counts:
        bucket = str(bucket)
        if not bucket.startswith("answer:"):
            continue
        answer = bucket[len("answer:") :]
        if answer not in seen:
            seen.add(answer)
            candidates.append(answer)
    return candidates


def _dbbench_semantic_bucket_canonicalizer_extra_body() -> dict[str, Any]:
    """Return the vLLM structured-output schema for semantic bucket canonicalization."""
    return {
        "chat_template_kwargs": {"enable_thinking": False},
        "structured_outputs": {
            "json": {
                "type": "object",
                "properties": {
                    "judgment": {"type": "string", "enum": ["same", "different", "unknown"]},
                    "canonical_outcome": {"type": "string"},
                    "mismatched_buckets": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["judgment", "canonical_outcome", "mismatched_buckets", "reason"],
                "additionalProperties": False,
            }
        },
    }


def _parse_dbbench_semantic_bucket_canonicalizer_output(output: str) -> dict[str, Any] | None:
    text = str(output or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    candidates = [text]
    for match in re.finditer(r"\{", text):
        suffix = text[match.start() :]
        for end in reversed([candidate.end() for candidate in re.finditer(r"\}", suffix)]):
            candidates.append(suffix[:end])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        judgment = payload.get("judgment")
        canonical_outcome = payload.get("canonical_outcome")
        mismatched_buckets = payload.get("mismatched_buckets")
        reason = payload.get("reason")
        if judgment not in {"same", "different", "unknown"}:
            continue
        if not isinstance(canonical_outcome, str):
            canonical_outcome = ""
        if not isinstance(mismatched_buckets, list):
            mismatched_buckets = []
        return {
            "judgment": judgment,
            "canonical_outcome": canonical_outcome,
            "mismatched_buckets": [str(item) for item in mismatched_buckets],
            "reason": str(reason or ""),
        }
    return None


async def _dbbench_semantic_canonicalize_answer_buckets(
    *,
    model_client: Any,
    question: str | None,
    deterministic_counts: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, Any]]:
    """Collapse answer-only bucket variants when Qwen judges them semantically identical."""
    answer_candidates = _dbbench_answer_candidates_from_counts(deterministic_counts)
    if len(answer_candidates) <= 1:
        return dict(deterministic_counts), {
            "method": "deterministic",
            "status": "already_collapsed",
            "qwen_called": False,
        }

    detailed_callable = getattr(model_client, "sample_many_detailed", None)
    inference_callable = getattr(model_client, "ainference", None)
    if not callable(detailed_callable) and not callable(inference_callable):
        return dict(deterministic_counts), {
            "method": "deterministic",
            "status": "model_client_missing_text_completion",
            "qwen_called": False,
        }

    system_prompt = (
        "You are a strict semantic normalizer for benchmark answer buckets.\n"
        "Decide only whether the candidate answers express the same final answer for the given task question.\n"
        "Do not decide whether the answer is correct.\n"
        "Ignore harmless surface differences: capitalization, quotes, punctuation, whitespace, and numeric formatting such as 167 vs 167.0.\n"
        "A concise answer and a sentence are equivalent if the sentence clearly identifies the same final answer.\n"
        "Answers are different if they name different entities, different numbers, different labels, or answer different fields.\n"
        "Use the task question to decide what the final answer type should be.\n"
        "When answers are equivalent, set canonical_outcome to the shortest concise final answer, not an explanatory sentence.\n"
        "Return only JSON."
    )
    candidates_text = "\n".join(f"- {json.dumps(candidate, ensure_ascii=False)}" for candidate in answer_candidates)
    user_prompt = (
        f"Task question:\n{question or ''}\n\n"
        f"Candidate answers:\n{candidates_text}\n\n"
        "Are all candidate answers semantically equivalent as final answers to the task?"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_output: str | None = None
    structured_requested = False
    try:
        if callable(detailed_callable):
            structured_requested = True
            result = detailed_callable(
                messages,
                temperature=0.0,
                n=1,
                extra_body=_dbbench_semantic_bucket_canonicalizer_extra_body(),
                max_tokens=_DBBENCH_SEMANTIC_BUCKET_CANONICALIZER_MAX_TOKENS,
            )
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)) and result:
                first = result[0]
                if isinstance(first, Mapping):
                    raw_output = str(first.get("output") or "")
        if raw_output is None and callable(inference_callable):
            result = inference_callable(messages, temperature=0.0)
            if asyncio.iscoroutine(result):
                result = await result
            raw_output = str(result)
    except (ModelGenerationError, TypeError, ValueError, OSError) as exc:
        return dict(deterministic_counts), {
            "method": "qwen_semantic_answer_bucket_canonicalizer",
            "status": "failed",
            "qwen_called": True,
            "structured_outputs_requested": structured_requested,
            "error": str(exc)[:500],
        }

    parsed = _parse_dbbench_semantic_bucket_canonicalizer_output(raw_output or "")
    if parsed is None:
        return dict(deterministic_counts), {
            "method": "qwen_semantic_answer_bucket_canonicalizer",
            "status": "invalid_output",
            "qwen_called": True,
            "structured_outputs_requested": structured_requested,
            "raw_output": str(raw_output or "")[:500],
        }

    metadata = {
        "method": "qwen_semantic_answer_bucket_canonicalizer",
        "status": "ok",
        "qwen_called": True,
        "structured_outputs_requested": structured_requested,
        "judgment": parsed["judgment"],
        "canonical_outcome": parsed["canonical_outcome"],
        "mismatched_buckets": parsed["mismatched_buckets"],
        "reason": parsed["reason"][:500],
        "raw_output": str(raw_output or "")[:500],
    }
    if parsed["judgment"] != "same":
        return dict(deterministic_counts), metadata

    canonical_answer = parsed["canonical_outcome"].strip()
    if canonical_answer.startswith("answer:"):
        canonical_answer = canonical_answer[len("answer:") :]
    canonical_answer = _dbbench_normalize_answer_bucket_value(canonical_answer)
    if not canonical_answer:
        metadata["status"] = "empty_canonical_outcome"
        return dict(deterministic_counts), metadata
    return {f"answer:{canonical_answer}": int(sum(deterministic_counts.values()))}, metadata


async def _canonicalize_dbbench_outcome_counts(
    *,
    model_client: Any,
    question: str | None,
    raw_outcome_counts: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, Any]]:
    """Return the canonical DBBench outcome counts used for outcome-level scores."""
    deterministic_counts = _dbbench_deterministic_canonical_outcome_counts(raw_outcome_counts)
    families = {_dbbench_bucket_family(str(bucket)) for bucket in raw_outcome_counts}
    metadata: dict[str, Any] = {
        "version": _DBBENCH_CANONICAL_OUTCOME_BUCKET_VERSION,
        "bucket_families": sorted(families),
        "qwen_called": False,
        "status": "deterministic",
    }
    if families == {"answer"}:
        canonical_counts, semantic_metadata = await _dbbench_semantic_canonicalize_answer_buckets(
            model_client=model_client,
            question=question,
            deterministic_counts=deterministic_counts,
        )
        metadata.update(semantic_metadata)
        metadata["version"] = _DBBENCH_CANONICAL_OUTCOME_BUCKET_VERSION
        metadata["bucket_families"] = sorted(families)
        return canonical_counts, metadata
    if "answer" in families:
        metadata["status"] = "deterministic_mixed_bucket_families"
    elif families & {"db_state", "db_state_no_change", "write_sequence", "write_sql", "write_error"}:
        metadata["status"] = "deterministic_db_or_write_bucket"
    return deterministic_counts, metadata


def _dbbench_positive_integer_outcome_counts(outcome_counts: Mapping[str, Any]) -> dict[str, int]:
    """Return finite positive bucket counts as integers for canonicalization metadata."""
    counts: dict[str, int] = {}
    for bucket, count in outcome_counts.items():
        if not isinstance(bucket, str) or not isinstance(count, (int, float)):
            continue
        value = float(count)
        if not math.isfinite(value) or value <= 0.0:
            continue
        counts[bucket] = int(value)
    return counts


def _dbbench_question_from_tdps(tdps: Sequence[TrajectoryDependentDecisionProcess]) -> str | None:
    """Return the first available DBBench task prompt from a sampled TDP list."""
    for tdp in tdps:
        prompt = getattr(tdp, "prompt", None)
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    return None


def _entropy_from_weighted_counts(counts: Mapping[str, Any]) -> float | None:
    total = float(sum(max(0.0, float(count)) for count in counts.values() if isinstance(count, (int, float))))
    if total <= 0.0:
        return None
    entropy = 0.0
    for count in counts.values():
        if not isinstance(count, (int, float)):
            continue
        probability = max(0.0, float(count)) / total
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return float(entropy)


def _most_frequent_dbbench_outcome_bucket(outcome_counts: Mapping[str, int]) -> str | None:
    if not outcome_counts:
        return None
    return max(outcome_counts.items(), key=lambda item: item[1])[0]


def _dbbench_has_only_task_limit_outcomes(outcome_counts: Mapping[str, int]) -> bool:
    return len(outcome_counts) == 1 and next(iter(outcome_counts), None) in _DBBENCH_UNANIMOUS_TASK_LIMIT_BUCKETS


def _deserialize_dbbench_counterfactual_branch(payload: Any) -> TDPCounterfactualBranch:
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid cached DBBench TDP counterfactual branch.")
    sampled_metadata = payload.get("target_sampled_output_metadata")
    return TDPCounterfactualBranch(
        source_decision=str(payload.get("source_decision", "")),
        target_sampled_decisions=[str(value) for value in payload.get("target_sampled_decisions", [])],
        target_sampled_output_metadata=[
            dict(value) for value in sampled_metadata if isinstance(value, Mapping)
        ] if isinstance(sampled_metadata, list) else [],
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
    )


def _deserialize_dbbench_counterfactual_record(payload: Any) -> TDPCounterfactualRecord:
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid cached DBBench TDP counterfactual record.")
    branches = payload.get("branches")
    return TDPCounterfactualRecord(
        source_step_index=int(payload.get("source_step_index", 0)),
        realized_source_decision=str(payload.get("realized_source_decision", "")),
        branches=[_deserialize_dbbench_counterfactual_branch(branch) for branch in branches] if isinstance(branches, list) else [],
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
    )


def _deserialize_dbbench_tdp_step(payload: Any) -> TDPStepRecord:
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid cached DBBench TDP step.")
    measurements = payload.get("uncertainty_measurements")
    counterfactual_records = payload.get("counterfactual_records")
    return TDPStepRecord(
        index=int(payload.get("index", 0)),
        realized_decision=str(payload.get("realized_decision", "")),
        sampled_decisions=[str(value) for value in payload.get("sampled_decisions", [])],
        uncertainty_measurements={
            str(key): float(value)
            for key, value in measurements.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        } if isinstance(measurements, Mapping) else {},
        counterfactual_records=[
            _deserialize_dbbench_counterfactual_record(record)
            for record in counterfactual_records
        ] if isinstance(counterfactual_records, list) else [],
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
    )


def _deserialize_dbbench_tdp(payload: Any) -> TrajectoryDependentDecisionProcess:
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid cached DBBench TDP.")
    steps = payload.get("steps")
    return TrajectoryDependentDecisionProcess(
        sample_id=str(payload.get("sample_id", "")),
        prompt=str(payload.get("prompt", "")),
        steps=[_deserialize_dbbench_tdp_step(step) for step in steps] if isinstance(steps, list) else [],
        final_answer=str(payload.get("final_answer")) if payload.get("final_answer") is not None else None,
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
    )


def _dbbench_tdp_cache_enabled(config: AgentBenchRunnerConfig) -> bool:
    return config.method == "uprop"


def _dbbench_hard_finalization_disabled() -> bool:
    return os.getenv(_DISABLE_HARD_FINALIZATION_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _dbbench_tdps_include_postprocessing(tdps: Sequence[TrajectoryDependentDecisionProcess]) -> bool:
    for tdp in tdps:
        if "semantic_postprocessing_applies" not in tdp.metadata:
            return False
        if (
            _dbbench_completed_write_tdp_needs_state_bucket(tdp)
            and _dbbench_tdp_state_outcome_bucket(tdp) is None
            and tdp.metadata.get("dbbench_state_bucket_attempted") is not True
        ):
            return False
    return True


def _dbbench_shared_tdp_cache_key(
    config: AgentBenchRunnerConfig,
    sample: AgentBenchSample,
    *,
    emulate_tool_calls: bool | None = None,
) -> dict[str, Any]:
    resolved_emulate_tool_calls = config.emulate_tool_calls if emulate_tool_calls is None else bool(emulate_tool_calls)
    model_signature = dict(_agentbench_model_signature(config))
    model_signature["emulate_tool_calls"] = resolved_emulate_tool_calls
    return {
        "cache_version": _DBBENCH_SHARED_TDP_CACHE_VERSION,
        "task_name": sample.task_name,
        "task_index": sample.task_index,
        "model": model_signature,
        "temperature": config.temperature,
        "seed": config.seed,
        "max_steps": config.max_steps,
        "max_tokens": config.max_tokens,
        "tdp_samples": config.tdp_samples,
        "per_step_samples": 1,
        "include_counterfactuals": False,
        "fair_trajectory_budget": True,
        "hard_finalization_disabled": _dbbench_hard_finalization_disabled(),
        "emulate_tool_calls": resolved_emulate_tool_calls,
        "executor": "agentbench_dbbench",
    }


def _load_dbbench_shared_tdp_cache(
    storage: SharedSamplingStorage,
    *,
    config: AgentBenchRunnerConfig,
    sample: AgentBenchSample,
) -> list[TrajectoryDependentDecisionProcess] | None:
    if not config.fair_trajectory_budget:
        return None
    if not _dbbench_tdp_cache_enabled(config):
        return None
    payload = None
    cache_key_emulate_tool_calls = config.emulate_tool_calls
    for candidate_emulate_tool_calls in (
        [config.emulate_tool_calls, False]
        if config.emulate_tool_calls
        else [config.emulate_tool_calls]
    ):
        payload = storage.load(
            _DBBENCH_SHARED_TDP_CACHE_CATEGORY,
            sample_id=sample.sample_id,
            key=_dbbench_shared_tdp_cache_key(config, sample, emulate_tool_calls=candidate_emulate_tool_calls),
        )
        if isinstance(payload, Mapping):
            cache_key_emulate_tool_calls = bool(candidate_emulate_tool_calls)
            break
    if not isinstance(payload, Mapping):
        return None
    raw_tdps = payload.get("tdps")
    required_tdps = config.tdp_samples
    if not isinstance(raw_tdps, list) or len(raw_tdps) < required_tdps:
        return None
    try:
        tdps = [_deserialize_dbbench_tdp(item) for item in raw_tdps[: required_tdps]]
    except (TypeError, ValueError) as exc:
        LOGGER.warning("Ignoring invalid DBBench shared TDP cache for sample_id=%s: %s", sample.sample_id, exc)
        return None
    if not _dbbench_tdps_include_postprocessing(tdps):
        LOGGER.info("Ignoring stale DBBench shared TDP cache without postprocessing metadata for %s.", sample.sample_id)
        return None
    for tdp in tdps:
        tdp.metadata["tdp_cache_status"] = "hit"
        tdp.metadata["tdp_cache_source_method"] = str(payload.get("source_method") or "unknown")
        tdp.metadata["tdp_cache_key_emulate_tool_calls"] = cache_key_emulate_tool_calls
        if cache_key_emulate_tool_calls != config.emulate_tool_calls:
            tdp.metadata["tdp_cache_status"] = "hit_legacy_emulate_false"
    return tdps


async def _extend_dbbench_cached_tdps_for_degree(
    *,
    config: AgentBenchRunnerConfig,
    sample: AgentBenchSample,
    executor: AgentBenchDBBenchExecutor,
    tdps: Sequence[TrajectoryDependentDecisionProcess],
) -> tuple[list[TrajectoryDependentDecisionProcess], int, int]:
    per_step_samples = max(config.per_step_samples, _DEGREE_SHARED_CACHE_PER_STEP_SAMPLES)
    extended_tdps: list[TrajectoryDependentDecisionProcess] = []
    extended_count = 0
    for position, tdp in enumerate(tdps):
        cached_per_step_samples = tdp.metadata.get("per_step_samples")
        if isinstance(cached_per_step_samples, int) and cached_per_step_samples >= per_step_samples:
            extended_tdps.append(tdp)
            continue
        trajectory_index = tdp.metadata.get("trajectory_index")
        if not isinstance(trajectory_index, int) or trajectory_index < 0:
            trajectory_index = position
        extended_tdp = await executor._extend_cached_tdp(
            sample=sample,
            trajectory_index=trajectory_index,
            cached_tdp=tdp,
            per_step_samples=per_step_samples,
            include_counterfactuals=False,
        )
        extended_tdp.metadata["tdp_cache_status"] = "hit_extended"
        extended_tdp.metadata["tdp_cache_source_method"] = str(tdp.metadata.get("tdp_cache_source_method") or "pe")
        extended_tdp.metadata["tdp_cache_extended_from_per_step_samples"] = cached_per_step_samples
        extended_tdp.metadata["tdp_cache_extended_to_per_step_samples"] = per_step_samples
        extended_tdps.append(extended_tdp)
        extended_count += 1
    return extended_tdps, extended_count, per_step_samples


def _store_dbbench_shared_tdp_cache(
    storage: SharedSamplingStorage,
    *,
    config: AgentBenchRunnerConfig,
    sample: AgentBenchSample,
    tdps: Sequence[TrajectoryDependentDecisionProcess],
) -> None:
    if config.method != "pe" or not _dbbench_tdp_cache_enabled(config):
        return
    if not config.fair_trajectory_budget:
        return
    try:
        storage.store(
            _DBBENCH_SHARED_TDP_CACHE_CATEGORY,
            sample_id=sample.sample_id,
            key=_dbbench_shared_tdp_cache_key(config, sample),
            value={
                "source_method": "pe",
                "tdps": [asdict(tdp) for tdp in tdps],
            },
        )
    except OSError as exc:
        LOGGER.warning("Failed to write DBBench shared TDP cache for sample_id=%s: %s", sample.sample_id, exc)


def _store_dbbench_degree_tdp_cache(
    *,
    executor: AgentBenchDBBenchExecutor,
    sample: AgentBenchSample,
    tdps: Sequence[TrajectoryDependentDecisionProcess],
) -> int:
    """Write final Degree TDPs to the reusable per-trajectory shared cache."""
    if executor.shared_sampling_storage is None:
        return 0

    written_count = 0
    for fallback_index, tdp in enumerate(tdps):
        trajectory_index = tdp.metadata.get("trajectory_index")
        if not isinstance(trajectory_index, int) or trajectory_index < 0:
            trajectory_index = fallback_index
        cache_key = executor._tdp_cache_key(
            sample,
            trajectory_index=trajectory_index,
            include_counterfactuals=False,
        )
        tdp.metadata["degree_tdp_cache_status"] = "written"
        tdp.metadata["degree_tdp_cache_source_method"] = "degree"
        try:
            executor._store_cached_tdp(sample, cache_key, tdp)
        except OSError as exc:
            tdp.metadata["degree_tdp_cache_status"] = "failed"
            tdp.metadata["degree_tdp_cache_error"] = str(exc)[:500]
            LOGGER.warning(
                "Failed to write DBBench Degree TDP cache for sample_id=%s trajectory_index=%s: %s",
                sample.sample_id,
                trajectory_index,
                exc,
            )
            continue
        written_count += 1
    return written_count


def _estimate_dbbench_outcome_entropy_from_tdps(
    *,
    config: AgentBenchRunnerConfig,
    tdps: Sequence[TrajectoryDependentDecisionProcess],
    cache_status: str,
) -> MultiStepBaselineEstimate:
    return MultiStepBaselineEstimate(
        total_uncertainty=None,
        baseline_name="outcome-entropy",
        aggregation="outcome",
        tdp_scores=[],
        tdp_step_scores=[],
        tdps=list(tdps),
        metadata={
            "executor": "agentbench_dbbench",
            "num_tdps": len(tdps),
            "per_step_samples": 1,
            "uncertainty_key": "final_outcome_entropy",
            "requires_logprobs": False,
            "uncertainty_available": False,
            "hard_finalization_count": 0,
            "fair_trajectory_budget": config.fair_trajectory_budget,
            "tdp_cache_status": cache_status,
            "tdp_cache_source_method": "pe" if cache_status == "hit" else None,
            "used_no_tool_call_samples": any(bool(tdp.metadata.get("used_no_tool_call_samples")) for tdp in tdps),
            "no_tool_call_sample_count": sum(
                [count for tdp in tdps if isinstance((count := tdp.metadata.get("no_tool_call_sample_count")), int)]
            ),
            "hard_requirement_failure": any(bool(tdp.metadata.get("hard_requirement_failure")) for tdp in tdps),
        },
    )


async def _apply_dbbench_outcome_entropy(estimate: Any, *, model_client: Any) -> Any:
    """Compute the single DBBench OE score from canonical prediction buckets."""
    _, raw_outcome_counts = _compute_dbbench_outcome_entropy(estimate.tdps)
    question = None
    if estimate.tdps:
        first_prompt = getattr(estimate.tdps[0], "prompt", None)
        if isinstance(first_prompt, str):
            question = first_prompt
    outcome_counts, canonicalization_metadata = await _canonicalize_dbbench_outcome_counts(
        model_client=model_client,
        question=question,
        raw_outcome_counts=raw_outcome_counts,
    )
    total_uncertainty = _entropy_from_weighted_counts(outcome_counts)
    most_frequent_bucket = _most_frequent_dbbench_outcome_bucket(outcome_counts)
    uncertainty_reason = "entropy"
    if _dbbench_has_only_task_limit_outcomes(outcome_counts):
        total_uncertainty = _OUTCOME_ENTROPY_TASK_LIMIT_UNCERTAINTY
        uncertainty_reason = "all_trajectories_task_limit_reached"
    estimate.total_uncertainty = total_uncertainty
    estimate.metadata["uncertainty_key"] = "dbbench_canonical_outcome_entropy"
    estimate.metadata["outcome_counts"] = outcome_counts
    estimate.metadata["canonical_outcome_counts"] = outcome_counts
    estimate.metadata["raw_outcome_counts_diagnostic"] = raw_outcome_counts
    estimate.metadata["outcome_bucket_canonicalization"] = canonicalization_metadata
    estimate.metadata["outcome_count"] = len(outcome_counts)
    estimate.metadata["most_frequent_outcome_bucket"] = most_frequent_bucket
    estimate.metadata["uncertainty_reason"] = uncertainty_reason
    estimate.metadata["uncertainty_available"] = total_uncertainty is not None
    return estimate


def _aggregate_no_tool_call_info(*sources: Any) -> dict[str, Any] | None:
    used_no_tool_call_samples = False
    no_tool_call_sample_count = 0
    hard_requirement_failure = False

    for source in sources:
        iterable: Sequence[Any]
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            iterable = source
        else:
            iterable = [source]
        for item in iterable:
            if not isinstance(item, Mapping):
                continue
            used_no_tool_call_samples = used_no_tool_call_samples or bool(item.get("used_no_tool_call_samples"))
            hard_requirement_failure = hard_requirement_failure or bool(item.get("hard_requirement_failure"))
            count = item.get("no_tool_call_sample_count")
            if isinstance(count, int):
                no_tool_call_sample_count += count

    if not used_no_tool_call_samples and not hard_requirement_failure and no_tool_call_sample_count == 0:
        return None

    info: dict[str, Any] = {
        "used_no_tool_call_samples": used_no_tool_call_samples,
        "no_tool_call_sample_count": no_tool_call_sample_count,
    }
    if hard_requirement_failure:
        info["hard_requirement_failure"] = True
    return info


def _merge_record_info(base_info: Any, *sources: Any) -> Any:
    extra_info = _aggregate_no_tool_call_info(*sources)
    if extra_info is None:
        return base_info
    if isinstance(base_info, Mapping):
        merged = dict(base_info)
        merged.update(extra_info)
        return merged
    if base_info is None:
        return extra_info
    return {"message": str(base_info), **extra_info}


def _serialize_uprop_result(sample: AgentBenchSample, estimate: Any) -> dict[str, Any]:
    reference_answers = _dbbench_reference_answers(sample.task_index)
    raw_predicted_answer = _dbbench_majority_answer(estimate.tdps)
    predicted_answer = _dbbench_majority_answer(estimate.tdps, semantic=True)
    is_correct = _dbbench_correctness_for_selected_answer(estimate.tdps, raw_predicted_answer)
    postprocessed_is_correct = _dbbench_correctness_for_selected_answer(
        estimate.tdps,
        predicted_answer,
        semantic=True,
    )
    status = _majority_final_answer(
        [str(tdp.metadata.get("status")) if tdp.metadata.get("status") is not None else None for tdp in estimate.tdps]
    )
    return {
        "sample_id": sample.sample_id,
        "task_name": sample.task_name,
        "task_index": sample.task_index,
        "question": estimate.tdps[0].prompt if estimate.tdps else None,
        "method": "uprop",
        "status": status,
        "reference_answer": reference_answers[0] if reference_answers else None,
        "reference_answers": reference_answers,
        "raw_predicted_answer": raw_predicted_answer,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "postprocessed_is_correct": postprocessed_is_correct,
        "uncertainty": float(estimate.total_uncertainty),
        "error": None,
        "info": _merge_record_info(None, estimate.metadata, [tdp.metadata for tdp in estimate.tdps]),
        "estimate": asdict(estimate),
    }


def _serialize_baseline_result(sample: AgentBenchSample, estimate: Any) -> dict[str, Any]:
    reference_answers = _dbbench_reference_answers(sample.task_index)
    raw_predicted_answer = _dbbench_majority_answer(estimate.tdps)
    predicted_answer = _dbbench_majority_answer(estimate.tdps, semantic=True)
    is_correct = _dbbench_correctness_for_selected_answer(estimate.tdps, raw_predicted_answer)
    postprocessed_is_correct = _dbbench_correctness_for_selected_answer(
        estimate.tdps,
        predicted_answer,
        semantic=True,
    )
    status = _majority_final_answer(
        [str(tdp.metadata.get("status")) if tdp.metadata.get("status") is not None else None for tdp in estimate.tdps]
    )
    return {
        "sample_id": sample.sample_id,
        "task_name": sample.task_name,
        "task_index": sample.task_index,
        "question": estimate.tdps[0].prompt if estimate.tdps else None,
        "method": estimate.baseline_name,
        "status": status,
        "reference_answer": reference_answers[0] if reference_answers else None,
        "reference_answers": reference_answers,
        "raw_predicted_answer": raw_predicted_answer,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "postprocessed_is_correct": postprocessed_is_correct,
        "uncertainty": float(estimate.total_uncertainty) if estimate.total_uncertainty is not None else None,
        "error": None,
        "info": _merge_record_info(None, estimate.metadata, [tdp.metadata for tdp in estimate.tdps]),
        "estimate": asdict(estimate),
    }


def _serialize_generation_failure(
    sample: AgentBenchSample,
    *,
    method: str,
    error: ModelGenerationError,
) -> dict[str, Any]:
    uncertainty = None
    info = error.to_record_metadata()
    return {
        "sample_id": sample.sample_id,
        "task_name": sample.task_name,
        "task_index": sample.task_index,
        "question": None,
        "method": method,
        "status": "generation_failed",
        "reference_answer": None,
        "raw_predicted_answer": None,
        "predicted_answer": None,
        "is_correct": False,
        "postprocessed_is_correct": False,
        "uncertainty": uncertainty,
        "error": str(error),
        "info": info,
        "estimate": None,
        "result": None,
        "trajectory": None,
        "agentbench_output": None,
    }


def _format_uncertainty_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "n/a"


async def run_agentbench_dbbench_samples(
    samples: Sequence[AgentBenchSample],
    *,
    model_client: Any,
    config: AgentBenchRunnerConfig,
    tracker: ExperimentTracker | None = None,
    sample_offset: int = 0,
    sample_total: int | None = None,
    record_callback: Callable[[AgentBenchSample, dict[str, Any], int, int], None] | None = None,
    controller_client: AgentBenchControllerClient | None = None,
) -> list[dict[str, Any]]:
    """Generate the paper's UProp trajectory pools for DBBench."""
    active_controller = controller_client or AgentBenchControllerClient(config.task_name, config.controller_url)
    shared_sampling_storage = SharedSamplingStorage(config.shared_sampling_dir)
    executor = AgentBenchDBBenchExecutor(
        active_controller,
        model_client,
        temperature=config.temperature,
        backbone_per_step_samples=config.backbone_per_step_samples,
        next_step_entropy_samples=config.next_step_entropy_samples,
        max_steps=config.max_steps,
        collect_sample_logprobs=True,
        shared_sampling_storage=shared_sampling_storage,
        model_signature=_agentbench_model_signature(config),
        strict_replay=config.strict_replay,
    )
    estimator = UPropEstimator(
        trajectory_samples=config.tdp_samples,
        per_step_samples=config.per_step_samples,
        tau=config.tau,
        ratio_epsilon=config.uprop_ratio_epsilon,
        ratio_cap=config.uprop_ratio_cap,
        intrinsic_cap=config.uprop_intrinsic_cap,
        intrinsic_transform=config.uprop_intrinsic_transform,
        outcome_bucket_fn=_dbbench_tdp_outcome_bucket,
    )
    records: list[dict[str, Any]] = []
    total_samples = sample_total or len(samples)
    for index, sample in enumerate(samples, start=sample_offset + 1):
        LOGGER.info(
            "Running AgentBench DBBench sample %s/%s with UProp: task=%s index=%s",
            index,
            total_samples,
            sample.task_name,
            sample.task_index,
        )
        if tracker is not None:
            tracker.log_event(
                "sample_started",
                sample_id=sample.sample_id,
                position=index,
                total_samples=total_samples,
                method="uprop",
            )
        try:
            cached_tdps = _load_dbbench_shared_tdp_cache(
                shared_sampling_storage,
                config=config,
                sample=sample,
            )
            if cached_tdps is None:
                estimate = await estimator.estimate(sample=sample, executor=executor)
                await _postprocess_dbbench_tdp_answers(
                    estimate.tdps,
                    model_client=model_client,
                    task_index=sample.task_index,
                )
                estimate.metadata.setdefault("tdp_cache_status", "miss")
            else:
                await _postprocess_dbbench_tdp_answers(
                    cached_tdps,
                    model_client=model_client,
                    task_index=sample.task_index,
                )
                estimate = estimator.estimate_from_tdps(
                    list(cached_tdps),
                    executor_name=executor.name,
                    metadata={
                        "tdp_cache_status": "hit",
                        "tdp_cache_source_method": str(
                            cached_tdps[0].metadata.get("tdp_cache_source_method") or "uprop"
                        ),
                        "fair_trajectory_budget": config.fair_trajectory_budget,
                    },
                )
            record = _serialize_uprop_result(sample, estimate)
        except ModelGenerationError as exc:
            LOGGER.warning(
                "DBBench sample_id=%s failed due to model error code=%s retryable=%s",
                sample.sample_id,
                exc.error_code,
                exc.retryable,
            )
            record = _serialize_generation_failure(sample, method="uprop", error=exc)
            if tracker is not None:
                tracker.log_event(
                    "sample_failed",
                    sample_id=sample.sample_id,
                    position=index,
                    method="uprop",
                    error=str(exc),
                    error_code=exc.error_code,
                    retryable=exc.retryable,
                )
        else:
            if tracker is not None:
                tracker.log_event(
                    "sample_completed",
                    sample_id=sample.sample_id,
                    position=index,
                    uncertainty=record["uncertainty"],
                    is_correct=record["is_correct"],
                    status=record["status"],
                )
        records.append(record)
        if record_callback is not None:
            record_callback(sample, record, index, total_samples)
    return records


def summarize_agentbench_dbbench_results(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return summarize_records(
        records,
        default_method="uprop",
        empty_mean_uncertainty=None,
    )


def write_agentbench_dbbench_results(output_path: str, records: Sequence[dict[str, Any]]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


def _build_experiment_name(config: AgentBenchRunnerConfig) -> str:
    if config.experiment_name:
        return config.experiment_name
    variant_suffix = _agentbench_variant_name_suffix(emulate_tool_calls=config.emulate_tool_calls)
    return f"agentbench-dbbench-{config.method}-{config.model.replace('/', '-')}-{config.task_name}{variant_suffix}"


def _agentbench_resume_fingerprint(config: AgentBenchRunnerConfig) -> str:
    payload = asdict(config)
    for key in _RESUME_IGNORED_CONFIG_KEYS:
        payload.pop(key, None)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha1(serialized.encode("utf-8")).hexdigest()[:12]


def _explicit_agentbench_indices(task_index: int | str | None) -> list[int | str] | None:
    """Return explicit AgentBench indices, accepting comma-separated CLI input."""
    if task_index is None:
        return None
    if isinstance(task_index, str) and "," in task_index:
        indices: list[int | str] = []
        for part in task_index.split(","):
            value = part.strip()
            if not value:
                continue
            indices.append(int(value) if value.isdigit() else value)
        return indices
    return [task_index]


def _default_run_name(config: AgentBenchRunnerConfig) -> str:
    explicit_indices = _explicit_agentbench_indices(config.task_index)
    if explicit_indices is not None:
        if len(explicit_indices) == 1:
            return f"task-{explicit_indices[0]}-{_agentbench_resume_fingerprint(config)}"
        return f"tasks-{len(explicit_indices)}-{_agentbench_resume_fingerprint(config)}"
    limit_label = "all" if config.limit is None else str(config.limit)
    return f"offset-{config.offset}-limit-{limit_label}-{_agentbench_resume_fingerprint(config)}"


def _agentbench_model_signature(config: AgentBenchRunnerConfig) -> dict[str, Any]:
    return reusable_model_signature(
        {
            "model": config.model,
            "emulate_tool_calls": config.emulate_tool_calls,
            "max_tokens": config.max_tokens,
        }
    )


def _agentbench_transport_signature(config: AgentBenchRunnerConfig) -> dict[str, Any]:
    return {
        "provider": config.provider,
        "base_url": config.base_url,
        "azure_endpoint": config.azure_endpoint,
        "api_version": config.api_version,
        "deployment_name": config.deployment_name,
    }


def _agentbench_backbone_compatibility(
    config: AgentBenchRunnerConfig,
    *,
    executor_name: str,
    replay_version: int = 3,
    strict_replay: bool = True,
    extra_executor_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    executor_signature: dict[str, Any] = {
        "executor": executor_name,
        "replay_version": replay_version,
        "sample_temperature": config.temperature,
        "strict_replay": strict_replay,
    }
    if extra_executor_fields:
        executor_signature.update(dict(extra_executor_fields))

    compatibility = {
        "family": "agentbench_backbone",
        "schema_version": 1,
        "model": _agentbench_model_signature(config),
        "executor": executor_signature,
    }
    serialized = json.dumps(compatibility, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    compatibility["fingerprint"] = sha1(serialized.encode("utf-8")).hexdigest()[:12]
    compatibility["transport"] = _agentbench_transport_signature(config)
    return compatibility


def _resolve_output_path(config: AgentBenchRunnerConfig, tracker: ExperimentTracker | None) -> str:
    if config.output_path is not None:
        return config.output_path
    if tracker is not None:
        return str(tracker.artifacts_dir / "results.jsonl")
    variant_suffix = _agentbench_variant_name_suffix(emulate_tool_calls=config.emulate_tool_calls)
    return f"outputs/agentbench-dbbench-{config.method}-{config.model.replace('/', '-')}-{config.task_name}{variant_suffix}.jsonl"


def _checkpoint_path(output_path: str, tracker: ExperimentTracker | None) -> Path:
    if tracker is not None:
        return tracker.artifacts_dir / "checkpoint.json"
    return Path(f"{output_path}.checkpoint.json")


def _load_agentbench_results(path: str | Path) -> list[dict[str, Any]]:
    result_path = Path(path)
    if not result_path.exists():
        return []
    with result_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_checkpoint(
    checkpoint_path: Path,
    *,
    config: AgentBenchRunnerConfig,
    selected_samples: Sequence[AgentBenchSample],
    output_path: str,
    records: Sequence[dict[str, Any]],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    payload = {
        "config_fingerprint": _agentbench_resume_fingerprint(config),
        "runner_config": asdict(config),
        "selected_sample_ids": [sample.sample_id for sample in selected_samples],
        "output_path": output_path,
        "records": list(records),
    }
    temp_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(checkpoint_path)
    _profile_log(
        "checkpoint_write",
        checkpoint_path=str(checkpoint_path),
        record_count=len(records),
        elapsed_ms=round((time.perf_counter() - start_time) * 1000, 3),
    )


def _filter_records_missing_dbbench_postprocessing(
    records: Sequence[dict[str, Any]],
    *,
    method: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    if method not in _DBBENCH_POSTPROCESSING_METHODS:
        return list(records), []
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for record in records:
        if isinstance(record, dict) and record.get("status") != "generation_failed":
            estimate = record.get("estimate")
            metadata = estimate.get("metadata") if isinstance(estimate, Mapping) else None
            if method in {OUTCOME_ENTROPY_BASELINE, "__unavailable_method"}:
                canonicalization = metadata.get("outcome_bucket_canonicalization") if isinstance(metadata, Mapping) else None
                if not isinstance(canonicalization, Mapping) or canonicalization.get("version") != _DBBENCH_CANONICAL_OUTCOME_BUCKET_VERSION:
                    dropped.append(str(record.get("sample_id")))
                    continue
            if "postprocessed_is_correct" not in record:
                dropped.append(str(record.get("sample_id")))
                continue
            tdps = estimate.get("tdps") if isinstance(estimate, Mapping) else None
            if isinstance(tdps, Sequence) and not isinstance(tdps, (str, bytes, bytearray)):
                missing_postprocessing_shape = False
                for tdp in tdps:
                    metadata = tdp.get("metadata") if isinstance(tdp, Mapping) else None
                    if isinstance(metadata, Mapping) and "semantic_postprocessing_applies" not in metadata:
                        missing_postprocessing_shape = True
                        break
                    if isinstance(metadata, Mapping) and metadata.get("semantic_postprocessing_applies") is False:
                        status = str(metadata.get("status") or "").strip().lower()
                        result = metadata.get("result")
                        result = result if isinstance(result, Mapping) else {}
                        has_state_bucket = isinstance(result.get("dbbench_state_bucket"), str) and bool(
                            str(result.get("dbbench_state_bucket")).strip()
                        )
                        canonicalization = result.get("dbbench_state_canonicalization")
                        if isinstance(canonicalization, Mapping):
                            has_state_bucket = has_state_bucket or (
                                isinstance(canonicalization.get("bucket"), str)
                                and bool(str(canonicalization.get("bucket")).strip())
                            )
                        state_bucket_attempted = metadata.get("dbbench_state_bucket_attempted") is True
                        if status in {"completed", "success"} and not has_state_bucket and not state_bucket_attempted:
                            missing_postprocessing_shape = True
                            break
                if missing_postprocessing_shape:
                    dropped.append(str(record.get("sample_id")))
                    continue
        kept.append(record)
    return kept, dropped


def _delete_checkpoint(checkpoint_path: Path) -> None:
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def _load_checkpoint_records(
    checkpoint_path: Path,
    *,
    config: AgentBenchRunnerConfig,
    selected_samples: Sequence[AgentBenchSample],
) -> list[dict[str, Any]]:
    if config.restart or not checkpoint_path.exists():
        return []
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    expected_fingerprint = _agentbench_resume_fingerprint(config)
    if payload.get("config_fingerprint") != expected_fingerprint:
        raise ValueError(
            "Existing checkpoint belongs to a different AgentBench dbbench configuration. Use --restart or change the run name/output path."
        )
    expected_sample_ids = [sample.sample_id for sample in selected_samples]
    if payload.get("selected_sample_ids") != expected_sample_ids:
        raise ValueError(
            "Existing checkpoint does not match the selected AgentBench dbbench samples. Use --restart or change the run name/output path."
        )
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("AgentBench dbbench checkpoint is invalid: records must be a list.")
    return records


def _build_tracker(config: AgentBenchRunnerConfig) -> ExperimentTracker | None:
    if not config.track_experiment:
        return None
    return ExperimentTracker(
        experiment=ExperimentConfig(
            name=_build_experiment_name(config),
            model=config.model,
            task_family=_agentbench_task_family("agentbench_dbbench", emulate_tool_calls=config.emulate_tool_calls),
            output_dir=config.tracking_dir,
            notes=config.notes,
        ),
        tracking_dir=config.tracking_dir,
        run_name=config.run_name or _default_run_name(config),
        tags=config.tags,
        metadata={
            "runner_config": asdict(config),
            "backbone_compatibility": _agentbench_backbone_compatibility(
                config,
                executor_name=AgentBenchSingleTrajectoryExecutor.name,
            ),
        },
        resume=not config.restart,
    )


def _wrap_model_client_for_tracking(
    model_client: Any,
    *,
    tracker: ExperimentTracker | None,
    provider: str,
) -> Any:
    if tracker is None:
        return model_client
    if not any(hasattr(model_client, attribute) for attribute in ("sample_many", "ainference", "inference")):
        return model_client
    return RecordedChatModelClient(model_client, tracker, provider=provider)


async def run_agentbench_dbbench(
    config: AgentBenchRunnerConfig,
    *,
    controller_client: AgentBenchControllerClient | None = None,
) -> dict[str, Any]:
    tracker = _build_tracker(config)
    output_path = _resolve_output_path(config, tracker)
    checkpoint_path = _checkpoint_path(output_path, tracker)
    progress_display = RunProgressDisplay(enabled=config.show_progress)
    configure_logging(
        config.log_level,
        config.log_path or (str(tracker.log_path) if tracker is not None else None),
        console_handler=progress_display.create_logging_handler(),
    )
    LOGGER.info(
        "Starting AgentBench dbbench run: method=%s model=%s provider=%s task=%s task_index=%s limit=%s offset=%s",
        config.method,
        config.model,
        config.provider,
        config.task_name,
        config.task_index,
        config.limit,
        config.offset,
    )
    if tracker is not None:
        tracker.log_event(
            "run_started",
            method=config.method,
            model=config.model,
            provider=config.provider,
            task_name=config.task_name,
        )
        if tracker.resumed:
            LOGGER.info("Resuming tracked run in %s", tracker.run_dir)
            tracker.log_event("run_resumed", previous_status=tracker.previous_status)

    active_controller = controller_client or AgentBenchControllerClient(config.task_name, config.controller_url)
    explicit_indices = _explicit_agentbench_indices(config.task_index)
    if explicit_indices is not None:
        selected_indices = explicit_indices
    else:
        indices = await asyncio.to_thread(active_controller.get_indices)
        selected_indices = indices[config.offset :]
        if config.limit is not None:
            selected_indices = selected_indices[: config.limit]

    if not selected_indices:
        raise ValueError("No AgentBench dbbench samples were selected for the requested task_index/offset/limit.")

    selected_samples = [
        AgentBenchSample(
            controller_url=config.controller_url,
            task_name=config.task_name,
            task_index=task_index,
        )
        for task_index in selected_indices
    ]
    if tracker is not None:
        tracker.log_event(
            "samples_selected",
            count=len(selected_samples),
            sample_ids=[sample.sample_id for sample in selected_samples],
        )

    model_client: Any | None = None
    refresh_missing_logprob_records = False

    async def ensure_refresh_decision(records: Sequence[dict[str, Any]]) -> bool:
        nonlocal model_client, refresh_missing_logprob_records
        if refresh_missing_logprob_records:
            return True
        if not any(
            isinstance(record, dict) and record_has_missing_logprob_state(record, method=config.method)
            for record in records
        ):
            return False
        if model_client is None:
            model_client = create_chat_model(
                ChatModelConfig(
                    provider=config.provider,
                    model=config.model,
                    api_key=config.api_key,
                    emulate_tool_calls=config.emulate_tool_calls,
                    max_tokens=config.max_tokens,
                    parallel_requests=config.parallel_requests,
                    seed=config.seed,
                    base_url=config.base_url,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.api_version,
                    deployment_name=config.deployment_name,
                )
            )
        refresh_missing_logprob_records = await probe_structured_logprob_support(
            model_client,
            probe_kind="tool",
            logger=LOGGER,
            model=config.model,
            provider=config.provider,
        )
        return refresh_missing_logprob_records

    checkpoint_records = _load_checkpoint_records(
        checkpoint_path,
        config=config,
        selected_samples=selected_samples,
    )
    dropped_checkpoint_sample_ids: list[str] = []
    if checkpoint_records:
        refresh_checkpoint_records = await ensure_refresh_decision(checkpoint_records)
        checkpoint_records, dropped_checkpoint_sample_ids = filter_stale_records(
            checkpoint_records,
            refresh_missing_logprob_records=refresh_checkpoint_records,
            resolve_uncertainty=resolve_record_uncertainty,
            method=config.method,
        )
        checkpoint_records, dropped_postprocessing_sample_ids = _filter_records_missing_dbbench_postprocessing(
            checkpoint_records,
            method=config.method,
        )
        dropped_checkpoint_sample_ids.extend(dropped_postprocessing_sample_ids)
        if dropped_checkpoint_sample_ids:
            LOGGER.info(
                "Dropping %s stale AgentBench dbbench checkpoint record(s): %s",
                len(dropped_checkpoint_sample_ids),
                ", ".join(dropped_checkpoint_sample_ids[:10]),
            )
            _write_checkpoint(
                checkpoint_path,
                config=config,
                selected_samples=selected_samples,
                output_path=output_path,
                records=checkpoint_records,
            )
            if tracker is not None:
                tracker.log_event(
                    "stale_records_dropped",
                    source="checkpoint",
                    count=len(dropped_checkpoint_sample_ids),
                    sample_ids=dropped_checkpoint_sample_ids,
                )
    if checkpoint_records:
        LOGGER.info(
            "Loaded %s checkpointed AgentBench dbbench records from %s",
            len(checkpoint_records),
            checkpoint_path,
        )
        if tracker is not None:
            tracker.log_event("checkpoint_loaded", records=len(checkpoint_records), checkpoint_path=str(checkpoint_path))

    if (
        tracker is not None
        and not config.restart
        and tracker.previous_status == "completed"
        and not checkpoint_path.exists()
        and Path(output_path).exists()
    ):
        existing_records = _load_agentbench_results(output_path)
        existing_ids = [record.get("sample_id") for record in existing_records]
        selected_ids = [sample.sample_id for sample in selected_samples]
        if existing_records and existing_ids == selected_ids:
            refresh_existing_records = await ensure_refresh_decision(existing_records)
            reusable_existing_records, dropped_existing_sample_ids = filter_stale_records(
                existing_records,
                refresh_missing_logprob_records=refresh_existing_records,
                resolve_uncertainty=resolve_record_uncertainty,
                method=config.method,
            )
            reusable_existing_records, dropped_postprocessing_sample_ids = _filter_records_missing_dbbench_postprocessing(
                reusable_existing_records,
                method=config.method,
            )
            dropped_existing_sample_ids.extend(dropped_postprocessing_sample_ids)
            if not dropped_existing_sample_ids:
                progress_display.start(
                    total=len(selected_samples),
                    description=f"AgentBench dbbench {config.method}",
                    completed=len(reusable_existing_records),
                )
                LOGGER.info("Existing completed run found at %s; reusing recorded results", output_path)
                summary = summarize_agentbench_dbbench_results(reusable_existing_records)
                tracker.finalize(
                    status="completed",
                    summary=summary,
                    artifacts={
                        "results": str(output_path),
                        "log": str(tracker.log_path),
                    },
                )
                await close_chat_model(model_client)
                progress_display.close(status="completed")
                return {
                    "summary": summary,
                    "output_path": str(output_path),
                    "records": reusable_existing_records,
                    "tracking_path": str(tracker.run_dir),
                }

            LOGGER.info(
                "Dropping %s stale AgentBench dbbench result record(s) from %s",
                len(dropped_existing_sample_ids),
                output_path,
            )
            checkpoint_records = reusable_existing_records
            _write_checkpoint(
                checkpoint_path,
                config=config,
                selected_samples=selected_samples,
                output_path=output_path,
                records=checkpoint_records,
            )
            if tracker is not None:
                tracker.log_event(
                    "stale_records_dropped",
                    source="results",
                    count=len(dropped_existing_sample_ids),
                    sample_ids=dropped_existing_sample_ids,
                )

    completed_sample_ids = {record.get("sample_id") for record in checkpoint_records}
    pending_samples = [sample for sample in selected_samples if sample.sample_id not in completed_sample_ids]
    all_records = list(checkpoint_records)
    progress_display.start(
        total=len(selected_samples),
        description=f"AgentBench dbbench {config.method}",
        completed=len(all_records),
    )
    LOGGER.info("AgentBench dbbench resume state: %s completed, %s pending", len(all_records), len(pending_samples))
    if tracker is not None:
        tracker.log_event(
            "resume_state",
            completed_records=len(all_records),
            pending_records=len(pending_samples),
        )

    def persist_record(sample: AgentBenchSample, record: dict[str, Any], index: int, total: int) -> None:
        del total
        all_records.append(record)
        progress_display.advance(
            position=index,
            sample_id=sample.sample_id,
            status=str(record.get("status") or "completed"),
            uncertainty=record.get("uncertainty"),
        )
        _write_checkpoint(
            checkpoint_path,
            config=config,
            selected_samples=selected_samples,
            output_path=output_path,
            records=all_records,
        )
        if tracker is not None:
            tracker.log_event("checkpoint_saved", records=len(all_records), checkpoint_path=str(checkpoint_path))

    try:
        if model_client is None:
            model_client = create_chat_model(
                ChatModelConfig(
                    provider=config.provider,
                    model=config.model,
                    api_key=config.api_key,
                    emulate_tool_calls=config.emulate_tool_calls,
                    max_tokens=config.max_tokens,
                    parallel_requests=config.parallel_requests,
                    seed=config.seed,
                    base_url=config.base_url,
                    azure_endpoint=config.azure_endpoint,
                    api_version=config.api_version,
                    deployment_name=config.deployment_name,
                )
            )
        model_client = _wrap_model_client_for_tracking(
            model_client,
            tracker=tracker,
            provider=config.provider,
        )
        if pending_samples:
            record_count_before_run = len(all_records)
            new_records = await run_agentbench_dbbench_samples(
                pending_samples,
                model_client=model_client,
                config=config,
                tracker=tracker,
                sample_offset=len(all_records),
                sample_total=len(selected_samples),
                record_callback=persist_record,
                controller_client=active_controller,
            )
            if len(all_records) == record_count_before_run and new_records:
                all_records.extend(new_records)
                _write_checkpoint(
                    checkpoint_path,
                    config=config,
                    selected_samples=selected_samples,
                    output_path=output_path,
                    records=all_records,
                )

        summary = summarize_agentbench_dbbench_results(all_records)
        written_path = write_agentbench_dbbench_results(output_path, all_records)
        _delete_checkpoint(checkpoint_path)
        if tracker is not None:
            tracker.log_event("results_written", output_path=str(written_path))
            tracker.finalize(
                status="completed",
                summary=summary,
                artifacts={
                    "results": str(written_path),
                    "log": str(tracker.log_path),
                },
            )
        progress_display.close(status="completed")
        return {
            "summary": summary,
            "output_path": str(written_path),
            "records": all_records,
            "tracking_path": str(tracker.run_dir) if tracker is not None else None,
        }
    except Exception as exc:
        LOGGER.exception("AgentBench dbbench run failed")
        progress_display.close(status="failed")
        if tracker is not None:
            tracker.log_event("run_failed", error=str(exc))
            tracker.finalize(
                status="failed",
                summary=summarize_agentbench_dbbench_results(all_records),
                artifacts={
                    "results": str(output_path),
                    "log": str(tracker.log_path),
                    "checkpoint": str(checkpoint_path),
                },
            )
        raise
    finally:
        try:
            await close_chat_model(model_client)
        except Exception as exc:
            LOGGER.warning("Failed to close AgentBench dbbench model client: %s", exc)
        progress_display.close()
