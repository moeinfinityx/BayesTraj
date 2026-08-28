from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from ..config import ExperimentConfig


_PROFILE_ENV_VAR = "LTUQ_PROFILE_AGENTBENCH_TIMINGS"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _profile_enabled() -> bool:
    value = os.getenv(_PROFILE_ENV_VAR, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "run"


def _jsonl_record_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


class ExperimentTracker:
    def __init__(
        self,
        experiment: ExperimentConfig,
        *,
        tracking_dir: str = "outputs/experiments",
        run_name: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        self.experiment = experiment
        self.run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        resolved_run_name = _slugify(run_name or self.run_id)
        self.run_dir = Path(tracking_dir) / _slugify(experiment.name) / resolved_run_name
        self.artifacts_dir = self.run_dir / "artifacts"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "run.json"
        self.log_path = self.run_dir / "run.log"
        self.llm_calls_path = self.artifacts_dir / "llm_calls.jsonl"
        self.resumed = resume and self.manifest_path.exists()
        self.previous_status: str | None = None
        self._llm_call_count = _jsonl_record_count(self.llm_calls_path)
        if self.resumed:
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.run_id = str(self._manifest.get("run_id", self.run_id))
            self.previous_status = self._manifest.get("status")
            merged_metadata = dict(self._manifest.get("metadata", {}))
            merged_metadata.update(dict(metadata or {}))
            self._manifest["experiment"] = asdict(experiment)
            self._manifest["tags"] = list(tags) if tags else list(self._manifest.get("tags", []))
            self._manifest["metadata"] = merged_metadata
            self._manifest["status"] = "running"
            self._manifest["resumed_at"] = _utc_timestamp()
        else:
            self._manifest = {
                "run_id": self.run_id,
                "status": "running",
                "created_at": _utc_timestamp(),
                "updated_at": _utc_timestamp(),
                "experiment": asdict(experiment),
                "tags": list(tags),
                "metadata": dict(metadata or {}),
                "artifacts": {},
                "summary": None,
            }
        self._manifest.setdefault("artifacts", {})
        self._manifest["artifacts"].setdefault("llm_calls", str(self.llm_calls_path))
        self._write_manifest()

    def _write_manifest(self) -> None:
        self._manifest["updated_at"] = _utc_timestamp()
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2, sort_keys=True), encoding="utf-8")

    def log_event(self, name: str, **payload: Any) -> None:
        event = {
            "timestamp": _utc_timestamp(),
            "name": name,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    def record_llm_call(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        responses: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        self._llm_call_count += 1
        entry = {
            "timestamp": _utc_timestamp(),
            "call_index": self._llm_call_count,
            "messages": [dict(message) for message in messages],
            "responses": list(responses),
            "metadata": dict(metadata or {}),
        }
        start_time = time.perf_counter()
        with self.llm_calls_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
        if _profile_enabled():
            self.log_event(
                "profile_llm_call_persisted",
                call_index=self._llm_call_count,
                elapsed_ms=round((time.perf_counter() - start_time) * 1000, 3),
                response_count=len(responses),
            )
        self.log_event(
            "llm_call_recorded",
            call_index=self._llm_call_count,
            response_count=len(responses),
            llm_calls_path=str(self.llm_calls_path),
            call_type=entry["metadata"].get("call_type"),
        )
        return self._llm_call_count

    def finalize(
        self,
        *,
        status: str,
        summary: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
    ) -> None:
        self._manifest["status"] = status
        self._manifest["summary"] = dict(summary or {})
        merged_artifacts = dict(artifacts or {})
        if self.llm_calls_path.exists():
            merged_artifacts.setdefault("llm_calls", str(self.llm_calls_path))
        self._manifest["artifacts"] = merged_artifacts
        self._write_manifest()

    @property
    def status(self) -> str | None:
        return self._manifest.get("status")
