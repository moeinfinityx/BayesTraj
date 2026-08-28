from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence


MODEL_SIGNATURE_TRANSPORT_KEYS = frozenset(
    {
        "api_key",
        "api_version",
        "azure_endpoint",
        "base_url",
        "deployment_name",
        "provider",
    }
)


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(nested_value) for key, nested_value in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sampling_fingerprint(payload: Mapping[str, Any]) -> str:
    normalized_payload = _normalize_json_value(payload)
    serialized = json.dumps(normalized_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return sha1(serialized.encode("utf-8")).hexdigest()


def reusable_model_signature(signature: Mapping[str, Any] | None) -> dict[str, Any]:
    if signature is None:
        return {}
    payload = {
        str(key): value
        for key, value in signature.items()
        if str(key) not in MODEL_SIGNATURE_TRANSPORT_KEYS
    }
    normalized_payload = _normalize_json_value(payload)
    if not isinstance(normalized_payload, dict):
        raise TypeError("Model signature must serialize to a JSON object.")
    return normalized_payload


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "sample"


class SharedSamplingStorage:
    schema_version = 1

    def __init__(self, root_dir: str = "outputs/shared_sampling") -> None:
        self.root_dir = Path(root_dir)

    def load(
        self,
        category: str,
        *,
        sample_id: str,
        key: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        normalized_key = self._normalized_key(key)
        path = self._record_path(category, sample_id=sample_id, key=normalized_key)
        if not path.exists():
            return self._load_legacy_record(category, sample_id=sample_id, normalized_key=normalized_key)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if self._normalized_stored_key(payload.get("key")) != normalized_key:
            return None
        value = payload.get("value")
        return value if isinstance(value, dict) else None

    def store(
        self,
        category: str,
        *,
        sample_id: str,
        key: Mapping[str, Any],
        value: Mapping[str, Any],
    ) -> Path:
        normalized_key = self._normalized_key(key)
        path = self._record_path(category, sample_id=sample_id, key=normalized_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "key": normalized_key,
            "value": _normalize_json_value(value),
        }
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(path)
        return path

    def _normalized_key(self, key: Mapping[str, Any]) -> dict[str, Any]:
        payload = {"schema_version": self.schema_version, **dict(key)}
        model_signature = payload.get("model")
        if isinstance(model_signature, Mapping):
            payload["model"] = reusable_model_signature(model_signature)
        normalized_payload = _normalize_json_value(payload)
        if not isinstance(normalized_payload, dict):
            raise TypeError("Sampling storage key must serialize to a JSON object.")
        return normalized_payload

    def _normalized_stored_key(self, key: Any) -> dict[str, Any] | None:
        if not isinstance(key, Mapping):
            return None
        payload = dict(key)
        model_signature = payload.get("model")
        if isinstance(model_signature, Mapping):
            payload["model"] = reusable_model_signature(model_signature)
        normalized_payload = _normalize_json_value(payload)
        return normalized_payload if isinstance(normalized_payload, dict) else None

    def _load_legacy_record(
        self,
        category: str,
        *,
        sample_id: str,
        normalized_key: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        sample_dir = self.root_dir / category / _slugify(sample_id)
        if not sample_dir.exists():
            return None
        for candidate in sample_dir.glob("*.json"):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if self._normalized_stored_key(payload.get("key")) != normalized_key:
                continue
            value = payload.get("value")
            return value if isinstance(value, dict) else None
        return None

    def _record_path(self, category: str, *, sample_id: str, key: Mapping[str, Any]) -> Path:
        return self.root_dir / category / _slugify(sample_id) / f"{sampling_fingerprint(key)}.json"


@dataclass(frozen=True)
class StepSample:
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    recovery_state: dict[str, Any] = field(default_factory=dict)


class SharedStepSampler:
    """Shared step-level sampling interface with append-only sample reuse."""

    def __init__(self, storage: SharedSamplingStorage | None) -> None:
        self.storage = storage

    async def sample(
        self,
        *,
        category: str,
        sample_id: str,
        key: Mapping[str, Any],
        sample_count: int,
        cursor: int = 0,
        sample_fn: Callable[[int], Awaitable[Sequence[StepSample | str | Mapping[str, Any]]]],
    ) -> list[StepSample]:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if cursor < 0:
            raise ValueError("cursor must be non-negative")

        if self.storage is None:
            generated = await sample_fn(sample_count)
            return self._coerce_samples(generated, expected_count=sample_count)

        cached_samples = self._load_samples(category=category, sample_id=sample_id, key=key)
        required_total = cursor + sample_count
        if len(cached_samples) < required_total:
            missing_count = required_total - len(cached_samples)
            generated = await sample_fn(missing_count)
            cached_samples.extend(self._coerce_samples(generated, expected_count=missing_count))
            self._store_samples(
                category=category,
                sample_id=sample_id,
                key=key,
                samples=cached_samples,
            )

        return cached_samples[cursor:required_total]

    def _load_samples(
        self,
        *,
        category: str,
        sample_id: str,
        key: Mapping[str, Any],
    ) -> list[StepSample]:
        if self.storage is None:
            return []
        payload = self.storage.load(category, sample_id=sample_id, key=key)
        if payload is None:
            return []
        raw_samples = payload.get("samples")
        if not isinstance(raw_samples, list):
            return []
        samples: list[StepSample] = []
        for raw_sample in raw_samples:
            try:
                samples.append(self._coerce_sample(raw_sample))
            except TypeError:
                continue
        return samples

    def _store_samples(
        self,
        *,
        category: str,
        sample_id: str,
        key: Mapping[str, Any],
        samples: Sequence[StepSample],
    ) -> None:
        if self.storage is None:
            return
        self.storage.store(
            category,
            sample_id=sample_id,
            key=key,
            value={
                "samples": [self._serialize_sample(sample) for sample in samples],
            },
        )

    def _coerce_samples(
        self,
        values: Sequence[StepSample | str | Mapping[str, Any]],
        *,
        expected_count: int,
    ) -> list[StepSample]:
        samples = [self._coerce_sample(value) for value in values]
        if len(samples) != expected_count:
            raise ValueError(
                f"Expected {expected_count} step samples but received {len(samples)}."
            )
        return samples

    def _coerce_sample(self, value: StepSample | str | Mapping[str, Any]) -> StepSample:
        if isinstance(value, StepSample):
            return value
        if isinstance(value, str):
            return StepSample(output=value)
        if not isinstance(value, Mapping):
            raise TypeError("Step samples must be strings, mappings, or StepSample instances.")

        output = value.get("output")
        if output is None:
            output = value.get("text")
        if output is None:
            output = value.get("content")
        if output is None:
            raise TypeError("Step sample mappings must include output, text, or content.")

        metadata = value.get("metadata")
        recovery_state = value.get("recovery_state")
        return StepSample(
            output=str(output),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
            recovery_state=dict(recovery_state) if isinstance(recovery_state, Mapping) else {},
        )

    def _serialize_sample(self, sample: StepSample) -> dict[str, Any]:
        return {
            "output": sample.output,
            "metadata": _normalize_json_value(sample.metadata),
            "recovery_state": _normalize_json_value(sample.recovery_state),
        }