from __future__ import annotations

import asyncio
import ast
import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
from urllib import error, parse, request

from ..logprobs import compute_predictive_entropy_from_metadata
from ..sampling import SharedSamplingStorage, SharedStepSampler, StepSample, reusable_model_signature, sampling_fingerprint
from ..trajectory import (
    BackboneTrajectory,
    LocalBranchHistory,
    StepRecord,
    TDPCounterfactualBranch,
    TDPCounterfactualRecord,
    TDPStepRecord,
    TrajectoryDependentDecisionProcess,
)
from ..models.base import ModelGenerationError
from .base import BranchingExecutor


LOGGER = logging.getLogger(__name__)

_TOOL_CALL_SAMPLING_ERROR_CODE = "invalid_tool_call_sampling"
_NO_TOOL_CALL_HARD_FAILURE_REASON = "no_callable_tool_call"
_PROFILE_ENV_VAR = "LTUQ_PROFILE_AGENTBENCH_TIMINGS"
_WORKER_UNAVAILABLE_RETRIES_ENV_VAR = "LTUQ_AGENTBENCH_WORKER_RETRIES"
_WORKER_UNAVAILABLE_RETRY_BASE_SLEEP_ENV_VAR = "LTUQ_AGENTBENCH_WORKER_RETRY_BASE_SECONDS"
_WORKER_UNAVAILABLE_RETRY_MAX_SLEEP_ENV_VAR = "LTUQ_AGENTBENCH_WORKER_RETRY_MAX_SECONDS"
_REQUIRE_FROZEN_TDP_CACHE_ENV_VAR = "LTUQ_REQUIRE_FROZEN_TDP_CACHE"
_DBBENCH_REPLAY_VERSION = 3
_DBBENCH_STEP_SAMPLING_VERSION = 6
_DBBENCH_STATE_BUCKET_VERSION = 1
_DBBENCH_SESSION_DATABASE_NAME_PATTERN = r"dbbench_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}"
_DBBENCH_SESSION_DATABASE_PATTERN = re.compile(rf"\b{_DBBENCH_SESSION_DATABASE_NAME_PATTERN}\b", re.IGNORECASE)
_DBBENCH_SESSION_DATABASE_QUALIFIER_PATTERN = re.compile(
    rf"(?:`{_DBBENCH_SESSION_DATABASE_NAME_PATTERN}`|\b{_DBBENCH_SESSION_DATABASE_NAME_PATTERN}\b)\s*\.",
    re.IGNORECASE,
)
_DBBENCH_SESSION_DATABASE_STRING_PATTERN = re.compile(
    rf"(?P<quote>['\"]){_DBBENCH_SESSION_DATABASE_NAME_PATTERN}(?P=quote)",
    re.IGNORECASE,
)


def _profile_enabled() -> bool:
    """Return whether AgentBench timing/profile log lines should be emitted."""
    value = os.getenv(_PROFILE_ENV_VAR, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _profile_log(event: str, **payload: Any) -> None:
    """Emit one structured profiling log event when profiling is enabled."""
    if not _profile_enabled():
        return
    LOGGER.info("AGENTBENCH_PROFILE %s", json.dumps({"event": event, **payload}, sort_keys=True, ensure_ascii=True))


class AgentBenchControllerError(RuntimeError):
    """Error raised when the AgentBench controller/runtime protocol fails."""

    pass


@dataclass(frozen=True)
class AgentBenchSample:
    """Reference to one AgentBench task sample."""

    controller_url: str
    task_name: str
    task_index: int | str

    @property
    def sample_id(self) -> str:
        """Return the stable task/index identifier used in caches and outputs."""
        return f"{self.task_name}:{self.task_index}"


@dataclass
class AgentBenchFunctionCallingSession:
    """Mutable function-calling session state returned by AgentBench."""

    sample: AgentBenchSample
    session_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    final_state: dict[str, Any] | None = None
    replay_mismatches: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AgentBenchSampledStep:
    """One realized step plus the candidate model samples considered at that step."""

    index: int
    assistant_message: dict[str, Any]
    sampled_messages: list[dict[str, Any]]
    sampled_actions: list[str]
    sampled_output_metadata: list[dict[str, Any]]
    chosen_output_index: int
    entropy: float
    observation_messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentBenchRollout:
    """A full rollout through an AgentBench task session."""

    prompt: str
    steps: list[AgentBenchSampledStep]
    final_answer: str | None
    status: str | None
    result: dict[str, Any] = field(default_factory=dict)
    raw_output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentBenchTaskExecution:
    """Single-trajectory execution result with normalized trajectory and scoring data."""

    sample: AgentBenchSample
    trajectory: BackboneTrajectory
    status: str | None
    result: dict[str, Any] = field(default_factory=dict)
    raw_output: dict[str, Any] | None = None
    error: str | None = None
    info: str | None = None


class AgentBenchControllerClient:
    """HTTP client and protocol helpers for AgentBench function-calling tasks.

    A wrapper around the AgentBench controller API that starts tasks, sends model tool calls, receives observations, and builds final results.

    """

    _DBBENCH_TOOLS: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Execute a single SQL query against the task database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SQL statement to execute.",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "commit_final_answer",
                "description": "Submit the final answer for evaluation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "The final answers to submit.",
                        }
                    },
                    "required": ["answers"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    def __init__(self, task_name: str, controller_url: str = "http://localhost:5020/api") -> None:
        """Store the task name and base URL for controller API requests."""
        self.task_name = task_name
        self.controller_url = controller_url.rstrip("/")

    """Which example IDs exist for this AgentBench task?
    The controller may return a list of integers or strings, depending on how the task is configured.
    Each number points to one task example.

    AgentBenchDBBench -> indices of database task examples
    AgentBenchWebShop -> indices of shopping task examples
    """
    def get_indices(self) -> list[int | str]:
        """Fetch available sample indices for the configured task."""
        payload = self._request(
            "/get_indices",
            method="GET",
            query={"name": self.task_name},
        )
        if not isinstance(payload, list):
            raise AgentBenchControllerError("AgentBench controller returned an invalid sample index list.")
        return [item for item in payload if isinstance(item, (int, str))]

    def start_function_calling_session(
        self,
        sample: AgentBenchSample,
    ) -> AgentBenchFunctionCallingSession:
        """Start one function-calling task session through the controller.

        Starts one AgentBench sample.

        It returns an AgentBenchFunctionCallingSession, which contains:

        session_id
        messages
        tools
        final_state

        The messages are the conversation so far. The tools are functions the model is allowed to call.

        `start_function_calling_session(...)` only **starts** a task session.

        It does **not** run multiple model steps.

        It creates the initial state:

        ```text
        system prompt
        user/task prompt
        available tools
        session id
        ```

        Example result:

        ```python
        session = start_function_calling_session(sample)
        ```

        Now `session` may contain:

        ```python
        session.messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "How many times did Alice sell a stock?"}
        ]

        session.tools = [
            {"function": {"name": "bash_action", ...}},
            {"function": {"name": "answer_action", ...}},
        ]
        ```

        At this point, no assistant action has happened yet.

        The actual step happens later with:

        ```python
        interact_function_calling(session, assistant_message)
        ```

        That sends **one assistant tool call** to the environment/controller.

        So the lifecycle is:

        ```text
        start_function_calling_session()
            -> create initial session, no model action yet

        model generates assistant_message
            -> one possible action

        interact_function_calling(session, assistant_message)
            -> execute that one action, update session

        model generates next assistant_message
        interact_function_calling(...)
            -> execute next one action
        ```

        Short answer:

        > `start_function_calling_session` runs zero model steps. It just initializes the task session.
        """
        result = self._request_with_worker_retry(
            "/start_sample",
            method="POST",
            payload={"name": sample.task_name, "index": sample.task_index},
            allowed_statuses={200, 406},
            return_headers=True,
            context=f"start_sample sample={sample.sample_id}",
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise AgentBenchControllerError("AgentBench start_sample did not return status, payload, and headers.")

        status_code, payload, response_headers = result
        if status_code == 406:
            raise AgentBenchControllerError(f"AgentBench sample {sample.sample_id} is not available: {payload}")
        if not isinstance(payload, Mapping):
            raise AgentBenchControllerError("AgentBench start_sample returned invalid JSON.")

        session_id = self._extract_session_id(payload, response_headers=response_headers)
        start_state = self._extract_protocol_state(payload)
        messages = self._extract_state_messages(start_state)
        tools = self._extract_state_tools(start_state)
        messages_required = str(sample.task_name) != "dbbench-std"
        if not session_id or (messages_required and not messages) or not tools:
            raise AgentBenchControllerError(
                self._format_start_sample_protocol_error(
                    sample=sample,
                    session_id=session_id,
                    messages=messages,
                    tools=tools,
                )
            )

        return AgentBenchFunctionCallingSession(
            sample=sample,
            session_id=str(session_id),
            messages=messages,
            tools=tools,
        )

    def _format_start_sample_protocol_error(
        self,
        *,
        sample: AgentBenchSample,
        session_id: str | None,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> str:
        """Describe which start_sample protocol fields were missing."""
        missing_fields: list[str] = []
        if not session_id:
            missing_fields.append("session_id")
        if not messages:
            missing_fields.append("messages")
        if not tools:
            missing_fields.append("tools")
        missing_suffix = f" missing {', '.join(missing_fields)}." if missing_fields else "."
        return (
            f"AgentBench {sample.task_name} start_sample did not return a usable FC session;"
            f" expected session_id, initial messages, and tools from /start_sample,{missing_suffix}"
        )

    def interact_function_calling(
        self,
        session: AgentBenchFunctionCallingSession,
        assistant_message: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Send one assistant tool call to AgentBench and recover if needed.

        This sends the model’s chosen action to AgentBench.

        Example assistant message:

        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "execute_sql",
                        "arguments": "{\"query\": \"SELECT * FROM users\"}"
                    }
                }
            ]
        }
        AgentBench executes it and returns observations.
        """
        normalized_assistant_message = dict(assistant_message)
        max_missing_session_recoveries = 3
        for attempt in range(max_missing_session_recoveries + 1):
            try:
                return self._interact_function_calling_once(session, normalized_assistant_message)
            except AgentBenchControllerError as exc:
                if not self._is_missing_session_error(exc) or attempt >= max_missing_session_recoveries:
                    raise
                try:
                    self._recover_missing_session(session)
                except AgentBenchControllerError as recovery_exc:
                    if not self._is_missing_session_error(recovery_exc) or attempt >= max_missing_session_recoveries:
                        raise
                    LOGGER.warning(
                        "AgentBench session recovery also lost session for sample=%s; retrying recovery (%s/%s)",
                        session.sample.sample_id,
                        attempt + 1,
                        max_missing_session_recoveries,
                    )
                    continue
        raise AgentBenchControllerError(
            f"AgentBench missing-session recovery loop exhausted for sample={session.sample.sample_id}."
        )

    def _interact_function_calling_once(
        self,
        session: AgentBenchFunctionCallingSession,
        assistant_message: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform a single /interact request and update the local session state."""
        interaction_response = self._request_with_worker_retry(
            "/interact",
            method="POST",
            payload={"messages": [dict(assistant_message)], "session_id": session.session_id},
            request_headers={"Session_id": session.session_id},
            context=f"interact sample={session.sample.sample_id} session={session.session_id}",
        )
        if not isinstance(interaction_response, Mapping):
            raise AgentBenchControllerError("AgentBench FC interact returned invalid JSON.")

        interaction_state = self._extract_protocol_state(interaction_response)
        returned_messages = self._extract_interaction_messages(
            interaction_state,
            prior_messages=session.messages,
            assistant_message=assistant_message,
        )
        session.messages.append(dict(assistant_message))
        session.messages.extend(returned_messages)
        session.final_state = dict(interaction_state)
        return returned_messages, dict(interaction_state)

    def _is_missing_session_error(self, exc: AgentBenchControllerError) -> bool:
        """Return whether a controller error indicates that session state was lost."""
        return "session not found" in str(exc).lower()

    def _recover_missing_session(self, session: AgentBenchFunctionCallingSession) -> None:
        """Restart a dropped session and replay prior assistant tool calls into it."""
        prior_messages = [dict(message) for message in session.messages]
        replay_messages = self._extract_replayable_assistant_messages(prior_messages)
        LOGGER.warning(
            "AgentBench controller dropped session_id=%s for sample=%s; restarting and replaying %s assistant turns",
            session.session_id,
            session.sample.sample_id,
            len(replay_messages),
        )
        restored_session = self.start_function_calling_session(session.sample)
        try:
            for replay_message in replay_messages:
                self._interact_function_calling_once(restored_session, replay_message)
        except Exception:
            try:
                self._cancel(restored_session.session_id, prefer_header=True)
            except Exception:
                pass
            raise

        session.session_id = restored_session.session_id
        session.messages = restored_session.messages
        session.tools = restored_session.tools
        session.final_state = restored_session.final_state

    def _extract_replayable_assistant_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Select assistant tool-call messages that can be replayed into a session."""
        replay_messages: list[dict[str, Any]] = []
        for message in messages:
            if str(message.get("role", "")) != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                continue
            replay_messages.append(dict(message))
        return replay_messages

    def run_sample(self, sample: AgentBenchSample, agent: Any) -> dict[str, Any]:
        """Run one AgentBench sample with a simple agent loop and return raw output.

        start session
        ask agent for tool call
        send tool call to controller
        repeat until finished
        return output
        """
        try:
            session = self.start_function_calling_session(sample)
        except AgentBenchControllerError as exc:
            return {"error": "NETWORK_ERROR", "info": str(exc), "output": None}

        try:
            for _ in range(64):
                try:
                    assistant_message = agent.inference_with_tools(session.messages, session.tools)
                except Exception as exc:
                    return {
                        "error": "AGENT_FAILED",
                        "info": str(exc),
                        "output": self._build_fc_output(sample, session.session_id, session.messages, session.final_state),
                    }

                try:
                    _, final_state = self.interact_function_calling(session, assistant_message)
                except AgentBenchControllerError as exc:
                    return {
                        "error": "NETWORK_ERROR",
                        "info": str(exc),
                        "output": self._build_fc_output(sample, session.session_id, session.messages, session.final_state),
                    }

                if bool(final_state.get("finish")):
                    break
            else:
                return {
                    "error": "INTERACT_FAILED",
                    "info": "AgentBench FC run exceeded the maximum number of interaction steps.",
                    "output": self._build_fc_output(sample, session.session_id, session.messages, session.final_state),
                }

            return {
                "error": None,
                "info": None,
                "output": self._build_fc_output(sample, session.session_id, session.messages, session.final_state),
            }
        finally:
            self._cancel(session.session_id, prefer_header=True)

    def _cancel(self, session_id: Any, *, prefer_header: bool = False) -> None:
        """Best-effort cancellation for a controller session."""
        if session_id is None:
            return
        attempts: list[tuple[dict[str, str] | None, dict[str, Any]]]
        attempts = [
            (None, {"session_id": session_id}),
            ({"Session_id": str(session_id)}, {}),
            ({"Session_id": str(session_id)}, {"session_id": session_id}),
        ]
        if prefer_header:
            attempts = [attempts[1], attempts[2], attempts[0]]

        for request_headers, payload in attempts:
            try:
                response = self._request(
                    "/cancel",
                    method="POST",
                    payload=payload,
                    request_headers=request_headers,
                    allowed_statuses={200, 400, 404},
                )
            except AgentBenchControllerError:
                continue

            status_code = response[0] if isinstance(response, tuple) else 200
            if status_code in {200, 404}:
                return

        LOGGER.debug("Ignoring AgentBench cancel failure for session_id=%s", session_id)

    def _request(
        self,
        path: str,
        *,
        method: str,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
        allowed_statuses: set[int] | None = None,
        request_headers: Mapping[str, Any] | None = None,
        return_headers: bool = False,
    ) -> Any:
        """Make one JSON HTTP request to the AgentBench controller."""
        url = f"{self.controller_url}{path}"
        if query:
            encoded_query = parse.urlencode({key: str(value) for key, value in query.items()})
            url = f"{url}?{encoded_query}"

        data: bytes | None = None
        headers: dict[str, str] = {str(key): str(value) for key, value in (request_headers or {}).items()}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(url, data=data, headers=headers, method=method.upper())
        expected_statuses = allowed_statuses or {200}
        start_time = time.perf_counter()
        try:
            with request.urlopen(req, timeout=120) as response:
                status_code = response.getcode()
                raw_body = response.read().decode("utf-8")
                response_headers = dict(response.headers.items())
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            _profile_log(
                "controller_request",
                path=path,
                method=method.upper(),
                status=exc.code,
                elapsed_ms=round((time.perf_counter() - start_time) * 1000, 3),
                expected=exc.code in expected_statuses,
            )
            if exc.code in expected_statuses:
                if return_headers:
                    return exc.code, self._decode_body(body), dict(exc.headers.items())
                return exc.code, self._decode_body(body)
            raise AgentBenchControllerError(f"HTTP {exc.code} from AgentBench controller: {body}") from exc
        except error.URLError as exc:
            _profile_log(
                "controller_request",
                path=path,
                method=method.upper(),
                status="url_error",
                elapsed_ms=round((time.perf_counter() - start_time) * 1000, 3),
                error=str(exc),
            )
            raise AgentBenchControllerError(f"Could not reach AgentBench controller at {url}: {exc}") from exc

        if status_code not in expected_statuses:
            _profile_log(
                "controller_request",
                path=path,
                method=method.upper(),
                status=status_code,
                elapsed_ms=round((time.perf_counter() - start_time) * 1000, 3),
                expected=False,
            )
            raise AgentBenchControllerError(f"Unexpected HTTP {status_code} from AgentBench controller: {raw_body}")
        _profile_log(
            "controller_request",
            path=path,
            method=method.upper(),
            status=status_code,
            elapsed_ms=round((time.perf_counter() - start_time) * 1000, 3),
            expected=True,
        )
        decoded = self._decode_body(raw_body)
        if return_headers:
            return status_code, decoded, response_headers
        if allowed_statuses:
            return status_code, decoded
        return decoded

    def _request_with_worker_retry(
        self,
        path: str,
        *,
        method: str,
        payload: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
        allowed_statuses: set[int] | None = None,
        request_headers: Mapping[str, Any] | None = None,
        return_headers: bool = False,
        context: str,
    ) -> Any:
        """Retry transient AgentBench worker-capacity errors before failing a branch."""
        retries = self._env_int(_WORKER_UNAVAILABLE_RETRIES_ENV_VAR, default=30)
        base_sleep = self._env_float(_WORKER_UNAVAILABLE_RETRY_BASE_SLEEP_ENV_VAR, default=0.5)
        max_sleep = self._env_float(_WORKER_UNAVAILABLE_RETRY_MAX_SLEEP_ENV_VAR, default=5.0)
        retries = max(0, retries)
        base_sleep = max(0.0, base_sleep)
        max_sleep = max(base_sleep, max_sleep)

        for attempt in range(retries + 1):
            try:
                return self._request(
                    path,
                    method=method,
                    payload=payload,
                    query=query,
                    allowed_statuses=allowed_statuses,
                    request_headers=request_headers,
                    return_headers=return_headers,
                )
            except AgentBenchControllerError as exc:
                if not self._is_worker_unavailable_error_message(str(exc)) or attempt >= retries:
                    raise
                sleep_seconds = min(max_sleep, base_sleep * (1.5 ** attempt))
                LOGGER.warning(
                    "AgentBench worker unavailable during %s; retrying in %.2fs (%s/%s)",
                    context,
                    sleep_seconds,
                    attempt + 1,
                    retries,
                )
                time.sleep(sleep_seconds)

        raise AgentBenchControllerError(f"AgentBench worker retry loop exhausted for {context}.")

    @staticmethod
    def _is_worker_unavailable_error_message(message: str) -> bool:
        lowered = message.lower()
        return "no workers available for task" in lowered or ("task " in lowered and " does not exist" in lowered)

    @staticmethod
    def _env_int(name: str, *, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default

    @staticmethod
    def _env_float(name: str, *, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError:
            return default

    def _build_fc_output(
        self,
        sample: AgentBenchSample,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        final_state: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Build a normalized function-calling output payload from session state."""
        answer = self._extract_final_answer(messages)
        reward = final_state.get("reward") if isinstance(final_state, Mapping) else None
        metrics = final_state.get("metrics") if isinstance(final_state, Mapping) else None
        score = metrics.get("score") if isinstance(metrics, Mapping) else None
        state_result = final_state.get("result") if isinstance(final_state, Mapping) else None
        result_payload = dict(state_result) if isinstance(state_result, Mapping) else {}
        result_payload.setdefault("answer", answer)
        if "is_correct" not in result_payload:
            result_payload["is_correct"] = bool(reward) if isinstance(reward, (int, float)) else None
        if "score" not in result_payload:
            result_payload["score"] = score
        return {
            "protocol": "agentbench-fc",
            "sample_id": sample.sample_id,
            "session_id": session_id,
            "status": final_state.get("status") if isinstance(final_state, Mapping) else None,
            "finish": bool(final_state.get("finish")) if isinstance(final_state, Mapping) else False,
            "reward": reward,
            "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
            "history": [dict(message) for message in messages],
            "result": result_payload,
        }

    def _iter_assistant_functions(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        reverse_tool_calls: bool = False,
    ):
        """Yield assistant tool-call dictionaries and their function payloads."""
        for message in reversed(messages):
            if str(message.get("role", "")) != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            ordered_tool_calls = reversed(tool_calls) if reverse_tool_calls else tool_calls
            for tool_call in ordered_tool_calls:
                if not isinstance(tool_call, Mapping):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, Mapping):
                    continue
                yield dict(tool_call), dict(function)

    def _decode_function_arguments(self, function: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Decode tool-call arguments, accepting either JSON strings or mappings."""
        arguments_text = function.get("arguments")
        if isinstance(arguments_text, Mapping):
            return dict(arguments_text)
        if not isinstance(arguments_text, str):
            return None
        try:
            arguments = json.loads(arguments_text)
        except json.JSONDecodeError:
            return None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        return dict(arguments) if isinstance(arguments, Mapping) else None

    def _extract_final_answer(self, messages: Sequence[Mapping[str, Any]]) -> str | None:
        """Extract the latest DBBench final answer from commit_final_answer calls."""
        for _, function in self._iter_assistant_functions(messages):
            if str(function.get("name", "")) != "commit_final_answer":
                continue
            arguments = self._decode_function_arguments(function)
            if arguments is None:
                continue
            answers = arguments.get("answers")
            if isinstance(answers, list) and answers:
                return self._coerce_text(answers[0])
        return None

    def _normalize_message_list(self, value: Any) -> list[dict[str, Any]]:
        """Return only mapping items from a value that should contain messages."""
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                normalized.append(dict(item))
        return normalized

    def _extract_protocol_state(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Unwrap controller payloads that nest state under an output field."""
        nested_output = payload.get("output")
        if isinstance(nested_output, Mapping):
            return dict(nested_output)
        return dict(payload)

    def _extract_state_messages(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Extract messages/history from a controller protocol state."""
        messages = self._normalize_message_list(state.get("messages"))
        if messages:
            return messages
        return self._normalize_message_list(state.get("history"))

    def _extract_state_tools(self, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Extract tool definitions, falling back to DBBench defaults when needed."""
        tools = state.get("tools")
        if isinstance(tools, list):
            normalized = [dict(tool) for tool in tools if isinstance(tool, Mapping)]
            if normalized:
                return normalized
        if self._extract_state_messages(state):
            return [dict(tool) for tool in self._DBBENCH_TOOLS]
        return []

    def _extract_session_id(
        self,
        payload: Mapping[str, Any],
        *,
        response_headers: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Find a session id in response headers, payload, or nested state."""
        header_value = self._lookup_header_value(response_headers, "session_id", "Session_id", "session-id")
        if header_value:
            return str(header_value)

        state = self._extract_protocol_state(payload)
        for candidate in (
            payload.get("session_id"),
            payload.get("Session_id"),
            payload.get("session-id"),
            state.get("session_id"),
            state.get("Session_id"),
            state.get("session-id"),
        ):
            if candidate is not None and str(candidate):
                return str(candidate)
        return None

    def _lookup_header_value(self, headers: Mapping[str, Any] | None, *names: str) -> Any:
        """Case/punctuation-insensitive lookup for HTTP header values."""
        if not isinstance(headers, Mapping):
            return None
        normalized_headers = {
            self._normalize_header_name(str(key)): value
            for key, value in headers.items()
            if value is not None
        }
        for name in names:
            candidate = normalized_headers.get(self._normalize_header_name(name))
            if candidate is not None:
                return candidate
        return None

    def _normalize_header_name(self, name: str) -> str:
        """Normalize header names to alphanumeric lowercase for comparison."""
        return "".join(character for character in name.lower() if character.isalnum())

    def _extract_interaction_messages(
        self,
        state: Mapping[str, Any],
        *,
        prior_messages: Sequence[Mapping[str, Any]],
        assistant_message: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Recover newly returned messages from several controller response shapes."""
        explicit_messages = self._normalize_message_list(state.get("messages"))
        if explicit_messages:
            return explicit_messages

        full_history = self._normalize_message_list(state.get("history"))
        if not full_history:
            return []

        prior_prefix = self._normalize_message_list(prior_messages)
        assistant_prefix = [dict(assistant_message)]
        if self._message_prefix_matches(full_history, [*prior_prefix, *assistant_prefix]):
            return full_history[len(prior_prefix) + 1 :]
        if self._message_prefix_matches(full_history, prior_prefix):
            trimmed = full_history[len(prior_prefix) :]
            if trimmed and self._messages_equal(trimmed[0], assistant_message):
                return trimmed[1:]
            return trimmed
        return full_history

    def _message_prefix_matches(
        self,
        messages: Sequence[Mapping[str, Any]],
        prefix: Sequence[Mapping[str, Any]],
    ) -> bool:
        """Return whether a normalized message sequence starts with a prefix."""
        if len(prefix) > len(messages):
            return False
        for index, expected in enumerate(prefix):
            if not self._messages_equal(messages[index], expected):
                return False
        return True

    def _messages_equal(self, left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        """Compare messages after recursive normalization."""
        return self._normalize_nested_mapping(dict(left)) == self._normalize_nested_mapping(dict(right))

    def _normalize_nested_mapping(self, value: Any) -> Any:
        """Recursively normalize mappings/lists/strings for stable comparison."""
        if isinstance(value, Mapping):
            return {str(key): self._normalize_nested_mapping(nested) for key, nested in sorted(value.items())}
        if isinstance(value, list):
            return [self._normalize_nested_mapping(item) for item in value]
        if isinstance(value, str):
            return value.replace("\r\n", "\n").replace("\r", "\n")
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)

    def _coerce_text(self, value: Any) -> str:
        """Convert scalar or structured values to deterministic text."""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else str(value)

    def _decode_body(self, body: str) -> Any:
        """Decode a controller response body as JSON when possible."""
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body


class AgentBenchModelAgent:
    """Adapter that exposes AgentBench-style inference on top of a chat model.

    AgentBenchModelAgent =
        wrapper around model_client
        that provides:
            inference()             for plain text
            inference_with_tools()  for tool calls

    In this project’s newer rollout code, _run_rollout_in_session usually calls the model client more
    directly through sampling helpers. But AgentBenchModelAgent is still useful for the simple run_sample(...) path.
    """

    def __init__(self, model_client: Any, *, temperature: float = 1.0) -> None:
        """Store the model client and sampling temperature for future calls."""
        self.model_client = model_client
        self.temperature = temperature

    def inference(self, history: Sequence[Mapping[str, Any]]) -> str:
        """Run a plain text model completion for legacy AgentBench interfaces.

        This is for plain text inference.

        It:

        Normalizes message roles.
        Calls either model_client.inference(...) or model_client.ainference(...).
        Returns a stripped string.
        So it asks the model:

        Given this conversation history, produce a text response.

        inference_with_tools(...)
        """
        messages = [self._normalize_message(message) for message in history]
        inference_callable = getattr(self.model_client, "inference", None)
        if callable(inference_callable):
            result = inference_callable(messages, temperature=self.temperature)
        else:
            async_inference = getattr(self.model_client, "ainference", None)
            if async_inference is None:
                raise TypeError("Configured model client does not expose inference or ainference.")
            result = asyncio.run(async_inference(messages, temperature=self.temperature))
        if not isinstance(result, str):
            result = str(result)
        return result.strip()

    def inference_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Run one tool-calling model completion and return the assistant message.

        This is for tool-calling inference.

        It:

        Copies the messages.
        Copies the tool definitions.
        Calls model_client.acompletion_with_tools(...).
        Returns the assistant message dictionary.
        So it asks the model:

        Given this conversation and these tools, produce an assistant message, probably with a tool call.

        Example output:

        {
            "role": "assistant",
            "content": "I should inspect the file.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "bash_action",
                        "arguments": "{\"script\": \"head /usr/stock.log\"}"
                    }
                }
            ]
        }

        You need `inference_with_tools(...)` when the model must respond by choosing a tool/function call, not just plain text.

        The model should respond by selecting one of the supplied task tools,
        rather than merely describing the action it intends to take.

        It needs to call a tool:

        ```python
        bash_action({"script": "grep -c '^Alice | Sell |' /usr/stock.log"})
        ```

        So the code calls:

        ```python
        assistant_message = agent.inference_with_tools(
            session.messages,
            session.tools,
        )
        ```

        Where:

        ```python
        session.messages
        ```

        contains the current conversation:

        ```python
        [
            {"role": "system", "content": "You are an assistant using Linux tools..."},
            {"role": "user", "content": "How many times did Alice sell a stock in /usr/stock.log?"}
        ]
        ```

        and:

        ```python
        session.tools
        ```

        contains the available tools:

        ```python
        [
            {
                "type": "function",
                "function": {
                    "name": "bash_action",
                    "description": "Execute bash code...",
                    "parameters": {...}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "answer_action",
                    "description": "Provide the answer...",
                    "parameters": {...}
                }
            }
        ]
        ```

        Then `inference_with_tools(...)` returns something like:

        ```python
        {
            "role": "assistant",
            "content": "I should count the matching lines.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash_action",
                        "arguments": "{\"script\": \"grep -c '^Alice | Sell |' /usr/stock.log\"}"
                    }
                }
            ]
        }
        ```

        That output can then be sent to:

        ```python
        interact_function_calling(session, assistant_message)
        ```

        which actually runs the bash command in the environment.

        Another example: DBBench.

        Available tools might be:

        ```text
        execute_sql
        commit_final_answer
        ```

        The model might return:

        ```python
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "execute_sql",
                        "arguments": "{\"query\": \"SELECT COUNT(*) FROM orders\"}"
                    }
                }
            ]
        }
        ```

        So use `inference_with_tools(...)` when:

        ```text
        the environment expects structured function/tool calls
        ```

        Use plain `inference(...)` when:

        ```text
        the model only needs to produce normal text
        ```
        """
        normalized_messages = [dict(message) for message in messages if isinstance(message, Mapping)]
        normalized_tools = [dict(tool) for tool in tools if isinstance(tool, Mapping)]
        tool_callable = getattr(self.model_client, "acompletion_with_tools", None)
        if tool_callable is None:
            raise TypeError("Configured model client does not support tool-calling completions.")
        result = tool_callable(normalized_messages, normalized_tools, temperature=self.temperature)
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        if not isinstance(result, Mapping):
            raise TypeError("Tool-calling model completion must return a message mapping.")
        return dict(result)

    def _normalize_message(self, message: Mapping[str, Any]) -> dict[str, str]:
        """Map AgentBench roles into chat-model roles and string content."""
        role = str(message.get("role", "user"))
        if role == "agent":
            role = "assistant"
        elif role not in {"system", "user", "assistant", "tool", "developer"}:
            role = "user"
        return {
            "role": role,
            "content": str(message.get("content", "")),
        }


class AgentBenchSingleTrajectoryExecutor(BranchingExecutor):
    """Executor that samples AgentBench trajectories and local branches.

    The engine that starts an AgentBench session, samples model actions, sends those
    actions to the environment, records the resulting trajectory, and supports local
    branching / uncertainty estimation.
    """

    name = "agentbench_dbbench"

    def __init__(
        self,
        controller_client: AgentBenchControllerClient, #  talks to AgentBench environment.
        model_client: Any, # calls the LLM.
        *,
        temperature: float = 1.0,
        backbone_per_step_samples: int = 4, # how many candidate actions to sample per backbone step.
        next_step_entropy_samples: int | None = None,  # how many candidates to sample when estimating next-step uncertainty.
        max_steps: int = 64, # maximum number of steps to sample in the backbone trajectory before stopping.
        # whether to raise an error if replaying prefix steps fails during session restoration;
        # set to False to allow silent recovery and continue sampling.
        strict_replay: bool = True,
        # whether to ask the controller for logprobs of sampled actions and include them in the trajectory metadata for use in uncertainty estimation.
        collect_sample_logprobs: bool = False,
        # optional shared storage for sampled candidates and their metadata to support advanced uncertainty
        # estimation techniques that require access to all sampled candidates (not just the chosen one) and/or
        # cross-step analysis; if None, a default in-memory storage will be used that supports basic use cases
        # but does not persist across different executor instances or runs.
        shared_sampling_storage: SharedSamplingStorage | None = None,
        model_signature: dict[str, Any] | None = None,
    ) -> None:
        """Configure controller/model access and shared sampling parameters."""
        if backbone_per_step_samples <= 0:
            raise ValueError("backbone_per_step_samples must be positive")
        resolved_next_step_entropy_samples = (
            backbone_per_step_samples if next_step_entropy_samples is None else next_step_entropy_samples
        )
        if resolved_next_step_entropy_samples <= 0:
            raise ValueError("next_step_entropy_samples must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        self.controller_client = controller_client
        self.model_client = model_client
        self.temperature = temperature
        self.backbone_per_step_samples = backbone_per_step_samples
        self.next_step_entropy_samples = resolved_next_step_entropy_samples
        self.max_steps = max_steps
        self.strict_replay = strict_replay
        self.collect_sample_logprobs = collect_sample_logprobs
        self.shared_sampling_storage = shared_sampling_storage
        self.shared_step_sampler = SharedStepSampler(shared_sampling_storage)
        self.model_signature = reusable_model_signature(model_signature or self._default_model_signature())
        self._history_state_cache: dict[str, dict[str, Any]] = {}

    async def sample_single_trajectory(self, sample: AgentBenchSample) -> AgentBenchTaskExecution:
        """Run the main trajectory once and wrap it as a task execution result."""
        try:
            trajectory = await self.sample_backbone(sample)
        except AgentBenchControllerError as exc:
            raise self._coerce_generation_failure(exc) from exc
        result = trajectory.metadata.get("result")
        resolved_result = dict(result) if isinstance(result, Mapping) else {}
        raw_output = self._backbone_raw_output(trajectory)
        return AgentBenchTaskExecution(
            sample=sample,
            trajectory=trajectory,
            status=self._coerce_optional_string(trajectory.metadata.get("status")),
            result=resolved_result,
            raw_output=raw_output,
            error=None,
            info=None,
        )

    def _coerce_generation_failure(self, error: AgentBenchControllerError) -> ModelGenerationError | AgentBenchControllerError:
        """Map known controller/runtime failures to normalized model errors."""
        if self._is_tool_call_sampling_failure(error):
            return ModelGenerationError(
                str(error),
                error_code=_TOOL_CALL_SAMPLING_ERROR_CODE,
                retryable=False,
                details={"source": "agentbench_tool_call_sampling"},
            )
        if self._is_worker_unavailable_failure(error):
            return ModelGenerationError(
                str(error),
                error_code="agentbench_worker_unavailable",
                retryable=True,
                details={"source": "agentbench_controller_capacity"},
            )
        if self._is_start_sample_protocol_failure(error):
            return ModelGenerationError(
                str(error),
                error_code="agentbench_start_sample_protocol",
                retryable=True,
                details={"source": "agentbench_controller_protocol"},
            )
        if "session not found" in str(error).lower():
            return ModelGenerationError(
                str(error),
                error_code="agentbench_session_not_found",
                retryable=True,
                details={"source": "agentbench_controller_session"},
            )
        if self._is_retryable_controller_failure(error):
            return ModelGenerationError(
                str(error),
                error_code="agentbench_controller_failure",
                retryable=True,
                details={"source": "agentbench_controller"},
            )
        return error

    @staticmethod
    def _is_tool_call_sampling_failure(error: AgentBenchControllerError) -> bool:
        """Return whether a failure came from unusable sampled tool calls."""
        message = str(error).lower()
        return (
            "valid agentbench tool-calling samples" in message
            or "no valid tool-calling samples were produced" in message
            or "tool call arguments must be a json object string" in message
            or "requires a string script argument" in message
        )

    @staticmethod
    def _is_worker_unavailable_failure(error: AgentBenchControllerError) -> bool:
        """Return whether the controller reports unavailable task workers."""
        message = str(error).lower()
        return "no workers available for task" in message or ("task " in message and " does not exist" in message)

    @staticmethod
    def _is_start_sample_protocol_failure(error: AgentBenchControllerError) -> bool:
        """Return whether start_sample omitted required session fields."""
        return "start_sample did not return a usable fc session" in str(error).lower()

    @staticmethod
    def _is_retryable_controller_failure(error: AgentBenchControllerError) -> bool:
        """Return whether a controller error is likely transient/retryable."""
        message = str(error).lower()
        return (
            "could not reach agentbench controller" in message
            or "failed to interact with session" in message
            or "http 500 from agentbench controller" in message
            or "http 502 from agentbench controller" in message
            or "http 503 from agentbench controller" in message
            or "http 504 from agentbench controller" in message
            or "unexpected http 500 from agentbench controller" in message
            or "unexpected http 502 from agentbench controller" in message
            or "unexpected http 503 from agentbench controller" in message
            or "unexpected http 504 from agentbench controller" in message
        )

    async def _start_function_calling_session(self, sample: AgentBenchSample) -> AgentBenchFunctionCallingSession:
        """Start a controller session and normalize start failures."""
        try:
            return await asyncio.to_thread(self.controller_client.start_function_calling_session, sample)
        except AgentBenchControllerError as exc:
            raise self._coerce_generation_failure(exc) from exc

    async def _restore_prefixed_session(
        self,
        sample: AgentBenchSample,
        prefix_steps: Sequence[StepRecord],
    ) -> AgentBenchFunctionCallingSession | None:
        """Optionally restore task state for a replay prefix; base executor cannot."""
        del sample, prefix_steps
        return None

    async def _start_session_with_prefix(
        self,
        sample: AgentBenchSample,
        prefix_steps: Sequence[StepRecord],
    ) -> AgentBenchFunctionCallingSession:
        """Start a fresh session, restoring or replaying prefix steps when provided."""
        restored_session = await self._restore_prefixed_session(sample, prefix_steps)
        if restored_session is not None:
            return restored_session
        session = await self._start_function_calling_session(sample)
        if prefix_steps:
            await self._replay_steps(session, prefix_steps)
        return session

    def _build_runtime_step_metadata(
        self,
        *,
        sample: AgentBenchSample,
        session: AgentBenchFunctionCallingSession,
        step_index: int,
    ) -> dict[str, Any]:
        """Collect executor-specific metadata for a realized runtime step."""
        del sample, session, step_index
        return {}

    async def sample_backbone(self, sample: Any) -> BackboneTrajectory:
        """Sample or load the primary trajectory used by uncertainty estimators."""
        cache_key = self._backbone_cache_key(sample)
        cached_backbone = self._load_cached_backbone(sample, cache_key)
        if cached_backbone is not None:
            return cached_backbone

        try:
            rollout = await self._sample_rollout(
                sample=sample,
                rollout_id="backbone",
                per_step_samples=self.backbone_per_step_samples,
            )
        except AgentBenchControllerError as exc:
            raise self._coerce_generation_failure(exc) from exc
        backbone = self._build_backbone_from_rollout(sample, rollout)
        self._store_cached_backbone(sample, cache_key, backbone)
        return backbone

    async def backbone_step_entropies(
        self,
        sample: Any,
        backbone: BackboneTrajectory,
    ) -> list[float]:
        """Read cached predictive entropy values from each backbone step."""
        del sample
        cached_entropies = backbone.metadata.get("step_entropies")
        if isinstance(cached_entropies, list) and all(isinstance(value, (int, float)) for value in cached_entropies):
            return [float(value) for value in cached_entropies]
        recovered: list[float] = []
        for step in backbone.steps:
            pe = step.metadata.get("pe")
            if not isinstance(pe, (int, float)):
                raise ValueError("Backbone trajectory is missing cached step entropies.")
            recovered.append(float(pe))
        return recovered

    async def sample_local_history(
        self,
        sample: Any,
        backbone: BackboneTrajectory,
        step_index: int,  # $t$
        window: int, # $L_t$
        branch_index: int,
    ) -> LocalBranchHistory:
        """Replay fixed prefix steps, sample a local branch, and cache its state.

        Args:
            sample: The AgentBench task instance being branched from. It provides
                the task name/index and stable sample id used for sessions and
                sampling caches.
            backbone: The already-sampled main trajectory. Its early steps are
                reused as a fixed prefix, and its recent steps define where this
                local branch is anchored.
            step_index: The 1-based target step whose next-step uncertainty will
                be estimated after constructing this local history.
            window: The local history width. Steps before ``step_index - window``
                are kept fixed from the backbone; steps inside the window may be
                resampled as the branch.
            branch_index: Which branch sample to use for the first branched step.
                Different branch indices use different sampling cursors so they
                can start from different candidate actions.

        Returns:
            A ``LocalBranchHistory`` containing:
                - ``prefix_steps``: fixed backbone steps before the branch window.
                - ``branched_steps``: newly sampled steps inside the local window.
                - ``window_start`` and ``step_index``: where the branch applies.
                - metadata with the branch id, sampling cursor, cached session
                  state key, and any tool-call/failure bookkeeping.
        """
        if step_index <= 0:
            raise ValueError("step_index must be positive")
        if window <= 0:
            raise ValueError("window must be positive")

        window_start = max(1, step_index - window) # $s_t = max(1, t-L_t)$
        fixed_prefix_count = window_start - 1 # $s_t - 1$
        prefix_steps = list(backbone.steps[:fixed_prefix_count]) # sub trajectory up to s_t-1
        branch_start_index = window_start # $s_t$
        branch_end_index = step_index - 1 # $t-1$
        #   The branch will cover steps from s_t to t-1, with the branch sampling cursor starting at b_t = L_t + i, where i is the branch index among the samples for this step.

        """
        `sample` is used to **create or restore** a session because it tells the controller which task example to open.

        It does not contain the conversation history itself. It is more like an address or key.

        Then the code can do:

        ```python
        session = await self._start_function_calling_session(sample)
        ```

        Inside that, the controller uses:

        ```python
        sample.task_name
        sample.task_index
        ```

        to start the correct task.

        For OS local runtime, this eventually does something like:

        ```python
        problem = self._load_problem_map(sample.task_name).get(int(sample.task_index))
        ```

        So:

        ```python
        task_name = "os-std"
        task_index = 0
        ```

        selects the correct OS problem.

        Then it creates a fresh session with initial messages:

        ```python
        session.messages = [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "How many times did Alice sell a stock...?"}
        ]
        ```

        So the flow is:

        ```text
        sample:
        task_name = os-std
        task_index = 0

        controller:
        uses task_name/task_index to load the task

        session:
        created from that task
        contains initial prompt/messages/tools/runtime
        ```

        In `sample_local_history`, the code calls:

        ```python
        session = await self._start_session_with_prefix(sample, prefix_steps)
        ```

        That function uses `sample` to start the correct task session, then uses `prefix_steps` to replay or restore the history.

        So:

        ```text
        sample -> opens the correct task
        prefix_steps -> restore/replay historical conversation
        session -> live state after that
        ```

        Analogy:

        ```text
        sample = book title + page number
        prefix_steps = notes about what you already read
        session = opened book plus your current position
        ```
        """
        session = await self._start_session_with_prefix(sample, prefix_steps)
        try:
            try:
                branched_steps: list[StepRecord] = []
                branch_rollout: AgentBenchRollout | None = None
                if branch_end_index >= branch_start_index and not self._session_finished(session):
                    branch_rollout = await self._run_rollout_in_session(
                        sample=sample,
                        session=session,
                        rollout_id=f"branch-{step_index}-{window}-{branch_index}",
                        per_step_samples=1,
                        start_step_index=branch_start_index,
                        stop_step_index=branch_end_index,
                        sampling_cursors={branch_start_index: self.backbone_per_step_samples + branch_index},
                    )
                    branched_steps = [self._build_backbone_step_record(step) for step in branch_rollout.steps]

                state_cache_key = self._cache_history_state(
                    sample=sample,
                    history_key=f"{step_index}:{window}:{branch_index}",
                    messages=session.messages,
                    tools=session.tools,
                    final_state=session.final_state,
                )
                return LocalBranchHistory(
                    step_index=step_index,
                    window_start=window_start,
                    prefix_steps=prefix_steps,
                    branched_steps=branched_steps,
                    metadata={
                        "executor": self.name,
                        "branch_index": branch_index,
                        "fixed_prefix_count": fixed_prefix_count,
                        "branch_rollout_id": branch_rollout.metadata.get("rollout_id") if branch_rollout is not None else None,
                        "branch_prompt": branch_rollout.prompt if branch_rollout is not None else None,
                        "branch_status": branch_rollout.status if branch_rollout is not None else None,
                        "branch_result": dict(branch_rollout.result) if branch_rollout is not None else None,
                        "branch_final_answer": branch_rollout.final_answer if branch_rollout is not None else None,
                        "branch_sampling_cursor": self.backbone_per_step_samples + branch_index,
                        "state_cache_key": state_cache_key,
                        "replay_mismatch_count": len(session.replay_mismatches),
                        "replay_mismatches": [dict(item) for item in session.replay_mismatches],
                        "replay_relaxed": bool(session.replay_mismatches),
                        "used_no_tool_call_samples": bool(branch_rollout.metadata.get("used_no_tool_call_samples")) if branch_rollout is not None else False,
                        "no_tool_call_sample_count": branch_rollout.metadata.get("no_tool_call_sample_count", 0) if branch_rollout is not None else 0,
                        "hard_requirement_failure": bool(branch_rollout.metadata.get("hard_requirement_failure")) if branch_rollout is not None else False,
                    },
                )
            except AgentBenchControllerError as exc:
                raise self._coerce_generation_failure(exc) from exc
        finally:
            await asyncio.to_thread(self.controller_client._cancel, session.session_id, prefer_header=True)

    async def estimate_next_step_entropy(
        self,
        sample: Any,
        history: LocalBranchHistory,
        step_index: int,
    ) -> float:
        """Sample candidate next actions from a local history and compute entropy.

        `estimate_next_step_entropy` estimates how uncertain the model is about what to
        do next from a given local history. It first checks whether the branch history already
        ended the task; if so, the next-step uncertainty is `0.0` because there is no next action
        to choose. Otherwise, it restores or loads the saved conversation/environment state for
        that history, asks the model to sample several possible next actions from that exact state,
        collects their output metadata such as logprobs, computes predictive entropy from those samples,
        saves the sampled actions and metadata back into the history object, and returns the entropy as a
        single number. Higher entropy means the model’s next action choices are more uncertain or spread out;
        lower entropy means the model is more confident or consistent.


        In this codebase, predictive entropy for a next action is computed by sampling several candidate model
        outputs from the same conversation state, reading each candidate’s `token_logprob_sum`, negating those
        logprob sums to get per-sample uncertainty contributions, and then averaging them into one scalar. For
        example, if four sampled candidates have logprob sums `-10`, `-12`, `-9`, and `-15`, their uncertainty
        contributions are `10`, `12`, `9`, and `15`, so the single predictive entropy is `(10 + 12 + 9 + 15) / 4 = 11.5`.
        Equivalently, the code computes `-sum(logprob_sums) / len(logprob_sums)`. This value summarizes how uncertain the
        model is about the next step from that exact conversation history: larger values mean the sampled candidate actions
        were lower probability on average, while smaller values mean the model assigned higher probability to its sampled
        candidates.

        Important: this function assumes ``history`` is positioned immediately
        before ``step_index``. In other words, ``history.prefix_steps`` plus
        ``history.branched_steps`` should represent the trajectory through
        ``step_index - 1``. The function does not verify that invariant from raw
        messages, so callers must keep the history and step index consistent.

        """
        if step_index <= 0:
            raise ValueError("step_index must be positive")

        realized_steps = history.prefix_steps + history.branched_steps
        if realized_steps:
            last_action = realized_steps[-1].action
            if isinstance(last_action, str) and self._is_finish_action(last_action):
                return 0.0

        # Restore the exact conversation/environment state represented by this
        # local history. The entropy estimate must be conditioned on this state,
        # not on the original backbone state or any other branch. This assumes
        # the restored history is the state immediately before step_index; no
        # explicit consistency check is performed here.
        state = await self._resolve_history_state(sample, history)
        final_state = state.get("final_state")
        if isinstance(final_state, Mapping) and bool(final_state.get("finish")):
            # If the restored state is already terminal, there is no next action
            # distribution to estimate, so its next-step entropy is zero.
            history.metadata["estimated_step_index"] = step_index
            history.metadata["estimated_entropy"] = 0.0
            history.metadata["estimated_sampled_actions"] = []
            history.metadata["estimated_sampled_messages"] = []
            history.metadata["estimated_sampled_output_metadata"] = []
            return 0.0

        try:
            # Sample several candidate next assistant actions from the restored
            # state. These candidates form the empirical set used to estimate
            # one predictive-entropy value for this step.
            sampled_step_samples = await self._sample_step_samples(
                sample=sample,
                # Pass the restored local-history conversation into the model.
                # It includes the initial prompt plus prior assistant actions
                # and tool/environment observations represented by
                # history.prefix_steps + history.branched_steps.
                messages=state["messages"],
                tools=state["tools"],
                per_step_samples=self.next_step_entropy_samples,
                step_index=step_index,
            )
        except AgentBenchControllerError as exc:
            raise self._coerce_generation_failure(exc) from exc
        # Clean empty outputs and retain metadata about non-tool-call samples so
        # the uncertainty record reflects what the model actually produced.
        prepared_step_samples = self._prepare_step_samples(
            sample_id=sample.sample_id,
            context_id="entropy-estimate",
            step_index=step_index,
            sampled_step_samples=sampled_step_samples,
        )
        sampling_metadata = self._summarize_step_sampling(prepared_step_samples)
        sampled_actions = [step_sample.output for step_sample in prepared_step_samples]
        # Each sample carries token logprob metadata. The predictive entropy
        # helper turns those per-sample logprob sums into one scalar by averaging
        # their negative log probabilities.
        sampled_output_metadata = [dict(step_sample.metadata) for step_sample in prepared_step_samples]
        entropy, _ = compute_predictive_entropy_from_metadata(sampled_output_metadata)
        resolved_entropy = 0.0 if entropy is None else float(entropy)
        history.metadata["estimated_step_index"] = step_index
        history.metadata["estimated_entropy"] = resolved_entropy
        history.metadata["estimated_sampled_actions"] = sampled_actions
        history.metadata["estimated_used_no_tool_call_samples"] = sampling_metadata["used_no_tool_call_samples"]
        history.metadata["estimated_no_tool_call_sample_count"] = sampling_metadata["no_tool_call_sample_count"]
        history.metadata["estimated_sampled_messages"] = [
            dict(step_sample.recovery_state.get("assistant_message", {}))
            for step_sample in prepared_step_samples
        ]
        history.metadata["estimated_sampled_output_metadata"] = sampled_output_metadata
        return resolved_entropy

    async def rollout_local_history_to_tdp(
        self,
        sample: Any,
        history: LocalBranchHistory,
        *,
        branch_index: int,
        rollout_horizon: int,
        per_step_samples: int = 1,
    ) -> TrajectoryDependentDecisionProcess:
        """Continue a local branch for a bounded horizon and return it as a TDP.

        ``sample_local_history(step_index=s + 1, window=1, ...)`` represents a
        local alternative for action ``s``. This method restores that branch
        state and then samples actions from ``s + 1`` onward, so adaptive local
        outcome estimators can bucket short branch futures instead of only
        estimating the next action's entropy.
        """
        if rollout_horizon <= 0:
            raise ValueError("rollout_horizon must be positive")
        if per_step_samples <= 0:
            raise ValueError("per_step_samples must be positive")

        realized_history_steps = list(history.prefix_steps) + list(history.branched_steps)
        terminal_history_tdp = self._terminal_local_history_tdp(
            sample=sample,
            history=history,
            realized_history_steps=realized_history_steps,
            branch_index=branch_index,
            rollout_horizon=rollout_horizon,
            per_step_samples=per_step_samples,
        )
        if terminal_history_tdp is not None:
            return terminal_history_tdp

        session = await self._start_session_with_prefix(sample, realized_history_steps)
        try:
            try:
                stop_step_index = min(self.max_steps, history.step_index + rollout_horizon - 1)
                rollout = await self._run_rollout_in_session(
                    sample=sample,
                    session=session,
                    rollout_id=f"adaptive-lb-oe-{history.step_index}-{branch_index}",
                    per_step_samples=per_step_samples,
                    start_step_index=history.step_index,
                    stop_step_index=stop_step_index,
                )
            except AgentBenchControllerError as exc:
                raise self._coerce_generation_failure(exc) from exc
        finally:
            await asyncio.to_thread(self.controller_client._cancel, session.session_id, prefer_header=True)

        prefix_tdp_steps = [self._build_tdp_step_record_from_step_record(step) for step in realized_history_steps]
        rollout_tdp_steps = [self._build_tdp_step_record(step) for step in rollout.steps]
        tdp = TrajectoryDependentDecisionProcess(
            sample_id=f"{sample.sample_id}-adaptive-lb-oe-{history.step_index}-{branch_index}",
            prompt=rollout.prompt,
            steps=[*prefix_tdp_steps, *rollout_tdp_steps],
            final_answer=rollout.final_answer,
            metadata={
                "executor": self.name,
                "task_name": sample.task_name,
                "task_index": sample.task_index,
                "status": rollout.status,
                "result": dict(rollout.result),
                "branch_index": branch_index,
                "branch_step_index": history.step_index - 1,
                "rollout_start_step_index": history.step_index,
                "rollout_horizon": rollout_horizon,
                "per_step_samples": per_step_samples,
                "local_history_metadata": dict(history.metadata),
                "used_no_tool_call_samples": bool(rollout.metadata.get("used_no_tool_call_samples"))
                or bool(history.metadata.get("used_no_tool_call_samples")),
                "no_tool_call_sample_count": self._coerce_nonnegative_int(
                    rollout.metadata.get("no_tool_call_sample_count")
                )
                + self._coerce_nonnegative_int(history.metadata.get("no_tool_call_sample_count")),
                "hard_requirement_failure": bool(rollout.metadata.get("hard_requirement_failure"))
                or bool(history.metadata.get("hard_requirement_failure")),
                "hard_requirement_failure_reason": rollout.metadata.get("hard_requirement_failure_reason"),
            },
        )
        if self.should_hard_finalize_tdp(sample, tdp):
            hard_final_answer = await self.hard_finalize_tdp(sample, tdp)
            tdp.metadata["hard_finalization"] = {
                "attempted": True,
                "answer": hard_final_answer,
                "source": "adaptive-lb-oe",
            }
            if isinstance(hard_final_answer, str) and hard_final_answer.strip():
                tdp.final_answer = hard_final_answer.strip()
        else:
            tdp.metadata["hard_finalization"] = {"attempted": False}
        return tdp

    def _terminal_local_history_tdp(
        self,
        *,
        sample: Any,
        history: LocalBranchHistory,
        realized_history_steps: Sequence[StepRecord],
        branch_index: int,
        rollout_horizon: int,
        per_step_samples: int,
    ) -> TrajectoryDependentDecisionProcess | None:
        """Return a TDP directly when the local branch already ended.

        Adaptive local-branch OE may sample a branch action that immediately
        submits an answer, or a branch action that fails because no callable
        tool call was produced. In both cases there is no live nonterminal
        environment state to restore and continue from. Treating that local
        history as a terminal short future preserves the outcome bucket and
        avoids requiring OS filesystem snapshots for terminal/no-tool steps.
        """
        if not realized_history_steps:
            return None
        last_step = realized_history_steps[-1]
        action = last_step.action
        branch_status = history.metadata.get("branch_status")
        hard_requirement_failure = bool(last_step.metadata.get("hard_requirement_failure")) or bool(
            history.metadata.get("hard_requirement_failure")
        )
        terminal_action = isinstance(action, str) and self._is_finish_action(action)
        terminal_status = isinstance(branch_status, str) and branch_status.strip() in {
            "completed",
            "generation_failed",
        }
        if not (hard_requirement_failure or terminal_action or terminal_status):
            return None

        branch_result = history.metadata.get("branch_result")
        resolved_result = dict(branch_result) if isinstance(branch_result, Mapping) else {}
        if hard_requirement_failure and "is_correct" not in resolved_result:
            resolved_result["is_correct"] = False
        resolved_status = (
            str(branch_status).strip()
            if isinstance(branch_status, str) and branch_status.strip()
            else "generation_failed"
            if hard_requirement_failure
            else "completed"
        )
        final_answer = history.metadata.get("branch_final_answer")
        if not isinstance(final_answer, str) or not final_answer.strip():
            history_messages = [
                dict(message)
                for step in realized_history_steps
                for message in step.messages
                if isinstance(message, Mapping)
            ]
            final_answer = self.controller_client._extract_final_answer(history_messages)

        return TrajectoryDependentDecisionProcess(
            sample_id=f"{sample.sample_id}-adaptive-lb-oe-{history.step_index}-{branch_index}",
            prompt=str(history.metadata.get("branch_prompt") or ""),
            steps=[self._build_tdp_step_record_from_step_record(step) for step in realized_history_steps],
            final_answer=final_answer.strip() if isinstance(final_answer, str) and final_answer.strip() else None,
            metadata={
                "executor": self.name,
                "task_name": sample.task_name,
                "task_index": sample.task_index,
                "status": resolved_status,
                "result": resolved_result,
                "branch_index": branch_index,
                "branch_step_index": history.step_index - 1,
                "rollout_start_step_index": history.step_index,
                "rollout_horizon": rollout_horizon,
                "per_step_samples": per_step_samples,
                "local_history_metadata": dict(history.metadata),
                "terminal_local_history": True,
                "used_no_tool_call_samples": bool(history.metadata.get("used_no_tool_call_samples")),
                "no_tool_call_sample_count": self._coerce_nonnegative_int(
                    history.metadata.get("no_tool_call_sample_count")
                ),
                "hard_requirement_failure": hard_requirement_failure,
                "hard_requirement_failure_reason": last_step.metadata.get("hard_requirement_failure_reason"),
                "hard_finalization": {"attempted": False},
            },
        )

    async def sample_tdp(
        self,
        sample: Any,
        trajectory_index: int,
        per_step_samples: int,
        *,
        include_counterfactuals: bool = False,
    ) -> TrajectoryDependentDecisionProcess:
        """Sample one trajectory-dependent decision process for baselines/UProp.

        Args:
            sample: The AgentBench task instance to run. It identifies the task
                name/index and is used in rollout ids, cache keys, and output
                metadata.
            trajectory_index: Which sampled trajectory to generate for this
                sample. ``0`` is treated as the backbone-compatible trajectory;
                larger values use shifted sampling cursors so they start from
                different candidate actions.
            per_step_samples: Number of candidate actions to sample at each
                decision step. These candidates are stored in each TDP step and
                used by baseline uncertainty methods.
            include_counterfactuals: Whether to also sample counterfactual
                target-step candidates after replacing earlier source decisions.
                This is more expensive and is mainly needed by methods that
                analyze cross-step decision effects.

        Returns:
            A ``TrajectoryDependentDecisionProcess`` containing the prompt,
            realized decision steps, sampled decisions and uncertainty
            measurements for each step, the final answer if one was produced,
            and metadata describing cache/replay settings and failure flags.
        """
        if per_step_samples <= 0:
            raise ValueError("per_step_samples must be positive")

        tdp_cache_key = self._tdp_cache_key(
            sample,
            trajectory_index=trajectory_index,
            include_counterfactuals=include_counterfactuals,
        )
        tdp_cache_key_fingerprint = sampling_fingerprint(
            {"schema_version": SharedSamplingStorage.schema_version, **tdp_cache_key}
        )
        cached_tdp: TrajectoryDependentDecisionProcess | None = None
        if self.shared_sampling_storage is not None:
            cached_tdp = self._load_cached_tdp(sample, tdp_cache_key)
            if cached_tdp is None:
                alternate_replay_key = copy.deepcopy(tdp_cache_key)
                executor_key = alternate_replay_key.get("executor")
                if isinstance(executor_key, dict):
                    executor_key["strict_replay"] = not self.strict_replay
                    replay_cached_tdp = self._load_cached_tdp(
                        sample,
                        alternate_replay_key,
                    )
                    if (
                        replay_cached_tdp is not None
                        and replay_cached_tdp.metadata.get(
                            "fixed_trajectory_candidate_extension_requested"
                        )
                        is True
                    ):
                        cached_tdp = replay_cached_tdp
                        cached_tdp.metadata[
                            "fixed_trajectory_replay_cache_fallback"
                        ] = True
            if cached_tdp is not None:
                cached_per_step_samples = cached_tdp.metadata.get("per_step_samples")
                if isinstance(cached_per_step_samples, int) and cached_per_step_samples >= per_step_samples:
                    cached_tdp.metadata["tdp_cache_status"] = "hit"
                    cached_tdp.metadata["tdp_cache_key"] = tdp_cache_key_fingerprint
                    cached_tdp.metadata["tdp_cache_requested_per_step_samples"] = per_step_samples
                    cached_tdp.metadata["tdp_cache_available_per_step_samples"] = cached_per_step_samples
                    return cached_tdp
                if isinstance(cached_per_step_samples, int) and cached_per_step_samples < per_step_samples:
                    if (
                        os.getenv(_REQUIRE_FROZEN_TDP_CACHE_ENV_VAR, "").strip().lower()
                        in {"1", "true", "yes", "on"}
                        and cached_tdp.metadata.get("fixed_trajectory_candidate_extension_requested") is not True
                    ):
                        raise AgentBenchControllerError(
                            "Strict frozen-pool execution refuses an unmarked cached TDP for "
                            f"sample={sample.sample_id} trajectory={trajectory_index}."
                        )
                    try:
                        extended_tdp = await self._extend_cached_tdp(
                            sample=sample,
                            trajectory_index=trajectory_index,
                            cached_tdp=cached_tdp,
                            per_step_samples=per_step_samples,
                            include_counterfactuals=include_counterfactuals,
                        )
                    except AgentBenchControllerError as exc:
                        if cached_tdp.metadata.get("fixed_trajectory_candidate_extension_requested") is True:
                            raise
                        LOGGER.warning(
                            "Ignoring cached TDP for sample=%s trajectory=%s because it could not be extended to %s per-step samples: %s",
                            sample.sample_id,
                            trajectory_index,
                            per_step_samples,
                            exc,
                        )
                    else:
                        extended_tdp.metadata["tdp_cache_status"] = "written"
                        extended_tdp.metadata["tdp_cache_lookup"] = "hit_extended"
                        extended_tdp.metadata["tdp_cache_key"] = tdp_cache_key_fingerprint
                        self._store_cached_tdp(sample, tdp_cache_key, extended_tdp)
                        return extended_tdp

        if os.getenv(_REQUIRE_FROZEN_TDP_CACHE_ENV_VAR, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise AgentBenchControllerError(
                "Strict frozen-pool execution refuses trajectory regeneration after a cache miss for "
                f"sample={sample.sample_id} trajectory={trajectory_index}."
            )

        backbone_cache_key = self._backbone_cache_key(sample)
        cached_backbone: BackboneTrajectory | None = None
        forced_sample_indices: dict[int, int] | None = None
        if trajectory_index == 0 and self.shared_sampling_storage is not None:
            cached_backbone = self._load_cached_backbone(sample, backbone_cache_key)
            if cached_backbone is not None:
                forced_sample_indices = self._backbone_sample_indices(cached_backbone)
        elif cached_tdp is not None:
            forced_sample_indices = self._tdp_sample_indices(cached_tdp)

        try:
            rollout = await self._sample_rollout(
                sample=sample,
                rollout_id=f"tdp-{trajectory_index}",
                per_step_samples=per_step_samples,
                sampling_cursors=({1: trajectory_index * per_step_samples} if trajectory_index > 0 else None),
                forced_sample_indices=forced_sample_indices,
            )
        except AgentBenchControllerError as exc:
            raise self._coerce_generation_failure(exc) from exc
        counterfactual_records = (
            await self._build_tdp_counterfactual_records(
                sample=sample,
                rollout=rollout,
                per_step_samples=per_step_samples,
            )
            if include_counterfactuals
            else [[] for _ in rollout.steps]
        )

        if (
            trajectory_index == 0
            and self.shared_sampling_storage is not None
            and cached_backbone is None
        ):
            self._store_cached_backbone(sample, backbone_cache_key, self._build_backbone_from_rollout(sample, rollout))

        tdp = TrajectoryDependentDecisionProcess(
            sample_id=f"{sample.sample_id}-tdp-{trajectory_index}",
            prompt=rollout.prompt,
            steps=[
                self._build_tdp_step_record(
                    step,
                    counterfactual_records=self._counterfactual_records_for_step(
                        counterfactual_records,
                        rollout.steps,
                        step,
                    ),
                )
                for step in rollout.steps
            ],
            final_answer=rollout.final_answer,
            metadata={
                "executor": self.name,
                "task_name": sample.task_name,
                "task_index": sample.task_index,
                "status": rollout.status,
                "result": dict(rollout.result),
                "trajectory_index": trajectory_index,
                "per_step_samples": per_step_samples,
                "include_counterfactuals": include_counterfactuals,
                "counterfactual_per_step_samples": per_step_samples if include_counterfactuals else 0,
                "used_no_tool_call_samples": bool(rollout.metadata.get("used_no_tool_call_samples")),
                "no_tool_call_sample_count": rollout.metadata.get("no_tool_call_sample_count", 0),
                "hard_requirement_failure": bool(rollout.metadata.get("hard_requirement_failure")),
                "hard_requirement_failure_reason": rollout.metadata.get("hard_requirement_failure_reason"),
                "tdp_cache_status": "written" if self.shared_sampling_storage is not None else "disabled",
                "tdp_cache_lookup": "miss" if self.shared_sampling_storage is not None else "disabled",
                "tdp_cache_key": tdp_cache_key_fingerprint if self.shared_sampling_storage is not None else None,
            },
        )
        if self.shared_sampling_storage is not None:
            self._store_cached_tdp(sample, tdp_cache_key, tdp)
        return tdp

    async def hard_finalize_tdp(
        self,
        sample: Any,
        tdp: TrajectoryDependentDecisionProcess,
    ) -> str | None:
        """Ask the backbone for a validated JSON final answer from an unfinished trajectory."""
        if isinstance(tdp.final_answer, str) and tdp.final_answer.strip():
            return tdp.final_answer.strip()

        rejected_attempts: list[dict[str, str]] = []
        tool_answer = await self._hard_finalize_with_answer_tool(sample, tdp, rejected_attempts)
        if tool_answer:
            if rejected_attempts:
                tdp.metadata["hard_finalization_rejected_attempts"] = rejected_attempts
            return tool_answer

        retry_reason: str | None = None
        previous_output: str | None = None
        for _attempt in range(2):
            messages = self._hard_finalization_messages(
                sample,
                tdp,
                retry_reason=retry_reason,
                previous_output=previous_output,
            )
            raw_output, output_metadata = await self._hard_finalization_text_completion(messages)
            if raw_output is None:
                return None
            if output_metadata.get("truncated") is True:
                invalid_reason = "text finalization was truncated"
                retry_reason = invalid_reason
                previous_output = raw_output
                rejected_attempts.append({"output": raw_output, "reason": invalid_reason})
                continue

            answer, invalid_reason = self._parse_hard_finalization_answer(raw_output)
            if answer:
                task_invalid_reason = self._validate_hard_finalization_answer_against_task(answer, sample, tdp)
                if task_invalid_reason is not None:
                    retry_reason = task_invalid_reason
                    previous_output = raw_output
                    rejected_attempts.append({"output": raw_output, "reason": task_invalid_reason})
                    continue
                if rejected_attempts:
                    tdp.metadata["hard_finalization_rejected_attempts"] = rejected_attempts
                tdp.metadata["hard_finalization_raw_output"] = raw_output
                return answer

            retry_reason = invalid_reason
            previous_output = raw_output
            rejected_attempts.append({"output": raw_output, "reason": invalid_reason})

        tdp.metadata["hard_finalization_rejected_attempts"] = rejected_attempts
        return None

    async def _hard_finalization_text_completion(self, messages: list[dict[str, str]]) -> tuple[str | None, dict[str, Any]]:
        """Run a text finalization completion, preserving truncation metadata when available."""
        detailed_callable = getattr(self.model_client, "sample_many_detailed", None)
        if detailed_callable is not None:
            result = detailed_callable(messages, temperature=0.0, n=1)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)) and result:
                first = result[0]
                if isinstance(first, Mapping):
                    output = first.get("output")
                    metadata = first.get("metadata")
                    return str(output) if output is not None else "", dict(metadata) if isinstance(metadata, Mapping) else {}

        inference_callable = getattr(self.model_client, "ainference", None)
        if inference_callable is not None:
            result = inference_callable(messages, temperature=0.0)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result), {}

        inference_callable = getattr(self.model_client, "inference", None)
        if inference_callable is None:
            return None, {}
        result = await asyncio.to_thread(inference_callable, messages, 0.0)
        return str(result), {}

    async def _hard_finalize_with_answer_tool(
        self,
        sample: Any,
        tdp: TrajectoryDependentDecisionProcess,
        rejected_attempts: list[dict[str, str]],
    ) -> str | None:
        """Prefer a structured final-answer tool call when the model client supports tools."""
        completion_callable = getattr(self.model_client, "acompletion_with_tools_detailed", None)
        if completion_callable is None:
            completion_callable = getattr(self.model_client, "acompletion_with_tools", None)
        if completion_callable is None:
            return None

        messages = self._hard_finalization_messages(sample, tdp)
        messages[-1]["content"] += "\n\nUse the submit_final_answer tool exactly once."
        tool_choice = self._hard_finalization_tool_choice()
        try:
            result = completion_callable(
                messages,
                [self._hard_finalization_answer_tool()],
                temperature=0.0,
                tool_choice=tool_choice,
            )
            if asyncio.iscoroutine(result):
                result = await result
        except TypeError:
            fallback_callable = getattr(self.model_client, "acompletion_with_tools", None)
            if fallback_callable is None:
                return None
            result = fallback_callable(messages, [self._hard_finalization_answer_tool()], temperature=0.0)
            if asyncio.iscoroutine(result):
                result = await result
        except ModelGenerationError as exc:
            rejected_attempts.append(
                {
                    "output": json.dumps(tool_choice, sort_keys=True, ensure_ascii=True),
                    "reason": f"forced tool finalization failed: {exc}",
                }
            )
            result = completion_callable(
                messages,
                [self._hard_finalization_answer_tool()],
                temperature=0.0,
            )
            if asyncio.iscoroutine(result):
                result = await result

        if self._payload_is_truncated(result):
            rejected_attempts.append(
                {"output": json.dumps(result, sort_keys=True, ensure_ascii=True), "reason": "tool finalization was truncated"}
            )
            return None

        answer, invalid_reason = self._extract_hard_finalization_tool_answer(result)
        if answer:
            task_invalid_reason = self._validate_hard_finalization_answer_against_task(answer, sample, tdp)
            if task_invalid_reason is not None:
                rejected_attempts.append(
                    {"output": json.dumps(result, sort_keys=True, ensure_ascii=True), "reason": task_invalid_reason}
                )
                return None
            tdp.metadata["hard_finalization_tool_output"] = result
            return answer

        content_answer = self._extract_hard_finalization_answer_from_content(result, sample, tdp)
        if content_answer:
            task_invalid_reason = self._validate_hard_finalization_answer_against_task(content_answer, sample, tdp)
            if task_invalid_reason is not None:
                rejected_attempts.append(
                    {"output": json.dumps(result, sort_keys=True, ensure_ascii=True), "reason": task_invalid_reason}
                )
                return None
            tdp.metadata["hard_finalization_tool_output"] = result
            tdp.metadata["hard_finalization_content_extracted"] = True
            return content_answer

        rejected_attempts.append({"output": json.dumps(result, sort_keys=True, ensure_ascii=True), "reason": invalid_reason})
        return None

    def _payload_is_truncated(self, payload: Any) -> bool:
        """Return whether a model payload was cut off by the output token limit."""
        if not isinstance(payload, Mapping):
            return False
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("truncated") is True:
            return True
        return payload.get("finish_reason") == "length"

    def _extract_hard_finalization_answer_from_content(
        self,
        payload: Any,
        sample: Any,
        tdp: TrajectoryDependentDecisionProcess,
    ) -> str | None:
        """Optionally recover an answer when a model ignores the forced tool call."""
        del payload, sample, tdp
        return None

    @staticmethod
    def _hard_finalization_tool_choice() -> dict[str, Any]:
        """Return the OpenAI-compatible directive that forces the final-answer tool."""
        return {"type": "function", "function": {"name": "submit_final_answer"}}

    @staticmethod
    def _hard_finalization_answer_tool() -> dict[str, Any]:
        """Return the strict tool schema used to collect hard-finalized answers."""
        return {
            "type": "function",
            "function": {
                "name": "submit_final_answer",
                "description": "Submit the short final answer inferred from the partial trajectory.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The exact final answer to submit, with no explanation.",
                        }
                    },
                    "required": ["answer"],
                },
            },
        }

    def _extract_hard_finalization_tool_answer(self, payload: Any) -> tuple[str | None, str]:
        """Extract and validate the answer field from a finalization tool call payload."""
        if not isinstance(payload, Mapping):
            return None, "tool finalization returned a non-object payload"
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            return None, "tool finalization did not call submit_final_answer"
        function = tool_calls[0].get("function") if isinstance(tool_calls[0], Mapping) else None
        if not isinstance(function, Mapping):
            return None, "tool call did not contain function arguments"
        if function.get("name") != "submit_final_answer":
            return None, "tool finalization called the wrong function"
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            return None, "tool finalization arguments were not a JSON string"
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return None, f"tool finalization arguments were not valid JSON: {exc.msg}"
        if not isinstance(parsed, Mapping):
            return None, "tool finalization arguments were not a JSON object"
        raw_answer = parsed.get("answer")
        if not isinstance(raw_answer, str):
            return None, "tool finalization did not contain a string answer field"
        answer = self._normalize_final_answer(raw_answer.strip()) or ""
        invalid_reason = self._validate_hard_finalization_answer(answer)
        if invalid_reason is not None:
            return None, invalid_reason
        return answer, ""

    def _hard_finalization_messages(
        self,
        sample: Any,
        tdp: TrajectoryDependentDecisionProcess,
        *,
        retry_reason: str | None = None,
        previous_output: str | None = None,
    ) -> list[dict[str, str]]:
        """Build a compact evidence-only prompt for hard-finalizing an unfinished trajectory."""
        prompt = tdp.prompt or getattr(sample, "question", None) or getattr(sample, "sample_id", "")
        transcript_lines = [f"Task prompt:\n{prompt}".strip(), "", "Observed command outputs:"]
        for step in tdp.steps:
            observation = step.metadata.get("observation")
            if isinstance(observation, str) and observation:
                transcript_lines.append(f"Step {step.index}:")
                transcript_lines.append(self._compact_hard_finalization_text(observation, limit=500))
        if len(transcript_lines) == 3:
            transcript_lines.append("No observations were recorded.")

        user_prompt = (
            "\n".join(transcript_lines).strip()
            + "\n\nThe trajectory stopped before an explicit final answer. "
            "Infer the answer that should be submitted now from the observed command outputs only. "
            "Do not continue debugging the task. Do not explain the trajectory. "
            "If there are conflicting observations, choose the concise answer most directly supported by command output. "
            'Return exactly one JSON object in this format: {"answer":"<short final answer>"}. '
            "Do not include markdown, reasoning, tool calls, or any text outside the JSON object. "
            "The answer value must be the submitted answer itself, not a description of the task."
        )
        if retry_reason is not None:
            user_prompt += (
                "\n\nYour previous output was rejected because: "
                f"{retry_reason}. Previous output: {previous_output!r}. "
                'Return only valid JSON now, for example {"answer":"108"} or {"answer":"/tmp/result.txt"}.'
            )

        return [
            {
                "role": "system",
                "content": (
                    "You are a strict final-answer extractor for partial environment trajectories. "
                    "Your job is extraction, not reasoning aloud or continuing the task. "
                    'You must output exactly one JSON object: {"answer":"..."}'
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    def _parse_hard_finalization_answer(self, output: str) -> tuple[str | None, str]:
        """Parse and validate the strict JSON object returned by hard finalization."""
        candidate = self._strip_json_fence(output).strip()
        parsed: Any
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None, "output was not exactly one valid JSON object"

        if not isinstance(parsed, dict):
            return None, "JSON value was not an object"
        raw_answer = parsed.get("answer")
        if not isinstance(raw_answer, str):
            return None, "JSON object did not contain a string answer field"

        answer = self._normalize_final_answer(raw_answer.strip()) or ""
        invalid_reason = self._validate_hard_finalization_answer(answer)
        if invalid_reason is not None:
            return None, invalid_reason
        return answer, ""

    @staticmethod
    def _strip_json_fence(output: str) -> str:
        """Remove a single markdown code fence around a model JSON response."""
        stripped = output.strip()
        fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            return fence_match.group(1)
        return stripped

    @staticmethod
    def _validate_hard_finalization_answer(answer: str) -> str | None:
        """Return an invalidity reason when the answer looks explanatory instead of final."""
        if not answer:
            return "answer was empty"
        if "\n" in answer or "\r" in answer:
            return "answer contained multiple lines"
        if len(answer) > 200:
            return "answer was too long to be a final answer"
        lowered = answer.lower()
        explanatory_prefixes = (
            "the user",
            "based on",
            "let me",
            "i ",
            "we ",
            "to determine",
            "looking at",
            "the trajectory",
            "the task",
            "step ",
        )
        if lowered.startswith(explanatory_prefixes):
            return "answer looked like reasoning or task description"
        if any(marker in answer for marker in ("```", "**", "Thought:", "Action:", "Observation:")):
            return "answer contained formatting or trajectory text"
        word_count = len(answer.split())
        path_like = answer.startswith(("/", "./", "../", "~")) or "\\" in answer
        numeric_like = bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer))
        if word_count > 12 and not path_like and not numeric_like:
            return "answer was too verbose to be a final answer"
        return None

    def _validate_hard_finalization_answer_against_task(
        self,
        answer: str,
        sample: Any,
        tdp: TrajectoryDependentDecisionProcess,
    ) -> str | None:
        """Conservatively reject hard-finalized answers that do not match the task shape or evidence.

        This validation is intentionally applied only to hard-finalized unfinished
        trajectories. Normal AgentBench answer_action outputs are left unchanged.
        If the task shape is unclear, the validator falls back to evidence checks
        instead of trying to repair or reinterpret the model's answer.
        """
        prompt = str(tdp.prompt or getattr(sample, "question", "") or "")
        prompt_lower = prompt.lower()
        answer = answer.strip()
        answer_lower = answer.lower()
        observations = self._hard_finalization_observation_text(tdp)

        expects_number = self._prompt_expects_number(prompt_lower)
        expects_path = self._prompt_expects_path(prompt_lower)
        expects_list_or_entity = self._prompt_expects_list_or_entity(prompt_lower)

        if expects_number and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer) is None:
            return "task appears to require a numeric answer, but hard-finalized answer was not numeric"

        if expects_path and not answer.startswith("/"):
            return "task appears to require a full path, but hard-finalized answer was not an absolute path"

        if expects_list_or_entity and not (expects_number or expects_path):
            if len(answer.split()) > 8:
                return "entity/list task hard-finalized answer was too verbose for a conservative answer"

        if observations and not self._answer_is_supported_by_observation(answer, observations):
            return "hard-finalized answer was not directly supported by observed command output"

        return None

    @staticmethod
    def _prompt_expects_number(prompt_lower: str) -> bool:
        """Return whether the task wording suggests a compact numeric answer."""
        number_markers = (
            "how many",
            "count ",
            "count the",
            "total number",
            "number of",
            "sum ",
            "summation",
            "average",
            "maximum",
            "minimum",
            "largest",
            "smallest",
            "size of",
        )
        return any(marker in prompt_lower for marker in number_markers)

    @staticmethod
    def _prompt_expects_path(prompt_lower: str) -> bool:
        """Return whether the task wording suggests an absolute filesystem path answer."""
        path_markers = (
            "full path",
            "absolute path",
            "path of",
            "path to",
            "directory",
            "folder",
            "where is",
        )
        return any(marker in prompt_lower for marker in path_markers)

    @staticmethod
    def _prompt_expects_list_or_entity(prompt_lower: str) -> bool:
        """Return whether the task wording asks for named entities or a ranked/list answer."""
        entity_markers = (
            "identify",
            "which ",
            "who ",
            "most active",
            "least active",
            "buyers/sellers",
            "trader",
            "traders",
        )
        return any(marker in prompt_lower for marker in entity_markers)

    @staticmethod
    def _hard_finalization_observation_text(tdp: TrajectoryDependentDecisionProcess) -> str:
        """Join observations from the last few steps as the evidence available to finalization."""
        observations: list[str] = []
        for step in tdp.steps[-5:]:
            observation = step.metadata.get("observation")
            if isinstance(observation, str) and observation.strip():
                observations.append(observation)
        return "\n".join(observations)

    @staticmethod
    def _answer_is_supported_by_observation(answer: str, observations: str) -> bool:
        """Return whether the exact answer appears as a standalone observed value."""
        normalized_answer = " ".join(answer.split())
        normalized_observations = " ".join(observations.split())
        if not normalized_answer:
            return False

        if answer.startswith("/"):
            path_pattern = rf"(?<!\S){re.escape(answer)}(?:/)?(?!\S)"
            return re.search(path_pattern, normalized_observations) is not None

        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", answer):
            numeric_pattern = rf"(?<![\w.]){re.escape(answer)}(?![\w.])"
            return re.search(numeric_pattern, normalized_observations) is not None

        return normalized_answer.lower() in normalized_observations.lower()

    @staticmethod
    def _compact_hard_finalization_text(text: str, *, limit: int) -> str:
        """Keep finalization transcripts short enough for small local model contexts."""
        compacted = " ".join(text.split())
        if len(compacted) <= limit:
            return compacted
        half = max(1, (limit - 20) // 2)
        return f"{compacted[:half]} ... {compacted[-half:]}"

    def _default_model_signature(self) -> dict[str, Any]:
        """Build cache identity fields from the configured model client."""
        return {
            "client_type": f"{type(self.model_client).__module__}.{type(self.model_client).__qualname__}",
            "model": getattr(self.model_client, "model", None),
        }

    def _executor_signature(self) -> dict[str, Any]:
        """Build cache identity fields for executor behavior that affects samples."""
        return {
            "executor": self.name,
            "replay_version": _DBBENCH_REPLAY_VERSION,
            "sample_temperature": self.temperature,
            "backbone_per_step_samples": self.backbone_per_step_samples,
            "next_step_entropy_samples": self.next_step_entropy_samples,
            "max_steps": self.max_steps,
            "strict_replay": self.strict_replay,
        }

    def _sample_signature(self, sample: AgentBenchSample) -> dict[str, Any]:
        """Build cache identity fields for one AgentBench sample."""
        return {
            "sample_id": sample.sample_id,
            "task_name": sample.task_name,
            "task_index": sample.task_index,
        }

    def _backbone_cache_key(self, sample: AgentBenchSample) -> dict[str, Any]:
        """Build the shared-sampling key for a cached backbone trajectory."""
        key = {
            "kind": "backbone",
            "sample": self._sample_signature(sample),
            "model": self.model_signature,
            "executor": {
                "executor": self.name,
                "replay_version": _DBBENCH_REPLAY_VERSION,
                "max_steps": self.max_steps,
                "sample_temperature": self.temperature,
                "strict_replay": self.strict_replay,
            },
        }
        if self.name == "agentbench_dbbench":
            key["executor"]["dbbench_state_bucket_version"] = _DBBENCH_STATE_BUCKET_VERSION
        return key

    def _tdp_cache_key(
        self,
        sample: AgentBenchSample,
        *,
        trajectory_index: int,
        include_counterfactuals: bool = False,
    ) -> dict[str, Any]:
        """Build the shared-sampling key for a cached TDP trajectory."""
        key: dict[str, Any] = {
            "kind": "tdp",
            "sample": self._sample_signature(sample),
            "model": self.model_signature,
            "executor": {
                "executor": self.name,
                "replay_version": _DBBENCH_REPLAY_VERSION,
                "max_steps": self.max_steps,
                "sample_temperature": self.temperature,
                "strict_replay": self.strict_replay,
            },
            "trajectory_index": trajectory_index,
            "include_counterfactuals": include_counterfactuals,
        }
        if self.name == "agentbench_dbbench":
            key["executor"]["dbbench_state_bucket_version"] = _DBBENCH_STATE_BUCKET_VERSION
        return key

    def _tdp_sample_indices(self, tdp: TrajectoryDependentDecisionProcess) -> dict[int, int]:
        """Recover absolute sampled-action indices chosen by a cached TDP."""
        cached_per_step_samples = tdp.metadata.get("per_step_samples")
        trajectory_index = tdp.metadata.get("trajectory_index")
        if not isinstance(cached_per_step_samples, int) or cached_per_step_samples <= 0:
            cached_per_step_samples = 0
        if not isinstance(trajectory_index, int) or trajectory_index < 0:
            trajectory_index = 0
        sample_indices: dict[int, int] = {}
        for step in tdp.steps:
            chosen_output_index = step.metadata.get("chosen_output_index")
            if not isinstance(chosen_output_index, int) or chosen_output_index < 0:
                continue
            if step.index == 1 and trajectory_index > 0 and cached_per_step_samples > 0:
                sample_indices[step.index] = trajectory_index * cached_per_step_samples + chosen_output_index
            else:
                sample_indices[step.index] = chosen_output_index
        return sample_indices

    def _step_sampling_cache_key(
        self,
        sample: AgentBenchSample,
        *,
        step_index: int,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the shared-sampling key for candidate next-step samples."""
        return {
            "kind": "step_samples",
            "step_sampling_version": _DBBENCH_STEP_SAMPLING_VERSION,
            "sample": self._sample_signature(sample),
            "model": self.model_signature,
            "executor": self._executor_signature(),
            "step_index": step_index,
            "messages": self._cache_key_messages(messages),
            "tools": self._cache_key_tools(tools),
        }

    def _step_sampling_pool_id(self) -> str:
        """Return the shared pool name for equivalent step-sampling contexts."""
        return "shared-context-pool"

    async def _sample_rollout(
        self,
        *,
        sample: AgentBenchSample,
        rollout_id: str,
        per_step_samples: int,
        replay_steps: Sequence[StepRecord] | None = None,
        start_step_index: int = 1,
        stop_step_index: int | None = None,
        sampling_cursors: dict[int, int] | None = None,
        forced_sample_indices: dict[int, int] | None = None,
    ) -> AgentBenchRollout:
        """Start a session, run a rollout, and always cancel the session afterward."""
        session = await self._start_session_with_prefix(sample, replay_steps or ())
        try:
            return await self._run_rollout_in_session(
                sample=sample,
                session=session,
                rollout_id=rollout_id,
                per_step_samples=per_step_samples,
                start_step_index=start_step_index,
                stop_step_index=stop_step_index,
                sampling_cursors=sampling_cursors,
                forced_sample_indices=forced_sample_indices,
            )
        finally:
            await asyncio.to_thread(self.controller_client._cancel, session.session_id, prefer_header=True)

    async def _run_rollout_in_session(
        self,
        *,
        sample: AgentBenchSample,
        session: AgentBenchFunctionCallingSession,
        rollout_id: str,
        per_step_samples: int,
        start_step_index: int = 1,
        stop_step_index: int | None = None,
        sampling_cursors: dict[int, int] | None = None,
        forced_sample_indices: dict[int, int] | None = None,
    ) -> AgentBenchRollout:
        """Advance an existing session by repeatedly sampling and applying actions.

        Args:
            sample: The AgentBench task instance being solved, for example
                ``os-std:0`` or a DBBench sample. It provides the task name,
                task index, and stable cache/sample identifiers.
            session: The live function-calling session to advance. It contains
                current messages, available tools, a session id, and any final
                state already returned by the controller/runtime.
            rollout_id: A stable label for this rollout, such as ``backbone``,
                ``branch-3-2-0``, or ``tdp-1``. It is used in metadata and as
                part of deterministic sampling behavior.
            per_step_samples: Number of candidate assistant actions to sample at
                each step before choosing one realized action.
            start_step_index: The 1-based step number where this rollout begins.
                Branch rollouts may start after a replayed prefix. The caller is
                responsible for making this consistent with ``session``: the
                session should already contain all completed action steps before
                ``start_step_index``. This function does not verify that
                invariant from raw messages.
            stop_step_index: Optional final 1-based step number for this rollout.
                When omitted, the rollout can continue until ``self.max_steps``.
            sampling_cursors: Optional map from step index to sample-pool offset.
                This lets local branches skip samples already used elsewhere, so
                each branch can start from a different candidate action.
            forced_sample_indices: Optional map from step index to absolute sample
                index that must be realized. This is used when replaying cached
                trajectories so the same sampled action is chosen again.
        """
        if per_step_samples <= 0:
            raise ValueError("per_step_samples must be positive")
        if start_step_index <= 0:
            raise ValueError("start_step_index must be positive")

        # Use a deterministic RNG per sample/rollout so repeated runs choose the
        # same sampled action whenever the candidate pool is unchanged.
        rng = random.Random(f"{sample.sample_id}:{rollout_id}")
        sampled_steps: list[AgentBenchSampledStep] = []
        final_answer: str | None = None
        # The caller must ensure the live session is already positioned right
        # before start_step_index; this function trusts that invariant.
        # A rollout can be bounded to a local branch window; otherwise it runs
        # until the task finishes or the executor-level max step limit is hit.
        last_step_index = self.max_steps if stop_step_index is None else min(stop_step_index, self.max_steps)
        resolved_sampling_cursors = dict(sampling_cursors or {})
        resolved_forced_sample_indices = dict(forced_sample_indices or {})
        used_no_tool_call_samples = False
        no_tool_call_sample_count = 0
        hard_requirement_failure = False

        for step_index in range(start_step_index, last_step_index + 1):

            # Before doing the next step, check whether the AgentBench task is already finished.
            # If it is finished, stop the rollout loop.
            if self._session_finished(session):
                break

            # For the current step, decide where to start sampling from.
            # If this step must use an exact sample, adjust the sampling window so that exact sample is included.

            # sampling_cursor controls which slice of the shared sample pool is
            # requested for this step. forced_sample_index is stricter: it
            # ensures a cached/replayed trajectory keeps the same realized action.
            forced_sample_index = resolved_forced_sample_indices.get(step_index)
            sampling_cursor = resolved_sampling_cursors.get(step_index, 0)
            if forced_sample_index is not None:
                if forced_sample_index < 0:
                    raise ValueError("forced sample indices must be non-negative")
                sampling_cursor = max(0, forced_sample_index - (per_step_samples - 1))

            # Ask the model for candidate actions for the current step, clean those candidates,
            # then summarize whether any of them were invalid/non-tool-call outputs.

            # Ask the model for candidate next actions from the current session
            # state. Shared sampling may reuse cached candidates for identical
            # messages/tools/context.
            sampled_step_samples = await self._sample_step_samples(
                sample=sample,
                # Pass the current conversation history into the model, so the model knows what has already happened.
                messages=session.messages,
                tools=session.tools,
                per_step_samples=per_step_samples,
                step_index=step_index,
                sampling_cursor=sampling_cursor,
            )
            prepared_step_samples = self._prepare_step_samples(
                sample_id=sample.sample_id,
                context_id=rollout_id,
                step_index=step_index,
                sampled_step_samples=sampled_step_samples,
            )
            # Remove unusable empty outputs, but keep non-callable samples in the
            # pool so uncertainty accounting can record that the model produced them.
            sampling_metadata = self._summarize_step_sampling(prepared_step_samples)
            if sampling_metadata["used_no_tool_call_samples"]:
                used_no_tool_call_samples = True
                no_tool_call_sample_count += sampling_metadata["no_tool_call_sample_count"]
            sampled_actions = [step_sample.output for step_sample in prepared_step_samples]
            # Only samples with decodable tool calls can normally be sent to the
            # AgentBench controller.
            selectable_indices = [
                index for index, step_sample in enumerate(prepared_step_samples) if self._step_sample_is_selectable(step_sample)
            ]

            if forced_sample_index is not None:
                # Convert the absolute forced sample index into an index inside
                # the current sampled window.
                chosen_output_index = forced_sample_index - sampling_cursor
                if not 0 <= chosen_output_index < len(prepared_step_samples):
                    raise ValueError(
                        f"Forced sample index {forced_sample_index} for step {step_index} was not available in the sampled pool."
                    )
                if chosen_output_index not in selectable_indices:
                    # Some executors may know how to recover a valid step from a
                    # non-tool-call output. If not, this becomes a hard generation
                    # failure because the controller cannot advance.
                    recovered_step = await self._recover_no_tool_call_step(
                        sample=sample,
                        session=session,
                        step_index=step_index,
                        prepared_step_samples=prepared_step_samples,
                        chosen_output_index=chosen_output_index,
                        sampling_metadata=sampling_metadata,
                    )
                    if recovered_step is not None:
                        sampled_steps.append(recovered_step)
                        final_answer = self.controller_client._extract_final_answer(session.messages)
                        chosen_action = self._format_assistant_message(recovered_step.assistant_message)
                        if self._session_finished(session) or (
                            isinstance(chosen_action, str) and self._is_finish_action(chosen_action)
                        ):
                            break
                        continue
                    failure_step = self._build_no_tool_call_failure_step(
                        step_index=step_index,
                        prepared_step_samples=prepared_step_samples,
                        chosen_output_index=chosen_output_index,
                        sampling_metadata=sampling_metadata,
                    )
                    sampled_steps.append(failure_step)
                    session.messages.append(dict(failure_step.assistant_message))
                    session.final_state = {
                        "status": "generation_failed",
                        "finish": True,
                        "reward": 0,
                        "metrics": {},
                    }
                    hard_requirement_failure = True
                    break
            else:
                # Normal rollouts pick a valid tool-call sample. If several are
                # available, the deterministic RNG chooses among them.
                if selectable_indices:
                    chosen_output_index = self._choose_step_sample_index(
                        prepared_step_samples=prepared_step_samples,
                        selectable_indices=selectable_indices,
                        rng=rng,
                    )
                else:
                    chosen_output_index = 0

            if not selectable_indices:
                # No candidate can be applied to the controller. Try executor
                # recovery first, then mark the rollout as a terminal generation
                # failure if recovery is unavailable.
                recovered_step = await self._recover_no_tool_call_step(
                    sample=sample,
                    session=session,
                    step_index=step_index,
                    prepared_step_samples=prepared_step_samples,
                    chosen_output_index=chosen_output_index,
                    sampling_metadata=sampling_metadata,
                )
                if recovered_step is not None:
                    sampled_steps.append(recovered_step)
                    chosen_action = self._format_assistant_message(recovered_step.assistant_message)
                    final_answer = self.controller_client._extract_final_answer(session.messages)
                    if self._session_finished(session) or (
                        isinstance(chosen_action, str) and self._is_finish_action(chosen_action)
                    ):
                        break
                    continue
                failure_step = self._build_no_tool_call_failure_step(
                    step_index=step_index,
                    prepared_step_samples=prepared_step_samples,
                    chosen_output_index=chosen_output_index,
                    sampling_metadata=sampling_metadata,
                )
                sampled_steps.append(failure_step)
                session.messages.append(dict(failure_step.assistant_message))
                session.final_state = {
                    "status": "generation_failed",
                    "finish": True,
                    "reward": 0,
                    "metrics": {},
                }
                hard_requirement_failure = True
                break

            # Apply the chosen assistant message to the controller/runtime. This
            # mutates session.messages and captures the resulting observations.

            # 1. Get the chosen sampled candidate.
            # 2. Extract its raw assistant tool-call message.
            # 3. Execute that message in the AgentBench session.
            # 4. Record the resulting step.
            # 5. Check whether the task finished.
            # 6. If finished, stop the rollout loop.

            chosen_sample = prepared_step_samples[chosen_output_index]
            assistant_message = chosen_sample.recovery_state.get("assistant_message")
            if not isinstance(assistant_message, Mapping):
                raise AgentBenchControllerError("Tool-calling step samples must preserve the raw assistant message.")

            candidate_indices = [chosen_output_index] + [
                index for index in selectable_indices if index != chosen_output_index
            ]
            failed_candidate_indices: list[int] = []
            interaction_failure_messages: list[str] = []
            realized_step: AgentBenchSampledStep | None = None
            chosen_sample_for_interaction = chosen_sample
            chosen_output_index_for_interaction = chosen_output_index
            for candidate_index in candidate_indices:
                candidate_sample = prepared_step_samples[candidate_index]
                candidate_assistant_message = candidate_sample.recovery_state.get("assistant_message")
                if not isinstance(candidate_assistant_message, Mapping):
                    raise AgentBenchControllerError("Tool-calling step samples must preserve the raw assistant message.")
                retry_metadata = dict(sampling_metadata)
                if failed_candidate_indices:
                    retry_metadata["interaction_retry_count"] = len(failed_candidate_indices)
                    retry_metadata["interaction_failed_candidate_indices"] = list(failed_candidate_indices)
                    retry_metadata["interaction_failure_messages"] = list(interaction_failure_messages)
                try:
                    # It sends the chosen assistant message to the controller/runtime, executing
                    # the tool call, like running bash in the container.
                    realized_step = await self._build_interacted_sampled_step(
                        sample=sample,
                        session=session,
                        step_index=step_index,
                        assistant_message=candidate_assistant_message,
                        prepared_step_samples=prepared_step_samples,
                        chosen_output_index=candidate_index,
                        sampling_metadata=retry_metadata,
                    )
                    chosen_sample_for_interaction = candidate_sample
                    chosen_output_index_for_interaction = candidate_index
                    break
                except Exception as exc:
                    if not self._should_retry_step_interaction_error(exc):
                        raise
                    failed_candidate_indices.append(candidate_index)
                    interaction_failure_messages.append(str(exc))
                    if len(failed_candidate_indices) >= len(candidate_indices):
                        raise
                    LOGGER.warning(
                        "Retrying AgentBench step after interaction error for sample=%s step=%s candidate=%s: %s",
                        sample.sample_id,
                        step_index,
                        candidate_index,
                        exc,
                    )
            if realized_step is None:
                raise AgentBenchControllerError("No selectable AgentBench step sample could be executed.")
            sampled_steps.append(realized_step)
            chosen_output_index = chosen_output_index_for_interaction
            chosen_sample = chosen_sample_for_interaction
            chosen_action = chosen_sample.output
            final_answer = self.controller_client._extract_final_answer(session.messages)
            # Stop once the controller says the task is finished or the chosen
            # action itself is a terminal action.
            if self._session_finished(session) or (
                isinstance(chosen_action, str) and self._is_finish_action(chosen_action)
            ):
                break

        if stop_step_index is None and not hard_requirement_failure and not self._session_finished(session):
            forced_finish_step = await self._force_finish_unfinished_session(
                sample=sample,
                session=session,
                step_index=last_step_index + 1,
            )
            if forced_finish_step is not None:
                sampled_steps.append(forced_finish_step)
                final_answer = self.controller_client._extract_final_answer(session.messages)

        # Convert the live controller/session state into the raw AgentBench-style
        # output payload used by downstream serializers.

        # 1. Build final raw AgentBench output from session state.
        # 2. Extract the result dictionary.
        # 3. If generation failed, mark that clearly.
        # 4. If final_answer is still missing, try to get it from result["answer"].


        # So it packages:
        # which sample was run
        # session id
        # full conversation history
        # final status
        # reward/score
        # answer if available
        raw_output = self.controller_client._build_fc_output(
            sample,
            session.session_id,
            session.messages,
            session.final_state,
        )
        result_payload = dict(raw_output.get("result", {})) if isinstance(raw_output.get("result"), Mapping) else {}

        if hard_requirement_failure:
            # Make non-recoverable no-tool-call failures explicit in the final
            # payload, since they are model-generation failures rather than task
            # answers.
            raw_output["status"] = "generation_failed"
            result_payload.update(
                {
                    "answer": None,
                    "is_correct": False,
                    "error_code": _TOOL_CALL_SAMPLING_ERROR_CODE,
                    "failure_reason": _NO_TOOL_CALL_HARD_FAILURE_REASON,
                }
            )
            raw_output["result"] = result_payload

        if final_answer is None:
            # If we have not found the final answer yet, try to get it from the final result payload.
            # Some controller implementations only expose the answer in the final
            # result payload, so use that as a fallback.

            # If result_payload is a dictionary-like object, read its "answer" field. Otherwise, use None.
            # if no final answer is generated, final_answer will still be None, which downstream serializers can interpret as "no answer found".
            answer = result_payload.get("answer") if isinstance(result_payload, Mapping) else None
            final_answer = self._normalize_final_answer(answer)

        # Return both the realized sampled steps and the aggregate metadata needed
        # by uncertainty estimators and result serialization.

        # These lines return the final rollout object containing the prompt, all executed steps,
        # final answer, status, raw result, and metadata needed later for uncertainty estimation and result serialization.
        return AgentBenchRollout(
            prompt=self._extract_prompt(session.messages),
            steps=sampled_steps,
            final_answer=final_answer,
            status=self._coerce_optional_string(raw_output.get("status")),
            result=result_payload,
            raw_output=raw_output,
            metadata={
                "executor": self.name,
                "rollout_id": rollout_id,
                "sampling_cursors": resolved_sampling_cursors,
                "session_id": session.session_id,
                "used_no_tool_call_samples": used_no_tool_call_samples,
                "no_tool_call_sample_count": no_tool_call_sample_count,
                "hard_requirement_failure": hard_requirement_failure,
                "hard_requirement_failure_reason": (
                    _NO_TOOL_CALL_HARD_FAILURE_REASON if hard_requirement_failure else None
                ),
            },
        )

    async def _replay_steps(
        self,
        session: AgentBenchFunctionCallingSession,
        steps: Sequence[StepRecord],
    ) -> None:
        """Replay recorded steps into a session, validating observations when strict."""
        for step in steps:
            assistant_message = self._prepare_assistant_message_for_interaction(self._assistant_message_from_step(step))
            expected_observations = self.controller_client._normalize_message_list(step.messages[1:])
            actual_observations, _ = await asyncio.to_thread(
                self.controller_client.interact_function_calling,
                session,
                assistant_message,
            )
            expected = self._normalize_replay_messages(expected_observations)
            actual = self._normalize_replay_messages(actual_observations)
            replay_matches = expected == actual or self._replay_messages_equivalent(
                assistant_message=assistant_message,
                expected=expected,
                actual=actual,
            )
            if not replay_matches:
                mismatch = self._replay_mismatch_metadata(step=step, expected=expected, actual=actual)
                session.replay_mismatches.append(mismatch)
                if self.strict_replay:
                    raise AgentBenchControllerError(
                        f"Deterministic replay diverged for sample={session.sample.sample_id} step={step.index}."
                    )
                LOGGER.warning(
                    "Relaxed AgentBench replay mismatch for sample=%s step=%s; continuing with cached observation context",
                    session.sample.sample_id,
                    step.index,
                )
            if self.strict_replay:
                if expected_observations != actual_observations:
                    self._restore_cached_replay_observations(session, expected_observations)
            elif expected_observations != actual_observations:
                self._restore_cached_replay_observations(session, expected_observations)
            if self._session_finished(session):
                break

    def _replay_mismatch_metadata(
        self,
        *,
        step: StepRecord,
        expected: Sequence[Mapping[str, Any]],
        actual: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build compact diagnostics for a replay observation mismatch."""
        return {
            "step_index": step.index,
            "expected_digest": self._messages_digest(expected),
            "actual_digest": self._messages_digest(actual),
            "expected_preview": self._messages_preview(expected),
            "actual_preview": self._messages_preview(actual),
        }

    def _messages_digest(self, messages: Sequence[Mapping[str, Any]]) -> str:
        payload = json.dumps(
            [dict(message) for message in messages],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _messages_preview(self, messages: Sequence[Mapping[str, Any]], *, limit: int = 300) -> str:
        text = " ".join(str(message.get("content", "")) for message in messages if isinstance(message, Mapping))
        normalized = " ".join(text.split())
        return normalized[:limit]

    def _replay_messages_equivalent(
        self,
        *,
        assistant_message: Mapping[str, Any],
        expected: Sequence[Mapping[str, Any]],
        actual: Sequence[Mapping[str, Any]],
    ) -> bool:
        """Allow DBBench SQL replay equality when unordered rows differ only by order."""
        if self.name != "agentbench_dbbench" or not self._dbbench_query_allows_unordered_rows(assistant_message):
            return False
        return self._normalize_dbbench_unordered_sql_observations(expected) == self._normalize_dbbench_unordered_sql_observations(actual)

    def _prepare_assistant_message_for_interaction(self, assistant_message: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize a saved assistant message before sending it to the controller."""
        if self.name != "agentbench_dbbench":
            return dict(assistant_message)
        return self._remap_dbbench_session_schema_in_assistant_message(assistant_message)

    def _remap_dbbench_session_schema_in_assistant_message(self, assistant_message: Mapping[str, Any]) -> dict[str, Any]:
        """Remove volatile DBBench session database names inside SQL tool calls."""
        remapped_message = copy.deepcopy(dict(assistant_message))
        tool_calls = remapped_message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return remapped_message

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or str(function.get("name", "")) != "execute_sql":
                continue
            arguments = self.controller_client._decode_function_arguments(function)
            if not isinstance(arguments, Mapping):
                continue
            remapped_arguments = dict(arguments)
            changed = False
            for key, value in list(remapped_arguments.items()):
                if not isinstance(value, str):
                    continue
                remapped_value = self._remap_dbbench_session_schema_in_sql(value)
                if remapped_value != value:
                    remapped_arguments[key] = remapped_value
                    changed = True
            if changed:
                function["arguments"] = json.dumps(remapped_arguments, ensure_ascii=True, separators=(",", ":"))
        return remapped_message

    def _remap_dbbench_session_schema_in_sql(self, query: str) -> str:
        """Rewrite session-specific DBBench SQL references to reusable forms."""
        without_qualifiers = _DBBENCH_SESSION_DATABASE_QUALIFIER_PATTERN.sub("", query)
        return _DBBENCH_SESSION_DATABASE_STRING_PATTERN.sub("DATABASE()", without_qualifiers)

    def _dbbench_query_allows_unordered_rows(self, assistant_message: Mapping[str, Any]) -> bool:
        """Return whether an SQL query lacks ORDER BY and may return rows unordered."""
        for _, function in self.controller_client._iter_assistant_functions([assistant_message]):
            if str(function.get("name", "")) != "execute_sql":
                continue
            arguments = self.controller_client._decode_function_arguments(function)
            if not isinstance(arguments, Mapping):
                continue
            query = arguments.get("query")
            if not isinstance(query, str):
                continue
            normalized_query = " ".join(query.lower().split())
            if " order by " in f" {normalized_query} ":
                return False
            return True
        return False

    def _normalize_dbbench_unordered_sql_observations(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize SQL result observations by sorting row multisets."""
        normalized: list[dict[str, Any]] = []
        for message in messages:
            normalized_message = self._normalize_nested_mapping(self._cache_key_message(message))
            if isinstance(normalized_message, dict):
                content = normalized_message.get("content")
                unordered_rows = self._dbbench_sql_result_multiset(content)
                if unordered_rows is not None:
                    normalized_message = dict(normalized_message)
                    normalized_message["content"] = unordered_rows
            normalized.append(normalized_message)
        return normalized

    def _dbbench_sql_result_multiset(self, content: Any) -> list[str] | None:
        """Parse a SQL result list and return canonical sorted row strings."""
        if not isinstance(content, str):
            return None
        try:
            parsed = ast.literal_eval(content)
        except (SyntaxError, ValueError):
            return None
        if not isinstance(parsed, list):
            return None
        canonical_rows = [
            json.dumps(self._normalize_nested_mapping(row), sort_keys=True, ensure_ascii=True)
            for row in parsed
        ]
        return sorted(canonical_rows)

    def _restore_cached_replay_observations(
        self,
        session: AgentBenchFunctionCallingSession,
        expected_observations: Sequence[Mapping[str, Any]],
    ) -> None:
        """Replace replayed observations with cached ones after accepted equivalence."""
        if not expected_observations:
            return
        replacement = [dict(message) for message in expected_observations if isinstance(message, Mapping)]
        if not replacement:
            return
        if len(session.messages) >= len(replacement):
            session.messages[-len(replacement) :] = replacement
        if isinstance(session.final_state, Mapping):
            session.final_state = dict(session.final_state)
            session.final_state["messages"] = [dict(message) for message in replacement]

    async def _apply_steps_without_validation(
        self,
        session: AgentBenchFunctionCallingSession,
        steps: Sequence[StepRecord],
    ) -> None:
        """Apply recorded steps to a session without checking returned observations."""
        for step in steps:
            assistant_message = self._prepare_assistant_message_for_interaction(self._assistant_message_from_step(step))
            await asyncio.to_thread(
                self.controller_client.interact_function_calling,
                session,
                assistant_message,
            )
            if self._session_finished(session):
                break

    def _cache_history_state(
        self,
        *,
        sample: AgentBenchSample,
        history_key: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        final_state: Mapping[str, Any] | None = None,
    ) -> str:
        """Store a local history state and return its in-memory cache key."""
        cache_key = f"{sample.sample_id}:{history_key}:{len(self._history_state_cache)}"
        self._history_state_cache[cache_key] = {
            "messages": [dict(message) for message in messages if isinstance(message, Mapping)],
            "tools": [dict(tool) for tool in tools if isinstance(tool, Mapping)],
            "final_state": dict(final_state) if isinstance(final_state, Mapping) else None,
        }
        return cache_key

    async def _resolve_history_state(
        self,
        sample: AgentBenchSample,
        history: LocalBranchHistory,
    ) -> dict[str, Any]:
        """Load a cached local history state or rebuild it from recorded steps."""
        cache_key = history.metadata.get("state_cache_key")
        if isinstance(cache_key, str):
            cached = self._history_state_cache.get(cache_key)
            if isinstance(cached, dict):
                return {
                    "messages": [dict(message) for message in cached.get("messages", []) if isinstance(message, Mapping)],
                    "tools": [dict(tool) for tool in cached.get("tools", []) if isinstance(tool, Mapping)],
                    "final_state": dict(cached.get("final_state", {})) if isinstance(cached.get("final_state"), Mapping) else None,
                }

        session = await self._rebuild_history_session(sample, history)
        return {
            "messages": session["messages"],
            "tools": session["tools"],
            "final_state": session.get("final_state"),
        }

    async def _rebuild_history_session(
        self,
        sample: AgentBenchSample,
        history: LocalBranchHistory,
    ) -> dict[str, Any]:
        """Reconstruct local history state by replaying prefix and branch steps."""
        session = await self._start_session_with_prefix(sample, history.prefix_steps + history.branched_steps)
        try:
            return {
                "messages": [dict(message) for message in session.messages],
                "tools": [dict(tool) for tool in session.tools],
                "final_state": dict(session.final_state) if isinstance(session.final_state, Mapping) else None,
            }
        finally:
            await asyncio.to_thread(self.controller_client._cancel, session.session_id, prefer_header=True)

    async def _sample_step_samples(
        self,
        *,
        sample: AgentBenchSample,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        per_step_samples: int,
        step_index: int,
        sampling_cursor: int = 0,
    ) -> list[StepSample]:
        """Fetch candidate assistant actions for one state from shared sampling.

        Args:
            sample: The AgentBench task instance being sampled. It is used for
                stable cache keys and profiling metadata, not as conversation
                history.
            messages: The exact conversation history/state to condition the
                model on. Candidate samples are only comparable for the same
                messages/tools/step context.
            tools: The tool definitions available to the model at this state.
            per_step_samples: Number of candidate assistant actions to return.

            step_index: The 1-based step number being sampled. This is part of
                the cache key and should match the state represented by
                ``messages``.
                `step_index` should match the state represented by `messages`
                means that `messages` must contain the conversation history
                immediately before that step. For example, if `step_index = 4`,
                then `messages` should already include the prompt plus the
                assistant actions and tool observations for steps 1, 2, and 3,
                so the model is sampling the next action for step 4. If `messages`
                only contains the initial prompt but `step_index = 4`, the code
                would actually be sampling the first action while labeling and
                caching it as step 4, which is inconsistent. In short, `messages`
                is what has happened so far, and `step_index` should be the next action step number.

            sampling_cursor: Offset into the candidate sample pool for this
                exact state. A cursor of ``0`` returns the first
                ``per_step_samples`` candidates; a cursor of ``4`` skips the
                first four candidates for this same messages/tools/step context.

        Returns:
            A list of ``StepSample`` objects. Each item contains a candidate
            action string, model metadata such as logprobs when available, and
            recovery state containing the raw assistant message used if the
            candidate is later selected and executed.
        """
        async def sample_fn(sample_count: int) -> Sequence[StepSample | Mapping[str, Any]]:
            """Generate the missing samples requested by shared sampling storage."""
            return await self._generate_step_samples(messages=messages, tools=tools, sample_count=sample_count)

        start_time = time.perf_counter()
        outputs = await self.shared_step_sampler.sample(
            category=f"{self.name}/step_samples",
            sample_id=self._step_sampling_pool_id(),
            key=self._step_sampling_cache_key(sample, step_index=step_index, messages=messages, tools=tools),
            sample_count=per_step_samples,
            cursor=sampling_cursor,
            sample_fn=sample_fn,
        )
        _profile_log(
            "step_sample_pool",
            sample_id=sample.sample_id,
            step_index=step_index,
            sample_count=per_step_samples,
            sampling_cursor=sampling_cursor,
            elapsed_ms=round((time.perf_counter() - start_time) * 1000, 3),
            returned_count=len(outputs),
        )
        return outputs

    async def _generate_step_samples(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        sample_count: int,
    ) -> list[StepSample]:
        """Ask the model for tool-call samples and wrap them as StepSample values."""
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")

        use_many = False
        tool_callable = getattr(self.model_client, "acompletion_with_tools_many_detailed", None)
        if tool_callable is not None:
            use_many = True
        else:
            tool_callable = getattr(self.model_client, "acompletion_with_tools_detailed", None)
        if tool_callable is None:
            tool_callable = getattr(self.model_client, "acompletion_with_tools_many", None)
            if tool_callable is not None:
                use_many = True
        if tool_callable is None:
            tool_callable = getattr(self.model_client, "acompletion_with_tools", None)
        if tool_callable is None:
            raise TypeError("Configured model client does not support tool-calling completions.")

        sampled: list[StepSample] = []
        sampled_without_tool_calls = 0
        max_attempts = max(4, sample_count * 3)
        attempt = 0
        while (len(sampled) < sample_count or not any(self._step_sample_is_selectable(sample) for sample in sampled)) and attempt < max_attempts:
            remaining_count = max(1, sample_count - len(sampled))
            response = await self._request_tool_completion_batch(
                tool_callable=tool_callable,
                use_many=use_many,
                messages=messages,
                tools=tools,
                sample_count=remaining_count,
            )
            attempt += 1

            if not isinstance(response, Sequence) or isinstance(response, (str, bytes, bytearray)):
                response = [response]

            for item in response:
                if not isinstance(item, Mapping):
                    raise TypeError("Tool-calling step generation must return a message mapping.")
                response_payload = dict(item)
                metadata = response_payload.pop("metadata", None)
                assistant_message = self._normalize_sampled_assistant_message(response_payload)
                action_text = self._format_assistant_message(assistant_message)
                if not action_text:
                    action_text = json.dumps(assistant_message, ensure_ascii=True, sort_keys=True)
                has_tool_call = self._assistant_message_has_callable_tool_call(assistant_message)
                if not has_tool_call:
                    sampled_without_tool_calls += 1
                sampled.append(
                    StepSample(
                        output=action_text,
                        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
                        recovery_state={
                            "assistant_message": assistant_message,
                            "has_tool_call": has_tool_call,
                        },
                    )
                )
                if len(sampled) >= sample_count and any(self._step_sample_is_selectable(sample) for sample in sampled):
                    break

        if sampled and not any(self._step_sample_is_selectable(sample) for sample in sampled):
            repair_response = await self._request_tool_completion_batch(
                tool_callable=tool_callable,
                use_many=use_many,
                messages=self._build_tool_call_repair_messages(messages, tools, sampled),
                tools=tools,
                sample_count=1,
                temperature=0.0,
            )
            if not isinstance(repair_response, Sequence) or isinstance(repair_response, (str, bytes, bytearray)):
                repair_response = [repair_response]

            for item in repair_response:
                if not isinstance(item, Mapping):
                    continue
                response_payload = dict(item)
                metadata = response_payload.pop("metadata", None)
                assistant_message = self._normalize_sampled_assistant_message(response_payload)
                action_text = self._format_assistant_message(assistant_message)
                if not action_text:
                    action_text = json.dumps(assistant_message, ensure_ascii=True, sort_keys=True)
                has_tool_call = self._assistant_message_has_callable_tool_call(assistant_message)
                repaired_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
                repaired_metadata["tool_call_repair_attempted"] = True
                repaired_metadata["tool_call_repair_source"] = "forced_retry"
                repaired_metadata["tool_call_repair_succeeded"] = has_tool_call
                if not has_tool_call:
                    sampled_without_tool_calls += 1
                replacement = StepSample(
                    output=action_text,
                    metadata=repaired_metadata,
                    recovery_state={
                        "assistant_message": assistant_message,
                        "has_tool_call": has_tool_call,
                    },
                )
                if has_tool_call:
                    sampled[-1] = replacement
                    break

        if sampled_without_tool_calls:
            LOGGER.warning(
                "Retained %s sampled assistant messages without callable tool calls for AgentBench uncertainty handling",
                sampled_without_tool_calls,
            )
        if len(sampled) != sample_count:
            if len(sampled) > sample_count:
                selected = list(sampled[:sample_count])
                if not any(self._step_sample_is_selectable(sample) for sample in selected):
                    replacement = next(
                        (sample for sample in sampled[sample_count:] if self._step_sample_is_selectable(sample)),
                        None,
                    )
                    if replacement is not None:
                        selected[-1] = replacement
                sampled = selected
        if len(sampled) != sample_count:
            raise AgentBenchControllerError(
                f"Expected {sample_count} AgentBench step samples but received {len(sampled)} after {attempt} attempts."
            )
        return sampled

    def _build_tool_call_repair_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        sampled: Sequence[StepSample],
    ) -> list[dict[str, Any]]:
        """Build a terse retry prompt after the model returned prose instead of a tool call."""
        repaired_messages = [dict(message) for message in messages if isinstance(message, Mapping)]
        previous_output = ""
        for step_sample in reversed(sampled):
            assistant_message = step_sample.recovery_state.get("assistant_message")
            if isinstance(assistant_message, Mapping):
                content = assistant_message.get("content")
                if isinstance(content, str) and content.strip():
                    previous_output = content.strip()
                    break
            if step_sample.output.strip():
                previous_output = step_sample.output.strip()
                break

        tool_names = [
            str(function.get("name"))
            for tool in tools
            if isinstance(tool, Mapping)
            and isinstance((function := tool.get("function")), Mapping)
            and isinstance(function.get("name"), str)
        ]
        repair_instruction = (
            "/no_think\n"
            "Your previous response did not contain a valid callable tool call. "
            "Return exactly one tool call now and no explanation. "
            "Begin with '{' and end with '}'. "
            "Use the same task context. "
            f"Allowed tool names: {', '.join(tool_names)}."
        )
        if previous_output:
            repaired_messages.append({"role": "assistant", "content": previous_output})
        repaired_messages.append({"role": "user", "content": repair_instruction})
        return repaired_messages

    async def _request_tool_completion_batch(
        self,
        *,
        tool_callable: Any,
        use_many: bool,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        sample_count: int,
        temperature: float | None = None,
    ) -> Any:
        """Call either batched or repeated single tool-completion APIs."""
        start_time = time.perf_counter()
        resolved_temperature = self.temperature if temperature is None else float(temperature)
        if use_many:
            response = tool_callable(
                [dict(message) for message in messages],
                [dict(tool) for tool in tools],
                temperature=resolved_temperature,
                n=sample_count,
            )
            resolved = await response if asyncio.iscoroutine(response) else response
            response_count = len(resolved) if isinstance(resolved, Sequence) and not isinstance(resolved, (str, bytes, bytearray, Mapping)) else None
            _profile_log(
                "tool_completion_batch",
                use_many=True,
                sample_count=sample_count,
                elapsed_ms=round((time.perf_counter() - start_time) * 1000, 3),
                response_count=response_count,
            )
            return resolved

        resolved = [
            await tool_callable(
                [dict(message) for message in messages],
                [dict(tool) for tool in tools],
                temperature=resolved_temperature,
            )
            for _ in range(sample_count)
        ]
        _profile_log(
            "tool_completion_batch",
            use_many=False,
            sample_count=sample_count,
            elapsed_ms=round((time.perf_counter() - start_time) * 1000, 3),
            response_count=len(resolved),
        )
        return resolved

    def _cache_key_messages(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Normalize messages for stable sampling-cache keys."""
        return [self._cache_key_message(message) for message in messages if isinstance(message, Mapping)]

    def _cache_key_tools(self, tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Normalize tool definitions for stable sampling-cache keys."""
        return [copy.deepcopy(dict(tool)) for tool in tools if isinstance(tool, Mapping)]

    def _cache_key_message(self, message: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize volatile tool-call ids inside a single cache-key message."""
        normalized = copy.deepcopy(dict(message))
        tool_calls = normalized.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_calls: list[dict[str, Any]] = []
            for index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, Mapping):
                    continue
                normalized_call = copy.deepcopy(dict(tool_call))
                normalized_call["id"] = f"tool-call-{index}"
                normalized_calls.append(normalized_call)
            normalized["tool_calls"] = normalized_calls

        tool_call_id = normalized.get("tool_call_id")
        if isinstance(tool_call_id, str):
            normalized["tool_call_id"] = "tool-call-0"
        return normalized

    def _normalize_sampled_assistant_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Normalize model output into one controller-compatible assistant message."""
        normalized = dict(self._extract_sampled_assistant_message(message))
        normalized["role"] = "assistant"

        tool_calls = normalized.get("tool_calls")
        if not isinstance(tool_calls, list):
            return normalized

        valid_tool_calls = [dict(tool_call) for tool_call in tool_calls if isinstance(tool_call, Mapping)]
        if not valid_tool_calls:
            normalized.pop("tool_calls", None)
            return normalized

        callable_tool_calls = []
        for tool_call in valid_tool_calls:
            function = tool_call.get("function")
            if not isinstance(function, Mapping):
                continue
            if self.controller_client._decode_function_arguments(function) is not None:
                callable_tool_calls.append(tool_call)

        if len(valid_tool_calls) > 1:
            LOGGER.warning(
                "Dropping %s extra tool calls from sampled assistant message to preserve controller-compatible history",
                len(valid_tool_calls) - 1,
            )
        if callable_tool_calls:
            normalized["tool_calls"] = [callable_tool_calls[0]]
        else:
            normalized["tool_calls"] = [valid_tool_calls[0]]
        return normalized

    def _extract_sampled_assistant_message(self, message: Mapping[str, Any]) -> Mapping[str, Any]:
        """Unwrap common response envelopes to find the assistant message payload."""
        for key in ("assistant_message", "message", "response"):
            nested = message.get(key)
            if not isinstance(nested, Mapping):
                continue
            if nested.get("tool_calls") is not None or nested.get("content") is not None or nested.get("role") is not None:
                return nested
        return message

    def _prepare_step_samples(
        self,
        *,
        sample_id: str,
        context_id: str,
        step_index: int,
        sampled_step_samples: list[StepSample],
    ) -> list[StepSample]:
        """Drop empty samples and keep metadata about non-callable tool outputs."""
        usable_samples: list[StepSample] = []
        dropped_empty_outputs = 0
        retained_no_tool_call_samples = 0
        for step_sample in sampled_step_samples:
            if not step_sample.output.strip():
                dropped_empty_outputs += 1
                continue
            if not self._step_sample_has_tool_call(step_sample):
                retained_no_tool_call_samples += 1
            usable_samples.append(step_sample)

        if dropped_empty_outputs:
            LOGGER.warning(
                "Empty AgentBench tool-calling outputs detected for sample_id=%s context=%s step=%s dropped=%s total=%s",
                sample_id,
                context_id,
                step_index,
                dropped_empty_outputs,
                len(sampled_step_samples),
            )
        if retained_no_tool_call_samples:
            LOGGER.warning(
                "Using %s sampled assistant messages without callable tool calls for sample_id=%s context=%s step=%s",
                retained_no_tool_call_samples,
                sample_id,
                context_id,
                step_index,
            )
        if usable_samples:
            return usable_samples
        raise AgentBenchControllerError(
            f"No AgentBench step samples were produced for sample_id={sample_id} context={context_id} step={step_index}."
        )

    def _step_sample_has_tool_call(self, step_sample: StepSample) -> bool:
        """Return whether a sampled output contains a callable tool call."""
        assistant_message = step_sample.recovery_state.get("assistant_message")
        if assistant_message is not None:
            return self._assistant_message_has_callable_tool_call(assistant_message)
        has_tool_call = step_sample.recovery_state.get("has_tool_call")
        if isinstance(has_tool_call, bool):
            return has_tool_call
        return False

    def _step_sample_is_selectable(self, step_sample: StepSample) -> bool:
        """Return whether a sample is eligible to be applied to the controller."""
        return self._step_sample_has_tool_call(step_sample)

    def _should_retry_step_interaction_error(self, exc: Exception) -> bool:
        """Return whether the rollout may try another sampled action after ``exc``."""
        del exc
        return False

    def _choose_step_sample_index(
        self,
        *,
        prepared_step_samples: Sequence[StepSample],
        selectable_indices: Sequence[int],
        rng: random.Random,
    ) -> int:
        """Choose one selectable sampled action for the realized rollout step."""
        del prepared_step_samples
        if not selectable_indices:
            raise AgentBenchControllerError("No selectable AgentBench step samples were available.")
        return selectable_indices[0] if len(selectable_indices) == 1 else rng.choice(list(selectable_indices))

    def _summarize_step_sampling(self, sampled_step_samples: Sequence[StepSample]) -> dict[str, Any]:
        """Count callable vs non-callable samples for run metadata."""
        no_tool_call_sample_count = sum(1 for step_sample in sampled_step_samples if not self._step_sample_has_tool_call(step_sample))
        return {
            "used_no_tool_call_samples": no_tool_call_sample_count > 0,
            "no_tool_call_sample_count": no_tool_call_sample_count,
            "callable_sample_count": len(sampled_step_samples) - no_tool_call_sample_count,
        }

    def _build_no_tool_call_failure_step(
        self,
        *,
        step_index: int,
        prepared_step_samples: Sequence[StepSample],
        chosen_output_index: int,
        sampling_metadata: Mapping[str, Any],
    ) -> AgentBenchSampledStep:
        """Construct a terminal failure step when no callable tool call exists."""
        if not prepared_step_samples:
            raise AgentBenchControllerError("No AgentBench step samples were available to construct a failure step.")
        chosen_sample = prepared_step_samples[chosen_output_index]
        assistant_message = chosen_sample.recovery_state.get("assistant_message")
        if isinstance(assistant_message, Mapping):
            resolved_message = dict(assistant_message)
        else:
            resolved_message = {"role": "assistant", "content": chosen_sample.output}
        sampled_actions = [step_sample.output for step_sample in prepared_step_samples]
        sampled_output_metadata = [dict(step_sample.metadata) for step_sample in prepared_step_samples]
        entropy, _ = compute_predictive_entropy_from_metadata(sampled_output_metadata)
        return AgentBenchSampledStep(
            index=step_index,
            assistant_message=resolved_message,
            sampled_messages=[
                dict(step_sample.recovery_state.get("assistant_message", {}))
                for step_sample in prepared_step_samples
            ],
            sampled_actions=sampled_actions,
            sampled_output_metadata=sampled_output_metadata,
            chosen_output_index=chosen_output_index,
            entropy=0.0 if entropy is None else float(entropy),
            observation_messages=[],
            metadata={
                **dict(sampling_metadata),
                "hard_requirement_failure": True,
                "hard_requirement_failure_reason": _NO_TOOL_CALL_HARD_FAILURE_REASON,
            },
        )

    async def _build_interacted_sampled_step(
        self,
        *,
        sample: AgentBenchSample,
        session: AgentBenchFunctionCallingSession,
        step_index: int,
        assistant_message: Mapping[str, Any],
        prepared_step_samples: Sequence[StepSample],
        chosen_output_index: int,
        sampling_metadata: Mapping[str, Any],
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> AgentBenchSampledStep:
        """Apply the chosen assistant message and record observations/entropy."""
        merged_extra_metadata = dict(extra_metadata) if isinstance(extra_metadata, Mapping) else {}
        start_time = time.perf_counter()
        interaction_assistant_message = self._prepare_assistant_message_for_interaction(assistant_message)
        observation_messages, _ = await asyncio.to_thread(
            self.controller_client.interact_function_calling,
            session,
            interaction_assistant_message,
        )
        _profile_log(
            "step_interact",
            sample_id=sample.sample_id,
            step_index=step_index,
            elapsed_ms=round((time.perf_counter() - start_time) * 1000, 3),
            observation_count=len(observation_messages),
        )
        runtime_step_metadata = self._build_runtime_step_metadata(
            sample=sample,
            session=session,
            step_index=step_index,
        )
        sampled_actions = [step_sample.output for step_sample in prepared_step_samples]
        sampled_output_metadata = [dict(step_sample.metadata) for step_sample in prepared_step_samples]
        entropy, _ = compute_predictive_entropy_from_metadata(sampled_output_metadata)
        return AgentBenchSampledStep(
            index=step_index,
            assistant_message=interaction_assistant_message,
            sampled_messages=[
                dict(step_sample.recovery_state.get("assistant_message", {}))
                for step_sample in prepared_step_samples
            ],
            sampled_actions=sampled_actions,
            sampled_output_metadata=sampled_output_metadata,
            chosen_output_index=chosen_output_index,
            entropy=0.0 if entropy is None else float(entropy),
            observation_messages=observation_messages,
            metadata={
                **dict(sampling_metadata),
                **runtime_step_metadata,
                "hard_requirement_failure": False,
                **merged_extra_metadata,
            },
        )

    async def _recover_no_tool_call_step(
        self,
        *,
        sample: AgentBenchSample,
        session: AgentBenchFunctionCallingSession,
        step_index: int,
        prepared_step_samples: Sequence[StepSample],
        chosen_output_index: int,
        sampling_metadata: Mapping[str, Any],
    ) -> AgentBenchSampledStep | None:
        """Hook for executors that can recover when a sample lacks tool calls."""
        del sample, session, step_index, prepared_step_samples, chosen_output_index, sampling_metadata
        return None

    async def _force_finish_unfinished_session(
        self,
        *,
        sample: AgentBenchSample,
        session: AgentBenchFunctionCallingSession,
        step_index: int,
    ) -> AgentBenchSampledStep | None:
        """Optional hook to submit a terminal action when a full rollout hits max steps."""
        del sample, session, step_index
        return None

    def _assistant_message_has_tool_call(self, assistant_message: Any) -> bool:
        """Return whether an assistant message includes any tool-call structure."""
        if not isinstance(assistant_message, Mapping):
            return False
        tool_calls = assistant_message.get("tool_calls")
        return isinstance(tool_calls, list) and any(isinstance(tool_call, Mapping) for tool_call in tool_calls)

    def _assistant_message_has_callable_tool_call(self, assistant_message: Any) -> bool:
        """Return whether an assistant message has decodable tool-call arguments."""
        if not isinstance(assistant_message, Mapping):
            return False
        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return False
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                continue
            function = tool_call.get("function")
            if not isinstance(function, Mapping):
                continue
            if self.controller_client._decode_function_arguments(function) is not None:
                return True
        return False

    def _build_backbone_step_record(self, step: AgentBenchSampledStep) -> StepRecord:
        """Convert a sampled AgentBench step into the common StepRecord format."""
        chosen_output_metadata = self._resolve_chosen_output_metadata(step)
        return StepRecord(
            index=step.index,
            action=self._format_assistant_message(step.assistant_message),
            observation=self._format_observation(step.observation_messages),
            messages=[dict(step.assistant_message), *[dict(message) for message in step.observation_messages]],
            metadata={
                **dict(step.metadata),
                "pe": step.entropy,
                "sampled_actions": list(step.sampled_actions),
                "sampled_messages": [dict(message) for message in step.sampled_messages],
                "sampled_output_metadata": list(step.sampled_output_metadata),
                "chosen_output_index": step.chosen_output_index,
                "chosen_output_metadata": chosen_output_metadata,
                "tool_calls": step.assistant_message.get("tool_calls") if isinstance(step.assistant_message.get("tool_calls"), list) else [],
                "raw_assistant_message": dict(step.assistant_message),
            },
        )

    def _build_tdp_step_record_from_step_record(self, step: StepRecord) -> TDPStepRecord:
        """Convert a cached backbone/local StepRecord into a TDP step."""
        metadata = dict(step.metadata)
        if step.observation is not None and "observation" not in metadata:
            metadata["observation"] = step.observation
        sampled_actions = metadata.get("sampled_actions")
        resolved_sampled_actions = (
            [str(action) for action in sampled_actions]
            if isinstance(sampled_actions, list)
            else ([step.action] if isinstance(step.action, str) else [])
        )
        uncertainty_measurements: dict[str, float] = {}
        pe = metadata.get("pe")
        if isinstance(pe, (int, float)) and math.isfinite(float(pe)):
            uncertainty_measurements["pe"] = float(pe)
        return TDPStepRecord(
            index=step.index,
            realized_decision=step.action or "",
            sampled_decisions=resolved_sampled_actions,
            uncertainty_measurements=uncertainty_measurements,
            metadata=metadata,
        )

    def _build_tdp_step_record(
        self,
        step: AgentBenchSampledStep,
        *,
        counterfactual_records: list[TDPCounterfactualRecord] | None = None,
    ) -> TDPStepRecord:
        """Convert a sampled step into a TDP step with uncertainty measurements."""
        chosen_output_metadata = self._resolve_chosen_output_metadata(step)
        metadata: dict[str, Any] = {
            **dict(step.metadata),
            "raw_assistant_message": dict(step.assistant_message),
            "sampled_messages": [dict(message) for message in step.sampled_messages],
            "sampled_output_metadata": list(step.sampled_output_metadata),
            "chosen_output_index": step.chosen_output_index,
            "chosen_output_metadata": chosen_output_metadata,
        }
        observation_text = self._format_observation(step.observation_messages)
        if observation_text is not None:
            metadata["observation"] = observation_text
        if step.observation_messages:
            metadata["raw_observation_messages"] = [dict(message) for message in step.observation_messages]

        uncertainty_measurements = {"pe": step.entropy}
        ppl = self._perplexity_from_output_metadata(chosen_output_metadata)
        if ppl is not None:
            uncertainty_measurements["ppl"] = ppl
        return TDPStepRecord(
            index=step.index,
            realized_decision=self._format_assistant_message(step.assistant_message),
            sampled_decisions=list(step.sampled_actions),
            uncertainty_measurements=uncertainty_measurements,
            counterfactual_records=list(counterfactual_records or []),
            metadata=metadata,
        )

    @staticmethod
    def _counterfactual_records_for_step(
        counterfactual_records: Sequence[list[TDPCounterfactualRecord]],
        steps: Sequence[AgentBenchSampledStep],
        step: AgentBenchSampledStep,
    ) -> list[TDPCounterfactualRecord]:
        """Return records for a rollout step even when step indexes are sparse.

        Some controllers append a synthetic terminal checker step at
        ``max_steps + 1``. In that case list position and ``step.index`` no
        longer match, so indexing by ``step.index - 1`` is unsafe.
        """
        by_index = {
            rollout_step.index: records
            for rollout_step, records in zip(steps, counterfactual_records, strict=False)
        }
        return list(by_index.get(step.index, []))

    async def _build_tdp_counterfactual_records(
        self,
        *,
        sample: AgentBenchSample,
        rollout: AgentBenchRollout,
        per_step_samples: int,
    ) -> list[list[TDPCounterfactualRecord]]:
        """Build source-to-target counterfactual sample records for each TDP step."""
        stepwise_records: list[list[TDPCounterfactualRecord]] = [[] for _ in rollout.steps]
        if len(rollout.steps) <= 1:
            return stepwise_records

        realized_step_records = [self._build_backbone_step_record(step) for step in rollout.steps]
        for target_step_index in range(2, len(rollout.steps) + 1):
            for source_step_index in range(1, target_step_index):
                stepwise_records[target_step_index - 1].append(
                    await self._sample_counterfactual_record(
                        sample=sample,
                        rollout=rollout,
                        realized_step_records=realized_step_records,
                        source_step_index=source_step_index,
                        target_step_index=target_step_index,
                        per_step_samples=per_step_samples,
                    )
                )
        return stepwise_records

    async def _sample_counterfactual_record(
        self,
        *,
        sample: AgentBenchSample,
        rollout: AgentBenchRollout,
        realized_step_records: list[StepRecord],
        source_step_index: int,
        target_step_index: int,
        per_step_samples: int,
    ) -> TDPCounterfactualRecord:
        """Sample target-step actions after replacing one earlier source decision."""
        source_step = rollout.steps[source_step_index - 1]
        prefix_steps = realized_step_records[: source_step_index - 1]
        branches: list[TDPCounterfactualBranch] = []

        for branch_index, sampled_message in enumerate(source_step.sampled_messages):
            branch_metadata: dict[str, Any] = {
                "branch_index": branch_index,
                "target_step_index": target_step_index,
            }
            target_sampled_decisions: list[str] = []
            target_sampled_output_metadata: list[dict[str, Any]] = []

            if not self._assistant_message_has_tool_call(sampled_message):
                branch_metadata["terminated_before_target"] = True
                branch_metadata["invalid_source_message"] = True
                branches.append(
                    TDPCounterfactualBranch(
                        source_decision=source_step.sampled_actions[branch_index],
                        target_sampled_decisions=target_sampled_decisions,
                        target_sampled_output_metadata=target_sampled_output_metadata,
                        metadata=branch_metadata,
                    )
                )
                continue

            session = await self._start_function_calling_session(sample)
            try:
                if prefix_steps:
                    await self._apply_steps_without_validation(session, prefix_steps)

                if not self._session_finished(session):
                    await asyncio.to_thread(
                        self.controller_client.interact_function_calling,
                        session,
                        dict(sampled_message),
                    )

                if not self._session_finished(session):
                    intermediate_steps = realized_step_records[source_step_index: target_step_index - 1]
                    if intermediate_steps:
                        await self._apply_steps_without_validation(session, intermediate_steps)

                if self._session_finished(session):
                    branch_metadata["terminated_before_target"] = True
                else:
                    sampled_step_samples = await self._sample_step_samples(
                        sample=sample,
                        messages=session.messages,
                        tools=session.tools,
                        per_step_samples=per_step_samples,
                        step_index=target_step_index,
                    )
                    prepared_step_samples = self._prepare_step_samples(
                        sample_id=sample.sample_id,
                        context_id=f"counterfactual-{source_step_index}-{target_step_index}-{branch_index}",
                        step_index=target_step_index,
                        sampled_step_samples=sampled_step_samples,
                    )
                    target_sampled_decisions = [step_sample.output for step_sample in prepared_step_samples]
                    target_sampled_output_metadata = [
                        dict(step_sample.metadata) for step_sample in prepared_step_samples
                    ]
            except AgentBenchControllerError as exc:
                branch_metadata["terminated_before_target"] = True
                branch_metadata["controller_error"] = str(exc)
            finally:
                await asyncio.to_thread(self.controller_client._cancel, session.session_id, prefer_header=True)

            branches.append(
                TDPCounterfactualBranch(
                    source_decision=source_step.sampled_actions[branch_index],
                    target_sampled_decisions=target_sampled_decisions,
                    target_sampled_output_metadata=target_sampled_output_metadata,
                    metadata=branch_metadata,
                )
            )

        return TDPCounterfactualRecord(
            source_step_index=source_step_index,
            realized_source_decision=self._format_assistant_message(source_step.assistant_message),
            branches=branches,
            metadata={"target_step_index": target_step_index},
        )

    def _resolve_chosen_output_metadata(self, step: AgentBenchSampledStep) -> dict[str, Any]:
        """Return metadata for the sampled output chosen as the realized step."""
        if 0 <= step.chosen_output_index < len(step.sampled_output_metadata):
            metadata = step.sampled_output_metadata[step.chosen_output_index]
            if isinstance(metadata, dict):
                return dict(metadata)
        return {}

    def _perplexity_from_output_metadata(self, metadata: dict[str, Any]) -> float | None:
        """Compute token-level perplexity from summed logprob metadata."""
        logprob_sum = metadata.get("token_logprob_sum")
        token_count = metadata.get("token_count")
        if not isinstance(logprob_sum, (int, float)) or not isinstance(token_count, int):
            return None
        if token_count <= 0:
            return None
        return float(math.exp(-float(logprob_sum) / float(token_count)))

    def _deserialize_tdp(self, payload: Mapping[str, Any]) -> TrajectoryDependentDecisionProcess:
        """Rebuild a TDP object from cached serialized data."""
        tdp_payload = payload.get("tdp")
        if not isinstance(tdp_payload, Mapping):
            raise ValueError("Invalid cached TDP payload.")
        return TrajectoryDependentDecisionProcess(
            sample_id=str(tdp_payload.get("sample_id", "")),
            prompt=str(tdp_payload.get("prompt", "")),
            steps=[self._deserialize_tdp_step_record(step) for step in tdp_payload.get("steps", [])],
            final_answer=self._coerce_optional_string(tdp_payload.get("final_answer")),
            metadata=dict(tdp_payload.get("metadata", {})) if isinstance(tdp_payload.get("metadata"), Mapping) else {},
        )

    def _deserialize_tdp_step_record(self, payload: Any) -> TDPStepRecord:
        """Rebuild one TDP step, including nested counterfactual records."""
        if not isinstance(payload, Mapping):
            raise ValueError("Invalid cached TDP step record.")
        raw_measurements = payload.get("uncertainty_measurements")
        uncertainty_measurements = (
            {str(key): float(value) for key, value in raw_measurements.items()}
            if isinstance(raw_measurements, Mapping)
            else {}
        )
        counterfactual_records: list[TDPCounterfactualRecord] = []
        for item in payload.get("counterfactual_records", []):
            if not isinstance(item, Mapping):
                continue
            branches: list[TDPCounterfactualBranch] = []
            for branch in item.get("branches", []):
                if not isinstance(branch, Mapping):
                    continue
                branches.append(
                    TDPCounterfactualBranch(
                        source_decision=str(branch.get("source_decision", "")),
                        target_sampled_decisions=[str(value) for value in branch.get("target_sampled_decisions", [])],
                        target_sampled_output_metadata=[dict(value) for value in branch.get("target_sampled_output_metadata", []) if isinstance(value, Mapping)],
                        metadata=dict(branch.get("metadata", {})) if isinstance(branch.get("metadata"), Mapping) else {},
                    )
                )
            counterfactual_records.append(
                TDPCounterfactualRecord(
                    source_step_index=int(item.get("source_step_index", 0)),
                    realized_source_decision=str(item.get("realized_source_decision", "")),
                    branches=branches,
                    metadata=dict(item.get("metadata", {})) if isinstance(item.get("metadata"), Mapping) else {},
                )
            )
        return TDPStepRecord(
            index=int(payload.get("index", 0)),
            realized_decision=str(payload.get("realized_decision", "")),
            sampled_decisions=[str(item) for item in payload.get("sampled_decisions", [])],
            uncertainty_measurements=uncertainty_measurements,
            counterfactual_records=counterfactual_records,
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
        )

    def _load_cached_tdp(
        self,
        sample: AgentBenchSample,
        cache_key: dict[str, Any],
    ) -> TrajectoryDependentDecisionProcess | None:
        """Load a cached TDP from shared sampling storage if available."""
        if self.shared_sampling_storage is None:
            return None
        payload = self.shared_sampling_storage.load(f"{self.name}/tdp", sample_id=sample.sample_id, key=cache_key)
        if payload is None:
            return None
        return self._deserialize_tdp(payload)

    def load_cached_tdp_for_local_branching(
        self,
        sample: AgentBenchSample,
        *,
        trajectory_index: int = 0,
        include_counterfactuals: bool = False,
        emulate_tool_calls_fallbacks: Sequence[bool | None] | None = None,
    ) -> TrajectoryDependentDecisionProcess | None:
        """Load a reusable per-trajectory TDP for a local-branching backbone."""
        if self.shared_sampling_storage is None:
            return None
        candidates: list[bool | None] = []
        for candidate in list(emulate_tool_calls_fallbacks or [None]):
            if candidate not in candidates:
                candidates.append(candidate)
        for candidate_emulate_tool_calls in candidates:
            cache_key = self._tdp_cache_key(
                sample,
                trajectory_index=trajectory_index,
                include_counterfactuals=include_counterfactuals,
            )
            if candidate_emulate_tool_calls is not None:
                model_signature = cache_key.get("model")
                if isinstance(model_signature, Mapping):
                    resolved_model_signature = dict(model_signature)
                    resolved_model_signature["emulate_tool_calls"] = bool(candidate_emulate_tool_calls)
                    cache_key["model"] = resolved_model_signature
            tdp = self._load_cached_tdp(sample, cache_key)
            if tdp is None:
                continue
            tdp.metadata["tdp_cache_status"] = (
                "hit_legacy_emulate_false"
                if candidate_emulate_tool_calls is False and self.model_signature.get("emulate_tool_calls") is True
                else "hit"
            )
            tdp.metadata["tdp_cache_key_emulate_tool_calls"] = (
                bool(candidate_emulate_tool_calls)
                if candidate_emulate_tool_calls is not None
                else self.model_signature.get("emulate_tool_calls")
            )
            degree_source_method = tdp.metadata.get("degree_tdp_cache_source_method")
            if degree_source_method:
                previous_source_method = tdp.metadata.get("tdp_cache_source_method")
                if previous_source_method and previous_source_method != degree_source_method:
                    tdp.metadata.setdefault("tdp_cache_lineage_source_method", previous_source_method)
                tdp.metadata["tdp_cache_source_method"] = degree_source_method
            else:
                tdp.metadata.setdefault(
                    "tdp_cache_source_method",
                    tdp.metadata.get("source_method") or "unknown",
                )
            return tdp
        return None

    def _store_cached_tdp(
        self,
        sample: AgentBenchSample,
        cache_key: dict[str, Any],
        tdp: TrajectoryDependentDecisionProcess,
    ) -> None:
        """Persist a sampled TDP to shared sampling storage."""
        if self.shared_sampling_storage is None:
            return
        self.shared_sampling_storage.store(
            f"{self.name}/tdp",
            sample_id=sample.sample_id,
            key=cache_key,
            value={"tdp": asdict(tdp)},
        )

    def _replay_step_from_tdp_step(self, step: TDPStepRecord) -> StepRecord:
        """Convert a cached TDP step back into a replayable StepRecord."""
        messages: list[dict[str, Any]] = []
        raw_assistant_message = step.metadata.get("raw_assistant_message")
        if isinstance(raw_assistant_message, Mapping):
            messages.append(dict(raw_assistant_message))
        raw_observation_messages = step.metadata.get("raw_observation_messages")
        if isinstance(raw_observation_messages, list):
            messages.extend(dict(message) for message in raw_observation_messages if isinstance(message, Mapping))
        return StepRecord(
            index=step.index,
            action=step.realized_decision,
            observation=self._coerce_optional_string(step.metadata.get("observation")),
            messages=messages,
            metadata=dict(step.metadata) if isinstance(step.metadata, Mapping) else {},
        )

    async def _extend_cached_tdp(
        self,
        *,
        sample: AgentBenchSample,
        trajectory_index: int,
        cached_tdp: TrajectoryDependentDecisionProcess,
        per_step_samples: int,
        include_counterfactuals: bool = False,
    ) -> TrajectoryDependentDecisionProcess:
        """Extend a cached TDP using its explicitly requested extension protocol."""
        if cached_tdp.metadata.get("fixed_trajectory_candidate_extension_requested") is True:
            return await self._extend_cached_tdp_candidates_only(
                sample=sample,
                trajectory_index=trajectory_index,
                cached_tdp=cached_tdp,
                per_step_samples=per_step_samples,
                include_counterfactuals=include_counterfactuals,
            )

        replay_steps = [self._replay_step_from_tdp_step(step) for step in cached_tdp.steps]
        forced_sample_indices = self._tdp_sample_indices(cached_tdp)
        sampled_steps: list[AgentBenchSampledStep] = []

        for position, replay_step in enumerate(replay_steps):
            step_index = replay_step.index
            forced_sample_index = forced_sample_indices.get(step_index)
            if not isinstance(forced_sample_index, int):
                raise AgentBenchControllerError(
                    f"Frozen TDP cache for sample={sample.sample_id} trajectory={trajectory_index} is missing chosen sample metadata for step={step_index}."
                )
            rollout = await self._sample_rollout(
                sample=sample,
                rollout_id=f"tdp-{trajectory_index}",
                per_step_samples=per_step_samples,
                replay_steps=replay_steps[:position],
                start_step_index=step_index,
                stop_step_index=step_index,
                forced_sample_indices={step_index: forced_sample_index},
            )
            if len(rollout.steps) != 1:
                raise AgentBenchControllerError(
                    f"Strict TDP extension expected exactly one sampled step for sample={sample.sample_id} step={step_index}."
                )
            sampled_step = rollout.steps[0]
            if sampled_step.index != step_index:
                raise AgentBenchControllerError(
                    f"Strict TDP extension produced mismatched step index for sample={sample.sample_id}: expected {step_index}, got {sampled_step.index}."
                )
            sampled_steps.append(sampled_step)

        rollout = AgentBenchRollout(
            prompt=cached_tdp.prompt,
            steps=sampled_steps,
            final_answer=cached_tdp.final_answer,
            status=self._coerce_optional_string(cached_tdp.metadata.get("status")),
            result=dict(cached_tdp.metadata.get("result", {})) if isinstance(cached_tdp.metadata.get("result"), Mapping) else {},
            raw_output={},
            metadata={
                "executor": self.name,
                "rollout_id": f"tdp-{trajectory_index}",
                "used_no_tool_call_samples": any(bool(step.metadata.get("used_no_tool_call_samples")) for step in sampled_steps),
                "no_tool_call_sample_count": sum(
                    count
                    for step in sampled_steps
                    if isinstance((count := step.metadata.get("no_tool_call_sample_count")), int)
                ),
                "hard_requirement_failure": any(bool(step.metadata.get("hard_requirement_failure")) for step in sampled_steps),
                "hard_requirement_failure_reason": cached_tdp.metadata.get("hard_requirement_failure_reason"),
            },
        )
        counterfactual_records = (
            await self._build_tdp_counterfactual_records(
                sample=sample,
                rollout=rollout,
                per_step_samples=per_step_samples,
            )
            if include_counterfactuals
            else [[] for _ in rollout.steps]
        )
        return TrajectoryDependentDecisionProcess(
            sample_id=f"{sample.sample_id}-tdp-{trajectory_index}",
            prompt=rollout.prompt,
            steps=[
                self._build_tdp_step_record(
                    step,
                    counterfactual_records=self._counterfactual_records_for_step(
                        counterfactual_records,
                        rollout.steps,
                        step,
                    ),
                )
                for step in rollout.steps
            ],
            final_answer=rollout.final_answer,
            metadata={
                "executor": self.name,
                "task_name": sample.task_name,
                "task_index": sample.task_index,
                "status": rollout.status,
                "result": dict(rollout.result),
                "trajectory_index": trajectory_index,
                "per_step_samples": per_step_samples,
                "include_counterfactuals": include_counterfactuals,
                "counterfactual_per_step_samples": per_step_samples if include_counterfactuals else 0,
                "used_no_tool_call_samples": bool(rollout.metadata.get("used_no_tool_call_samples")),
                "no_tool_call_sample_count": rollout.metadata.get("no_tool_call_sample_count", 0),
                "hard_requirement_failure": bool(rollout.metadata.get("hard_requirement_failure")),
                "hard_requirement_failure_reason": rollout.metadata.get("hard_requirement_failure_reason"),
            },
        )

    async def _extend_cached_tdp_candidates_only(
        self,
        *,
        sample: AgentBenchSample,
        trajectory_index: int,
        cached_tdp: TrajectoryDependentDecisionProcess,
        per_step_samples: int,
        include_counterfactuals: bool = False,
    ) -> TrajectoryDependentDecisionProcess:
        """Append candidates at frozen TDP states without changing the trajectory."""
        if include_counterfactuals:
            raise AgentBenchControllerError(
                "Frozen candidate-only TDP extension does not support counterfactual regeneration."
            )

        replay_steps = [self._replay_step_from_tdp_step(step) for step in cached_tdp.steps]
        extended_steps: list[TDPStepRecord] = []

        for position, cached_step in enumerate(cached_tdp.steps):
            step_index = cached_step.index
            chosen_output_index = cached_step.metadata.get("chosen_output_index")
            if not isinstance(chosen_output_index, int):
                raise AgentBenchControllerError(
                    f"Frozen TDP cache for sample={sample.sample_id} trajectory={trajectory_index} is missing chosen sample metadata for step={step_index}."
                )

            existing_actions = list(cached_step.sampled_decisions)
            existing_messages = cached_step.metadata.get("sampled_messages")
            existing_metadata = cached_step.metadata.get("sampled_output_metadata")
            if not isinstance(existing_messages, list) or not isinstance(existing_metadata, list):
                raise AgentBenchControllerError(
                    f"Frozen TDP cache for sample={sample.sample_id} trajectory={trajectory_index} "
                    f"is missing persisted candidate payloads for step={step_index}."
                )
            existing_count = len(existing_actions)
            if (
                existing_count <= 0
                or len(existing_messages) != existing_count
                or len(existing_metadata) != existing_count
                or not 0 <= chosen_output_index < existing_count
            ):
                raise AgentBenchControllerError(
                    f"Frozen TDP candidate payloads are inconsistent for sample={sample.sample_id} "
                    f"trajectory={trajectory_index} step={step_index}."
                )
            if existing_count > per_step_samples:
                raise AgentBenchControllerError(
                    f"Frozen TDP for sample={sample.sample_id} trajectory={trajectory_index} "
                    f"already has {existing_count} candidates, exceeding requested N={per_step_samples}."
                )

            additional_count = per_step_samples - existing_count
            additional_samples: list[StepSample] = []
            if additional_count:
                session = await self._start_session_with_prefix(sample, replay_steps[:position])
                try:
                    generated = await self._generate_step_samples(
                        messages=session.messages,
                        tools=session.tools,
                        sample_count=additional_count,
                    )
                finally:
                    await asyncio.to_thread(
                        self.controller_client._cancel,
                        session.session_id,
                        prefer_header=True,
                    )
                additional_samples = self._prepare_step_samples(
                    sample_id=sample.sample_id,
                    context_id=f"tdp-{trajectory_index}-fixed-extension",
                    step_index=step_index,
                    sampled_step_samples=generated,
                )
                if len(additional_samples) != additional_count:
                    raise AgentBenchControllerError(
                        f"Expected {additional_count} additional candidates for sample={sample.sample_id} "
                        f"trajectory={trajectory_index} step={step_index}, received {len(additional_samples)}."
                    )

            combined_actions = [
                *existing_actions,
                *[step_sample.output for step_sample in additional_samples],
            ]
            combined_messages = [
                *[dict(message) if isinstance(message, Mapping) else {} for message in existing_messages],
                *[
                    dict(step_sample.recovery_state.get("assistant_message", {}))
                    for step_sample in additional_samples
                ],
            ]
            combined_metadata = [
                *[dict(value) if isinstance(value, Mapping) else {} for value in existing_metadata],
                *[dict(step_sample.metadata) for step_sample in additional_samples],
            ]
            entropy, _ = compute_predictive_entropy_from_metadata(combined_metadata)
            measurements = dict(cached_step.uncertainty_measurements)
            measurements["pe"] = 0.0 if entropy is None else float(entropy)
            metadata = dict(cached_step.metadata)
            no_tool_call_sample_count = sum(
                not self._assistant_message_has_callable_tool_call(message)
                for message in combined_messages
            )
            metadata.update(
                {
                    "sampled_messages": combined_messages,
                    "sampled_output_metadata": combined_metadata,
                    "used_no_tool_call_samples": no_tool_call_sample_count > 0,
                    "no_tool_call_sample_count": no_tool_call_sample_count,
                    "callable_sample_count": len(combined_messages)
                    - no_tool_call_sample_count,
                    "fixed_trajectory_candidate_extension": True,
                    "fixed_trajectory_source_candidate_count": existing_count,
                    "fixed_trajectory_added_candidate_count": additional_count,
                }
            )
            extended_steps.append(
                TDPStepRecord(
                    index=cached_step.index,
                    realized_decision=cached_step.realized_decision,
                    sampled_decisions=combined_actions,
                    uncertainty_measurements=measurements,
                    counterfactual_records=list(cached_step.counterfactual_records),
                    metadata=metadata,
                )
            )

        metadata = dict(cached_tdp.metadata)
        total_no_tool_call_samples = sum(
            int(step.metadata.get("no_tool_call_sample_count", 0))
            for step in extended_steps
        )
        metadata.update(
            {
                "trajectory_index": trajectory_index,
                "per_step_samples": per_step_samples,
                "include_counterfactuals": False,
                "counterfactual_per_step_samples": 0,
                "fixed_trajectory_candidate_extension": True,
                "fixed_trajectory_source_per_step_samples": cached_tdp.metadata.get(
                    "per_step_samples"
                ),
                "used_no_tool_call_samples": total_no_tool_call_samples > 0,
                "no_tool_call_sample_count": total_no_tool_call_samples,
            }
        )
        return TrajectoryDependentDecisionProcess(
            sample_id=cached_tdp.sample_id,
            prompt=cached_tdp.prompt,
            steps=extended_steps,
            final_answer=cached_tdp.final_answer,
            metadata=metadata,
        )

    def _load_cached_backbone(
        self,
        sample: AgentBenchSample,
        cache_key: dict[str, Any],
    ) -> BackboneTrajectory | None:
        """Load a cached backbone trajectory from shared sampling storage."""
        if self.shared_sampling_storage is None:
            return None
        payload = self.shared_sampling_storage.load(f"{self.name}/backbone", sample_id=sample.sample_id, key=cache_key)
        if payload is None:
            return None
        backbone_payload = payload.get("backbone")
        if not isinstance(backbone_payload, Mapping):
            return None
        return self._deserialize_backbone(backbone_payload)

    def _store_cached_backbone(
        self,
        sample: AgentBenchSample,
        cache_key: dict[str, Any],
        backbone: BackboneTrajectory,
    ) -> None:
        """Persist a backbone trajectory to shared sampling storage."""
        if self.shared_sampling_storage is None:
            return
        self.shared_sampling_storage.store(
            f"{self.name}/backbone",
            sample_id=sample.sample_id,
            key=cache_key,
            value={"backbone": asdict(backbone)},
        )

    def _deserialize_backbone(self, payload: Mapping[str, Any]) -> BackboneTrajectory:
        """Rebuild a backbone trajectory from cached serialized data."""
        return BackboneTrajectory(
            sample_id=str(payload.get("sample_id", "")),
            prompt=str(payload.get("prompt", "")),
            steps=[self._deserialize_step_record(step) for step in payload.get("steps", [])],
            final_answer=self._coerce_optional_string(payload.get("final_answer")),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {},
        )

    def _deserialize_step_record(self, payload: Any) -> StepRecord:
        """Rebuild a generic StepRecord from cached serialized data."""
        if not isinstance(payload, Mapping):
            raise ValueError("Invalid cached step record.")
        messages = payload.get("messages")
        resolved_messages = [dict(message) for message in messages if isinstance(message, Mapping)] if isinstance(messages, list) else []
        metadata = payload.get("metadata")
        return StepRecord(
            index=int(payload.get("index", 0)),
            thought=self._coerce_optional_string(payload.get("thought")),
            action=self._coerce_optional_string(payload.get("action")),
            observation=self._coerce_optional_string(payload.get("observation")),
            messages=resolved_messages,
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )

    def _backbone_sample_indices(self, backbone: BackboneTrajectory) -> dict[int, int]:
        """Recover chosen sampled-action indices from a cached backbone."""
        sample_indices: dict[int, int] = {}
        for step in backbone.steps:
            chosen_output_index = step.metadata.get("chosen_output_index")
            if isinstance(chosen_output_index, int) and chosen_output_index >= 0:
                sample_indices[step.index] = chosen_output_index
        return sample_indices

    def _backbone_raw_output(self, trajectory: BackboneTrajectory) -> dict[str, Any]:
        """Reconstruct an AgentBench-like raw output payload from a trajectory."""
        history: list[dict[str, Any]] = []
        if trajectory.prompt:
            for prompt_part in str(trajectory.prompt).split("\n\n"):
                if prompt_part:
                    history.append({"role": "user", "content": prompt_part})
        for step in trajectory.steps:
            history.extend(dict(message) for message in step.messages if isinstance(message, Mapping))

        raw_output: dict[str, Any] = {"history": history}
        status = trajectory.metadata.get("status")
        if status is not None:
            raw_output["status"] = status
        result = trajectory.metadata.get("result")
        if isinstance(result, Mapping):
            raw_output["result"] = dict(result)
        protocol = trajectory.metadata.get("protocol")
        if protocol is not None:
            raw_output["protocol"] = protocol
        return raw_output

    def _build_backbone_from_rollout(
        self,
        sample: AgentBenchSample,
        rollout: AgentBenchRollout,
    ) -> BackboneTrajectory:
        """Convert a rollout into the common BackboneTrajectory format."""
        return BackboneTrajectory(
            sample_id=f"{sample.sample_id}-backbone",
            prompt=rollout.prompt,
            steps=[self._build_backbone_step_record(step) for step in rollout.steps],
            final_answer=rollout.final_answer,
            metadata={
                "executor": self.name,
                "task_name": sample.task_name,
                "task_index": sample.task_index,
                "controller_url": sample.controller_url,
                "status": rollout.status,
                "result": dict(rollout.result),
                "step_entropies": [step.entropy for step in rollout.steps],
                "per_step_samples": self.backbone_per_step_samples,
                "protocol": rollout.raw_output.get("protocol"),
                "used_no_tool_call_samples": bool(rollout.metadata.get("used_no_tool_call_samples")),
                "no_tool_call_sample_count": rollout.metadata.get("no_tool_call_sample_count", 0),
                "hard_requirement_failure": bool(rollout.metadata.get("hard_requirement_failure")),
                "hard_requirement_failure_reason": rollout.metadata.get("hard_requirement_failure_reason"),
            },
        )

    def backbone_from_cached_tdp(
        self,
        sample: AgentBenchSample,
        tdp: TrajectoryDependentDecisionProcess,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> BackboneTrajectory:
        """Convert a cached TDP trajectory into a replayable local-branching backbone."""
        raw_tdp_steps = list(tdp.steps)
        effective_tdp_steps = self._effective_cached_backbone_steps(tdp)
        steps = [self._replay_step_from_tdp_step(step) for step in effective_tdp_steps]
        step_entropies: list[float] = []
        for step in effective_tdp_steps:
            pe = step.uncertainty_measurements.get("pe")
            if not isinstance(pe, (int, float)) or not math.isfinite(float(pe)):
                pe = step.metadata.get("pe")
            step_entropies.append(float(pe) if isinstance(pe, (int, float)) and math.isfinite(float(pe)) else 0.0)

        tdp_metadata = dict(tdp.metadata)
        no_tool_count = sum(
            count
            for step in steps
            if isinstance((count := step.metadata.get("no_tool_call_sample_count")), int)
        )
        metadata_no_tool_count = tdp_metadata.get("no_tool_call_sample_count")
        backbone_metadata: dict[str, Any] = {
            "executor": self.name,
            "task_name": sample.task_name,
            "task_index": sample.task_index,
            "controller_url": sample.controller_url,
            "status": tdp_metadata.get("status"),
            "result": dict(tdp_metadata.get("result", {})) if isinstance(tdp_metadata.get("result"), Mapping) else {},
            "step_entropies": step_entropies,
            "per_step_samples": tdp_metadata.get("per_step_samples", 1),
            "used_no_tool_call_samples": bool(tdp_metadata.get("used_no_tool_call_samples"))
            or any(bool(step.metadata.get("used_no_tool_call_samples")) for step in steps),
            "no_tool_call_sample_count": (
                self._coerce_nonnegative_int(metadata_no_tool_count)
                if metadata_no_tool_count is not None
                else no_tool_count
            ),
            "hard_requirement_failure": bool(tdp_metadata.get("hard_requirement_failure"))
            or any(bool(step.metadata.get("hard_requirement_failure")) for step in steps),
            "hard_requirement_failure_reason": tdp_metadata.get("hard_requirement_failure_reason"),
            "cached_tdp_sample_id": tdp.sample_id,
            "cached_tdp_metadata": tdp_metadata,
            "raw_cached_tdp_length": len(raw_tdp_steps),
            "effective_cached_tdp_length": len(effective_tdp_steps),
            "excluded_cached_synthetic_terminal_step": len(effective_tdp_steps) < len(raw_tdp_steps),
        }
        backbone_metadata.update(dict(metadata or {}))
        return BackboneTrajectory(
            sample_id=f"{sample.sample_id}-cached-tdp-backbone",
            prompt=tdp.prompt,
            steps=steps,
            final_answer=tdp.final_answer,
            metadata=backbone_metadata,
        )

    def _effective_cached_backbone_steps(
        self,
        tdp: TrajectoryDependentDecisionProcess,
    ) -> list[TDPStepRecord]:
        """Return cached TDP steps after dropping one explicit synthetic terminal step."""
        steps = list(tdp.steps)
        if len(steps) <= 1:
            return steps
        last_step = steps[-1]
        if self._is_synthetic_cached_terminal_step(last_step, tdp.metadata):
            return steps[:-1]
        return steps

    def _is_synthetic_cached_terminal_step(
        self,
        step: TDPStepRecord,
        tdp_metadata: Mapping[str, Any] | None,
    ) -> bool:
        """Identify artificial forced-finalization steps in reusable TDP caches."""
        metadata = step.metadata if isinstance(step.metadata, Mapping) else {}
        if bool(metadata.get("forced_terminal_finish")) or isinstance(metadata.get("forced_terminal_finish_reason"), str):
            return True
        if bool(metadata.get("synthetic_terminal_step") or metadata.get("forced_terminal_step") or metadata.get("task_limit_terminal_step")):
            return True

        decision = str(step.realized_decision or "").lower()
        if "finish_action" not in decision:
            return False
        if "reached the maximum rollout length" in decision and "official environment checker" in decision:
            return True
        if not isinstance(tdp_metadata, Mapping):
            return False
        status = tdp_metadata.get("status")
        task_limit_status = isinstance(status, str) and "task" in status.lower() and "limit" in status.lower()
        return task_limit_status and (
            "task limit" in decision
            or "maximum rollout" in decision
            or "max step" in decision
            or "max_steps" in decision
        )

    def _assistant_message_from_step(self, step: StepRecord) -> dict[str, Any]:
        """Recover the raw assistant message needed to replay a recorded step."""
        if step.messages:
            first = step.messages[0]
            if isinstance(first, Mapping) and str(first.get("role", "")) == "assistant":
                return dict(first)
        raw_message = step.metadata.get("raw_assistant_message")
        if isinstance(raw_message, Mapping):
            return dict(raw_message)
        raise AgentBenchControllerError("Cannot replay dbbench step because the assistant message was not preserved.")

    def _normalize_replay_messages(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Normalize observation messages before strict replay comparison."""
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            normalized.append(self._normalize_nested_mapping(self._cache_key_message(message)))
        return normalized

    def _normalize_nested_mapping(self, value: Any) -> Any:
        """Recursively normalize values for cache keys and replay comparison."""
        if isinstance(value, Mapping):
            return {str(key): self._normalize_nested_mapping(nested) for key, nested in sorted(value.items())}
        if isinstance(value, list):
            return [self._normalize_nested_mapping(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\r\n", "\n").replace("\r", "\n")
            if self.name == "agentbench_dbbench":
                normalized = _DBBENCH_SESSION_DATABASE_PATTERN.sub("dbbench_<session>", normalized)
            return normalized
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)

    def _session_finished(self, session: AgentBenchFunctionCallingSession) -> bool:
        """Return whether a session final_state marks the task as finished."""
        return bool(session.final_state and session.final_state.get("finish"))

    def _extract_prompt(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """Extract initial user prompt text from a message history."""
        prompt_parts: list[str] = []
        for item in messages:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role", ""))
            if role == "system":
                continue
            if role != "user":
                break
            prompt_parts.append(str(item.get("content", "")))
        return "\n\n".join(part for part in prompt_parts if part)

    def _format_observation(self, observation_messages: Sequence[Mapping[str, Any]]) -> str | None:
        """Join tool/observation message content into one readable string."""
        parts = [str(message.get("content", "")) for message in observation_messages if isinstance(message, Mapping)]
        combined = "\n\n".join(part for part in parts if part)
        return combined or None

    def _is_finish_action(self, action: str) -> bool:
        """Return whether a formatted action submits the DBBench final answer."""
        return "commit_final_answer" in action.lower()

    def _build_trajectory(
        self,
        sample: AgentBenchSample,
        output: Mapping[str, Any] | None,
    ) -> BackboneTrajectory:
        """Parse an AgentBench raw output history into a BackboneTrajectory."""
        history = output.get("history") if isinstance(output, Mapping) else None
        history_items = list(history) if isinstance(history, list) else []
        initial_messages: list[dict[str, Any]] = []
        prompt_parts: list[str] = []
        cursor = 0
        while cursor < len(history_items):
            item = history_items[cursor]
            if not isinstance(item, Mapping):
                cursor += 1
                continue
            if str(item.get("role", "")) == "system":
                initial_messages.append(dict(item))
                cursor += 1
                continue
            if str(item.get("role", "")) != "user":
                break
            initial_messages.append(dict(item))
            prompt_parts.append(str(item.get("content", "")))
            cursor += 1

        steps: list[StepRecord] = []
        step_index = 1
        while cursor < len(history_items):
            item = history_items[cursor]
            cursor += 1
            if not isinstance(item, Mapping):
                continue
            if str(item.get("role", "")) != "assistant":
                continue

            response_text = self._format_assistant_message(item)
            observation_messages: list[dict[str, Any]] = []
            while cursor < len(history_items):
                next_item = history_items[cursor]
                if not isinstance(next_item, Mapping):
                    cursor += 1
                    continue
                if str(next_item.get("role", "")) == "assistant":
                    break
                observation_messages.append(dict(next_item))
                cursor += 1

            steps.append(
                StepRecord(
                    index=step_index,
                    action=response_text,
                    observation=self._format_observation(observation_messages),
                    messages=[dict(item), *observation_messages],
                    metadata={
                        "task_name": sample.task_name,
                        "task_index": sample.task_index,
                        "tool_calls": item.get("tool_calls") if isinstance(item.get("tool_calls"), list) else [],
                        "raw_assistant_message": dict(item),
                    },
                )
            )
            step_index += 1

        raw_result = output.get("result") if isinstance(output, Mapping) else None
        final_answer: str | None = None
        if isinstance(raw_result, Mapping):
            final_answer = self._normalize_final_answer(raw_result.get("answer"))
        if final_answer is None:
            final_answer = self.controller_client._extract_final_answer(history_items)

        return BackboneTrajectory(
            sample_id=sample.sample_id,
            prompt="\n\n".join(part for part in prompt_parts if part),
            steps=steps,
            final_answer=final_answer,
            metadata={
                "executor": self.name,
                "task_name": sample.task_name,
                "task_index": sample.task_index,
                "controller_url": sample.controller_url,
                "status": output.get("status") if isinstance(output, Mapping) else None,
                "history_length": len(history_items),
                "protocol": output.get("protocol") if isinstance(output, Mapping) else None,
                "result": dict(raw_result) if isinstance(raw_result, Mapping) else {},
                "initial_messages": initial_messages,
            },
        )

    def _format_assistant_message(self, message: Mapping[str, Any]) -> str:
        """Render assistant content and tool calls as a compact action string."""
        content = message.get("content")
        parts: list[str] = []
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, Mapping):
                    continue
                name = str(function.get("name", ""))
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    parts.append(f"Tool: {name}({arguments})")
                else:
                    parts.append(f"Tool: {name}")
        return "\n".join(part for part in parts if part)

    def _normalize_final_answer(self, answer: Any) -> str | None:
        """Normalize final-answer payloads to optional strings."""
        if answer is None:
            return None
        if isinstance(answer, list):
            if len(answer) == 1 and isinstance(answer[0], str):
                return answer[0]
            return json.dumps(answer, ensure_ascii=True)
        return str(answer)

    def _coerce_optional_string(self, value: Any) -> str | None:
        """Convert a value to string while preserving None."""
        if value is None:
            return None
        return str(value)

    def _coerce_nonnegative_int(self, value: Any) -> int:
        """Return a nonnegative integer for trusted metadata counters."""
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return max(0, value)
        return 0
