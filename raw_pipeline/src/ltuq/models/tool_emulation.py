from __future__ import annotations

import json
import logging
import os
import re
import copy
from collections.abc import Mapping, Sequence
from typing import Any

from .base import ChatModelInterface


LOGGER = logging.getLogger(__name__)


class ToolCallEmulationChatModelClient(ChatModelInterface):
    def __init__(self, wrapped_client: ChatModelInterface) -> None:
        self.wrapped_client = wrapped_client
        self.model = getattr(wrapped_client, "model", "unknown")

    async def ainference(
        self,
        history: list[dict[str, str]],
        temperature: float = 1.0,
    ) -> str:
        return await self.wrapped_client.ainference(history, temperature=temperature)

    async def sample_many(
        self,
        history: list[dict[str, str]],
        temperature: float = 1.0,
        n: int = 1,
    ) -> list[str]:
        return await self.wrapped_client.sample_many(history, temperature=temperature, n=n)

    async def sample_many_detailed(
        self,
        history: list[dict[str, str]],
        temperature: float = 1.0,
        n: int = 1,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        detailed_callable = getattr(self.wrapped_client, "sample_many_detailed", None)
        if callable(detailed_callable):
            try:
                return await detailed_callable(
                    history,
                    temperature=temperature,
                    n=n,
                    extra_body=extra_body,
                    max_tokens=max_tokens,
                )
            except TypeError:
                if extra_body is not None or max_tokens is not None:
                    LOGGER.warning(
                        "Wrapped model client does not accept detailed sampling controls; falling back to plain sampling."
                    )
                return await detailed_callable(history, temperature=temperature, n=n)
        return [
            {"output": output, "metadata": {"logprobs_unavailable": True}}
            for output in await self.wrapped_client.sample_many(history, temperature=temperature, n=n)
        ]

    async def acompletion_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        tool_choice: Any | None = None,
    ) -> dict[str, Any]:
        return (
            await self.acompletion_with_tools_many(
                messages,
                tools,
                temperature=temperature,
                n=1,
                tool_choice=tool_choice,
            )
        )[0]

    async def acompletion_with_tools_many(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        n: int = 1,
        tool_choice: Any | None = None,
    ) -> list[dict[str, Any]]:
        detailed_outputs = await self.acompletion_with_tools_many_detailed(
            messages,
            tools,
            temperature=temperature,
            n=n,
            tool_choice=tool_choice,
        )
        return [
            {key: value for key, value in item.items() if key != "metadata"}
            for item in detailed_outputs
        ]

    async def acompletion_with_tools_detailed(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        tool_choice: Any | None = None,
    ) -> dict[str, Any]:
        return (
            await self.acompletion_with_tools_many_detailed(
                messages,
                tools,
                temperature=temperature,
                n=1,
                tool_choice=tool_choice,
            )
        )[0]

    async def acompletion_with_tools_many_detailed(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 1.0,
        n: int = 1,
        tool_choice: Any | None = None,
    ) -> list[dict[str, Any]]:
        del tool_choice
        if n <= 0:
            raise ValueError("n must be positive")

        emulation_history = self._build_emulation_history(messages, tools)
        outputs = await self._sample_emulated_tool_outputs(
            emulation_history,
            tools,
            temperature=temperature,
            n=n,
        )
        return [self._normalize_emulated_output(item, tools) for item in outputs]

    async def aclose(self) -> None:
        await self.wrapped_client.aclose()

    def _build_emulation_history(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        merged_system_parts: list[str] = []
        conversation_messages: list[dict[str, str]] = []
        for rendered in self._render_history_messages(messages):
            if rendered["role"] in {"system", "developer"} and rendered["content"].strip():
                merged_system_parts.append(rendered["content"])
                continue
            conversation_messages.append(rendered)

        merged_system_parts.append(self._build_emulation_instruction(tools))
        history: list[dict[str, str]] = [
            {
                "role": "system",
                "content": "\n\n".join(merged_system_parts),
            }
        ]
        history.extend(conversation_messages)
        history.append(
            {
                "role": "user",
                "content": self._build_emulation_response_reminder(tools),
            }
        )
        return history

    def _build_emulation_response_reminder(self, tools: Sequence[Mapping[str, Any]]) -> str:
        final_answer_clause = ""
        if self._allows_plain_final_answer(tools):
            final_answer_clause = (
                " If the final answer is already known and the task requires Final Answer: #N, "
                "reply with exactly that final-answer text and nothing else."
            )
        single_action_clause = ""
        action_tool_name = self._single_action_tool_name(tools)
        if action_tool_name is not None:
            single_action_clause = (
                " When selecting the action tool, emit JSON exactly in this shape: "
                '{"name":"' + action_tool_name + '","arguments":{"action":"<verbatim action>"}}. '
                "Do not move the action text into name, content, or prose."
            )
        return (
            "Given the conversation above, emit only the next assistant action now. "
            "If a tool is needed, reply with exactly one JSON object matching "
            '{"name":"<tool name>","arguments":{...}} using the allowed tool schema. '
            "Do not include prose, markdown, or explanation outside the JSON object."
            f"{single_action_clause}"
            f"{final_answer_clause}"
        )

    def _build_emulation_instruction(self, tools: Sequence[Mapping[str, Any]]) -> str:
        summarized_tools = []
        for tool in tools:
            if not isinstance(tool, Mapping):
                continue
            function = tool.get("function")
            if not isinstance(function, Mapping):
                continue
            summarized_tools.append(
                {
                    "name": str(function.get("name", "")),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                }
            )
        return (
            "You are operating in emulated tool-calling mode.\n"
            "Your entire reply MUST be a single valid JSON object with no surrounding text.\n"
            "Do not output explanations, prose, markdown fences, or chain-of-thought outside the JSON object.\n"
            "The conversation may ask you to think first. Do not write that thought in plain text. Translate it directly into the tool call JSON.\n"
            "If you want to keep a short rationale, put it inside an optional top-level field named \"content\" within the JSON object.\n"
            "The required shape is {\"name\": \"<tool name>\", \"arguments\": {...}}.\n"
            "Rules:\n"
            "- Choose exactly one tool from the allowed list.\n"
            "- arguments must be a JSON object, not a string.\n"
            "- The reply must be parseable by json.loads without preprocessing.\n"
            "- Never answer with text like 'I need to inspect the file'. Emit the actual tool call instead.\n"
            "Valid reply examples:\n"
            "{\"name\": \"<tool name>\", \"arguments\": {\"arg\": \"value\"}}\n"
            "{\"name\": \"<tool name>\", \"arguments\": {\"arg\": \"value\"}, \"content\": \"brief rationale\"}\n"
            f"Allowed tools: {json.dumps(summarized_tools, ensure_ascii=True, sort_keys=True)}"
        )

    async def _sample_emulated_tool_outputs(
        self,
        history: list[dict[str, str]],
        tools: Sequence[Mapping[str, Any]],
        *,
        temperature: float,
        n: int,
    ) -> list[dict[str, Any]]:
        """Sample emulated tool-call text, optionally with vLLM guided JSON."""
        extra_body = self._structured_outputs_extra_body(tools)
        if extra_body is None:
            return await self.sample_many_detailed(history, temperature=temperature, n=n)

        detailed_callable = getattr(self.wrapped_client, "sample_many_detailed", None)
        if callable(detailed_callable):
            try:
                outputs = await detailed_callable(history, temperature=temperature, n=n, extra_body=extra_body)
                return [self._mark_structured_outputs_metadata(output) for output in outputs]
            except TypeError:
                LOGGER.warning(
                    "Wrapped model client does not accept structured-output extra_body; falling back to prompt-only tool emulation."
                )
        return await self.sample_many_detailed(history, temperature=temperature, n=n)

    def _mark_structured_outputs_metadata(self, output: Mapping[str, Any]) -> dict[str, Any]:
        """Annotate one raw sampled output as using structured-output decoding."""
        marked = dict(output)
        metadata = dict(marked.get("metadata", {})) if isinstance(marked.get("metadata"), Mapping) else {}
        metadata["structured_outputs_requested"] = True
        marked["metadata"] = metadata
        return marked

    def _structured_outputs_extra_body(self, tools: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        """Return a vLLM structured-output request body when enabled by environment."""
        if os.getenv("LTUQ_EMULATED_TOOL_GUIDED_JSON", "").lower() not in {"1", "true", "yes", "on"}:
            return None

        names = [
            str(function.get("name"))
            for tool in tools
            if isinstance(tool, Mapping)
            and isinstance((function := tool.get("function")), Mapping)
            and isinstance(function.get("name"), str)
        ]
        if not names:
            return None

        # Preserve requirements shared by every tool in the guided arguments
        # schema. This keeps shared fields a decoding invariant while still
        # allowing tool-specific arguments.
        common_required: set[str] | None = None
        parameter_properties: list[Mapping[str, Any]] = []
        for tool in tools:
            function = tool.get("function") if isinstance(tool, Mapping) else None
            parameters = function.get("parameters") if isinstance(function, Mapping) else None
            if not isinstance(parameters, Mapping):
                common_required = set()
                parameter_properties.append({})
                continue
            required = parameters.get("required")
            required_names = {
                str(value)
                for value in required
                if isinstance(value, str)
            } if isinstance(required, Sequence) and not isinstance(required, (str, bytes)) else set()
            common_required = required_names if common_required is None else common_required & required_names
            properties = parameters.get("properties")
            parameter_properties.append(properties if isinstance(properties, Mapping) else {})

        shared_names = sorted(common_required or set())
        shared_properties: dict[str, Any] = {}
        for name in shared_names:
            for properties in parameter_properties:
                candidate = properties.get(name)
                if isinstance(candidate, Mapping):
                    shared_properties[name] = copy.deepcopy(dict(candidate))
                    break

        arguments_schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": True,
        }
        if shared_names:
            arguments_schema["properties"] = shared_properties
            arguments_schema["required"] = shared_names

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": names},
                "arguments": arguments_schema,
                "content": {"type": "string"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        return {"structured_outputs": {"json": schema}}

    def _render_history_messages(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
        rendered: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role", "user"))
            if role == "agent":
                role = "assistant"
            elif role == "tool":
                role = "user"
            elif role not in {"system", "user", "assistant", "tool", "developer"}:
                role = "user"
            rendered.append(
                {
                    "role": role,
                    "content": self._render_message_content(message),
                }
            )
        return rendered

    def _render_message_content(self, message: Mapping[str, Any]) -> str:
        if str(message.get("role", "")) == "tool":
            tool_call_id = message.get("tool_call_id")
            prefix = "Tool result"
            if isinstance(tool_call_id, str) and tool_call_id:
                prefix = f"Tool result for {tool_call_id}"
            content = message.get("content")
            if isinstance(content, str):
                return f"{prefix}:\n{content}"
            if content is None:
                return prefix
            return f"{prefix}:\n{content}"

        tool_calls = message.get("tool_calls")
        content = message.get("content")
        if isinstance(tool_calls, list):
            payload: dict[str, Any] = {
                "role": str(message.get("role", "assistant")),
                "tool_calls": [dict(tool_call) for tool_call in tool_calls if isinstance(tool_call, Mapping)],
            }
            if content is not None:
                payload["content"] = content
            return json.dumps(payload, ensure_ascii=True, sort_keys=True)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [getattr(part, "text", "") for part in content]
            return "".join(text_parts)
        if content is None:
            return ""
        return str(content)

    def _normalize_emulated_output(self, item: Mapping[str, Any], tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        output = item.get("output") if isinstance(item, Mapping) else None
        raw_output = output if isinstance(output, str) else str(output or "")
        metadata = dict(item.get("metadata", {})) if isinstance(item.get("metadata"), Mapping) else {}
        metadata["tool_call_emulated"] = True
        assistant_message, parse_metadata = self._parse_emulated_message(raw_output, tools)
        metadata.update(parse_metadata)
        return {
            **assistant_message,
            "metadata": metadata,
        }

    def _parse_emulated_message(
        self,
        raw_output: str,
        tools: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = self._decode_json_payload(raw_output)
        if not isinstance(payload, Mapping):
            recovered_tool_call = self._recover_tool_call_from_text(raw_output, tools)
            if recovered_tool_call is not None:
                return recovered_tool_call, {"emulated_tool_call_recovered": "text_pattern"}
            return (
                {"role": "assistant", "content": raw_output},
                {"emulated_tool_call_parse_error": "invalid_json"},
            )

        tool_call = self._extract_tool_call_payload(payload)
        if not isinstance(tool_call, Mapping):
            recovered_tool_call = self._recover_tool_call_from_text(raw_output, tools)
            if recovered_tool_call is not None:
                return recovered_tool_call, {"emulated_tool_call_recovered": "text_pattern"}
            return (
                {"role": "assistant", "content": raw_output},
                {"emulated_tool_call_parse_error": "missing_tool_call"},
            )

        function = tool_call.get("function") if isinstance(tool_call.get("function"), Mapping) else tool_call
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not name:
            return (
                {"role": "assistant", "content": raw_output},
                {"emulated_tool_call_parse_error": "missing_name"},
            )

        allowed_names = {
            str(tool.get("function", {}).get("name", ""))
            for tool in tools
            if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
        }
        arguments = self._normalize_arguments(function.get("arguments") if isinstance(function, Mapping) else None)
        recovery_metadata: dict[str, Any] = {}
        if allowed_names and name not in allowed_names:
            recovered = self._recover_single_action_tool_call(name, arguments, tools)
            if recovered is None:
                return (
                    {"role": "assistant", "content": raw_output},
                    {
                        "emulated_tool_call_parse_error": "invalid_name",
                        "emulated_tool_call_invalid_name": name,
                    },
                )
            name, arguments, recovery_metadata = recovered

        if arguments is None:
            return (
                {"role": "assistant", "content": raw_output},
                {"emulated_tool_call_parse_error": "invalid_arguments"},
            )

        content = payload.get("content") if isinstance(payload.get("content"), str) else None
        message: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": str(tool_call.get("id", "emulated-tool-call-0")),
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=True, sort_keys=True),
                    },
                }
            ],
        }
        if content:
            message["content"] = content
        return message, recovery_metadata

    def _extract_tool_call_payload(self, payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, Mapping):
                    return tool_call

        function = payload.get("function")
        if isinstance(function, Mapping):
            return {
                "id": payload.get("id", "emulated-tool-call-0"),
                "type": "function",
                "function": function,
            }

        if isinstance(payload.get("name"), str):
            return {
                "id": payload.get("id", "emulated-tool-call-0"),
                "type": "function",
                "function": {
                    "name": payload.get("name"),
                    "arguments": payload.get("arguments"),
                },
            }

        for key in ("assistant_message", "message", "response"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                extracted = self._extract_tool_call_payload(nested)
                if extracted is not None:
                    return extracted
        return None

    def _normalize_arguments(self, arguments: Any) -> dict[str, Any] | None:
        if isinstance(arguments, Mapping):
            return dict(arguments)
        if not isinstance(arguments, str):
            return None
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, str):
            try:
                decoded = json.loads(decoded)
            except json.JSONDecodeError:
                return None
        return dict(decoded) if isinstance(decoded, Mapping) else None

    def _recover_single_action_tool_call(
        self,
        invalid_name: str,
        arguments: Mapping[str, Any] | None,
        tools: Sequence[Mapping[str, Any]],
    ) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
        action_tool_name = self._single_action_tool_name(tools)
        if action_tool_name is None:
            return None

        normalized_arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        action = normalized_arguments.get("action")
        if action is None:
            if normalized_arguments:
                return None
            action = invalid_name
        if not isinstance(action, str) or not action.strip():
            return None

        normalized_arguments["action"] = action
        return (
            action_tool_name,
            normalized_arguments,
            {
                "emulated_tool_call_recovered": "single_action_tool_name",
                "emulated_tool_call_invalid_name": invalid_name,
            },
        )

    def _single_action_tool_name(self, tools: Sequence[Mapping[str, Any]]) -> str | None:
        functions = [
            tool.get("function")
            for tool in tools
            if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
        ]
        if len(functions) != 1:
            return None

        function = functions[0]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            return None
        if name == "take_action":
            return name

        parameters = function.get("parameters")
        if not isinstance(parameters, Mapping):
            return None
        properties = parameters.get("properties")
        if not isinstance(properties, Mapping):
            return None
        action_property = properties.get("action")
        if not isinstance(action_property, Mapping):
            return None
        if action_property.get("type") not in {None, "string"}:
            return None
        required = parameters.get("required")
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes, bytearray)) and "action" not in required:
            return None
        return name

    def _decode_json_payload(self, raw_output: str) -> Any:
        stripped = raw_output.strip()
        for candidate in self._json_candidates(stripped):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                prefix_payload = self._decode_json_prefix_with_trailing_closers(candidate)
                if prefix_payload is not None:
                    return prefix_payload
        repaired = self._repair_common_json_errors(stripped)
        if repaired is not None:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        for candidate in self._json_candidates(stripped):
            prefix_payload = self._decode_first_json_object(candidate)
            if prefix_payload is not None:
                return prefix_payload
        LOGGER.debug("Failed to parse emulated tool-call JSON for model=%s output=%r", self.model, raw_output)
        return None

    def _decode_first_json_object(self, candidate: str) -> Any | None:
        """Return the first JSON object when a model appends extra text.

        GPT-OSS sometimes emits a valid emulated tool call and then continues
        with another object or prose. For tool execution, the first complete
        object is the actionable decision; rejecting it turns recoverable model
        chatter into a hard no-tool-call failure.
        """
        start = candidate.find("{")
        if start == -1:
            return None
        try:
            payload, _ = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, Mapping) else None

    def _decode_json_prefix_with_trailing_closers(self, candidate: str) -> Any | None:
        try:
            payload, end_index = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError:
            return None
        trailing = candidate[end_index:].strip()
        if trailing and any(character != "}" for character in trailing):
            return None
        return payload

    def _json_candidates(self, text: str) -> list[str]:
        candidates = [text]
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                candidates.append("\n".join(lines[1:-1]).strip())
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])
        deduped: list[str] = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                deduped.append(candidate)
                seen.add(candidate)
        return deduped

    def _repair_common_json_errors(self, text: str) -> str | None:
        repaired = text.strip()
        if not repaired:
            return None
        repaired = re.sub(r'\}\s*,\s*"content"\s*:', ', "content":', repaired, count=1)
        repaired = re.sub(r'\}\s*,\s*"role"\s*:', ', "role":', repaired, count=1)
        return repaired if repaired != text.strip() else None

    def _recover_tool_call_from_text(
        self,
        raw_output: str,
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        allowed_names = {
            str(tool.get("function", {}).get("name", ""))
            for tool in tools
            if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
        }
        text = raw_output.strip()
        single_action_call = self._recover_single_action_tool_call_from_text(text, tools)
        if single_action_call is not None:
            return single_action_call
        patterns = [
            r'^(?:Tool|Action)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\((\{.*\})\)\s*$',
            r'^([A-Za-z_][A-Za-z0-9_]*)\((\{.*\})\)\s*$',
        ]
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.DOTALL)
            if match is None:
                continue
            name = match.group(1)
            if allowed_names and name not in allowed_names:
                continue
            arguments = self._normalize_arguments(match.group(2))
            if arguments is None:
                continue
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "emulated-tool-call-0",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, ensure_ascii=True, sort_keys=True),
                        },
                    }
                ],
            }
        return None

    def _recover_single_action_tool_call_from_text(
        self,
        text: str,
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        action_tool_name = self._single_action_tool_name(tools)
        if action_tool_name is None:
            return None
        action = self._extract_action_text_candidate(text)
        if action is None:
            return None
        return self._build_emulated_tool_call(action_tool_name, {"action": action})

    def _extract_action_text_candidate(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None

        decoded = self._decode_json_payload(stripped)
        if isinstance(decoded, Mapping):
            tool_call = self._extract_tool_call_payload(decoded)
            if isinstance(tool_call, Mapping):
                function = tool_call.get("function") if isinstance(tool_call.get("function"), Mapping) else tool_call
                arguments = self._normalize_arguments(function.get("arguments") if isinstance(function, Mapping) else None)
                if isinstance(arguments, Mapping):
                    action = arguments.get("action")
                    if isinstance(action, str):
                        normalized = self._clean_action_text_candidate(action)
                        if normalized is not None:
                            return normalized

        tool_match = re.search(r"\btake_action\s*\((\{.*?\})\)", stripped, flags=re.IGNORECASE | re.DOTALL)
        if tool_match is not None:
            arguments = self._normalize_arguments(tool_match.group(1))
            if isinstance(arguments, Mapping):
                action = arguments.get("action")
                if isinstance(action, str):
                    normalized = self._clean_action_text_candidate(action)
                    if normalized is not None:
                        return normalized

        for match in re.finditer(r"(?im)^\s*(?:next\s+action|action)\s*[:=]\s*(.+?)\s*$", stripped):
            normalized = self._clean_action_text_candidate(match.group(1))
            if normalized is not None:
                return normalized

        non_empty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if non_empty_lines:
            normalized = self._clean_action_text_candidate(non_empty_lines[-1])
            if normalized is not None:
                return normalized

        normalized = self._clean_action_text_candidate(stripped)
        if normalized is not None:
            return normalized
        return None

    def _clean_action_text_candidate(self, text: str) -> str | None:
        cleaned = text.strip().strip("` ")
        cleaned = re.sub(r"^(?:next\s+action|action)\s*[:=]\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^[-*]\s*", "", cleaned)
        cleaned = re.sub(r"^[\"']|[\"']$", "", cleaned)
        cleaned = re.sub(r"[.。]+$", "", cleaned).strip()
        if not cleaned:
            return None
        if re.match(r"^(?:i|we)\s+(?:should|will|can|need|must|think)\b", cleaned, flags=re.IGNORECASE):
            return None
        if len(cleaned) > 120:
            return None
        return cleaned

    def _build_emulated_tool_call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "emulated-tool-call-0",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(dict(arguments), ensure_ascii=True, sort_keys=True),
                    },
                }
            ],
        }
