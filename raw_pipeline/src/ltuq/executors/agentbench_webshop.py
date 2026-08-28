from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .agentbench import (
    AgentBenchControllerError,
    AgentBenchControllerClient,
    AgentBenchSample,
    StepSample,
    AgentBenchSingleTrajectoryExecutor,
)
from ..trajectory import TrajectoryDependentDecisionProcess

_WEBSHOP_SUCCESS_REWARD_THRESHOLD = 1.0 - 1e-9
_WEBSHOP_PRODUCT_ID_PATTERN = re.compile(r"^b[0-9a-z]{9}$", re.IGNORECASE)
_WEBSHOP_PRICE_PATTERN = re.compile(r"\bPrice:\s*(\$[0-9][0-9,]*(?:\.[0-9]{1,2})?)", re.IGNORECASE)
_WEBSHOP_PRODUCT_CONTROLS = {
    "< prev",
    "back to search",
    "buy now",
    "description",
    "features",
    "next >",
    "prev",
    "reviews",
    "search",
}
_WEBSHOP_DETAIL_PAGES = {"description", "features", "reviews"}
_WEBSHOP_CONTEXT_RESET_CLICKS = {"back to search", "next >", "search"}


@dataclass(frozen=True)
class AgentBenchWebShopSample(AgentBenchSample):
    pass


class AgentBenchWebShopControllerClient(AgentBenchControllerClient):
    def _extract_final_answer(self, messages: Sequence[Mapping[str, Any]]) -> str | None:
        return self._extract_last_action(messages)

    def _build_fc_output(
        self,
        sample: AgentBenchWebShopSample,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        final_state: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        reward = final_state.get("reward") if isinstance(final_state, Mapping) else None
        metrics = final_state.get("metrics") if isinstance(final_state, Mapping) else None
        score = metrics.get("score") if isinstance(metrics, Mapping) else None
        terminal_action = self._extract_last_action(messages)

        result_payload: dict[str, Any] = {
            "answer": terminal_action,
            "is_correct": float(reward) >= _WEBSHOP_SUCCESS_REWARD_THRESHOLD if isinstance(reward, (int, float)) else None,
            "score": score,
            "terminal_action": terminal_action,
        }
        task_result = final_state.get("result") if isinstance(final_state, Mapping) else None
        if isinstance(task_result, Mapping):
            result_payload.update(dict(task_result))
            result_payload.setdefault("answer", terminal_action)
            result_payload.setdefault("terminal_action", terminal_action)
        elif task_result is not None:
            result_payload["result"] = task_result

        resolved_reward = result_payload.get("reward")
        if not isinstance(resolved_reward, (int, float)) or isinstance(resolved_reward, bool):
            resolved_reward = result_payload.get("score")
        if isinstance(resolved_reward, (int, float)) and not isinstance(resolved_reward, bool):
            result_payload["is_correct"] = float(resolved_reward) >= _WEBSHOP_SUCCESS_REWARD_THRESHOLD

        normalized_answer = result_payload.get("answer")
        if normalized_answer is not None:
            result_payload["answer"] = self._coerce_text(normalized_answer)

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

    def _extract_last_action(self, messages: Sequence[Mapping[str, Any]]) -> str | None:
        for _, function in self._iter_assistant_functions(messages, reverse_tool_calls=True):
            name = str(function.get("name", ""))
            if name not in {"search_action", "click_action"}:
                continue
            arguments = self._decode_function_arguments(function)
            if arguments is None:
                continue
            if name == "search_action":
                keywords = arguments.get("keywords")
                if keywords is not None:
                    return f"search[{self._coerce_text(keywords)}]"
                continue
            value = arguments.get("value")
            if value is not None:
                return f"click[{self._coerce_text(value)}]"
        return None


class AgentBenchWebShopExecutor(AgentBenchSingleTrajectoryExecutor):
    name = "agentbench_webshop"
    _MAX_SAMPLING_MESSAGES = 8
    _MAX_WEBSHOP_OBSERVATION_CHARS = 2600
    _MAX_WEBSHOP_VISIBLE_CONTEXT_CHARS = 700

    _COMPLETION_HINT_PATTERNS = (
        re.compile(
            r"\b(task is complete|task seems to be complete|completed the action|successfully identified|found a suitable product|meets all the criteria|satisfies all your requirements)\b",
            re.IGNORECASE,
        ),
    )

    def _webshop_visible_purchase_context_lines(self, tdp: TrajectoryDependentDecisionProcess) -> list[str]:
        summary = self._build_visible_webshop_purchase_summary(tdp)
        if not summary:
            return []

        lines = [
            "WebShop visible purchase summary derived only from recorded actions and observations:",
        ]
        terminal_action = summary.get("terminal_action")
        if isinstance(terminal_action, str) and terminal_action:
            lines.append(f"Terminal action: {terminal_action}.")
        purchased_asin = summary.get("purchased_asin")
        if isinstance(purchased_asin, str) and purchased_asin:
            lines.append(f"Purchased ASIN inferred from trajectory: {purchased_asin}.")
        selected_options = summary.get("selected_options")
        if isinstance(selected_options, list):
            rendered_options = ", ".join(str(option) for option in selected_options) if selected_options else "none"
            lines.append(f"Selected options observed before purchase: {rendered_options}.")
        title = summary.get("visible_title")
        if isinstance(title, str) and title:
            lines.append(f"Visible product title before purchase: {title}.")
        price = summary.get("visible_price")
        if isinstance(price, str) and price:
            lines.append(f"Visible product price before purchase: {price}.")
        detail_pages = summary.get("observed_detail_pages")
        if isinstance(detail_pages, list):
            rendered_pages = ", ".join(str(page) for page in detail_pages) if detail_pages else "none"
            lines.append(f"Observed product detail pages before purchase: {rendered_pages}.")
        detail_excerpt = summary.get("observed_detail_excerpt")
        if isinstance(detail_excerpt, str) and detail_excerpt:
            lines.append(f"Observed detail-page text excerpt: {detail_excerpt}.")
        if summary.get("post_purchase_observation") is None:
            lines.append(
                "WebShop click[buy now] terminates the task and commonly has no post-purchase observation; do not treat that absence alone as evidence failure."
            )
        return lines

    def _build_visible_webshop_purchase_summary(self, tdp: TrajectoryDependentDecisionProcess) -> dict[str, Any] | None:
        current_asin: str | None = None
        selected_options: list[str] = []
        product_pages: dict[str, dict[str, Any]] = {}

        for step in tdp.steps:
            action = self._webshop_tdp_step_action(step)
            if not action:
                continue
            kind, value = self._split_webshop_action(action)
            if not kind:
                continue

            observation = self._webshop_step_observation(step)
            normalized_value = " ".join(value.strip().split())
            lowered_value = normalized_value.lower()

            if kind == "search":
                current_asin = None
                selected_options = []
                continue

            if kind != "click":
                continue

            product_id = self._normalize_webshop_product_id(normalized_value)
            if product_id is not None:
                current_asin = product_id
                selected_options = []
                product_pages[current_asin] = self._parse_visible_webshop_product_page(observation)
                continue

            if lowered_value in _WEBSHOP_CONTEXT_RESET_CLICKS:
                current_asin = None
                selected_options = []
                continue

            if lowered_value in _WEBSHOP_DETAIL_PAGES and current_asin is not None:
                product_summary = product_pages.setdefault(current_asin, {})
                observed_pages = product_summary.setdefault("observed_detail_pages", [])
                if lowered_value not in observed_pages:
                    observed_pages.append(lowered_value)
                if observation:
                    detail_texts = product_summary.setdefault("observed_detail_text", {})
                    detail_texts[lowered_value] = self._compact_visible_webshop_text(observation)
                continue

            if lowered_value == "buy now":
                product_summary = dict(product_pages.get(current_asin or "", {}))
                detail_texts = product_summary.get("observed_detail_text")
                detail_excerpt = None
                if isinstance(detail_texts, Mapping):
                    detail_excerpt = " | ".join(
                        f"{key}: {value}" for key, value in detail_texts.items() if isinstance(value, str) and value
                    )
                return {
                    "terminal_action": "click[buy now]",
                    "purchased_asin": current_asin,
                    "selected_options": list(selected_options),
                    "visible_title": product_summary.get("visible_title"),
                    "visible_price": product_summary.get("visible_price"),
                    "observed_detail_pages": list(product_summary.get("observed_detail_pages") or []),
                    "observed_detail_excerpt": self._compact_visible_webshop_text(detail_excerpt),
                    "post_purchase_observation": observation,
                }

            if current_asin is not None and self._is_visible_webshop_option_click(lowered_value):
                option = self._normalize_webshop_option(normalized_value)
                if option not in selected_options:
                    selected_options.append(option)

        return None

    def _parse_visible_webshop_product_page(self, observation: str | None) -> dict[str, Any]:
        summary: dict[str, Any] = {"observed_detail_pages": []}
        if not observation:
            return summary

        price_match = _WEBSHOP_PRICE_PATTERN.search(observation)
        if price_match is not None:
            summary["visible_price"] = price_match.group(1)

        tokens = [" ".join(token.strip().split()) for token in observation.split("[SEP]")]
        tokens = [token for token in tokens if token]
        for index, token in enumerate(tokens):
            if token.lower().startswith("price:") and index > 0:
                title = tokens[index - 1]
                if title and not title.lower().startswith(("instruction:", "back to search", "< prev")):
                    summary["visible_title"] = self._compact_visible_webshop_text(title, limit=240)
                break
        if "visible_title" not in summary:
            title_match = re.search(
                r"(?:Observation:\s*)?(.*?)\s+Price:\s*\$[0-9]",
                " ".join(observation.split()),
                flags=re.IGNORECASE,
            )
            if title_match is not None:
                title = re.sub(
                    r"^.*?(?:Back to Search|< Prev|Next >)\s+",
                    "",
                    title_match.group(1),
                    flags=re.IGNORECASE,
                ).strip()
                if title:
                    summary["visible_title"] = self._compact_visible_webshop_text(title, limit=240)
        return summary

    def _webshop_tdp_step_action(self, step: Any) -> str | None:
        metadata = step.metadata if isinstance(getattr(step, "metadata", None), Mapping) else {}
        chosen_metadata = metadata.get("chosen_output_metadata")
        if isinstance(chosen_metadata, Mapping):
            action = chosen_metadata.get("webshop_action_text")
            if isinstance(action, str) and action.strip():
                return self._normalize_webshop_action_text(action)
        realized_decision = getattr(step, "realized_decision", None)
        if isinstance(realized_decision, str):
            return self._extract_webshop_action_text(realized_decision)
        return None

    @staticmethod
    def _webshop_step_observation(step: Any) -> str | None:
        metadata = step.metadata if isinstance(getattr(step, "metadata", None), Mapping) else {}
        observation = metadata.get("observation")
        return observation if isinstance(observation, str) and observation.strip() else None

    @staticmethod
    def _split_webshop_action(action: str) -> tuple[str, str]:
        match = re.fullmatch(r"\s*(search|click)\[([^\]]*)\]\s*", action, flags=re.IGNORECASE)
        if match is None:
            return "", ""
        return match.group(1).lower(), match.group(2).strip()

    @staticmethod
    def _normalize_webshop_product_id(value: str) -> str | None:
        normalized = value.strip().lower()
        return normalized if _WEBSHOP_PRODUCT_ID_PATTERN.fullmatch(normalized) else None

    @staticmethod
    def _normalize_webshop_option(value: str) -> str:
        return " ".join(value.strip().lower().split())

    @staticmethod
    def _is_visible_webshop_option_click(value: str) -> bool:
        if not value or value in _WEBSHOP_PRODUCT_CONTROLS:
            return False
        return _WEBSHOP_PRODUCT_ID_PATTERN.fullmatch(value) is None

    def _extract_webshop_action_text(self, text: str) -> str | None:
        tool_matches = list(re.finditer(r"\b(search_action|click_action)\s*\((\{.*?\})\)", text, flags=re.DOTALL))
        for match in reversed(tool_matches):
            name = match.group(1)
            try:
                arguments = json.loads(match.group(2))
            except json.JSONDecodeError:
                continue
            if not isinstance(arguments, Mapping):
                continue
            if name == "search_action":
                keywords = arguments.get("keywords")
                if isinstance(keywords, str) and keywords.strip():
                    return self._normalize_webshop_action_text(f"search[{keywords.strip()}]")
                continue
            value = arguments.get("value")
            if isinstance(value, str) and value.strip():
                return self._normalize_webshop_action_text(f"click[{value.strip()}]")

        direct_matches = list(re.finditer(r"\b(search|click)\[([^\]]+)\]", text, flags=re.IGNORECASE))
        for match in reversed(direct_matches):
            return self._normalize_webshop_action_text(f"{match.group(1).lower()}[{match.group(2).strip()}]")
        return None

    def _normalize_webshop_action_text(self, action: str) -> str:
        kind, value = self._split_webshop_action(action)
        if not kind:
            return action.strip()
        if kind == "click":
            product_id = self._normalize_webshop_product_id(value)
            if product_id is not None:
                value = product_id
            elif value.strip().lower() == "buy now":
                value = "buy now"
            else:
                value = " ".join(value.strip().split())
        else:
            value = " ".join(value.strip().split())
        return f"{kind}[{value}]"

    def _compact_visible_webshop_text(self, value: Any, *, limit: int | None = None) -> str | None:
        if not isinstance(value, str):
            return None
        compacted = " ".join(value.split())
        if not compacted:
            return None
        resolved_limit = self._MAX_WEBSHOP_VISIBLE_CONTEXT_CHARS if limit is None else limit
        if len(compacted) <= resolved_limit:
            return compacted
        return f"{compacted[: max(0, resolved_limit - 16)].rstrip()}...[truncated]"

    def _latest_available_actions(self, messages: Sequence[Mapping[str, Any]]) -> list[str]:
        for message in reversed(messages):
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str):
                continue
            marker = "Available Actions:"
            if marker not in content:
                continue
            _, _, tail = content.partition(marker)
            return [first or second for first, second in re.findall(r"'([^']+)'|\"([^\"]+)\"", tail)]
        return []

    def _compact_messages_for_sampling(self, messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Keep WebShop prompts inside the model context without changing live session state.

        WebShop observations can include long product listings. The controller still
        receives the full session, but model sampling only needs the original user
        instruction, recent tool-call history, and the current page/action list.
        """
        normalized = [dict(message) for message in messages if isinstance(message, Mapping)]
        if len(normalized) <= self._MAX_SAMPLING_MESSAGES:
            return [self._compact_single_sampling_message(message, is_current=(index == len(normalized) - 1)) for index, message in enumerate(normalized)]

        first_message = normalized[0]
        recent = normalized[-(self._MAX_SAMPLING_MESSAGES - 2) :]
        compacted: list[dict[str, Any]] = [
            self._compact_single_sampling_message(first_message, is_current=False),
            {
                "role": "user",
                "content": "Earlier WebShop turns were omitted to fit the context. Continue from the current page below and use only valid available actions.",
            },
        ]
        compacted.extend(
            self._compact_single_sampling_message(message, is_current=(index == len(recent) - 1))
            for index, message in enumerate(recent)
        )
        return compacted

    def _compact_single_sampling_message(self, message: Mapping[str, Any], *, is_current: bool) -> dict[str, Any]:
        compacted = dict(message)
        role = compacted.get("role")
        content = compacted.get("content")
        if role == "assistant" and compacted.get("tool_calls") is not None:
            compacted["content"] = ""
        elif role == "tool" and isinstance(content, str) and not is_current:
            action_line = next((line for line in content.splitlines() if line.startswith("Action:")), "")
            compacted["content"] = f"{action_line}\nObservation omitted; use the latest observation for the current available actions.".strip()
        elif isinstance(content, str) and len(content) > self._MAX_WEBSHOP_OBSERVATION_CHARS:
            marker = "Available Actions:"
            if marker in content:
                head, _, tail = content.partition(marker)
                budget = max(500, self._MAX_WEBSHOP_OBSERVATION_CHARS - len(tail) - len(marker) - 80)
                compacted["content"] = f"{head[:budget]}\n...[observation truncated]...\n{marker}{tail}"
            else:
                compacted["content"] = f"{content[: self._MAX_WEBSHOP_OBSERVATION_CHARS]}\n...[truncated]..."
        return compacted

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
        """Sample WebShop actions from a compact prompt to avoid context overflow."""
        compacted_messages = self._compact_messages_for_sampling(messages)
        return await super()._sample_step_samples(
            sample=sample,
            messages=compacted_messages,
            tools=tools,
            per_step_samples=per_step_samples,
            step_index=step_index,
            sampling_cursor=sampling_cursor,
        )

    async def _generate_step_samples(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        sample_count: int,
    ) -> list[StepSample]:
        """Generate candidates and repair/replace invalid WebShop actions."""
        available_actions = self._latest_available_actions(messages)
        samples = await super()._generate_step_samples(messages=messages, tools=tools, sample_count=sample_count)
        samples = [self._annotate_webshop_action_validity(sample, available_actions) for sample in samples]
        if self._valid_webshop_sample_count(samples) >= sample_count:
            return self._prefer_valid_webshop_samples(samples, sample_count)

        repair_messages = self._build_webshop_valid_action_repair_messages(messages, available_actions, samples)
        extra_attempts = max(2, sample_count)
        for _ in range(extra_attempts):
            needed = max(1, sample_count - self._valid_webshop_sample_count(samples))
            repaired = await super()._generate_step_samples(messages=repair_messages, tools=tools, sample_count=needed)
            samples.extend(self._annotate_webshop_action_validity(sample, available_actions) for sample in repaired)
            if self._valid_webshop_sample_count(samples) >= sample_count:
                break
        return self._prefer_valid_webshop_samples(samples, sample_count)

    def _step_sample_is_selectable(self, step_sample: StepSample) -> bool:
        """Only execute tool calls that match the current WebShop action list."""
        valid = step_sample.recovery_state.get("webshop_action_valid")
        if isinstance(valid, bool):
            return valid
        return super()._step_sample_is_selectable(step_sample)

    def _annotate_webshop_action_validity(self, step_sample: StepSample, available_actions: Sequence[str]) -> StepSample:
        valid, reason, action = self._webshop_step_sample_validity(step_sample, available_actions)
        recovery_state = dict(step_sample.recovery_state)
        recovery_state["webshop_action_valid"] = valid
        recovery_state["webshop_action_invalid_reason"] = reason
        recovery_state["webshop_action_text"] = action
        metadata = dict(step_sample.metadata)
        metadata["webshop_action_valid"] = valid
        metadata["webshop_action_invalid_reason"] = reason
        if action is not None:
            metadata["webshop_action_text"] = action
        return StepSample(output=step_sample.output, metadata=metadata, recovery_state=recovery_state)

    def _webshop_step_sample_validity(self, step_sample: StepSample, available_actions: Sequence[str]) -> tuple[bool, str | None, str | None]:
        assistant_message = step_sample.recovery_state.get("assistant_message")
        if not isinstance(assistant_message, Mapping):
            return False, "missing_assistant_message", None
        functions = list(self.controller_client._iter_assistant_functions([assistant_message]))
        if len(functions) != 1:
            return False, "expected_exactly_one_tool_call", None
        _, function = functions[0]
        name = str(function.get("name", ""))
        arguments = self.controller_client._decode_function_arguments(function)
        if not isinstance(arguments, Mapping):
            return False, "invalid_tool_arguments", None
        if not available_actions:
            return True, None, self._format_assistant_message(assistant_message)
        normalized_actions = {str(action).strip().lower() for action in available_actions}
        if name == "search_action":
            keywords = arguments.get("keywords")
            if not isinstance(keywords, str) or not keywords.strip():
                return False, "empty_search_keywords", None
            action = f"search[{keywords.strip()}]"
            return ("search" in normalized_actions), None if "search" in normalized_actions else "search_not_available", action
        if name == "click_action":
            value = arguments.get("value")
            if not isinstance(value, str) or not value.strip():
                return False, "empty_click_value", None
            action_value = value.strip()
            action = f"click[{action_value}]"
            if action_value.lower() in normalized_actions:
                return True, None, action
            return False, "click_value_not_available", action
        return False, f"unsupported_tool:{name}", None

    def _valid_webshop_sample_count(self, samples: Sequence[StepSample]) -> int:
        return sum(1 for sample in samples if sample.recovery_state.get("webshop_action_valid") is True)

    def _prefer_valid_webshop_samples(self, samples: Sequence[StepSample], sample_count: int) -> list[StepSample]:
        valid = [sample for sample in samples if sample.recovery_state.get("webshop_action_valid") is True]
        invalid = [sample for sample in samples if sample.recovery_state.get("webshop_action_valid") is not True]
        selected = (valid + invalid)[:sample_count]
        if len(selected) != sample_count:
            raise AgentBenchControllerError(f"Expected {sample_count} WebShop samples but received {len(selected)}.")
        return selected

    def _build_webshop_valid_action_repair_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        available_actions: Sequence[str],
        samples: Sequence[StepSample],
    ) -> list[dict[str, Any]]:
        repaired = [dict(message) for message in messages if isinstance(message, Mapping)]
        invalid_reasons = [
            str(sample.recovery_state.get("webshop_action_invalid_reason"))
            for sample in samples
            if sample.recovery_state.get("webshop_action_valid") is not True
        ]
        action_list = ", ".join(str(action) for action in available_actions) or "(none)"
        repaired.append(
            {
                "role": "user",
                "content": (
                    "/no_think\n"
                    "Your previous WebShop action was invalid or not callable. "
                    "Return exactly one tool call and no explanation. "
                    f"Valid actions on the current page are: {action_list}. "
                    "Use search_action only when 'search' is available. "
                    "Use click_action only with a value copied exactly from the valid action list. "
                    f"Invalid reasons seen: {', '.join(invalid_reasons[:5])}."
                ),
            }
        )
        return repaired

    def _message_suggests_completion_without_tool_call(self, assistant_message: Mapping[str, Any]) -> bool:
        content = assistant_message.get("content")
        if not isinstance(content, str):
            return False
        normalized = content.strip()
        if not normalized:
            return False
        return any(pattern.search(normalized) for pattern in self._COMPLETION_HINT_PATTERNS)

    async def _recover_no_tool_call_step(
        self,
        *,
        sample: AgentBenchWebShopSample,
        session,
        step_index: int,
        prepared_step_samples,
        chosen_output_index: int,
        sampling_metadata: Mapping[str, Any],
    ):
        chosen_sample = prepared_step_samples[chosen_output_index]
        assistant_message = chosen_sample.recovery_state.get("assistant_message")
        if not isinstance(assistant_message, Mapping):
            return None
        if not self._message_suggests_completion_without_tool_call(assistant_message):
            return None
        available_actions = self._latest_available_actions(session.messages)
        if not any(str(action).lower() == "buy now" for action in available_actions):
            return None
        fallback_message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "fallback-buy-now",
                    "type": "function",
                    "function": {
                        "name": "click_action",
                        "arguments": json.dumps({"value": "Buy Now"}, sort_keys=True),
                    },
                }
            ],
        }
        return await self._build_interacted_sampled_step(
            sample=sample,
            session=session,
            step_index=step_index,
            assistant_message=fallback_message,
            prepared_step_samples=prepared_step_samples,
            chosen_output_index=chosen_output_index,
            sampling_metadata=sampling_metadata,
            extra_metadata={
                "no_tool_call_fallback_action": "click[Buy Now]",
                "no_tool_call_fallback_reason": "completion_without_tool_call",
            },
        )
