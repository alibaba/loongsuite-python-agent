# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import ast
import json
import logging
import sys
from typing import Any

from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    Text,
    ToolCall,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base64 image filtering (used by Chain content processing)
# ---------------------------------------------------------------------------


def recursive_size(obj: Any, max_size: int = 10240) -> int:
    """递归计算对象大小，超过阈值时快速返回"""
    total_size = 0
    if isinstance(obj, dict):
        total_size += sys.getsizeof(obj)
        if total_size > max_size:
            return total_size
        for key, value in obj.items():
            total_size += recursive_size(
                key, max_size - total_size
            ) + recursive_size(value, max_size - total_size)
            if total_size > max_size:
                return total_size
    elif isinstance(obj, list):
        total_size += sys.getsizeof(obj)
        if total_size > max_size:
            return total_size
        for item in obj:
            total_size += recursive_size(item, max_size - total_size)
            if total_size > max_size:
                return total_size
    else:
        total_size += sys.getsizeof(obj)
    return total_size


def _is_base64_image(item: Any) -> bool:
    """检查是否为base64编码的图片数据"""
    if not isinstance(item, dict):
        return False
    if not isinstance(item.get("image_url"), dict):
        return False
    if "data:image/" not in item.get("image_url", {}).get("url", ""):
        return False
    return True


def _filter_base64_images(obj: Any) -> Any:
    """递归过滤掉base64图片数据，保留其他信息"""
    if recursive_size(obj) < 10240:  # 10KB
        return obj

    if isinstance(obj, list):
        filtered_list = []
        for item in obj:
            if isinstance(item, str) and "data:image/" in item:
                start_idx = item.find("[")
                end_idx = item.rfind("]")

                if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                    filtered_list.append(item)
                    continue

                try:
                    filtered_obj = item[start_idx : end_idx + 1]
                    parsed_list = ast.literal_eval(filtered_obj)
                    if isinstance(parsed_list, list):
                        filtered_parsed_list = _filter_base64_images(
                            parsed_list
                        )
                        filtered_item = (
                            item[:start_idx]
                            + str(filtered_parsed_list)
                            + item[end_idx + 1 :]
                        )
                        filtered_list.append(filtered_item)
                    else:
                        filtered_list.append(item)
                except Exception:
                    logger.debug(
                        "Failed to parse/filter base64 in list item",
                        exc_info=True,
                    )
                    filtered_list.append(item)
            elif _is_base64_image(item):
                filtered_item = {
                    "type": item.get("type", "image_url"),
                    "image_url": {"url": "BASE64_IMAGE_DATA_FILTERED"},
                }
                filtered_list.append(filtered_item)
            else:
                filtered_list.append(_filter_base64_images(item))
        return filtered_list
    elif isinstance(obj, dict):
        filtered_dict = {}
        for key, value in obj.items():
            if _is_base64_image(value):
                filtered_dict[key] = {
                    "type": value.get("type", "image_url"),
                    "image_url": {"url": "BASE64_IMAGE_DATA_FILTERED"},
                }
            else:
                filtered_dict[key] = _filter_base64_images(value)
        return filtered_dict
    else:
        return obj


# ---------------------------------------------------------------------------
# Agent detection
# ---------------------------------------------------------------------------

AGENT_RUN_NAMES = frozenset(
    {
        "AgentExecutor",
        "MRKLChain",
        "ReActChain",
        "ReActTextWorldAgent",
        "SelfAskWithSearchChain",
    }
)

_LANGGRAPH_REACT_METADATA_KEY = "_loongsuite_react_agent"

LANGGRAPH_REACT_STEP_NODE = "agent"


def _is_agent_run(run: Any) -> bool:
    """Return *True* for classic LangChain agents (name-based check only).

    LangGraph agents are detected separately via metadata — see
    ``_has_langgraph_react_metadata`` — because their metadata propagates
    to ALL child callbacks and must be disambiguated in the tracer.
    """
    name = getattr(run, "name", "") or ""
    return name in AGENT_RUN_NAMES


def _has_langgraph_react_metadata(run: Any) -> bool:
    """Return *True* if *run* carries the LangGraph ReAct agent metadata.

    This flag is injected by ``loongsuite-instrumentation-langgraph``
    into ``config["metadata"]`` when ``Pregel.stream`` is called on a
    graph marked with ``_loongsuite_react_agent = True``.

    Note: the metadata propagates to child runs, so the caller must
    distinguish the top-level graph from child nodes.
    """
    metadata = getattr(run, "metadata", None) or {}
    return bool(metadata.get(_LANGGRAPH_REACT_METADATA_KEY))


# ---------------------------------------------------------------------------
# Run data extraction helpers
# ---------------------------------------------------------------------------


def _extract_model_name(run: Any) -> str | None:
    extra = getattr(run, "extra", None) or {}
    params = extra.get("invocation_params") or {}
    return (
        params.get("model_name")
        or params.get("model")
        or params.get("model_id")
    )


def _extract_provider(run: Any) -> str:
    serialized = getattr(run, "serialized", None) or {}
    id_list = serialized.get("id") or []
    if len(id_list) >= 3:
        return id_list[2]
    return "langchain"


def _extract_invocation_params(run: Any) -> dict[str, Any]:
    extra = getattr(run, "extra", None) or {}
    return extra.get("invocation_params") or {}


# ---------------------------------------------------------------------------
# LangChain message ↔ util-genai message conversion
# ---------------------------------------------------------------------------


def _convert_lc_message_to_input(msg: Any) -> InputMessage | None:
    """Convert a LangChain message dict (dumpd format) to InputMessage."""
    if isinstance(msg, dict):
        kwargs = msg.get("kwargs") or {}
        role = msg.get("id", ["", "", ""])
        if isinstance(role, list) and len(role) >= 3:
            role_name = role[-1].lower().replace("message", "")
            role_map = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
                "function": "tool",
                "tool": "tool",
                "chat": "user",
            }
            role_str = role_map.get(role_name, role_name)
        else:
            role_str = "user"

        content = kwargs.get("content", "")
        parts = []
        if isinstance(content, str) and content:
            parts.append(Text(content=content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(Text(content=part.get("text", "")))
                elif isinstance(part, str):
                    parts.append(Text(content=part))

        tool_calls = kwargs.get("tool_calls") or []
        for tc in tool_calls:
            if isinstance(tc, dict):
                parts.append(
                    ToolCall(
                        name=tc.get("name", ""),
                        arguments=tc.get("args", {}),
                        id=tc.get("id"),
                    )
                )
        if parts:
            return InputMessage(role=role_str, parts=parts)
    return None


def _extract_llm_input_messages(run: Any) -> list[InputMessage]:
    """Extract input messages from a Run's inputs."""
    inputs = getattr(run, "inputs", None) or {}
    messages: list[InputMessage] = []

    raw_messages = inputs.get("messages")
    if raw_messages:
        for batch in raw_messages:
            if isinstance(batch, list):
                for msg in batch:
                    converted = _convert_lc_message_to_input(msg)
                    if converted:
                        messages.append(converted)
        if messages:
            return messages

    prompts = inputs.get("prompts")
    if prompts and isinstance(prompts, list):
        for p in prompts:
            if isinstance(p, str):
                messages.append(
                    InputMessage(role="user", parts=[Text(content=p)])
                )
        return messages

    return messages


def _extract_llm_output_messages(run: Any) -> list[OutputMessage]:
    """Extract output messages from a completed Run."""
    outputs = getattr(run, "outputs", None) or {}
    result: list[OutputMessage] = []

    generations = outputs.get("generations") or []
    for gen_list in generations:
        if not isinstance(gen_list, list):
            continue
        for gen in gen_list:
            if not isinstance(gen, dict):
                continue
            text = gen.get("text", "")
            parts = []
            if text:
                parts.append(Text(content=text))

            msg_data = gen.get("message") or {}
            msg_kwargs = {}
            if isinstance(msg_data, dict):
                msg_kwargs = msg_data.get("kwargs") or {}

            tool_calls = msg_kwargs.get("tool_calls") or []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    parts.append(
                        ToolCall(
                            name=tc.get("name", ""),
                            arguments=tc.get("args", {}),
                            id=tc.get("id"),
                        )
                    )

            finish_reason = (gen.get("generation_info") or {}).get(
                "finish_reason", "stop"
            )
            if parts:
                result.append(
                    OutputMessage(
                        role="assistant",
                        parts=parts,
                        finish_reason=finish_reason or "stop",
                    )
                )
    return result


def _parse_token_usage_dict(token_usage: Any) -> tuple[int | None, int | None]:
    """Parse a token_usage/usage dict into (input_tokens, output_tokens)."""
    if not isinstance(token_usage, dict):
        return None, None
    inp = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
    out = token_usage.get("completion_tokens") or token_usage.get(
        "output_tokens"
    )
    return (
        int(inp) if inp is not None else None,
        int(out) if out is not None else None,
    )


def _extract_token_usage(run: Any) -> tuple[int | None, int | None]:
    """Return (input_tokens, output_tokens) from a completed LLM Run.

    Tries multiple LangChain formats in order:
    1. outputs["llm_output"]["token_usage"] or ["usage"]
    2. generations[i][j]["generation_info"]["token_usage"] or ["usage"]
    3. generations[i][j]["message"].response_metadata or ["kwargs"]["response_metadata"]
    """
    outputs = getattr(run, "outputs", None) or {}

    # 1. Primary: llm_output.token_usage / llm_output.usage
    llm_output = outputs.get("llm_output") or {}
    token_usage = (
        llm_output.get("token_usage") or llm_output.get("usage") or {}
    )
    inp, out = _parse_token_usage_dict(token_usage)
    if inp is not None or out is not None:
        return inp, out

    # 2. Fallback: generations[][].generation_info["token_usage"] or ["usage"]
    # 3. Fallback: generations[][].message.response_metadata["token_usage"]
    for gen_list in outputs.get("generations") or []:
        if not isinstance(gen_list, list):
            continue
        for gen in gen_list:
            if not isinstance(gen, dict):
                continue
            # Try generation_info
            gen_info = gen.get("generation_info") or {}
            token_usage = (
                gen_info.get("token_usage") or gen_info.get("usage") or {}
            )
            inp, out = _parse_token_usage_dict(token_usage)
            if inp is not None or out is not None:
                return inp, out
            # Try message.response_metadata (serialized: kwargs.response_metadata)
            msg = gen.get("message")
            if msg is None:
                continue
            if isinstance(msg, dict):
                metadata = (msg.get("kwargs") or {}).get(
                    "response_metadata"
                ) or {}
            else:
                metadata = getattr(msg, "response_metadata", None) or {}
            if isinstance(metadata, dict):
                token_usage = (
                    metadata.get("token_usage") or metadata.get("usage") or {}
                )
                inp, out = _parse_token_usage_dict(token_usage)
                if inp is not None or out is not None:
                    return inp, out

    return None, None


def _extract_finish_reasons(run: Any) -> list[str] | None:
    outputs = getattr(run, "outputs", None) or {}
    reasons: list[str] = []
    for gen_list in outputs.get("generations") or []:
        if not isinstance(gen_list, list):
            continue
        for gen in gen_list:
            if not isinstance(gen, dict):
                continue
            info = gen.get("generation_info") or {}
            reason = info.get("finish_reason")
            if reason:
                reasons.append(reason)
    return reasons or None


def _extract_response_model(run: Any) -> str | None:
    outputs = getattr(run, "outputs", None) or {}
    llm_output = outputs.get("llm_output") or {}
    return llm_output.get("model_name") or llm_output.get("model")


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _safe_json(obj: Any, max_len: int = 4096) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        logger.debug(
            "Failed to JSON serialize object, using str()", exc_info=True
        )
        s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s
