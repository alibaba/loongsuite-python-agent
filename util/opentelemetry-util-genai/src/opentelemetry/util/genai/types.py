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

import base64
from contextvars import Token
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional, Protocol, Type, Union

from opentelemetry.context import Context
from opentelemetry.semconv._incubating.attributes import \
    gen_ai_attributes as GenAI
from opentelemetry.trace import Span
from typing_extensions import TypeAlias


class _ContentTruncateTool(Protocol):
    """内容截断工具协议"""

    def truncate_content(self, content: str) -> str: ...
    def should_truncate(self, content: str) -> bool: ...


# Aliyun Python Agent Extension: Add type alias for ContextToken to avoid failure in python 3.8
ContextToken: TypeAlias = "Token[Context]"


class ContentCapturingMode(Enum):
    # Do not capture content (default).
    NO_CONTENT = 0
    # Only capture content in spans.
    SPAN_ONLY = 1
    # Only capture content in events.
    EVENT_ONLY = 2
    # Capture content in both spans and events.
    SPAN_AND_EVENT = 3


@dataclass()
class ToolCall:
    """Represents a tool call requested by the model

    This model is specified as part of semconv in `GenAI messages Python models - ToolCallRequestPart
    <https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/non-normative/models.ipynb>`__.
    """

    arguments: Any
    name: str
    id: str | None
    type: Literal["tool_call"] = "tool_call"


@dataclass()
class ToolCallResponse:
    """Represents a tool call result sent to the model or a built-in tool call outcome and details

    This model is specified as part of semconv in `GenAI messages Python models - ToolCallResponsePart
    <https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/non-normative/models.ipynb>`__.
    """

    response: Any
    id: str | None
    type: Literal["tool_call_response"] = "tool_call_response"


@dataclass()
class Text:
    """Represents text content sent to or received from the model

    This model is specified as part of semconv in `GenAI messages Python models - TextPart
    <https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/non-normative/models.ipynb>`__.
    """

    content: str
    type: Literal["text"] = "text"


@dataclass()
class Reasoning:
    """Represents reasoning/thinking content received from the model

    This model is specified as part of semconv in `GenAI messages Python models - ReasoningPart
    <https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/non-normative/models.ipynb>`__.
    """

    content: str
    type: Literal["reasoning"] = "reasoning"


Modality = Literal["image", "video", "audio"]


class Blob:
    """Represents blob binary data sent inline to the model.

    Supports automatic base64 encoding/decoding with lazy caching.
    Can be initialized with either `content` (bytes) or `base64_content` (str).

    This model is specified as part of semconv in `GenAI messages Python models - BlobPart
    <https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/non-normative/models.ipynb>`__.
    """

    def __init__(
        self,
        modality: Union[Modality, str],
        mime_type: Optional[str] = None,
        content: Optional[bytes] = None,
        base64_content: Optional[str] = None,
        type: Literal["blob"] = "blob",  # noqa: A002
    ):
        """Initialize Blob.

        Args:
            modality: Modality type (image/video/audio)
            mime_type: MIME type
            content: Raw bytes data (optional, provide either content or base64_content)
            base64_content: Base64 encoded data (optional, provide either content or base64_content)
            type: Type literal, defaults to "blob"
        """
        if content is None and base64_content is None:
            raise ValueError(
                "either content or base64_content must be provided"
            )

        self.modality = modality
        self.mime_type = mime_type
        self.type = type
        self._content: Optional[bytes] = content
        self._base64_content: Optional[str] = base64_content

    @property
    def content(self) -> bytes:
        """Get raw bytes data (auto-decodes from base64 if needed)."""
        if self._content is None:
            if self._base64_content is None:
                raise ValueError("content is not set")
            self._content = base64.b64decode(
                self._base64_content, validate=True
            )
        return self._content

    @content.setter
    def content(self, value: bytes) -> None:
        """Set raw bytes data (clears base64 cache)."""
        self._content = value
        self._base64_content = None

    @property
    def base64_content(self) -> str:
        """Get base64 encoded data (auto-encodes from bytes if needed)."""
        if self._base64_content is None:
            if self._content is None:
                raise ValueError("content is not set")
            self._base64_content = base64.b64encode(self._content).decode(
                "utf-8"
            )
        return self._base64_content

    @base64_content.setter
    def base64_content(self, value: str) -> None:
        """Set base64 encoded data (clears content cache)."""
        self._base64_content = value
        self._content = None

    def to_serializable_object(
        self, truncate_tool: Optional[_ContentTruncateTool] = None
    ) -> Dict[str, Any]:
        """Convert to serializable dict."""
        content = self.base64_content
        if truncate_tool and truncate_tool.should_truncate(content):
            content = truncate_tool.truncate_content(content)
        return {
            "type": self.type,
            "content": content,
            "mime_type": self.mime_type,
            "modality": self.modality,
        }


@dataclass()
class File:
    """Represents an external referenced file sent to the model by file id

    This model is specified as part of semconv in `GenAI messages Python models - FilePart
    <https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/non-normative/models.ipynb>`__.
    """

    mime_type: str | None
    modality: Union[Modality, str]
    file_id: str
    type: Literal["file"] = "file"


@dataclass()
class Uri:
    """Represents an external referenced file sent to the model by URI

    This model is specified as part of semconv in `GenAI messages Python models - UriPart
    <https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/non-normative/models.ipynb>`__.
    """

    modality: Union[Modality, str]
    mime_type: Optional[str]
    uri: str
    type: Literal["uri"] = "uri"

    def to_serializable_object(
        self, truncate_tool: Optional[_ContentTruncateTool] = None
    ) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "type": self.type,
            "mime_type": self.mime_type,
            "uri": self.uri,
            "modality": self.modality,
        }


MessagePart = Union[
    Text, ToolCall, ToolCallResponse, Blob, File, Uri, Reasoning, Any
]


FinishReason = Literal[
    "content_filter", "error", "length", "stop", "tool_calls"
]


@dataclass()
class InputMessage:
    role: str
    parts: list[MessagePart]


@dataclass()
class OutputMessage:
    role: str
    parts: list[MessagePart]
    finish_reason: str | FinishReason


# LoongSuite Extension


@dataclass()
class FunctionToolDefinition:
    name: str
    description: str | None
    parameters: Any | None
    type: Literal["function"] = "function"


ToolDefinition = Union[FunctionToolDefinition, Any]


def _new_input_messages() -> list[InputMessage]:
    return []


def _new_output_messages() -> list[OutputMessage]:
    return []


def _new_system_instruction() -> list[MessagePart]:
    return []


def _new_str_any_dict() -> dict[str, Any]:
    return {}


# LoongSuite Extension


def _new_tool_definitions() -> list[ToolDefinition]:
    return []


@dataclass
class LLMInvocation:
    """
    Represents a single LLM call invocation. When creating an LLMInvocation object,
    only update the data attributes. The span and context_token attributes are
    set by the TelemetryHandler.
    """

    request_model: str
    # Chat by default
    operation_name: str = GenAI.GenAiOperationNameValues.CHAT.value
    context_token: ContextToken | None = None
    span: Span | None = None
    input_messages: list[InputMessage] = field(
        default_factory=_new_input_messages
    )
    output_messages: list[OutputMessage] = field(
        default_factory=_new_output_messages
    )
    system_instruction: list[MessagePart] = field(
        default_factory=_new_system_instruction
    )
    tool_definitions: list[ToolDefinition] = field(  # LoongSuite Extension
        default_factory=_new_tool_definitions
    )
    provider: str | None = None
    response_model_name: str | None = None
    response_id: str | None = None
    finish_reasons: list[str] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    attributes: dict[str, Any] = field(default_factory=_new_str_any_dict)
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    seed: int | None = None
    monotonic_start_s: float | None = None
    """
    Monotonic start time in seconds (from timeit.default_timer) used
    for duration calculations to avoid mixing clock sources. This is
    populated by the TelemetryHandler when starting an invocation.
    """
    monotonic_end_s: float | None = None
    """
    Monotonic end time in seconds (from timeit.default_timer) used
    for duration calculations in async multimodal processing. This is
    populated by the ExtendedTelemetryHandler when stopping an invocation
    with multimodal data.
    """


@dataclass
class Error:
    message: str
    type: Type[BaseException]


def obj_to_dict(obj: object) -> dict[str, Any]:
    """Convert object to dict, handling both dataclass and custom types."""
    if hasattr(obj, "to_serializable_object"):
        return obj.to_serializable_object()  # type: ignore[no-any-return, union-attr]
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj  # type: ignore[return-value]
