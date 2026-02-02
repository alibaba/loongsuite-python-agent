import dataclasses
import json
import logging
from base64 import b64encode
from typing import Any, Dict, List, Optional

from opentelemetry.trace import Span
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCall,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_SIZE = 10000


class _GenAiJsonEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, bytes):
            return b64encode(o).decode()
        return super().default(o)


def _safe_json_dumps(
    obj: Any, default: str = "", max_size: int = MAX_MESSAGE_SIZE
) -> str:
    """
    Safely serialize object to JSON string with size limit.
    """
    if obj is None:
        return default

    # Fast path for simple types
    if isinstance(obj, (str, int, float, bool)):
        result = str(obj)
    else:
        try:
            result = json.dumps(
                obj,
                separators=(",", ":"),
                cls=_GenAiJsonEncoder,
                ensure_ascii=False,
            )
        except Exception as e:
            logger.debug(f"Failed to serialize to JSON: {e}")
            result = str(obj)

    if len(result) > max_size:
        return result[:max_size] + "...[truncated]"
    return result


GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"

OP_NAME_CREW = "crew.kickoff"
OP_NAME_AGENT = "agent.execute"
OP_NAME_TASK = "task.execute"
OP_NAME_TOOL = "tool.execute"


class GenAIHookHelper:
    def __init__(self, capture_content: bool = True):
        self.capture_content = capture_content

    def on_completion(
        self,
        span: Span,
        inputs: List[InputMessage],
        outputs: List[OutputMessage],
        system_instructions: Optional[List[MessagePart]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        if self.capture_content and span.is_recording():
            if inputs:
                span.set_attribute(
                    GEN_AI_INPUT_MESSAGES,
                    _safe_json_dumps([dataclasses.asdict(i) for i in inputs]),
                )
            if outputs:
                span.set_attribute(
                    GEN_AI_OUTPUT_MESSAGES,
                    _safe_json_dumps([dataclasses.asdict(o) for o in outputs]),
                )
            if system_instructions:
                span.set_attribute(
                    GEN_AI_SYSTEM_INSTRUCTIONS,
                    _safe_json_dumps(
                        [dataclasses.asdict(i) for i in system_instructions]
                    ),
                )

            if attributes:
                span.set_attributes(attributes)


def to_input_message(role: str, content: Any) -> List[InputMessage]:
    if not content:
        return []

    text_content = content if isinstance(content, str) else str(content)

    return [InputMessage(role=role, parts=[Text(content=text_content)])]


def to_output_message(
    role: str, content: Any, finish_reason: str = "stop"
) -> List[OutputMessage]:
    if not content:
        return []

    text_content = content if isinstance(content, str) else str(content)

    return [
        OutputMessage(
            role=role,
            parts=[Text(content=text_content)],
            finish_reason=finish_reason,
        )
    ]


def extract_agent_inputs(
    task_obj: Any, context: str, tools: List[Any]
) -> List[InputMessage]:
    description = getattr(task_obj, "description", "")

    parts = []
    if description:
        parts.append(Text(content=f"Task: {description}"))
    if context:
        parts.append(Text(content=f"Context: {context}"))
    if tools:
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        parts.append(Text(content=f"Tools Available: {', '.join(tool_names)}"))

    return [InputMessage(role="user", parts=parts)]


def extract_tool_inputs(tool_name: str, arguments: Any) -> List[InputMessage]:
    args_str = (
        json.dumps(arguments)
        if isinstance(arguments, dict)
        else str(arguments)
    )

    return [
        InputMessage(
            role="assistant",
            parts=[ToolCall(id=tool_name, name=tool_name, arguments=args_str)],
        )
    ]
