import json
import math
import time
import traceback
import logging
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from itertools import chain
from threading import RLock
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)
from uuid import UUID

import wrapt
from langchain_core.messages import BaseMessage
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.trace import SpanAttributes as OTELSpanAttributes
from opentelemetry.trace import SpanKind
from opentelemetry.util.types import AttributeValue
from opentelemetry.metrics import Meter
import os

from .semconv import *
from ._utils import _filter_base64_images, process_content

logger = logging.getLogger(__name__)

ENABLE_LANGCHAIN_INSTRUMENTOR = "ENABLE_LANGCHAIN_INSTRUMENTOR"


def _is_enable():
    enable_instrumentor = os.getenv(ENABLE_LANGCHAIN_INSTRUMENTOR)
    if enable_instrumentor is None:
        return True
    if enable_instrumentor.lower() == "false":
        return False
    else:
        return True


_AUDIT_TIMING = False


class _Run(NamedTuple):
    span: trace_api.Span
    context: context_api.Context


class LoongsuiteTracer(BaseTracer):
    __slots__ = ("_tracer", "_runs", "_lock", "_meter")

    def __init__(self, tracer: trace_api.Tracer, meter: Meter, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer
        self._runs: Dict[UUID, _Run] = {}
        self._lock = RLock()  # handlers may be run in a thread by langchain
        self._meter = meter

    def _start_trace(self, run: Run) -> None:
        super()._start_trace(run)
        if not _is_enable():
            return
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return
        with self._lock:
            parent_context = (
                parent.context
                if (parent_run_id := run.parent_run_id)
                   and (parent := self._runs.get(parent_run_id))
                else None
            )
        # We can't use real time because the handler may be
        # called in a background thread.
        start_time_utc_nano = _as_utc_nano(run.start_time)
        span = self._tracer.start_span(
            name=run.name,
            context=parent_context,
            start_time=start_time_utc_nano,
        )
        context = trace_api.set_span_in_context(span)
        # The following line of code is commented out to serve as a reminder that in a system
        # of callbacks, attaching the context can be hazardous because there is no guarantee
        # that the context will be detached. An error could happen between callbacks leaving
        # the context attached forever, and all future spans will use it as parent. What's
        # worse is that the error could have also prevented the span from being exported,
        # leaving all future spans as orphans. That is a very bad scenario.
        # token = context_api.attach(context)
        with self._lock:
            self._runs[run.id] = _Run(span=span, context=context)

    def _end_trace(self, run: Run) -> None:
        super()._end_trace(run)
        if not _is_enable():
            return
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return
        with self._lock:
            if run.id not in self._runs:
                logger.warning(f"Run Id: {run.id} is not in event data.")
                return
            event_data = self._runs.pop(run.id, None)
        if event_data:
            span = event_data.span
            try:
                _update_span(span, run)
            except Exception:
                logger.exception("Failed to update span with run data.")
            # We can't use real time because the handler may be
            # called in a background thread.
            end_time_utc_nano = _as_utc_nano(run.end_time) if run.end_time else None
            span.end(end_time=end_time_utc_nano)

    def _persist_run(self, run: Run) -> None:
        pass

    def on_llm_error(self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any) -> Run:
        logger.debug(f"on_llm_error: {error}")
        with self._lock:
            event_data = self._runs.get(run_id)
        if event_data:
            _record_exception(event_data.span, error)
        return super().on_llm_error(error, *args, run_id=run_id, **kwargs)

    def on_chain_error(self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any) -> Run:
        logger.debug(f"on_chain_error: {error}")
        with self._lock:
            event_data = self._runs.get(run_id)
        if event_data:
            _record_exception(event_data.span, error)
        return super().on_chain_error(error, *args, run_id=run_id, **kwargs)

    def on_retriever_error(
            self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any
    ) -> Run:
        logger.debug(f"on_retriever_error: {error}")
        with self._lock:
            event_data = self._runs.get(run_id)
        if event_data:
            _record_exception(event_data.span, error)
        return super().on_retriever_error(error, *args, run_id=run_id, **kwargs)

    def on_tool_error(self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any) -> Run:
        logger.debug(f"on_tool_error: {error}")
        with self._lock:
            event_data = self._runs.get(run_id)
        if event_data:
            _record_exception(event_data.span, error)
        return super().on_tool_error(error, *args, run_id=run_id, **kwargs)


def _record_exception(span: trace_api.Span, error: BaseException) -> None:
    if isinstance(error, Exception):
        span.record_exception(error)
        return
    exception_type = error.__class__.__name__
    exception_message = str(error)
    if not exception_message:
        exception_message = repr(error)
    attributes: Dict[str, AttributeValue] = {
        OTELSpanAttributes.EXCEPTION_TYPE: exception_type,
        OTELSpanAttributes.EXCEPTION_MESSAGE: exception_message,
        OTELSpanAttributes.EXCEPTION_ESCAPED: False,
    }
    try:
        # See e.g. https://github.com/open-telemetry/opentelemetry-python/blob/e9c7c7529993cd13b4af661e2e3ddac3189a34d0/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py#L967  # noqa: E501
        attributes[OTELSpanAttributes.EXCEPTION_STACKTRACE] = traceback.format_exc()
    except Exception:
        logger.exception("Failed to record exception stacktrace.")
    span.add_event(name="exception", attributes=attributes)


def _update_span(span: trace_api.Span, run: Run) -> None:
    logger.debug(f"_update_span: {run}")
    if run.error is None:
        span.set_status(trace_api.StatusCode.OK)
    else:
        span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, run.error))
    span_kind = (
        SpanKindValues.AGENT
        if "agent" in run.name.lower()
        else _langchain_run_type_to_span_kind(run.run_type)
    )
    span.set_attribute(LLM_SPAN_KIND, span_kind.value)
    deepcopy_inputs = deepcopy(run.inputs)
    filtered_inputs = _filter_base64_images(deepcopy_inputs)
    span.set_attributes(
        dict(
            _flatten(
                chain(
                    _as_input(_convert_io(filtered_inputs)),
                    _as_output(_convert_io(run.outputs)),
                    _prompts(filtered_inputs),
                    _input_messages(filtered_inputs),
                    _output_messages(run.outputs),
                    _prompt_template(run),
                    _model_name(run.extra),
                    _token_counts(run.outputs),
                    _tools(run),
                    _retrieval_documents(run),
                    _metadata(run),
                )
            )
        )
    )


def _langchain_run_type_to_span_kind(run_type: str) -> SpanKindValues:
    try:
        return SpanKindValues(run_type.lower())
    except ValueError:
        return SpanKindValues.UNKNOWN


def _serialize_json(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def stop_on_exception(
        wrapped: Callable[..., Iterator[Tuple[str, Any]]],
) -> Callable[..., Iterator[Tuple[str, Any]]]:
    def wrapper(*args: Any, **kwargs: Any) -> Iterator[Tuple[str, Any]]:
        start_time = time.perf_counter()
        try:
            yield from wrapped(*args, **kwargs)
        except Exception:
            logger.exception("Failed to get attribute.")
        finally:
            if _AUDIT_TIMING:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.debug(f"{wrapped.__name__}: {latency_ms:.3f}ms")

    return wrapper


@stop_on_exception
def _flatten(key_values: Iterable[Tuple[str, Any]]) -> Iterator[Tuple[str, AttributeValue]]:
    for key, value in key_values:
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value.items()):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, List) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping.items()):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value


@stop_on_exception
def _as_input(values: Iterable[str]) -> Iterator[Tuple[str, str]]:
    return zip((INPUT_VALUE, INPUT_MIME_TYPE), values)


@stop_on_exception
def _as_output(values: Iterable[str]) -> Iterator[Tuple[str, str]]:
    return zip((OUTPUT_VALUE, OUTPUT_MIME_TYPE), values)


def _convert_io(obj: Optional[Mapping[str, Any]]) -> Iterator[str]:
    if not obj:
        return
    assert isinstance(obj, dict), f"expected dict, found {type(obj)}"
    if len(obj) == 1 and isinstance(value := next(iter(obj.values())), str):
        yield process_content(value)
    else:
        obj = dict(_replace_nan(obj))
        content = process_content(json.dumps(obj, default=_serialize_json, ensure_ascii=False))
        yield process_content(content)
        yield MimeTypeValues.JSON.value


def _replace_nan(obj: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    for k, v in obj.items():
        if isinstance(v, float) and not math.isfinite(v):
            yield k, None
        else:
            yield k, v


@stop_on_exception
def _prompts(inputs: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, List[str]]]:
    """Yields prompts if present."""
    if not inputs:
        return
    if not hasattr(inputs, "get"):
        logger.warning(f"Invalid input type for parameter 'inputs': expected mapping, found {type(inputs).__name__}")
        return
    if prompts := inputs.get("prompts"):
        idx = 0
        for prompt in prompts:
            if isinstance(prompt, dict):
                if "role" in prompt:
                    yield f"gen_ai.prompt.{idx}.role", f"{prompt['role']}"
                if "text" in prompt:
                    yield f"gen_ai.prompt.{idx}.content", process_content(f"{prompt['text']}")
            elif isinstance(prompt, str):
                yield f"gen_ai.prompt.{idx}.content", process_content(prompt)
            idx += 1


@stop_on_exception
def _input_messages(
        inputs: Optional[Mapping[str, Any]],
) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    """Yields chat messages if present."""
    if not inputs:
        return
    assert hasattr(inputs, "get"), f"expected Mapping, found {type(inputs)}"
    # There may be more than one set of messages. We'll use just the first set.
    if multiple_messages := inputs.get("messages"):
        assert isinstance(
            multiple_messages, Iterable
        ), f"expected Iterable, found {type(multiple_messages)}"
        # This will only get the first set of messages.
        if not (first_messages := next(iter(multiple_messages), None)):
            return
        parsed_messages = []
        if isinstance(first_messages, list):
            for message_data in first_messages:
                if isinstance(message_data, BaseMessage):
                    parsed_messages.append(dict(_parse_message_data(message_data.to_json())))
                elif hasattr(message_data, "get"):
                    parsed_messages.append(dict(_parse_message_data(message_data)))
                elif isinstance(message_data, Sequence) and len(message_data) == 2:
                    # Handle tuple format (role, content)
                    role, content = message_data
                    parsed_messages.append({MESSAGE_ROLE: role, MESSAGE_CONTENT: content})
                else:
                    logger.warning(f"Failed to parse message of type {type(message_data)}")
        elif isinstance(first_messages, BaseMessage):
            parsed_messages.append(dict(_parse_message_data(first_messages.to_json())))
        elif hasattr(first_messages, "get"):
            parsed_messages.append(dict(_parse_message_data(first_messages)))
        elif isinstance(first_messages, Sequence) and len(first_messages) == 2:
            # Handle tuple format (role, content)
            role, content = first_messages
            parsed_messages.append({MESSAGE_ROLE: role, MESSAGE_CONTENT: content})
        else:
            logger.warning(f"Failed to parse messages of type {type(first_messages)}")
        if parsed_messages:
            yield LLM_INPUT_MESSAGES, parsed_messages
    elif multiple_prompts := inputs.get("prompts"):
        assert isinstance(
            multiple_prompts, Iterable
        ), f"expected Iterable, found {type(multiple_prompts)}"
        parsed_prompts = []
        for prompt_data in multiple_prompts:
            assert isinstance(prompt_data, str), f"expected str, found {type(prompt_data)}"
            parsed_prompts.append(dict(_parse_prompt_data(prompt_data)))
        if parsed_prompts:
            yield LLM_INPUT_MESSAGES, parsed_prompts


@stop_on_exception
def _output_messages(
        outputs: Optional[Mapping[str, Any]],
) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    """Yields chat messages if present."""
    if not outputs:
        return
    assert hasattr(outputs, "get"), f"expected Mapping, found {type(outputs)}"
    # There may be more than one set of generations. We'll use just the first set.
    if not (multiple_generations := outputs.get("generations")):
        return
    assert isinstance(
        multiple_generations, Iterable
    ), f"expected Iterable, found {type(multiple_generations)}"
    # This will only get the first set of generations.
    if not (first_generations := next(iter(multiple_generations), None)):
        return
    assert isinstance(
        first_generations, Iterable
    ), f"expected Iterable, found {type(first_generations)}"
    parsed_messages = []
    for generation in first_generations:
        assert hasattr(generation, "get"), f"expected Mapping, found {type(generation)}"
        if message_data := generation.get("message"):
            if isinstance(message_data, BaseMessage):
                parsed_messages.append(dict(_parse_message_data(message_data.to_json())))
            elif hasattr(message_data, "get"):
                parsed_messages.append(dict(_parse_message_data(message_data)))
            else:
                logger.warning(f"Failed to parse message of type {type(message_data)}")
        elif text := generation.get("text"):
            parsed_messages.append(text)
        if generation_info := generation.get("generation_info"):
            assert hasattr(generation_info, "get"), f"expected Mapping, found {type(generation_info)}"
            if finish_reason := generation_info.get("finish_reason"):
                yield LLM_RESPONSE_FINISH_REASON, finish_reason
            if token_usage := generation_info.get("token_usage"):
                assert hasattr(token_usage, "get"), f"expected Mapping, found {type(token_usage)}"
                for attribute_name, key in [
                    (LLM_USAGE_PROMPT_TOKENS, "input_tokens"),
                    (LLM_USAGE_COMPLETION_TOKENS, "output_tokens"),
                    (LLM_USAGE_TOTAL_TOKENS, "total_tokens"),
                ]:
                    if (token_count := token_usage.get(key)) is not None:
                        yield attribute_name, token_count
    if parsed_messages:
        yield LLM_OUTPUT_MESSAGES, parsed_messages
    if not (llm_output := outputs.get("llm_output")):
        return
    assert hasattr(llm_output, "get"), f"expected Mapping, found {type(llm_output)}"
    if model_name := llm_output.get("model_name"):
        yield LLM_RESPONSE_MODEL_NAME, model_name


@stop_on_exception
def _parse_prompt_data(prompt_data: Optional[str]) -> Iterator[Tuple[str, Any]]:
    if not prompt_data:
        return
    assert isinstance(prompt_data, str), f"expected str, found {type(prompt_data)}"
    yield CONTENT, process_content(prompt_data)


@stop_on_exception
def _parse_message_data(message_data: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, Any]]:
    """Parses message data to grab message role, content, etc.
    
    Example HumanMessage structure:
    {
        'additional_kwargs': {'session_id': 's456', 'user_id': 'u123'},
        'content': '分析这段代码的性能问题',
        'example': False,
        'id': None,
        'name': None,
        'response_metadata': {},
        'type': 'human'
    }
    """
    if not message_data:
        return
    if not hasattr(message_data, "get"):
        logger.warning(f"Invalid message_data type: expected Mapping, found {type(message_data)}")
        return
    id_ = message_data.get("id")
    if not isinstance(id_, list) or not id_:
        logger.warning(f"Invalid message id format: expected non-empty list, found {type(id_)}")
        return
    message_class_name = id_[-1]
    if message_class_name.startswith("HumanMessage"):
        role = "user"
    elif message_class_name.startswith("AIMessage"):
        role = "assistant"
    elif message_class_name.startswith("SystemMessage"):
        role = "system"
    elif message_class_name.startswith("FunctionMessage"):
        role = "function"
    elif message_class_name.startswith("ToolMessage"):
        role = "tool"
    elif message_class_name.startswith("ChatMessage"):
        role = message_data["kwargs"]["role"]
    else:
        raise ValueError(f"Cannot parse message of type: {message_class_name}")
    yield MESSAGE_ROLE, role
    if kwargs := message_data.get("kwargs"):
        assert hasattr(kwargs, "get"), f"expected Mapping, found {type(kwargs)}"
        if content := kwargs.get("content"):
            if isinstance(content, str):
                yield MESSAGE_CONTENT, process_content(content)
            elif isinstance(content, list):
                # Handle list content (e.g., multimodal content)
                for i, obj in enumerate(content):
                    if isinstance(obj, str):
                        yield f"{MESSAGE_CONTENT}.{i}", process_content(obj)
                    elif hasattr(obj, "get"):
                        yield f"{MESSAGE_CONTENT}.{i}", process_content(str(obj))
                    else:
                        logger.warning(f"Unexpected content object type: {type(obj)}")
            else:
                logger.warning(f"Unexpected content type: {type(content)}")

        if name := kwargs.get("name"):
            if isinstance(name, str):
                yield MESSAGE_NAME, name
            else:
                logger.warning(f"Expected str for name, found {type(name)}")
        if additional_kwargs := kwargs.get("additional_kwargs"):
            assert hasattr(
                additional_kwargs, "get"
            ), f"expected Mapping, found {type(additional_kwargs)}"
            if function_call := additional_kwargs.get("function_call"):
                assert hasattr(
                    function_call, "get"
                ), f"expected Mapping, found {type(function_call)}"
                if name := function_call.get("name"):
                    assert isinstance(name, str), f"expected str, found {type(name)}"
                    yield MESSAGE_FUNCTION_CALL_NAME, name
                if arguments := function_call.get("arguments"):
                    assert isinstance(arguments, str), f"expected str, found {type(arguments)}"
                    yield MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON, arguments
            if tool_calls := additional_kwargs.get("tool_calls"):
                assert isinstance(
                    tool_calls, Iterable
                ), f"expected Iterable, found {type(tool_calls)}"
                message_tool_calls = []
                for tool_call in tool_calls:
                    if message_tool_call := dict(_get_tool_call(tool_call)):
                        message_tool_calls.append(message_tool_call)
                if message_tool_calls:
                    yield MESSAGE_TOOL_CALLS, message_tool_calls


@stop_on_exception
def _get_tool_call(tool_call: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, Any]]:
    if not tool_call:
        return
    assert hasattr(tool_call, "get"), f"expected Mapping, found {type(tool_call)}"
    if function := tool_call.get("function"):
        assert hasattr(function, "get"), f"expected Mapping, found {type(function)}"
        if name := function.get("name"):
            assert isinstance(name, str), f"expected str, found {type(name)}"
            yield TOOL_CALL_FUNCTION_NAME, name
        if arguments := function.get("arguments"):
            assert isinstance(arguments, str), f"expected str, found {type(arguments)}"
            yield TOOL_CALL_FUNCTION_ARGUMENTS_JSON, arguments


@stop_on_exception
def _prompt_template(run: Run) -> Iterator[Tuple[str, Any]]:
    """
    A best-effort attempt to locate the PromptTemplate object among the
    keyword arguments of a serialized object, e.g. an LLMChain object.
    """
    serialized: Optional[Mapping[str, Any]] = run.serialized
    if not serialized:
        return
    assert hasattr(serialized, "get"), f"expected Mapping, found {type(serialized)}"
    if not (kwargs := serialized.get("kwargs")):
        return
    assert isinstance(kwargs, dict), f"expected dict, found {type(kwargs)}"
    for obj in kwargs.values():
        if not hasattr(obj, "get") or not (id_ := obj.get("id")):
            continue
        # The `id` field of the object is a list indicating the path to the
        # object's class in the LangChain package, e.g. `PromptTemplate` in
        # the `langchain.prompts.prompt` module is represented as
        # ["langchain", "prompts", "prompt", "PromptTemplate"]
        assert isinstance(id_, Sequence), f"expected list, found {type(id_)}"
        if id_[-1].endswith("PromptTemplate"):
            if not (kwargs := obj.get("kwargs")):
                continue
            assert hasattr(kwargs, "get"), f"expected Mapping, found {type(kwargs)}"
            if not (template := kwargs.get("template", "")):
                continue
            yield LLM_PROMPT_TEMPLATE, template
            if input_variables := kwargs.get("input_variables"):
                assert isinstance(
                    input_variables, list
                ), f"expected list, found {type(input_variables)}"
                template_variables = {}
                for variable in input_variables:
                    if (value := run.inputs.get(variable)) is not None:
                        template_variables[variable] = value
                if template_variables:
                    yield LLM_PROMPT_TEMPLATE_VARIABLES, json.dumps(template_variables, cls=_SafeJSONEncoder)
            break


@stop_on_exception
def _model_name(extra: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, str]]:
    """Yields model name if present."""
    if not extra:
        return
    assert hasattr(extra, "get"), f"expected Mapping, found {type(extra)}"
    if not (invocation_params := extra.get("invocation_params")):
        return
    for key in ["model_name", "model"]:
        if name := invocation_params.get(key):
            yield LLM_MODEL_NAME, name
            return


def get_attr_or_key(obj, name, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def extract_token_counts_from_token_usage(token_usage):
    mapping = [
        (LLM_USAGE_PROMPT_TOKENS, ["prompt_tokens", "PromptTokens", "input_tokens"]),
        (LLM_USAGE_COMPLETION_TOKENS, ["completion_tokens", "CompletionTokens", "output_tokens"]),
        (LLM_USAGE_TOTAL_TOKENS, ["total_tokens", "TotalTokens"]),
    ]
    for attr, keys in mapping:
        for key in keys:
            if (value := get_attr_or_key(token_usage, key)) is not None:
                yield attr, value
                break


@stop_on_exception
def _token_counts(outputs: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, int]]:
    """Yields token count information if present, from llm_output, generation_info, or response_metadata, only once."""
    if not outputs or not hasattr(outputs, "get"):
        return
    # 1. llm_output
    if (llm_output := outputs.get("llm_output", None)) is not None:
        if (token_usage := get_attr_or_key(llm_output, "token_usage")) is not None:
            yield from extract_token_counts_from_token_usage(token_usage)
            logger.debug("token get from llm_output")
            return
    # 2. generations
    generations = outputs.get("generations")
    if generations and isinstance(generations, list):
        for group in generations:
            if not isinstance(group, list):
                continue
            for generation in group:
                generation_info = get_attr_or_key(generation, "generation_info")
                if (token_usage := get_attr_or_key(generation_info, "token_usage")):
                    yield from extract_token_counts_from_token_usage(token_usage)
                    logger.debug("token get from generations generation_info")
                    return
                message = get_attr_or_key(generation, "message")
                if (response_metadata := get_attr_or_key(message, "response_metadata")):
                    if (token_usage := get_attr_or_key(response_metadata, "token_usage")):
                        yield from extract_token_counts_from_token_usage(token_usage)
                        logger.debug("token get from generations message response_metadata")
                        return
    return


@stop_on_exception
def _tools(run: Run) -> Iterator[Tuple[str, str]]:
    """Yields tool attributes if present."""
    if run.run_type.lower() != "tool":
        return
    if not (serialized := run.serialized):
        return
    assert hasattr(serialized, "get"), f"expected Mapping, found {type(serialized)}"
    if name := serialized.get("name"):
        yield TOOL_NAME, name
    if description := serialized.get("description"):
        yield TOOL_DESCRIPTION, description


@stop_on_exception
def _retrieval_documents(run: Run) -> Iterator[Tuple[str, List[Mapping[str, Any]]]]:
    if run.run_type.lower() != "retriever":
        return
    if not (outputs := run.outputs):
        return
    assert hasattr(outputs, "get"), f"expected Mapping, found {type(outputs)}"
    documents = outputs.get("documents")
    assert isinstance(documents, Iterable), f"expected Iterable, found {type(documents)}"
    yield RETRIEVAL_DOCUMENTS, [dict(_as_document(document)) for document in documents]


@stop_on_exception
def _metadata(run: Run) -> Iterator[Tuple[str, str]]:
    """
    Takes the LangChain chain metadata and adds it to the trace
    """
    if not run.extra or not (metadata := run.extra.get("metadata")):
        return
    if not isinstance(metadata, Mapping):
        logger.warning(f"Invalid metadata type: expected Mapping, found {type(metadata)}")
        return

    # 获取 session_id
    if session_id := (
        metadata.get("session_id")
        or metadata.get("conversation_id")
        or metadata.get("thread_id")
    ):
        yield LLM_SESSION_ID, session_id
    
    # 获取 user_id
    if user_id := metadata.get("user_id"):
        yield LLM_USER_ID, user_id

    yield METADATA, json.dumps(metadata)


@stop_on_exception
def _as_document(document: Any) -> Iterator[Tuple[str, Any]]:
    if page_content := getattr(document, "page_content", None):
        assert isinstance(page_content, str), f"expected str, found {type(page_content)}"
        yield DOCUMENT_CONTENT, process_content(page_content)
    if metadata := getattr(document, "metadata", None):
        assert isinstance(metadata, Mapping), f"expected Mapping, found {type(metadata)}"
        yield DOCUMENT_METADATA, json.dumps(metadata, cls=_SafeJSONEncoder)


class _SafeJSONEncoder(json.JSONEncoder):
    """
    A JSON encoder that falls back to the string representation of a
    non-JSON-serializable object rather than raising an error.
    """

    def default(self, obj: Any) -> Any:
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def _as_utc_nano(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1_000_000_000)