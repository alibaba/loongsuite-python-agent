import json

import threading

from opentelemetry.semconv.trace import SpanAttributes, MessageAttributes, SpanKindValues, DocumentAttributes
#  dify packages path
fromopentelemetry.sdk.extension.semconv.metrics import CommonServiceMetrics
from opentelemetry.metrics import get_meter
from logging import getLogger
from opentelemetry.instrumentation.dify.contants import _get_dify_app_name_key, DIFY_APP_ID_KEY
from abc import ABC
from opentelemetry import trace as trace_api
from opentelemetry import context as context_api
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    TypeVar,
)
from typing_extensions import TypeAlias
from opentelemetry.instrumentation.dify.entities import NodeType
# dify packages
from models.model import App
from extensions.ext_database import db
from models.model import Message
from opentelemetry.instrumentation.dify.version import __version__
from opentelemetry.sdk.extension.arms.common.utils.metrics_utils import get_llm_common_attributes
from opentelemetry.instrumentation.dify._strategy import StrategyFactory

_logger = getLogger(__name__)

_DIFY_APP_NAME_KEY = _get_dify_app_name_key()

_EventId: TypeAlias = str
_ParentId: TypeAlias = str

from .entities import _EventData

_Value = TypeVar("_Value")


class _BoundedDict(OrderedDict[str, _Value]):
    """
    One use case for this is when the LLM raises an exception in the following code location, in
    which case the LLM event will never be popped and will remain in the container forever.
    https://github.com/run-llama/llama_index/blob/dcef41ee67925cccf1ee7bb2dd386bcf0564ba29/llama_index/llms/base.py#L62
    Therefore, to prevent memory leak, this container is limited to a certain capacity, and when it
    reaches that capacity, the oldest item by insertion order will be popped.
    """  # noqa: E501

    def __init__(
            self,
            capacity: int = 1000,
            on_evict_fn: Optional[Callable[[_Value], None]] = None,
    ) -> None:
        super().__init__()
        self._capacity = capacity
        self._on_evict_fn = on_evict_fn

    def __setitem__(self, key: str, value: _Value) -> None:
        if key not in self and len(self) >= self._capacity > 0:
            # pop the oldest item by insertion order
            _, oldest = self.popitem(last=False)
            if self._on_evict_fn:
                self._on_evict_fn(oldest)
        super().__setitem__(key, value)


import copy


class AppGeneratorHandler(ABC):

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[type, Any],
            kwargs: Mapping[str, Any],
    ) -> Any:
        method = wrapped.__qualname__
        res = None
        if method.endswith("generate"):
            instance._otel_context = context_api.get_current()
            return wrapped(*args, **kwargs)
        if method.endswith("_generate_worker"):
            token = None
            try:
                token = context_api.attach(instance._otel_context)
                return wrapped(*args, **kwargs)
            finally:
                context_api.detach(token)


class QueueHandler(ABC):

    def __init__(self, tracer) -> None:
        self._tracer = tracer

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[type, Any],
            kwargs: Mapping[str, Any],
    ) -> Any:
        method = wrapped.__qualname__
        res = None
        if method.endswith("run"):
            from opentelemetry import context
            ctx = context.get_current()
            with self._tracer.start_as_current_span(name="node_run") as span:
                res = wrapped(*args, **kwargs)
        if method.endswith("_publish"):
            from opentelemetry import context
            otel_ctx = context.get_current()
            args[0]._otel_ctx = otel_ctx
            res = wrapped(*args, **kwargs)
        else:
            res = wrapped(*args, **kwargs)
        return res


def stop_on_exception(
        wrapped: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return wrapped(*args, **kwargs)
        except Exception:
            _logger.exception(f"Fail to process data, func name: {wrapped.__name__}")
            return None

    return wrapper


class DifyHandler:
    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer
        self._meter = get_meter(
            __name__,
            __version__,
            None,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
        self._lock = threading.Lock()
        self._event_data: Dict[str, _EventData] = _BoundedDict()
        self._logger = getLogger(__name__)
        self._strategy_factory = StrategyFactory(self)
        self._app_list: Dict[str, str] = _BoundedDict()
        self._init_metrics()

    def _init_metrics(self):
        meter = self._meter
        self.calls_count = CommonServiceMetrics(meter).calls_count
        self.calls_duration_seconds = CommonServiceMetrics(meter).calls_duration_seconds
        self.llm_context_size = CommonServiceMetrics(meter).llm_context_size
        self.llm_prompt_size = CommonServiceMetrics(meter).llm_prompt_size
        self.llm_output_token_seconds = CommonServiceMetrics(meter).llm_output_token_seconds
        self.llm_usage_tokens = CommonServiceMetrics(meter).llm_usage_tokens
        self.llm_first_token_seconds = CommonServiceMetrics(meter).llm_first_token_seconds
        self.calls_error_count = CommonServiceMetrics(meter).call_error_count

    def get_app_name_by_id(self, app_id: str) -> str:
        if app_id is None:
            return "NO_FOUND"
        app_name = None
        app_name = self._app_list.get(app_id, None)
        if app_name is not None:
            return app_name
        app_info = (
            db.session.query(
                App.id,
                App.name,
            )
            .filter(App.id == app_id)
            .all()
        )
        if len(app_info) <= 0:
            return app_id
        app_name = app_info[0].name
        if app_name is None:
            return app_id
        with self._lock:
            self._app_list[app_id] = app_name
        return app_name

    def get_llm_common_attributes(self) -> dict:
        return {
            "component": "dify",
            "service.name": "dify",
        }

    def _get_message_data(self, message_id: str):
        return db.session.query(Message).filter(Message.id == message_id).first()

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[type, Any],
            kwargs: Mapping[str, Any],
    ) -> Any:
        try:
            method = wrapped.__name__
            self._before_process(method, instance, args, kwargs)
        except:
            pass
        res = wrapped(*args, **kwargs)
        try:
            method = wrapped.__name__
            self._after_process(method, instance, args, kwargs, res)
        except Exception as e:
            pass
        return res

    def _before_process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any]):
        strategy = self._strategy_factory.get_strategy(method)
        if strategy:
            strategy.before_process(method, instance, args, kwargs)

    def _after_process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any],
                       res: Any) -> None:
        strategy = self._strategy_factory.get_strategy(method)
        if strategy:
            strategy.process(method, instance, args, kwargs, res)

    def _record_metrics(self, event_data, metrics_attributes, error=None):
        if error:
            self.calls_error_count.add(1, metrics_attributes)
        self.calls_count.add(1, metrics_attributes)
        if event_data.end_time:
            duration = (event_data.end_time - event_data.start_time) / 1e9
            self.calls_duration_seconds.record(duration, metrics_attributes)


@stop_on_exception
def _extract_workflow_node_attributes(event: Any) -> str:
    node_type = _getattr(event, "node_type", None)
    if node_type is None:
        node_type = "DEFAULT_NODE_TYPE"
    node_type = _getattr(node_type, "value", "DEFAULT_NODE_TYPE")
    span_kind = _get_span_kind_by_node_type(node_type)
    span_attriubtes = {}
    span_attriubtes[SpanAttributes.GEN_AI_SPAN_KIND] = span_kind
    inputs = _getattr(event, "inputs", None)
    input_attributes = _extract_inputs(inputs)
    if input_attributes is not None:
        span_attriubtes.update(input_attributes)
    outputs = _getattr(event, "outputs", None)
    output_attributes = _extract_outputs(outputs)
    span_attriubtes.update(output_attributes)
    if span_kind == SpanKindValues.LLM.value:
        llm_attributes = _extract_llm_attributes(event)
        span_attriubtes.update(llm_attributes)
        metrics_attriubtes = get_llm_common_attributes()
        if SpanAttributes.GEN_AI_REQUEST_MODEL_NAME in span_attriubtes:
            metrics_attriubtes["modelName"] = span_attriubtes[SpanAttributes.GEN_AI_REQUEST_MODEL_NAME]
        if SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS in span_attriubtes:
            input_tokens = span_attriubtes[SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS]
            metrics_attriubtes["usageType"] = "input"
            # self.llm_usage_tokens.add(input_tokens, attributes=metrics_attriubtes)
        if SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS in span_attriubtes:
            output_tokens = span_attriubtes[SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS]
            metrics_attriubtes["usageType"] = "output"
            # self.llm_usage_tokens.add(output_tokens, attributes=metrics_attriubtes)
    if span_kind == SpanKindValues.RETRIEVER.value:
        retriever_attributes = _extract_retrieval_attributes(event)
        span_attriubtes.update(retriever_attributes)
    return span_attriubtes


@stop_on_exception
def _extract_retrieval_attributes(event):
    retrieval_attributes = {}
    output = None
    output = _getattr(event, "outputs", None)
    if output is None:
        return retrieval_attributes
    result = None
    if "result" in output:
        result = output["result"]
    if result is not None:
        idx = 0
        for document in result:
            k_prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{idx}"
            if "metadata" in document:
                metadata = document["metadata"]
                if "document_id" in metadata:
                    retrieval_attributes[f"{k_prefix}.{DocumentAttributes.DOCUMENT_ID}"] = metadata["document_id"]
                if "score" in metadata:
                    retrieval_attributes[f"{k_prefix}.{DocumentAttributes.DOCUMENT_SCORE}"] = metadata["score"]
                retrieval_attributes[f"{k_prefix}.{DocumentAttributes.DOCUMENT_METADATA}"] = json.dumps(metadata,
                                                                                                        ensure_ascii=False)
            retrieval_attributes[f"{k_prefix}.{DocumentAttributes.DOCUMENT_CONTENT}"] = document["content"]
            idx += 1
    return retrieval_attributes


@stop_on_exception
def _extract_llm_attributes(event):
    llm_attributes = {}
    if node_data := _getattr(event, "node_data", None):
        if single_retrieval_config := _getattr(node_data, "single_retrieval_config", None):
            # single_retrieval_config = node_data["single_retrieval_config"]
            if model := _getattr(single_retrieval_config, "model", None):
                # if "model" in single_retrieval_config:
                model = single_retrieval_config["model"]
                if "name" in model:
                    llm_attributes[SpanAttributes.GEN_AI_REQUEST_MODEL_NAME] = model["name"]
                    llm_attributes[SpanAttributes.GEN_AI_MODEL_NAME] = model["name"]
                if "provider" in model:
                    llm_attributes[SpanAttributes.GEN_AI_SYSTEM] = model["provider"]
        if model := _getattr(node_data, "model", None):
            # if "model" in node_data:
            #     model = node_data["model"]
            if name := _getattr(model, "name", None):
                llm_attributes[SpanAttributes.GEN_AI_REQUEST_MODEL_NAME] = name
                llm_attributes[SpanAttributes.GEN_AI_MODEL_NAME] = name
            if provider := _getattr(model, "provider", None):
                llm_attributes[SpanAttributes.GEN_AI_SYSTEM] = provider

    if process_data := _getattr(event, "process_data", None):
        if "prompts" in process_data:
            prompts = process_data["prompts"]
            idx = 0
            for prompt in prompts:
                if "role" in prompt:
                    llm_attributes[
                        f"{SpanAttributes.GEN_AI_PROMPT}.{idx}.{MessageAttributes.MESSAGE_ROLE}"] = \
                        f"{prompt['role']}"
                if "text" in prompt:
                    llm_attributes[
                        f"{SpanAttributes.GEN_AI_PROMPT}.{idx}.{MessageAttributes.MESSAGE_CONTENT}"] = \
                        f"{prompt['text']}"
                idx += 1
        if usage := _get_data(src=process_data, key="usage"):
            if prompt_tokens := _get_data(src=usage, key="prompt_tokens"):
                llm_attributes[SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = prompt_tokens
            if completion_tokens := _get_data(src=usage, key="completion_tokens"):
                llm_attributes[SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = completion_tokens
            if total_tokens := _get_data(src=usage, key="total_tokens"):
                llm_attributes[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] = total_tokens
        if finish_reason := _get_data(src=process_data, key="finish_reason"):
            llm_attributes[SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON] = finish_reason

    if outputs := _getattr(event, "outputs", None):
        if text := _get_data(outputs, "text", None):
            key = f"{SpanAttributes.GEN_AI_PROMPT}.{SpanAttributes.CONTENT}"
            llm_attributes[key] = text
        if usage := _get_data(outputs, "usage", None):
            if prompt_tokens := _get_data(usage, "prompt_tokens", None):
                llm_attributes[SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = prompt_tokens
            if completion_tokens := _get_data(usage, "completion_tokens"):
                llm_attributes[SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = completion_tokens
            if total_tokens := _get_data(usage, "total_tokens", None):
                llm_attributes[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] = total_tokens
        if finish_reason := _get_data(outputs, "finish_reason", None):
            llm_attributes[SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON] = finish_reason
    return llm_attributes


def _getattr(o, k, default=None):
    r = getattr(o, k, default)
    if r is None:
        _logger.debug(f"can not get {k}")
    return r


def _get_data(src, key, default=None):
    if key in src:
        return src[key]
    else:
        return default


@stop_on_exception
def _extract_inputs(inputs):
    if inputs is None:
        return {}
    input_attributes = {}
    input_value = ""
    input_key = SpanAttributes.INPUT_VALUE
    if inputs is None:
        input_attributes[input_key] = "{}"
        return input_attributes
    if "sys.query" in inputs:
        input_value = inputs["sys.query"]
    else:
        input_value = f"{inputs}"
    if input_value is None:
        return input_attributes
    input_attributes[input_key] = input_value
    return input_attributes


@stop_on_exception
def _extract_outputs(outputs):
    if outputs is None:
        return {}
    output_attributes = {}
    output = ""
    output_key = SpanAttributes.OUTPUT_VALUE
    if "sys.query" in outputs:
        output = outputs["sys.query"]
    elif "answer" in outputs:
        output = outputs["answer"]
    elif "text" in outputs:
        output = outputs["text"]
    else:
        output = f"{outputs}"
    if output is None:
        return output_attributes
    output_attributes[output_key] = output
    return output_attributes


def get_app_id():
    pass


def _get_span_kind_by_node_type(node_type):
    span_attributes = {}
    if (node_type == NodeType.LLM.value
            or node_type == NodeType.QUESTION_CLASSIFIER.value
            or node_type == NodeType.PARAMETER_EXTRACTOR.value):
        span_kind = SpanKindValues.LLM.value
    elif node_type == NodeType.TOOL.value or node_type == NodeType.HTTP_REQUEST.value:
        span_kind = SpanKindValues.TOOL.value
    elif node_type == NodeType.KNOWLEDGE_RETRIEVAL.value:
        span_kind = SpanKindValues.RETRIEVER.value
    else:
        span_kind = SpanKindValues.TASK.value
    return span_kind
