from opentelemetry.instrumentation.agentscope._model_call_attributes_extractor import(
    ModelRequestAttributesExtractor,
    ModelResponseAttributesExtractor,
)
from opentelemetry.instrumentation.agentscope._tool_call_attributes_extractor import(
    ToolRequestAttributesExtractor,
    ToolResponseAttributesExtractor,
)
from opentelemetry.instrumentation.agentscope._with_span import _WithSpan
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Tuple,
)
from abc import ABC
from opentelemetry import trace as trace_api
from contextlib import contextmanager
from opentelemetry.util.types import AttributeValue
from opentelemetry.trace import INVALID_SPAN
from agentscope.models.response import ModelResponse
from functools import wraps
from aliyun.semconv.logger import getLogger
from os import environ

logger = getLogger(__name__)
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)
class _WithTracer(ABC):
    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @contextmanager
    def _start_as_current_span(
            self,
            span_name: str,
            attributes: Iterable[Tuple[str, AttributeValue]],
            extra_attributes: Iterable[Tuple[str, AttributeValue]],
    ) -> Iterator[_WithSpan]:
        # Because OTEL has a default limit of 128 attributes, we split our attributes into
        # two tiers, where the addition of "extra_attributes" is deferred until the end
        # and only after the "attributes" are added.
        try:
            span = self._tracer.start_span(name=span_name, attributes=dict(attributes))
        except Exception:
            logger.exception("Failed to start span")
            span = INVALID_SPAN
        with trace_api.use_span(
                span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
        ) as span:
            yield _WithSpan(span=span, extra_attributes=dict(extra_attributes))

class AgentscopeRequestWrapper(_WithTracer):

    def __init__(self, tracer, *args, **kwargs):
        super().__init__(tracer, *args, **kwargs)
        self.class_replaced = []
        self._request_attributes_extractor = ModelRequestAttributesExtractor()
        self._response_attributes_extractor = ModelResponseAttributesExtractor()

    def _finalize_response(
            self,
            response: Any,
            with_span: _WithSpan,
    ) -> Any:
        resp_attr = self._response_attributes_extractor.extract(response)
        with_span.finish_tracing(
            status=trace_api.Status(status_code=trace_api.StatusCode.OK),
            attributes=dict(resp_attr),
            extra_attributes=dict(resp_attr),
        )
        return response

    def _enable_genai_capture(self) -> bool:
        capture_content = environ.get(
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
        )
        return capture_content.lower() == "true"

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[type, Any],
            kwargs: Mapping[str, Any],
    ) -> None:
        wrapped(*args, **kwargs)
        model_class = instance.__class__
        if model_class in self.class_replaced:
            return
        original_call = model_class.__call__
        @wraps(original_call)
        def wrapped_call(
            model_instance: Any,
            *args: Tuple[type, Any],
            **kwargs: Mapping[str, Any],
        ) -> ModelResponse:
            if not self._enable_genai_capture():
                return original_call(model_instance, *args, **kwargs)
            if instance is None:
                return original_call(model_instance, *args, **kwargs)
            with self._start_as_current_span(
                span_name="LLM",
                attributes=self._request_attributes_extractor.extract(*args),
                extra_attributes=self._request_attributes_extractor.extract(**kwargs),
            ) as with_span:
                try:
                    response = original_call(model_instance, *args, **kwargs)
                except Exception as exception:
                    with_span.record_exception(exception)
                    status = trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        # Follow the format in OTEL SDK for description, see:
                        # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                        description=f"{type(exception).__name__}: {exception}",
                    )
                    with_span.finish_tracing(status=status)
                    raise
                try:
                    response = self._finalize_response(
                        response=response,
                        with_span=with_span,
                    )
                except Exception:
                    logger.exception(f"Failed to finalize response of type {type(response)}")
                    with_span.finish_tracing()

            return response
        instance.__class__.__call__ = wrapped_call
        self.class_replaced.append(model_class)

class AgentscopeToolcallWrapper(_WithTracer):

    def __init__(self, tracer, *args, **kwargs):
        super().__init__(tracer, *args, **kwargs)
        self._request_attributes_extractor = ToolRequestAttributesExtractor()
        self._response_attributes_extractor = ToolResponseAttributesExtractor()

    def _finalize_response(
            self,
            response: Any,
            with_span: _WithSpan,
    ) -> Any:
        resp_attr = self._response_attributes_extractor.extract(response)
        with_span.finish_tracing(
            status=trace_api.Status(status_code=trace_api.StatusCode.OK),
            attributes=dict(resp_attr),
            extra_attributes=dict(resp_attr),
        )
        return response

    def _enable_genai_capture(self) -> bool:
        capture_content = environ.get(
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false"
        )
        return capture_content.lower() == "true"

    def __call__(
            self,
            wrapped: Callable[..., Any],
            instance: Any,
            args: Tuple[type, Any],
            kwargs: Mapping[str, Any],
    ) -> None:
        tool_use_block = args[0] if args else kwargs.get("tool_call")
        if not self._enable_genai_capture():
            return wrapped(*args, **kwargs)
        if instance is None:
            return wrapped(*args, **kwargs)
        with self._start_as_current_span(
            span_name="ToolCall",
            attributes=self._request_attributes_extractor.extract(tool_use_block),
            extra_attributes=self._request_attributes_extractor.extract(tool_use_block),
        ) as with_span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                with_span.record_exception(exception)
                status = trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    # Follow the format in OTEL SDK for description, see:
                    # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                    description=f"{type(exception).__name__}: {exception}",
                )
                with_span.finish_tracing(status=status)
                raise
            try:
                response = self._finalize_response(
                    response=response,
                    with_span=with_span,
                )
            except Exception:
                logger.exception(f"Failed to finalize response of type {type(response)}")
                with_span.finish_tracing()
        return response