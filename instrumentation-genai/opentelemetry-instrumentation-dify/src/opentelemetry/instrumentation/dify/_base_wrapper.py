from typing import Any, Dict
from abc import ABC
from opentelemetry.metrics import get_meter
fromopentelemetry.sdk.extension.semconv.metrics import CommonServiceMetrics
from opentelemetry.sdk.extension.arms.common.utils.metrics_utils import get_llm_common_attributes
from opentelemetry.semconv.trace import SpanAttributes

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer
from opentelemetry.instrumentation.dify.version import __version__
from logging import getLogger
from opentelemetry.context import get_value
from opentelemetry.instrumentation.dify.contants import _get_dify_app_name_key, DIFY_APP_ID_KEY

_logger = getLogger(__name__)

_DIFY_APP_NAME_KEY = _get_dify_app_name_key()


class BaseWrapper(ABC):
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self._span_kind = "TASK"
        self._meter = get_meter(
            __name__,
            __version__,
            None,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
        self._logger = getLogger(__name__)
        self._app_list: Dict[str, str] = {}
        self._init_metrics()

    def _init_metrics(self):
        meter = self._meter
        self.calls_count = CommonServiceMetrics(meter).calls_count
        self.calls_duration_seconds = CommonServiceMetrics(meter).calls_duration_seconds
        self.llm_context_size = CommonServiceMetrics(meter).llm_context_size
        self.llm_prompt_size = CommonServiceMetrics(meter).llm_prompt_size
        self.calls_error_count = CommonServiceMetrics(meter).call_error_count

    def set_span_kind(self, span_kind: str):
        self._span_kind = span_kind

    def span_kind(self):
        return self._span_kind

    def get_common_attributes(self):
        attributes = get_llm_common_attributes()
        attributes["spanKind"] = self.span_kind()
        return attributes

    def extract_attributes_from_context(self) -> Dict:
        attributes = {}
        app_name = get_value(_DIFY_APP_NAME_KEY)
        app_id = get_value(DIFY_APP_ID_KEY)
        user_id = get_value(SpanAttributes.GEN_AI_USER_ID)
        session_id = get_value(SpanAttributes.GEN_AI_SESSION_ID)
        if app_name:
            attributes[_DIFY_APP_NAME_KEY] = app_name
        if app_id:
            attributes[DIFY_APP_ID_KEY] = app_id
        if user_id:
            attributes[SpanAttributes.GEN_AI_USER_ID] = user_id
        if session_id:
            attributes[SpanAttributes.GEN_AI_SESSION_ID] = session_id
        return attributes

    def before_process(self):
        pass

    def after_process(self):
        pass

    def record_call_count(self, attributes: Dict[str, Any] = None, span_kind: str = None):
        """记录调用次数"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)
        self.calls_count.add(1, common_attrs)

    def record_duration(self, duration: float, attributes: Dict[str, Any] = None, span_kind: str = None):
        """记录调用持续时间"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)
        self.calls_duration_seconds.record(duration, common_attrs)

    def record_call_error_count(self, attributes: Dict[str, Any] = None, span_kind: str = None):
        """记录调用错误次数"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)
        self.calls_error_count.add(1, common_attrs)


class LLMBaseWrapper(BaseWrapper):
    def __init__(self, tracer: Tracer):
        super().__init__(tracer)

    def get_trace_headers(self, current_span=None):
        # Get current context
        if current_span is None:
            current_span = trace.get_current_span()
        if not current_span:
            # logger.debug("No current span found")
            return {}
        current_context = current_span.get_span_context()
        # Only inject if we have a valid context
        if current_context and hasattr(current_context, "trace_id") and hasattr(current_context, "span_id"):
            # Create trace headers
            trace_headers = {}
            # Create traceparent header
            trace_id_hex = format(current_context.trace_id, "032x")
            span_id_hex = format(current_context.span_id, "016x")
            flags = format(int(current_context.trace_flags) if hasattr(current_context, "trace_flags") else 1,
                           "02x")
            traceparent = f"00-{trace_id_hex}-{span_id_hex}-{flags}"
            trace_headers["traceparent"] = traceparent
            # Add tracestate if available
            if hasattr(current_context, "trace_state") and current_context.trace_state:
                trace_headers["tracestate"] = str(current_context.trace_state)

            return trace_headers


    def _init_metrics(self):
        super()._init_metrics()
        self.llm_output_token_seconds = CommonServiceMetrics(self._meter).llm_output_token_seconds
        self.llm_usage_tokens = CommonServiceMetrics(self._meter).llm_usage_tokens
        self.llm_first_token_seconds = CommonServiceMetrics(self._meter).llm_first_token_seconds

    def record_call_count(self, model_name: str, attributes: Dict[str, Any] = None, span_kind: str = "LLM"):
        """记录调用次数"""
        if attributes is None:
            attributes = {}
        attributes["modelName"] = model_name
        super().record_call_count(attributes, span_kind)

    def record_duration(self, duration: float, model_name: str, attributes: Dict[str, Any] = None,
                        span_kind: str = "LLM"):
        """记录调用持续时间"""
        if attributes is None:
            attributes = {}
        attributes["modelName"] = model_name
        super().record_duration(duration, attributes, span_kind)

    def record_call_error_count(self, model_name: str, attributes: Dict[str, Any] = None, span_kind: str = "LLM"):
        """记录调用错误次数"""
        if attributes is None:
            attributes = {}
        attributes["modelName"] = model_name
        super().record_call_error_count(attributes, span_kind)

    def record_llm_output_token_seconds(self, duration: float, attributes: Dict[str, Any] = None,
                                        span_kind: str = "LLM"):
        """记录LLM输出token的持续时间"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)
        self.llm_output_token_seconds.record(duration, common_attrs)

    def record_first_token_seconds(self, duration: float, model_name: str, attributes: Dict[str, Any] = None,
                                   span_kind: str = "LLM"):
        """记录首包耗时"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)
        common_attrs["modelName"] = model_name
        self.llm_first_token_seconds.record(duration, common_attrs)

    def _record_llm_tokens(self, tokens: int, usage_type: str, model_name: str, attributes: Dict[str, Any] = None,
                           span_kind: str = "LLM"):
        """记录LLM token数量的通用方法"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)
        common_attrs["usageType"] = usage_type
        common_attrs["modelName"] = model_name
        self.llm_usage_tokens.add(tokens, common_attrs)

    def record_llm_input_tokens(self, tokens: int, model_name: str, attributes: Dict[str, Any] = None,
                                span_kind: str = "LLM"):
        """记录LLM输入token的数量"""
        self._record_llm_tokens(tokens, "input", model_name, attributes, span_kind)

    def record_llm_output_tokens(self, tokens: int, model_name: str, attributes: Dict[str, Any] = None,
                                 span_kind: str = "LLM"):
        """记录LLM输出token的数量"""
        self._record_llm_tokens(tokens, "output", model_name, attributes, span_kind)


class TOOLBaseWrapper(BaseWrapper):
    def __init__(self, tracer: Tracer):
        super().__init__(tracer)

    def _init_metrics(self):
        super()._init_metrics()

    def record_call_count(self, tool_name: str, attributes: Dict[str, Any] = None, span_kind: str = "TOOL"):
        """记录调用次数"""
        if attributes is None:
            attributes = {}
        attributes["rpc"] = tool_name
        super().record_call_count(attributes, span_kind)

    def record_duration(self, duration: float, tool_name: str, attributes: Dict[str, Any] = None,
                        span_kind: str = "TOOL"):
        """记录调用持续时间"""
        if attributes is None:
            attributes = {}
        attributes["rpc"] = tool_name
        super().record_duration(duration, attributes, span_kind)

    def record_call_error_count(self, tool_name: str, attributes: Dict[str, Any] = None, span_kind: str = "TOOL"):
        """记录调用错误次数"""
        if attributes is None:
            attributes = {}
        attributes["rpc"] = tool_name
        super().record_call_error_count(attributes, span_kind)
