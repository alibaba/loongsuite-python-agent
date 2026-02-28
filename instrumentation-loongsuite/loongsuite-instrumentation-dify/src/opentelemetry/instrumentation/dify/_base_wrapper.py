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

from abc import ABC
from typing import Any, Dict

from opentelemetry import trace
from opentelemetry.context import get_value
from opentelemetry.instrumentation.dify.constants import (
    DIFY_APP_ID_KEY,
    _get_dify_app_name_key,
)
from opentelemetry.instrumentation.dify.semconv import (
    GEN_AI_SESSION_ID,
    GEN_AI_USER_ID,
)
from opentelemetry.instrumentation.dify.utils import get_llm_common_attributes
from opentelemetry.instrumentation.dify.version import __version__
from opentelemetry.metrics import get_meter
from opentelemetry.trace import Tracer

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
        self._app_list: Dict[str, str] = {}
        self._init_metrics()

    def _init_metrics(self):
        # FIXME: ruff failed
        meter = self._meter  # noqa: F841

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
        user_id = get_value(GEN_AI_USER_ID)
        session_id = get_value(GEN_AI_SESSION_ID)
        if app_name:
            attributes[_DIFY_APP_NAME_KEY] = app_name
        if app_id:
            attributes[DIFY_APP_ID_KEY] = app_id
        if user_id:
            attributes[GEN_AI_USER_ID] = user_id
        if session_id:
            attributes[GEN_AI_SESSION_ID] = session_id
        return attributes

    def before_process(self):
        pass

    def after_process(self):
        pass

    def record_call_count(
        self, attributes: Dict[str, Any] = None, span_kind: str = None
    ):
        """记录调用次数"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)

    def record_duration(
        self,
        duration: float,
        attributes: Dict[str, Any] = None,
        span_kind: str = None,
    ):
        """记录调用持续时间"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)

    def record_call_error_count(
        self, attributes: Dict[str, Any] = None, span_kind: str = None
    ):
        """记录调用错误次数"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)


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
        if (
            current_context
            and hasattr(current_context, "trace_id")
            and hasattr(current_context, "span_id")
        ):
            # Create trace headers
            trace_headers = {}
            # Create traceparent header
            trace_id_hex = format(current_context.trace_id, "032x")
            span_id_hex = format(current_context.span_id, "016x")
            flags = format(
                int(current_context.trace_flags)
                if hasattr(current_context, "trace_flags")
                else 1,
                "02x",
            )
            traceparent = f"00-{trace_id_hex}-{span_id_hex}-{flags}"
            trace_headers["traceparent"] = traceparent
            # Add tracestate if available
            if (
                hasattr(current_context, "trace_state")
                and current_context.trace_state
            ):
                trace_headers["tracestate"] = str(current_context.trace_state)

            return trace_headers

    def record_call_count(
        self,
        model_name: str,
        attributes: Dict[str, Any] = None,
        span_kind: str = "LLM",
    ):
        """记录调用次数"""
        if attributes is None:
            attributes = {}
        attributes["modelName"] = model_name
        super().record_call_count(attributes, span_kind)

    def record_duration(
        self,
        duration: float,
        model_name: str,
        attributes: Dict[str, Any] = None,
        span_kind: str = "LLM",
    ):
        """记录调用持续时间"""
        if attributes is None:
            attributes = {}
        attributes["modelName"] = model_name
        super().record_duration(duration, attributes, span_kind)

    def record_call_error_count(
        self,
        model_name: str,
        attributes: Dict[str, Any] = None,
        span_kind: str = "LLM",
    ):
        """记录调用错误次数"""
        if attributes is None:
            attributes = {}
        attributes["modelName"] = model_name
        super().record_call_error_count(attributes, span_kind)

    def record_llm_output_token_seconds(
        self,
        duration: float,
        attributes: Dict[str, Any] = None,
        span_kind: str = "LLM",
    ):
        """记录LLM输出token的持续时间"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)

    def record_first_token_seconds(
        self,
        duration: float,
        model_name: str,
        attributes: Dict[str, Any] = None,
        span_kind: str = "LLM",
    ):
        """记录首包耗时"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)
        common_attrs["modelName"] = model_name

    def _record_llm_tokens(
        self,
        tokens: int,
        usage_type: str,
        model_name: str,
        attributes: Dict[str, Any] = None,
        span_kind: str = "LLM",
    ):
        """记录LLM token数量的通用方法"""
        common_attrs = self.get_common_attributes()
        if span_kind:
            common_attrs["spanKind"] = span_kind
        if attributes:
            common_attrs.update(attributes)
        common_attrs["usageType"] = usage_type
        common_attrs["modelName"] = model_name

    def record_llm_input_tokens(
        self,
        tokens: int,
        model_name: str,
        attributes: Dict[str, Any] = None,
        span_kind: str = "LLM",
    ):
        """记录LLM输入token的数量"""
        self._record_llm_tokens(
            tokens, "input", model_name, attributes, span_kind
        )

    def record_llm_output_tokens(
        self,
        tokens: int,
        model_name: str,
        attributes: Dict[str, Any] = None,
        span_kind: str = "LLM",
    ):
        """记录LLM输出token的数量"""
        self._record_llm_tokens(
            tokens, "output", model_name, attributes, span_kind
        )


class TOOLBaseWrapper(BaseWrapper):
    def __init__(self, tracer: Tracer):
        super().__init__(tracer)

    def _init_metrics(self):
        super()._init_metrics()

    def record_call_count(
        self,
        tool_name: str,
        attributes: Dict[str, Any] = None,
        span_kind: str = "TOOL",
    ):
        """记录调用次数"""
        if attributes is None:
            attributes = {}
        attributes["rpc"] = tool_name
        super().record_call_count(attributes, span_kind)

    def record_duration(
        self,
        duration: float,
        tool_name: str,
        attributes: Dict[str, Any] = None,
        span_kind: str = "TOOL",
    ):
        """记录调用持续时间"""
        if attributes is None:
            attributes = {}
        attributes["rpc"] = tool_name
        super().record_duration(duration, attributes, span_kind)

    def record_call_error_count(
        self,
        tool_name: str,
        attributes: Dict[str, Any] = None,
        span_kind: str = "TOOL",
    ):
        """记录调用错误次数"""
        if attributes is None:
            attributes = {}
        attributes["rpc"] = tool_name
        super().record_call_error_count(attributes, span_kind)
