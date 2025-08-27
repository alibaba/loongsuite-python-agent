import time
from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Tuple,
    Union,
)

from opentelemetry.instrumentation.mcp.metrics import ClientMetrics
from opentelemetry.instrumentation.mcp.semconv import MCPAttributes, _method_names_with_target, _metric_attribute_names
from opentelemetry import trace as trace_api
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.instrumentation.mcp.utils import (
    _is_capture_content_enabled,
    _safe_dump_attributes,
    _get_resource_result_size,
    _get_call_tool_result_size,
    _get_prompt_result_size,
    _get_complete_result_size,
    _get_logger,
)
from opentelemetry.instrumentation.mcp.session_handler import _SessionContext

logger = _get_logger(__name__)

_has_mcp_types = False
try:
    from mcp.types import (
        LATEST_PROTOCOL_VERSION,
        CallToolResult,
        CompleteResult,
        GetPromptResult,
        ReadResourceResult,
    )

    _has_mcp_types = True
except ImportError:
    _has_mcp_types = False


class RequestHandler(ABC):
    def __init__(self, method_name: str, tracer: trace_api.Tracer, metrics: ClientMetrics):
        self._tracer = tracer
        self._method_name = method_name
        self._metrics = metrics

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        start_time = time.time()
        input_attributes = self._get_input_attributes(instance, args, kwargs)
        span_name = self._get_span_name(instance, args, kwargs)

        with self._tracer.start_as_current_span(
            name=span_name,
            kind=SpanKind.CLIENT,
        ) as span:
            try:
                if span.is_recording():
                    span.set_attributes(input_attributes)
            except Exception as e:
                logger.debug("Failed to set span attributes", exc_info=True)

            # do request
            try:
                result = await wrapped(*args, **kwargs)
            except Exception as e:
                try:
                    if span.is_recording():
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    self._record_metrics(start_time, input_attributes, None)
                except Exception:
                    logger.debug("Failed to set span attributes while handling wrapped exception", exc_info=True)
                raise

            # after request
            try:
                if (
                    self._method_name == "initialize"
                    and (session_context := self._get_session_context(instance))
                    and (session_id := session_context._parse_session_id())
                ):
                    span.set_attribute(MCPAttributes.MCP_SESSION_ID, session_id)

                output_attributes, error_message = self._get_output_attributes(result)
                if error_message is not None:
                    span.set_status(Status(StatusCode.ERROR, error_message))
                else:
                    span.set_status(Status(StatusCode.OK))
                if span.is_recording():
                    span.set_attributes(output_attributes)
                self._record_metrics(start_time, input_attributes, output_attributes)
            except Exception:
                logger.debug("Failed to set span attributes and record metrics", exc_info=True)
            return result

    def _get_target_name(self, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
        if len(args) > 0:
            return str(args[0])
        _, arg_name = _method_names_with_target[self._method_name]
        if arg_name in kwargs:
            return str(kwargs[arg_name])
        return ""

    def _get_span_name(self, instance: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
        try:
            if self._method_name in _method_names_with_target:
                target_name = self._get_target_name(args, kwargs)
                return f"{self._method_name} {target_name}"
            return self._method_name
        except Exception:
            logger.debug(f"Failed to get span name for method {self._method_name}", exc_info=True)
            return self._method_name

    def _get_input_attributes(
        self,
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Dict[str, str]:
        try:
            input_attributes = {
                MCPAttributes.COMPONENT_NAME: MCPAttributes.MCP_CLIENT,
                MCPAttributes.MCP_METHOD_NAME: self._method_name,
                MCPAttributes.RPC_REQUEST_ID: str(instance._request_id),
            }
            if _has_mcp_types:
                input_attributes[MCPAttributes.MCP_CLIENT_VERSION] = LATEST_PROTOCOL_VERSION
            if self._method_name in _method_names_with_target:
                key, _ = _method_names_with_target[self._method_name]
                input_attributes[key] = self._get_target_name(args, kwargs)

            if _is_capture_content_enabled():
                input_value = _safe_dump_attributes(
                    {
                        "args": args,
                        "kwargs": kwargs,
                    },
                )
                if input_value:
                    input_attributes[MCPAttributes.MCP_PARAMETERS] = input_value

            if session_context := self._get_session_context(instance):
                input_attributes.update(session_context._get_session_attributes())

            return input_attributes
        except Exception:
            logger.debug("Failed to get input attributes", exc_info=True)
            return {}

    def _get_session_context(self, instance: Any) -> Union[_SessionContext, None]:
        if hasattr(instance, "_read_stream") and hasattr(instance._read_stream, "_session_context"):
            return instance._read_stream._session_context
        return None

    def _get_output_attributes(self, response: Any) -> Tuple[Dict[str, str], Union[str, None]]:
        try:
            output_attributes : Dict[str, str] = {}
            if _is_capture_content_enabled():
                output_value = _safe_dump_attributes(response)
                if output_value:
                    output_attributes[MCPAttributes.OUTPUT_VALUE] = output_value

            response_size = RequestHandler._calculate_response_size(response)
            if response_size is not None:
                output_attributes[MCPAttributes.MCP_OUTPUT_SIZE] = str(response_size)

            if _has_mcp_types and isinstance(response, CallToolResult) and response.isError:
                output_attributes[MCPAttributes.ERROR_TYPE] = "tool_error"
                return output_attributes, _safe_dump_attributes(response.content)

            return output_attributes, None
        except Exception:
            logger.debug("Failed to get output attributes", exc_info=True)
            return {}, None

    @classmethod
    def _calculate_response_size(cls, response: Any) -> Union[int, None]:
        if not _has_mcp_types:
            return None
        if isinstance(response, CallToolResult):
            return _get_call_tool_result_size(response)
        if isinstance(response, ReadResourceResult):
            return _get_resource_result_size(response)
        if isinstance(response, GetPromptResult):
            return _get_prompt_result_size(response)
        if isinstance(response, CompleteResult):
            return _get_complete_result_size(response)

        return None

    def _record_metrics(self, start_time: float, input_attributes: Union[Dict[str, str], None], output_attributes: Union[Dict[str, str], None]):
        metric_attributes : Dict[str, str] = {}
        for key in _metric_attribute_names:
            if input_attributes is not None and key in input_attributes:
                if value := input_attributes[key]:
                    metric_attributes[key] = value
            elif output_attributes is not None and key in output_attributes:
                if value := output_attributes[key]:
                    metric_attributes[key] = value
        self._metrics.operation_duration.record(time.time() - start_time, metric_attributes)
        self._metrics.operation_count.add(1, metric_attributes)
