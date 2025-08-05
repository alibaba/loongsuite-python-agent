from abc import ABC, abstractmethod
from typing import Any, Tuple, Mapping

from opentelemetry.metrics import get_meter

from opentelemetry.instrumentation.dify.capture_content import set_dict_value
from opentelemetry.instrumentation.dify.dify_utils import get_message_data
from opentelemetry.instrumentation.dify.semconv import INPUT_VALUE
from opentelemetry.instrumentation.dify.version import __version__


class ProcessStrategy(ABC):
    """Base abstract class for all process strategies in Dify instrumentation.

    This class provides the foundation for tracking and monitoring different aspects of Dify's operations.
    It handles metrics collection, tracing, and event management for various Dify components.

    Attributes:
        _handler: The handler instance that manages the overall instrumentation process
        _tracer: OpenTelemetry tracer for creating and managing spans
        _lock: Thread lock for thread-safe operations
        _event_data: Dictionary storing event-related data
        _logger: Logger instance for recording events and errors
        _meter: OpenTelemetry meter for collecting metrics
    """

    def __init__(self, handler: Any):
        self._handler = handler
        self._tracer = handler._tracer
        self._lock = handler._lock
        self._event_data = handler._event_data
        self._logger = handler._logger
        self._meter = get_meter(
            __name__,
            __version__,
            None,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )

    def _extract_inputs(self, inputs):
        if inputs is None:
            return {}
        input_attributes = {}
        input_value = ""
        input_key = INPUT_VALUE
        if inputs is None:
            input_attributes[input_key] = "{}"
            return input_attributes
        if "sys.query" in inputs:
            input_value = inputs["sys.query"]
        else:
            input_value = f"{inputs}"
        if input_value is None:
            return input_attributes
        set_dict_value(input_attributes, input_key, input_value)
        return input_attributes


    @abstractmethod
    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        pass

    def before_process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], ):
        pass

    def _get_data(self, src, key, default=None):
        if key in src:
            return src[key]
        else:
            return default

    def _get_message_data(self, message_id: str):
        return get_message_data(message_id)
