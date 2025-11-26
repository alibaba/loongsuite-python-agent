"""
OpenTelemetry GenAI Metrics Collector for Google ADK.

This module implements standard OpenTelemetry GenAI metrics collection
following the latest GenAI semantic conventions.
"""

import logging
from typing import Optional

from opentelemetry.metrics import Meter
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics

_logger = logging.getLogger(__name__)


class Instruments:
    """
    Standard OpenTelemetry GenAI instrumentation instruments.
    
    This class follows the same pattern as openai-v2/instruments.py
    and implements only the 2 standard GenAI client metrics.
    """
    
    def __init__(self, meter: Meter):
        """
        Initialize standard GenAI instruments.
        
        Args:
            meter: OpenTelemetry meter instance
        """
        # ✅ Standard GenAI client metric 1: Operation duration
        self.operation_duration_histogram = (
            gen_ai_metrics.create_gen_ai_client_operation_duration(meter)
        )
        
        # ✅ Standard GenAI client metric 2: Token usage
        self.token_usage_histogram = (
            gen_ai_metrics.create_gen_ai_client_token_usage(meter)
        )


class AdkMetricsCollector:
    """
    Metrics collector for Google ADK following OpenTelemetry GenAI conventions.
    
    This collector implements ONLY the 2 standard GenAI client metrics:
    - gen_ai.client.operation.duration (Histogram, unit: seconds)
    - gen_ai.client.token.usage (Histogram, unit: tokens)
    
    All ARMS-specific metrics have been removed.
    """
    
    def __init__(self, meter: Meter):
        """
        Initialize the metrics collector.
        
        Args:
            meter: OpenTelemetry meter instance
        """
        self._instruments = Instruments(meter)
        _logger.debug("AdkMetricsCollector initialized with standard OTel GenAI metrics")

    def record_llm_call(
        self,
        operation_name: str,
        model_name: str,
        duration: float,
        error_type: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Record LLM call metrics following standard OTel GenAI conventions.
        
        Args:
            operation_name: Operation name (e.g., "chat")
            model_name: Model name
            duration: Duration in seconds
            error_type: Error type if error occurred
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            conversation_id: Conversation ID (not used in metrics due to high cardinality)
            user_id: User ID (not used in metrics due to high cardinality)
        """
        try:
            # ✅ Build standard attributes for operation.duration
            attributes = {
                "gen_ai.operation.name": operation_name,
                "gen_ai.provider.name": "google_adk",  # ✅ Required attribute
                "gen_ai.request.model": model_name,     # ✅ Recommended attribute
            }
            
            # ✅ Add error.type only if error occurred (Conditionally Required)
            if error_type:
                attributes["error.type"] = error_type
            
            # ✅ Record operation duration (Histogram, unit: seconds)
            self._instruments.operation_duration_histogram.record(
                duration, 
                attributes=attributes
            )
            
            # ✅ Record token usage (Histogram, unit: tokens)
            # Note: session_id and user_id are NOT included in metrics (high cardinality)
            if prompt_tokens > 0:
                self._instruments.token_usage_histogram.record(
                    prompt_tokens,
                    attributes={
                        **attributes,
                        "gen_ai.token.type": "input",  # ✅ Required for token.usage
                    }
                )
                
            if completion_tokens > 0:
                self._instruments.token_usage_histogram.record(
                    completion_tokens,
                    attributes={
                        **attributes,
                        "gen_ai.token.type": "output",  # ✅ Required for token.usage
                    }
                )
                
            _logger.debug(
                f"Recorded LLM metrics: operation={operation_name}, model={model_name}, "
                f"duration={duration:.3f}s, prompt_tokens={prompt_tokens}, "
                f"completion_tokens={completion_tokens}, error={error_type}"
            )
            
        except Exception as e:
            _logger.exception(f"Error recording LLM metrics: {e}")

    def record_agent_call(
        self,
        operation_name: str,
        agent_name: str,
        duration: float,
        error_type: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Record Agent call metrics following standard OTel GenAI conventions.
        
        Args:
            operation_name: Operation name (e.g., "invoke_agent")
            agent_name: Agent name
            duration: Duration in seconds
            error_type: Error type if error occurred
            conversation_id: Conversation ID (not used in metrics due to high cardinality)
            user_id: User ID (not used in metrics due to high cardinality)
        """
        try:
            # ✅ Build standard attributes
            attributes = {
                "gen_ai.operation.name": operation_name,
                "gen_ai.provider.name": "google_adk",  # ✅ Required
                "gen_ai.request.model": agent_name,     # ✅ Agent name as model
            }
            
            # ✅ Add error.type only if error occurred
            if error_type:
                attributes["error.type"] = error_type
            
            # ✅ Record operation duration (Histogram, unit: seconds)
            self._instruments.operation_duration_histogram.record(
                duration, 
                attributes=attributes
            )
            
            _logger.debug(
                f"Recorded Agent metrics: operation={operation_name}, agent={agent_name}, "
                f"duration={duration:.3f}s, error={error_type}"
            )
            
        except Exception as e:
            _logger.exception(f"Error recording Agent metrics: {e}")

    def record_tool_call(
        self,
        operation_name: str,
        tool_name: str,
        duration: float,
        error_type: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Record Tool call metrics following standard OTel GenAI conventions.
        
        Args:
            operation_name: Operation name (e.g., "execute_tool")
            tool_name: Tool name
            duration: Duration in seconds
            error_type: Error type if error occurred
            conversation_id: Conversation ID (not used in metrics due to high cardinality)
            user_id: User ID (not used in metrics due to high cardinality)
        """
        try:
            # ✅ Build standard attributes
            attributes = {
                "gen_ai.operation.name": operation_name,
                "gen_ai.provider.name": "google_adk",  # ✅ Required
                "gen_ai.request.model": tool_name,     # ✅ Tool name as model
            }
            
            # ✅ Add error.type only if error occurred
            if error_type:
                attributes["error.type"] = error_type
            
            # ✅ Record operation duration (Histogram, unit: seconds)
            self._instruments.operation_duration_histogram.record(
                duration, 
                attributes=attributes
            )
            
            _logger.debug(
                f"Recorded Tool metrics: operation={operation_name}, tool={tool_name}, "
                f"duration={duration:.3f}s, error={error_type}"
            )
            
        except Exception as e:
            _logger.exception(f"Error recording Tool metrics: {e}")

