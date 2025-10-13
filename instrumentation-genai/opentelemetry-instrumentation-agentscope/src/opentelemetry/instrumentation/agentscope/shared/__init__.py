# -*- coding: utf-8 -*-
"""
OpenTelemetry AgentScope Instrumentation Shared Module

This module provides shared components for AgentScope instrumentation,
including GenAI semantic conventions compliance, telemetry options,
and common attribute definitions.
"""

from .constants import *
from .telemetry_options import GenAITelemetryOptions, get_telemetry_options, set_telemetry_options
from .attributes import *

__all__ = [
    # Constants
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MAX_LENGTH", 
    "OTEL_INSTRUMENTATION_GENAI_MESSAGE_STRATEGY",
    # Telemetry options
    "GenAITelemetryOptions",
    "get_telemetry_options",
    "set_telemetry_options",
    # Attributes
    "LLMRequestAttributes",
    "LLMResponseAttributes", 
    "EmbeddingRequestAttributes",
    "AgentRequestAttributes",
    "ToolRequestAttributes", 
    # Enums
    "GenAiSpanKind",
    # Attribute constants
    "CommonAttributes",
]
