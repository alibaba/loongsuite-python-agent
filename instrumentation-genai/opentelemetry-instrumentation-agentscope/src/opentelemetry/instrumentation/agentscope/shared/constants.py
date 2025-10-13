# -*- coding: utf-8 -*-
"""
Constants for OpenTelemetry GenAI Semantic Conventions
"""

from enum import Enum

# Environment Variables
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MAX_LENGTH = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MAX_LENGTH"
OTEL_INSTRUMENTATION_GENAI_MESSAGE_STRATEGY = "OTEL_INSTRUMENTATION_GENAI_MESSAGE_STRATEGY"


class CommonAttributes:
    """Common GenAI attributes shared across all span types"""
    GEN_AI_SPAN_KIND = "gen_ai.span.kind"
    GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
    GEN_AI_REQUEST_TOOL_DEFINITIONS = "gen_ai.tool.definitions"

    # extended attributes
    GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"


class GenAiSpanKind(str, Enum):
    """GenAI span kinds"""
    LLM = "llm"
    EMBEDDING = "embedding"
    AGENT = "agent"
    TOOL = "tool"
    FORMATTER = "formatter"

class AgentScopeGenAiProviderName(str, Enum):
    OLLAMA = "ollama"
    DASHSCOPE = "dashscope"
