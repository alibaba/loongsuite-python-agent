# -*- coding: utf-8 -*-
"""
GenAI Attributes Classes for AgentScope Instrumentation

This module provides dataclasses for handling GenAI semantic convention attributes
for different types of operations (LLM, Embedding, Agent, Tool).
"""

from abc import ABC
from dataclasses import dataclass, fields
from typing import Dict, List, Optional
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.types import AttributeValue
from .constants import (
    CommonAttributes,
    GenAiSpanKind,
)
from agentscope import _config


class ASBaseAttributes(ABC):
    """属性基类"""

    SPAN_KIND: Optional[str] = None

    def get_span_attributes(self) -> Dict[str, AttributeValue]:
        """获取用于 span 的属性"""
        attributes = {}

        if self.SPAN_KIND:
            attributes[CommonAttributes.GEN_AI_SPAN_KIND] = self.SPAN_KIND
            attributes[GenAIAttributes.GEN_AI_CONVERSATION_ID] = _config.run_id

        # 遍历所有字段并设置属性
        for field_info in fields(self):
            field_name = field_info.name
            field_value = getattr(self, field_name)

            # 如果字段有值且在映射表中
            if field_value is not None and field_name in ATTRIBUTE_MAPPING:
                attribute_key = ATTRIBUTE_MAPPING[field_name]
                attributes[attribute_key] = field_value

        return attributes

@dataclass
class LLMRequestAttributes(ASBaseAttributes):
    """LLM 请求属性类
    
    用于存储和管理 LLM 请求相关的 GenAI 语义约定属性。
    """
    SPAN_KIND = GenAiSpanKind.LLM.value

    provider_name: Optional[str] = None
    operation_name: Optional[str] = None
    request_model: Optional[str] = None
    request_choice_count: Optional[int] = None
    request_seed: Optional[int] = None
    request_frequency_penalty: Optional[float] = None
    request_presence_penalty: Optional[float] = None
    request_max_tokens: Optional[int] = None
    request_top_p: Optional[float] = None
    request_top_k: Optional[float] = None
    request_temperature: Optional[float] = None
    request_stop_sequences: Optional[List[str]] = None
    system_instructions: Optional[AttributeValue] = None
    request_tool_definitions: Optional[AttributeValue] = None
    input_messages: Optional[AttributeValue] = None


@dataclass
class LLMResponseAttributes(ASBaseAttributes):
    """LLM 响应属性类
    
    用于存储和管理 LLM 响应相关的 GenAI 语义约定属性。
    """
    response_id: Optional[str] = None
    response_model: Optional[str] = None
    output_type: Optional[str] = None
    response_finish_reasons: Optional[List[str]] = None
    usage_input_tokens: Optional[int] = None
    usage_output_tokens: Optional[int] = None


@dataclass
class EmbeddingRequestAttributes(ASBaseAttributes):
    """Embedding 请求属性类
    
    用于存储和管理 Embedding 请求相关的 GenAI 语义约定属性。
    """
    SPAN_KIND = GenAiSpanKind.EMBEDDING.value

    provider_name: Optional[str] = None
    operation_name: Optional[str] = None
    request_model: Optional[str] = None
    request_encoding_formats: Optional[List[str]] = None
    input_messages: Optional[AttributeValue] = None


@dataclass
class AgentRequestAttributes(ASBaseAttributes):
    """Agent 请求属性类
    
    用于存储和管理 Agent 请求相关的 GenAI 语义约定属性。
    """
    SPAN_KIND = GenAiSpanKind.AGENT.value

    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    agent_description: Optional[str] = None
    conversation_id: Optional[str] = None
    operation_name: Optional[str] = None
    system_instructions: Optional[str] = None
    request_model: Optional[str] = None
    input_messages: Optional[AttributeValue] = None


@dataclass
class ToolRequestAttributes(ASBaseAttributes):
    """Tool 请求属性类
    
    用于存储和管理 Tool 请求相关的 GenAI 语义约定属性。
    """
    SPAN_KIND = GenAiSpanKind.TOOL.value

    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_description: Optional[str] = None
    tool_call_arguments: Optional[str] = None
    tool_type: Optional[str] = None
    operation_name: Optional[str] = None


ATTRIBUTE_MAPPING = {
    "provider_name": GenAIAttributes.GEN_AI_PROVIDER_NAME,
    "operation_name": GenAIAttributes.GEN_AI_OPERATION_NAME,
    "request_model": GenAIAttributes.GEN_AI_REQUEST_MODEL,
    "request_choice_count": GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT,
    "request_seed": GenAIAttributes.GEN_AI_REQUEST_SEED,
    "request_frequency_penalty": GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY,
    "request_presence_penalty": GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY,
    "request_max_tokens": GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS,
    "request_top_p": GenAIAttributes.GEN_AI_REQUEST_TOP_P,
    "request_top_k": GenAIAttributes.GEN_AI_REQUEST_TOP_K,
    "request_temperature": GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE,
    "request_stop_sequences": GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES,
    "system_instructions": GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
    "input_messages": GenAIAttributes.GEN_AI_INPUT_MESSAGES,
    "request_tool_definitions": CommonAttributes.GEN_AI_REQUEST_TOOL_DEFINITIONS, # agentscope semconv

    "response_id": GenAIAttributes.GEN_AI_RESPONSE_ID,
    "response_model": GenAIAttributes.GEN_AI_RESPONSE_MODEL,
    "output_type": GenAIAttributes.GEN_AI_OUTPUT_TYPE,
    "response_finish_reasons": GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS,
    "usage_input_tokens": GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS,
    "usage_output_tokens": GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS,

    "request_encoding_formats": GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS,

    "agent_id": GenAIAttributes.GEN_AI_AGENT_ID,
    "agent_name": GenAIAttributes.GEN_AI_AGENT_NAME,
    "agent_description": GenAIAttributes.GEN_AI_AGENT_DESCRIPTION,
    "conversation_id": GenAIAttributes.GEN_AI_CONVERSATION_ID,

    "tool_call_id": GenAIAttributes.GEN_AI_TOOL_CALL_ID,
    "tool_name": GenAIAttributes.GEN_AI_TOOL_NAME,
    "tool_description": GenAIAttributes.GEN_AI_TOOL_DESCRIPTION,
    "tool_type": GenAIAttributes.GEN_AI_TOOL_TYPE,
    "tool_call_arguments": CommonAttributes.GEN_AI_TOOL_CALL_ARGUMENTS, # agentscope semconv
}
