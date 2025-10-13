# -*- coding: utf-8 -*-
"""Request attributes extractor for AgentScope instrumentation.

This module provides utilities to extract and normalize message formats
from different AI model providers into a unified format for frontend display.
"""

from typing import Any, Dict, List, Optional, Tuple
import inspect

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from agentscope.message import Msg
from agentscope.model import (
    ChatModelBase, 
    OpenAIChatModel,
    GeminiChatModel,
    OllamaChatModel,
    AnthropicChatModel,
    DashScopeChatModel,
)
from agentscope.embedding import EmbeddingModelBase
from agentscope.agent import AgentBase

from ..shared import (
    AgentScopeGenAiProviderName,
    LLMRequestAttributes,
    EmbeddingRequestAttributes,
    AgentRequestAttributes,
)

from .utils import (
    _format_msg_to_parts,
    _serialize_to_str,
)
from opentelemetry.instrumentation.agentscope.v1.message_converter import get_message_converter

import logging
logger = logging.getLogger(__name__)


class RequestAttributesExtractor:
    """请求参数提取器类，支持有状态的环境变量管理和参数解析。
    
    该类负责从不同类型的函数调用参数中提取输入消息，并转换为统一格式。
    支持LLM、Embedding和Agent三种类型的参数解析。
    """
    
    def __init__(self):
        """初始化提取器，获取遥测配置选项。"""
        # self.telemetry_options = get_telemetry_options()
        
    def extract_llm_input_messages(
        self,
        call_instance: ChatModelBase ,
        call_args: Tuple[Any, ...],
        call_kwargs: Dict[str, Any]
    ) -> LLMRequestAttributes:
        """提取LLM调用的输入消息。
        
        Args:
            call_instance: llm 模型调用实例
            call_args: 位置参数
            call_kwargs: 关键字参数
            
        Returns:
            转换后的统一格式消息列表
        """
        # 创建LLM请求属性
        request_attrs = LLMRequestAttributes(
            operation_name = GenAIAttributes.GenAiOperationNameValues.CHAT.value, 
            provider_name = self._parse_provider_name(call_instance),
            request_model = getattr(call_instance, "model_name", "unknown_model"),
            request_max_tokens = call_kwargs.get("max_tokens"),
            request_temperature = call_kwargs.get("temperature"),
            request_top_p = call_kwargs.get("top_p"),
            request_top_k = call_kwargs.get("top_k"),
            request_stop_sequences = call_kwargs.get("stop_sequences"),
            request_tool_definitions = self._get_tool_definitions(tools=call_kwargs.get("tools"), tool_choice=call_kwargs.get("tool_choice"), structured_model=call_kwargs.get("structured_model")),
        )
        
        # input_attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = request_attrs.request_model

        messages = None
        if call_args and len(call_args) > 0:
            messages = call_args[0]
        elif "messages" in call_kwargs:
            messages = call_kwargs["messages"]
        if messages:
            input_messages = get_message_converter(request_attrs.provider_name)(messages)
        else:
            logger.debug(" ChatModelWrapper No messages provided. Skipping input message conversion.")
            input_messages = {
                "args": call_args,
                "kwargs": call_kwargs,
            }

        request_attrs.input_messages = _serialize_to_str(input_messages)
        
        return request_attrs
    
    def extract_embedding_input_messages(
        self,
        call_instance: EmbeddingModelBase,
        call_args: Tuple[Any, ...],
        call_kwargs: Dict[str, Any]
    ) -> EmbeddingRequestAttributes:
        """提取Embedding调用的输入消息。
        
        Args:
            call_args: 位置参数
            call_kwargs: 关键字参数
            
        Returns:
            转换后的统一格式消息列表
        """
        # 创建embedding请求属性
        request_attrs = EmbeddingRequestAttributes(
            operation_name = GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value, 
            request_model = getattr(call_instance, "model_name", "unknown_model"),
            request_encoding_formats = call_kwargs.get("encoding_formats"),
        )

        # input_attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = request_attrs.request_model

        text_for_embedding = None
        if call_args and len(call_args) > 0:
            text_for_embedding = call_args[0]
        elif "text" in call_kwargs:
            text_for_embedding = call_kwargs["text"]
        if text_for_embedding:
            input_messages = self._get_embedding_message(text_for_embedding)
        else:
            logger.debug(" EmbeddingModelWrapper No text provided. Skipping input message conversion.")
            input_messages = {
                "args": call_args,
                "kwargs": call_kwargs,
            }

        request_attrs.input_messages = _serialize_to_str(input_messages)

        return request_attrs
    
    def extract_agent_input_messages(
        self,
        reply_instance: AgentBase,
        reply_args: Tuple[Any, ...],
        reply_kwargs: Dict[str, Any]
    ) -> AgentRequestAttributes:
        """提取Agent reply调用的输入消息。
        
        Args:
            reply_args: 位置参数
            reply_kwargs: 关键字参数
            
        Returns:
            转换后的统一格式消息列表
        """
        # 创建agent reply请求属性
        request_attrs = AgentRequestAttributes(
            operation_name = GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value, 
            agent_id = getattr(reply_instance, "id", "unknown"),
            agent_name = getattr(reply_instance, "name", "unknown_agent"),
            agent_description = inspect.getdoc(reply_instance.__class__) or "No description available",
            system_instructions = reply_instance.sys_prompt if hasattr(reply_instance, "sys_prompt") else None,
        )
        if hasattr(reply_instance, "model") and reply_instance.model:
            request_attrs.request_model = getattr(reply_instance.model, "model_name", "unknown_model")

        # input_attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = request_attrs.request_model
        
        msg = None
        if reply_args and len(reply_args) > 0:
            msg = reply_args[0]
        elif "msg" in reply_kwargs:
            msg = reply_kwargs["msg"]
        if msg:
            input_messages = self._get_agent_message(msg)
        else:
            logger.debug(" AgentWrapper No msg provided. Skipping input message conversion.")
            input_messages = {
                "args": reply_args,
                "kwargs": reply_kwargs,
            }
        request_attrs.input_messages = _serialize_to_str(input_messages)

        return request_attrs

    def _get_tool_definitions(self, tools: Optional[List[Dict[str, Any]]], tool_choice: Optional[str], structured_model: Optional[bool] = False) -> Optional[str]:
        if structured_model is True or tools is None or tool_choice is None or tool_choice == "none":
            return None
        else:
            return _serialize_to_str(tools)

    def _parse_provider_name(self, chat_model: ChatModelBase) -> str:
        if isinstance(chat_model, OpenAIChatModel):
            return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
        elif isinstance(chat_model, GeminiChatModel):
            return GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value
        elif isinstance(chat_model, AnthropicChatModel):
            return GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value
        elif isinstance(chat_model, DashScopeChatModel):
            if hasattr(chat_model, "base_http_api_url") and chat_model.base_http_api_url:
                base_url = chat_model.base_http_api_url
                if "openai.com" in base_url:
                    return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
                elif "api.deepseek.com" in base_url:
                    return GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value
                elif "dashscope.aliyuncs.com" in base_url:
                    return AgentScopeGenAiProviderName.DASHSCOPE.value
            return AgentScopeGenAiProviderName.DASHSCOPE.value
        elif isinstance(chat_model, OllamaChatModel):
            return AgentScopeGenAiProviderName.OLLAMA.value
        else:
            return "unknown"
    
    def _get_embedding_message(self, text: List[str]) -> List[Dict[str, Any]]:
        input_message = []
        for text_item in text:
            input_message.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_item
                    }
                ]
            })
        return input_message

    def _get_agent_message(self, msg: Any) -> List[Dict[str, Any]]:
        try:
            if isinstance(msg, Msg):
                return [_format_msg_to_parts(msg)]
            elif isinstance(msg, list):
                return [_format_msg_to_parts(msg_item) for msg_item in msg]
            else:
                return []
        except Exception as e:
            logger.debug(f"Error formatting messages: {e}")
            return []
