# -*- coding: utf-8 -*-
"""Message format converter for AgentScope instrumentation.

This module provides utilities to convert messages from different AI model providers
into a unified format for OpenTelemetry instrumentation and frontend display.
"""

import json
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from ..shared import AgentScopeGenAiProviderName

import logging
logger = logging.getLogger(__name__)


# ==================== 标准消息部件定义 ====================

@dataclass
class MessagePart:
    """消息部件基类"""
    type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {"type": self.type}
        for key, value in self.__dict__.items():
            if key != "type" and value is not None:
                result[key] = value
        return result


@dataclass
class TextPart(MessagePart):
    """文本内容部件"""
    content: str


@dataclass  
class ToolCallRequestPart(MessagePart):
    """工具调用请求部件"""
    name: str
    id: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


@dataclass
class ToolCallResponsePart(MessagePart):
    """工具调用响应部件"""
    response: Any
    id: Optional[str] = None


class GenericPart(MessagePart):
    """通用部件，支持任意属性"""
    def __init__(self, type: str, **kwargs: Any):
        super().__init__(type)
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str
    parts: List[MessagePart]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "role": self.role,
            "parts": [part.to_dict() for part in self.parts]
        }


# ==================== 消息部件工厂函数 ====================

class PartFactory:
    """消息部件工厂类"""
    
    @staticmethod
    def create_text_part(content: str) -> TextPart:
        """创建文本部件"""
        return TextPart(type="text", content=content)
    
    @staticmethod
    def create_tool_call_part(name: str, id: Optional[str] = None, 
                             arguments: Optional[Dict[str, Any]] = None) -> ToolCallRequestPart:
        """创建工具调用部件"""
        return ToolCallRequestPart(type="tool_call", name=name, id=id, arguments=arguments)
    
    @staticmethod
    def create_tool_response_part(response: Any, id: Optional[str] = None) -> ToolCallResponsePart:
        """创建工具响应部件"""
        return ToolCallResponsePart(type="tool_call_response", response=response, id=id)
    
    @staticmethod
    def create_generic_part(part_type: str, **kwargs: Any) -> GenericPart:
        """创建通用部件"""
        return GenericPart(part_type, **kwargs)


# ==================== 基础解析器 ====================

class BaseMessageParser(ABC):
    """消息解析器基类"""
    
    @abstractmethod
    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        """解析单个消息"""
        pass
    
    def parse_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """解析消息列表"""
        parsed_messages = []
        for msg in messages:
            try:
                parsed_msg = self.parse_message(msg)
                if parsed_msg.parts:  # 只添加有内容的消息
                    parsed_messages.append(parsed_msg.to_dict())
            except Exception as e:
                logger.warning(f"Failed to parse message: {e}, message: {msg}")
                continue
        return parsed_messages
    
    def _get_role(self, msg: Dict[str, Any]) -> str:
        """获取角色，支持角色映射"""
        role = msg.get('role', 'user')
        # Gemini特殊处理
        if role == 'model':
            return 'assistant'
        return role
    
    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        """安全的JSON解析"""
        if isinstance(text, dict):
            return text
        try:
            return json.loads(text) if isinstance(text, str) else {}
        except json.JSONDecodeError:
            return {}


# ==================== 提供商特定解析器 ====================

class OpenAIMessageParser(BaseMessageParser):
    """OpenAI格式消息解析器"""
    
    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []
        
        # 处理tool消息
        if role == 'tool':
            part = PartFactory.create_tool_response_part(
                response=msg.get('content', ''),
                id=msg.get('tool_call_id', '')
            )
            parts.append(part)
        else:
            # 处理tool_calls
            if 'tool_calls' in msg and msg['tool_calls']:
                for tool_call in msg['tool_calls']:
                    function_info = tool_call.get('function', {})
                    arguments = self._safe_json_loads(function_info.get('arguments', '{}'))
                    
                    part = PartFactory.create_tool_call_part(
                        name=function_info.get('name', 'unknown_tool'),
                        id=tool_call.get('id', 'unknown_id'),
                        arguments=arguments
                    )
                    parts.append(part)
            
            # 处理content
            content = msg.get('content')
            if isinstance(content, str):
                if content.strip():
                    parts.append(PartFactory.create_text_part(content))
            elif isinstance(content, list):
                parts.extend(self._parse_content_list(content))
        
        # 如果没有任何部件且没有tool_calls，添加空文本部件
        if not parts and not msg.get('tool_calls'):
            parts.append(PartFactory.create_text_part(''))
        
        return ChatMessage(role=role, parts=parts)
    
    def _parse_content_list(self, content_list: List[Any]) -> List[MessagePart]:
        """解析content列表"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get('type', 'text')
                if content_type == 'text':
                    text_content = item.get('text', '')
                    if text_content:
                        parts.append(PartFactory.create_text_part(text_content))
                elif content_type == 'image_url':
                    image_url = item.get('image_url', {}).get('url', '')
                    if image_url:
                        parts.append(PartFactory.create_generic_part('image', content=image_url))
                elif content_type == 'input_audio':
                    audio_data = item.get('input_audio', {}).get('data', '')
                    if audio_data:
                        parts.append(PartFactory.create_generic_part('audio', content=audio_data))
                else:
                    parts.append(PartFactory.create_generic_part(
                        content_type, content=str(item), hint='unsupported content type'
                    ))
            elif isinstance(item, str):
                parts.append(PartFactory.create_text_part(item))
        return parts


class AnthropicMessageParser(BaseMessageParser):
    """Anthropic格式消息解析器"""
    
    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []
        
        content = msg.get('content', [])
        if isinstance(content, str):
            parts.append(PartFactory.create_text_part(content))
        elif isinstance(content, list):
            parts.extend(self._parse_content_list(content))
        
        if not parts:
            parts.append(PartFactory.create_text_part(''))
        
        return ChatMessage(role=role, parts=parts)
    
    def _parse_content_list(self, content_list: List[Any]) -> List[MessagePart]:
        """解析Anthropic content列表"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get('type', 'text')
                if content_type == 'text':
                    text_content = item.get('text')
                    if not text_content:
                        continue
                    parts.append(PartFactory.create_text_part(text_content))
                elif content_type == 'tool_use':
                    parts.append(PartFactory.create_tool_call_part(
                        name=item.get('name', ''),
                        id=item.get('id', ''),
                        arguments=item.get('input', {})
                    ))
                elif content_type == 'tool_result':
                    result_text = self._extract_tool_result(item.get('content', []))
                    parts.append(PartFactory.create_tool_response_part(
                        response=result_text,
                        id=item.get('tool_use_id', '')
                    ))
                elif content_type == 'image':
                    source = item.get('source', {})
                    if source.get('type') == 'url':
                        parts.append(PartFactory.create_generic_part('image', content=source.get('url', '')))
                    elif source.get('type') == 'base64':
                        content_url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                        parts.append(PartFactory.create_generic_part('image', content=content_url))
                else:
                    parts.append(PartFactory.create_generic_part(
                        content_type, content=str(item), hint='unsupported content type'
                    ))
            elif isinstance(item, str):
                parts.append(PartFactory.create_text_part(item))
        return parts
    
    def _extract_tool_result(self, content_blocks: List[Any]) -> str:
        """提取工具调用结果"""
        if isinstance(content_blocks, list):
            result_text = ""
            for block in content_blocks:
                if isinstance(block, dict) and block.get('type') == 'text':
                    result_text += block.get('text', '')
            return result_text
        return str(content_blocks)


class GeminiMessageParser(BaseMessageParser):
    """Gemini格式消息解析器"""
    
    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []
        
        parts_list = msg.get('parts', [])
        if isinstance(parts_list, list):
            parts.extend(self._parse_parts_list(parts_list))
        
        if not parts:
            parts.append(PartFactory.create_text_part(''))
        
        return ChatMessage(role=role, parts=parts)
    
    def _parse_parts_list(self, parts_list: List[Any]) -> List[MessagePart]:
        """解析Gemini parts列表"""
        parts = []
        for item in parts_list:
            if isinstance(item, dict):
                if 'text' in item:
                    if not item['text']:
                        continue
                    parts.append(PartFactory.create_text_part(item['text']))
                elif 'function_call' in item:
                    func_call = item['function_call']
                    parts.append(PartFactory.create_tool_call_part(
                        name=func_call.get('name', ''),
                        id=func_call.get('id', ''),
                        arguments=func_call.get('args', {})
                    ))
                elif 'function_response' in item:
                    func_resp = item['function_response']
                    response_content = func_resp.get('response', {})
                    result_text = response_content.get('output', '') if isinstance(response_content, dict) else str(response_content)
                    parts.append(PartFactory.create_tool_response_part(
                        response=result_text,
                        id=func_resp.get('id', '')
                    ))
                elif 'inline_data' in item:
                    parts.append(self._parse_inline_data(item['inline_data']))
                else:
                    parts.append(PartFactory.create_generic_part(
                        'unknown', content=str(item), hint='unsupported part type'
                    ))
            elif isinstance(item, str):
                parts.append(PartFactory.create_text_part(item))
        return parts
    
    def _parse_inline_data(self, inline_data: Dict[str, Any]) -> MessagePart:
        """解析内联数据"""
        mime_type = inline_data.get('mime_type', '')
        data = inline_data.get('data', '')
        
        if mime_type.startswith('image/'):
            return PartFactory.create_generic_part('image', content=f"data:{mime_type};base64,{data}")
        elif mime_type.startswith('audio/'):
            return PartFactory.create_generic_part('audio', content=data)
        else:
            return PartFactory.create_generic_part('unknown', content=str(inline_data), hint='unsupported inline_data type')


class DashScopeMessageParser(BaseMessageParser):
    """DashScope格式消息解析器"""
    
    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []
        
        # 处理tool消息
        if role == 'tool':
            parts.append(PartFactory.create_tool_response_part(
                response=msg.get('content', ''),
                id=msg.get('tool_call_id', '')
            ))
        else:
            # 处理tool_calls
            if 'tool_calls' in msg and msg['tool_calls']:
                for tool_call in msg['tool_calls']:
                    function_info = tool_call.get('function', {})
                    arguments = self._safe_json_loads(function_info.get('arguments', '{}'))
                    
                    parts.append(PartFactory.create_tool_call_part(
                        name=function_info.get('name', ''),
                        id=tool_call.get('id', ''),
                        arguments=arguments
                    ))
            
            # 处理content
            content = msg.get('content')
            if isinstance(content, str) and content.strip():
                parts.append(PartFactory.create_text_part(content))
            elif isinstance(content, list):
                parts.extend(self._parse_content_list(content))
        
        if not parts and not msg.get('tool_calls'):
            parts.append(PartFactory.create_text_part(''))
        
        return ChatMessage(role=role, parts=parts)
    
    def _parse_content_list(self, content_list: List[Any]) -> List[MessagePart]:
        """解析DashScope content列表"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                if 'text' in item:
                    if not item['text']:
                        continue
                    parts.append(PartFactory.create_text_part(item['text']))
                elif 'image' in item:
                    parts.append(PartFactory.create_generic_part('image', content=item['image']))
                elif 'audio' in item:
                    parts.append(PartFactory.create_generic_part('audio', content=item['audio']))
                else:
                    parts.append(PartFactory.create_generic_part(
                        'unknown', content=str(item), hint='unsupported content type'
                    ))
            elif isinstance(item, str):
                parts.append(PartFactory.create_text_part(item))
        return parts


class OllamaMessageParser(BaseMessageParser):
    """Ollama格式消息解析器"""
    
    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []
        
        # 处理tool消息
        if role == 'tool':
            parts.append(PartFactory.create_tool_response_part(
                response=msg.get('content', ''),
                id=msg.get('tool_call_id', '')
            ))
        else:
            # 处理tool_calls
            if 'tool_calls' in msg and msg['tool_calls']:
                for tool_call in msg['tool_calls']:
                    function_info = tool_call.get('function', {})
                    parts.append(PartFactory.create_tool_call_part(
                        name=function_info.get('name', ''),
                        id=tool_call.get('id', ''),
                        arguments=function_info.get('arguments', {})
                    ))
            
            # 处理文本内容
            content = msg.get('content')
            if isinstance(content, str) and content.strip():
                parts.append(PartFactory.create_text_part(content))
            elif content is None and not msg.get('tool_calls'):
                parts.append(PartFactory.create_text_part(''))
            
            # 处理图片（Ollama特有的images字段）
            if 'images' in msg and msg['images']:
                for image_data in msg['images']:
                    if isinstance(image_data, str):
                        parts.append(PartFactory.create_generic_part(
                            'image', content=f"data:image/png;base64,{image_data}"
                        ))
                    else:
                        parts.append(PartFactory.create_generic_part('image', content=str(image_data)))
        
        if not parts and not msg.get('tool_calls'):
            parts.append(PartFactory.create_text_part(''))
        
        return ChatMessage(role=role, parts=parts)


class DeepSeekMessageParser(BaseMessageParser):
    """DeepSeek格式消息解析器"""
    
    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []
        
        # 处理tool消息
        if role == 'tool':
            parts.append(PartFactory.create_tool_response_part(
                response=msg.get('content', ''),
                id=msg.get('tool_call_id', '')
            ))
        else:
            # 处理tool_calls
            if 'tool_calls' in msg and msg['tool_calls']:
                for tool_call in msg['tool_calls']:
                    function_info = tool_call.get('function', {})
                    arguments = self._safe_json_loads(function_info.get('arguments', '{}'))
                    
                    parts.append(PartFactory.create_tool_call_part(
                        name=function_info.get('name', ''),
                        id=tool_call.get('id', ''),
                        arguments=arguments
                    ))
            
            # 处理content
            content = msg.get('content')
            if isinstance(content, str) and content and content.strip():
                parts.append(PartFactory.create_text_part(content))
            elif content is None and msg.get('tool_calls'):
                # DeepSeek格式中，当有tool_calls时content可能为None
                pass  # 不需要添加空文本
            elif isinstance(content, list):
                parts.extend(self._parse_content_list(content))
        
        if not parts and not msg.get('tool_calls'):
            parts.append(PartFactory.create_text_part(''))
        
        return ChatMessage(role=role, parts=parts)
    
    def _parse_content_list(self, content_list: List[Any]) -> List[MessagePart]:
        """解析DeepSeek content列表"""
        parts = []
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get('type', 'text')
                if content_type == 'text':
                    text_content = item.get('text')
                    if not text_content:
                        continue
                    parts.append(PartFactory.create_text_part(text_content))
                else:
                    parts.append(PartFactory.create_generic_part(
                        content_type, content=str(item), hint='unsupported content type'
                    ))
            elif isinstance(item, str):
                parts.append(PartFactory.create_text_part(item))
        return parts


class DefaultMessageParser(BaseMessageParser):
    """默认消息解析器，用于未知格式"""
    
    def parse_message(self, msg: Dict[str, Any]) -> ChatMessage:
        role = self._get_role(msg)
        parts = []
        
        # 尝试从content字段提取文本
        content = msg.get('content', '')
        if isinstance(content, str) and content.strip():
            parts.append(PartFactory.create_text_part(content))
        elif isinstance(content, list):
            # 尝试解析列表格式的content
            for item in content:
                if isinstance(item, dict):
                    if 'text' in item:
                        if not item['text']:
                            continue
                        parts.append(PartFactory.create_text_part(str(item['text'])))
                    else:
                        parts.append(PartFactory.create_generic_part(
                            'unknown', content=str(item), hint='unknown format'
                        ))
                else:
                    parts.append(PartFactory.create_text_part(str(item)))
        else:
            # 将整个消息转换为文本
            parts.append(PartFactory.create_text_part(str(msg)))
        
        if not parts:
            parts.append(PartFactory.create_text_part(''))
        
        return ChatMessage(role=role, parts=parts)


# ==================== 消息转换器工厂 ====================

def get_message_converter(
    provide_name: Optional[str],
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    根据模型类型返回相应的消息格式转换函数。
    
    Args:
        provide_name: AI模型提供商
    
    Returns:
        对应的消息转换函数，函数签名为:
        (messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]
        
        转换后的统一格式符合JSON Schema标准:
        [
            {
                "role": "user|assistant|tool|system",
                "parts": [
                    {
                        "type": "text",
                        "content": "..."
                    },
                    {
                        "type": "tool_call", 
                        "name": "...",
                        "id": "...",
                        "arguments": {...}
                    },
                    {
                        "type": "tool_call_response",
                        "response": "...",
                        "id": "..."
                    }
                ]
            }
        ]
    """
    # 选择对应的解析器
    if provide_name == GenAIAttributes.GenAiProviderNameValues.OPENAI.value:
        parser = OpenAIMessageParser()
    elif provide_name == GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value:
        parser = AnthropicMessageParser()
    elif provide_name == GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value:
        parser = GeminiMessageParser()
    elif provide_name == GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value:
        parser = DeepSeekMessageParser()
    elif provide_name == AgentScopeGenAiProviderName.DASHSCOPE.value:
        parser = DashScopeMessageParser()
    elif provide_name == AgentScopeGenAiProviderName.OLLAMA.value:
        parser = OllamaMessageParser()
    else:
        parser = DefaultMessageParser()
    
    return parser.parse_messages