# -*- coding: utf-8 -*-
"""
GenAI Telemetry Options for AgentScope Instrumentation
"""

import os
from dataclasses import dataclass
from typing import Optional

from .constants import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MAX_LENGTH,
)


@dataclass
class GenAITelemetryOptions:
    """遥测配置选项
    
    用于控制 GenAI 插装的行为，包括内容捕获、消息策略等。
    支持通过环境变量进行配置。
    """
    capture_message_content: Optional[bool] = None
    capture_message_content_max_length: Optional[int] = None
    
    def __post_init__(self):
        """初始化后处理，设置默认值"""
        if self.capture_message_content is None:
            self.capture_message_content = (
                os.getenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false").lower() == "true"
            )
        
        if self.capture_message_content_max_length is None:
            self.capture_message_content_max_length = int(
                os.getenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT_MAX_LENGTH, "1048576")
            )
        
    def should_capture_content(self) -> bool:
        """是否应该捕获内容"""
        return self.capture_message_content
    
    def truncate_content(self, content: str) -> str:
        """截断内容到指定长度
        
        Args:
            content: 要截断的内容
            
        Returns:
            截断后的内容，如果超过最大长度会添加截断标记
        """
        if not content:
            return content
        if len(content) > self.capture_message_content_max_length:
            return content[:self.capture_message_content_max_length] + "...[truncated]"
        return content
    
    @classmethod
    def from_env(cls) -> "GenAITelemetryOptions":
        """从环境变量创建配置实例
        
        Returns:
            配置实例
        """
        return cls()


# 全局默认配置实例
_global_options: Optional[GenAITelemetryOptions] = None


def get_telemetry_options() -> GenAITelemetryOptions:
    """获取全局遥测配置
    
    Returns:
        全局配置实例
    """
    global _global_options
    if _global_options is None:
        _global_options = GenAITelemetryOptions.from_env()
    return _global_options


def set_telemetry_options(options: GenAITelemetryOptions) -> None:
    """设置全局遥测配置
    
    Args:
        options: 新的配置实例
    """
    global _global_options
    _global_options = options