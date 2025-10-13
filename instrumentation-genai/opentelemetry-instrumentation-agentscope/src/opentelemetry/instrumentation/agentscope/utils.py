import os
import json
import datetime
import enum
import inspect
from importlib.metadata import version
from packaging import version as pkg_version
from dataclasses import is_dataclass
from typing import Any, Dict, List, Optional, Union, Tuple, Iterable

from opentelemetry.util.types import AttributeValue

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None

# 获取 AgentScope 版本
try:
    import agentscope
    _AGENTSCOPE_VERSION = version("agentscope")
except Exception:
    _AGENTSCOPE_VERSION = "0.0.0"

# 环境变量配置
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)


def is_agentscope_v1():
    """检查是否为 AgentScope v1.0.0 及以上版本"""
    return pkg_version.parse(_AGENTSCOPE_VERSION) >= pkg_version.parse("1.0.0")

def is_content_enabled() -> bool:
    """检查是否应该捕获消息内容"""
    return (
        os.getenv(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "false").lower() == "true"
    )
