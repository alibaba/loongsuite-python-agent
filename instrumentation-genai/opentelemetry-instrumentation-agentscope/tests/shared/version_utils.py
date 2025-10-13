# -*- coding: utf-8 -*-
"""版本检测和测试跳过工具"""

import pytest
import sys
import os

# 添加src目录到Python路径
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from opentelemetry.instrumentation.agentscope.utils import is_agentscope_v1, _AGENTSCOPE_VERSION


def get_agentscope_version():
    """获取当前agentscope版本"""
    return _AGENTSCOPE_VERSION


def skip_if_not_v0():
    """如果不是v0版本则跳过测试"""
    return pytest.mark.skipif(
        is_agentscope_v1(),
        reason=f"需要 agentscope < 1.0.0，当前版本: {_AGENTSCOPE_VERSION}"
    )


def skip_if_not_v1():
    """如果不是v1版本则跳过测试"""
    return pytest.mark.skipif(
        not is_agentscope_v1(),
        reason=f"需要 agentscope >= 1.0.0，当前版本: {_AGENTSCOPE_VERSION}"
    )


def skip_if_no_agentscope():
    """如果没有安装agentscope则跳过测试"""
    try:
        import agentscope
        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skipif(True, reason="需要安装 agentscope")


# pytest markers
pytestmark = [
    pytest.mark.agentscope,  # 标记所有agentscope相关测试
]
