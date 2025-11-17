# -*- coding: utf-8 -*-
"""版本检测和测试跳过工具"""

import importlib.util

import pytest

from opentelemetry.instrumentation.agentscope.utils import (
    _AGENTSCOPE_VERSION,
    is_agentscope_v1,
)


def get_agentscope_version():
    """获取当前agentscope版本"""
    return _AGENTSCOPE_VERSION


def skip_if_not_v0():
    """如果不是v0版本则跳过测试"""
    return pytest.mark.skipif(
        is_agentscope_v1(),
        reason=f"需要 agentscope < 1.0.0，当前版本: {_AGENTSCOPE_VERSION}",
    )


def skip_if_not_v1():
    """如果不是v1版本则跳过测试"""
    return pytest.mark.skipif(
        not is_agentscope_v1(),
        reason=f"需要 agentscope >= 1.0.0，当前版本: {_AGENTSCOPE_VERSION}",
    )


def skip_if_no_agentscope():
    """如果没有安装agentscope则跳过测试"""
    if importlib.util.find_spec("agentscope") is None:
        return pytest.mark.skipif(True, reason="需要安装 agentscope")
    return pytest.mark.skipif(False, reason="")
