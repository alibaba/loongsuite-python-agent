"""Pytest configuration for Claude Agent SDK instrumentation tests."""

import yaml
from pathlib import Path
from typing import Any, Dict, List
import pytest


def load_cassette(filename: str) -> Dict[str, Any]:
    """从 cassettes 目录加载测试用例。
    
    Args:
        filename: cassette 文件名
        
    Returns:
        测试用例数据字典
    """
    cassette_path = Path(__file__).parent / "cassettes" / filename
    
    with open(cassette_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_all_cassettes() -> List[str]:
    """获取所有 cassette 文件名。
    
    Returns:
        cassette 文件名列表
    """
    cassettes_dir = Path(__file__).parent / "cassettes"
    return sorted([f.name for f in cassettes_dir.glob("*.yaml")])


# Pytest fixture for cassettes
@pytest.fixture
def cassette(request):
    """加载指定的 cassette 文件。"""
    filename = request.param
    return load_cassette(filename)
