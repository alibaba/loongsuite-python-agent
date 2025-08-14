import pytest
import json
from unittest.mock import Mock
from opentelemetry.instrumentation.langchain.internal._tracer import _metadata

# 定义常量以替代aliyun.semconv.trace中的引用
LLM_SESSION_ID = "gen_ai.session.id"
LLM_USER_ID = "gen_ai.user.id"
METADATA = "metadata"


class TestMetadata:
    """测试_metadata函数的功能"""
    
    @pytest.mark.parametrize("test_case", [
        # 测试基本的metadata - 包含session_id和user_id
        {
            "name": "basic_metadata_with_session_and_user",
            "run_extra": {
                "metadata": {
                    "session_id": "session_123",
                    "user_id": "user_456",
                    "env": "production"
                }
            },
            "expected": [
                (LLM_SESSION_ID, "session_123"),
                (LLM_USER_ID, "user_456"),
                (METADATA, json.dumps({
                    "session_id": "session_123",
                    "user_id": "user_456",
                    "env": "production"
                }))
            ]
        },
        # 测试使用conversation_id作为session_id
        {
            "name": "metadata_with_conversation_id",
            "run_extra": {
                "metadata": {
                    "conversation_id": "conv_789",
                    "user_id": "user_abc",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            },
            "expected": [
                (LLM_SESSION_ID, "conv_789"),
                (LLM_USER_ID, "user_abc"),
                (METADATA, json.dumps({
                    "conversation_id": "conv_789",
                    "user_id": "user_abc",
                    "timestamp": "2024-01-01T00:00:00Z"
                }))
            ]
        },
        # 测试使用thread_id作为session_id
        {
            "name": "metadata_with_thread_id",
            "run_extra": {
                "metadata": {
                    "thread_id": "thread_xyz",
                    "user_id": "user_def",
                    "version": "1.0.0"
                }
            },
            "expected": [
                (LLM_SESSION_ID, "thread_xyz"),
                (LLM_USER_ID, "user_def"),
                (METADATA, json.dumps({
                    "thread_id": "thread_xyz",
                    "user_id": "user_def",
                    "version": "1.0.0"
                }))
            ]
        },
        # 测试只有session_id，没有user_id
        {
            "name": "metadata_session_id_only",
            "run_extra": {
                "metadata": {
                    "session_id": "session_only",
                    "env": "development"
                }
            },
            "expected": [
                (LLM_SESSION_ID, "session_only"),
                (METADATA, json.dumps({
                    "session_id": "session_only",
                    "env": "development"
                }))
            ]
        },
        # 测试只有user_id，没有session_id
        {
            "name": "metadata_user_id_only",
            "run_extra": {
                "metadata": {
                    "user_id": "user_only",
                    "env": "test"
                }
            },
            "expected": [
                (LLM_USER_ID, "user_only"),
                (METADATA, json.dumps({
                    "user_id": "user_only",
                    "env": "test"
                }))
            ]
        },
        # 测试session_id优先级 - session_id > conversation_id > thread_id
        {
            "name": "session_id_priority",
            "run_extra": {
                "metadata": {
                    "session_id": "session_priority",
                    "conversation_id": "conv_ignored",
                    "thread_id": "thread_ignored",
                    "user_id": "user_priority"
                }
            },
            "expected": [
                (LLM_SESSION_ID, "session_priority"),
                (LLM_USER_ID, "user_priority"),
                (METADATA, json.dumps({
                    "session_id": "session_priority",
                    "conversation_id": "conv_ignored",
                    "thread_id": "thread_ignored",
                    "user_id": "user_priority"
                }))
            ]
        },
        # 测试conversation_id优先级 - conversation_id > thread_id
        {
            "name": "conversation_id_priority",
            "run_extra": {
                "metadata": {
                    "conversation_id": "conv_priority",
                    "thread_id": "thread_ignored",
                    "user_id": "user_conv"
                }
            },
            "expected": [
                (LLM_SESSION_ID, "conv_priority"),
                (LLM_USER_ID, "user_conv"),
                (METADATA, json.dumps({
                    "conversation_id": "conv_priority",
                    "thread_id": "thread_ignored",
                    "user_id": "user_conv"
                }))
            ]
        },
        # 测试包含特殊字符的metadata
        {
            "name": "metadata_with_special_characters",
            "run_extra": {
                "metadata": {
                    "session_id": "session-123_456",
                    "user_id": "user.abc@def",
                    "env": "test-env_1.0"
                }
            },
            "expected": [
                (LLM_SESSION_ID, "session-123_456"),
                (LLM_USER_ID, "user.abc@def"),
                (METADATA, json.dumps({
                    "session_id": "session-123_456",
                    "user_id": "user.abc@def",
                    "env": "test-env_1.0"
                }))
            ]
        },
        # 测试包含中文的metadata
        {
            "name": "metadata_with_chinese",
            "run_extra": {
                "metadata": {
                    "session_id": "会话_123",
                    "user_id": "用户_456",
                    "env": "生产环境"
                }
            },
            "expected": [
                (LLM_SESSION_ID, "会话_123"),
                (LLM_USER_ID, "用户_456"),
                (METADATA, json.dumps({
                    "session_id": "会话_123",
                    "user_id": "用户_456",
                    "env": "生产环境"
                }))
            ]
        },
        # 测试空metadata
        {
            "name": "empty_metadata",
            "run_extra": {
                "metadata": {}
            },
            "expected": []
        },
        # 测试没有metadata键
        {
            "name": "no_metadata_key",
            "run_extra": {
                "other_key": "other_value"
            },
            "expected": []
        },
        # 测试None的run_extra
        {
            "name": "none_run_extra",
            "run_extra": None,
            "expected": []
        },
        # 测试空run_extra
        {
            "name": "empty_run_extra",
            "run_extra": {},
            "expected": []
        },
        # 测试包含无效类型的metadata
        {
            "name": "invalid_metadata_type",
            "run_extra": {
                "metadata": "not a dict"
            },
            "expected": []
        },
        # 测试包含Unicode字符的metadata
        {
            "name": "unicode_metadata",
            "run_extra": {
                "metadata": {
                    "session_id": "session_unicode_测试",
                    "user_id": "user_unicode_测试",
                    "message": "Hello 世界 🌍",
                    "emoji": "🚀✨🎉"
                }
            },
            "expected": [
                (LLM_SESSION_ID, "session_unicode_测试"),
                (LLM_USER_ID, "user_unicode_测试"),
                (METADATA, json.dumps({
                    "session_id": "session_unicode_测试",
                    "user_id": "user_unicode_测试",
                    "message": "Hello 世界 🌍",
                    "emoji": "🚀✨🎉"
                }))
            ]
        }
    ])
    def test_metadata_parsing(self, test_case):
        """测试metadata解析功能"""
        # 创建模拟的Run对象
        mock_run = Mock()
        mock_run.extra = test_case["run_extra"]
        
        result = list(_metadata(mock_run))
        
        # 验证结果数量
        assert len(result) == len(test_case["expected"]), \
            f"Expected {len(test_case['expected'])} results, got {len(result)} for {test_case['name']}"
        
        # 验证每个结果
        for i, (actual_key, actual_value) in enumerate(result):
            expected_key, expected_value = test_case["expected"][i]
            assert actual_key == expected_key, \
                f"Expected key {expected_key}, got {actual_key} for {test_case['name']}"
            assert actual_value == expected_value, \
                f"Expected value {expected_value}, got {actual_value} for {test_case['name']}"