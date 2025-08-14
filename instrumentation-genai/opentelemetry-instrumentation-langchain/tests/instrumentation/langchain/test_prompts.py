import pytest
from opentelemetry.instrumentation.langchain.internal._tracer import _prompts

# 定义常量以替代aliyun.semconv.trace中的引用
class SpanAttributes:
    GEN_AI_PROMPT = "gen_ai.prompt"
    CONTENT = "content"

class MessageAttributes:
    MESSAGE_ROLE = "role"
    MESSAGE_CONTENT = "content"


class TestPrompts:
    """测试_prompts函数的功能"""
    
    @pytest.mark.parametrize("test_case", [
        # 测试dict类型的prompt - 包含role和text
        {
            "name": "dict_prompt_with_role_and_text",
            "inputs": {
                "prompts": [
                    {"role": "user", "text": "Hello, how are you?"},
                    {"role": "assistant", "text": "I'm doing well, thank you!"}
                ]
            },
            "expected": [
                (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_ROLE}", "user"),
                (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_CONTENT}", "Hello, how are you?"),
                (f"{SpanAttributes.GEN_AI_PROMPT}.1.{MessageAttributes.MESSAGE_ROLE}", "assistant"),
                (f"{SpanAttributes.GEN_AI_PROMPT}.1.{MessageAttributes.MESSAGE_CONTENT}", "I'm doing well, thank you!")
            ]
        },
        # 测试dict类型的prompt - 只包含role
        {
            "name": "dict_prompt_with_role_only",
            "inputs": {
                "prompts": [
                    {"role": "system"}
                ]
            },
            "expected": [
                (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_ROLE}", "system")
            ]
        },
        # 测试dict类型的prompt - 只包含text
        {
            "name": "dict_prompt_with_text_only",
            "inputs": {
                "prompts": [
                    {"text": "This is a system message"}
                ]
            },
            "expected": [
                (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_CONTENT}", "This is a system message")
            ]
        },
        # 测试dict类型的prompt - 空dict
        {
            "name": "dict_prompt_empty",
            "inputs": {
                "prompts": [
                    {}
                ]
            },
            "expected": []
        },
        # 测试string类型的prompt
        {
            "name": "string_prompt_single",
            "inputs": {
                "prompts": ["Hello, this is a simple prompt"]
            },
            "expected": [
                (f"{SpanAttributes.GEN_AI_PROMPT}.0.{SpanAttributes.CONTENT}", "Hello, this is a simple prompt")
            ]
        },
        # 测试string类型的prompt - 多个
        {
            "name": "string_prompt_multiple",
            "inputs": {
                "prompts": [
                    "First prompt",
                    "Second prompt",
                    "Third prompt"
                ]
            },
            "expected": [
                (f"{SpanAttributes.GEN_AI_PROMPT}.0.{SpanAttributes.CONTENT}", "First prompt"),
                (f"{SpanAttributes.GEN_AI_PROMPT}.1.{SpanAttributes.CONTENT}", "Second prompt"),
                (f"{SpanAttributes.GEN_AI_PROMPT}.2.{SpanAttributes.CONTENT}", "Third prompt")
            ]
        },
        # 测试混合类型 - dict和string
        {
            "name": "mixed_prompt_types",
            "inputs": {
                "prompts": [
                    {"role": "user", "text": "User message"},
                    "Simple string prompt",
                    {"role": "assistant", "text": "Assistant response"}
                ]
            },
            "expected": [
                (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_ROLE}", "user"),
                (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_CONTENT}", "User message"),
                (f"{SpanAttributes.GEN_AI_PROMPT}.1.{SpanAttributes.CONTENT}", "Simple string prompt"),
                (f"{SpanAttributes.GEN_AI_PROMPT}.2.{MessageAttributes.MESSAGE_ROLE}", "assistant"),
                (f"{SpanAttributes.GEN_AI_PROMPT}.2.{MessageAttributes.MESSAGE_CONTENT}", "Assistant response")
            ]
        },
        # 测试边界情况 - 空prompts列表
        {
            "name": "empty_prompts_list",
            "inputs": {
                "prompts": []
            },
            "expected": []
        },
        # 测试边界情况 - 没有prompts键
        {
            "name": "no_prompts_key",
            "inputs": {
                "messages": ["some message"]
            },
            "expected": []
        },
        # 测试边界情况 - None输入
        {
            "name": "none_input",
            "inputs": None,
            "expected": []
        },
        # 测试边界情况 - 空dict输入
        {
            "name": "empty_dict_input",
            "inputs": {},
            "expected": []
        },
        # 测试边界情况 - 非Mapping类型输入
        {
            "name": "non_mapping_input",
            "inputs": "not a dict",
            "expected": []
        }
    ])
    def test_prompts_parsing(self, test_case):
        """测试prompts解析功能"""
        result = list(_prompts(test_case["inputs"]))
        
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
    
    def test_prompts_with_invalid_prompt_types(self):
        """测试包含无效prompt类型的情况"""
        inputs = {
            "prompts": [
                {"role": "user", "text": "Valid dict prompt"},
                123,  # 无效类型
                "Valid string prompt",
                None,  # 无效类型
                {"role": "assistant", "text": "Another valid dict prompt"}
            ]
        }
        
        result = list(_prompts(inputs))
        
        # 应该只处理有效的prompt类型
        expected = [
            (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_ROLE}", "user"),
            (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_CONTENT}", "Valid dict prompt"),
        ]

        # 验证expected中的key和value是否在result中
        for key, value in expected:
            assert (key, value) in result
    
    def test_prompts_with_complex_dict_structure(self):
        """测试复杂的dict结构"""
        inputs = {
            "prompts": [
                {
                    "role": "user",
                    "text": "Complex user message",
                    "additional_info": "This should be ignored"
                },
                {
                    "role": "system",
                    "text": "System configuration",
                    "metadata": {"version": "1.0"}
                }
            ]
        }
        
        result = list(_prompts(inputs))
        
        # 只应该提取role和text字段
        expected = [
            (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_ROLE}", "user"),
            (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_CONTENT}", "Complex user message"),
            (f"{SpanAttributes.GEN_AI_PROMPT}.1.{MessageAttributes.MESSAGE_ROLE}", "system"),
            (f"{SpanAttributes.GEN_AI_PROMPT}.1.{MessageAttributes.MESSAGE_CONTENT}", "System configuration")
        ]
        
        assert len(result) == len(expected)
        for i, (actual_key, actual_value) in enumerate(result):
            expected_key, expected_value = expected[i]
            assert actual_key == expected_key
            assert actual_value == expected_value
    
    def test_prompts_with_special_characters(self):
        """测试包含特殊字符的prompt"""
        inputs = {
            "prompts": [
                {"role": "user", "text": "Hello\nWorld\tTab"},
                "String with \"quotes\" and 'apostrophes'",
                {"role": "assistant", "text": "Response with unicode: 你好世界"}
            ]
        }
        
        result = list(_prompts(inputs))
        
        expected = [
            (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_ROLE}", "user"),
            (f"{SpanAttributes.GEN_AI_PROMPT}.0.{MessageAttributes.MESSAGE_CONTENT}", "Hello\nWorld\tTab"),
            (f"{SpanAttributes.GEN_AI_PROMPT}.1.{SpanAttributes.CONTENT}", "String with \"quotes\" and 'apostrophes'"),
            (f"{SpanAttributes.GEN_AI_PROMPT}.2.{MessageAttributes.MESSAGE_ROLE}", "assistant"),
            (f"{SpanAttributes.GEN_AI_PROMPT}.2.{MessageAttributes.MESSAGE_CONTENT}", "Response with unicode: 你好世界")
        ]
        
        assert len(result) == len(expected)
        for i, (actual_key, actual_value) in enumerate(result):
            expected_key, expected_value = expected[i]
            assert actual_key == expected_key
            assert actual_value == expected_value