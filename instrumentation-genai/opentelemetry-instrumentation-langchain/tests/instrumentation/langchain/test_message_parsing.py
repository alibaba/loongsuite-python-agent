import pytest
from opentelemetry.instrumentation.langchain.internal._tracer import _parse_message_data, _input_messages, _get_tool_call, _parse_prompt_data
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    FunctionMessage, 
    ToolMessage, 
    ChatMessage
)

# 定义常量以替代aliyun.semconv.trace中的引用
class MessageAttributes:
    MESSAGE_ROLE = "role"
    MESSAGE_CONTENT = "content"
    MESSAGE_NAME = "name"
    MESSAGE_FUNCTION_CALL_NAME = "function_call.name"
    MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = "function_call.arguments"
    MESSAGE_TOOL_CALLS = "tool_calls"

class ToolCallAttributes:
    TOOL_CALL_FUNCTION_NAME = "tool_call.function.name"
    TOOL_CALL_FUNCTION_ARGUMENTS_JSON = "tool_call.function.arguments"


class TestMessageParsing:
    """测试消息解析功能"""
    
    @pytest.mark.parametrize("test_case", [
        # 基础消息类型测试
        {
            "name": "human_message_basic",
            "message": HumanMessage(content="Hello, how are you?"),
            "expected": {
                MessageAttributes.MESSAGE_ROLE: "user",
                MessageAttributes.MESSAGE_CONTENT: "Hello, how are you?"
            }
        },
        {
            "name": "ai_message_basic", 
            "message": AIMessage(content="I'm doing well, thank you!"),
            "expected": {
                MessageAttributes.MESSAGE_ROLE: "assistant",
                MessageAttributes.MESSAGE_CONTENT: "I'm doing well, thank you!"
            }
        },
        {
            "name": "system_message_basic",
            "message": SystemMessage(content="You are a helpful assistant."),
            "expected": {
                MessageAttributes.MESSAGE_ROLE: "system", 
                MessageAttributes.MESSAGE_CONTENT: "You are a helpful assistant."
            }
        },
        {
            "name": "function_message_basic",
            "message": FunctionMessage(content="The weather is sunny", name="get_weather"),
            "expected": {
                MessageAttributes.MESSAGE_ROLE: "function",
                MessageAttributes.MESSAGE_CONTENT: "The weather is sunny",
                MessageAttributes.MESSAGE_NAME: "get_weather"
            }
        },
        {
            "name": "tool_message_basic",
            "message": ToolMessage(content="Tool execution completed", name="calculator", tool_call_id="call_123"),
            "expected": {
                MessageAttributes.MESSAGE_ROLE: "tool",
                MessageAttributes.MESSAGE_CONTENT: "Tool execution completed", 
                MessageAttributes.MESSAGE_NAME: "calculator"
            }
        },
        {
            "name": "chat_message_basic",
            "message": ChatMessage(content="Hello from chat", role="user"),
            "expected": {
                MessageAttributes.MESSAGE_ROLE: "user",
                MessageAttributes.MESSAGE_CONTENT: "Hello from chat"
            }
        },
        
        # 函数调用测试
        {
            "name": "ai_message_with_function_call",
            "message": AIMessage(
                content="I'll call a function to get weather information",
                additional_kwargs={
                    "function_call": {
                        "name": "get_weather",
                        "arguments": '{"location": "Beijing", "unit": "celsius"}'
                    }
                }
            ),
            "expected": {
                MessageAttributes.MESSAGE_ROLE: "assistant",
                MessageAttributes.MESSAGE_CONTENT: "I'll call a function to get weather information",
                MessageAttributes.MESSAGE_FUNCTION_CALL_NAME: "get_weather",
                MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON: '{"location": "Beijing", "unit": "celsius"}'
            }
        },
        # 工具调用测试
        {
            "name": "ai_message_with_tool_calls",
            "message": AIMessage(
                content="I'll use a calculator tool",
                additional_kwargs={
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calculator",
                                "arguments": '{"operation": "add", "a": 1, "b": 2}'
                            }
                        }
                    ]
                }
            ),
            "expected": {
                MessageAttributes.MESSAGE_ROLE: "assistant",
                MessageAttributes.MESSAGE_CONTENT: "I'll use a calculator tool",
                MessageAttributes.MESSAGE_TOOL_CALLS: [
                    {
                        ToolCallAttributes.TOOL_CALL_FUNCTION_NAME: "calculator",
                        ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON: '{"operation": "add", "a": 1, "b": 2}'
                    }
                ]
            }
        },
        # 多模态内容测试
        {
            "name": "human_message_with_multimodal_content",
            "message": HumanMessage(content=[
                "Here's an image:",
                {"type": "image", "url": "http://example.com/image.jpg"}
            ]),
            "expected": {
                MessageAttributes.MESSAGE_ROLE: "user",
                f"{MessageAttributes.MESSAGE_CONTENT}.0": "Here's an image:",
                f"{MessageAttributes.MESSAGE_CONTENT}.1": "{'type': 'image', 'url': 'http://example.com/image.jpg'}"
            }
        },
        # 边界情况测试
        {
            "name": "message_with_non_string_name",
            "message_data": {
                "id": ["langchain", "core", "messages", "HumanMessage"],
                "kwargs": {
                    "content": "Hello",
                    "name": 123  # 非字符串name
                }
            },
            "expected": {
                MessageAttributes.MESSAGE_ROLE: "user",
                MessageAttributes.MESSAGE_CONTENT: "Hello"
            }
        },
        {
            "name": "message_with_non_string_content",
            "message_data": {
                "id": ["langchain", "core", "messages", "HumanMessage"],
                "kwargs": {
                    "content": 123  # 非字符串content
                }
            },
            "expected": {
                MessageAttributes.MESSAGE_ROLE: "user"
            }
        }
    ])
    def test_parse_message_data(self, test_case):
        """测试消息数据解析"""
        # 处理两种输入格式：LangChain消息对象或字典数据
        if "message" in test_case:
            message_data = test_case["message"].to_json()
        else:
            message_data = test_case["message_data"]
            
        result = dict(_parse_message_data(message_data))
        
        for key, expected_value in test_case["expected"].items():
            if key == MessageAttributes.MESSAGE_TOOL_CALLS:
                tool_calls = result.get(key)
                assert tool_calls is not None, f"Expected tool_calls for {test_case['name']}"
                # 检查tool_calls列表中是否包含期望的函数名
                found = False
                for tool_call in tool_calls:
                    if tool_call.get(ToolCallAttributes.TOOL_CALL_FUNCTION_NAME) == "calculator":
                        found = True
                        break
                assert found, f"Expected calculator function in tool_calls for {test_case['name']}"
            else:
                actual_value = result.get(key)
                assert actual_value == expected_value, f"Expected {expected_value}, got {actual_value} for {test_case['name']}"


class TestInputMessagesParsing:
    """测试输入消息解析功能"""
    
    @pytest.mark.parametrize("test_case", [
        # 基础消息列表测试
        {
            "name": "basic_messages_list",
            "input": {
                "messages": [
                    [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
                ]
            },
            "expected_count": 1
        },
        # 单个消息测试
        {
            "name": "single_message",
            "input": {
                "messages": HumanMessage(content="Single message")
            },
            "expected_count": 1
        },
        # 空输入测试
        {
            "name": "empty_messages",
            "input": {"messages": [[]]},
            "expected_count": 0
        },
        {
            "name": "none_input",
            "input": None,
            "expected_count": 0
        },
        {
            "name": "invalid_message_types",
            "input": {"messages": [[123, "invalid", {"invalid": "format"}]], "prompts": None},
            "expected_count": 1
        }
    ])
    def test_input_messages_parsing(self, test_case):
        """测试输入消息解析"""
        result = list(_input_messages(test_case["input"]))
        assert len(result) == test_case["expected_count"], f"Expected {test_case['expected_count']} results for {test_case['name']}"