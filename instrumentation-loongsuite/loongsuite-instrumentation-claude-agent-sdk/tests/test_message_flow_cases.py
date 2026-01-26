"""测试用例：基于真实的 message 流来测试 _process_agent_invocation_stream 函数。

这些测试用例基于 claude-agent-sdk-python 的真实运行示例：
1. foo.sh 命令示例 - 展示了工具调用和错误处理（多轮对话，包含错误场景）
2. echo 命令示例 - 展示了简单的成功路径（单轮对话，成功场景）
3. pretooluse 示例 - 展示了 PreToolUse hook 的使用（工具调用被 hook 阻止）

测试数据来源：/Users/sipercai/.cursor/projects/Users-sipercai-project-pyins/terminals/6.txt:229-246
"""

import json
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import MagicMock

import pytest


# ============================================================================
# 辅助函数：从测试数据创建模拟的 message 对象
# ============================================================================


def create_mock_message_from_data(message_data: Dict[str, Any]) -> Any:
    """从测试数据字典创建模拟的 message 对象。
    
    Args:
        message_data: 包含消息数据的字典
        
    Returns:
        模拟的 message 对象，具有与真实对象相同的属性
    """
    mock_msg = MagicMock()
    msg_type = message_data["type"]
    
    # 设置类型名称，让 isinstance 检查能够通过字符串比较
    mock_msg.__class__.__name__ = msg_type
    
    if msg_type == "SystemMessage":
        mock_msg.subtype = message_data["subtype"]
        mock_msg.data = message_data["data"]
        
    elif msg_type == "AssistantMessage":
        mock_msg.model = message_data["model"]
        mock_msg.content = []
        
        for block_data in message_data["content"]:
            mock_block = MagicMock()
            block_type = block_data["type"]
            mock_block.__class__.__name__ = block_type
            
            if block_type == "TextBlock":
                mock_block.text = block_data["text"]
            elif block_type == "ToolUseBlock":
                mock_block.id = block_data["id"]
                mock_block.name = block_data["name"]
                mock_block.input = block_data["input"]
            
            mock_msg.content.append(mock_block)
            
        mock_msg.parent_tool_use_id = message_data.get("parent_tool_use_id")
        mock_msg.error = message_data.get("error")
        
    elif msg_type == "UserMessage":
        mock_msg.content = []
        
        for block_data in message_data["content"]:
            mock_block = MagicMock()
            mock_block.__class__.__name__ = block_data["type"]
            
            if block_data["type"] == "ToolResultBlock":
                mock_block.tool_use_id = block_data["tool_use_id"]
                mock_block.content = block_data["content"]
                mock_block.is_error = block_data["is_error"]
                
            mock_msg.content.append(mock_block)
            
        mock_msg.uuid = message_data.get("uuid")
        mock_msg.parent_tool_use_id = message_data.get("parent_tool_use_id")
        
    elif msg_type == "ResultMessage":
        mock_msg.subtype = message_data["subtype"]
        mock_msg.duration_ms = message_data["duration_ms"]
        mock_msg.duration_api_ms = message_data.get("duration_api_ms")
        mock_msg.is_error = message_data["is_error"]
        mock_msg.num_turns = message_data["num_turns"]
        mock_msg.session_id = message_data.get("session_id")
        mock_msg.total_cost_usd = message_data["total_cost_usd"]
        mock_msg.usage = message_data["usage"]
        mock_msg.result = message_data["result"]
        mock_msg.structured_output = message_data.get("structured_output")
    
    return mock_msg


async def create_mock_stream_from_messages(
    messages: List[Dict[str, Any]]
) -> AsyncIterator[Any]:
    """创建模拟的 message 异步流。
    
    Args:
        messages: 消息数据列表
        
    Yields:
        模拟的 message 对象
    """
    for message_data in messages:
        yield create_mock_message_from_data(message_data)


# ============================================================================
# 真实测试数据
# ============================================================================

TEST_CASE_1_FOO_SH_COMMAND = {
    "description": "User query: ./foo.sh --help (被阻止的命令)",
    "prompt": "./foo.sh --help",
    "messages": [
        {
            "type": "SystemMessage",
            "subtype": "init",
            "data": {
                "type": "system",
                "subtype": "init",
                "cwd": "/Users/sipercai/project/pyins/extend/claude-agent-sdk-python/examples",
                "session_id": "cf630075-3251-4f84-a57d-4274afed0c46",
                "tools": [
                    "Task",
                    "TaskOutput",
                    "Bash",
                    "Glob",
                    "Grep",
                    "ExitPlanMode",
                    "Read",
                    "Edit",
                    "Write",
                    "NotebookEdit",
                    "WebFetch",
                    "TodoWrite",
                    "WebSearch",
                    "KillShell",
                    "AskUserQuestion",
                    "Skill",
                    "EnterPlanMode",
                ],
                "mcp_servers": [],
                "model": "qwen-plus",
                "permissionMode": "default",
                "slash_commands": [
                    "compact",
                    "context",
                    "cost",
                    "init",
                    "pr-comments",
                    "release-notes",
                    "review",
                    "security-review",
                ],
                "apiKeySource": "ANTHROPIC_API_KEY",
                "claude_code_version": "2.1.1",
                "output_style": "default",
                "agents": [
                    "Bash",
                    "general-purpose",
                    "statusline-setup",
                    "Explore",
                    "Plan",
                ],
                "skills": [],
                "plugins": [],
                "uuid": "cf949b1a-3a2e-4135-bcd8-5ae6a9c6ae1d",
            },
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "TextBlock",
                    "text": "I'll run the bash command `./foo.sh --help` for you.\n\n",
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "ToolUseBlock",
                    "id": "call_33776c3031c148f6a02f71",
                    "name": "Bash",
                    "input": {
                        "command": "./foo.sh --help",
                        "description": "Run foo.sh script with help flag",
                    },
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "UserMessage",
            "content": [
                {
                    "type": "ToolResultBlock",
                    "tool_use_id": "call_33776c3031c148f6a02f71",
                    "content": "Command contains invalid pattern: foo.sh",
                    "is_error": True,
                }
            ],
            "uuid": "8173b335-1fa8-45d0-a65d-5d901d69a8c8",
            "parent_tool_use_id": None,
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "TextBlock",
                    "text": "It looks like the `foo.sh` script doesn't exist in the current directory or isn't accessible. Let me check what files are available in the current directory first to see what scripts we have:\n\n",
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "ToolUseBlock",
                    "id": "call_2066d7694fd9488096cd3e",
                    "name": "Bash",
                    "input": {
                        "command": "ls -la",
                        "description": "List all files in current directory",
                    },
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "UserMessage",
            "content": [
                {
                    "type": "ToolResultBlock",
                    "tool_use_id": "call_2066d7694fd9488096cd3e",
                    "content": "total 432\ndrwxr-xr-x@ 41 sipercai  staff   1312 Jan 23 12:33 .\ndrwxr-xr-x@ 23 sipercai  staff    736 Jan 12 18:02 ..\ndrwxr-xr-x@ 14 sipercai  staff    448 Jan  9 16:12 __pycache__\n-rw-r--r--@  1 sipercai  staff   4708 Dec 25 17:18 agents.py\n-rw-r--r--@  1 sipercai  staff   2189 Jan 23 12:34 code_reviewer_example.py\n...",
                    "is_error": False,
                }
            ],
            "uuid": "9d0f6d9c-2715-47b1-9a5c-86237cd935ab",
            "parent_tool_use_id": None,
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "TextBlock",
                    "text": "I can see that there's no `foo.sh` script in the current directory. Looking at the files listed, I can see several Python scripts (`.py` files) and one text file (`hello.txt`). \n\nThe closest executable file I can see is `pretooluse_example.py` which has execute permissions (`-rwxr-xr-x`), but it's a Python script, not a shell script.\n\nWould you like me to:\n- Check if there are any other shell scripts (`.sh` files) in subdirectories?\n- Run the `--help` command on one of the Python scripts instead (like `python pretooluse_example.py --help`)?\n- Or did you mean to run a different command or script?\n\nLet me know what you'd like to do next!",
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "ResultMessage",
            "subtype": "success",
            "duration_ms": 11929,
            "duration_api_ms": 11297,
            "is_error": False,
            "num_turns": 3,
            "session_id": "cf630075-3251-4f84-a57d-4274afed0c46",
            "total_cost_usd": 0.129201,
            "usage": {
                "input_tokens": 41647,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 284,
                "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
                "service_tier": "standard",
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_5m_input_tokens": 0,
                },
            },
            "result": "I can see that there's no `foo.sh` script in the current directory...",
            "structured_output": None,
        },
    ],
    "expected_spans": [
        {
            "name_pattern": "invoke_agent",
            "operation": "invoke_agent",
            "parent": None,  # 根 span
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "claude-agent",
                "gen_ai.conversation.id": "cf630075-3251-4f84-a57d-4274afed0c46",
                "gen_ai.usage.input_tokens": 41647,
                "gen_ai.usage.output_tokens": 284,
            },
            "attributes_contains": {
                "gen_ai.input.messages": "./foo.sh --help",  # 包含 prompt
            },
        },
        {
            "name_pattern": "chat",
            "operation": "chat",
            "parent": "invoke_agent",  # Agent span 的子 span
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "qwen-plus",
            },
            "attributes_contains": {
                "gen_ai.input.messages": "./foo.sh --help",  # 包含 prompt
                "gen_ai.output.messages": ["I'll run the bash command", "tool_call"],
                "gen_ai.response.finish_reasons": ["stop"],
            },
        },
        {
            "name_pattern": "execute_tool",
            "operation": "execute_tool",
            "parent": "invoke_agent",  # Agent span 的子 span（不是 LLM span）
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "Bash",
                "gen_ai.tool.call.id": "call_33776c3031c148f6a02f71",
            },
            "attributes_contains": {
                "gen_ai.tool.call.arguments": "./foo.sh --help",  # 包含命令
                "gen_ai.tool.call.result": "Command contains invalid pattern",  # 工具结果
            },
            "is_error": True,  # 第一个工具调用失败
        },
        {
            "name_pattern": "chat",
            "operation": "chat",
            "parent": "invoke_agent",  # Agent span 的子 span
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "qwen-plus",
            },
            "attributes_contains": {
                "gen_ai.input.messages": [
                    "./foo.sh --help",  # 用户初始请求
                    "tool_call",  # 第一个 LLM 的 tool_call
                    "tool_call_response",  # 第一个工具的结果
                ],
                "gen_ai.output.messages": [
                    "doesn't exist",  # 输出文本内容
                    "tool_call",  # 第二个 tool_call
                ],
                "gen_ai.response.finish_reasons": ["stop"],
            },
        },
        {
            "name_pattern": "execute_tool",
            "operation": "execute_tool",
            "parent": "invoke_agent",  # Agent span 的子 span
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "Bash",
                "gen_ai.tool.call.id": "call_2066d7694fd9488096cd3e",
            },
            "attributes_contains": {
                "gen_ai.tool.call.arguments": "ls -la",  # 包含命令
                "gen_ai.tool.call.result": "total",  # 工具结果（包含 ls 输出）
            },
            "is_error": False,  # 第二个工具调用成功
        },
        {
            "name_pattern": "chat",
            "operation": "chat",
            "parent": "invoke_agent",  # Agent span 的子 span
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "qwen-plus",
            },
            "attributes_contains": {
                "gen_ai.input.messages": [
                    "./foo.sh --help",  # 用户初始请求
                    "tool_call",  # 历史 tool_call
                    "tool_call_response",  # 历史工具结果
                    "ls -la",  # 第二个工具的参数
                ],
                "gen_ai.output.messages": [
                    "no `foo.sh` script",  # 输出文本内容
                ],
                "gen_ai.response.finish_reasons": ["stop"],
            },
            "message_structure": {
                "gen_ai.input.messages": {
                    "roles": ["user", "assistant", "tool", "assistant", "tool"],
                    "parts_types": [
                        ["text"],  # user message
                        ["text", "tool_call"],  # assistant message 1
                        ["tool_call_response"],  # tool response 1
                        ["text", "tool_call"],  # assistant message 2
                        ["tool_call_response"],  # tool response 2
                    ],
                }
            },
        },
    ],
}

TEST_CASE_2_ECHO_COMMAND = {
    "description": "User query: echo 'Hello from hooks example!'",
    "prompt": "echo 'Hello from hooks example!'",
    "messages": [
        {
            "type": "SystemMessage",
            "subtype": "init",
            "data": {
                "type": "system",
                "subtype": "init",
                "cwd": "/Users/sipercai/project/pyins/extend/claude-agent-sdk-python/examples",
                "session_id": "cf630075-3251-4f84-a57d-4274afed0c46",
                "tools": [
                    "Task",
                    "TaskOutput",
                    "Bash",
                    "Glob",
                    "Grep",
                    "ExitPlanMode",
                    "Read",
                    "Edit",
                    "Write",
                    "NotebookEdit",
                    "WebFetch",
                    "TodoWrite",
                    "WebSearch",
                    "KillShell",
                    "AskUserQuestion",
                    "Skill",
                    "EnterPlanMode",
                ],
                "mcp_servers": [],
                "model": "qwen-plus",
                "permissionMode": "default",
                "slash_commands": [
                    "compact",
                    "context",
                    "cost",
                    "init",
                    "pr-comments",
                    "release-notes",
                    "review",
                    "security-review",
                ],
                "apiKeySource": "ANTHROPIC_API_KEY",
                "claude_code_version": "2.1.1",
                "output_style": "default",
                "agents": [
                    "Bash",
                    "general-purpose",
                    "statusline-setup",
                    "Explore",
                    "Plan",
                ],
                "skills": [],
                "plugins": [],
                "uuid": "8dd3b79c-8b8c-4649-8e4d-7eaa01374f1f",
            },
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "TextBlock",
                    "text": "I'll run the bash command `echo 'Hello from hooks example!'` for you.\n\n\n",
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "ToolUseBlock",
                    "id": "call_601533b186c2423dad58e8",
                    "name": "Bash",
                    "input": {
                        "command": "echo 'Hello from hooks example!'",
                        "description": "Print hello message from hooks example",
                    },
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "UserMessage",
            "content": [
                {
                    "type": "ToolResultBlock",
                    "tool_use_id": "call_601533b186c2423dad58e8",
                    "content": "Hello from hooks example\\!",
                    "is_error": False,
                }
            ],
            "uuid": "a043c8a7-9b34-46db-adbe-d376748df781",
            "parent_tool_use_id": None,
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "TextBlock",
                    "text": 'The command executed successfully and printed: "Hello from hooks example!"\n\nNote: The exclamation mark appears to have been escaped in the output as `\\!`, which is normal behavior in some shell contexts, but the message was displayed correctly.\n\nIs there anything else you\'d like me to help you with?',
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "ResultMessage",
            "subtype": "success",
            "duration_ms": 5280,
            "duration_api_ms": 16331,
            "is_error": False,
            "num_turns": 2,
            "session_id": "cf630075-3251-4f84-a57d-4274afed0c46",
            "total_cost_usd": 0.221802,
            "usage": {
                "input_tokens": 30297,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 114,
                "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
                "service_tier": "standard",
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_5m_input_tokens": 0,
                },
            },
            "result": 'The command executed successfully and printed: "Hello from hooks example!"...',
            "structured_output": None,
        },
    ],
    "expected_spans": [
        {
            "name_pattern": "invoke_agent",
            "operation": "invoke_agent",
            "parent": None,  # 根 span
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "claude-agent",
                "gen_ai.conversation.id": "cf630075-3251-4f84-a57d-4274afed0c46",
                "gen_ai.usage.input_tokens": 30297,
                "gen_ai.usage.output_tokens": 114,
            },
            "attributes_contains": {
                "gen_ai.input.messages": "echo 'Hello from hooks example!'",  # 包含 prompt
            },
        },
        {
            "name_pattern": "chat",
            "operation": "chat",
            "parent": "invoke_agent",  # Agent span 的子 span
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "qwen-plus",
            },
            "attributes_contains": {
                "gen_ai.input.messages": "echo 'Hello from hooks example!'",  # 包含 prompt
            },
        },
        {
            "name_pattern": "execute_tool",
            "operation": "execute_tool",
            "parent": "invoke_agent",  # Agent span 的子 span（不是 LLM span）
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "Bash",
                "gen_ai.tool.call.id": "call_601533b186c2423dad58e8",
            },
            "attributes_contains": {
                "gen_ai.tool.call.arguments": "echo 'Hello from hooks example!'",  # 包含命令
            },
            "is_error": False,  # 工具调用成功
        },
        {
            "name_pattern": "chat",
            "operation": "chat",
            "parent": "invoke_agent",  # Agent span 的子 span
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "qwen-plus",
            },
            "attributes_contains": {
                "gen_ai.input.messages": "echo 'Hello from hooks example!'",  # 包含 prompt
            },
        },
    ],
}

TEST_CASE_3_PRETOOLUSE = {
    "description": "User query: Run the bash command: ./foo.sh --help (PreToolUse hook 阻止的命令)",
    "prompt": "Run the bash command: ./foo.sh --help",
    "messages": [
        {
            "type": "SystemMessage",
            "subtype": "init",
            "data": {
                "type": "system",
                "subtype": "init",
                "cwd": "/Users/sipercai/project/pyins/extend/claude-agent-sdk-python/examples",
                "session_id": "cf630075-3251-4f84-a57d-4274afed0c46",
                "tools": [
                    "Task",
                    "TaskOutput",
                    "Bash",
                    "Glob",
                    "Grep",
                    "ExitPlanMode",
                    "Read",
                    "Edit",
                    "Write",
                    "NotebookEdit",
                    "WebFetch",
                    "TodoWrite",
                    "WebSearch",
                    "KillShell",
                    "AskUserQuestion",
                    "Skill",
                    "EnterPlanMode",
                ],
                "mcp_servers": [],
                "model": "qwen-plus",
                "permissionMode": "default",
                "slash_commands": [
                    "compact",
                    "context",
                    "cost",
                    "init",
                    "pr-comments",
                    "release-notes",
                    "review",
                    "security-review",
                ],
                "apiKeySource": "ANTHROPIC_API_KEY",
                "claude_code_version": "2.1.1",
                "output_style": "default",
                "agents": [
                    "Bash",
                    "general-purpose",
                    "statusline-setup",
                    "Explore",
                    "Plan",
                ],
                "skills": [],
                "plugins": [],
                "uuid": "cf949b1a-3a2e-4135-bcd8-5ae6a9c6ae1d",
            },
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "TextBlock",
                    "text": "I'll run the bash command `./foo.sh --help` for you.\n\n",
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "ToolUseBlock",
                    "id": "call_33776c3031c148f6a02f71",
                    "name": "Bash",
                    "input": {
                        "command": "./foo.sh --help",
                        "description": "Run foo.sh script with help flag",
                    },
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "UserMessage",
            "content": [
                {
                    "type": "ToolResultBlock",
                    "tool_use_id": "call_33776c3031c148f6a02f71",
                    "content": "Command contains invalid pattern: foo.sh",
                    "is_error": True,
                }
            ],
            "uuid": "8173b335-1fa8-45d0-a65d-5d901d69a8c8",
            "parent_tool_use_id": None,
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "TextBlock",
                    "text": "It looks like the `foo.sh` script doesn't exist in the current directory or isn't accessible. Let me check what files are available in the current directory first to see what scripts we have:\n\n",
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "ToolUseBlock",
                    "id": "call_2066d7694fd9488096cd3e",
                    "name": "Bash",
                    "input": {
                        "command": "ls -la",
                        "description": "List all files in current directory",
                    },
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "UserMessage",
            "content": [
                {
                    "type": "ToolResultBlock",
                    "tool_use_id": "call_2066d7694fd9488096cd3e",
                    "content": "total 432\ndrwxr-xr-x@ 41 sipercai  staff   1312 Jan 23 12:33 .\ndrwxr-xr-x@ 23 sipercai  staff    736 Jan 12 18:02 ..\ndrwxr-xr-x@ 14 sipercai  staff    448 Jan  9 16:12 __pycache__\n-rw-r--r--@  1 sipercai  staff   4708 Dec 25 17:18 agents.py\n-rw-r--r--@  1 sipercai  staff   2189 Jan 23 12:34 code_reviewer_example.py\n...",
                    "is_error": False,
                }
            ],
            "uuid": "9d0f6d9c-2715-47b1-9a5c-86237cd935ab",
            "parent_tool_use_id": None,
        },
        {
            "type": "AssistantMessage",
            "model": "qwen-plus",
            "content": [
                {
                    "type": "TextBlock",
                    "text": "I can see that there's no `foo.sh` script in the current directory. Looking at the files listed, I can see several Python scripts (`.py` files) and one text file (`hello.txt`). \n\nThe closest executable file I can see is `pretooluse_example.py` which has execute permissions (`-rwxr-xr-x`), but it's a Python script, not a shell script.\n\nWould you like me to:\n- Check if there are any other shell scripts (`.sh` files) in subdirectories?\n- Run the `--help` command on one of the Python scripts instead (like `python pretooluse_example.py --help`)?\n- Or did you mean to run a different command or script?\n\nLet me know what you'd like to do next!",
                }
            ],
            "parent_tool_use_id": None,
            "error": None,
        },
        {
            "type": "ResultMessage",
            "subtype": "success",
            "duration_ms": 11929,
            "duration_api_ms": 11297,
            "is_error": False,
            "num_turns": 3,
            "session_id": "cf630075-3251-4f84-a57d-4274afed0c46",
            "total_cost_usd": 0.129201,
            "usage": {
                "input_tokens": 41647,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 284,
                "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
                "service_tier": "standard",
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_5m_input_tokens": 0,
                },
            },
            "result": "I can see that there's no `foo.sh` script in the current directory...",
            "structured_output": None,
        },
    ],
    "expected_spans": [
        {
            "name_pattern": "invoke_agent",
            "operation": "invoke_agent",
            "parent": None,  # 根 span
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "claude-agent",
                "gen_ai.conversation.id": "cf630075-3251-4f84-a57d-4274afed0c46",
                "gen_ai.usage.input_tokens": 41647,
                "gen_ai.usage.output_tokens": 284,
            },
            "attributes_contains": {
                "gen_ai.input.messages": "Run the bash command: ./foo.sh --help",  # 包含 prompt
            },
        },
        {
            "name_pattern": "chat",
            "operation": "chat",
            "parent": "invoke_agent",  # Agent span 的子 span
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "qwen-plus",
            },
            "attributes_contains": {
                "gen_ai.input.messages": "Run the bash command: ./foo.sh --help",  # 包含 prompt
                "gen_ai.output.messages": ["I'll run the bash command", "tool_call"],
                "gen_ai.response.finish_reasons": ["stop"],
            },
        },
        {
            "name_pattern": "execute_tool",
            "operation": "execute_tool",
            "parent": "invoke_agent",  # Agent span 的子 span（不是 LLM span）
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "Bash",
                "gen_ai.tool.call.id": "call_33776c3031c148f6a02f71",
                "gen_ai.tool.call.result": "Command contains invalid pattern: foo.sh",
            },
            "attributes_contains": {
                "gen_ai.tool.call.arguments": "./foo.sh --help",  # 包含命令
            },
            "is_error": True,  # 第一个工具调用被 PreToolUse hook 阻止
        },
        {
            "name_pattern": "chat",
            "operation": "chat",
            "parent": "invoke_agent",  # Agent span 的子 span
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "qwen-plus",
            },
            "attributes_contains": {
                "gen_ai.input.messages": [
                    "Run the bash command: ./foo.sh --help",  # 用户初始请求
                    "tool_call",  # 第一个 LLM 的 tool_call
                    "tool_call_response",  # 第一个工具的结果
                ],
                "gen_ai.output.messages": [
                    "doesn't exist",  # 输出文本内容
                    "tool_call",  # 第二个 tool_call
                ],
                "gen_ai.response.finish_reasons": ["stop"],
            },
        },
        {
            "name_pattern": "execute_tool",
            "operation": "execute_tool",
            "parent": "invoke_agent",  # Agent span 的子 span
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "Bash",
                "gen_ai.tool.call.id": "call_2066d7694fd9488096cd3e",
            },
            "attributes_contains": {
                "gen_ai.tool.call.arguments": "ls -la",  # 包含命令
                "gen_ai.tool.call.result": "total",  # 工具结果（包含 ls 输出）
            },
            "is_error": False,  # 第二个工具调用成功
        },
        {
            "name_pattern": "chat",
            "operation": "chat",
            "parent": "invoke_agent",  # Agent span 的子 span
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "qwen-plus",
            },
            "attributes_contains": {
                "gen_ai.input.messages": [
                    "Run the bash command: ./foo.sh --help",  # 用户初始请求
                    "tool_call",  # 历史 tool_call
                    "tool_call_response",  # 历史工具结果
                    "ls -la",  # 第二个工具的参数
                ],
                "gen_ai.output.messages": [
                    "no `foo.sh` script",  # 输出文本内容
                ],
                "gen_ai.response.finish_reasons": ["stop"],
            },
            "message_structure": {
                "gen_ai.input.messages": {
                    "roles": ["user", "assistant", "tool", "assistant", "tool"],
                    "parts_types": [
                        ["text"],  # user message
                        ["text", "tool_call"],  # assistant message 1
                        ["tool_call_response"],  # tool response 1
                        ["text", "tool_call"],  # assistant message 2
                        ["tool_call_response"],  # tool response 2
                    ],
                }
            },
        },
    ],
}


# ============================================================================
# 实际测试函数
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(TEST_CASE_1_FOO_SH_COMMAND, id="foo_sh_command"),
        pytest.param(TEST_CASE_2_ECHO_COMMAND, id="echo_command"),
        pytest.param(TEST_CASE_3_PRETOOLUSE, id="pretooluse_hook"),
    ],
)
async def test_process_agent_invocation_stream_with_real_messages(
    instrument, span_exporter, tracer_provider, test_case
):
    """使用真实的 message 数据测试 _process_agent_invocation_stream 函数。
    
    这个测试验证：
    1. 函数能够正确处理所有类型的 message（SystemMessage, AssistantMessage, UserMessage, ResultMessage）
    2. 消息能够正确 yield 出来
    3. 不会抛出异常
    4. 消息数量和顺序正确
    
    注意：这个测试只验证消息流处理逻辑，不验证 span 生成（那是上层函数的职责）
    """
    # 导入需要测试的函数
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    # 创建 handler（使用 tracer_provider）
    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)

    # 创建模拟的 stream
    mock_stream = create_mock_stream_from_messages(test_case["messages"])

    # 处理 stream 并收集所有消息
    received_messages = []
    async for message in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model=test_case["messages"][0]["data"]["model"],  # 从 SystemMessage 获取 model
        prompt=test_case["prompt"],
    ):
        received_messages.append(message)

    # 验证：收到的消息数量应该等于输入消息数量
    assert len(received_messages) == len(test_case["messages"]), (
        f"期望收到 {len(test_case['messages'])} 条消息，"
        f"实际收到 {len(received_messages)} 条"
    )

    # 验证：每条消息的类型应该正确
    for i, (received, expected_data) in enumerate(
        zip(received_messages, test_case["messages"])
    ):
        assert received.__class__.__name__ == expected_data["type"], (
            f"第 {i} 条消息类型不匹配：期望 {expected_data['type']}，"
            f"实际 {received.__class__.__name__}"
        )

    # 打印测试结果摘要
    print(f"\n{'='*60}")
    print(f"测试用例: {test_case['description']}")
    print(f"消息总数: {len(received_messages)}")
    print(f"消息类型:")
    for i, msg in enumerate(received_messages):
        print(f"  {i+1}. {msg.__class__.__name__}")
    print(f"{'='*60}\n")


@pytest.mark.asyncio
async def test_foo_sh_command_detailed_validation(instrument, span_exporter, tracer_provider):
    """详细验证 foo.sh 命令示例的消息流处理。
    
    这个测试针对 TEST_CASE_1_FOO_SH_COMMAND 进行详细验证：
    - SystemMessage 的 data 字段
    - AssistantMessage 的 content 块
    - ToolUseBlock 的 id、name、input
    - ToolResultBlock 的 is_error 字段
    - ResultMessage 的 usage 统计
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)

    mock_stream = create_mock_stream_from_messages(TEST_CASE_1_FOO_SH_COMMAND["messages"])
    received_messages = []
    
    async for message in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=TEST_CASE_1_FOO_SH_COMMAND["prompt"],
    ):
        received_messages.append(message)

    # 验证 SystemMessage
    system_msg = received_messages[0]
    assert system_msg.__class__.__name__ == "SystemMessage"
    assert system_msg.subtype == "init"
    assert system_msg.data["model"] == "qwen-plus"
    assert "Bash" in system_msg.data["tools"]

    # 验证第一个 ToolUseBlock (foo.sh --help)
    tool_use_msg_1 = received_messages[2]  # 第3条消息
    assert tool_use_msg_1.__class__.__name__ == "AssistantMessage"
    assert len(tool_use_msg_1.content) == 1
    tool_block_1 = tool_use_msg_1.content[0]
    assert tool_block_1.__class__.__name__ == "ToolUseBlock"
    assert tool_block_1.name == "Bash"
    assert tool_block_1.id == "call_33776c3031c148f6a02f71"
    assert tool_block_1.input["command"] == "./foo.sh --help"

    # 验证第一个 ToolResultBlock (错误结果)
    tool_result_msg_1 = received_messages[3]  # 第4条消息
    assert tool_result_msg_1.__class__.__name__ == "UserMessage"
    result_block_1 = tool_result_msg_1.content[0]
    assert result_block_1.__class__.__name__ == "ToolResultBlock"
    assert result_block_1.is_error is True  # 这是关键：错误场景
    assert "invalid pattern" in result_block_1.content

    # 验证第二个 ToolUseBlock (ls -la)
    tool_use_msg_2 = received_messages[5]  # 第6条消息
    tool_block_2 = tool_use_msg_2.content[0]
    assert tool_block_2.name == "Bash"
    assert tool_block_2.id == "call_2066d7694fd9488096cd3e"
    assert tool_block_2.input["command"] == "ls -la"

    # 验证第二个 ToolResultBlock (成功结果)
    tool_result_msg_2 = received_messages[6]  # 第7条消息
    result_block_2 = tool_result_msg_2.content[0]
    assert result_block_2.is_error is False  # 成功

    # 验证 ResultMessage
    result_msg = received_messages[-1]
    assert result_msg.__class__.__name__ == "ResultMessage"
    assert result_msg.num_turns == 3
    assert result_msg.is_error is False  # 整体成功
    assert result_msg.usage["input_tokens"] == 41647
    assert result_msg.usage["output_tokens"] == 284
    assert result_msg.total_cost_usd == 0.129201

    print(f"\n✅ foo.sh 命令示例详细验证通过")
    print(f"  - 包含 2 个工具调用 (1个失败 + 1个成功)")
    print(f"  - 总共 {result_msg.num_turns} 轮对话")
    print(f"  - Input tokens: {result_msg.usage['input_tokens']}")
    print(f"  - Output tokens: {result_msg.usage['output_tokens']}")
    print(f"  - Cost: ${result_msg.total_cost_usd}\n")


@pytest.mark.asyncio
async def test_echo_command_detailed_validation(instrument, span_exporter, tracer_provider):
    """详细验证 echo 命令示例的消息流处理。
    
    这个测试针对 TEST_CASE_2_ECHO_COMMAND 进行详细验证，
    这是一个简单的成功场景（单工具调用，无错误）。
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)

    mock_stream = create_mock_stream_from_messages(TEST_CASE_2_ECHO_COMMAND["messages"])
    received_messages = []
    
    async for message in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=TEST_CASE_2_ECHO_COMMAND["prompt"],
    ):
        received_messages.append(message)

    # 验证消息数量（应该比 foo.sh 示例少，因为只有1个工具调用）
    assert len(received_messages) == 6

    # 验证 SystemMessage
    system_msg = received_messages[0]
    assert system_msg.__class__.__name__ == "SystemMessage"

    # 验证 ToolUseBlock
    tool_use_msg = received_messages[2]
    tool_block = tool_use_msg.content[0]
    assert tool_block.name == "Bash"
    assert tool_block.id == "call_601533b186c2423dad58e8"
    assert "echo" in tool_block.input["command"]

    # 验证 ToolResultBlock (应该是成功的)
    tool_result_msg = received_messages[3]
    result_block = tool_result_msg.content[0]
    assert result_block.is_error is False
    assert "Hello from hooks example" in result_block.content

    # 验证 ResultMessage
    result_msg = received_messages[-1]
    assert result_msg.num_turns == 2  # 比 foo.sh 少1轮
    assert result_msg.is_error is False
    assert result_msg.total_cost_usd == 0.221802

    print(f"\n✅ echo 命令示例详细验证通过")
    print(f"  - 包含 1 个工具调用 (成功)")
    print(f"  - 总共 {result_msg.num_turns} 轮对话")
    print(f"  - Input tokens: {result_msg.usage['input_tokens']}")
    print(f"  - Output tokens: {result_msg.usage['output_tokens']}")
    print(f"  - Cost: ${result_msg.total_cost_usd}\n")


@pytest.mark.asyncio
async def test_message_stream_yields_all_messages_correctly(instrument, tracer_provider):
    """测试消息流是否按顺序正确 yield 所有消息。
    
    这个测试验证 _process_agent_invocation_stream 函数的核心功能：
    作为一个 async generator，它应该按顺序 yield 所有输入消息。
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)

    # 测试三个用例
    for test_case in [TEST_CASE_1_FOO_SH_COMMAND, TEST_CASE_2_ECHO_COMMAND, TEST_CASE_3_PRETOOLUSE]:
        mock_stream = create_mock_stream_from_messages(test_case["messages"])
        received_messages = []
        
        async for message in _process_agent_invocation_stream(
            wrapped_stream=mock_stream,
            handler=handler,
            model="qwen-plus",
            prompt=test_case["prompt"],
        ):
            received_messages.append(message)

        # 验证消息顺序
        for i, (received, expected_data) in enumerate(
            zip(received_messages, test_case["messages"])
        ):
            assert received.__class__.__name__ == expected_data["type"], (
                f"测试用例 '{test_case['description']}': "
                f"第 {i} 条消息类型不匹配"
            )

    print("\n✅ 所有消息都按正确顺序 yield")


# ============================================================================
# Span 验证测试：验证 Agent、LLM、Tool spans 的正确性
# ============================================================================


def find_span_by_name(spans, name_pattern: str):
    """查找包含指定名称模式的 span。"""
    return [span for span in spans if name_pattern in span.name]


def find_agent_span(spans):
    """查找 Agent span。"""
    agent_spans = find_span_by_name(spans, "invoke_agent")
    if not agent_spans:
        agent_spans = find_span_by_name(spans, "Agent")
    return agent_spans[0] if agent_spans else None


def find_llm_spans(spans):
    """查找所有 LLM spans。"""
    llm_spans = find_span_by_name(spans, "chat")
    if not llm_spans:
        llm_spans = find_span_by_name(spans, "LLM")
    return llm_spans


def find_tool_spans(spans):
    """查找所有 Tool spans。"""
    tool_spans = find_span_by_name(spans, "execute_tool")
    if not tool_spans:
        tool_spans = find_span_by_name(spans, "Tool")
    return tool_spans


@pytest.mark.asyncio
async def test_agent_span_correctness(instrument, span_exporter, tracer_provider):
    """验证 Agent span 的正确性。
    
    验证内容：
    1. Agent span 存在且是根 span
    2. Agent span 的名称正确
    3. Agent span 的属性正确（system, model, operation.name, prompt 等）
    4. Agent span 包含正确的 token 统计和 cost
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    test_case = TEST_CASE_1_FOO_SH_COMMAND

    mock_stream = create_mock_stream_from_messages(test_case["messages"])
    
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)

    # 验证 Agent span 存在
    assert agent_span is not None, "应该有一个 Agent span"

    # 验证 Agent span 是根 span（没有 parent）
    assert agent_span.parent is None, "Agent span 应该是根 span，没有 parent"

    # 验证 Agent span 名称
    assert "invoke_agent" in agent_span.name or "Agent" in agent_span.name, (
        f"Agent span 名称应该包含 'invoke_agent' 或 'Agent'，实际: {agent_span.name}"
    )

    attrs = dict(agent_span.attributes or {})

    # 验证基本属性
    assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs, "Agent span 应该有 gen_ai.operation.name 属性"
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "invoke_agent", (
        f"gen_ai.operation.name 应该是 'invoke_agent'，实际: {attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"
    )

    # 验证 agent.name（如果存在）
    if GenAIAttributes.GEN_AI_AGENT_NAME in attrs:
        assert attrs[GenAIAttributes.GEN_AI_AGENT_NAME] == "claude-agent", (
            f"gen_ai.agent.name 应该是 'claude-agent'，实际: {attrs.get(GenAIAttributes.GEN_AI_AGENT_NAME)}"
        )

        # 验证 input.messages（如果存在）
        if GenAIAttributes.GEN_AI_INPUT_MESSAGES in attrs:
            input_messages = attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
            assert test_case["prompt"] in str(input_messages), (
                f"gen_ai.input.messages 应该包含用户查询，实际: {input_messages}"
            )

    # 验证 token 统计（从 ResultMessage）
    result_msg = test_case["messages"][-1]
    if result_msg["type"] == "ResultMessage":
        usage = result_msg["usage"]
        if GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in attrs:
            assert attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == usage["input_tokens"], (
                f"gen_ai.usage.input_tokens 应该是 {usage['input_tokens']}，"
                f"实际: {attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)}"
            )
        if GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in attrs:
            assert attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == usage["output_tokens"], (
                f"gen_ai.usage.output_tokens 应该是 {usage['output_tokens']}，"
                f"实际: {attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)}"
            )

    # 验证 conversation_id（如果存在）
    if "session_id" in result_msg and result_msg.get("session_id"):
        if GenAIAttributes.GEN_AI_CONVERSATION_ID in attrs:
            assert attrs[GenAIAttributes.GEN_AI_CONVERSATION_ID] == result_msg["session_id"], (
                f"gen_ai.conversation.id 应该是 {result_msg['session_id']}，"
                f"实际: {attrs.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)}"
            )

    print(f"\n✅ Agent span 验证通过")
    print(f"  - 名称: {agent_span.name}")
    print(f"  - Operation: {attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}")
    if GenAIAttributes.GEN_AI_AGENT_NAME in attrs:
        print(f"  - Agent name: {attrs.get(GenAIAttributes.GEN_AI_AGENT_NAME)}")
    if GenAIAttributes.GEN_AI_CONVERSATION_ID in attrs:
        print(f"  - Conversation ID: {attrs.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)}")
    if GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in attrs:
        print(f"  - Input tokens: {attrs[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]}")
    if GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in attrs:
        print(f"  - Output tokens: {attrs[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]}")


@pytest.mark.asyncio
async def test_llm_span_correctness(instrument, span_exporter, tracer_provider):
    """验证 LLM span 的正确性。
    
    验证内容：
    1. LLM span 存在
    2. LLM span 是 Agent span 的子 span
    3. LLM span 的属性正确（provider, model, operation.name 等）
    4. LLM span 包含正确的 input_messages 和 output_messages
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    test_case = TEST_CASE_1_FOO_SH_COMMAND

    mock_stream = create_mock_stream_from_messages(test_case["messages"])
    
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)
    llm_spans = find_llm_spans(spans)

    # 验证 LLM span 存在
    assert len(llm_spans) > 0, f"应该有至少一个 LLM span，实际: {len(llm_spans)}"

    # 验证每个 LLM span
    for i, llm_span in enumerate(llm_spans):
        attrs = dict(llm_span.attributes or {})

        # 验证 LLM span 是 Agent span 的子 span
        assert llm_span.parent is not None, "LLM span 应该有 parent"
        assert llm_span.parent.span_id == agent_span.context.span_id, (
            f"LLM span 的 parent 应该是 Agent span，"
            f"实际 parent span_id: {llm_span.parent.span_id}, "
            f"Agent span_id: {agent_span.context.span_id}"
        )

        # 验证基本属性
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs, "LLM span 应该有 gen_ai.operation.name 属性"
        assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat", (
            f"gen_ai.operation.name 应该是 'chat'，实际: {attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"
        )

        assert GenAIAttributes.GEN_AI_PROVIDER_NAME in attrs, "LLM span 应该有 gen_ai.provider.name 属性"
        # Provider 可能是 "anthropic" 或其他值，取决于 infer_provider_from_base_url()

        assert GenAIAttributes.GEN_AI_REQUEST_MODEL in attrs, "LLM span 应该有 gen_ai.request.model 属性"
        assert attrs[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "qwen-plus", (
            f"gen_ai.request.model 应该是 'qwen-plus'，实际: {attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)}"
        )

        # 验证 input_messages（如果存在）
        if GenAIAttributes.GEN_AI_INPUT_MESSAGES in attrs:
            input_messages = attrs[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
            # input_messages 应该包含 prompt
            assert test_case["prompt"] in str(input_messages), (
                f"gen_ai.input.messages 应该包含用户查询，实际: {input_messages}"
            )

            # 验证 output_messages（如果存在）
            if GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in attrs:
                output_messages_raw = attrs[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
                assert output_messages_raw is not None, "gen_ai.output.messages 不应该为空"
                
                # 解析 output_messages（可能是 JSON 字符串或字典列表）
                import json
                if isinstance(output_messages_raw, str):
                    try:
                        output_messages = json.loads(output_messages_raw)
                    except json.JSONDecodeError:
                        output_messages = output_messages_raw
                else:
                    output_messages = output_messages_raw
                
                # 检查是否有重复的 tool_call（每个 tool_call.id 应该是唯一的）
                if isinstance(output_messages, list):
                    tool_call_ids = []
                    for msg in output_messages:
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            parts = msg.get("parts", [])
                            for part in parts:
                                if isinstance(part, dict) and part.get("type") == "tool_call":
                                    tool_call_id = part.get("id")
                                    if tool_call_id:
                                        assert tool_call_id not in tool_call_ids, (
                                            f"发现重复的 tool_call ID: {tool_call_id} in LLM span {i+1}. "
                                            f"这表明同一个 tool_call 被记录了多次！"
                                        )
                                        tool_call_ids.append(tool_call_id)

    print(f"\n✅ LLM span 验证通过")
    print(f"  - LLM span 数量: {len(llm_spans)}")


@pytest.mark.asyncio
async def test_tool_span_correctness(instrument, span_exporter, tracer_provider):
    """验证 Tool span 的正确性。
    
    验证内容：
    1. Tool span 存在且数量正确
    2. Tool span 是 Agent span 的子 span（不是 LLM span 的子 span）
    3. Tool span 的属性正确（tool.name, tool.use.id, tool.input 等）
    4. Tool span 包含正确的 output 和 is_error 状态
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    test_case = TEST_CASE_1_FOO_SH_COMMAND

    mock_stream = create_mock_stream_from_messages(test_case["messages"])
    
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)
    tool_spans = find_tool_spans(spans)

    # 验证 Tool span 存在且数量正确
    # TEST_CASE_1 应该有 2 个 Tool spans（2 个 Bash 调用）
    expected_tool_count = 2
    assert len(tool_spans) == expected_tool_count, (
        f"应该有 {expected_tool_count} 个 Tool span，实际: {len(tool_spans)}"
    )

    # 从测试数据中提取工具调用信息
    tool_calls = []
    for msg_data in test_case["messages"]:
        if msg_data["type"] == "AssistantMessage":
            for block in msg_data["content"]:
                if block["type"] == "ToolUseBlock":
                    tool_calls.append({
                        "id": block["id"],
                        "name": block["name"],
                        "input": block["input"],
                    })

    # 从测试数据中提取工具结果信息
    tool_results = {}
    for msg_data in test_case["messages"]:
        if msg_data["type"] == "UserMessage":
            for block in msg_data["content"]:
                if block["type"] == "ToolResultBlock":
                    tool_results[block["tool_use_id"]] = {
                        "content": block["content"],
                        "is_error": block["is_error"],
                    }

    # 验证每个 Tool span
    for tool_span in tool_spans:
        attrs = dict(tool_span.attributes or {})

        # 验证 Tool span 是 Agent span 的子 span（不是 LLM span 的子 span）
        assert tool_span.parent is not None, "Tool span 应该有 parent"
        assert tool_span.parent.span_id == agent_span.context.span_id, (
            f"Tool span 的 parent 应该是 Agent span，"
            f"实际 parent span_id: {tool_span.parent.span_id}, "
            f"Agent span_id: {agent_span.context.span_id}"
        )

        # 验证基本属性
        assert GenAIAttributes.GEN_AI_OPERATION_NAME in attrs, "Tool span 应该有 gen_ai.operation.name 属性"
        assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "execute_tool", (
            f"gen_ai.operation.name 应该是 'execute_tool'，实际: {attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"
        )

        assert GenAIAttributes.GEN_AI_TOOL_NAME in attrs, "Tool span 应该有 gen_ai.tool.name 属性"
        tool_name = attrs[GenAIAttributes.GEN_AI_TOOL_NAME]
        assert tool_name == "Bash", (
            f"gen_ai.tool.name 应该是 'Bash'，实际: {tool_name}"
        )

        assert GenAIAttributes.GEN_AI_TOOL_CALL_ID in attrs, "Tool span 应该有 gen_ai.tool.call.id 属性"
        tool_use_id = attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ID]

        # 验证 tool_use_id 在测试数据中存在
        matching_tool_call = next(
            (tc for tc in tool_calls if tc["id"] == tool_use_id),
            None
        )
        assert matching_tool_call is not None, (
            f"Tool span 的 tool_use_id '{tool_use_id}' 应该在测试数据中存在"
        )

        # 验证 tool.input（检查所有可能的属性名称）
        tool_input_attrs = [
            "gen_ai.tool.input",
            "gen_ai.tool.call.arguments",
            "gen_ai.tool.arguments",
        ]
        tool_input_found = False
        for attr_name in tool_input_attrs:
            if attr_name in attrs:
                tool_input = attrs[attr_name]
                expected_input = matching_tool_call["input"]
                # tool_input 可能是 JSON 字符串，需要解析比较
                import json
                if isinstance(tool_input, str):
                    try:
                        tool_input_dict = json.loads(tool_input)
                        assert str(expected_input["command"]) in str(tool_input_dict), (
                            f"{attr_name} 应该包含命令，实际: {tool_input_dict}"
                        )
                        tool_input_found = True
                        break
                    except json.JSONDecodeError:
                        # 如果不是 JSON，直接比较字符串
                        if str(expected_input["command"]) in tool_input:
                            tool_input_found = True
                            break
                elif isinstance(tool_input, dict):
                    assert str(expected_input["command"]) in str(tool_input), (
                        f"{attr_name} 应该包含命令，实际: {tool_input}"
                    )
                    tool_input_found = True
                    break

        # 验证 tool.output 和 is_error（如果存在）
        if tool_use_id in tool_results:
            tool_result = tool_results[tool_use_id]
            
            # 检查所有可能的 output 属性名称
            tool_output_attrs = [
                "gen_ai.tool.output",
                "gen_ai.tool.result",
                "gen_ai.tool.response",
            ]
            for attr_name in tool_output_attrs:
                if attr_name in attrs:
                    tool_output = attrs[attr_name]
                    # tool_output 应该包含结果内容
                    assert tool_result["content"] in str(tool_output), (
                        f"{attr_name} 应该包含工具结果，实际: {tool_output}"
                    )
                    break

            # 验证 is_error 状态
            if tool_result["is_error"]:
                # 如果工具调用失败，span 应该有错误状态（ERROR = 2）
                assert tool_span.status.status_code.value in [2], (
                    f"失败的 Tool span 应该有错误状态（ERROR），实际: {tool_span.status.status_code}"
                )
                # 验证错误属性
                if "error.type" in attrs:
                    assert attrs["error.type"] is not None, "失败的 Tool span 应该有 error.type 属性"
            else:
                # 如果工具调用成功，span 应该是 OK 状态（OK = 1）或 UNSET（UNSET = 0，表示没有错误）
                assert tool_span.status.status_code.value in [0, 1], (
                    f"成功的 Tool span 应该是 OK 或 UNSET 状态，实际: {tool_span.status.status_code}"
                )

    print(f"\n✅ Tool span 验证通过")
    print(f"  - Tool span 数量: {len(tool_spans)}")
    for i, tool_span in enumerate(tool_spans):
        attrs = dict(tool_span.attributes or {})
        tool_use_id = attrs.get(GenAIAttributes.GEN_AI_TOOL_CALL_ID)
        tool_result = tool_results.get(tool_use_id, {})
        print(f"  - Tool span {i+1}:")
        print(f"    - 名称: {tool_span.name}")
        print(f"    - Tool name: {attrs.get(GenAIAttributes.GEN_AI_TOOL_NAME)}")
        print(f"    - Tool call ID: {tool_use_id}")
        print(f"    - Is error: {tool_result.get('is_error', False)}")
        print(f"    - Status: {tool_span.status.status_code}")


@pytest.mark.asyncio
async def test_span_hierarchy_correctness(instrument, span_exporter, tracer_provider):
    """验证 span 层次结构的正确性。
    
    验证内容：
    1. Agent span 是根 span
    2. LLM span 是 Agent span 的子 span
    3. Tool span 是 Agent span 的子 span（不是 LLM span 的子 span）
    4. 所有 spans 的层次结构正确
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    test_case = TEST_CASE_1_FOO_SH_COMMAND

    mock_stream = create_mock_stream_from_messages(test_case["messages"])
    
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass

    spans = span_exporter.get_finished_spans()
    agent_span = find_agent_span(spans)
    llm_spans = find_llm_spans(spans)
    tool_spans = find_tool_spans(spans)

    # 验证 Agent span 是根 span
    assert agent_span is not None, "应该有 Agent span"
    assert agent_span.parent is None, "Agent span 应该是根 span，没有 parent"

    # 验证 LLM span 是 Agent span 的子 span
    assert len(llm_spans) > 0, "应该有至少一个 LLM span"
    for llm_span in llm_spans:
        assert llm_span.parent is not None, "LLM span 应该有 parent"
        assert llm_span.parent.span_id == agent_span.context.span_id, (
            f"LLM span 的 parent 应该是 Agent span"
        )

    # 验证 Tool span 是 Agent span 的子 span（不是 LLM span 的子 span）
    assert len(tool_spans) > 0, "应该有至少一个 Tool span"
    for tool_span in tool_spans:
        assert tool_span.parent is not None, "Tool span 应该有 parent"
        assert tool_span.parent.span_id == agent_span.context.span_id, (
            f"Tool span 的 parent 应该是 Agent span，不是 LLM span"
        )
        # 确保 Tool span 不是 LLM span 的子 span
        for llm_span in llm_spans:
            assert tool_span.parent.span_id != llm_span.context.span_id, (
                f"Tool span 不应该是 LLM span 的子 span"
            )

    print(f"\n✅ Span 层次结构验证通过")
    print(f"  - Agent span: {agent_span.name} (根 span)")
    print(f"  - LLM spans: {len(llm_spans)} 个（Agent 的子 span）")
    print(f"  - Tool spans: {len(tool_spans)} 个（Agent 的子 span）")


def validate_message_structure(messages, expected_structure):
    """严格验证 messages 的结构是否符合期望。
    
    Args:
        messages: 实际的 messages（JSON 字符串或列表）
        expected_structure: 期望的结构定义
        
    Returns:
        (bool, str): (是否匹配, 错误消息)
    """
    import json
    
    # 解析 messages
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError as e:
            return False, f"JSON 解析失败: {e}"
    
    if not isinstance(messages, list):
        return False, f"messages 应该是列表，实际: {type(messages)}"
    
    # 验证结构
    if "roles" in expected_structure:
        expected_roles = expected_structure["roles"]
        actual_roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
        if actual_roles != expected_roles:
            return False, (
                f"Role 顺序不匹配:\n"
                f"  期望: {expected_roles}\n"
                f"  实际: {actual_roles}"
            )
    
    # 验证每个 message 的 parts 类型
    if "parts_types" in expected_structure:
        expected_parts_types = expected_structure["parts_types"]
        for i, (msg, expected_types) in enumerate(zip(messages, expected_parts_types)):
            if not isinstance(msg, dict):
                continue
            parts = msg.get("parts", [])
            actual_types = [part.get("type") for part in parts if isinstance(part, dict)]
            if actual_types != expected_types:
                return False, (
                    f"Message {i+1} 的 parts 类型不匹配:\n"
                    f"  期望: {expected_types}\n"
                    f"  实际: {actual_types}"
                )
    
    return True, ""


def match_span_to_expected(span, expected_span_def, parent_map):
    """匹配实际 span 与期望的 span 定义。
    
    Args:
        span: 实际的 span 对象
        expected_span_def: 期望的 span 定义字典
        parent_map: 父 span 的映射 {operation: span}
        
    Returns:
        (bool, str): (是否匹配, 错误消息)
    """
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    
    attrs = dict(span.attributes or {})
    
    # 验证名称模式
    if "name_pattern" in expected_span_def:
        if expected_span_def["name_pattern"] not in span.name:
            return False, f"Span 名称不匹配: 期望包含 '{expected_span_def['name_pattern']}', 实际: '{span.name}'"
    
    # 验证 operation
    if "operation" in expected_span_def:
        if GenAIAttributes.GEN_AI_OPERATION_NAME not in attrs:
            return False, f"缺少 gen_ai.operation.name 属性"
        if attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] != expected_span_def["operation"]:
            return False, (
                f"gen_ai.operation.name 不匹配: "
                f"期望 '{expected_span_def['operation']}', "
                f"实际 '{attrs[GenAIAttributes.GEN_AI_OPERATION_NAME]}'"
            )
    
    # 验证父 span
    if "parent" in expected_span_def:
        expected_parent = expected_span_def["parent"]
        if expected_parent is None:
            if span.parent is not None:
                return False, f"期望根 span，但实际有 parent"
        else:
            if span.parent is None:
                return False, f"期望 parent 是 '{expected_parent}'，但实际是根 span"
            elif expected_parent in parent_map:
                expected_parent_span = parent_map[expected_parent]
                if span.parent.span_id != expected_parent_span.context.span_id:
                    return False, (
                        f"Parent span 不匹配: "
                        f"期望 '{expected_parent}' (span_id={expected_parent_span.context.span_id}), "
                        f"实际 span_id={span.parent.span_id}"
                    )
    
    # 验证精确匹配的属性
    if "attributes" in expected_span_def:
        for attr_key, expected_value in expected_span_def["attributes"].items():
            if attr_key not in attrs:
                return False, f"缺少属性: {attr_key}"
            if attrs[attr_key] != expected_value:
                return False, (
                    f"属性 {attr_key} 不匹配: "
                    f"期望 {expected_value!r}, 实际 {attrs[attr_key]!r}"
                )
    
    # 验证包含匹配的属性（值包含指定字符串或列表）
    if "attributes_contains" in expected_span_def:
        for attr_key, expected_value in expected_span_def["attributes_contains"].items():
            if attr_key not in attrs:
                return False, f"缺少属性: {attr_key}"
            attr_value = attrs[attr_key]
            
            # 如果期望值是列表，验证每个元素
            if isinstance(expected_value, list):
                attr_value_str = str(attr_value)
                for expected_item in expected_value:
                    if expected_item not in attr_value_str:
                        return False, (
                            f"属性 {attr_key} 不包含期望内容: "
                            f"期望包含 '{expected_item}', 实际: {attr_value_str[:200]}..."
                        )
            else:
                # 字符串匹配
                attr_value_str = str(attr_value)
                if expected_value not in attr_value_str:
                    return False, (
                        f"属性 {attr_key} 不包含期望内容: "
                        f"期望包含 '{expected_value}', 实际: {attr_value_str[:200]}..."
                    )
    
    # 验证消息结构（如果定义了 message_structure）
    if "message_structure" in expected_span_def:
        for attr_key, expected_structure in expected_span_def["message_structure"].items():
            if attr_key not in attrs:
                return False, f"缺少属性: {attr_key}"
            is_valid, error_msg = validate_message_structure(attrs[attr_key], expected_structure)
            if not is_valid:
                return False, f"属性 {attr_key} 结构验证失败: {error_msg}"
    
    # 验证错误状态
    if "is_error" in expected_span_def:
        expected_is_error = expected_span_def["is_error"]
        actual_status = span.status.status_code.value
        if expected_is_error:
            # 期望错误状态（ERROR = 2）
            if actual_status != 2:
                return False, (
                    f"期望错误状态，但实际状态码是 {actual_status}"
                )
        else:
            # 期望成功状态（OK = 1 或 UNSET = 0）
            if actual_status not in [0, 1]:
                return False, (
                    f"期望成功状态，但实际状态码是 {actual_status}"
                )
    
    return True, ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(TEST_CASE_1_FOO_SH_COMMAND, id="foo_sh_command"),
        pytest.param(TEST_CASE_2_ECHO_COMMAND, id="echo_command"),
        pytest.param(TEST_CASE_3_PRETOOLUSE, id="pretooluse_hook"),
    ],
)
async def test_all_spans_match_expected(
    instrument, span_exporter, tracer_provider, test_case
):
    """验证实际生成的 spans 与 expected_spans 完全匹配。
    
    这个测试验证：
    1. 生成的 spans 数量与 expected_spans 一致
    2. 每个 span 的名称、操作类型、父 span 都匹配
    3. 每个 span 的属性都完全匹配 expected_spans 中的定义
    4. Span 的层次结构正确
    """
    from opentelemetry.instrumentation.claude_agent_sdk.patch import (  # noqa: PLC0415
        _process_agent_invocation_stream,
    )
    from opentelemetry.semconv._incubating.attributes import (  # noqa: PLC0415
        gen_ai_attributes as GenAIAttributes,
    )
    from opentelemetry.util.genai.extended_handler import (  # noqa: PLC0415
        ExtendedTelemetryHandler,
    )

    handler = ExtendedTelemetryHandler(tracer_provider=tracer_provider)
    expected_spans = test_case.get("expected_spans", [])

    if not expected_spans:
        pytest.skip(f"测试用例 '{test_case['description']}' 没有定义 expected_spans")

    mock_stream = create_mock_stream_from_messages(test_case["messages"])
    
    async for _ in _process_agent_invocation_stream(
        wrapped_stream=mock_stream,
        handler=handler,
        model="qwen-plus",
        prompt=test_case["prompt"],
    ):
        pass

    spans = span_exporter.get_finished_spans()
    
    # 构建父 span 映射（按 operation 名称）
    parent_map = {}
    for span in spans:
        attrs = dict(span.attributes or {})
        if GenAIAttributes.GEN_AI_OPERATION_NAME in attrs:
            operation = attrs[GenAIAttributes.GEN_AI_OPERATION_NAME]
            parent_map[operation] = span
    
    # 验证 spans 数量
    assert len(spans) == len(expected_spans), (
        f"生成的 spans 数量不匹配: "
        f"期望 {len(expected_spans)} 个，实际 {len(spans)} 个"
    )
    
    # 按 operation 类型分组 spans
    spans_by_operation = {}
    for span in spans:
        attrs = dict(span.attributes or {})
        if GenAIAttributes.GEN_AI_OPERATION_NAME in attrs:
            operation = attrs[GenAIAttributes.GEN_AI_OPERATION_NAME]
            if operation not in spans_by_operation:
                spans_by_operation[operation] = []
            spans_by_operation[operation].append(span)
    
    # 验证每个期望的 span
    operation_index_map = {}  # 跟踪每个 operation 类型的索引
    for i, expected_span_def in enumerate(expected_spans):
        expected_operation = expected_span_def.get("operation")
        
        if expected_operation not in spans_by_operation:
            pytest.fail(
                f"期望的 span #{i+1} (operation={expected_operation}) 不存在于生成的 spans 中"
            )
        
        # 获取对应 operation 类型的 span（按顺序）
        if expected_operation not in operation_index_map:
            operation_index_map[expected_operation] = 0
        
        operation_index = operation_index_map[expected_operation]
        if operation_index >= len(spans_by_operation[expected_operation]):
            pytest.fail(
                f"期望的 span #{i+1} (operation={expected_operation}) "
                f"超出了该类型的实际数量 ({len(spans_by_operation[expected_operation])})"
            )
        
        actual_span = spans_by_operation[expected_operation][operation_index]
        operation_index_map[expected_operation] += 1
        
        # 匹配 span
        is_match, error_msg = match_span_to_expected(actual_span, expected_span_def, parent_map)
        assert is_match, (
            f"Span #{i+1} (operation={expected_operation}) 不匹配:\n"
            f"  {error_msg}\n"
            f"  Span 名称: {actual_span.name}\n"
            f"  Span 属性: {dict(actual_span.attributes or {})}"
        )
    
    print(f"\n✅ 测试用例 '{test_case['description']}' 的所有 spans 验证通过")
    print(f"  - 总 spans 数: {len(spans)}")
    for operation, span_list in spans_by_operation.items():
        print(f"  - {operation} spans: {len(span_list)} 个")
