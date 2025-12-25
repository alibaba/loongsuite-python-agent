#!/usr/bin/env python3
"""
工具使用示例 (HTTP 服务版本)
展示如何在 ADK Agent 中使用各种工具函数并部署为 HTTP 服务
"""

import asyncio
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查环境变量
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("❌ 请设置 DASHSCOPE_API_KEY 环境变量:")
    print("   export DASHSCOPE_API_KEY='your-dashscope-api-key'")
    print("🔗 获取地址: https://dashscope.console.aliyun.com/")
    sys.exit(1)

try:
    # 导入 ADK 相关模块
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import (
        InMemorySessionService,
    )
    from google.adk.tools import FunctionTool
    from google.genai import types
except ImportError as e:
    print(f"❌ 导入 ADK 模块失败: {e}")
    print("📦 请确保已正确安装 Google ADK:")
    print("   pip install google-adk")
    sys.exit(1)

# 导入自定义工具
try:
    from tools import (
        calculate_math,
        check_prime_numbers,
        get_current_time,
        get_weather_info,
        roll_dice,
        search_web,
        translate_text,
    )
except ImportError as e:
    print(f"❌ 导入自定义工具失败: {e}")
    sys.exit(1)

# 配置阿里云百炼模型
DASHSCOPE_CONFIG = {
    "model": "dashscope/qwen-plus",
    "api_key": api_key,
    "temperature": 0.7,
    "max_tokens": 1000,
}

# 设置LiteLLM的环境变量
os.environ["DASHSCOPE_API_KEY"] = api_key

# ==================== 数据模型定义 ====================


class ToolsRequest(BaseModel):
    """工具使用请求模型"""

    task: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "default_user"


class ApiResponse(BaseModel):
    """API 响应模型"""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str
    session_id: Optional[str] = None


def extract_content_text(content) -> str:
    """
    从 Content 对象中提取文本内容

    Args:
        content: Content 对象，包含 parts 列表

    Returns:
        提取到的文本内容
    """
    if not content:
        return ""

    # 如果 content 是字符串，直接返回
    if isinstance(content, str):
        return content

    # 如果 content 有 parts 属性
    if hasattr(content, "parts") and content.parts:
        text_parts = []
        for part in content.parts:
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)
        return "".join(text_parts)

    # 如果 content 有 text 属性
    if hasattr(content, "text") and content.text:
        return content.text

    # 如果都没有，返回空字符串
    return ""


async def create_agent() -> LlmAgent:
    """创建带工具的 LLM Agent 实例"""

    # 创建 LiteLlm 模型实例
    dashscope_model = LiteLlm(
        model=DASHSCOPE_CONFIG["model"],
        api_key=DASHSCOPE_CONFIG["api_key"],
        temperature=DASHSCOPE_CONFIG["temperature"],
        max_tokens=DASHSCOPE_CONFIG["max_tokens"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 创建工具
    time_tool = FunctionTool(func=get_current_time)
    calc_tool = FunctionTool(func=calculate_math)
    dice_tool = FunctionTool(func=roll_dice)
    prime_tool = FunctionTool(func=check_prime_numbers)
    weather_tool = FunctionTool(func=get_weather_info)
    search_tool = FunctionTool(func=search_web)
    translate_tool = FunctionTool(func=translate_text)

    # 创建 Agent
    agent = LlmAgent(
        name="tools_assistant",
        model=dashscope_model,
        instruction="""你是一个功能丰富的智能助手，可以使用多种工具来帮助用户。

你可以使用的工具包括：
1. 🕐 get_current_time - 获取当前时间
2. 🧮 calculate_math - 进行数学计算
3. 🎲 roll_dice - 掷骰子
4. 🔢 check_prime_numbers - 检查质数
5. 🌤️ get_weather_info - 获取天气信息
6. 🔍 search_web - 搜索网络信息
7. 🌍 translate_text - 翻译文本

使用原则：
- 用中文与用户交流
- 对用户友好和专业
- 当需要使用工具时，主动调用相应的工具函数
- 基于工具返回的结果给出完整回答
- 如果用户请求的功能没有对应工具，要诚实说明""",
        description="一个可以使用多种工具的智能助手",
        tools=[
            time_tool,
            calc_tool,
            dice_tool,
            prime_tool,
            weather_tool,
            search_tool,
            translate_tool,
        ],
    )

    return agent


# ==================== 服务实现 ====================

# 全局变量存储服务组件
session_service = None
runner = None
agent = None


async def initialize_services():
    """初始化服务组件"""
    global session_service, runner, agent

    if session_service is None:
        logger.info("🔧 初始化服务组件...")
        session_service = InMemorySessionService()
        agent = await create_agent()
        runner = Runner(
            app_name="tools_agent_demo",
            agent=agent,
            session_service=session_service,
        )
        logger.info("✅ 服务组件初始化完成")


async def run_conversation(
    user_input: str, user_id: str, session_id: str = "default_session"
) -> str:
    """运行对话并返回回复"""
    try:
        # 初始化服务
        await initialize_services()

        # 直接创建新会话，不检查是否存在
        logger.info(f"创建新会话: {session_id}")
        session = await session_service.create_session(
            app_name="tools_agent_demo", user_id=user_id, session_id=session_id
        )

        logger.info(f"使用会话: {session.id}")

        # 创建用户消息
        user_message = types.Content(
            role="user", parts=[types.Part(text=user_input)]
        )

        # 运行对话
        events = []
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=user_message
        ):
            events.append(event)

        # 获取回复
        for event in events:
            if hasattr(event, "content") and event.content:
                # 提取 Content 对象中的文本
                content_text = extract_content_text(event.content)
                if content_text:
                    logger.info(f"收到回复: {content_text[:100]}...")
                    return content_text

        logger.warning("未收到有效回复")
        return "抱歉，我没有收到有效的回复。"

    except Exception as e:
        logger.error(f"处理消息时出错: {e}")
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"处理消息失败: {str(e)}")


# ==================== FastAPI 应用 ====================

# 创建 FastAPI 应用
app = FastAPI(
    title="ADK 工具使用 Agent HTTP 服务",
    description="基于 Google ADK 框架的工具使用 Agent HTTP 服务",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化服务"""
    logger.info("🚀 启动工具使用 Agent HTTP 服务...")
    await initialize_services()
    logger.info("✅ 服务启动完成")


@app.get("/")
async def root():
    """服务状态检查"""
    return ApiResponse(
        success=True,
        message="工具使用 Agent HTTP 服务运行正常",
        data={
            "service": "ADK Tools Agent HTTP Service",
            "version": "1.0.0",
            "available_tools": [
                "get_current_time: 获取当前时间",
                "calculate_math: 数学计算",
                "roll_dice: 投骰子",
                "check_prime_numbers: 质数检查",
                "get_weather_info: 天气信息",
                "search_web: 网络搜索",
                "translate_text: 文本翻译",
            ],
            "capabilities": [
                "工具自动调用",
                "多种实用功能",
                "智能任务处理",
                "结果整合分析",
            ],
        },
        timestamp=datetime.now().isoformat(),
    )


@app.post("/tools")
async def tools(request: ToolsRequest):
    """工具使用任务处理接口"""
    try:
        session_id = (
            request.session_id
            or f"tools_{request.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        response = await run_conversation(
            user_input=request.task,
            user_id=request.user_id or "default_user",
            session_id=session_id,
        )

        return ApiResponse(
            success=True,
            message="工具任务处理成功",
            data={"task": request.task, "response": response},
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
        )

    except Exception as e:
        logger.error(f"工具任务处理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"全局异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": f"服务器内部错误: {str(exc)}",
            "timestamp": datetime.now().isoformat(),
        },
    )


def main():
    """主函数 - 启动 HTTP 服务"""
    print("🚀 ADK 工具使用 Agent HTTP 服务")
    print("=" * 50)
    print("🔑 API Key 已设置")
    print("🔧 可用工具:")
    print("   1. get_current_time - 获取当前时间")
    print("   2. calculate_math - 数学计算")
    print("   3. roll_dice - 投骰子")
    print("   4. check_prime_numbers - 质数检查")
    print("   5. get_weather_info - 天气信息")
    print("   6. search_web - 网络搜索")
    print("   7. translate_text - 文本翻译")
    print("=" * 50)
    print("\n📡 HTTP 接口说明:")
    print("  GET  / - 服务状态检查")
    print("  POST /tools - 工具使用任务处理接口")
    print("\n💡 示例请求:")
    print("  curl -X POST http://localhost:8000/tools \\")
    print("       -H 'Content-Type: application/json' \\")
    print('       -d \'{"task": "现在几点了？"}\'')
    print("\n🌐 启动服务...")

    # 启动 FastAPI 服务
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )


# 保留原有的命令行测试功能
async def run_test_conversation():
    """运行测试对话"""
    print("🚀 启动工具使用示例")
    print("=" * 50)
    print("🔑 API Key 已设置")
    print(f"🤖 模型: {DASHSCOPE_CONFIG['model']}")
    print("=" * 50)

    try:
        # 初始化服务
        await initialize_services()
        print("✅ Agent 初始化成功")

        # 示例对话
        test_inputs = [
            "现在几点了？",
            "计算 123 乘以 456",
            "掷一个六面骰子",
            "检查 17, 25, 29, 33 是否为质数",
            "北京的天气怎么样？",
            "搜索人工智能的定义",
            "翻译'你好'成英文",
        ]

        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n💬 测试 {i}: {user_input}")
            print("-" * 30)

            response = await run_conversation(
                user_input, "default_user", session_id
            )
            print(f"🤖 回复: {response}")

            # 添加延迟避免请求过快
            await asyncio.sleep(1)

        print("\n✅ 所有测试已完成，程序结束")

    except Exception as e:
        print(f"❌ 运行失败: {e}")
        logger.exception("运行失败")


def run_test():
    """运行测试对话"""
    asyncio.run(run_test_conversation())


if __name__ == "__main__":
    # 检查是否要运行测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_test()
    else:
        # 启动 HTTP 服务
        try:
            main()
        except KeyboardInterrupt:
            print("\n👋 服务已停止")
        except Exception as e:
            print(f"❌ 服务启动失败: {e}")
            logger.exception("Service startup failed")
