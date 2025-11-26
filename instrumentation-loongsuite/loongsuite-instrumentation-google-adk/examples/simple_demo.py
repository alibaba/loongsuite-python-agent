#!/usr/bin/env python3
"""
Google ADK 工具使用精简示例
展示如何在 ADK Agent 中使用工具函数
"""

import os
import sys
import asyncio
import math
import random
from datetime import datetime
from typing import List, Dict, Any

# ==================== 工具函数定义 ====================

def get_current_time() -> str:
    """获取当前时间"""
    return f"当前时间是: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"

def calculate_math(expression: str) -> str:
    """数学计算工具"""
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names.update({"abs": abs, "round": round, "pow": pow, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"

def roll_dice(sides: int = 6) -> int:
    """掷骰子"""
    if sides < 2:
        sides = 6
    return random.randint(1, sides)

def check_prime_numbers(numbers: List[int]) -> Dict[str, Any]:
    """检查质数"""
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [num for num in numbers if is_prime(num)]
    non_primes = [num for num in numbers if not is_prime(num)]
    
    return {
        "primes": primes,
        "non_primes": non_primes,
        "summary": f"质数: {primes}, 非质数: {non_primes}"
    }

def get_weather_info(city: str) -> str:
    """获取天气信息（模拟）"""
    weather_data = {
        "北京": "晴朗，温度 15°C",
        "上海": "多云，温度 18°C",
        "深圳": "小雨，温度 25°C",
        "杭州": "阴天，温度 20°C"
    }
    weather = weather_data.get(city, f"{city}的天气信息暂时无法获取")
    return f"{city}的天气：{weather}"

# ==================== ADK Agent 设置 ====================

async def create_agent():
    """创建带工具的 ADK Agent"""
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.tools import FunctionTool
    
    # 检查环境变量
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ 请设置 DASHSCOPE_API_KEY 环境变量")
        print("   export DASHSCOPE_API_KEY='your-api-key'")
        sys.exit(1)
    
    # 创建模型
    model = LiteLlm(
        model="dashscope/qwen-plus",
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 创建工具
    tools = [
        FunctionTool(func=get_current_time),
        FunctionTool(func=calculate_math),
        FunctionTool(func=roll_dice),
        FunctionTool(func=check_prime_numbers),
        FunctionTool(func=get_weather_info)
    ]
    
    # 创建 Agent
    agent = LlmAgent(
        name="simple_assistant",
        model=model,
        instruction="""你是一个智能助手，可以使用多种工具帮助用户。
可用工具：
1. get_current_time - 获取当前时间
2. calculate_math - 数学计算
3. roll_dice - 掷骰子
4. check_prime_numbers - 检查质数
5. get_weather_info - 获取天气

用中文友好地与用户交流，根据需要调用工具。""",
        description="一个简单的工具助手",
        tools=tools
    )
    
    return agent

async def run_conversation(user_input: str) -> str:
    """运行对话并返回回复"""
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.genai import types
    
    # 初始化服务
    session_service = InMemorySessionService()
    agent = await create_agent()
    runner = Runner(
        app_name="simple_demo",
        agent=agent,
        session_service=session_service
    )
    
    # 创建会话
    session = await session_service.create_session(
        app_name="simple_demo",
        user_id="demo_user",
        session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # 创建用户消息
    user_message = types.Content(
        role="user",
        parts=[types.Part(text=user_input)]
    )
    
    # 运行对话并收集事件
    events = []
    async for event in runner.run_async(
        user_id="demo_user",
        session_id=session.id,
        new_message=user_message
    ):
        events.append(event)
    
    # 提取回复文本
    for event in events:
        if hasattr(event, 'content') and event.content:
            if hasattr(event.content, 'parts') and event.content.parts:
                text_parts = [part.text for part in event.content.parts if hasattr(part, 'text') and part.text]
                if text_parts:
                    return ''.join(text_parts)
    
    return "未收到有效回复"

# ==================== 主程序 ====================

async def main():
    """主函数"""
    print("🚀 Google ADK 工具使用精简示例")
    print("=" * 50)
    
    # 测试用例
    test_cases = [
        "现在几点了？",
        "计算 123 乘以 456",
        "掷一个六面骰子",
        "检查 17, 25, 29 是否为质数",
        "北京的天气怎么样？"
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"\n💬 测试 {i}: {user_input}")
        print("-" * 40)
        
        try:
            response = await run_conversation(user_input)
            print(f"🤖 回复: {response}")
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        # 避免请求过快
        if i < len(test_cases):
            await asyncio.sleep(1)
    
    print("\n✅ 所有测试完成")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序已停止")
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


