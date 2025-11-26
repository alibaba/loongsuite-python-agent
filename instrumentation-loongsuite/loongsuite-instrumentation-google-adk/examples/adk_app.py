"""
Google ADK Demo Application
演示 Agent、Tool、LLM 的集成使用
"""
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from datetime import datetime
import json


# 定义工具函数
def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """
    计算数学表达式
    
    Args:
        expression: 数学表达式，例如 "2 + 3"
    """
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


# 创建 Tools
time_tool = FunctionTool(
    name="get_current_time",
    description="获取当前时间",
    func=get_current_time
)

calculator_tool = FunctionTool(
    name="calculate",
    description="计算数学表达式，支持加减乘除等基本运算",
    func=calculate
)

# 创建 Agent
math_assistant = Agent(
    name="math_assistant",
    description="一个能够执行数学计算和查询时间的智能助手",
    tools=[time_tool, calculator_tool],
    model="gemini-1.5-flash",  # 或使用其他支持的模型
    instruction="你是一个专业的数学助手，可以帮助用户进行计算和查询时间。"
)

# 创建 Runner
runner = Runner(app_name="math_assistant_demo", agent=math_assistant)


def main():
    """主函数"""
    print("Google ADK Demo - Math Assistant")
    print("=" * 50)
    
    # 测试场景 1：计算
    print("\n场景 1：数学计算")
    result1 = runner.run("帮我计算 (125 + 375) * 2 的结果")
    print(f"用户：帮我计算 (125 + 375) * 2 的结果")
    print(f"助手：{result1}")
    
    # 测试场景 2：查询时间
    print("\n场景 2：查询时间")
    result2 = runner.run("现在几点了？")
    print(f"用户：现在几点了？")
    print(f"助手：{result2}")
    
    # 测试场景 3：组合使用
    print("\n场景 3：组合使用")
    result3 = runner.run("现在几点了？顺便帮我算一下 100 / 4")
    print(f"用户：现在几点了？顺便帮我算一下 100 / 4")
    print(f"助手：{result3}")
    
    print("\n" + "=" * 50)
    print("Demo 完成")


if __name__ == "__main__":
    main()

