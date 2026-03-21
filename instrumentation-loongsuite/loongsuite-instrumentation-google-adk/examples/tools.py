#!/usr/bin/env python3

# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
工具函数定义
包含各种类型的工具函数供 Agent 使用
"""

import math
import random
from datetime import datetime
from typing import Any, Dict, List


def get_current_time() -> str:
    """
    获取当前时间

    Returns:
        当前时间的字符串表示
    """
    return f"当前时间是: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"


def calculate_math(expression: str) -> str:
    """
    数学计算工具函数

    Args:
        expression: 数学表达式字符串

    Returns:
        计算结果的字符串
    """
    try:
        # 安全的数学表达式计算
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update(
            {"abs": abs, "round": round, "pow": pow, "min": min, "max": max}
        )

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"🔢 计算结果：{expression} = {result}"
    except Exception as e:
        return f"❌ 计算错误：{str(e)}"


def roll_dice(sides: int = 6) -> int:
    """
    掷骰子工具函数

    Args:
        sides: 骰子面数，默认为6

    Returns:
        掷骰子的结果
    """
    if sides < 2:
        sides = 6
    return random.randint(1, sides)


def check_prime_numbers(numbers: List[int]) -> Dict[str, Any]:
    """
    检查数字是否为质数

    Args:
        numbers: 要检查的数字列表

    Returns:
        包含检查结果的字典
    """

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

    results = {}
    primes = []
    non_primes = []

    for num in numbers:
        if is_prime(num):
            primes.append(num)
        else:
            non_primes.append(num)
        results[str(num)] = is_prime(num)

    return {
        "results": results,
        "primes": primes,
        "non_primes": non_primes,
        "summary": f"在 {numbers} 中，质数有: {primes}，非质数有: {non_primes}",
    }


def get_weather_info(city: str) -> str:
    """
    获取天气信息工具函数（模拟）

    Args:
        city: 城市名称

    Returns:
        天气信息字符串
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴朗，温度 15°C，湿度 45%，微风",
        "上海": "多云，温度 18°C，湿度 60%，东南风",
        "深圳": "小雨，温度 25°C，湿度 80%，南风",
        "杭州": "阴天，温度 20°C，湿度 55%，西北风",
        "广州": "晴朗，温度 28°C，湿度 65%，东风",
    }

    weather = weather_data.get(city, f"{city}的天气信息暂时无法获取")
    return f"📍 {city}的天气：{weather}"


def search_web(query: str) -> str:
    """
    网络搜索工具函数（模拟）

    Args:
        query: 搜索查询

    Returns:
        搜索结果字符串
    """
    # 模拟搜索结果
    mock_results = {
        "人工智能": "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "机器学习": "机器学习是人工智能的一个分支，是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。",
        "深度学习": "深度学习是机器学习的一个分支，它基于人工神经网络，利用多层非线性变换对数据进行特征提取和转换。",
        "自然语言处理": "自然语言处理是计算机科学领域与人工智能领域中的一个重要方向，它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。",
    }

    for key, value in mock_results.items():
        if key in query:
            return value

    return f"🔍 关于'{query}'的搜索结果：这是模拟的搜索结果，实际应用中会连接真实的搜索引擎API。"


def translate_text(text: str, target_language: str = "en") -> str:
    """
    文本翻译工具函数（模拟）

    Args:
        text: 要翻译的文本
        target_language: 目标语言代码

    Returns:
        翻译结果字符串
    """
    # 模拟翻译结果
    translations = {
        "你好": "Hello",
        "谢谢": "Thank you",
        "再见": "Goodbye",
        "人工智能": "Artificial Intelligence",
        "机器学习": "Machine Learning",
    }

    if target_language.lower() == "en":
        return translations.get(text, f"Translated: {text}")
    else:
        return f"翻译到{target_language}：{text}"


# 导出所有工具函数
__all__ = [
    "get_current_time",
    "calculate_math",
    "roll_dice",
    "check_prime_numbers",
    "get_weather_info",
    "search_web",
    "translate_text",
]
