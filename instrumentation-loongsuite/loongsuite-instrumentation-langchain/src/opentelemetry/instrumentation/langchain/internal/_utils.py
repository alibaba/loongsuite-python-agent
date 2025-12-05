from __future__ import annotations

import ast
import logging
import sys
from os import environ
from typing import Any

OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT = (
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
)

logger = logging.getLogger(__name__)


def recursive_size(obj: Any, max_size: int = 10240) -> int:
    """递归计算对象大小，超过阈值时快速返回"""
    total_size = 0
    if isinstance(obj, dict):
        total_size += sys.getsizeof(obj)
        if total_size > max_size:
            return total_size
        for key, value in obj.items():
            total_size += recursive_size(
                key, max_size - total_size
            ) + recursive_size(value, max_size - total_size)
            if total_size > max_size:
                return total_size
    elif isinstance(obj, list):
        total_size += sys.getsizeof(obj)
        if total_size > max_size:
            return total_size
        for item in obj:
            total_size += recursive_size(item, max_size - total_size)
            if total_size > max_size:
                return total_size
    else:
        total_size += sys.getsizeof(obj)
    return total_size


def _is_base64_image(item: Any) -> bool:
    """检查是否为base64编码的图片数据"""
    if not isinstance(item, dict):
        return False
    if not isinstance(item.get("image_url"), dict):
        return False
    if "data:image/" not in item.get("image_url", {}).get("url", ""):
        return False
    return True


def _filter_base64_images(obj: Any) -> Any:
    """递归过滤掉base64图片数据，保留其他信息"""
    # 使用内存大小检测 - 如果数据量不大，直接返回
    # 256x256 图片 base64 约 12K 字符长度，这里设置阈值为 10KB
    if recursive_size(obj) < 10240:  # 10KB
        return obj

    if isinstance(obj, list):
        filtered_list = []
        for item in obj:
            if isinstance(item, str) and "data:image/" in item:
                # 处理字符串中包含base64图片数据的情况
                # 例如: "Human: [{'type': 'text', 'text': '简述这个图片'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,...'}}]"
                start_idx = item.find("[")
                end_idx = item.rfind("]")

                if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                    filtered_list.append(item)
                    continue

                try:
                    filtered_obj = item[start_idx : end_idx + 1]
                    # 解析列表
                    parsed_list = ast.literal_eval(filtered_obj)
                    if isinstance(parsed_list, list):
                        # 递归处理解析后的列表
                        filtered_parsed_list = _filter_base64_images(
                            parsed_list
                        )
                        # 替换原字符串中的列表
                        filtered_item = (
                            item[:start_idx]
                            + str(filtered_parsed_list)
                            + item[end_idx + 1 :]
                        )
                        filtered_list.append(filtered_item)
                    else:
                        filtered_list.append(item)
                except Exception:
                    # 如果解析失败，保持原样
                    filtered_list.append(item)
            elif _is_base64_image(item):
                # 保留图片信息但不包含base64数据
                filtered_item = {
                    "type": item.get("type", "image_url"),
                    "image_url": {"url": "BASE64_IMAGE_DATA_FILTERED"},
                }
                filtered_list.append(filtered_item)
            else:
                filtered_list.append(_filter_base64_images(item))
        return filtered_list
    elif isinstance(obj, dict):
        filtered_dict = {}
        for key, value in obj.items():
            if _is_base64_image(value):
                # 如果字典值本身就是base64图片
                filtered_dict[key] = {
                    "type": value.get("type", "image_url"),
                    "image_url": {"url": "BASE64_IMAGE_DATA_FILTERED"},
                }
            else:
                filtered_dict[key] = _filter_base64_images(value)
        return filtered_dict
    else:
        return obj


max_content_length = 4 * 1024


def process_content(content: str | None) -> str:
    if is_capture_content_enabled():
        if content is not None and len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        return content
    elif content is None:
        return "<0size>"
    else:
        return to_size(content)


def to_size(content: str) -> str:
    if content is None:
        return "<0size>"
    size = len(content)
    return f"<{size}size>"


def is_capture_content_enabled() -> bool:
    capture_content = environ.get(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, "true"
    )
    return is_true_value(capture_content)


def is_true_value(value) -> bool:
    return value.lower() in {"1", "y", "yes", "true"}
