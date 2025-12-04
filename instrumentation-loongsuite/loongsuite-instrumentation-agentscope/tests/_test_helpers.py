# -*- coding: utf-8 -*-
"""Test Utility Functions"""

from typing import Any, Dict, List, Optional

from opentelemetry.sdk._logs import LogRecord
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def remove_none_values(d: Dict[str, Any]) -> Dict[str, Any]:
    """移除字典中值为 None 的键值对"""
    return {k: v for k, v in d.items() if v is not None}


def assert_span_attributes(
    span: ReadableSpan,
    expected_operation: str,
    expected_model: Optional[str] = None,
    expected_provider: Optional[str] = None,
    expected_response_id: Optional[str] = None,
    expected_response_model: Optional[str] = None,
    expected_input_tokens: Optional[int] = None,
    expected_output_tokens: Optional[int] = None,
    expected_error_type: Optional[str] = None,
):
    """
    验证 span 的属性

    Args:
        span: 要验证的 span
        expected_operation: 期望的操作名称 (e.g., "chat", "embed")
        expected_model: 期望的请求模型名称
        expected_provider: 期望的提供商名称
        expected_response_id: 期望的响应 ID
        expected_response_model: 期望的响应模型名称
        expected_input_tokens: 期望的输入 token 数
        expected_output_tokens: 期望的输出 token 数
        expected_error_type: 期望的错误类型
    """
    attrs = span.attributes

    # 验证操作名称
    assert attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == expected_operation

    # 验证模型
    if expected_model:
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == expected_model

    # 验证提供商
    if expected_provider:
        assert attrs.get("gen_ai.provider.name") == expected_provider

    # 验证响应 ID
    if expected_response_id:
        assert attrs.get(GenAIAttributes.GEN_AI_RESPONSE_ID) == expected_response_id

    # 验证响应模型
    if expected_response_model:
        assert attrs.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL) == expected_response_model

    # 验证 token 使用
    if expected_input_tokens is not None:
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == expected_input_tokens

    if expected_output_tokens is not None:
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == expected_output_tokens

    # 验证错误类型
    if expected_error_type:
        assert attrs.get(ErrorAttributes.ERROR_TYPE) == expected_error_type


def assert_log_parent(log: LogRecord, span: ReadableSpan):
    """验证日志记录是否正确关联到 span"""
    assert log.trace_id == span.context.trace_id
    assert log.span_id == span.context.span_id
    assert log.trace_flags == span.context.trace_flags


def find_spans_by_operation(spans: List[ReadableSpan], operation: str) -> List[ReadableSpan]:
    """根据操作名称查找 spans"""
    return [
        span for span in spans
        if span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == operation
    ]


def find_spans_by_name_prefix(spans: List[ReadableSpan], prefix: str) -> List[ReadableSpan]:
    """根据 span 名称前缀查找 spans"""
    return [span for span in spans if span.name.startswith(prefix)]


def assert_metrics_recorded(
    metric_reader,
    expected_metric_name: str,
    expected_attributes: Optional[Dict[str, Any]] = None,
    min_value: Optional[float] = None,
):
    """
    验证 metrics 是否被正确记录

    Args:
        metric_reader: metric reader 实例
        expected_metric_name: 期望的 metric 名称
        expected_attributes: 期望的 metric 属性
        min_value: 期望的最小值
    """
    metrics = metric_reader.get_metrics_data().resource_metrics
    assert len(metrics) > 0, "No metrics recorded"

    metric_data = metrics[0].scope_metrics[0].metrics
    target_metric = next(
        (m for m in metric_data if m.name == expected_metric_name),
        None,
    )

    assert target_metric is not None, f"Metric {expected_metric_name} not found"

    # 验证至少有一个数据点
    data_points = target_metric.data.data_points
    assert len(data_points) > 0, "No data points in metric"

    # 如果提供了期望的属性，验证它们
    if expected_attributes:
        found = False
        for point in data_points:
            if all(
                point.attributes.get(k) == v
                for k, v in expected_attributes.items()
            ):
                found = True
                # 如果提供了最小值，验证它
                if min_value is not None:
                    if hasattr(point, "sum"):
                        assert point.sum >= min_value
                    elif hasattr(point, "value"):
                        assert point.value >= min_value
                break

        assert found, f"No data point with attributes {expected_attributes}"


def get_event_attributes(log: LogRecord) -> Dict[str, Any]:
    """获取日志事件的属性"""
    if hasattr(log, "attributes") and log.attributes:
        return dict(log.attributes)
    return {}


def assert_event_in_logs(
    logs: List[LogRecord],
    event_name: str,
    expected_attributes: Optional[Dict[str, Any]] = None,
    parent_span: Optional[ReadableSpan] = None,
):
    """
    验证日志中是否包含特定事件

    Args:
        logs: 日志列表
        event_name: 期望的事件名称
        expected_attributes: 期望的事件属性
        parent_span: 期望关联的父 span
    """
    matching_logs = [
        log for log in logs
        if hasattr(log, "attributes") and
        log.attributes.get("event.name") == event_name
    ]

    assert len(matching_logs) > 0, f"Event {event_name} not found in logs"

    # 如果提供了期望的属性，验证它们
    if expected_attributes:
        found = False
        for log in matching_logs:
            attrs = get_event_attributes(log)
            if all(attrs.get(k) == v for k, v in expected_attributes.items()):
                found = True
                # 如果提供了父 span，验证关联关系
                if parent_span:
                    assert_log_parent(log, parent_span)
                break

        assert found, f"No event with attributes {expected_attributes}"
    elif parent_span:
        # 即使没有提供属性，也验证至少一个日志与父 span 关联
        assert_log_parent(matching_logs[0], parent_span)


def print_span_tree(spans: List[ReadableSpan], indent: int = 0):
    """打印 span 树结构，用于调试"""
    # 按照开始时间排序
    sorted_spans = sorted(spans, key=lambda s: s.start_time)

    for span in sorted_spans:
        print("  " * indent + f"- {span.name}")
        print("  " * indent + f"  Operation: {span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}")
        print("  " * indent + f"  Model: {span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)}")
        print("  " * indent + f"  Duration: {(span.end_time - span.start_time) / 1e9:.3f}s")

        # 打印子 spans（如果有的话）
        child_spans = [
            s for s in spans
            if hasattr(s, "parent") and s.parent and
            s.parent.span_id == span.context.span_id
        ]
        if child_spans:
            print_span_tree(child_spans, indent + 1)

