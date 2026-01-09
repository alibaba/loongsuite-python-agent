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

"""Common utility functions for DashScope instrumentation."""

from __future__ import annotations

from typing import Any, Optional


def _get_parameter(
    kwargs: dict, param_name: str, parameters: Optional[dict] = None
) -> Optional[Any]:
    """Get parameter from kwargs or parameters dict.

    Checks kwargs first (direct arguments), then kwargs["parameters"] if provided.

    Args:
        kwargs: Method kwargs
        param_name: Parameter name to extract
        parameters: Optional parameters dict (if None, will extract from kwargs.get("parameters"))

    Returns:
        Parameter value if found, None otherwise
    """
    # Check kwargs first (direct arguments)
    if param_name in kwargs:
        return kwargs[param_name]

    # Check parameters dict if provided
    if parameters is None:
        parameters = kwargs.get("parameters", {})
    if isinstance(parameters, dict) and param_name in parameters:
        return parameters[param_name]

    return None


def _extract_usage(response: Any) -> tuple[Optional[int], Optional[int]]:
    """Extract token usage from DashScope response.

    Args:
        response: DashScope response object

    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    if not response:
        return None, None

    try:
        # Use getattr with default None to safely access attributes
        # DashScope response uses __getattr__ which raises KeyError for missing attributes
        usage = getattr(response, "usage", None)
        if not usage:
            return None, None

        # Use getattr with default None for safe access
        input_tokens = getattr(usage, "input_tokens", None) or getattr(
            usage, "prompt_tokens", None
        )
        output_tokens = getattr(usage, "output_tokens", None) or getattr(
            usage, "completion_tokens", None
        )

        return input_tokens, output_tokens
    except (KeyError, AttributeError):
        # If any attribute access fails, return None for both tokens
        return None, None


def _extract_task_id(task: Any) -> Optional[str]:
    """Extract task_id from task parameter (can be str or Response object).

    Args:
        task: Task parameter (str task_id or Response object)

    Returns:
        task_id string if found, None otherwise
    """
    if not task:
        return None

    if isinstance(task, str):
        return task

    try:
        # Try to get task_id from response object
        if hasattr(task, "output") and hasattr(task.output, "get"):
            task_id = task.output.get("task_id")
            if task_id:
                return task_id
    except (KeyError, AttributeError):
        pass

    return None


# Context key for skipping instrumentation in nested calls
_SKIP_INSTRUMENTATION_KEY = "dashscope.skip_instrumentation"

