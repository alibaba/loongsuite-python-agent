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

from typing import Any

from opentelemetry.instrumentation.dify.strategy.agent_strategy import (
    AppRunnerStrategy,
    MessageEndStrategy,
)
from opentelemetry.instrumentation.dify.strategy.strategy import (
    ProcessStrategy,
)
from opentelemetry.instrumentation.dify.strategy.workflow_strategy import (
    WorkflowNodeExecutionFailedStrategy,
    WorkflowNodeFinishStrategy,
    WorkflowNodeInitStrategy,
    WorkflowNodeStartStrategy,
    WorkflowRunFailedStrategy,
    WorkflowRunStartStrategy,
    WorkflowRunSuccessStrategy,
)


class StrategyFactory:
    """Factory class for creating and managing process strategies.

    This factory provides a centralized way to create and access different process
    strategies based on method names. It implements the Factory pattern for strategy
    creation and management.

    The factory maintains a mapping of method suffixes to their corresponding strategies,
    allowing for dynamic strategy selection based on the method being called.

    Attributes:
        _handler: The handler instance that manages the overall instrumentation process
        _strategies: Dictionary mapping method suffixes to their corresponding strategy instances
    """

    def __init__(self, handler: Any):
        self._handler = handler
        self._strategies = {
            ### agent/chat
            "run": AppRunnerStrategy(handler),
            "_message_end_to_stream_response": MessageEndStrategy(handler),
            ### workflow
            "__init__": WorkflowNodeInitStrategy(handler),
            "_handle_workflow_run_start": WorkflowRunStartStrategy(
                handler
            ),  # v1
            "handle_workflow_run_start": WorkflowRunStartStrategy(
                handler
            ),  # v2
            "_handle_workflow_run_success": WorkflowRunSuccessStrategy(
                handler
            ),  # v1
            "handle_workflow_run_success": WorkflowRunSuccessStrategy(
                handler
            ),  # v2
            "_handle_workflow_run_failed": WorkflowRunFailedStrategy(
                handler
            ),  # v1
            "handle_workflow_run_failed": WorkflowRunFailedStrategy(
                handler
            ),  # v2
            "_workflow_node_start_to_stream_response": WorkflowNodeStartStrategy(
                handler
            ),  # v1
            "workflow_node_start_to_stream_response": WorkflowNodeStartStrategy(
                handler
            ),  # v2
            "_workflow_node_finish_to_stream_response": WorkflowNodeFinishStrategy(
                handler
            ),  # v1
            "workflow_node_finish_to_stream_response": WorkflowNodeFinishStrategy(
                handler
            ),  # v2
            "_handle_workflow_node_execution_failed": WorkflowNodeExecutionFailedStrategy(
                handler
            ),  # v1
            "handle_workflow_node_execution_failed": WorkflowNodeExecutionFailedStrategy(
                handler
            ),  # v2
        }

    def get_strategy(self, method: str) -> ProcessStrategy:
        for suffix, strategy in self._strategies.items():
            if method.endswith(suffix):
                return strategy
        return None
