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
LongSuite LangChain Tracer (framework-only phase).

This tracer extends langchain_core's BaseTracer to receive callback events
from all LangChain Runnable executions. In this phase, it only prints
instrumentation logs to verify that all instrumentation points are triggered.

Supported run types:
- llm: Chat Model / LLM calls
- chain: Chain / Agent executions
- tool: Tool invocations
- retriever: RAG retrieval operations
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

logger = logging.getLogger(__name__)


class LoongsuiteTracer(BaseTracer):
    """LangChain tracer that receives callback events.

    Framework-only phase: only prints instrumentation logs.
    Data extraction (span creation, attribute extraction) will be
    implemented in the next phase.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _start_trace(self, run: Run) -> None:
        super()._start_trace(run)
        parent_info = (
            f", parent_run_id={run.parent_run_id}"
            if run.parent_run_id
            else ""
        )
        input_keys = list(run.inputs.keys()) if run.inputs else []
        print(
            f"[INSTRUMENTATION] _start_trace: name={run.name}, "
            f"run_type={run.run_type}, run_id={run.id}{parent_info}"
        )
        print(f"[INSTRUMENTATION]   input_keys={input_keys}")

    def _end_trace(self, run: Run) -> None:
        super()._end_trace(run)
        output_keys = list(run.outputs.keys()) if run.outputs else []
        has_error = run.error is not None
        print(
            f"[INSTRUMENTATION] _end_trace: name={run.name}, "
            f"run_type={run.run_type}, has_error={has_error}"
        )
        print(f"[INSTRUMENTATION]   output_keys={output_keys}")

    def _persist_run(self, run: Run) -> None:
        pass

    def on_llm_error(
        self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any
    ) -> Run:
        print(
            f"[INSTRUMENTATION] on_llm_error: run_id={run_id}, "
            f"error_type={type(error).__name__}, error={error}"
        )
        return super().on_llm_error(error, *args, run_id=run_id, **kwargs)

    def on_chain_error(
        self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any
    ) -> Run:
        print(
            f"[INSTRUMENTATION] on_chain_error: run_id={run_id}, "
            f"error_type={type(error).__name__}, error={error}"
        )
        return super().on_chain_error(error, *args, run_id=run_id, **kwargs)

    def on_retriever_error(
        self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any
    ) -> Run:
        print(
            f"[INSTRUMENTATION] on_retriever_error: run_id={run_id}, "
            f"error_type={type(error).__name__}, error={error}"
        )
        return super().on_retriever_error(
            error, *args, run_id=run_id, **kwargs
        )

    def on_tool_error(
        self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any
    ) -> Run:
        print(
            f"[INSTRUMENTATION] on_tool_error: run_id={run_id}, "
            f"error_type={type(error).__name__}, error={error}"
        )
        return super().on_tool_error(error, *args, run_id=run_id, **kwargs)
