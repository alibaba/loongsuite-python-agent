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
LongSuite LangChain Tracer — data extraction phase.

Extends ``langchain_core.tracers.base.BaseTracer`` and overrides the
fine-grained ``_on_*`` hooks to extract telemetry data from LangChain
``Run`` objects and emit OpenTelemetry spans via ``util-genai``.

Run type → handler mapping
--------------------------
* **LLM / chat_model** → ``handler.start_llm`` / ``stop_llm`` / ``fail_llm``
* **Chain (Agent)**     → ``handler.start_invoke_agent`` / …
* **Chain (generic)**   → direct span creation (no ``util-genai``)
* **Tool**              → ``handler.start_execute_tool`` / …
* **Retriever**         → ``handler.start_retrieve`` / …
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from opentelemetry import context as otel_context
from opentelemetry.trace import Span, SpanKind, StatusCode, set_span_in_context
from opentelemetry.util.genai.extended_handler import ExtendedTelemetryHandler
from opentelemetry.util.genai.extended_types import (
    ExecuteToolInvocation,
    InvokeAgentInvocation,
    RetrieveInvocation,
)
from opentelemetry.util.genai.handler import _safe_detach
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
)

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

from opentelemetry.instrumentation.langchain.internal._utils import (
    _extract_finish_reasons,
    _extract_invocation_params,
    _extract_llm_input_messages,
    _extract_llm_output_messages,
    _extract_model_name,
    _extract_provider,
    _extract_response_model,
    _extract_token_usage,
    _is_agent_run,
    _safe_json,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# _RunData — per-run bookkeeping
# ---------------------------------------------------------------------------
RunKind = Literal["llm", "agent", "chain", "tool", "retriever"]


@dataclass
class _RunData:
    run_kind: RunKind
    invocation: Any = None  # util-genai invocation object (LLM/Agent/Tool/Retriever)
    span: Span | None = None  # only used for generic chain spans
    parent_context_token: object | None = None  # token from attaching parent ctx


# ---------------------------------------------------------------------------
# LoongsuiteTracer
# ---------------------------------------------------------------------------

class LoongsuiteTracer(BaseTracer):
    """LangChain tracer that emits OpenTelemetry spans via util-genai."""

    def __init__(self, handler: ExtendedTelemetryHandler, **kwargs: Any) -> None:
        super().__init__(_schema_format="original+chat", **kwargs)
        self._handler = handler
        self._runs: dict[UUID, _RunData] = {}

    def _persist_run(self, run: Run) -> None:
        pass

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _attach_parent_context(self, run: Run) -> object | None:
        """Attach the parent run's span context so child spans nest correctly."""
        parent_id = getattr(run, "parent_run_id", None)
        if parent_id and parent_id in self._runs:
            parent_data = self._runs[parent_id]
            span = parent_data.invocation.span if parent_data.invocation else parent_data.span
            if span is not None:
                return otel_context.attach(set_span_in_context(span))
        return None

    def _detach_parent_context(self, token: object | None) -> None:
        _safe_detach(token)

    # ------------------------------------------------------------------
    # _start_trace / _end_trace — context lifecycle only
    # ------------------------------------------------------------------

    def _start_trace(self, run: Run) -> None:
        super()._start_trace(run)

    def _end_trace(self, run: Run) -> None:
        super()._end_trace(run)
        # Cleanup of self._runs is done in _on_*_end / _on_*_error hooks,
        # which are called AFTER _end_trace in the BaseTracer lifecycle.

    # ------------------------------------------------------------------
    # LLM hooks
    # ------------------------------------------------------------------

    def _on_llm_start(self, run: Run) -> None:
        self._handle_llm_start(run)

    def _on_chat_model_start(self, run: Run) -> None:
        self._handle_llm_start(run)

    def _handle_llm_start(self, run: Run) -> None:
        parent_token = self._attach_parent_context(run)
        try:
            params = _extract_invocation_params(run)
            invocation = LLMInvocation(
                request_model=_extract_model_name(run) or run.name or "",
                provider=_extract_provider(run),
                input_messages=_extract_llm_input_messages(run),
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                max_tokens=params.get("max_tokens") or params.get("max_output_tokens"),
            )
            self._handler.start_llm(invocation)
            self._detach_parent_context(parent_token)
            _safe_detach(invocation.context_token)
            self._runs[run.id] = _RunData(
                run_kind="llm",
                invocation=invocation,
                parent_context_token=None,
            )
        except Exception:
            self._detach_parent_context(parent_token)
            logger.debug("Failed to start LLM span", exc_info=True)

    def _on_llm_end(self, run: Run) -> None:
        rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "llm":
            return
        try:
            inv: LLMInvocation = rd.invocation
            inv.output_messages = _extract_llm_output_messages(run)
            inv.input_tokens, inv.output_tokens = _extract_token_usage(run)
            inv.finish_reasons = _extract_finish_reasons(run)
            inv.response_model_name = _extract_response_model(run)
            self._handler.stop_llm(inv)
        except Exception:
            logger.debug("Failed to stop LLM span", exc_info=True)

    def _on_llm_error(self, run: Run) -> None:
        rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "llm":
            return
        try:
            err_str = getattr(run, "error", None) or "Unknown error"
            self._handler.fail_llm(
                rd.invocation,
                Error(message=str(err_str), type=Exception),
            )
        except Exception:
            logger.debug("Failed to fail LLM span", exc_info=True)

    # ------------------------------------------------------------------
    # Chain / Agent hooks
    # ------------------------------------------------------------------

    def _on_chain_start(self, run: Run) -> None:
        parent_token = self._attach_parent_context(run)
        try:
            if _is_agent_run(run):
                self._start_agent(run, parent_token)
            else:
                self._start_chain(run, parent_token)
        except Exception:
            self._detach_parent_context(parent_token)
            logger.debug("Failed to start Chain/Agent span", exc_info=True)

    def _start_agent(self, run: Run, parent_token: object | None) -> None:
        inputs = getattr(run, "inputs", None) or {}
        input_messages: list[InputMessage] = []
        input_val = inputs.get("input") or inputs.get("query") or ""
        if isinstance(input_val, str) and input_val:
            input_messages.append(
                InputMessage(role="user", parts=[Text(content=input_val)])
            )

        invocation = InvokeAgentInvocation(
            provider="langchain",
            agent_name=run.name,
            input_messages=input_messages,
        )
        self._handler.start_invoke_agent(invocation)
        self._detach_parent_context(parent_token)
        _safe_detach(invocation.context_token)
        self._runs[run.id] = _RunData(
            run_kind="agent",
            invocation=invocation,
        )

    def _start_chain(self, run: Run, parent_token: object | None) -> None:
        tracer = self._handler._tracer  # noqa: SLF001
        span = tracer.start_span(
            name=f"chain {run.name}",
            kind=SpanKind.INTERNAL,
        )
        ctx_token = otel_context.attach(set_span_in_context(span))

        inputs = getattr(run, "inputs", None) or {}
        span.set_attribute("gen_ai.span.kind", "chain")
        input_str = _safe_json(inputs)
        span.set_attribute("input.value", input_str)

        self._detach_parent_context(parent_token)
        _safe_detach(ctx_token)

        self._runs[run.id] = _RunData(
            run_kind="chain",
            span=span,
        )

    def _on_chain_end(self, run: Run) -> None:
        rd = self._runs.pop(run.id, None)
        if rd is None:
            return
        try:
            if rd.run_kind == "agent":
                self._stop_agent(run, rd)
            elif rd.run_kind == "chain":
                self._stop_chain(run, rd)
        except Exception:
            logger.debug("Failed to stop Chain/Agent span", exc_info=True)

    def _stop_agent(self, run: Run, rd: _RunData) -> None:
        inv: InvokeAgentInvocation = rd.invocation
        outputs = getattr(run, "outputs", None) or {}
        output_val = outputs.get("output") or outputs.get("result") or ""
        if isinstance(output_val, str) and output_val:
            inv.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[Text(content=output_val)],
                    finish_reason="stop",
                )
            ]
        self._handler.stop_invoke_agent(inv)

    def _stop_chain(self, run: Run, rd: _RunData) -> None:
        span = rd.span
        if span is None:
            return
        outputs = getattr(run, "outputs", None) or {}
        output_str = _safe_json(outputs)
        span.set_attribute("output.value", output_str)
        span.set_status(StatusCode.OK)
        span.end()

    def _on_chain_error(self, run: Run) -> None:
        rd = self._runs.pop(run.id, None)
        if rd is None:
            return
        try:
            err_str = getattr(run, "error", None) or "Unknown error"
            if rd.run_kind == "agent":
                self._handler.fail_invoke_agent(
                    rd.invocation,
                    Error(message=str(err_str), type=Exception),
                )
            elif rd.run_kind == "chain":
                span = rd.span
                if span is not None:
                    span.set_status(StatusCode.ERROR, str(err_str))
                    span.record_exception(Exception(str(err_str)))
                    span.end()
        except Exception:
            logger.debug("Failed to fail Chain/Agent span", exc_info=True)

    # ------------------------------------------------------------------
    # Tool hooks
    # ------------------------------------------------------------------

    def _on_tool_start(self, run: Run) -> None:
        parent_token = self._attach_parent_context(run)
        try:
            inputs = getattr(run, "inputs", None) or {}
            input_str = inputs.get("input") or inputs.get("query") or ""
            if not isinstance(input_str, str):
                input_str = _safe_json(input_str)

            invocation = ExecuteToolInvocation(
                tool_name=run.name or "unknown_tool",
                tool_call_arguments=input_str,
            )
            self._handler.start_execute_tool(invocation)
            self._detach_parent_context(parent_token)
            _safe_detach(invocation.context_token)
            self._runs[run.id] = _RunData(
                run_kind="tool",
                invocation=invocation,
            )
        except Exception:
            self._detach_parent_context(parent_token)
            logger.debug("Failed to start Tool span", exc_info=True)

    def _on_tool_end(self, run: Run) -> None:
        rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "tool":
            return
        try:
            inv: ExecuteToolInvocation = rd.invocation
            outputs = getattr(run, "outputs", None) or {}
            output = outputs.get("output") or ""
            if not isinstance(output, str):
                output = _safe_json(output)
            inv.tool_call_result = output
            self._handler.stop_execute_tool(inv)
        except Exception:
            logger.debug("Failed to stop Tool span", exc_info=True)

    def _on_tool_error(self, run: Run) -> None:
        rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "tool":
            return
        try:
            err_str = getattr(run, "error", None) or "Unknown error"
            self._handler.fail_execute_tool(
                rd.invocation,
                Error(message=str(err_str), type=Exception),
            )
        except Exception:
            logger.debug("Failed to fail Tool span", exc_info=True)

    # ------------------------------------------------------------------
    # Retriever hooks
    # ------------------------------------------------------------------

    def _on_retriever_start(self, run: Run) -> None:
        parent_token = self._attach_parent_context(run)
        try:
            inputs = getattr(run, "inputs", None) or {}
            query = inputs.get("query") or ""

            invocation = RetrieveInvocation(query=query)
            self._handler.start_retrieve(invocation)
            self._detach_parent_context(parent_token)
            _safe_detach(invocation.context_token)
            self._runs[run.id] = _RunData(
                run_kind="retriever",
                invocation=invocation,
            )
        except Exception:
            self._detach_parent_context(parent_token)
            logger.debug("Failed to start Retriever span", exc_info=True)

    def _on_retriever_end(self, run: Run) -> None:
        rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "retriever":
            return
        try:
            inv: RetrieveInvocation = rd.invocation
            outputs = getattr(run, "outputs", None) or {}
            documents = outputs.get("documents") or []
            if documents:
                inv.documents = _safe_json(documents)
            self._handler.stop_retrieve(inv)
        except Exception:
            logger.debug("Failed to stop Retriever span", exc_info=True)

    def _on_retriever_error(self, run: Run) -> None:
        rd = self._runs.pop(run.id, None)
        if rd is None or rd.run_kind != "retriever":
            return
        try:
            err_str = getattr(run, "error", None) or "Unknown error"
            self._handler.fail_retrieve(
                rd.invocation,
                Error(message=str(err_str), type=Exception),
            )
        except Exception:
            logger.debug("Failed to fail Retriever span", exc_info=True)

    # ------------------------------------------------------------------
    # Deep copy / copy — return self (shared singleton)
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo: dict) -> LoongsuiteTracer:
        return self

    def __copy__(self) -> LoongsuiteTracer:
        return self
