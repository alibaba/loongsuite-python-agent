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

Context propagation follows the Robin/OpenLLMetry pattern: parent-child
span relationships are established by passing ``context`` explicitly to
``start_span`` / ``handler.start_*``, rather than using hazardous
``context_api.attach`` / ``detach`` in a callback system.

The only exception is Chain spans: they use ``attach``/``detach`` so that
non-LangChain child operations (e.g. HTTP calls) nest correctly.

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
import timeit
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal, Optional
from uuid import UUID

from opentelemetry import context as otel_context
from opentelemetry.context import Context
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
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    get_content_capturing_mode,
    is_experimental_mode,
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
    span: Span | None = None
    context: Context | None = None
    invocation: Any = None
    context_token: object | None = None  # only used for Chain attach/detach


def _should_capture_chain_content() -> bool:
    """Check if chain input/output content should be recorded."""
    try:
        if not is_experimental_mode():
            return False
        return get_content_capturing_mode() in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# LoongsuiteTracer
# ---------------------------------------------------------------------------

class LoongsuiteTracer(BaseTracer):
    """LangChain tracer that emits OpenTelemetry spans via util-genai.

    Context propagation is done explicitly — parent-child relationships
    are established by passing the stored ``Context`` of the parent run
    to ``handler.start_*(…, context=parent_ctx)`` or to
    ``tracer.start_span(…, context=parent_ctx)``.

    Chain spans are the sole exception: they ``attach``/``detach`` the
    context so that non-LangChain child operations nest correctly.

    All access to ``self._runs`` is protected by an ``RLock`` because
    LangChain callbacks may be fired from different threads.
    """

    def __init__(self, handler: ExtendedTelemetryHandler, **kwargs: Any) -> None:
        super().__init__(_schema_format="original+chat", **kwargs)
        self._handler = handler
        self._runs: dict[UUID, _RunData] = {}
        self._lock = RLock()

    def _persist_run(self, run: Run) -> None:
        pass

    # ------------------------------------------------------------------
    # Context helper
    # ------------------------------------------------------------------

    def _get_parent_context(self, run: Run) -> Context | None:
        """Return the stored context of the parent run, or *None*."""
        parent_id = getattr(run, "parent_run_id", None)
        if parent_id:
            with self._lock:
                rd = self._runs.get(parent_id)
            if rd is not None:
                return rd.context
        return None

    # ------------------------------------------------------------------
    # _start_trace / _end_trace
    # ------------------------------------------------------------------

    def _start_trace(self, run: Run) -> None:
        super()._start_trace(run)

    def _end_trace(self, run: Run) -> None:
        super()._end_trace(run)

    # ------------------------------------------------------------------
    # TTFT (Time To First Token) — streaming support
    # ------------------------------------------------------------------

    def on_llm_new_token(  # type: ignore[override]
        self,
        token: str,
        *,
        chunk: Optional[Any] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Run | None:
        """Record the first-token timestamp for TTFT calculation."""
        with self._lock:
            rd = self._runs.get(run_id)
        if rd is not None and rd.run_kind == "llm" and rd.invocation is not None:
            inv: LLMInvocation = rd.invocation
            if inv.monotonic_first_token_s is None:
                inv.monotonic_first_token_s = timeit.default_timer()
        return None

    # ------------------------------------------------------------------
    # LLM hooks
    # ------------------------------------------------------------------

    def _on_llm_start(self, run: Run) -> None:
        self._handle_llm_start(run)

    def _on_chat_model_start(self, run: Run) -> None:
        self._handle_llm_start(run)

    def _handle_llm_start(self, run: Run) -> None:
        try:
            parent_ctx = self._get_parent_context(run)
            params = _extract_invocation_params(run)
            invocation = LLMInvocation(
                request_model=_extract_model_name(run) or run.name or "",
                provider=_extract_provider(run),
                input_messages=_extract_llm_input_messages(run),
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                max_tokens=params.get("max_tokens") or params.get("max_output_tokens"),
            )
            self._handler.start_llm(invocation, context=parent_ctx)
            rd = _RunData(
                run_kind="llm",
                span=invocation.span,
                context=set_span_in_context(invocation.span) if invocation.span else None,
                invocation=invocation,
            )
            with self._lock:
                self._runs[run.id] = rd
        except Exception:
            logger.debug("Failed to start LLM span", exc_info=True)

    def _on_llm_end(self, run: Run) -> None:
        with self._lock:
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
        with self._lock:
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
        try:
            if _is_agent_run(run):
                self._start_agent(run)
            else:
                self._start_chain(run)
        except Exception:
            logger.debug("Failed to start Chain/Agent span", exc_info=True)

    def _start_agent(self, run: Run) -> None:
        parent_ctx = self._get_parent_context(run)
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
        self._handler.start_invoke_agent(invocation, context=parent_ctx)
        rd = _RunData(
            run_kind="agent",
            span=invocation.span,
            context=set_span_in_context(invocation.span) if invocation.span else None,
            invocation=invocation,
        )
        with self._lock:
            self._runs[run.id] = rd

    def _start_chain(self, run: Run) -> None:
        parent_ctx = self._get_parent_context(run)
        tracer = self._handler._tracer  # noqa: SLF001
        span = tracer.start_span(
            name=f"chain {run.name}",
            kind=SpanKind.INTERNAL,
            context=parent_ctx,
        )

        span.set_attribute("gen_ai.span.kind", "CHAIN")
        if _should_capture_chain_content():
            inputs = getattr(run, "inputs", None) or {}
            span.set_attribute("input.value", _safe_json(inputs))

        # Attach chain span context so non-LangChain children nest correctly.
        ctx = set_span_in_context(span)
        token = otel_context.attach(ctx)

        rd = _RunData(
            run_kind="chain",
            span=span,
            context=ctx,
            context_token=token,
        )
        with self._lock:
            self._runs[run.id] = rd

    def _on_chain_end(self, run: Run) -> None:
        with self._lock:
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
        if _should_capture_chain_content():
            outputs = getattr(run, "outputs", None) or {}
            span.set_attribute("output.value", _safe_json(outputs))
        span.set_status(StatusCode.OK)
        span.end()
        _safe_detach(rd.context_token)

    def _on_chain_error(self, run: Run) -> None:
        with self._lock:
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
                _safe_detach(rd.context_token)
        except Exception:
            logger.debug("Failed to fail Chain/Agent span", exc_info=True)

    # ------------------------------------------------------------------
    # Tool hooks
    # ------------------------------------------------------------------

    def _on_tool_start(self, run: Run) -> None:
        try:
            parent_ctx = self._get_parent_context(run)
            inputs = getattr(run, "inputs", None) or {}
            input_str = inputs.get("input") or inputs.get("query") or ""
            if not isinstance(input_str, str):
                input_str = _safe_json(input_str)

            invocation = ExecuteToolInvocation(
                tool_name=run.name or "unknown_tool",
                tool_call_arguments=input_str,
            )
            self._handler.start_execute_tool(invocation, context=parent_ctx)
            rd = _RunData(
                run_kind="tool",
                span=invocation.span,
                context=set_span_in_context(invocation.span) if invocation.span else None,
                invocation=invocation,
            )
            with self._lock:
                self._runs[run.id] = rd
        except Exception:
            logger.debug("Failed to start Tool span", exc_info=True)

    def _on_tool_end(self, run: Run) -> None:
        with self._lock:
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
        with self._lock:
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
        try:
            parent_ctx = self._get_parent_context(run)
            inputs = getattr(run, "inputs", None) or {}
            query = inputs.get("query") or ""

            invocation = RetrieveInvocation(query=query)
            self._handler.start_retrieve(invocation, context=parent_ctx)
            rd = _RunData(
                run_kind="retriever",
                span=invocation.span,
                context=set_span_in_context(invocation.span) if invocation.span else None,
                invocation=invocation,
            )
            with self._lock:
                self._runs[run.id] = rd
        except Exception:
            logger.debug("Failed to start Retriever span", exc_info=True)

    def _on_retriever_end(self, run: Run) -> None:
        with self._lock:
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
        with self._lock:
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
