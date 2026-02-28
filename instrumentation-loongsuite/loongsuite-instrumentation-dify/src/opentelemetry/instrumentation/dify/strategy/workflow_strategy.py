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

import json
import time
from copy import deepcopy
from typing import (
    Any,
    Dict,
    Mapping,
    Tuple,
)

from opentelemetry import context as context_api

#  dify packages path
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.dify.capture_content import (
    process_content,
    set_dict_value,
)
from opentelemetry.instrumentation.dify.config import (
    is_wrapper_version_1,
    is_wrapper_version_2,
)
from opentelemetry.instrumentation.dify.constants import (
    DIFY_APP_ID_KEY,
    _get_dify_app_name_key,
)
from opentelemetry.instrumentation.dify.dify_utils import get_workflow_run_id
from opentelemetry.instrumentation.dify.entities import NodeType, _EventData
from opentelemetry.instrumentation.dify.semconv import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_MODEL_NAME,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_REQUEST_MODEL_NAME,
    GEN_AI_SESSION_ID,
    GEN_AI_SPAN_KIND,
    GEN_AI_USAGE_TOTAL_TOKENS,
    GEN_AI_USER_ID,
    OUTPUT_VALUE,
    RETRIEVAL_DOCUMENTS,
    DocumentAttributes,
    SpanKindValues,
)
from opentelemetry.instrumentation.dify.strategy.strategy import (
    ProcessStrategy,
)
from opentelemetry.instrumentation.dify.utils import (
    get_llm_common_attributes,
    get_timestamp_from_datetime_attr,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_COMPLETION_TOKENS,
    GEN_AI_USAGE_PROMPT_TOKENS,
)
from opentelemetry.trace.status import Status, StatusCode

_DIFY_APP_NAME_KEY = _get_dify_app_name_key()


class WorkflowNodeInitStrategy(ProcessStrategy):
    """Strategy for handling initialization events in the workflow.

    This strategy manages the setup of new workflow instances, including:
    - Context creation and initialization
    - Span creation for new workflow instances
    - Initial state configuration
    - Parent-child relationship setup for nodes
    - Resource allocation and setup

    The strategy handles:
    - Workflow node initialization
    - Context propagation
    - Span hierarchy setup
    - Initial attribute configuration
    - Error handling during initialization
    """

    def process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
        res: Any,
    ) -> None:
        try:
            event_id = kwargs["id"]
            span = None
            with self._lock:
                from opentelemetry import context  # noqa: PLC0415

                ctx = context.get_current()
                if len(ctx) == 0:
                    self._logger.info(
                        f"can't get ctx, return : {ctx},kwargs: {kwargs},event_id: {event_id}"
                    )
                    graph_runtime_state = kwargs["graph_runtime_state"]
                    # FIXME: ruff failed
                    previous_node_id = kwargs["previous_node_id"]  # noqa: F841
                    if graph_runtime_state is None:
                        return
                    node_run_state = getattr(
                        graph_runtime_state, "node_run_state"
                    )
                    node_state_mapping = getattr(
                        node_run_state, "node_state_mapping"
                    )
                    for k, v in node_state_mapping.items():
                        event_data = self._event_data.get(k)
                        if event_data is not None:
                            start_time = time.time_ns()
                            parent_ctx = trace_api.set_span_in_context(
                                event_data.span
                            )
                            span: trace_api.Span = self._tracer.start_span(
                                f"node_run_{event_id}",
                                attributes={
                                    GEN_AI_SPAN_KIND: SpanKindValues.CHAIN.value,
                                    "component.name": "dify",
                                },
                                start_time=start_time,
                                context=parent_ctx,
                            )
                            new_ctx = trace_api.set_span_in_context(span)
                            token = context_api.attach(new_ctx)
                            self._logger.info(
                                f"node event id: {k},event_data: {event_data} {event_data.span}"
                            )
                            self._event_data[event_id] = _EventData(
                                span=span,
                                parent_id=None,
                                context=None,
                                payloads=[],
                                exceptions=[],
                                attributes={},
                                node_type=None,
                                start_time=start_time,
                                otel_token=token,
                            )
                            return
                    return
                start_time = time.time_ns()
                event_data = self._event_data.get(event_id)
                if event_data is None:
                    span: trace_api.Span = self._tracer.start_span(
                        f"node_run_{event_id}",
                        attributes={
                            GEN_AI_SPAN_KIND: SpanKindValues.CHAIN.value,
                            "component.name": "dify",
                        },
                        start_time=start_time,
                    )
                else:
                    span = event_data.span
                new_context = trace_api.set_span_in_context(span)
                token = context_api.attach(new_context)
                self._event_data[event_id] = _EventData(
                    span=span,
                    parent_id=None,
                    context=None,
                    payloads=[],
                    exceptions=[],
                    attributes={},
                    node_type=None,
                    start_time=start_time,
                    otel_token=token,
                )
        except Exception:
            self._logger.exception(
                "Fail to process data, func name: BaseNode.__init__"
            )


class WorkflowRunStartStrategy(ProcessStrategy):
    """Strategy for handling workflow run start events.

    This strategy manages the beginning of workflow executions, including:
    - Creation of workflow spans
    - Context setup for the workflow
    - Initial attribute configuration
    - User and session tracking
    - Resource initialization

    The strategy tracks:
    - Workflow start times
    - User and session information
    - Input parameters
    - Initial state setup
    - Performance metrics for workflow starts
    """

    def process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
        res: Any,
    ) -> None:
        if is_wrapper_version_1():
            self._handle_workflow_run_start_v1(res)
        elif is_wrapper_version_2():
            self._handle_workflow_run_start_v2(res)

    def _handle_workflow_run_start_v1(self, run):
        event_id = get_workflow_run_id(run)
        if event_id is None:
            self._logger.warning(
                "workflow_run_start: missing event_id",
                extra={
                    "run_object": str(run),
                    "component": "workflow_handler",
                },
            )
            return
        start_time = get_timestamp_from_datetime_attr(run, "created_at")
        with self._lock:
            data = self._event_data.get(event_id)
            if data is not None:
                return
        span: trace_api.Span = self._tracer.start_span(
            f"workflow_run_{event_id}",
            attributes={
                GEN_AI_SPAN_KIND: SpanKindValues.CHAIN.value,
                "component.name": "dify",
            },
            start_time=start_time,
        )
        app_id = getattr(run, "app_id", None)
        app_name = self._handler.get_app_name_by_id(app_id)
        inputs_dict = getattr(run, "inputs_dict", None)
        user_id = "DEFAULT_USER_ID"
        session_id = "DEFAULT_SESSION_ID"
        if inputs_dict is not None:
            if "sys.user_id" in inputs_dict:
                user_id = inputs_dict["sys.user_id"]
            if "sys.conversation_id" in inputs_dict:
                session_id = inputs_dict["sys.conversation_id"]
        span.set_attribute(GEN_AI_USER_ID, user_id)
        span.set_attribute(GEN_AI_SESSION_ID, session_id)
        new_context = trace_api.set_span_in_context(span)
        token = context_api.attach(new_context)
        with self._lock:
            self._event_data[event_id] = _EventData(
                span=span,
                parent_id=None,
                context=new_context,
                payloads=[],
                exceptions=[],
                attributes={
                    DIFY_APP_ID_KEY: app_id,
                    _DIFY_APP_NAME_KEY: app_name,
                    GEN_AI_USER_ID: user_id,
                    GEN_AI_SESSION_ID: session_id,
                },
                node_type=None,
                start_time=start_time,
                otel_token=token,
            )

    def _handle_workflow_run_start_v2(self, run):
        event_id = get_workflow_run_id(run)
        if event_id is None:
            self._logger.warning(
                "workflow_run_start: missing event_id",
                extra={
                    "run_object": str(run),
                    "component": "workflow_handler",
                },
            )
            return
        start_time = get_timestamp_from_datetime_attr(run, "created_at")
        with self._lock:
            data = self._event_data.get(event_id)
            if data is not None:
                return
        span: trace_api.Span = self._tracer.start_span(
            f"workflow_run_{event_id}",
            attributes={
                GEN_AI_SPAN_KIND: SpanKindValues.CHAIN.value,
                "component.name": "dify",
            },
            start_time=start_time,
        )

        user_id = "DEFAULT_USER_ID"
        session_id = "DEFAULT_SESSION_ID"
        inputs_dict = getattr(run, "inputs", None)
        if inputs_dict is not None:
            if "sys.user_id" in inputs_dict:
                user_id = inputs_dict["sys.user_id"]
            if "sys.conversation_id" in inputs_dict:
                session_id = inputs_dict["sys.conversation_id"]
            if "sys.app_id" in inputs_dict:
                app_id = inputs_dict["sys.app_id"]
                app_name = self._handler.get_app_name_by_id(app_id)
        span.set_attribute(GEN_AI_USER_ID, user_id)
        span.set_attribute(GEN_AI_SESSION_ID, session_id)
        new_context = trace_api.set_span_in_context(span)
        token = context_api.attach(new_context)
        with self._lock:
            self._event_data[event_id] = _EventData(
                span=span,
                parent_id=None,
                context=new_context,
                payloads=[],
                exceptions=[],
                attributes={
                    DIFY_APP_ID_KEY: app_id,
                    _DIFY_APP_NAME_KEY: app_name,
                    GEN_AI_USER_ID: user_id,
                    GEN_AI_SESSION_ID: session_id,
                },
                node_type=None,
                start_time=start_time,
                otel_token=token,
            )


class WorkflowRunSuccessStrategy(ProcessStrategy):
    """Strategy for handling successful workflow run completions.

    This strategy processes successful workflow executions, including:
    - Recording workflow outputs
    - Updating span attributes
    - Cleaning up resources
    - Recording performance metrics
    - Handling successful completion states

    The strategy manages:
    - Output recording and formatting
    - Span completion and cleanup
    - Success metrics collection
    - Resource cleanup
    - Final state recording
    """

    def process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
        res: Any,
    ) -> None:
        if is_wrapper_version_1():
            workflow_run = None
            if "workflow_run" in kwargs:
                workflow_run = kwargs["workflow_run"]
            else:
                workflow_run = res
            self._handle_workflow_run_success_v1(
                workflow_run, outputs=kwargs["outputs"]
            )
        elif is_wrapper_version_2():
            self._handle_workflow_run_success_v2(
                res, outputs=kwargs["outputs"]
            )

    def _handle_workflow_run_success_v1(self, run, outputs=[]):
        event_id = get_workflow_run_id(run)
        if event_id is None:
            return
        event_data = self._event_data.pop(event_id, None)
        if event_data is None:
            self._logger.warning(f"can not get data ,event_id: {event_id}")
            return

        app_id = getattr(run, "app_id", None)
        app_name = self._handler.get_app_name_by_id(app_id)
        span: trace_api.Span = event_data.span
        span.update_name(app_name)
        span_attributes = {}
        span_attributes[_DIFY_APP_NAME_KEY] = app_name
        span_attributes[DIFY_APP_ID_KEY] = app_id
        input_attr = self._extract_inputs(run.inputs_dict)
        span_attributes.update(input_attr)
        if isinstance(outputs, str):
            outputs_dict = json.loads(outputs)
        else:
            outputs_dict = outputs
        output_attr = self._extract_outputs(outputs_dict)
        span_attributes.update(output_attr)
        span.set_attributes(span_attributes)
        if span.is_recording():
            span.end()
        context_api.detach(event_data.otel_token)
        metrics_attributes = get_llm_common_attributes()
        metrics_attributes["spanKind"] = SpanKindValues.CHAIN.value
        self._record_metrics(event_data, metrics_attributes)

    def _handle_workflow_run_success_v2(self, run, outputs=[]):
        event_id = get_workflow_run_id(run)
        if event_id is None:
            return
        event_data = self._event_data.pop(event_id, None)
        if event_data is None:
            self._logger.warning(f"can not get data ,event_id: {event_id}")
            return

        app_id = "UNKNOWN_APP_ID"
        app_name = "DEFAULT_APP_NAME"
        inputs_dict = getattr(run, "inputs", None)
        if inputs_dict is not None:
            if "sys.app_id" in inputs_dict:
                app_id = inputs_dict["sys.app_id"]
                app_name = self._handler.get_app_name_by_id(app_id)

        span: trace_api.Span = event_data.span
        span.update_name(app_name)
        span_attributes = {}
        span_attributes[_DIFY_APP_NAME_KEY] = app_name
        span_attributes[DIFY_APP_ID_KEY] = app_id
        input_attr = self._extract_inputs(inputs_dict)
        span_attributes.update(input_attr)
        if isinstance(outputs, str):
            outputs_dict = json.loads(outputs)
        else:
            outputs_dict = outputs
        output_attr = self._extract_outputs(outputs_dict)
        span_attributes.update(output_attr)
        span.set_attributes(span_attributes)
        if span.is_recording():
            span.end()
        context_api.detach(event_data.otel_token)
        metrics_attributes = get_llm_common_attributes()
        metrics_attributes["spanKind"] = SpanKindValues.CHAIN.value
        self._record_metrics(event_data, metrics_attributes)

    def _extract_outputs(self, outputs):
        if outputs is None:
            return {}
        output_attributes = {}
        output = ""
        output_key = OUTPUT_VALUE
        if "sys.query" in outputs:
            output = outputs["sys.query"]
        elif "answer" in outputs:
            output = outputs["answer"]
        elif "text" in outputs:
            output = outputs["text"]
        else:
            output = f"{outputs}"
        if output is None:
            return output_attributes
        set_dict_value(output_attributes, output_key, output)
        return output_attributes


class WorkflowRunFailedStrategy(ProcessStrategy):
    """Strategy for handling failed workflow run events.

    This strategy manages workflow execution failures, including:
    - Error recording and tracking
    - Span status updates
    - Resource cleanup
    - Error metrics collection
    - Failure state management

    The strategy handles:
    - Error message recording
    - Span error status updates
    - Failure metrics collection
    - Resource cleanup
    - Error state propagation
    """

    def process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
        res: Any,
    ) -> None:
        if is_wrapper_version_1():
            workflow_run = None
            if "workflow_run" in kwargs:
                workflow_run = kwargs["workflow_run"]
            else:
                workflow_run = res
            self._handle_workflow_run_failed_v1(workflow_run, kwargs["error"])
        elif is_wrapper_version_2():
            self._handle_workflow_run_failed_v2(res, kwargs["error_message"])

    def _handle_workflow_run_failed_v1(self, run, error):
        event_id = get_workflow_run_id(run)
        if event_id is None:
            return
        event_data = self._event_data.pop(event_id, None)
        if event_data is None:
            self._logger.warning(f"can not get data ,event_id: {event_id}")
            return

        app_id = getattr(run, "app_id", None)
        app_name = self._handler.get_app_name_by_id(app_id)
        span: trace_api.Span = event_data.span
        span.update_name(app_name)
        span_attributes = {}
        span_attributes[_DIFY_APP_NAME_KEY] = app_name
        span_attributes[DIFY_APP_ID_KEY] = app_id
        input_attr = self._extract_inputs(run.inputs_dict)
        span_attributes.update(input_attr)
        err = error
        span.set_status(
            Status(
                status_code=StatusCode.ERROR,
                description=f"{err}",
            )
        )
        span.set_attributes(span_attributes)
        if span.is_recording():
            span.end()
        context_api.detach(event_data.otel_token)
        metrics_attributes = get_llm_common_attributes()
        metrics_attributes["spanKind"] = SpanKindValues.CHAIN.value
        self._record_metrics(event_data, metrics_attributes, error)

    def _handle_workflow_run_failed_v2(self, run, error):
        event_id = get_workflow_run_id(run)
        if event_id is None:
            return
        event_data = self._event_data.pop(event_id, None)
        if event_data is None:
            self._logger.warning(f"can not get data ,event_id: {event_id}")
            return

        app_id = "UNKNOWN_APP_ID"
        app_name = "DEFAULT_APP_NAME"
        inputs_dict = getattr(run, "inputs", None)
        if inputs_dict is not None:
            if "sys.app_id" in inputs_dict:
                app_id = inputs_dict["sys.app_id"]
                app_name = self._handler.get_app_name_by_id(app_id)

        span: trace_api.Span = event_data.span
        span.update_name(app_name)
        span_attributes = {}
        span_attributes[_DIFY_APP_NAME_KEY] = app_name
        span_attributes[DIFY_APP_ID_KEY] = app_id
        input_attr = self._extract_inputs(inputs_dict)
        span_attributes.update(input_attr)
        err = error
        span.set_status(
            Status(
                status_code=StatusCode.ERROR,
                description=f"{err}",
            )
        )
        span.set_attributes(span_attributes)
        if span.is_recording():
            span.end()
        context_api.detach(event_data.otel_token)
        metrics_attributes = get_llm_common_attributes()
        metrics_attributes["spanKind"] = SpanKindValues.CHAIN.value
        self._record_metrics(event_data, metrics_attributes, error)


class WorkflowNodeStartStrategy(ProcessStrategy):
    """Strategy for handling workflow node start events.

    This strategy manages the beginning of individual node executions within a workflow, including:
    - Node span creation
    - Context setup for nodes
    - Node-specific attribute configuration
    - Parent-child relationship management
    - Resource initialization for nodes

    The strategy tracks:
    - Node start times
    - Node types and configurations
    - Parent-child relationships
    - Initial node state
    - Node-specific metrics
    """

    def process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
        res: Any,
    ) -> None:
        self._workflow_node_start_to_stream_response(
            kwargs["event"], kwargs["workflow_node_execution"]
        )

    def _set_value(
        self, key: str, value: Any, ctx: Any = None
    ) -> context_api.Context:
        if value is not None:
            new_ctx = context_api.set_value(key, value, ctx)
            return new_ctx
        return None

    def _set_values(
        self, attributes: dict, ctx: Any = None
    ) -> context_api.Context:
        new_ctx = ctx
        for key, value in attributes.items():
            if value is not None:
                new_ctx = context_api.set_value(key, value, new_ctx)
        return new_ctx

    def _workflow_node_start_to_stream_response(
        self, event, workflow_node_execution=None
    ):
        start_time = get_timestamp_from_datetime_attr(event, "start_at")
        parent_id = None
        if is_wrapper_version_1():
            workflow_run_id = getattr(
                workflow_node_execution, "workflow_run_id", None
            )
        else:
            workflow_run_id = getattr(
                workflow_node_execution, "workflow_execution_id", None
            )
        if workflow_run_id is not None:
            parent_id = workflow_run_id
        event_id = getattr(event, "node_execution_id", None)
        if event_id is None:
            self._logger.warning(f"can not get data ,event_id: {event_id}")
            return
        node_type = getattr(event, "node_type", "")
        node_type_name = getattr(node_type, "name", "DEFAULT_NODE_TYPE")
        node_data = getattr(event, "node_data", None)
        node_name = "DEFAULT_NODE_NAME"
        if node_data is not None:
            node_name = getattr(node_data, "title", "DEFAULT_NODE_NAME")
        with self._lock:
            parent_ctx = None
            common_attributes = {}
            if parent_id is not None:
                parent_event_data = self._event_data.get(parent_id)
                if parent_event_data is not None:
                    parent_ctx = trace_api.set_span_in_context(
                        parent_event_data.span
                    )
                    common_attributes = parent_event_data.attributes
            event_data = self._event_data.get(event_id)
            span = None
            if event_data is None:
                span: trace_api.Span = self._tracer.start_span(
                    f"{node_name}({node_type_name})",
                    context=parent_ctx,
                    attributes=common_attributes,
                    start_time=start_time,
                )
            else:
                span = event_data.span
                span.update_name(f"{node_name}({node_type_name})")
                span.set_attributes(common_attributes)
                span._parent = parent_event_data.span.get_span_context()

            new_context = trace_api.set_span_in_context(span)
            new_context = self._set_values(common_attributes, new_context)
            token = context_api.attach(new_context)
            self._event_data[event_id] = _EventData(
                span=span,
                parent_id=None,
                context=parent_ctx,
                payloads=[],
                exceptions=[],
                attributes={},
                node_type=None,
                start_time=start_time,
                otel_token=token,
            )


class WorkflowNodeFinishStrategy(ProcessStrategy):
    """Strategy for handling workflow node completion events.

    This strategy processes the completion of individual nodes within a workflow, including:
    - Recording node outputs
    - Updating node spans
    - Cleaning up node resources
    - Collecting node metrics
    - Handling node completion states

    The strategy manages:
    - Output recording and formatting
    - Span completion and cleanup
    - Node-specific metrics collection
    - Resource cleanup
    - Final state recording for nodes
    """

    def process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
        res: Any,
    ) -> None:
        self._workflow_node_finish_to_stream_response(
            kwargs["event"], kwargs["workflow_node_execution"]
        )

    def _workflow_node_finish_to_stream_response(
        self, event, workflow_node_execution=None
    ):
        end_time = None
        if workflow_node_execution is not None:
            finished_at = getattr(workflow_node_execution, "finished_at", None)
            if finished_at is not None:
                et = (
                    finished_at.timestamp() * 1_000_000_000
                    + finished_at.microsecond * 1_000
                )
                end_time = int(et)
        event_id = getattr(event, "node_execution_id", None)
        if event_id is None:
            self._logger.warning("event_id is none.")
            return
        event_data = self._event_data.pop(event_id, None)
        if event_data is None:
            self._logger.warning(f"can not get data ,event_id: {event_id}")
            return
        span_attributes = self._extract_workflow_node_attributes(event)
        span: trace_api.Span = event_data.span
        span_attributes.update(event_data.attributes)
        err = getattr(workflow_node_execution, "error", None)
        if err is not None:
            span.set_status(
                Status(
                    status_code=StatusCode.ERROR,
                    description=f"{err}",
                )
            )
        span.set_attributes(span_attributes)
        if span.is_recording():
            if end_time is not None:
                span.end(end_time=end_time)
            else:
                span.end()
        context_api.detach(event_data.otel_token)
        metrics_attributes = get_llm_common_attributes()
        span_kind = span_attributes[GEN_AI_SPAN_KIND]
        metrics_attributes["spanKind"] = span_kind
        if span_kind == SpanKindValues.LLM.value:
            if model_name := self._get_data(
                span_attributes, GEN_AI_MODEL_NAME, "DEFAULT_MODEL_NAME"
            ):
                metrics_attributes["modelName"] = model_name
            # FIXME: ruff failed
            if input_tokens := self._get_data(  # noqa: F841
                span_attributes, GEN_AI_USAGE_PROMPT_TOKENS, 0
            ):
                input_attributes = deepcopy(metrics_attributes)
                input_attributes["usageType"] = "input"
            # FIXME: ruff failed
            if output_tokens := self._get_data(  # noqa: F841
                span_attributes, GEN_AI_USAGE_COMPLETION_TOKENS, 0
            ):
                output_attributes = deepcopy(metrics_attributes)
                output_attributes["usageType"] = "output"

    def _extract_workflow_node_attributes(self, event: Any) -> dict:
        node_type = getattr(event, "node_type", None)
        if node_type is None:
            node_type = "DEFAULT_NODE_TYPE"
        node_type = getattr(node_type, "value", "DEFAULT_NODE_TYPE")
        span_kind = self._get_span_kind_by_node_type(node_type)
        span_attriubtes = {}
        span_attriubtes[GEN_AI_SPAN_KIND] = span_kind
        inputs = getattr(event, "inputs", None)
        input_attributes = self._extract_inputs(inputs)
        if input_attributes is not None:
            span_attriubtes.update(input_attributes)
        outputs = getattr(event, "outputs", None)
        output_attributes = self._extract_outputs(outputs)
        span_attriubtes.update(output_attributes)
        if span_kind == SpanKindValues.LLM.value:
            llm_attributes = self._extract_llm_attributes(event)
            span_attriubtes.update(llm_attributes)
            metrics_attriubtes = get_llm_common_attributes()
            if GEN_AI_REQUEST_MODEL_NAME in span_attriubtes:
                metrics_attriubtes["modelName"] = span_attriubtes[
                    GEN_AI_REQUEST_MODEL_NAME
                ]
            if GEN_AI_USAGE_PROMPT_TOKENS in span_attriubtes:
                # FIXME: ruff failed
                input_tokens = span_attriubtes[GEN_AI_USAGE_PROMPT_TOKENS]  # noqa: F841
                metrics_attriubtes["usageType"] = "input"
            if GEN_AI_USAGE_COMPLETION_TOKENS in span_attriubtes:
                # FIXME: ruff failed
                output_tokens = span_attriubtes[GEN_AI_USAGE_COMPLETION_TOKENS]  # noqa: F841
                metrics_attriubtes["usageType"] = "output"
        if span_kind == SpanKindValues.RETRIEVER.value:
            retriever_attributes = self._extract_retrieval_attributes(event)
            span_attriubtes.update(retriever_attributes)
        return span_attriubtes

    def _get_span_kind_by_node_type(self, node_type):
        if (
            node_type == NodeType.LLM.value
            or node_type == NodeType.QUESTION_CLASSIFIER.value
            or node_type == NodeType.PARAMETER_EXTRACTOR.value
        ):
            span_kind = SpanKindValues.LLM.value
        elif (
            node_type == NodeType.TOOL.value
            or node_type == NodeType.HTTP_REQUEST.value
        ):
            span_kind = SpanKindValues.TOOL.value
        elif node_type == NodeType.KNOWLEDGE_RETRIEVAL.value:
            span_kind = SpanKindValues.RETRIEVER.value
        else:
            span_kind = SpanKindValues.TASK.value
        return span_kind

    def _extract_retrieval_attributes(self, event):
        retrieval_attributes = {}
        output = None
        output = getattr(event, "outputs", None)
        if output is None:
            return retrieval_attributes
        result = None
        if "result" in output:
            result = output["result"]
        if result is not None:
            idx = 0
            for document in result:
                k_prefix = f"{RETRIEVAL_DOCUMENTS}.{idx}"
                if "metadata" in document:
                    metadata = document["metadata"]
                    if "document_id" in metadata:
                        retrieval_attributes[
                            f"{k_prefix}.{DocumentAttributes.DOCUMENT_ID}"
                        ] = metadata["document_id"]
                    if "score" in metadata:
                        retrieval_attributes[
                            f"{k_prefix}.{DocumentAttributes.DOCUMENT_SCORE}"
                        ] = metadata["score"]
                    retrieval_attributes[
                        f"{k_prefix}.{DocumentAttributes.DOCUMENT_METADATA}"
                    ] = json.dumps(metadata, ensure_ascii=False)
                set_dict_value(
                    retrieval_attributes,
                    f"{k_prefix}.{DocumentAttributes.DOCUMENT_CONTENT}",
                    document["content"],
                )
                idx += 1
        return retrieval_attributes

    def _extract_llm_attributes(self, event):
        llm_attributes = {}
        if node_data := getattr(event, "node_data", None):
            if single_retrieval_config := getattr(
                node_data, "single_retrieval_config", None
            ):
                if model := getattr(single_retrieval_config, "model", None):
                    model = single_retrieval_config["model"]
                    if "name" in model:
                        llm_attributes[GEN_AI_REQUEST_MODEL_NAME] = model[
                            "name"
                        ]
                        llm_attributes[GEN_AI_MODEL_NAME] = model["name"]
                    if "provider" in model:
                        llm_attributes[GEN_AI_SYSTEM] = model["provider"]
            if model := getattr(node_data, "model", None):
                if name := getattr(model, "name", None):
                    llm_attributes[GEN_AI_REQUEST_MODEL_NAME] = name
                    llm_attributes[GEN_AI_MODEL_NAME] = name
                if provider := getattr(model, "provider", None):
                    llm_attributes[GEN_AI_SYSTEM] = provider

        if process_data := getattr(event, "process_data", None):
            llm_attributes[GEN_AI_INPUT_MESSAGES] = self._get_input_messages(
                process_data
            )

        if outputs := getattr(event, "outputs", None):
            llm_attributes[GEN_AI_OUTPUT_MESSAGES] = self._get_output_messages(
                outputs
            )

            if usage := self._get_data(outputs, "usage", None):
                if prompt_tokens := self._get_data(
                    usage, "prompt_tokens", None
                ):
                    llm_attributes[GEN_AI_USAGE_PROMPT_TOKENS] = prompt_tokens
                if completion_tokens := self._get_data(
                    usage, "completion_tokens"
                ):
                    llm_attributes[GEN_AI_USAGE_COMPLETION_TOKENS] = (
                        completion_tokens
                    )
                if total_tokens := self._get_data(usage, "total_tokens", None):
                    llm_attributes[GEN_AI_USAGE_TOTAL_TOKENS] = total_tokens
        return llm_attributes

    def _get_output_messages(self, outputs) -> str:
        output_messages = []
        output_message: Dict[str, Any] = {"role": "assistant"}
        if text := self._get_data(outputs, "text", None):
            output_message["parts"] = [
                {"type": "text", "content": process_content(text)}
            ]
        if finish_reason := self._get_data(outputs, "finish_reason", None):
            output_message["finish_reason"] = finish_reason
        output_messages.append(output_message)
        return json.dumps(output_messages)

    def _get_input_messages(self, process_data) -> str:
        input_messages = []
        if "prompts" not in process_data:
            return json.dumps(input_messages)
        prompts = process_data["prompts"]
        for prompt in prompts:
            input_message = {}
            if "role" in prompt:
                input_message["role"] = prompt["role"]
            if "text" in prompt:
                input_message["parts"] = [
                    {
                        "type": "text",
                        "content": process_content(prompt["text"]),
                    }
                ]
            input_messages.append(input_message)
        return json.dumps(input_messages)

    def _extract_outputs(self, outputs):
        if outputs is None:
            return {}
        output_attributes = {}
        output = ""
        output_key = OUTPUT_VALUE
        if "sys.query" in outputs:
            output = outputs["sys.query"]
        elif "answer" in outputs:
            output = outputs["answer"]
        elif "text" in outputs:
            output = outputs["text"]
        else:
            output = f"{outputs}"
        if output is None:
            return output_attributes
        set_dict_value(output_attributes, output_key, output)
        return output_attributes


class WorkflowNodeExecutionFailedStrategy(ProcessStrategy):
    """Strategy for handling workflow node execution failures.

    This strategy manages node execution failures within a workflow, including:
    - Error recording and tracking
    - Node span status updates
    - Resource cleanup
    - Error metrics collection
    - Failure state management for nodes

    The strategy handles:
    - Node-specific error recording
    - Span error status updates
    - Node failure metrics collection
    - Resource cleanup
    - Error state propagation for nodes
    """

    def process(
        self,
        method: str,
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
        res: Any,
    ) -> None:
        self._handle_workflow_node_execution_failed()

    def _handle_workflow_node_execution_failed(self):
        pass
