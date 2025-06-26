from abc import ABC, abstractmethod
import json
import time

from opentelemetry.semconv.trace import SpanAttributes, MessageAttributes, SpanKindValues, DocumentAttributes

#  dify packages path
fromopentelemetry.sdk.extension.semconv.metrics import CommonServiceMetrics
from opentelemetry.metrics import get_meter
from logging import getLogger
from abc import ABC
from opentelemetry import trace as trace_api
from opentelemetry import context as context_api
from opentelemetry.trace.status import Status, StatusCode
from typing import (
    Any,
    Mapping,
    Tuple,
)
from opentelemetry.instrumentation.dify.entities import NodeType
# dify packages
from opentelemetry.instrumentation.dify.version import __version__
from opentelemetry.sdk.extension.arms.semconv.attributes import arms_attributes
from opentelemetry.sdk.extension.arms.common.utils.metrics_utils import get_llm_common_attributes
from copy import deepcopy
from opentelemetry.instrumentation.dify.contants import _get_dify_app_name_key, DIFY_APP_ID_KEY
from .entities import _EventData

_logger = getLogger(__name__)

_DIFY_APP_NAME_KEY = _get_dify_app_name_key()


class ProcessStrategy(ABC):
    """Base abstract class for all process strategies in Dify instrumentation.

    This class provides the foundation for tracking and monitoring different aspects of Dify's operations.
    It handles metrics collection, tracing, and event management for various Dify components.

    Attributes:
        _handler: The handler instance that manages the overall instrumentation process
        _tracer: OpenTelemetry tracer for creating and managing spans
        _lock: Thread lock for thread-safe operations
        _event_data: Dictionary storing event-related data
        _logger: Logger instance for recording events and errors
        _meter: OpenTelemetry meter for collecting metrics
    """

    def __init__(self, handler: Any):
        self._handler = handler
        self._tracer = handler._tracer
        self._lock = handler._lock
        self._event_data = handler._event_data
        self._logger = handler._logger
        self._meter = get_meter(
            __name__,
            __version__,
            None,
            schema_url="https://opentelemetry.io/schemas/1.11.0",
        )
        self._init_metrics()

    def _init_metrics(self):
        meter = self._meter
        self.calls_count = CommonServiceMetrics(meter).calls_count
        self.calls_duration_seconds = CommonServiceMetrics(meter).calls_duration_seconds
        self.llm_context_size = CommonServiceMetrics(meter).llm_context_size
        self.llm_prompt_size = CommonServiceMetrics(meter).llm_prompt_size
        self.llm_output_token_seconds = CommonServiceMetrics(meter).llm_output_token_seconds
        self.llm_usage_tokens = CommonServiceMetrics(meter).llm_usage_tokens
        self.llm_first_token_seconds = CommonServiceMetrics(meter).llm_first_token_seconds
        self.calls_error_count = CommonServiceMetrics(meter).call_error_count

    def _record_metrics(self, event_data, metrics_attributes, error=None):
        self.calls_count.add(1, attributes=metrics_attributes)
        if error is not None:
            self.calls_error_count.add(1, attributes=metrics_attributes)
        duration = (time.time_ns() - event_data.start_time) / 1_000_000_000
        self.calls_duration_seconds.record(duration, attributes=metrics_attributes)

    def _extract_inputs(self, inputs):
        if inputs is None:
            return {}
        input_attributes = {}
        input_value = ""
        input_key = SpanAttributes.INPUT_VALUE
        if inputs is None:
            input_attributes[input_key] = "{}"
            return input_attributes
        if "sys.query" in inputs:
            input_value = inputs["sys.query"]
        else:
            input_value = f"{inputs}"
        if input_value is None:
            return input_attributes
        input_attributes[input_key] = input_value
        return input_attributes


    @abstractmethod
    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        pass

    def before_process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], ):
        pass

    def _get_data(self, src, key, default=None):
        if key in src:
            return src[key]
        else:
            return default

    def _get_message_data(self, message_id: str):
        return self._handler._get_message_data(message_id)


class AgentChatAppRunnerStrategy(ProcessStrategy):
    """Strategy for handling agent chat application runner events.

    This strategy manages the lifecycle of agent chat sessions, including:
    - Initialization of chat sessions
    - Message handling and processing
    - Context management for agent-based conversations
    - Span creation and management for agent interactions
    - Metrics collection for agent operations

    The strategy tracks:
    - User sessions and conversations
    - Agent responses and interactions
    - Performance metrics for agent operations
    - Error handling and reporting
    """

    def before_process(self, method: str, instance: Any, args: Tuple[type], kwargs: Mapping[str, Any], ):
        message = self._get_data(kwargs, "message", None)
        if message is None:
            message_id = getattr(instance, "_message_id", None)
            message = self._get_message_data(message_id)
        event_id = getattr(message, "id", None)
        self._handle_agent_start_message(event_id, message)

    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        pass

    def _set_value(self, key: str, value: Any, ctx: Any = None) -> context_api.Context:
        if value is not None:
            new_ctx = context_api.set_value(key, value, ctx)
            return new_ctx
        return None

    def _handle_agent_start_message(self, event_id, message):
        start_time = time.time_ns()
        with self._lock:
            data = self._event_data.get(event_id)
            if data is not None:
                return None
        span: trace_api.Span = self._tracer.start_span(f"agent_{event_id}", attributes={
            SpanAttributes.GEN_AI_SPAN_KIND: SpanKindValues.AGENT.value, "component.name": "dify"},
                                                       start_time=start_time,
                                                       )
        app_id = getattr(message, "app_id", None)
        app_name = self._handler.get_app_name_by_id(app_id)
        session_id = getattr(message, "conversation_id", "DEFAULT_SESSION_ID")
        user_id = getattr(message, "from_account_id", "DEFAULT_USER_ID")

        span.set_attribute(SpanAttributes.GEN_AI_USER_ID, user_id)
        span.set_attribute(SpanAttributes.GEN_AI_SESSION_ID, session_id)
        span.set_attribute(DIFY_APP_ID_KEY, app_id)
        span.set_attribute(_DIFY_APP_NAME_KEY, app_name)
        span.update_name(f"{app_name}({SpanKindValues.AGENT.value})")

        new_context = trace_api.set_span_in_context(span)
        new_context = self._set_value(_DIFY_APP_NAME_KEY, app_name, ctx=new_context)
        new_context = self._set_value(DIFY_APP_ID_KEY, app_id, ctx=new_context)
        new_context = self._set_value(SpanAttributes.GEN_AI_USER_ID, user_id, ctx=new_context)
        new_context = self._set_value(SpanAttributes.GEN_AI_SESSION_ID, session_id, ctx=new_context)
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
                    arms_attributes.COMPONENT_NAME: arms_attributes.ComponentNameValue.DIFY.value,
                    SpanAttributes.GEN_AI_USER_ID: user_id,
                    SpanAttributes.GEN_AI_SESSION_ID: session_id,
                },
                node_type=None,
                start_time=start_time,
                otel_token=None,
            )


class MessageEndStrategy(ProcessStrategy):
    """Strategy for handling message end events in conversations.

    This strategy processes the completion of messages in conversations, including:
    - Finalizing spans for completed messages
    - Recording message outputs and responses
    - Cleaning up resources associated with the message
    - Handling agent thoughts and final answers
    - Updating metrics for completed messages

    The strategy ensures proper cleanup and recording of:
    - Message queries and answers
    - Agent thought processes
    - Final response content
    - Performance metrics
    """

    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        task_state = getattr(instance, "_task_state", None)
        message = getattr(instance, "_message", None)
        if message is None:
            message_id = getattr(instance, "_message_id", None)
            message = self._get_message_data(message_id)
        try:
            self._handle_agent_end_message(task_state, message)
        except Exception as e:
            self._logger.warning(f"[_handle_agent_end_message] error, error info: {e}")

    def _handle_agent_end_message(self, task_state, message):
        if task_state is None:
            self._logger.warning("task_state is None, skipping agent end message handling")
            return
        if message is None:
            self._logger.warning("message is None, skipping agent end message handling")
            return
        event_id = getattr(message, "id", None)
        if event_id not in self._event_data:
            self._logger.warning("event_is is not in event data")
            return
        with self._lock:
            event_data = self._event_data.pop(event_id)
            span: trace_api.Span = event_data.span
            if query := getattr(message, "query", None):
                span.set_attribute(SpanAttributes.INPUT_VALUE, f"{query}")
            if answer := getattr(message, "answer", None):
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"{answer}")
            if agent_thoughts := getattr(message, "agent_thoughts", None):
                if isinstance(agent_thoughts, list) and len(agent_thoughts) > 0:
                    last_thought = agent_thoughts[-1]
                    if last_answer := getattr(last_thought, "answer", None):
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"{last_answer}")
            if span.is_recording():
                span.end()


class InitStrategy(ProcessStrategy):
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

    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        try:
            event_id = kwargs["id"]
            span = None
            with self._lock:
                from opentelemetry import context
                ctx = context.get_current()
                if len(ctx) == 0:
                    self._logger.info(f"can't get ctx, return : {ctx},kwargs: {kwargs},event_id: {event_id}")
                    graph_runtime_state = kwargs["graph_runtime_state"]
                    previous_node_id = kwargs["previous_node_id"]
                    if graph_runtime_state is None:
                        return
                    node_run_state = getattr(graph_runtime_state, "node_run_state")
                    node_state_mapping = getattr(node_run_state, "node_state_mapping")
                    for k, v in node_state_mapping.items():
                        event_data = self._event_data.get(k)
                        if event_data is not None:
                            start_time = time.time_ns()
                            parent_ctx = trace_api.set_span_in_context(event_data.span)
                            span: trace_api.Span = self._tracer.start_span(f"node_run_{event_id}", attributes={
                                SpanAttributes.GEN_AI_SPAN_KIND: SpanKindValues.CHAIN.value,
                                "component.name": "dify"}, start_time=start_time, context=parent_ctx)
                            new_ctx = trace_api.set_span_in_context(span)
                            token = context_api.attach(new_ctx)
                            self._logger.info(f"node event id: {k},event_data: {event_data} {event_data.span}")
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
                    span: trace_api.Span = self._tracer.start_span(f"node_run_{event_id}", attributes={
                        SpanAttributes.GEN_AI_SPAN_KIND: SpanKindValues.CHAIN.value,
                        "component.name": "dify"}, start_time=start_time)
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
        except:
            self._logger.exception("Fail to process data, func name: BaseNode.__init__")


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

    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        self._handle_workflow_run_start(res)

    def _handle_workflow_run_start(self, run):
        event_id = getattr(run, "id", None)
        if event_id is None:
            self._logger.warning("workflow_run_start: missing event_id", extra={
                "run_object": str(run),
                "component": "workflow_handler"
            })
            return
        created_at = getattr(run, "created_at", None)
        start_time = None
        if created_at is not None:
            st = created_at.timestamp() * 1_000_000_000 + created_at.microsecond * 1_000
            start_time = int(st)
        if start_time is None:
            start_time = time.time_ns()
        with self._lock:
            data = self._event_data.get(event_id)
            if data is not None:
                return
        span: trace_api.Span = self._tracer.start_span(f"workflow_run_{event_id}", attributes={
            SpanAttributes.GEN_AI_SPAN_KIND: SpanKindValues.CHAIN.value, "component.name": "dify"},
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
        span.set_attribute(SpanAttributes.GEN_AI_USER_ID, user_id)
        span.set_attribute(SpanAttributes.GEN_AI_SESSION_ID, session_id)
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
                    arms_attributes.COMPONENT_NAME: arms_attributes.ComponentNameValue.DIFY.value,
                    SpanAttributes.GEN_AI_USER_ID: user_id,
                    SpanAttributes.GEN_AI_SESSION_ID: session_id,
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

    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        workflow_run = None
        if "workflow_run" in kwargs:
            workflow_run = kwargs['workflow_run']
        else:
            workflow_run = res
        self._handle_workflow_run_success(workflow_run, outputs=kwargs['outputs'])

    def _handle_workflow_run_success(self, run, outputs=[]):
        event_id = getattr(run, "id", None)
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


    def _extract_outputs(self, outputs):
        if outputs is None:
            return {}
        output_attributes = {}
        output = ""
        output_key = SpanAttributes.OUTPUT_VALUE
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
        output_attributes[output_key] = output
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

    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        workflow_run = None
        if "workflow_run" in kwargs:
            workflow_run = kwargs['workflow_run']
        else:
            workflow_run = res
        self._handle_workflow_run_failed(workflow_run, kwargs['error'])

    def _handle_workflow_run_failed(self, run, error):
        event_id = getattr(run, "id", None)
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

    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        self._workflow_node_start_to_stream_response(kwargs['event'], kwargs['workflow_node_execution'])

    def _set_value(self, key: str, value: Any, ctx: Any = None) -> context_api.Context:
        if value is not None:
            new_ctx = context_api.set_value(key, value, ctx)
            return new_ctx
        return None

    def _set_values(self, attributes: dict, ctx: Any = None) -> context_api.Context:
        new_ctx = ctx
        for key, value in attributes.items():
            if value is not None:
                new_ctx = context_api.set_value(key, value, new_ctx)
        return new_ctx

    def _workflow_node_start_to_stream_response(self, event, workflow_node_execution=None):
        start_at = getattr(event, "start_at", None)
        start_time = None
        if start_time is None:
            start_time = time.time_ns()
        parent_id = None
        workflow_run_id = getattr(workflow_node_execution, "workflow_run_id", None)
        if workflow_run_id is not None:
            parent_id = workflow_run_id
        event_id = getattr(event, "node_execution_id", None)
        if event_id is None:
            self._logger.warning(f"can not get data ,event_id: {event_id}")
            return
        node_type = getattr(event, "node_type", None)
        node_type_name = "DEFAULT_NODE_TYPE"
        if node_type is None:
            node_type = ""
        node_type_name = getattr(node_type, "name", "DEFAULT_NODE_TYPE")
        node_data = getattr(event, "node_data", None)
        node_name = "DEFAULT_NODE_NAME"
        if node_data is not None:
            node_name = getattr(node_data, "title", "DEFAULT_NODE_NAME")
        with self._lock:
            parent_ctx = None
            if parent_id is not None:
                parent_event_data = self._event_data.get(parent_id)
                if parent_event_data is not None:
                    parent_ctx = trace_api.set_span_in_context(parent_event_data.span)
                    common_attributes = parent_event_data.attributes
            event_data = self._event_data.get(event_id)
            span = None
            if event_data is None:
                span: trace_api.Span = self._tracer.start_span(f"{node_name}({node_type_name})", context=parent_ctx,
                                                               attributes=common_attributes, start_time=start_time)
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

    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        self._workflow_node_finish_to_stream_response(kwargs['event'], kwargs['workflow_node_execution'])

    def _workflow_node_finish_to_stream_response(self, event, workflow_node_execution=None):
        end_time = None
        if workflow_node_execution is not None:
            finished_at = getattr(workflow_node_execution, "finished_at", None)
            if finished_at is not None:
                et = finished_at.timestamp() * 1_000_000_000 + finished_at.microsecond * 1_000
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
        span_kind = span_attributes[SpanAttributes.GEN_AI_SPAN_KIND]
        metrics_attributes["spanKind"] = span_kind
        if span_kind == SpanKindValues.LLM.value:
            if model_name := self._get_data(span_attributes, SpanAttributes.GEN_AI_MODEL_NAME, "DEFAULT_MODEL_NAME"):
                metrics_attributes["modelName"] = model_name
            if input_tokens := self._get_data(span_attributes, SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS, 0):
                input_attributes = deepcopy(metrics_attributes)
                input_attributes["usageType"] = "input"
                self.llm_usage_tokens.add(input_tokens, attributes=input_attributes)
            if output_tokens := self._get_data(span_attributes, SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS, 0):
                output_attributes = deepcopy(metrics_attributes)
                output_attributes["usageType"] = "output"
                self.llm_usage_tokens.add(output_tokens, attributes=output_attributes)

        self._record_metrics(event_data, metrics_attributes, err)

    def _extract_workflow_node_attributes(self, event: Any) -> str:
        node_type = getattr(event, "node_type", None)
        if node_type is None:
            node_type = "DEFAULT_NODE_TYPE"
        node_type = getattr(node_type, "value", "DEFAULT_NODE_TYPE")
        span_kind = self._get_span_kind_by_node_type(node_type)
        span_attriubtes = {}
        span_attriubtes[SpanAttributes.GEN_AI_SPAN_KIND] = span_kind
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
            if SpanAttributes.GEN_AI_REQUEST_MODEL_NAME in span_attriubtes:
                metrics_attriubtes["modelName"] = span_attriubtes[SpanAttributes.GEN_AI_REQUEST_MODEL_NAME]
            if SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS in span_attriubtes:
                input_tokens = span_attriubtes[SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS]
                metrics_attriubtes["usageType"] = "input"
            if SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS in span_attriubtes:
                output_tokens = span_attriubtes[SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS]
                metrics_attriubtes["usageType"] = "output"
        if span_kind == SpanKindValues.RETRIEVER.value:
            retriever_attributes = self._extract_retrieval_attributes(event)
            span_attriubtes.update(retriever_attributes)
        return span_attriubtes

    def _get_span_kind_by_node_type(self, node_type):
        span_attributes = {}
        if (node_type == NodeType.LLM.value
                or node_type == NodeType.QUESTION_CLASSIFIER.value
                or node_type == NodeType.PARAMETER_EXTRACTOR.value):
            span_kind = SpanKindValues.LLM.value
        elif node_type == NodeType.TOOL.value or node_type == NodeType.HTTP_REQUEST.value:
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
                k_prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{idx}"
                if "metadata" in document:
                    metadata = document["metadata"]
                    if "document_id" in metadata:
                        retrieval_attributes[f"{k_prefix}.{DocumentAttributes.DOCUMENT_ID}"] = metadata["document_id"]
                    if "score" in metadata:
                        retrieval_attributes[f"{k_prefix}.{DocumentAttributes.DOCUMENT_SCORE}"] = metadata["score"]
                    retrieval_attributes[f"{k_prefix}.{DocumentAttributes.DOCUMENT_METADATA}"] = json.dumps(metadata,
                                                                                                            ensure_ascii=False)
                retrieval_attributes[f"{k_prefix}.{DocumentAttributes.DOCUMENT_CONTENT}"] = document["content"]
                idx += 1
        return retrieval_attributes

    def _extract_llm_attributes(self, event):
        llm_attributes = {}
        if node_data := getattr(event, "node_data", None):
            if single_retrieval_config := getattr(node_data, "single_retrieval_config", None):
                if model := getattr(single_retrieval_config, "model", None):
                    model = single_retrieval_config["model"]
                    if "name" in model:
                        llm_attributes[SpanAttributes.GEN_AI_REQUEST_MODEL_NAME] = model["name"]
                        llm_attributes[SpanAttributes.GEN_AI_MODEL_NAME] = model["name"]
                    if "provider" in model:
                        llm_attributes[SpanAttributes.GEN_AI_SYSTEM] = model["provider"]
            if model := getattr(node_data, "model", None):
                if name := getattr(model, "name", None):
                    llm_attributes[SpanAttributes.GEN_AI_REQUEST_MODEL_NAME] = name
                    llm_attributes[SpanAttributes.GEN_AI_MODEL_NAME] = name
                if provider := getattr(model, "provider", None):
                    llm_attributes[SpanAttributes.GEN_AI_SYSTEM] = provider

        if process_data := getattr(event, "process_data", None):
            if "prompts" in process_data:
                prompts = process_data["prompts"]
                idx = 0
                for prompt in prompts:
                    if "role" in prompt:
                        llm_attributes[
                            f"{SpanAttributes.GEN_AI_PROMPT}.{idx}.{MessageAttributes.MESSAGE_ROLE}"] = \
                            f"{prompt['role']}"
                    if "text" in prompt:
                        llm_attributes[
                            f"{SpanAttributes.GEN_AI_PROMPT}.{idx}.{MessageAttributes.MESSAGE_CONTENT}"] = \
                            f"{prompt['text']}"
                    idx += 1
            if usage := self._get_data(src=process_data, key="usage"):
                if prompt_tokens := self._get_data(src=usage, key="prompt_tokens"):
                    llm_attributes[SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = prompt_tokens
                if completion_tokens := self._get_data(src=usage, key="completion_tokens"):
                    llm_attributes[SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = completion_tokens
                if total_tokens := self._get_data(src=usage, key="total_tokens"):
                    llm_attributes[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] = total_tokens
            if finish_reason := self._get_data(src=process_data, key="finish_reason"):
                llm_attributes[SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON] = finish_reason

        if outputs := getattr(event, "outputs", None):
            if text := self._get_data(outputs, "text", None):
                key = f"{SpanAttributes.GEN_AI_PROMPT}.{SpanAttributes.CONTENT}"
                llm_attributes[key] = text
            if usage := self._get_data(outputs, "usage", None):
                if prompt_tokens := self._get_data(usage, "prompt_tokens", None):
                    llm_attributes[SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS] = prompt_tokens
                if completion_tokens := self._get_data(usage, "completion_tokens"):
                    llm_attributes[SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS] = completion_tokens
                if total_tokens := self._get_data(usage, "total_tokens", None):
                    llm_attributes[SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS] = total_tokens
            if finish_reason := self._get_data(outputs, "finish_reason", None):
                llm_attributes[SpanAttributes.GEN_AI_RESPONSE_FINISH_REASON] = finish_reason
        return llm_attributes

    def _extract_outputs(self, outputs):
        if outputs is None:
            return {}
        output_attributes = {}
        output = ""
        output_key = SpanAttributes.OUTPUT_VALUE
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
        output_attributes[output_key] = output
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

    def process(self, method: str, instance: Any, args: Tuple[type, Any], kwargs: Mapping[str, Any], res: Any) -> None:
        self._handle_workflow_node_execution_failed()

    def _handle_workflow_node_execution_failed(self):
        pass


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
            "run": AgentChatAppRunnerStrategy(handler),
            "_message_end_to_stream_response": MessageEndStrategy(handler),
            "__init__": InitStrategy(handler),
            "_handle_workflow_run_start": WorkflowRunStartStrategy(handler),
            "_handle_workflow_run_success": WorkflowRunSuccessStrategy(handler),
            "_handle_workflow_run_failed": WorkflowRunFailedStrategy(handler),
            "_workflow_node_start_to_stream_response": WorkflowNodeStartStrategy(handler),
            "_workflow_node_finish_to_stream_response": WorkflowNodeFinishStrategy(handler),
            "_handle_workflow_node_execution_failed": WorkflowNodeExecutionFailedStrategy(handler),
        }

    def get_strategy(self, method: str) -> ProcessStrategy:
        for suffix, strategy in self._strategies.items():
            if method.endswith(suffix):
                return strategy
        return None
