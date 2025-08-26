import json
from typing import (
    Any,
    Iterable,
    Tuple,
    Dict,
    List,
)
from opentelemetry.util.types import AttributeValue
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

class AgentRunRequestExtractor(object):

    def extract(self, agent : Any, arguments : Dict[Any, Any]) -> Iterable[Tuple[str, AttributeValue]]:
        if agent.name:
            yield GenAIAttributes.GEN_AI_AGENT_NAME, f"{agent.name}"

        if agent.session_id:
            yield GenAIAttributes.GEN_AI_AGENT_ID, f"{agent.session_id}"

        if agent.knowledge:
            yield f"{GenAIAttributes.GEN_AI_AGENT_NAME}.knowledge", f"{agent.knowledge.__class__.__name__}"

        if agent.tools:
            tool_names = []
            from agno.tools.toolkit import Toolkit
            from agno.tools.function import Function
            for tool in agent.tools:
                if isinstance(tool, Function):
                    tool_names.append(tool.name)
                elif isinstance(tool, Toolkit):
                    tool_names.extend([f for f in tool.functions.keys()])
                elif callable(tool):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(tool))
            yield GenAIAttributes.GEN_AI_TOOL_NAME, ", ".join(tool_names)
        
        for key in arguments.keys():
            if key == "run_response":
                yield GenAIAttributes.GEN_AI_RESPONSE_ID, f"{arguments[key].run_id}"
            elif key == "run_messages":
                messages = arguments[key].messages
                for idx in range(len(messages)):  
                    message = messages[idx]
                    yield f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message", f"{json.dumps(message.to_dict(), indent=2)}"
            elif key == "response_format":
                yield GenAIAttributes.GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT, f"{arguments[key]}"

class AgentRunResponseExtractor(object):

    def extract(self, response : Any) -> Iterable[Tuple[str, AttributeValue]]:
        yield GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS, f"{response.to_json()}"

class FunctionCallRequestExtractor(object):

    def extract(self, function_call : Any) -> Iterable[Tuple[str, AttributeValue]]:

        if function_call.function.name:
            yield GenAIAttributes.GEN_AI_TOOL_NAME, f"{function_call.function.name}"

        if function_call.function.description:
            yield GenAIAttributes.GEN_AI_TOOL_DESCRIPTION, f"{function_call.function.description}"

        if function_call.call_id:
            yield GenAIAttributes.GEN_AI_TOOL_CALL_ID, f"{function_call.call_id}"

        if function_call.arguments:
            yield f"{GenAIAttributes.GEN_AI_TOOL_TYPE}.arguments", f"{json.dumps(function_call.arguments, indent=2)}"

class FunctionCallResponseExtractor(object):

    def extract(self, response : Any) -> Iterable[Tuple[str, AttributeValue]]:
        yield f"{GenAIAttributes.GEN_AI_TOOL_TYPE}.response", f"{response.result}"

class ModelRequestExtractor(object):

    def extract(self, model : Any, arguments : Dict[Any, Any]) -> Iterable[Tuple[str, AttributeValue]]:

        request_kwargs = {}
        if getattr(model, "request_kwargs", None):
            request_kwargs = model.request_kwargs
        if getattr(model, "request_params", None):
            request_kwargs = model.request_params
        if getattr(model, "get_request_kwargs", None):
            request_kwargs = model.get_request_kwargs()
        if getattr(model, "get_request_params", None):
            request_kwargs = model.get_request_params()

        if request_kwargs:
            yield GenAIAttributes.GEN_AI_REQUEST_MODEL, f"{json.dumps(request_kwargs, indent=2)}"
        
        for key in arguments.keys():
            if key == "response_format":
                yield GenAIAttributes.GEN_AI_OPENAI_REQUEST_RESPONSE_FORMAT,f"{arguments[key]}"
            elif key == "messages":
                messages = arguments["messages"]
                for idx in range(len(messages)):
                    message = messages[idx]
                    yield f"{GenAIAttributes.GEN_AI_PROMPT}.{idx}.message", f"{json.dumps(message.to_dict(), indent=2)}"
            elif key == "tools":
                tools = arguments["tools"]
                for idx in range(len(tools)):
                    yield f"{GenAIAttributes.GEN_AI_TOOL_DESCRIPTION}.{idx}", f"{json.dumps(tools[idx], indent=2)}"

class ModelResponseExtractor(object):

    def extract(self, responses: List[Any]) -> Iterable[Tuple[str, AttributeValue]]:
        content = ""
        for response in responses:
        # basic response fields
            if getattr(response, "role", None):
                yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.role", response.role
            if getattr(response, "content", None):
                content += response.content
            if getattr(response, "audio", None):
                yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.audio", json.dumps(response.audio.to_dict(), indent=2)
            if getattr(response, "image", None):
                yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.image", json.dumps(response.image.to_dict(), indent=2)
            for idx, exec in enumerate(getattr(response, "tool_executions", []) or []):
                yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.tool_executions.{idx}", json.dumps(exec.to_dict(), indent=2)
            # other metadata
            if getattr(response, "event", None):
                yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.event", response.event
            if getattr(response, "provider_data", None):
                yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.provider_data", json.dumps(response.provider_data, indent=2)
            if getattr(response, "thinking", None):
                yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.thinking", response.thinking
            if getattr(response, "redacted_thinking", None):
                yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.redacted_thinking", response.redacted_thinking
            if getattr(response, "reasoning_content", None):
                yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.reasoning_content", response.reasoning_content
            if getattr(response, "extra", None):
                yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.extra", json.dumps(response.extra, indent=2)
        if len(content):
            yield f"{GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS}.content", f"{content}"
        