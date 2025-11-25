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

"""Utility functions for DashScope instrumentation."""

import json
from typing import Any, List, Optional

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    LLMInvocation,
    OutputMessage,
    Text,
    ToolCall,
    ToolCallResponse,
    ToolDefinitions,
)


def _get_parameter(kwargs: dict, param_name: str, parameters: Optional[dict] = None) -> Optional[Any]:
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


def _extract_input_messages(kwargs: dict) -> List[InputMessage]:
    """Extract input messages from DashScope API kwargs.

    DashScope supports both `prompt` (string) and `messages` (list) formats.
    Also supports tool call responses in messages.
    """
    input_messages = []

    # Check for messages format (preferred)
    if "messages" in kwargs and kwargs["messages"]:
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                tool_call_id = msg.get("tool_call_id")
                tool_calls = msg.get("tool_calls")
                
                parts = []
                
                # Handle tool call response (role="tool")
                if role == "tool":
                    # For tool role, use ToolCallResponse
                    parts.append(
                        ToolCallResponse(
                            response=content,
                            id=tool_call_id,
                            type="tool_call_response",
                        )
                    )
                else:
                    # Add text content if present (for non-tool roles)
                    if content:
                        if isinstance(content, str):
                            parts.append(Text(content=content, type="text"))
                        elif isinstance(content, list):
                            # Handle multimodal content (qwen-vl)
                            for part in content:
                                if isinstance(part, dict):
                                    if "text" in part:
                                        parts.append(Text(content=part["text"], type="text"))
                                elif isinstance(part, str):
                                    parts.append(Text(content=part, type="text"))
                
                # Add tool calls if present (for assistant messages with tool calls)
                if tool_calls and isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tc_id = tool_call.get("id")
                            function = tool_call.get("function", {})
                            tool_name = function.get("name") if isinstance(function, dict) else None
                            tool_args = function.get("arguments") if isinstance(function, dict) else None
                            
                            if tool_name:
                                parts.append(
                                    ToolCall(
                                        name=tool_name,
                                        arguments=tool_args,
                                        id=tc_id,
                                        type="tool_call",
                                    )
                                )
                
                if parts:
                    input_messages.append(
                        InputMessage(
                            role=role,
                            parts=parts,
                        )
                    )
                elif content:  # Fallback for text-only messages
                    input_messages.append(
                        InputMessage(
                            role=role,
                            parts=[Text(content=str(content), type="text")],
                        )
                    )
            elif hasattr(msg, "role"):
                # Handle message objects
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
                tool_call_id = getattr(msg, "tool_call_id", None)
                tool_calls = getattr(msg, "tool_calls", None)
                
                parts = []
                
                # Handle tool call response (role="tool")
                if role == "tool" :
                    # For tool role, use ToolCallResponse
                    parts.append(
                        ToolCallResponse(
                            response=str(content),
                            id=tool_call_id,
                            type="tool_call_response",
                        )
                    )
                else:
                    # Add text content if present (for non-tool roles)
                    if content:
                        if isinstance(content, str):
                            parts.append(Text(content=content, type="text"))
                        elif isinstance(content, list):
                            # TODO: Handle multimodal content
                            for part in content:
                                if isinstance(part, dict) and "text" in part:
                                    parts.append(Text(content=part["text"], type="text"))
                                elif isinstance(part, str):
                                    parts.append(Text(content=part, type="text"))
                
                # Add tool calls if present
                if tool_calls:
                    for tool_call in tool_calls:
                        if hasattr(tool_call, "function"):
                            function = getattr(tool_call, "function", None)
                            if function:
                                tool_name = getattr(function, "name", None)
                                tool_args = getattr(function, "arguments", None)
                                tc_id = getattr(tool_call, "id", None)
                                
                                if tool_name:
                                    parts.append(
                                        ToolCall(
                                            name=tool_name,
                                            arguments=tool_args,
                                            id=tc_id,
                                            type="tool_call",
                                        )
                                    )
                
                if parts:
                    input_messages.append(
                        InputMessage(
                            role=role,
                            parts=parts,
                        )
                    )
                elif content:
                    input_messages.append(
                        InputMessage(
                            role=role,
                            parts=[Text(content=str(content), type="text")],
                        )
                    )

    # Check for prompt format (legacy)
    elif "prompt" in kwargs and kwargs["prompt"]:
        prompt = kwargs["prompt"]
        if isinstance(prompt, str):
            input_messages.append(
                InputMessage(
                    role="user",
                    parts=[Text(content=prompt, type="text")],
                )
            )

    return input_messages


def _extract_tool_definitions(kwargs: dict) -> ToolDefinitions:
    """Extract tool definitions from DashScope API kwargs and convert to FunctionToolDefinition objects.
    
    DashScope supports both `tools` and `plugins` parameters for tool definitions.
    - `tools`: Direct list of tool definitions (preferred)
    - `plugins`: Can be a string (JSON) or a dict containing tools
    
    Args:
        kwargs: Generation.call kwargs
        
    Returns:
        List of FunctionToolDefinition objects, or empty list if not found
    """
    tool_definitions: ToolDefinitions = []
    
    # Check for tools parameter first (preferred)
    tools = kwargs.get("tools")
    if not tools:
        # Check for plugins parameter
        plugins = kwargs.get("plugins")
        if plugins:
            try:
                # If plugins is a string, parse it as JSON
                if isinstance(plugins, str):
                    plugins = json.loads(plugins)
                
                # If plugins is a dict, extract tools
                if isinstance(plugins, dict):
                    # DashScope plugins format: {"tools": [...]} or direct list
                    if "tools" in plugins:
                        tools = plugins["tools"]
                    # Check if plugins itself is a list-like structure
                    elif isinstance(plugins, list):
                        tools = plugins
                
                # If plugins is already a list, use it
                if isinstance(plugins, list):
                    tools = plugins
            except (json.JSONDecodeError, TypeError, AttributeError):
                # If parsing fails, return empty list
                return tool_definitions
    
    # Convert tool definitions to FunctionToolDefinition objects
    if tools and isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                # Extract function definition (DashScope format: {"type": "function", "function": {...}})
                function = tool.get("function", {})
                if isinstance(function, dict):
                    tool_def = FunctionToolDefinition(
                        name=function.get("name", ""),
                        description=function.get("description"),
                        parameters=function.get("parameters"),
                        response=function.get("response"),
                        type="function",
                    )
                    tool_definitions.append(tool_def)
                # Handle case where tool itself is a function definition
                elif "name" in tool:
                    tool_def = FunctionToolDefinition(
                        name=tool.get("name", ""),
                        description=tool.get("description"),
                        parameters=tool.get("parameters"),
                        response=tool.get("response"),
                        type="function",
                    )
                    tool_definitions.append(tool_def)
            elif isinstance(tool, FunctionToolDefinition):
                # Already a FunctionToolDefinition, add directly
                tool_definitions.append(tool)
    
    return tool_definitions


def _extract_output_messages(response: Any) -> List[OutputMessage]:
    """Extract output messages from DashScope GenerationResponse.

    Supports both standard format (output.text) and qwen-vl format (output.choices).

    Args:
        response: DashScope GenerationResponse object

    Returns:
        List of OutputMessage objects
    """
    output_messages = []

    if not response:
        return output_messages

    try:
        # Use getattr with default None to safely access attributes
        # DashScope response uses __getattr__ which raises KeyError for missing attributes
        output = getattr(response, "output", None)
        if not output:
            return output_messages

        # Check for choices format (qwen-vl and some models)
        choices = getattr(output, "choices", None)
        if choices and isinstance(choices, list) and len(choices) > 0:
            # Process each choice
            for choice in choices:
                if not choice:
                    continue
                
                # Extract message from choice
                message = getattr(choice, "message", None)
                if not message:
                    continue
                
                # Extract content and tool_calls
                content = getattr(message, "content", None)
                tool_calls = getattr(message, "tool_calls", None)
                finish_reason = getattr(choice, "finish_reason", None) or getattr(
                    output, "finish_reason", "stop"
                )
                
                parts = []
                
                # Add text content if present
                if content:
                    if isinstance(content, str):
                        parts.append(Text(content=content, type="text"))
                    elif isinstance(content, list):
                        # Handle multimodal content (qwen-vl)
                        for part in content:
                            if isinstance(part, dict):
                                if "text" in part:
                                    parts.append(Text(content=part["text"], type="text"))
                            elif isinstance(part, str):
                                parts.append(Text(content=part, type="text"))
                
                # Add tool calls if present
                if tool_calls and isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call_id = tool_call.get("id")
                            function = tool_call.get("function", {})
                            tool_name = function.get("name") if isinstance(function, dict) else None
                            tool_args = function.get("arguments") if isinstance(function, dict) else None
                            
                            if tool_name:
                                parts.append(
                                    ToolCall(
                                        name=tool_name,
                                        arguments=tool_args,
                                        id=tool_call_id,
                                        type="tool_call",
                                    )
                                )
                        elif hasattr(tool_call, "function"):
                            # Handle tool call objects
                            function = getattr(tool_call, "function", None)
                            if function:
                                tool_name = getattr(function, "name", None)
                                tool_args = getattr(function, "arguments", None)
                                tool_call_id = getattr(tool_call, "id", None)
                                
                                if tool_name:
                                    parts.append(
                                        ToolCall(
                                            name=tool_name,
                                            arguments=tool_args,
                                            id=tool_call_id,
                                            type="tool_call",
                                        )
                                    )
                
                # Create output message if we have parts OR if finish_reason indicates tool_calls
                # (even if content is empty, tool calls should be captured)
                if parts or (finish_reason == "tool_calls" and tool_calls):
                    output_messages.append(
                        OutputMessage(
                            role="assistant",
                            parts=parts,
                            finish_reason=finish_reason or "stop",
                        )
                    )
        else:
            # Standard format: output.text
            text = getattr(output, "text", None) or getattr(output, "content", None)
            finish_reason = getattr(output, "finish_reason", "stop")

            if text:
                output_messages.append(
                    OutputMessage(
                        role="assistant",
                        parts=[Text(content=text, type="text")],
                        finish_reason=finish_reason or "stop",
                    )
                )
    except (KeyError, AttributeError):
        # If any attribute access fails, return empty list
        return output_messages

    return output_messages


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


def _create_invocation_from_generation(
    kwargs: dict, model: Optional[str] = None
) -> LLMInvocation:
    """Create LLMInvocation from Generation.call kwargs.

    Args:
        kwargs: Generation.call kwargs
        model: Model name (if not in kwargs)

    Returns:
        LLMInvocation object
    """
    request_model = kwargs.get("model") or model
    if not request_model:
        raise ValueError("Model name is required")

    invocation = LLMInvocation(request_model=request_model)
    invocation.provider = "dashscope"
    invocation.input_messages = _extract_input_messages(kwargs)
    
    # Extract tool definitions and convert to FunctionToolDefinition objects
    invocation.tool_definitions = _extract_tool_definitions(kwargs)

    # Extract parameters from kwargs or kwargs["parameters"] dict
    # Parameters can be passed directly as kwargs (e.g., temperature=0.7) or
    # explicitly in a parameters dict. Check kwargs first, then parameters dict.
    parameters = kwargs.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    
    # Temperature
    temperature = _get_parameter(kwargs, "temperature", parameters)
    if temperature is not None:
        invocation.attributes["gen_ai.request.temperature"] = temperature

    # Top-p
    top_p = _get_parameter(kwargs, "top_p", parameters)
    if top_p is not None:
        invocation.attributes["gen_ai.request.top_p"] = top_p

    # Top-k
    top_k = _get_parameter(kwargs, "top_k", parameters)
    if top_k is not None:
        invocation.attributes["gen_ai.request.top_k"] = top_k

    # Max tokens
    max_tokens = _get_parameter(kwargs, "max_tokens", parameters)
    if max_tokens is not None:
        invocation.attributes["gen_ai.request.max_tokens"] = max_tokens

    # Repetition penalty
    repetition_penalty = _get_parameter(kwargs, "repetition_penalty", parameters)
    if repetition_penalty is not None:
        invocation.attributes["gen_ai.request.repetition_penalty"] = repetition_penalty
    else:
        # Fallback to frequency_penalty and presence_penalty if repetition_penalty not present
        frequency_penalty = _get_parameter(kwargs, "frequency_penalty", parameters)
        if frequency_penalty is not None:
            invocation.attributes["gen_ai.request.frequency_penalty"] = frequency_penalty

        presence_penalty = _get_parameter(kwargs, "presence_penalty", parameters)
        if presence_penalty is not None:
            invocation.attributes["gen_ai.request.presence_penalty"] = presence_penalty

    # Stop sequences
    stop_sequences = _get_parameter(kwargs, "stop", parameters)
    if stop_sequences is not None:
        if isinstance(stop_sequences, list):
            invocation.attributes["gen_ai.request.stop_sequences"] = stop_sequences
        elif isinstance(stop_sequences, str):
            invocation.attributes["gen_ai.request.stop_sequences"] = [stop_sequences]

    # Seed
    seed = _get_parameter(kwargs, "seed", parameters)
    if seed is not None:
        invocation.attributes["gen_ai.request.seed"] = seed

    return invocation


def _update_invocation_from_response(
    invocation: LLMInvocation, response: Any
) -> None:
    """Update LLMInvocation with response data.

    Args:
        invocation: LLMInvocation to update
        response: DashScope response object
    """
    if not response:
        return

    try:
        # Extract output messages
        invocation.output_messages = _extract_output_messages(response)

        # Extract token usage
        input_tokens, output_tokens = _extract_usage(response)
        invocation.input_tokens = input_tokens
        invocation.output_tokens = output_tokens

        # Extract response model name (if available)
        # Use try-except to safely access attributes
        # DashScope response uses __getattr__ which raises KeyError for missing attributes
        try:
            response_model = getattr(response, "model", None)
            if response_model:
                invocation.response_model_name = response_model
        except (KeyError, AttributeError):
            pass

        # Extract request ID (if available)
        try:
            request_id = getattr(response, "request_id", None)
            if request_id:
                invocation.response_id = request_id
        except (KeyError, AttributeError):
            pass
    except (KeyError, AttributeError):
        # If any attribute access fails, silently continue with available data
        pass


def _create_accumulated_response(original_response, accumulated_text):
    """Create a response object with accumulated text for incremental output mode.
    
    Args:
        original_response: The last chunk response object
        accumulated_text: The accumulated text from all chunks
        
    Returns:
        A response object with accumulated text, or original_response if modification fails
    """
    try:
        output = getattr(original_response, "output", None)
        if output and hasattr(output, "text"):
            # Try to set the accumulated text directly
            try:
                output.text = accumulated_text
                return original_response
            except (AttributeError, TypeError):
                # If we can't modify, create a wrapper object
                pass
        
        # Create wrapper objects with accumulated text
        class AccumulatedOutput:
            def __init__(self, original_output, accumulated_text):
                self.text = accumulated_text
                self.finish_reason = getattr(original_output, "finish_reason", "stop")
                self.content = accumulated_text
        
        class AccumulatedResponse:
            def __init__(self, original_response, accumulated_output):
                self.output = accumulated_output
                # Copy other attributes from original response
                for attr in ["usage", "request_id", "model"]:
                    try:
                        value = getattr(original_response, attr, None)
                        if value is not None:
                            setattr(self, attr, value)
                    except (KeyError, AttributeError):
                        pass
        
        accumulated_output = AccumulatedOutput(output, accumulated_text)
        return AccumulatedResponse(original_response, accumulated_output)
    except (KeyError, AttributeError):
        # If modification fails, return original response
        return original_response

