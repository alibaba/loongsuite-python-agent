"""
ADK Attribute Extractors following OpenTelemetry GenAI Semantic Conventions.

This module extracts trace attributes from Google ADK objects according 
to OpenTelemetry GenAI semantic conventions (latest version).
"""

import json
import logging
from typing import Any, Dict, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from ._utils import (
    safe_json_dumps, safe_json_dumps_large, extract_content_safely,
    safe_json_dumps_for_input_output, extract_content_safely_for_input_output,
    should_capture_content, process_content
)

_logger = logging.getLogger(__name__)


class AdkAttributeExtractors:
    """
    Attribute extractors for Google ADK following OpenTelemetry GenAI semantic conventions.
    
    Extracts trace attributes from ADK objects according to:
    - gen_ai.* attributes for GenAI-specific information
    - Standard OpenTelemetry attributes for general information
    """

    def extract_common_attributes(
        self, 
        operation_name: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract common GenAI attributes required for all spans.
        
        Args:
            operation_name: Operation name (chat, invoke_agent, execute_tool, etc.)
            conversation_id: Conversation/session ID (optional)
            user_id: User ID (optional)
            
        Returns:
            Dictionary of common attributes
        """
        attrs = {
            "gen_ai.operation.name": operation_name,
            "gen_ai.provider.name": "google_adk"  # ✅ 使用 provider.name 而非 system
        }
        
        # ✅ conversation.id 而非 session.id
        if conversation_id and isinstance(conversation_id, str):
            attrs["gen_ai.conversation.id"] = conversation_id
        
        # ✅ 使用标准 enduser.id 而非 gen_ai.user.id    
        if user_id and isinstance(user_id, str):
            attrs["enduser.id"] = user_id
            
        return attrs

    def extract_runner_attributes(
        self, 
        invocation_context: InvocationContext
    ) -> Dict[str, Any]:
        """
        Extract attributes for Runner spans (top-level invoke_agent span).
        
        Args:
            invocation_context: ADK invocation context
            
        Returns:
            Dictionary of runner attributes
        """
        try:
            _logger.debug("Extracting runner attributes")
            
            # Extract conversation_id and user_id from invocation_context
            conversation_id = None
            user_id = None
            
            try:
                conversation_id = invocation_context.session.id
            except AttributeError:
                _logger.debug("Failed to extract conversation_id from invocation_context")
                
            try:
                user_id = getattr(invocation_context, 'user_id', None)
                if not user_id and hasattr(invocation_context, 'session'):
                    user_id = getattr(invocation_context.session, 'user_id', None)
            except AttributeError:
                _logger.debug("Failed to extract user_id from invocation_context")

            if conversation_id is None:
                _logger.debug("conversation_id not found on invocation_context")
            if user_id is None:
                _logger.debug("user_id not found on invocation_context")
            
            # ✅ 使用 invoke_agent 操作名称
            attrs = self.extract_common_attributes(
                operation_name="invoke_agent",
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # Add ADK-specific attributes (非标准，作为自定义扩展)
            if hasattr(invocation_context, 'app_name'):
                attrs["google_adk.runner.app_name"] = invocation_context.app_name
                
            if hasattr(invocation_context, 'invocation_id'):
                attrs["google_adk.runner.invocation_id"] = invocation_context.invocation_id
                
            # Agent spans use input.value/output.value
            attrs["input.mime_type"] = "application/json"
            attrs["output.mime_type"] = "application/json"
                
            return attrs
            
        except Exception as e:
            _logger.exception(f"Error extracting runner attributes: {e}")
            return self.extract_common_attributes("invoke_agent")

    def extract_agent_attributes(
        self, 
        agent: BaseAgent,
        callback_context: CallbackContext
    ) -> Dict[str, Any]:
        """
        Extract attributes for Agent spans.
        
        Args:
            agent: ADK agent instance
            callback_context: ADK callback context
            
        Returns:
            Dictionary of agent attributes
        """
        try:
            _logger.debug("Extracting agent attributes")
            
            # Extract conversation_id and user_id from callback_context
            conversation_id = None
            user_id = None
            
            try:
                conversation_id = callback_context._invocation_context.session.id
            except AttributeError:
                _logger.debug("Failed to extract conversation_id from callback_context")
                
            try:
                user_id = getattr(callback_context, 'user_id', None)
                if not user_id:
                    user_id = getattr(callback_context._invocation_context, 'user_id', None)
            except AttributeError:
                _logger.debug("Failed to extract user_id from callback_context")

            if conversation_id is None:
                _logger.debug("conversation_id not found on callback_context")
            if user_id is None:
                _logger.debug("user_id not found on callback_context")
            
            # ✅ 使用 invoke_agent 操作名称（无论是 agent 还是 chain）
            attrs = self.extract_common_attributes(
                operation_name="invoke_agent",
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # ✅ 使用 gen_ai.agent.* 属性（带前缀）
            if hasattr(agent, 'name') and agent.name:
                attrs["gen_ai.agent.name"] = agent.name
                
            # ✅ 尝试获取 agent.id（如果可用）
            if hasattr(agent, 'id') and agent.id:
                attrs["gen_ai.agent.id"] = agent.id
                
            if hasattr(agent, 'description') and agent.description:
                attrs["gen_ai.agent.description"] = agent.description
                
            # Add input/output placeholder
            attrs["input.mime_type"] = "application/json"
            attrs["output.mime_type"] = "application/json"
            
            return attrs
            
        except Exception as e:
            _logger.exception(f"Error extracting agent attributes: {e}")
            return self.extract_common_attributes("invoke_agent")

    def extract_llm_request_attributes(
        self, 
        llm_request: LlmRequest,
        callback_context: CallbackContext
    ) -> Dict[str, Any]:
        """
        Extract attributes for LLM request spans.
        
        Args:
            llm_request: ADK LLM request
            callback_context: ADK callback context
            
        Returns:
            Dictionary of LLM request attributes
        """
        try:
            # Extract conversation_id and user_id
            conversation_id = None
            user_id = None
            
            try:
                conversation_id = callback_context._invocation_context.session.id
            except AttributeError:
                _logger.debug("Failed to extract conversation_id from callback_context")
                
            try:
                user_id = getattr(callback_context, 'user_id', None)
                if not user_id:
                    user_id = getattr(callback_context._invocation_context, 'user_id', None)
            except AttributeError:
                _logger.debug("Failed to extract user_id from callback_context")
            
            # ✅ 使用 chat 操作名称
            attrs = self.extract_common_attributes(
                operation_name="chat",
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # Add LLM request attributes according to GenAI conventions
            if hasattr(llm_request, 'model') and llm_request.model:
                # ✅ 只使用 gen_ai.request.model（移除冗余的 model_name）
                attrs["gen_ai.request.model"] = llm_request.model
                # ✅ 使用 _extract_provider_name 而非 _extract_system_from_model
                attrs["gen_ai.provider.name"] = self._extract_provider_name(llm_request.model)
                
            # Extract request parameters
            if hasattr(llm_request, 'config') and llm_request.config:
                config = llm_request.config
                
                if hasattr(config, 'max_tokens') and config.max_tokens:
                    attrs["gen_ai.request.max_tokens"] = config.max_tokens
                    
                if hasattr(config, 'temperature') and config.temperature is not None:
                    if isinstance(config.temperature, (int, float)):
                        attrs["gen_ai.request.temperature"] = config.temperature
                    
                if hasattr(config, 'top_p') and config.top_p is not None:
                    if isinstance(config.top_p, (int, float)):
                        attrs["gen_ai.request.top_p"] = config.top_p
                    
                if hasattr(config, 'top_k') and config.top_k is not None:
                    if isinstance(config.top_k, (int, float)):
                        attrs["gen_ai.request.top_k"] = config.top_k
            
            # Extract input messages (with content capture control)
            if should_capture_content() and hasattr(llm_request, 'contents') and llm_request.contents:
                try:
                    input_messages = []
                    for content in llm_request.contents:
                        if hasattr(content, 'role') and hasattr(content, 'parts'):
                            # Convert to GenAI message format
                            message = {
                                "role": content.role,
                                "parts": []
                            }
                            for part in content.parts:
                                if hasattr(part, 'text'):
                                    message["parts"].append({
                                        "type": "text",
                                        "content": process_content(part.text)
                                    })
                            input_messages.append(message)
                    
                    if input_messages:
                        attrs["gen_ai.input.messages"] = safe_json_dumps_large(input_messages)
                        
                except Exception as e:
                    _logger.debug(f"Failed to extract input messages: {e}")
                
            attrs["input.mime_type"] = "application/json"
            # ❌ 移除 gen_ai.request.is_stream (非标准属性)
            
            return attrs
            
        except Exception as e:
            _logger.exception(f"Error extracting LLM request attributes: {e}")
            return self.extract_common_attributes("chat")

    def extract_llm_response_attributes(
        self, 
        llm_response: LlmResponse
    ) -> Dict[str, Any]:
        """
        Extract attributes for LLM response.
        
        Args:
            llm_response: ADK LLM response
            
        Returns:
            Dictionary of LLM response attributes
        """
        try:
            attrs = {}
            
            # Add response model
            if hasattr(llm_response, 'model') and llm_response.model:
                attrs["gen_ai.response.model"] = llm_response.model
                
            # ✅ finish_reasons (复数数组)
            if hasattr(llm_response, 'finish_reason'):
                finish_reason = llm_response.finish_reason or 'stop'
                attrs["gen_ai.response.finish_reasons"] = [finish_reason]  # 必须是数组
                
            # Add token usage
            if hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata:
                usage = llm_response.usage_metadata
                
                if hasattr(usage, 'prompt_token_count') and usage.prompt_token_count:
                    attrs["gen_ai.usage.input_tokens"] = usage.prompt_token_count
                    
                if hasattr(usage, 'candidates_token_count') and usage.candidates_token_count:
                    attrs["gen_ai.usage.output_tokens"] = usage.candidates_token_count
                # ❌ 移除 gen_ai.usage.total_tokens (非标准，可自行计算)
                    
            # Extract output messages (with content capture control)
            if should_capture_content() and hasattr(llm_response, 'content') and llm_response.content:
                try:
                    output_messages = []
                    # Check if response has text content
                    if hasattr(llm_response, 'text') and llm_response.text is not None:
                        extracted_text = extract_content_safely_for_input_output(llm_response.text)
                        message = {
                            "role": "assistant",
                            "parts": [{
                                "type": "text",
                                "content": process_content(extracted_text)
                            }],
                            "finish_reason": getattr(llm_response, 'finish_reason', None) or 'stop'
                        }
                        output_messages.append(message)
                    elif hasattr(llm_response, 'content') and llm_response.content is not None:
                        extracted_text = extract_content_safely_for_input_output(llm_response.content)
                        message = {
                            "role": "assistant",
                            "parts": [{
                                "type": "text",
                                "content": process_content(extracted_text)
                            }],
                            "finish_reason": getattr(llm_response, 'finish_reason', None) or 'stop'
                        }
                        output_messages.append(message)
                    
                    if output_messages:
                        attrs["gen_ai.output.messages"] = safe_json_dumps_large(output_messages)
                        
                except Exception as e:
                    _logger.debug(f"Failed to extract output messages: {e}")
                
            attrs["output.mime_type"] = "application/json"
            
            return attrs
            
        except Exception as e:
            _logger.exception(f"Error extracting LLM response attributes: {e}")
            return {}

    def extract_tool_attributes(
        self, 
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext
    ) -> Dict[str, Any]:
        """
        Extract attributes for Tool spans.
        
        Args:
            tool: ADK tool instance
            tool_args: Tool arguments
            tool_context: Tool context
            
        Returns:
            Dictionary of tool attributes
        """
        try:
            # 尝试从tool_context提取conversation_id
            conversation_id = None
            user_id = None
            
            if hasattr(tool_context, 'session_id'):
                conversation_id = tool_context.session_id
            elif hasattr(tool_context, 'context') and hasattr(tool_context.context, 'session_id'):
                conversation_id = tool_context.context.session_id
                
            # ✅ 使用 execute_tool 操作名称
            attrs = self.extract_common_attributes(
                operation_name="execute_tool",
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # ✅ Tool 属性使用 gen_ai.tool.* 前缀
            if hasattr(tool, 'name') and tool.name:
                attrs["gen_ai.tool.name"] = tool.name
                
            if hasattr(tool, 'description') and tool.description:
                attrs["gen_ai.tool.description"] = tool.description
            
            # ✅ 默认 tool type 为 function
            attrs["gen_ai.tool.type"] = "function"
            
            # ✅ 尝试获取 tool.call.id（如果可用）
            if hasattr(tool_context, 'call_id') and tool_context.call_id:
                attrs["gen_ai.tool.call.id"] = tool_context.call_id
                
            # ✅ tool.call.arguments 而非 tool.parameters (Opt-In)
            if should_capture_content() and tool_args:
                attrs["gen_ai.tool.call.arguments"] = safe_json_dumps(tool_args)
                attrs["input.value"] = safe_json_dumps_for_input_output(tool_args)
                attrs["input.mime_type"] = "application/json"
                
            return attrs
            
        except Exception as e:
            _logger.exception(f"Error extracting tool attributes: {e}")
            return self.extract_common_attributes("execute_tool")

    def _extract_provider_name(self, model_name: str) -> str:
        """
        Extract provider name from model name according to OTel GenAI conventions.
        
        Args:
            model_name: Model name string
            
        Returns:
            Provider name following OTel GenAI standard values
        """
        if not model_name:
            return "google_adk"
            
        model_lower = model_name.lower()
        
        # Google models - use standard values from OTel spec
        if "gemini" in model_lower:
            return "gcp.gemini"  # AI Studio API
        elif "vertex" in model_lower:
            return "gcp.vertex_ai"  # Vertex AI
        # OpenAI models
        elif "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        # Anthropic models
        elif "claude" in model_lower:
            return "anthropic"
        # Other providers
        elif "llama" in model_lower or "meta" in model_lower:
            return "meta"
        elif "mistral" in model_lower:
            return "mistral_ai"
        elif "cohere" in model_lower:
            return "cohere"
        else:
            # Default to google_adk for unknown models
            return "google_adk"


