from typing import Any

import logging
logger = logging.getLogger(__name__)

def _get_chatmodel_output_messages(
    chat_response: Any, 
) -> list[dict[str, Any]]:
    """将ChatResponse转换为OpenTelemetry标准的输出消息格式
    
    Args:
        chat_response: ChatResponse对象或其他响应对象
        telemetry_options: 遥测配置选项
        
    Returns:
        list[dict[str, Any]]: 格式化的输出消息列表
    """
    try:

        from agentscope.model import ChatResponse
        
        if not isinstance(chat_response, ChatResponse):
            # logger.warning(f"Expected ChatResponse, got {type(chat_response)}")
            return chat_response
        
        # 构建parts列表
        parts = []
        finish_reason = "stop"  # 默认完成原因
        
        # 处理content中的每个block
        for block in chat_response.content:
            block_type = block.get("type")
            
            if block_type == "text":
                # 文本块处理
                text_content = block.get("text", "")
                # 应用内容处理策略
                # processed_content = _process_content(text_content, telemetry_options)
                processed_content = text_content
                parts.append({
                    "type": "text",
                    "content": processed_content
                })
                
            elif block_type == "thinking":
                # 思考块处理 - 转换为文本块
                thinking_content = block.get("thinking", "")
                # processed_content = _process_content(thinking_content, telemetry_options)
                processed_content = thinking_content
                parts.append({
                    "type": "text", 
                    "content": f"[Thinking] {processed_content}"
                })
                
            elif block_type == "tool_use":
                # 工具调用块处理
                tool_call_data = {
                    "type": "tool_call",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": block.get("input", {})
                }
                tool_call_data["arguments"] = tool_call_data["arguments"]
                # # 对工具参数应用内容处理
                # if telemetry_options.should_capture_content():
                #     # 序列化参数
                #     args_str = _serialize_to_str(tool_call_data["arguments"])
                #     tool_call_data["arguments"] = _process_content(args_str, telemetry_options)
                # else:
                #     # 不捕获内容时，只显示大小
                #     args_str = _serialize_to_str(tool_call_data["arguments"])
                #     tool_call_data["arguments"] = f"<{len(args_str)}size>"
                
                parts.append(tool_call_data)
                
            else:
                logger.debug(f"Unsupported block type: {block_type}")
        
        # 如果没有任何parts，添加一个空的文本块
        if not parts:
            parts.append({
                "type": "text",
                "content": ""
            })
        
        # 构建最终的输出消息
        output_message = {
            "role": "assistant",
            "parts": parts,
            "finish_reason": finish_reason
        }
        
        return [output_message]
        
    except Exception as e:
        logger.warning(f"Error processing ChatResponse to output messages: {e}")
        # 返回一个基本的错误消息
        return [{
            "role": "assistant",
            "parts": [{
                "type": "text",
                "content": "<error processing response>"
            }],
            "finish_reason": "error"
        }]
