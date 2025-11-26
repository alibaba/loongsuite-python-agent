"""
Google ADK + FastAPI Service
将 Google ADK Agent 封装为 RESTful API 服务
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
import uvicorn
from datetime import datetime


# 定义请求和响应模型
class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    user_id: str = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    token_usage: dict = None


# 创建 FastAPI 应用
app = FastAPI(title="Google ADK API Service")


# 定义工具
def get_weather(city: str) -> str:
    """获取城市天气（模拟）"""
    # 实际应用中这里应该调用真实的天气API
    return f"{city}的天气：晴，温度25°C"


def search_knowledge(query: str) -> str:
    """搜索知识库（模拟）"""
    # 实际应用中这里应该连接真实的知识库
    return f"关于'{query}'的知识：这是模拟的知识库返回结果"


# 创建 Tools
weather_tool = FunctionTool(
    name="get_weather",
    description="获取指定城市的天气信息",
    func=get_weather
)

knowledge_tool = FunctionTool(
    name="search_knowledge",
    description="搜索内部知识库",
    func=search_knowledge
)

# 创建 Agent
assistant_agent = Agent(
    name="customer_service_agent",
    description="智能客服助手，可以查询天气和搜索知识库",
    tools=[weather_tool, knowledge_tool],
    model="gemini-1.5-flash",
    system_instruction="你是一个专业的客服助手，态度友好，回答准确。"
)

# 创建 Runner
runner = Runner(agent=assistant_agent)


# API 端点
@app.get("/")
def root():
    """健康检查"""
    return {
        "service": "Google ADK API Service",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    处理聊天请求
    
    Args:
        request: 包含用户消息和会话信息的请求
        
    Returns:
        ChatResponse: 包含 Agent 响应的结果
    """
    try:
        # 执行 Agent
        response = runner.run(
            request.message,
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        return ChatResponse(
            response=response,
            session_id=request.session_id or "default",
            token_usage={"note": "Token usage info would be here"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """健康检查端点"""
    return {"status": "healthy"}


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
