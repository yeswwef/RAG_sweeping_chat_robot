# api/server.py
import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.react_agent import ReactAgent

# ===== 1. 创建 FastAPI 应用 =====
app = FastAPI(
    title="扫地机器人 AI Agent API",
    version="1.0.0",
)

# ===== 2. 添加 CORS 中间件 =====
# 允许前端网页（不同域名）调用我们的 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许所有来源（开发阶段）
    allow_methods=["*"],       # 允许所有 HTTP 方法
    allow_headers=["*"],       # 允许所有请求头
)

# ===== 3. 创建 Agent 单例 =====
# 全局只创建一次 Agent，每次请求复用
agent = ReactAgent()


# ===== 4. 定义请求/响应格式 =====
class ChatRequest(BaseModel):
    """客户端发来的请求体"""
    query: str = Field(..., min_length=1, max_length=2000)  # 用户问题
    session_id: str | None = None  # 会话 ID（不传则自动生成）
    stream: bool = False  # 是否流式输出


class ChatResponse(BaseModel):
    """返回给客户端的数据"""
    session_id: str
    answer: str
    latency_ms: int = 0  # 响应耗时（毫秒）


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"


# ===== 5. 定义接口 =====

@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查接口"""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    对话接口

    用法：
    curl -X POST http://localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"query": "扫地机器人怎么保养"}'
    """
    start = time.time()  # 记开始时间

    # 如果没传 session_id，自动生成一个
    session_id = req.session_id or str(uuid.uuid4())

    try:
        # 调用 Agent，收集所有输出
        result = []
        for chunk in agent.execute_stream(req.query):
            result.append(chunk)
        answer = "".join(result)
    except Exception as e:
        # 出错了，返回 500 错误
        raise HTTPException(status_code=500, detail=str(e))

    # 计算耗时
    latency = int((time.time() - start) * 1000)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        latency_ms=latency,
    )
