# api/server.py
"""扫地机器人 AI Agent HTTP API（FastAPI）。

接口：
- GET    /health                   健康检查（含数据库连通性）
- POST   /chat                     对话：stream=false 返回 JSON，stream=true 返回 SSE 流式
- GET    /conversations            会话列表（与 Streamlit 侧边栏同源）
- DELETE /conversations/{id}       删除会话（同时清理 LangGraph 记忆线程）

设计说明：
- 会话统一写入 utils/db.py 管理的 conversations.db，与 Streamlit 前端共享，
  API 产生的会话在 UI 历史列表里同样可见。
- 同步端点使用普通 def，FastAPI 自动放入线程池执行，避免长时间生成阻塞事件循环。
- 可选鉴权（API_KEY）与每 IP 简单限流；默认仅适合本地/内网，公网部署请配合网关。
"""
import hmac
import json
import os
import threading
import time
from collections import defaultdict

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from agent.react_agent import ReactAgent
from utils.db import (add_message, create_conversation, delete_conversation,
                      get_conversation, get_conn, list_conversations,
                      list_messages, make_title, update_conversation_time)
from utils.logger_handler import logger

APP_VERSION = "1.1.0"

app = FastAPI(
    title="扫地机器人 AI Agent API",
    version=APP_VERSION,
)

# ===== CORS：默认仅放行本机 Streamlit，可通过 API_ALLOW_ORIGINS 配置（逗号分隔，* 表示全部） =====
_DEFAULT_ALLOW_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]
_raw_origins = os.getenv("API_ALLOW_ORIGINS", "").strip()
ALLOW_ORIGINS = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins else _DEFAULT_ALLOW_ORIGINS
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 可选鉴权：设置 API_KEY 后，业务接口必须携带 X-API-Key =====
API_KEY = os.getenv("API_KEY", "").strip()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """API Key 校验（未配置 API_KEY 时跳过，适合本地开发）。"""
    if not API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# ===== 简单限流：内存滑动窗口，按客户端 IP，默认 30 次/分钟（0 表示关闭） =====
RATE_LIMIT_PER_MINUTE = int(os.getenv("API_RATE_LIMIT_PER_MINUTE", "30"))


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0):
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = [t for t in self._hits[key] if now - t < self._window]
            if len(hits) >= self._limit:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True


_rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)


def rate_limit(request: Request) -> None:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    client = request.client.host if request.client else "unknown"
    if not _rate_limiter.allow(client):
        raise HTTPException(status_code=429, detail="too many requests, please slow down")


# ===== Agent 单例：进程内复用；多 worker 部署时每个 worker 各持一个实例 =====
agent = ReactAgent()


# ===== 请求/响应模型 =====
class ChatRequest(BaseModel):
    """客户端发来的请求体"""
    query: str = Field(..., min_length=1, max_length=2000)  # 用户问题
    session_id: str | None = None  # 会话 ID（不传则自动创建）
    stream: bool = False           # true 时以 SSE 流式返回


class ChatResponse(BaseModel):
    """非流式返回给客户端的数据"""
    session_id: str
    answer: str
    latency_ms: int = 0  # 响应耗时（毫秒）


def _ensure_conversation(session_id: str | None) -> str:
    """校验/创建会话记录，保证 API 会话与 UI 历史列表一致。"""
    if session_id and get_conversation(session_id):
        return session_id
    return create_conversation(session_id=session_id)


def _finalize_title(session_id: str, query: str) -> None:
    """会话第一条用户消息时自动命名（与 app.py 行为一致）。"""
    msgs = list_messages(session_id)
    if len([m for m in msgs if m["role"] == "user"]) == 1:
        update_conversation_time(session_id, title=make_title(query))


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ===== 接口 =====

@app.get("/health")
async def health():
    """健康检查：含数据库连通性验证。"""
    try:
        get_conn().execute("SELECT 1").fetchone()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"database unavailable: {e}")
    return {"status": "ok", "version": APP_VERSION}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key), Depends(rate_limit)])
def chat(req: ChatRequest):
    """对话接口。

    stream=false（默认，返回 JSON）：
      curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
        -d '{"query": "扫地机器人怎么保养"}'
    stream=true（SSE 流式）：
      curl -N -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
        -d '{"query": "扫地机器人怎么保养", "stream": true}'
    """
    start = time.time()
    session_id = _ensure_conversation(req.session_id)
    add_message(session_id, "user", req.query)
    _finalize_title(session_id, req.query)

    if req.stream:
        return _chat_stream(session_id, req.query, start)

    try:
        answer = "".join(agent.execute_stream(req.query, session_id=session_id))
    except Exception as e:
        logger.error(f"[api] agent execute failed: {e}")
        raise HTTPException(status_code=500, detail=f"agent error: {e}")

    add_message(session_id, "assistant", answer)
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        latency_ms=int((time.time() - start) * 1000),
    )


def _chat_stream(session_id: str, query: str, start: float) -> StreamingResponse:
    """SSE 流式：把 agent 生成器产生的片段逐块推给客户端。"""
    async def gen():
        collected: list[str] = []
        error: str | None = None
        try:
            async for chunk in iterate_in_threadpool(
                agent.execute_stream(query, session_id=session_id)
            ):
                collected.append(chunk)
                yield _sse({"session_id": session_id, "chunk": chunk})
            yield _sse({
                "session_id": session_id,
                "done": True,
                "latency_ms": int((time.time() - start) * 1000),
            })
        except Exception as e:
            error = str(e)
            logger.error(f"[api] stream error for {session_id}: {e}")
            yield _sse({"error": error})
        finally:
            # 流结束（无论成败）都把已生成内容落库，避免会话记录丢失
            content = "".join(collected)
            if error and content:
                content += f"\n[输出中断：{error}]"
            elif error:
                content = f"[回答出错：{error}]"
            if content:
                add_message(session_id, "assistant", content)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/conversations", dependencies=[Depends(require_api_key)])
def list_conversations_api():
    """会话列表（与 Streamlit 侧边栏同一数据源）。"""
    return {"conversations": list_conversations()}


@app.delete("/conversations/{session_id}", dependencies=[Depends(require_api_key)])
def delete_conversation_api(session_id: str):
    """删除会话：同时清理数据库记录与 LangGraph 记忆线程。"""
    delete_conversation(session_id)
    try:
        agent.delete_thread(session_id)
    except Exception as e:
        # 记忆线程清理失败不阻塞删除，仅记录
        logger.warning(f"[api] delete_thread failed for {session_id}: {e}")
    return {"deleted": True, "session_id": session_id}
