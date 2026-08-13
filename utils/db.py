# -*- coding: utf-8 -*-
"""
utils/db.py - SQLite 持久化层
================================
负责会话（conversations）与消息（messages）的增删查改。

设计要点：
- 每个会话一个 uuid（conversations.id），对应一整场对话
- messages.conversation_id 外键关联，删除会话时级联删除其消息
- 前端展示查 messages 表；Agent 记忆（LangGraph SqliteSaver）也用同一 uuid 作 thread_id
"""
import os
import sqlite3
import uuid
from datetime import datetime

# 数据库文件路径：项目根目录下 conversations.db（与 Dockerfile WORKDIR /app 一致）
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conversations.db")

_conn = None


def get_conn() -> sqlite3.Connection:
    """获取全局数据库连接（懒加载，单例）。"""
    global _conn

    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA foreign_keys = ON")
        init_db(_conn)
    return _conn


def init_db(conn: sqlite3.Connection) -> None:
    """建表（幂等）。"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id          TEXT PRIMARY KEY,                 -- uuid4 字符串
            title       TEXT NOT NULL DEFAULT '新对话',    -- 侧边栏显示名
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role            TEXT NOT NULL,                -- 'user' / 'assistant'
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)")
    conn.commit()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==================== 会话 ====================

def create_conversation(title: str = "新对话") -> str:
    """新建会话，返回新 uuid。"""
    conn = get_conn()
    sid = str(uuid.uuid4())
    now = _now()
    conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (sid, title, now, now),
    )
    conn.commit()
    return sid


def list_conversations() -> list[dict]:
    """获取全部会话（按更新时间倒序）。"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(session_id: str) -> dict | None:
    """按 uuid 查询单个会话。"""
    conn = get_conn()
    row = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def update_conversation_time(session_id: str, title: str | None = None) -> None:
    """刷新会话的 updated_at；可选更新标题。"""
    conn = get_conn()
    now = _now()
    if title:
        conn.execute(
            "UPDATE conversations SET updated_at = ?, title = ? WHERE id = ?",
            (now, title, session_id),
        )
    else:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
    conn.commit()


def delete_conversation(session_id: str) -> None:
    """删除会话（messages 由外键级联删除）。"""
    conn = get_conn()
    conn.execute("DELETE FROM conversations WHERE id = ?", (session_id,))
    conn.commit()


def count_conversations() -> int:
    conn = get_conn()
    return conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]


# ==================== 消息 ====================

def add_message(session_id: str, role: str, content: str) -> None:
    """向某会话追加一条消息，并刷新会话时间。"""
    conn = get_conn()
    now = _now()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now),
    )
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.commit()


def list_messages(session_id: str) -> list[dict]:
    """按顺序取出某会话的全部消息。"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def make_title(first_question: str, max_len: int = 20) -> str:
    """根据第一句用户问题自动生成会话标题（截取前 max_len 字）。"""
    title = first_question.strip().replace("\n", " ")
    return title[:max_len] if title else "新对话"
