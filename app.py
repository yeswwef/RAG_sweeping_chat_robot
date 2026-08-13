import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from rag.rag_service import RagSummarizeService
from agent.react_agent import ReactAgent
from utils.db import (create_conversation, list_conversations, get_conversation,
                      delete_conversation, add_message, list_messages, make_title,
                      update_conversation_time, count_conversations)

st.set_page_config(page_title="AI扫地机器人", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 AI扫地机器人知识问答系统")
st.divider()

# ==================== 会话状态初始化 ====================
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
if "message" not in st.session_state:
    st.session_state["message"] = []

# 启动时：若没有会话记录，先建一个；若已有 session_id 则校验其是否仍存在（可能被删过）
if "session_id" not in st.session_state:

    st.session_state["session_id"] = create_conversation()

    # 从数据库加载该会话的历史消息
    st.session_state["message"] = list_messages(st.session_state["session_id"])

# ==================== 侧边栏：会话管理 ====================
with st.sidebar:
    st.subheader("💬 会话记录")

    # 新建对话
    if st.button("➕ 新建对话", use_container_width=True):
        # 保存当前会话（消息已在每次对话后实时写入数据库，此处只需切换）
        st.session_state["session_id"] = create_conversation()
        st.session_state["message"] = []
        st.rerun()

    st.divider()
    st.caption("历史会话")

    # 会话列表（按更新时间倒序）
    conversations = list_conversations()
    current_sid = st.session_state.get("session_id")

    if not conversations:
        st.info("暂无历史会话")
    else:
        for conv in conversations:
            col1, col2 = st.columns([4, 1])
            with col1:
                # 当前会话高亮标记
                label = conv["title"]
                if conv["id"] == current_sid:
                    label = f"👉 {label}"
                if st.button(label, use_container_width=True,
                             key=f"load_{conv['id']}"):
                    # 切换到该会话：切换 uuid + 加载其历史消息
                    st.session_state["session_id"] = conv["id"]
                    st.session_state["message"] = list_messages(conv["id"])
                    st.rerun()
            with col2:
                if st.button("🗑", key=f"del_{conv['id']}", help="删除该会话"):
                    delete_conversation(conv["id"])
                    # 删的是当前会话：session_id 和 message 必须一起重置（新 uuid + 空列表），
                    # 否则会继续往已删除的 thread 里写
                    if conv["id"] == current_sid:
                        remaining = list_conversations()
                        if remaining:
                            st.session_state["session_id"] = remaining[0]["id"]
                            st.session_state["message"] = list_messages(remaining[0]["id"])
                        else:
                            st.session_state["session_id"] = create_conversation()
                            st.session_state["message"] = []
                    st.rerun()

    st.divider()
    st.caption(f"当前会话：{str(current_sid)[:8]}...")

# ==================== 聊天区 ====================
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("请输入内容")

if prompt:
    sid = st.session_state["session_id"]

    # 用户消息：先写数据库（保证持久化），再展示
    add_message(sid, "user", prompt)
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 新会话自动命名：第一句用户问题做标题
    if count_conversations() > 0 and get_conversation(sid):
        msgs = list_messages(sid)
        if len([m for m in msgs if m["role"] == "user"]) == 1:
            update_conversation_time(sid, title=make_title(prompt))

    res_messages = []
    with st.spinner("智能客服思考中..."):
        # 流式输出拦截器函数
        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        res_stream = st.session_state["agent"].execute_stream(prompt, session_id=sid)
        st.chat_message("assistant").write_stream(capture(res_stream, res_messages))
        answer = "".join(res_messages)
        # 助手回复：写数据库 + 更新内存
        add_message(sid, "assistant", answer)
        st.session_state["message"].append({"role": "assistant", "content": answer})
        st.rerun()
