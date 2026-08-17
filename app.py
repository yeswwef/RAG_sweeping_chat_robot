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

st.set_page_config(page_title="AI扫地机器人", page_icon="🤖")
st.title("🤖 AI扫地机器人知识问答系统")
st.divider()

# ==================== 初始化 ====================
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
if "message" not in st.session_state:
    st.session_state["message"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = None  # 懒创建：第一次提问才写入数据库
    st.session_state["message"] = []

# ==================== 侧边栏：会话管理 ====================
with st.sidebar:
    st.subheader("💬 会话记录")

    # 新建对话
    if st.button("➕ 新建对话", use_container_width=True):
        # 懒创建：不立即写库，等用户第一次提问才创建会话（避免空会话残留）
        st.session_state["session_id"] = None
        st.session_state["message"] = []
        st.rerun()

    st.divider()  # 画一条水平分割线
    st.caption("历史会话")  # 显示一行灰色小字"历史会话"

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
                    # 同步清理 LangGraph 记忆线程，避免 UI 删除后记忆残留
                    try:
                        st.session_state["agent"].delete_thread(conv["id"])
                    except Exception:
                        pass
                    # 删的是当前会话：session_id 和 message 必须一起重置（新 uuid + 空列表），
                    # 否则会继续往已删除的 thread 里写
                    if conv["id"] == current_sid:
                        remaining = list_conversations()
                        if remaining:
                            st.session_state["session_id"] = remaining[0]["id"]
                            st.session_state["message"] = list_messages(remaining[0]["id"])
                        else:
                            st.session_state["session_id"] = None  # 没有会话了，等下次提问再建
                            st.session_state["message"] = []
                    st.rerun()

    st.divider()
    st.caption(f"当前会话：{str(current_sid)[:8]}..." if current_sid else "当前会话：未开始")

# ==================== 聊天区 ====================
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input("请输入内容")

if prompt:
    sid = st.session_state["session_id"]

    # 懒创建：第一次提问才真正创建会话记录（空会话不入库）
    if sid is None:
        sid = create_conversation()
        st.session_state["session_id"] = sid

    # 用户消息：先写数据库（保证持久化），再展示
    add_message(sid, "user", prompt)
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 新会话自动命名：第一句用户问题做标题
    if count_conversations() > 0 and get_conversation(sid):
        msgs = list_messages(sid)
        #只有第一次提问改名后面提问均不改名
        if len([m for m in msgs if m["role"] == "user"]) == 1:
            update_conversation_time(sid, title=make_title(prompt))

    res_messages = []
    with st.spinner("智能客服思考中..."):
        # 小"管道"函数：遍历底层生成器吐出的每个片段
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
