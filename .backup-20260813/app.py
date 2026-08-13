import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from rag.rag_service import RagSummarizeService
from agent.react_agent import ReactAgent
st.title("ai扫地机器人")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"]=ReactAgent()
if "message" not in st.session_state:
    st.session_state["message"]=[]
if "session_id" not in st.session_state:
    # 每个浏览器会话一个固定 session_id，作为 Agent 的 thread_id，实现多轮对话记忆
    st.session_state["session_id"] = str(uuid.uuid4())

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt=st.chat_input("请输入内容")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user" ,"content":prompt})


    res_messages=[]
    with st.spinner("智能客服思考中..."):
        # 流式输出拦截器函数
        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
        res_stream = st.session_state["agent"].execute_stream(prompt, session_id=st.session_state["session_id"])
        st.chat_message("assistant").write_stream(capture(res_stream,res_messages))
        st.session_state["message"].append({"role": "assistant", "content": res_messages[-1]})
        st.rerun()