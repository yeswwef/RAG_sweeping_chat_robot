import sys
import os
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
        res_stream = st.session_state["agent"].execute_stream(prompt)
        st.chat_message("assistant").write_stream(capture(res_stream,res_messages))
        st.session_state["message"].append({"role": "assistant", "content": res_messages[-1]})
        st.rerun()