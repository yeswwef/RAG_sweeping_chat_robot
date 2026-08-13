# -*- coding: utf-8 -*-
"""多轮记忆验证：同一 session_id 应记住上文；不同 session_id 应互相隔离"""
import os
import sys

os.environ["NO_PROXY"] = "dashscope.aliyuncs.com"  # 开着代理时保证能连上 API

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.react_agent import ReactAgent


def ask(agent, session_id, question):
    print(f"\n>>> [{session_id}] 用户: {question}")
    answer = ""
    for chunk in agent.execute_stream(question, session_id=session_id):
        answer += chunk
    print(f"<<< [{session_id}] 助手: {answer.strip()}")
    return answer


if __name__ == "__main__":
    agent = ReactAgent()

    # ===== 测试1：同一会话，第二轮应记得第一轮的信息 =====
    print("=" * 60)
    print("测试1：多轮记忆（同一 session_id='user_001'）")
    ask(agent, "user_001", "记住，我的城市是北京，而且我家是小户型")
    ask(agent, "user_001", "根据我刚才告诉你的信息，我的城市是哪里？我家是什么户型？")

    # ===== 测试2：不同会话，不应串号 =====
    print("\n" + "=" * 60)
    print("测试2：会话隔离（新 session_id='user_002'，不应记得 user_001 的信息）")
    ask(agent, "user_002", "我的城市是哪里？你知道我家户型吗？")


