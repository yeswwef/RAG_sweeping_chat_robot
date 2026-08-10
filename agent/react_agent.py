from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from agent.tools import middleware
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                    get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.checkpointer=MemorySaver()
        self.agent=create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                                    get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            checkpointer=self.checkpointer
        )

    def execute_stream(self, query: str, session_id: str = "default"):  # ← 加参数
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }
        # ← 关键：thread_id 决定会话归属，同一 session_id 自动带出历史
        config = {"configurable": {"thread_id": session_id}}
        for chunk in self.agent.stream(input_dict, stream_mode="values", config=config, context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chunk, end="", flush=True)