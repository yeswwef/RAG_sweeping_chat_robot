import os
from abc import ABC, abstractmethod
from typing import Optional

from utils.env import settings  # 从 .env 读取配置

# 把 .env 里的 Key 注入环境变量（setdefault：如果系统里已设置则不覆盖）
os.environ.setdefault("DASHSCOPE_API_KEY", settings.dashscope_api_key)

from langchain_community import embeddings
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_config


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

#聊天模型
class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_config["chat_model_name"])

#文本嵌入模型
class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])

chat_model=ChatModelFactory().generator()
embedding_model=EmbeddingsFactory().generator()
