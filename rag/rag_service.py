from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rag.query_rewriter import QueryRewriter
from rag.vector_store import VectorStoreService
from rag.hybrid_retriever import HybridRetriever
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
from model.factory import embedding_model


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService(embedding_model)
        self.hybrid_retriever =  HybridRetriever(self.vector_store)
        self.query_rewriter=QueryRewriter()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self.get_chain()

    def get_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        """混合检索（稠密向量 + BM25 稀疏 + RRF 融合），返回按融合分降序的文档列表。"""
        queries=self.query_rewriter.rewrite(query)
        results = self.hybrid_retriever.retrieve(queries)
        print("query_rewriter",queries)
        return [doc for doc, _ in results]

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        context = ''
        counter = 0
        for doc in context_docs:
            counter += 1
            context += '[' + str(counter) + ']: ' + doc.page_content + ' | ' + str(doc.metadata) + chr(10)
        return self.chain.invoke({'input': query, 'context': context})


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.rag_summarize('小户型适合哪些扫地机器人'))
