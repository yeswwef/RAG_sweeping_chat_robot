from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
from model.factory import embedding_model


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService(embedding_model)
        self.retriver = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self.get_chain()

    def get_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriver.invoke(query)

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
