import os
from typing import Optional
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import self_hosted
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import txt_loader,pdf_loader,listdir_with_allowed_type,get_file_md5
from utils.path_tool import get_abs_path
from utils.config_handler import chroma_config
from model.factory import embedding_model
from utils.logger_handler import logger


class VectorStoreService:

    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=self.embedding,
            persist_directory=chroma_config["persist_directory"]
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len
        )

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        return self.vector_store.as_retriever(
            search_kwargs={'k': chroma_config["k"]}
        )

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """稠密向量检索，返回 k 条最相似文档（供混合检索的稠密路使用）。"""
        return self.vector_store.similarity_search(query, k=k)

    def get_all_documents(self) -> list[Document]:
        """拉取向量库中全部分块（供 BM25 构建稀疏索引）。"""
        data = self.vector_store.get()
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or [None] * len(ids)
        docs = []
        for doc_id, content, meta in zip(ids, documents, metadatas):
            if not content:
                continue
            meta = dict(meta) if meta else {}
            meta.setdefault("chunk_id", doc_id)
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def check_md5_hex(self, md5_check: str):
        md5_path = get_abs_path(chroma_config["md5_hex_store"])
        if not os.path.exists(md5_path):
            open(md5_path, "w", encoding="utf-8").close()
            return False
        with open(md5_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line.strip() == md5_check:
                    return True
        return False

    def save_md5_hex(self, md5_check: str):
        with open(get_abs_path(chroma_config["md5_hex_store"]),"a",encoding="utf-8") as f:
            f.write(md5_check + "")

    def get_file_documents(self, read_path: str):
        if read_path.endswith("txt"):
            return txt_loader(read_path)
        if read_path.endswith("pdf"):
            return pdf_loader(read_path)
        return None

    def load_document(self):
        allow_file_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"])
        )
        for path in allow_file_path:
            md5_hex = get_file_md5(path)
            if self.check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue
            try:
                documents: list[Document] = self.get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue
                split_document = self.spliter.split_documents(documents)
                logger.info(f"[加载知识库]{path}分片后内容：{split_document}")
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue
                self.vector_store.add_documents(split_document)
                self.save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败: {str(e)}", exc_info=True)

if __name__ == '__main__':
    vs = VectorStoreService(embedding_model)
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("小户型适合哪些扫地机器人")
    for r in res:
        print(r.page_content)
        print("-"*20)
