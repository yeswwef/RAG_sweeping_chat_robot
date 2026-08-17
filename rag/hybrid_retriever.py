"""混合检索：稠密向量（Chroma）+ 稀疏检索（BM25），RRF 融合。

score(d) = Σ 1/(k + rank(d))，k=60（论文推荐值）
- 稠密路：语义相似
- 稀疏路：关键词精确命中（中文用 jieba 分词）
- BM25 索引惰性构建，进程内只建一次（内存缓存）
"""
import hashlib
import re
import threading
from typing import Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from model.factory import embedding_model
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_config
from utils.logger_handler import logger

#判断jieba库是否安装成功
#jieba 是 Python 中最常用的中文分词库，用来把中文句子切分成一个个有意义的词语（比如把“我爱北京”切成“我”、“爱”、“北京”）
try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


def tokenize(text: str) -> list[str]:
    """中文分词：优先 jieba；缺失时退化为 CJK 二元组 + 英文单词（零依赖兜底）。"""
    if _HAS_JIEBA:#jieba.cut(text)：把句子切成词，返回生成器（“我爱北京”→ 我/爱/北京）
        return [w.strip() for w in jieba.cut(text) if w.strip()]
    #下面就是如果jieba库没导进去的做法，效果一样
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = ["".join(cjk[i:i + 2]) for i in range(len(cjk) - 1)]
    return words + cjk + bigrams



#计算返回的文档的md5值,同一段正文不管从稠密路还是稀疏路拿到，指纹都一样，可用来对齐、去重。
def _doc_key(doc: Document) -> str:
    """正文哈希作为唯一键：稠密/稀疏两路拿到的 Document 内容一致，键可对齐去重。"""
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

###########      混合检索器     #############
class HybridRetriever:
    """稠密 + 稀疏 双路召回，RRF 融合排序。"""

    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store
        cfg = rag_config["hybrid_retrieval"]
        self.enabled = cfg["enabled"]#是否启用
        self.dense_top_n = cfg["dense_top_n"]#稠密各取多少候选
        self.sparse_top_n = cfg["sparse_top_n"]#稀疏各取多少候选
        self.rrf_k = cfg["rrf_k"]#RRF 的 k
        self._bm25: Optional[BM25Okapi] = None  # BM25 索引，初始没有
        self._corpus: list[Document] = []  # 全量语料缓存
        self._bm25_built = False  # 索引是否建过
        self._corpus_version: tuple[int, str] | None = None  # 建索引时的语料指纹
        self._bm25_lock = threading.Lock()  # 锁：防并发重复建

    def _ensure_bm25(self):
        """版本化惰性构建：语料版本（count + id 指纹）变化时才全量重建；进程内只建一次。

        注意：空语料时置 None，避免 rank_bm25 除零崩溃；语料后来入库后版本变化会触发重建。
        """
        with self._bm25_lock:#加锁,保证只有一个进程
            current_version = self.vector_store.get_corpus_version()
            if self._bm25_built and current_version == self._corpus_version:
                return
            self._bm25_built = True
            self._corpus_version = current_version
            docs = self.vector_store.get_all_documents()
            

            if not docs:
                logger.warning("[HybridRetriever] 向量库为空，跳过 BM25 索引构建")
                self._corpus, self._bm25 = [], None
                return
            self._corpus = docs
            self._bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])
            logger.info(
                f"[HybridRetriever] BM25 索引构建完成，共 {len(docs)} 个分块（语料版本 {current_version}）"
            )

    #稠密
    def _dense_hits(self, query: str) -> list[Document]:
        return self.vector_store.similarity_search(query, k=self.dense_top_n)
    #稀疏
    def _sparse_hits(self, query: str) -> list[Document]:
        self._ensure_bm25()
        if self._bm25 is None or not self._corpus:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.sparse_top_n]
        return [self._corpus[i] for i in top_idx if scores[i] > 0]

    def retrieve(self, queries: list[str]) -> list[tuple[Document, float]]:
        """对每个查询做稠密+稀疏召回，RRF 融合后返回 (文档, 融合分)，降序。"""
        rrf_score: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for q in queries:
            for rank, doc in enumerate(self._dense_hits(q)):
                key = _doc_key(doc)
                rrf_score[key] = rrf_score.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                doc_map.setdefault(key, doc)
            if self.enabled:
                for rank, doc in enumerate(self._sparse_hits(q)):
                    key = _doc_key(doc)
                    rrf_score[key] = rrf_score.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                    doc_map.setdefault(key, doc)
        ranked = sorted(rrf_score.items(), key=lambda kv: kv[1], reverse=True)
        results = [(doc_map[k], s) for k, s in ranked]
        logger.info(f"[HybridRetriever] 混合检索完成，候选 {len(results)} 条")
        return results


if __name__ == '__main__':
    """功能自测：分词 → 去重键 → BM25 构建 → 单路召回 → RRF 融合 → 降级验证。"""
    print("=" * 60)
    print("[1/6] tokenize 中文分词测试")
    print("  中文: ", tokenize("小户型适合哪些扫地机器人"))
    print("  中英混: ", tokenize("DustBot X3 电池续航 120分钟"))

    print("=" * 60)
    print("[2/6] _doc_key 去重键测试")
    d1 = Document(page_content="扫地机器人保养方法")
    d2 = Document(page_content="扫地机器人保养方法")
    d3 = Document(page_content="扫地机器人维修方法")
    print(f"  相同内容键一致: {_doc_key(d1) == _doc_key(d2)}")
    print(f"  不同内容键不同: {_doc_key(d1) != _doc_key(d3)}")

    print("=" * 60)
    print("[3/6] HybridRetriever 初始化 + BM25 索引构建")
    vs = VectorStoreService(embedding_model)
    hr = HybridRetriever(vs)
    hr._ensure_bm25()
    print(f"  BM25 索引: {'已构建' if hr._bm25 is not None else '未构建(空库)'}, 语料分块数: {len(hr._corpus)}")
    if hr._bm25 is None:
        print("  [警告] 向量库为空，后续稀疏路结果会为空列表——属正常降级")

    print("=" * 60)
    print("[4/6] 单路召回测试 (query='小户型适合哪些扫地机器人')")
    dense = hr._dense_hits("小户型适合哪些扫地机器人")
    sparse = hr._sparse_hits("小户型适合哪些扫地机器人")
    print(f"  稠密路召回 {len(dense)} 条")
    for i, d in enumerate(dense[:3], 1):
        print(f"    #{i} {d.page_content[:50]}...")
    print(f"  稀疏路召回 {len(sparse)} 条")
    for i, d in enumerate(sparse[:3], 1):
        print(f"    #{i} {d.page_content[:50]}...")

    print("=" * 60)
    print("[5/6] retrieve 混合检索（RRF 融合，多查询）")
    queries = ["小户型适合哪些扫地机器人", "扫地机器人怎么保养"]
    results = hr.retrieve(queries)
    print(f"  共返回 {len(results)} 条，分数降序: ")
    scores = [s for _, s in results]
    print(f"  分数列表: {[round(s, 4) for s in scores]}")
    print(f"  降序验证: {all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))}")
    keys = [_doc_key(d) for d, _ in results]
    print(f"  去重验证: {len(keys) == len(set(keys))}")
    for i, (doc, score) in enumerate(results[:5], 1):
        src = doc.metadata.get("source", doc.metadata.get("chunk_id", "?"))
        print(f"    #{i} score={score:.4f} | {doc.page_content[:45]}... | 来源:{src}")

    print("=" * 60)
    print("[6/6] enabled=False 退化纯稠密验证")
    hr.enabled = False
    results2 = hr.retrieve(["小户型适合哪些扫地机器人"])
    print(f"  关闭混合后返回 {len(results2)} 条（应为纯稠密结果）")
    hr.enabled = True
    print("=" * 60)
    print("[完成] 全部测试执行完毕")
