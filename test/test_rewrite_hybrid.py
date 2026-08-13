# -*- coding: utf-8 -*-
"""
改造 2 测试脚本：查询改写 + 混合检索

运行方式（在项目根目录 agent扫地机器人 下）：
    C:\Users\HP\PycharmProjects\PythonProject3\.venv\Scripts\python.exe test_rewrite_hybrid.py

注意：如果开着加速器/代理，先设置环境变量再跑：
    $env:NO_PROXY = "dashscope.aliyuncs.com"
"""
import sys
sys.path.insert(0, "..")

from rag.query_rewriter import QueryRewriter
from rag.vector_store import VectorStoreService
from rag.hybrid_retriever import HybridRetriever
from model.factory import embedding_model

# ========== 测试 1：QueryRewriter 单元测试 ==========
def test_rewriter_unit():
    print("=" * 60)
    print("[测试1] QueryRewriter 单元测试")
    rw = QueryRewriter()
    print(f"实例化 OK | enabled={rw.enabled} | max_queries={rw.max_queries}")

    # _parse 容错测试（不调 API）
    cases = [
        '["a", "b", "c"]',                              # 干净 JSON
        '```json\n["x", "y"]\n```',                     # markdown 包裹
        '好的，结果如下：["m", "n"] 请查收',              # 带废话
        "第一行\n第二行",                                 # 纯文本按行
        "not json at all",                              # 完全非法
    ]
    for c in cases:
        print(f"  _parse({c[:30]}...) -> {QueryRewriter._parse(c)}")

    # 开关回退
    rw.enabled = False
    assert rw.rewrite("随便问问") == ["随便问问"], "enabled=False 应回退原查询"
    rw.enabled = True
    print("  enabled=False 回退测试通过")


# ========== 测试 2：真实改写（调 API） ==========
def test_rewrite_api():
    print("=" * 60)
    print("[测试2] 真实改写（需要 .env 里的通义千问 Key）")
    rw = QueryRewriter()
    query = "小户型适合哪些扫地机器人"
    queries = rw.rewrite(query)
    print(f"原始: {query}")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    return queries


# ========== 测试 3：集成对比（改写 vs 不改写） ==========
def test_integration():
    print("=" * 60)
    print("[测试3] 集成对比：口语化问题 直接检索 vs 改写后检索")
    query = "我家50平，买哪个好"  # 口语化 + 指代

    vs = VectorStoreService(embedding_model)
    hr = HybridRetriever(vs)

    # 3a. 不改写
    raw_docs = hr.retrieve([query])
    print(f"\n[不改写] 候选 {len(raw_docs)} 条，Top5:")
    for doc, score in raw_docs[:5]:
        print(f"  {score:.3f} | {doc.page_content[:40]}...")

    # 3b. 改写后
    rw = QueryRewriter()
    queries = rw.rewrite(query)
    print(f"\n[改写后] {len(queries)} 条查询:")
    for q in queries:
        print(f"  - {q}")
    new_docs = hr.retrieve(queries)
    print(f"[改写后] 候选 {len(new_docs)} 条，Top5:")
    for doc, score in new_docs[:5]:
        print(f"  {score:.3f} | {doc.page_content[:40]}...")

    print(f"\n结论: 候选 {len(raw_docs)} → {len(new_docs)} 条")
    print("如果改写后 Top 结果更相关（命中'选购/户型'），说明改造 2 生效 ✅")


if __name__ == "__main__":
    test_rewriter_unit()
    test_rewrite_api()
    test_integration()
    print("\n全部测试执行完毕")
