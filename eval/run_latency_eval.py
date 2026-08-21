# -*- coding: utf-8 -*-
"""端到端分阶段耗时 / Token / 成本评测。

阶段：查询改写 → 稠密嵌入 → 稀疏+RRF 融合 → 生成。
token 来自生成模型的 response_metadata.token_usage（改写/嵌入阶段的 token 未单独统计）。

用法（项目根目录，需先 build）：
    python eval/run_latency_eval.py --limit 5
    python eval/run_latency_eval.py --limit 20 --price-input 2.0 --price-output 8.0
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_DIR))

from rag.hybrid_retriever import HybridRetriever, _doc_key, tokenize
from rag.query_rewriter import QueryRewriter
from rag.vector_store import VectorStoreService
from rag.rag_service import get_rag_service
from model.factory import embedding_model
from utils.config_handler import rag_config

import gold
import metrics

RESULTS_DIR = EVAL_DIR / "results"
EVAL_SET = EVAL_DIR / "eval_set.json"


def sparse_top(hr, q, k):
    hr._ensure_bm25()
    if hr._bm25 is None:
        return []
    scores = hr._bm25.get_scores(tokenize(q))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [hr._corpus[i] for i in top if scores[i] > 0]


def rrf_fuse(dense_sets, sparse_sets, k=60):
    """与 hybrid_retriever.retrieve 的 RRF 融合保持一致（评测脚本专用，避免重复计时污染）。"""
    rrf = {}
    doc_map = {}
    for docs in dense_sets:
        for rank, doc in enumerate(docs):
            key = _doc_key(doc)
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map.setdefault(key, doc)
    for docs in sparse_sets:
        for rank, doc in enumerate(docs):
            key = _doc_key(doc)
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map.setdefault(key, doc)
    return [doc_map[key] for key, _ in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)]


def build_context(docs):
    context = ""
    for i, doc in enumerate(docs, 1):
        context += f"[{i}]: {doc.page_content} | {doc.metadata}" + "\n"
    return context


def main():
    parser = argparse.ArgumentParser(description="分阶段耗时/成本评测")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--price-input", type=float, default=0.0)
    parser.add_argument("--price-output", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not EVAL_SET.exists():
        sys.exit("评测集不存在，请先运行: python eval/build_eval_set.py")

    data = gold.load_eval_set()
    items = [i for i in data["items"] if i["kind"] in ("qa", "fault")][: args.limit]
    if not items:
        sys.exit("评测集为空")

    cfg = rag_config["hybrid_retrieval"]
    dense_k, sparse_k, rrf_k = cfg["dense_top_n"], cfg["sparse_top_n"], cfg["rrf_k"]

    vs = VectorStoreService(embedding_model)
    hr = HybridRetriever(vs)
    rw = QueryRewriter()
    service = get_rag_service()

    rows = []
    stages = ["rewrite_ms", "dense_ms", "sparse_fuse_ms", "summarize_ms", "total_ms"]
    agg = {s: [] for s in stages}
    tokens_in, tokens_out = [], []

    for item in items:
        q = item["query"]
        row = {"id": item["id"], "file": item["file"], "kind": item["kind"], "query": q}

        t0 = time.perf_counter()
        queries = rw.rewrite(q)
        row["n_queries"] = len(queries)
        t1 = time.perf_counter()

        dense_sets = [vs.similarity_search(qq, k=dense_k) for qq in queries]
        t2 = time.perf_counter()

        sparse_sets = [sparse_top(hr, qq, sparse_k) for qq in queries]
        docs = rrf_fuse(dense_sets, sparse_sets, rrf_k)
        context = build_context(docs)
        t3 = time.perf_counter()

        raw = (service.prompt_template | service.model).invoke({"input": q, "context": context})
        answer = raw.content if hasattr(raw, "content") else str(raw)
        t4 = time.perf_counter()

        usage = (raw.response_metadata or {}).get("token_usage") or {}
        row["answer_len"] = len(answer)
        row["n_candidates"] = len(docs)
        row["rewrite_ms"] = round((t1 - t0) * 1000, 1)
        row["dense_ms"] = round((t2 - t1) * 1000, 1)
        row["sparse_fuse_ms"] = round((t3 - t2) * 1000, 1)
        row["summarize_ms"] = round((t4 - t3) * 1000, 1)
        row["total_ms"] = round((t4 - t0) * 1000, 1)
        row["input_tokens"] = usage.get("input_tokens")
        row["output_tokens"] = usage.get("output_tokens")
        for s in stages:
            agg[s].append(row[s])
        if row["input_tokens"] is not None:
            tokens_in.append(row["input_tokens"])
            tokens_out.append(row["output_tokens"] or 0)
        rows.append(row)
        print(
            f"[{row['id']}] 改写 {row['rewrite_ms']}ms | 稠密 {row['dense_ms']}ms | "
            f"稀疏+融合 {row['sparse_fuse_ms']}ms | 生成 {row['summarize_ms']}ms | 总 {row['total_ms']}ms"
        )

    summary = {
        "stage": {
            s: {
                "p50_ms": round(metrics.percentile(agg[s], 0.5), 1),
                "p95_ms": round(metrics.percentile(agg[s], 0.95), 1),
                "mean_ms": round(metrics.mean(agg[s]), 1),
            }
            for s in stages
        },
        "tokens": {
            "input_mean": round(metrics.mean(tokens_in), 1) if tokens_in else None,
            "output_mean": round(metrics.mean(tokens_out), 1) if tokens_out else None,
        },
    }
    if args.price_input or args.price_output:
        cost = (
            sum(tokens_in) / 1_000_000 * args.price_input
            + sum(tokens_out) / 1_000_000 * args.price_output
        )
        summary["cost_estimate_yuan"] = round(cost, 4)
        summary["price_yuan_per_m"] = {"input": args.price_input, "output": args.price_output}

    out_path = args.out or (RESULTS_DIR / f"latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"meta": {"generated_at": datetime.now().isoformat(timespec="seconds"), "n": len(rows)}, "summary": summary, "rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== 分阶段耗时汇总 =====")
    labels = {
        "rewrite_ms": "查询改写",
        "dense_ms": "稠密嵌入",
        "sparse_fuse_ms": "稀疏+融合",
        "summarize_ms": "LLM 生成",
        "total_ms": "总耗时",
    }
    print(f"{'阶段':<14}{'P50(ms)':>10}{'P95(ms)':>10}{'均值(ms)':>10}")
    for s in stages:
        v = summary["stage"][s]
        print(f"{labels[s]:<14}{v['p50_ms']:>10}{v['p95_ms']:>10}{v['mean_ms']:>10}")
    print(f"Token 均值: 输入 {summary['tokens']['input_mean']} / 输出 {summary['tokens']['output_mean']}")
    if "cost_estimate_yuan" in summary:
        print(f"成本估算: {summary['cost_estimate_yuan']} 元（{len(rows)} 条）")
    print(f"原始数据: {out_path}")


if __name__ == "__main__":
    main()
