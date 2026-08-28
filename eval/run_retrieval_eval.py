# -*- coding: utf-8 -*-
"""检索指标评测：四组基线 + context precision。

策略：纯稠密 / 纯 BM25 / 混合 RRF / 混合+改写。
指标：hit@k / recall@k / mrr@k / ndcg@k / context_precision@k（新增，RAGAS AP）。
ground truth：由 gold.derive_gold 派生（答案最长前缀命中 chunk）。

用法（项目根目录，需先 build）：
    python eval/run_retrieval_eval.py --sample 20
    python eval/run_retrieval_eval.py --sample 100
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
from model.factory import embedding_model
from utils.config_handler import rag_config

import gold
import metrics

RESULTS_DIR = EVAL_DIR / "results"
EVAL_SET = EVAL_DIR / "eval_set.json"
STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rewrite")#评测参赛选手

#分层均匀抽样
def sample_items(items, n):
    if n >= len(items):
        return items

    by_file = {}#空字典  key=文件名，value=该文件的题列表
    order = []#空列表，记住文件出现的先后顺序


    for item in items:
        if item["file"] not in by_file:
            by_file[item["file"]] = []
            order.append(item["file"])
        by_file[item["file"]].append(item)

    # 关键：把每个文件里的“原版/变体”交错排开。
    # 否则均衡抽样永远只取到排在前面的原版题，变体题一次都抽不到。
    for f in order:
        orig = [i for i in by_file[f] if not i.get("variant")]
        var = [i for i in by_file[f] if i.get("variant")]
        interleaved = []
        oi = vi = 0
        while oi < len(orig) or vi < len(var):
            if oi < len(orig):
                interleaved.append(orig[oi])
                oi += 1
            if vi < len(var):
                interleaved.append(var[vi])
                vi += 1
        by_file[f] = interleaved

    result = []
    idx = {f: 0 for f in order}
    while len(result) < n:
        progressed = False
        for f in order:
            if idx[f] < len(by_file[f]):
                result.append(by_file[f][idx[f]])
                idx[f] += 1
                progressed = True
                if len(result) >= n:
                    break
        if not progressed:
            break
    return result


def strategy_dense(vs, q, k):
    return vs.similarity_search(q, k=k)


def strategy_bm25(hr, q, k):
    hr._ensure_bm25()
    if hr._bm25 is None:
        return []
    scores = hr._bm25.get_scores(tokenize(q))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [hr._corpus[i] for i in top if scores[i] > 0]


def strategy_hybrid(hr, q):
    return [d for d, _ in hr.retrieve([q])]


def strategy_hybrid_rewrite(hr, rw, q):
    queries = rw.rewrite(q)

    return queries, [d for d, _ in hr.retrieve(queries)]


def main():
    parser = argparse.ArgumentParser(description="检索指标评测（含 context precision）")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--include-tips", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not EVAL_SET.exists():
        sys.exit("评测集不存在，请先运行: python eval/build_eval_set.py")

    data = gold.load_eval_set()
    items = [i for i in data["items"] if i["kind"] in ("qa", "fault", "knowledge") or args.include_tips]
    items = sample_items(items, args.sample)
    print(f"评测集: 共 {data['meta']['total']} 条，本次抽样 {len(items)} 条")

    cfg = rag_config["hybrid_retrieval"]
    dense_k = cfg["dense_top_n"]
    sparse_k = cfg["sparse_top_n"]

    vs = VectorStoreService(embedding_model)
    hr = HybridRetriever(vs)
    rw = QueryRewriter()

    t0 = time.time()
    hr._ensure_bm25()
    print(f"BM25 索引构建: {time.time() - t0:.2f}s（语料 {len(hr._corpus)} 分块）")
    derived, unmatched = gold.derive_gold(hr._corpus, items)
    usable = [i for i in items if derived.get(i["id"])]
    print(f"ground truth 命中: {len(usable)}/{len(items)}，无匹配已排除: {len(unmatched)}")

    # 与 metrics.retrieval_metrics 的键保持一致（含新增 recall_full）
    metric_names = list(metrics.retrieval_metrics([], set()).keys())
    results = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "eval_set": EVAL_SET.name,
            "sample_size": len(items),
            "usable_size": len(usable),
            "excluded_no_gold": len(unmatched),
            "variant_items": sum(1 for i in usable if i.get("variant")),
            "config": {"dense_top_n": dense_k, "sparse_top_n": sparse_k, "rrf_k": cfg["rrf_k"]},
            "rewrite_enabled": rw.enabled,
        },
        "strategies": {},
        "per_item": {},
    }

    for strat in STRATEGIES:
        agg = {m: [] for m in metric_names}
        latencies = []
        groups: dict[str, dict[str, list]] = {}
        for item in usable:
            q = item["query"]
            g = derived[item["id"]]
            if strat == "dense":
                t = time.perf_counter()
                docs = strategy_dense(vs, q, dense_k)
                lat = time.perf_counter() - t
            elif strat == "bm25":
                t = time.perf_counter()
                docs = strategy_bm25(hr, q, sparse_k)
                lat = time.perf_counter() - t
            elif strat == "hybrid":
                t = time.perf_counter()
                docs = strategy_hybrid(hr, q)
                lat = time.perf_counter() - t
            else:
                t = time.perf_counter()
                queries, docs = strategy_hybrid_rewrite(hr, rw, q)
                lat = time.perf_counter() - t
            ranked_keys = [_doc_key(d) for d in docs]
            m = metrics.retrieval_metrics(ranked_keys, g)
            gk = f"{item['kind']}/{'variant' if item.get('variant') else 'original'}"
            groups.setdefault(gk, {mm: [] for mm in metric_names})
            for k, v in m.items():
                agg[k].append(v)
                groups[gk][k].append(v)
            latencies.append(lat * 1000)
            results["per_item"].setdefault(item["id"], {})[strat] = {
                **m,
                "latency_ms": round(lat * 1000, 1),
                "n_candidates": len(docs),
            }
        results["strategies"][strat] = {k: round(metrics.mean(v), 4) for k, v in agg.items()}
        results["strategies"][strat]["by_group"] = {
            gk: {k: round(metrics.mean(v), 4) for k, v in vals.items()}
            for gk, vals in groups.items()
        }
        results["strategies"][strat]["avg_latency_ms"] = round(metrics.mean(latencies), 1)
        results["strategies"][strat]["p50_latency_ms"] = round(metrics.percentile(latencies, 0.5), 1)
        print(f"[{strat}] 完成，平均耗时 {results['strategies'][strat]['avg_latency_ms']}ms")

    out_path = args.out or (RESULTS_DIR / f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 检索指标对比 =====")
    display = [
        ("hit@5", "hit@5"),
        ("recall@5", "recall@5"),
        ("mrr@10", "mrr@10"),
        ("ndcg@10", "ndcg@10"),
        ("context_precision@5", "context_precision@5"),
        ("recall_full", "recall_full"),
        ("avg_ms", "avg_latency_ms"),
    ]
    header = f"{'策略':<16}" + "".join(f"{label:>20}" for label, _ in display)
    print(header)
    for strat in STRATEGIES:
        s = results["strategies"][strat]
        row = f"{strat:<16}" + "".join(f"{s[key]:>20}" for _, key in display)
        print(row)

    print("\n===== 按 类型/原版vs变体 分组（recall@5 / recall_full / context_precision@5）=====")
    print(f"{'策略':<16}{'分组':<24}{'recall@5':>12}{'recall_full':>12}{'cp@5':>12}")
    for strat in STRATEGIES:
        s = results["strategies"][strat]
        for gk, vals in s.get("by_group", {}).items():
            print(
                f"{strat:<16}{gk:<24}"
                f"{vals.get('recall@5', 0):>12}"
                f"{vals.get('recall_full', 0):>12}"
                f"{vals.get('context_precision@5', 0):>12}"
            )
    print(f"\n原始数据: {out_path}")


if __name__ == "__main__":
    main()
