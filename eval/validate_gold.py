# -*- coding: utf-8 -*-
"""黄金集质量校验（离线，不调 LLM API）。

对 eval_set.json 与派生 gold 输出质量报告：
- 覆盖率：能派生到 >=1 个相关 chunk 的条目占比（空 gold 会被检索/生成评测自动排除）。
- 分 kind 的空 gold 明细。
- query / answer 重复率与疑似重复对。
- 每条平均相关 chunk 数（>1 说明答案跨多个分块）。
- 可选 --save-gold 把派生 gold 落盘（含相关 chunk 正文），便于人工复核。

用法（项目根目录，需先 build，且向量库已灌入语料）：
    python eval/validate_gold.py
    python eval/validate_gold.py --save-gold results/gold_contexts.json
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_DIR))

from rag.hybrid_retriever import HybridRetriever, _doc_key
from rag.vector_store import VectorStoreService
from model.factory import embedding_model

import gold


def load_corpus():
    vs = VectorStoreService(embedding_model)
    hr = HybridRetriever(vs)
    hr._ensure_bm25()
    return hr._corpus


def duplicate_pairs(items, field):
    seen = {}
    for it in items:
        key = gold.norm_text(it.get(field, ""))
        if not key:
            continue
        seen.setdefault(key, []).append(it["id"])
    return [{"norm": k, "ids": ids} for k, ids in seen.items() if len(ids) > 1]


def main():
    parser = argparse.ArgumentParser(description="黄金集质量校验")
    parser.add_argument("--eval-set", type=Path, default=None)
    parser.add_argument("--save-gold", type=Path, default=None, help="把派生 gold 落盘为 JSON")
    args = parser.parse_args()

    data = gold.load_eval_set(args.eval_set)
    items = data["items"]
    corpus = load_corpus()

    if not corpus:
        print("[警告] 向量库为空：请先灌入语料（运行 app 或 rag/vector_store.py 的 load_document）后再校验。")

    derived, unmatched = gold.derive_gold(corpus, items)
    id2item = {i["id"]: i for i in items}
    unmatched_by_kind = Counter(id2item[i]["kind"] for i in unmatched)
    rel_counts = [len(v) for v in derived.values()]
    originals = [i for i in items if not i.get("variant")]

    n_total = len(items)
    n_covered = len(derived)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_chunks": len(corpus),
        "eval_items": n_total,
        "by_kind": dict(Counter(i["kind"] for i in items)),
        "gold_coverage": {
            "covered": n_covered,
            "unmatched": len(unmatched),
            "coverage_rate": round(n_covered / n_total, 4) if n_total else 0.0,
            "unmatched_by_kind": dict(unmatched_by_kind),
        },
        "avg_relevant_chunks": round(sum(rel_counts) / len(rel_counts), 2) if rel_counts else 0.0,
        # 变体条目故意与原版共用答案，去重检查只针对原版题目
        "duplicate_queries": len(duplicate_pairs(originals, "query")),
        "duplicate_answers": len(duplicate_pairs(originals, "answer")),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if unmatched:
        print(f"\n[空 gold 明细] 共 {len(unmatched)} 条：")
        for i in unmatched:
            print(f"  - {i} [{id2item[i]['kind']}] {id2item[i]['query'][:40]}")

    if args.save_gold:
        payload = {
            "meta": report,
            "items": [
                {
                    "id": item["id"],
                    "file": item["file"],
                    "kind": item["kind"],
                    "query": item["query"],
                    "reference_answer": item["answer"],
                    "relevant_chunks": [
                        next((d.page_content for d in corpus if _doc_key(d) == k), "")
                        for k in derived.get(item["id"], [])
                    ],
                }
                for item in items
                if item["id"] in derived
            ],
        }
        args.save_gold.parent.mkdir(parents=True, exist_ok=True)
        args.save_gold.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n派生 gold 已落盘: {args.save_gold}")


if __name__ == "__main__":
    main()
