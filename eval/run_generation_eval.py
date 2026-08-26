# -*- coding: utf-8 -*-
"""生成质量评测：忠实度 / 答案相关性 / 完整度（LLM-as-Judge）。

评测对象是 RAG 生成阶段：直接用生产 RAG 服务做"改写→检索→拼接→生成"，
并把真实检索上下文交给裁判，从而准确度量：
- faithfulness 忠实度：回答是否被检索到的参考资料支撑（vs 上下文）。
- relevance    相关性：回答是否切题（vs 问题）。
- completeness 完整度：是否覆盖标准答案要点（vs 标准答案）。

与生产一致：上下文格式为 `[n]: 内容 | metadata`，生成 prompt 复用 rag_summarize。

用法（项目根目录，需先 build）：
    python eval/run_generation_eval.py --limit 3
    python eval/run_generation_eval.py --limit 50 --judge-model qwen-plus
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_DIR))

from rag.rag_service import get_rag_service

import gold
import judge

RESULTS_DIR = EVAL_DIR / "results"
EVAL_SET = EVAL_DIR / "eval_set.json"


def build_context(docs):
    context = ""
    for i, doc in enumerate(docs, 1):
        context += f"[{i}]: {doc.page_content} | {doc.metadata}" + "\n"
    return context


def generate(service, query, context):
    raw = (service.prompt_template | service.model).invoke({"input": query, "context": context})
    return raw.content if hasattr(raw, "content") else str(raw)


def main():
    parser = argparse.ArgumentParser(description="生成质量评测（faithfulness / relevance / completeness）")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--judge-model", default="qwen-plus")
    parser.add_argument("--kind", choices=["qa", "fault", "tip", "knowledge"], default="qa")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not EVAL_SET.exists():
        sys.exit("评测集不存在，请先运行: python eval/build_eval_set.py")

    data = gold.load_eval_set()
    items = [i for i in data["items"] if i["kind"] == args.kind][: args.limit]
    if not items:
        sys.exit(f"评测集内没有 kind={args.kind} 条目")

    service = get_rag_service()
    rows = []
    for item in items:
        q = item["query"]
        print(f"[{item['id']}] 生成中...")
        context_docs = service.retriever_docs(q)
        context = build_context(context_docs)
        answer = generate(service, q, context)
        print(f"  回答({len(answer)}字): {answer[:60]}...")

        f = judge.judge_faithfulness(q, context, answer, args.judge_model)
        r = judge.judge_answer_relevance(q, answer, args.judge_model)
        c = judge.judge_completeness(q, item["answer"], answer, args.judge_model)

        row = {
            "id": item["id"],
            "file": item["file"],
            "kind": item["kind"],
            "query": q,
            "reference": item["answer"],
            "n_context_chunks": len(context_docs),
            "context": context,
            "answer": answer,
            "faithfulness": f,
            "relevance": r,
            "completeness": c,
        }
        rows.append(row)
        print(
            f"  faithfulness={f.get('faithfulness')} relevance={r.get('relevance')} "
            f"completeness={c.get('completeness')}"
        )

    n = len(rows)
    summary = {
        "n": n,
        "faithfulness_mean": round(sum(r["faithfulness"].get("faithfulness", 0) for r in rows) / n, 2),
        "relevance_mean": round(sum(r["relevance"].get("relevance", 0) for r in rows) / n, 2),
        "completeness_mean": round(sum(r["completeness"].get("completeness", 0) for r in rows) / n, 2),
        "hallucination_rate": round(sum(1 for r in rows if r["faithfulness"].get("hallucination")) / n, 2),
    }
    out_path = args.out or (RESULTS_DIR / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n===== 生成质量汇总（LLM-as-Judge）=====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"原始数据: {out_path}")


if __name__ == "__main__":
    main()
