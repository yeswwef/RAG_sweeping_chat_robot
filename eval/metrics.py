# -*- coding: utf-8 -*-
"""检索指标纯函数实现（无 I/O、无网络调用）。

约定：
- ranked：检索结果按相关性降序的文档键列表（chunk 正文 md5）。
- gold  ：该 query 的相关文档键集合（由 gold.py 派生）。

所有指标均为二值相关性口径；context_precision 即 RAGAS 语境精确率，
数学上等价于 Average Precision@k，用来量化"相关 chunk 是否排前、无关 chunk 是否稀释上下文"。
"""
import math

KS = (5, 10)

#取前 k 个结果，只要里面有一个是相关 chunk 就记 1，否则 0。衡量“有没有命中”
def hit_at_k(ranked, gold, k):
    top = ranked[:k]
    return 1.0 if any(g in top for g in gold) else 0.0

#前 k 里命中多少个相关 chunk，除以相关 chunk 总数。gold 为空时避免除零返回 0。衡量“找回了多少
def recall_at_k(ranked, gold, k):
    if not gold:
        return 0.0
    top = set(ranked[:k])
    return len(top & set(gold)) / len(gold)

#从第 1 位开始找第一个相关 chunk，返回 1/排名；没找到返回 0。排名越靠前分越高
def mrr_at_k(ranked, gold, k):
    for i, key in enumerate(ranked[:k], 1):
        if key in gold:
            return 1.0 / i
    return 0.0

#每个位置 i 上，相关就给 1/log2(i+1)（位置越靠前权重越大，因为有折损）
def ndcg_at_k(ranked, gold, k):
    top = ranked[:k]
    dcg = sum(
        (1.0 if key in gold else 0.0) / math.log2(i + 1)
        for i, key in enumerate(top, 1)
    )
    ideal_n = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1)) if gold else 0.0
    return dcg / idcg if idcg > 0 else 0.0

#running 是“到当前位置为止的累计相关数”，running/i 是当前位置的 precision；
# 只在相关位置上累加这个 precision，最后除以相关总数。它惩罚“相关 chunk 被排到后面、前面被无关 chunk 稀释”的情况
def context_precision_at_k(ranked, gold, k):
    """Average Precision@k，等价于 RAGAS context precision。"""
    top = ranked[:k]
    if not top:
        return 0.0
    rel = [1 if key in gold else 0 for key in top]
    total_rel = sum(rel)
    if total_rel == 0:
        return 0.0
    running = 0
    score = 0.0
    for i, r in enumerate(rel, 1):
        running += r
        if r:
            score += running / i
    return score / total_rel

#把前 k 个结果转成 0/1 相关序列 rel，并统计其中相关总数；没有相关或没有结果就直接返回 0
def context_recall_at_k(ranked, gold, k):
    """RAGAS context recall：top-k 命中相关 chunk 数 / 相关 chunk 总数。"""
    return recall_at_k(ranked, gold, k)


def recall_full(ranked, gold):
    """整批召回：检索返回的全部候选里命中相关 chunk 数 / 相关 chunk 总数（不截断 top-k）。

    用途：RAG 生成阶段会把检索返回的全部候选拼进 prompt，因此"整批有没有捞全"
    比只看 top-5/10 更贴近线上实际。
    """
    return recall_at_k(ranked, gold, len(ranked))


def retrieval_metrics(ranked, gold):
    """返回该 query 的全部检索指标 dict。"""
    out = {}
    for k in KS:
        out[f"hit@{k}"] = hit_at_k(ranked, gold, k)
        out[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"mrr@{k}"] = mrr_at_k(ranked, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
        out[f"context_precision@{k}"] = context_precision_at_k(ranked, gold, k)
    out["recall_full"] = recall_full(ranked, gold)
    return out


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def percentile(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]
