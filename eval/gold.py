# -*- coding: utf-8 -*-
"""黄金集派生与加载。

ground truth 口径（自动派生、可复现）：
语料是 QA 结构，eval_set.json 每条 item 的 answer 即"应被检索到的资料原文"。
用"答案的最长前缀在语料分块正文中的出现"判定相关 chunk：
- 从完整归一化答案开始，逐步缩短前缀，直到命中某个分块；
- 命中分块即视为相关 chunk（用正文 md5 作为稳定键，与生产 _doc_key 一致）。

注意：这是自动派生 gold，不是人工标注；语义等价的改写答案仍可能漏判，
validate_gold.py 会输出空 gold 率供人工复核。
"""
import json
import re
from pathlib import Path

from rag.hybrid_retriever import _doc_key

EVAL_DIR = Path(__file__).resolve().parent
EVAL_SET = EVAL_DIR / "eval_set.json"
MIN_PREFIX = 12


def norm_text(s):
    return re.sub(r"\s+", "", s or "")


def derive_relevant_keys(corpus, answer, min_prefix=MIN_PREFIX):
    """返回答案命中的相关 chunk md5 键（升序、去重）。"""
    probe = norm_text(answer)
    if len(probe) < min_prefix:
        return []
    corpus_norm = [norm_text(d.page_content) for d in corpus]
    for end in range(len(probe), min_prefix - 1, -1):
        prefix = probe[:end]
        keys = sorted({_doc_key(d) for d, t in zip(corpus, corpus_norm) if prefix in t})
        if keys:
            return keys
    return []


def derive_gold(corpus, items):
    """返回 ({item_id: [keys]}, unmatched_ids)。"""
    gold = {}
    unmatched = []
    for item in items:
        keys = derive_relevant_keys(corpus, item.get("answer", ""))
        if keys:
            gold[item["id"]] = keys
        else:
            unmatched.append(item["id"])
    return gold, unmatched


def load_eval_set(path=None):
    p = Path(path) if path else EVAL_SET
    return json.loads(p.read_text(encoding="utf-8"))
