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
SEGMENT_MIN = 8
_SEG_SPLIT = re.compile(r"[；;。！？!?\n]+")


def norm_text(s):
    return re.sub(r"\s+", "", s or "")


def _match_keys(corpus, corpus_norm, probe):
    """返回正文包含 probe 的分块 md5 键（升序、去重）。"""
    return sorted({_doc_key(d) for d, t in zip(corpus, corpus_norm) if probe and probe in t})


def derive_relevant_keys(corpus, answer, min_prefix=MIN_PREFIX, segment_min=SEGMENT_MIN):
    """返回答案命中的相关 chunk md5 键（升序、去重）。

    策略升级（支持一题多个相关分块）：
    1. 先把答案按句级分隔符（；;。！？!?换行）拆成多段，每段长度 >= segment_min
       就去语料分块里找包含它的 chunk；多段命中会并成多个相关分块，
       不再像旧逻辑那样只取"答案开头所在的那 1 块"。
    2. 若整段拆分一无所获，退化为"答案最长前缀逐级缩短"匹配，
       保证至少能找到答案开头所在的分块（覆盖率不会因此下降）。
    """
    probe = norm_text(answer)
    if len(probe) < min_prefix:
        return []
    corpus_norm = [norm_text(d.page_content) for d in corpus]

    keys: set[str] = set()
    for seg in _SEG_SPLIT.split(probe):
        seg = seg.strip()
        if len(seg) >= segment_min:
            keys.update(_match_keys(corpus, corpus_norm, seg))
    if keys:
        return sorted(keys)

    for end in range(len(probe), min_prefix - 1, -1):
        prefix = probe[:end]
        keys.update(_match_keys(corpus, corpus_norm, prefix))
        if keys:
            return sorted(keys)
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
