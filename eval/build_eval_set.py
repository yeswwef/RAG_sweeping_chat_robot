# -*- coding: utf-8 -*-
"""评测集构造：从 data/ 语料解析 QA/故障/建议/陈述句知识，生成 eval/eval_set.json。

语料结构：
- qa   ：`N. **问题**` + `- 答案`（100问 × 2，query 就是真实问题）
- fault：`N. 故障现象：…；检测：…；修复：…`（query 取故障现象）
- tip  ：`N. 陈述句`（维护保养/选购指南，启发式 query，默认不参与）
- knowledge：`N. 陈述句`（产品知识/使用技巧/安装设置等 8 个新知识文件，默认参与，
  query 由"主题短语 + 领域问法模板"生成，避免直接用陈述句当问题）

用法（项目根目录）：
    python eval/build_eval_set.py
    python eval/build_eval_set.py --include-tips
    python eval/build_eval_set.py --include-pdf
"""
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_FILE = Path(__file__).resolve().parent / "eval_set.json"

QA_FILES = ["扫地机器人100问2.txt", "扫拖一体机器人100问.txt"]
FAULT_FILE = "故障排除.txt"
TIP_FILES = ["维护保养.txt", "选购指南.txt"]
PDF_FILE = "扫地机器人100问.pdf"

# 新知识文件（陈述句风格，默认参与评测）：文件名 -> 问法模板类型
KNOWLEDGE_FILES = {
    "术语词典.txt": "term",        # X是什么意思？
    "产品知识.txt": "what",        # X是什么？
    "使用技巧.txt": "how",         # X怎么做？
    "安装设置.txt": "setup",       # X怎么设置？
    "耗材配件.txt": "consumable",  # X怎么选购更换？
    "安全须知.txt": "safety",      # X要注意什么？
    "智能功能.txt": "feature",     # X怎么用？
    "售后服务.txt": "service",     # X怎么办？
}

_HEADER_RE = re.compile(r"^\s*#")
_QA_RE = re.compile(r"^\s*(\d+)\.\s*\*\*(.+?)\*\*\s*$")
_ITEM_RE = re.compile(r"^\s*(\d+)\.\s*(.+)$")

# 同义/口语化变体规则（按"最长优先、单向替换"排列，避免互相撤销）
SYNONYMS = [
    ("扫拖一体机器人", "扫拖机器人"),
    ("扫地机器人", "扫地机"),
    ("充电座", "充电桩"),
    ("怎么处理", "怎么解决"),
    ("怎么办", "如何处理"),
    ("无法连接", "连不上"),
    ("无法", "不能"),
    ("不工作", "没反应"),
    ("无反应", "没反应"),
    ("不旋转", "不转"),
    ("不转动", "不转"),
    ("怎么", "如何"),
    ("保养", "维护"),
]

#归一化文本——把空格、换行、tab 等所有空白删掉，返回紧凑字符串
def norm_text(s):
    return re.sub(r"\s+", "", s or "")

#评测条目的“工厂函数”——用给定的参数拼出一条标准格式的字典
def _make_item(file_stem, idx, kind, query, answer, variant=0):
    #文件名:类型:序号[:变体]
    seed = f"{file_stem}:{kind}:{idx}:v{variant}" if variant else f"{file_stem}:{kind}:{idx}"
    item_id = hashlib.md5(seed.encode("utf-8")).hexdigest()[:10]#做 md5 取前 10 位
    suffix = f"-v{variant}" if variant else ""
    return {
        "id": f"{file_stem}-{idx:03d}{suffix}-{item_id}",
        "file": file_stem,
        "kind": kind,
        "query": query,
        "answer": answer,
        "variant": bool(variant),
    }

#解析“问答”格式语料
def parse_qa(text, file_stem):
    items = []
    cur_q = None
    cur_a = []

    def flush():
        nonlocal cur_q, cur_a
        if cur_q and cur_a:
            items.append(_make_item(file_stem, len(items) + 1, "qa", cur_q, "\n".join(cur_a)))
        cur_q, cur_a = None, []

    for line in text.splitlines():
        if _HEADER_RE.match(line) or not line.strip():
            continue
        m = _QA_RE.match(line)
        if m:
            flush()
            cur_q = m.group(2).strip()
            cur_a = []
            continue
        if cur_q is not None:
            cur_a.append(line.strip().lstrip("-").strip())
    flush()
    return items

#解析故障文件——每行是 N. 故障现象：…；检测：…；修复：…
def parse_fault(text, file_stem):
    items = []
    for line in text.splitlines():
        if _HEADER_RE.match(line) or not line.strip():
            continue
        m = _ITEM_RE.match(line)
        if not m:
            continue
        body = m.group(2).strip()
        fm = re.search(r"故障现象：(.*?)(?:；检测|$)", body)
        if not fm:
            continue
        query = fm.group(1).strip()
        items.append(_make_item(file_stem, len(items) + 1, "fault", query, body))
    return items

#从一条陈述句里提取“主题短语”
def _extract_topic(body):
    for sep in ("：", ":", "，", ",", "；", "。", "？", "（", "(", " "):
        idx = body.find(sep)
        if idx > 0:
            topic = body[:idx].strip()
            if topic:
                return topic
    return body[:20].strip()

#解析保养/选购这类陈述句文件
def parse_tip(text, file_stem):
    items = []
    for line in text.splitlines():
        if _HEADER_RE.match(line) or not line.strip():
            continue
        m = _ITEM_RE.match(line)
        if not m:
            continue
        body = m.group(2).strip()
        items.append(_make_item(file_stem, len(items) + 1, "tip", _extract_topic(body), body))
    return items


def _knowledge_query(topic, style):
    """按领域问法模板把主题短语变成更接近真实用户的问题。"""
    topic = topic.strip()
    if style == "term":
        return f"{topic}是什么意思？"
    if style == "what":
        return f"{topic}是什么？"
    if style == "how":
        return f"{topic}怎么做？"
    if style == "setup":
        return f"{topic}怎么设置？"
    if style == "consumable":
        return f"{topic}怎么选购更换？"
    if style == "safety":
        return f"{topic}要注意什么？"
    if style == "feature":
        return f"{topic}怎么用？"
    if style == "service":
        return f"{topic}怎么办？"
    return f"{topic}是什么？"

#解析 8 个陈述句知识文件，生成 knowledge 条目
def parse_knowledge(text, file_stem, style):
    """陈述句知识文件：`N. 陈述句`，query 由主题短语 + 领域问法生成，answer 为整条陈述。"""
    items = []
    for line in text.splitlines():
        if _HEADER_RE.match(line) or not line.strip():
            continue
        m = _ITEM_RE.match(line)
        if not m:
            continue
        body = m.group(2).strip()
        if not body:
            continue
        topic = _extract_topic(body)
        # 去掉括号补充（如"主刷（滚刷）" -> "主刷"），再清掉残留分隔符
        topic = re.sub(r"[（(][^（）()]*[）)]", "", topic).strip("：:，,。；;？? ")
        if not topic:
            topic = body[:12]
        query = _knowledge_query(topic, style)
        items.append(_make_item(file_stem, len(items) + 1, "knowledge", query, body))
    return items


def parse_pdf(path, file_stem):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return parse_qa(text, file_stem)


def _make_variant(query):
    """生成一条更难的同义/口语化 query；无法生成时返回 None。"""
    variant = query
    replaced = 0
    for src, dst in SYNONYMS:
        if src in variant:
            variant = variant.replace(src, dst, 1)
            replaced += 1
            if replaced >= 2:
                break
    if variant == query:
        base = query.rstrip("？?")
        if base:
            variant = f"请问{base}？"
    return variant if variant != query else None


def _make_variant_items(items):
    """为 qa/fault/knowledge 原版条目各生成一条变体（kind 不变，gold 仍指向同一答案）。"""
    extra = []
    for item in items:
        if item["kind"] not in ("qa", "fault", "knowledge"):
            continue
        vq = _make_variant(item["query"])
        if not vq:
            continue
        try:
            idx = int(item["id"].split("-")[1])
        except (IndexError, ValueError):
            idx = len(items)
        extra.append(_make_item(item["file"], idx, item["kind"], vq, item["answer"], variant=1))
    return extra


def main():
    #注册评测集
    parser = argparse.ArgumentParser(description="构造评测集")
    parser.add_argument("--include-tips", action="store_true")#tip 条目注册tips条目
    parser.add_argument("--include-pdf", action="store_true")#注册是否解析pdf
    parser.add_argument("--out", type=Path, default=OUT_FILE)#指定输出文件路径
    args = parser.parse_args()

    #解析所有语料
    """
        {
          "id": "扫地机器人100问2-001-391d30789c",
          "file": "扫地机器人100问2",      // 来自哪个文件（去掉扩展名）
          "kind": "qa",                    // 类型：qa/fault/tip/knowledge
          "query": "首次使用扫地机器人需要做什么？",  // 用户问题
          "answer": "拆除机身所有包装配件…",        // 标准答案
          "variant": false                 // 是不是变体
        }
    """

    all_items = []
    for f in QA_FILES:
        #(DATA_DIR / f)：拼出文件的完整路径（比如 data/扫地机器人100问2.txt）
        all_items += parse_qa((DATA_DIR / f).read_text(encoding="utf-8"), Path(f).stem)
    all_items += parse_fault((DATA_DIR / FAULT_FILE).read_text(encoding="utf-8"), Path(FAULT_FILE).stem)

    for f, style in KNOWLEDGE_FILES.items():
        all_items += parse_knowledge((DATA_DIR / f).read_text(encoding="utf-8"), Path(f).stem, style)
    if args.include_tips:
        for f in TIP_FILES:
            all_items += parse_tip((DATA_DIR / f).read_text(encoding="utf-8"), Path(f).stem)
    if args.include_pdf:
        all_items += parse_pdf(DATA_DIR / PDF_FILE, Path(PDF_FILE).stem)

#第一次去重
    seen = set()#建一个空集合，
    deduped = []#去重后的结果列表
    for item in all_items:
        key = norm_text(item["query"])
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    # 变体增强：更难的同义/口语化 query，让召回评测不再是"原样问题"的简单复读
    augmented = deduped + _make_variant_items(deduped)
#第二次去重
    seen2 = set()
    final = []
    for item in augmented:
        key = norm_text(item["query"])
        if key and key not in seen2:
            seen2.add(key)
            final.append(item)

    #统计每种类型各多少条qa/fault/tip/knowledge
    by_kind = {k: sum(1 for i in final if i["kind"] == k) for k in ("qa", "fault", "tip", "knowledge")}
    #数一下变体条目有多少条（i.get("variant") 为 True 的）
    variants = sum(1 for i in final if i.get("variant"))
    payload = {#把“元信息 + 题目”组装成要写入文件的整体结构
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total": len(final),
            "by_kind": by_kind,
            "variants": variants,
        },
        "items": final,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评测集已生成: {args.out}")
    print(f"  总数 {len(final)} 条（原版 {len(final) - variants} + 变体 {variants}），分布 {by_kind}")


if __name__ == "__main__":
    main()
