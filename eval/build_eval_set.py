# -*- coding: utf-8 -*-
"""评测集构造：从 data/ 语料解析 QA/故障/建议，生成 eval/eval_set.json。

语料结构：
- qa   ：`N. **问题**` + `- 答案`（100问 × 2，query 就是真实问题）
- fault：`N. 故障现象：…；检测：…；修复：…`（query 取故障现象）
- tip  ：`N. 陈述句`（维护保养/选购指南，启发式 query，默认不参与）

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

_HEADER_RE = re.compile(r"^\s*#")
_QA_RE = re.compile(r"^\s*(\d+)\.\s*\*\*(.+?)\*\*\s*$")
_ITEM_RE = re.compile(r"^\s*(\d+)\.\s*(.+)$")


def norm_text(s):
    return re.sub(r"\s+", "", s or "")


def _make_item(file_stem, idx, kind, query, answer):
    item_id = hashlib.md5(f"{file_stem}:{kind}:{idx}".encode("utf-8")).hexdigest()[:10]
    return {
        "id": f"{file_stem}-{idx:03d}-{item_id}",
        "file": file_stem,
        "kind": kind,
        "query": query,
        "answer": answer,
    }


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


def _extract_topic(body):
    for sep in ("：", ":", "，", ",", "；", "。", "？", "（", "(", " "):
        idx = body.find(sep)
        if idx > 0:
            topic = body[:idx].strip()
            if topic:
                return topic
    return body[:20].strip()


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


def parse_pdf(path, file_stem):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return parse_qa(text, file_stem)


def main():
    parser = argparse.ArgumentParser(description="构造评测集")
    parser.add_argument("--include-tips", action="store_true")
    parser.add_argument("--include-pdf", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_FILE)
    args = parser.parse_args()

    all_items = []
    for f in QA_FILES:
        all_items += parse_qa((DATA_DIR / f).read_text(encoding="utf-8"), Path(f).stem)
    all_items += parse_fault((DATA_DIR / FAULT_FILE).read_text(encoding="utf-8"), Path(FAULT_FILE).stem)
    if args.include_tips:
        for f in TIP_FILES:
            all_items += parse_tip((DATA_DIR / f).read_text(encoding="utf-8"), Path(f).stem)
    if args.include_pdf:
        all_items += parse_pdf(DATA_DIR / PDF_FILE, Path(PDF_FILE).stem)

    seen = set()
    deduped = []
    for item in all_items:
        key = norm_text(item["query"])
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    by_kind = {k: sum(1 for i in deduped if i["kind"] == k) for k in ("qa", "fault", "tip")}
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total": len(deduped),
            "by_kind": by_kind,
        },
        "items": deduped,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评测集已生成: {args.out}")
    print(f"  总数 {len(deduped)} 条，分布 {by_kind}")


if __name__ == "__main__":
    main()
