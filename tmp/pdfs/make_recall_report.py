# -*- coding: utf-8 -*-
"""生成《召回率评测报告》PDF：评测结果 + 面试问答 + 优化方向。"""
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
FONT = "STSong-Light"

OUT = Path("output/pdf/扫地机器人RAG项目-召回率评测报告.pdf")


def s(name, size, leading, color=colors.black, bold=False, align=0, space_after=6):
    return ParagraphStyle(
        name=name,
        fontName=FONT,
        fontSize=size,
        leading=leading,
        textColor=color,
        alignment=align,
        spaceAfter=space_after,
        wordWrap="CJK",
    )


ST_TITLE = s("title", 20, 26, color=colors.HexColor("#1f3864"), align=1, space_after=4)
ST_SUB = s("sub", 11, 16, color=colors.HexColor("#555555"), align=1, space_after=12)
ST_H1 = s("h1", 15, 20, color=colors.HexColor("#1f3864"), space_after=8)
ST_H2 = s("h2", 12.5, 17, color=colors.HexColor("#2e5395"), space_after=6)
ST_BODY = s("body", 10.5, 16, space_after=6)
ST_BULLET = s("bullet", 10.5, 16, space_after=4, align=0)
ST_SMALL = s("small", 9, 13, color=colors.HexColor("#666666"), space_after=4)
ST_CELL = ParagraphStyle(
    "cell",
    fontName=FONT,
    fontSize=9.5,
    leading=14,
    wordWrap="CJK",
)


def H1(t):
    return Paragraph(t, ST_H1)


def H2(t):
    return Paragraph(t, ST_H2)


def P(t):
    return Paragraph(t, ST_BODY)


def B(t):
    return Paragraph(t, ST_BULLET)


def style_table(table, header_bg=colors.HexColor("#1f3864")):
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f8")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawCentredString(A4[0] / 2, 10 * mm, f"扫地机器人 RAG 项目  |  召回率评测报告  |  第 {doc.page} 页")
    canvas.restoreState()


def build_story():
    story = []
    story.append(Paragraph("扫地机器人 RAG 项目 召回率评测报告", ST_TITLE))
    story.append(Paragraph(
        f"评测日期：2026-08-28  |  数据来源：eval_set.json（2474 条） + retrieval_20260828_144102.json（抽样 100 条）",
        ST_SUB,
    ))

    # ============ 一、项目与评测背景 ============
    story.append(H1("一、项目与评测背景"))
    story.append(P(
        "本项目是“扫地/扫拖一体机器人智能客服 RAG 系统”：用户问题先经查询改写，再走混合检索"
        "（稠密向量 Chroma + BM25 稀疏 + RRF 融合），最后把检索结果拼进提示词交给大模型生成回答。"
        "召回率决定“资料是否捞全”，是 RAG 链路的核心指标，直接决定生成质量的上界。"
    ))
    story.append(P(
        "评测集由 build_eval_set.py 自动构建：共 2474 条（原版 1237 + 同义变体 1237），"
        "覆盖 qa 问答 400、fault 故障 400、knowledge 知识 1674 三类，每类都包含原版与“换说法”变体，"
        "用于检验检索对同义改写是否敏感。本次评测抽样 100 条。"
    ))

    # ============ 二、评测方法 ============
    story.append(H1("二、评测方法"))
    story.append(H2("2.1 四组检索策略（考生）"))
    story.append(B("dense：纯稠密，query 向量化后在 Chroma 里按语义相似度取 Top-20。"))
    story.append(B("bm25：纯稀疏，jieba 分词 + BM25 关键词打分取 Top-20。"))
    story.append(B("hybrid：稠密 + 稀疏两路召回，用 RRF 公式 1/(k+rank) 融合排序。"))
    story.append(B("hybrid_rewrite：先由 LLM 把问题改写成多条查询，再走混合检索。"))
    story.append(Spacer(1, 4))

    story.append(H2("2.2 黄金标准（gold）"))
    story.append(P(
        "对每条评测题，用标准答案（answer）去向量库 866 个分块里匹配：先按句拆段逐段找，"
        "找不到再按最长前缀回退；命中分块的正文 md5 组成该题的相关分块集合 gold。"
        "检索返回的分块也转成 md5 列表，与 gold 比对即可算分。"
    ))

    story.append(H2("2.3 指标口径"))
    metrics_data = [
        ["指标", "含义"],
        ["recall@5 / recall_full", "召回率：前5名 / 整批候选里捞回的相关分块比例"],
        ["hit@5", "前5名是否出现正确答案（0/1）"],
        ["mrr@10", "第一个正确答案的排名倒数均值"],
        ["ndcg@10", "整体排序质量"],
        ["context_precision@5", "前5名中相关内容的占比（精确率）"],
        ["avg_ms", "平均检索耗时"],
    ]
    story.append(style_table(Table(metrics_data, colWidths=[55 * mm, 115 * mm])))
    story.append(Spacer(1, 6))

    # ============ 三、评测结果 ============
    story.append(H1("三、评测结果"))
    story.append(H2("3.1 总体指标（100 条抽样）"))
    overall = [
        ["策略", "hit@5", "recall@5", "mrr@10", "ndcg@10", "cp@5", "recall_full", "avg_ms"],
        ["dense", "0.63", "0.63", "0.5496", "1.1108*", "0.5422", "0.80", "591.9"],
        ["bm25", "0.71", "0.71", "0.6422", "1.2079*", "0.6484", "0.86", "3.4"],
        ["hybrid", "0.82", "0.82", "0.5911", "0.67", "0.5752", "0.94", "635.4"],
        ["hybrid_rewrite", "0.86", "0.86", "0.6438", "0.7127", "0.6347", "0.95", "3765.6"],
    ]
    story.append(style_table(Table(overall, colWidths=[32 * mm] + [19 * mm] * 7)))
    story.append(Paragraph("* ndcg@10 出现大于 1 的异常值：语料中存在内容完全相同的重复分块，"
                           "dense/bm25 策略未去重导致 NDCG 被重复计数，该列暂不可采信。", ST_SMALL))
    story.append(Spacer(1, 6))

    story.append(H2("3.2 按 类型/原版变体 分组（recall@5 / recall_full / cp@5）"))
    group = [
        ["策略", "分组", "recall@5", "recall_full", "cp@5"],
        ["dense", "qa/original", "0.80", "0.90", "0.495"],
        ["dense", "fault/original", "0.60", "1.00", "0.600"],
        ["dense", "knowledge/original", "0.55", "0.725", "0.482"],
        ["dense", "qa/variant", "0.778", "0.889", "0.633"],
        ["dense", "fault/variant", "0.50", "1.00", "0.500"],
        ["dense", "knowledge/variant", "0.656", "0.781", "0.603"],
        ["bm25", "qa/original", "0.90", "1.00", "0.800"],
        ["bm25", "fault/original", "1.00", "1.00", "1.000"],
        ["bm25", "knowledge/original", "0.65", "0.825", "0.597"],
        ["bm25", "qa/variant", "0.778", "0.889", "0.778"],
        ["bm25", "fault/variant", "0.75", "0.75", "0.563"],
        ["bm25", "knowledge/variant", "0.656", "0.844", "0.585"],
        ["hybrid", "qa/original", "0.70", "1.00", "0.160"],
        ["hybrid", "fault/original", "0.60", "1.00", "0.120"],
        ["hybrid", "knowledge/original", "0.90", "0.90", "0.733"],
        ["hybrid", "qa/variant", "0.444", "1.00", "0.115"],
        ["hybrid", "fault/variant", "0.50", "1.00", "0.113"],
        ["hybrid", "knowledge/variant", "0.938", "0.938", "0.766"],
        ["hybrid_rewrite", "qa/original", "0.80", "1.00", "0.270"],
        ["hybrid_rewrite", "fault/original", "0.80", "1.00", "0.250"],
        ["hybrid_rewrite", "knowledge/original", "0.875", "0.925", "0.751"],
        ["hybrid_rewrite", "qa/variant", "0.778", "1.00", "0.278"],
        ["hybrid_rewrite", "fault/variant", "0.75", "1.00", "0.175"],
        ["hybrid_rewrite", "knowledge/variant", "0.906", "0.938", "0.821"],
    ]
    story.append(style_table(Table(group, colWidths=[32 * mm, 40 * mm] + [30 * mm] * 3)))
    story.append(Spacer(1, 4))
    story.append(P(
        "样本分布：knowledge 组 72 条（原版 40 + 变体 32）、qa 组 19 条（10+9）、fault 组仅 9 条（5+4）。"
        "fault 组样本过小，其数字（如 0.5、0.6）基本是噪声，不建议单独下结论。"
    ))

    # ============ 四、结果分析 ============
    story.append(H1("四、结果分析"))
    story.append(H2("4.1 混合检索是否更好"))
    story.append(P(
        "结论：在这套更大、更难（含变体与知识类）的评测集上，混合检索确实更好。"
        "hybrid_rewrite 的 recall@5 0.86、recall_full 0.95 全面优于单路 dense（0.63/0.80）与 bm25（0.71/0.86）。"
    ))
    story.append(P(
        "但提升主要来自 knowledge 类问题：hybrid 在 knowledge 组 recall@5 达 0.90~0.94、cp@5 达 0.73~0.77；"
        "而 qa/fault 短问答组，hybrid 的 cp@5 仍只有 0.11~0.27，说明融合对短问答仍会把正确分块稀释到后排。"
    ))
    story.append(H2("4.2 召回率水平解读"))
    story.append(P(
        "recall_full 0.94~0.95 说明答案基本都被捞到（整批候选内）；recall@5 0.82~0.86 说明前 5 名仍有提升空间。"
        "二者落差意味着下一步应聚焦“重排”（rerank），把捞到的正确答案提到前排。"
    ))
    story.append(H2("4.3 需要修的问题"))
    story.append(B("NDCG>1 异常：语料含重复分块，dense/bm25 未去重导致指标失真，需在策略层去重或修指标。"))
    story.append(B("fault 组样本量太小（4~5 条），结论置信度低，需加大 fault 抽样。"))
    story.append(B("变体组普遍低于原版组，说明检索对同义改写敏感，这是真实短板。"))

    # ============ 五、面试问答 ============
    story.append(H1("五、面试问答（召回率相关）"))
    qa = [
        ["问题", "参考回答要点"],
        ["Q1 什么是召回率？和精确率有什么区别？",
         "召回率 = 检索结果中命中的相关分块数 / 全部相关分块数，衡量捞得全不全；"
         "精确率 = 命中数 / 检索结果总数，衡量捞得准不准。两者是一对跷跷板。"],
        ["Q2 你项目里召回率怎么算的？",
         "先用标准答案在语料分块里匹配生成 gold（相关分块 md5 集合），再对四种策略算 Recall@5 和整批 recall_full。"
         "评测集自动构建 2400+ 条，分 qa/fault/knowledge 三类，并含同义变体。"],
        ["Q3 你的召回率多少？怎么解读？",
         "融合+改写策略 recall@5 约 0.86、recall_full 约 0.95。recall_full 高说明捞得全，"
         "recall@5 说明前 5 名还有空间，下一步做重排。"],
        ["Q4 为什么用混合检索？RRF 是什么？",
         "稠密抓语义、BM25 抓关键词，互补。RRF 用 1/(k+rank) 融合两路排名。"
         "实测发现融合在短问答上会把正确分块稀释到后排（cp@5 掉到 0.1~0.2），所以评测里专门用 context_precision 盯这个问题。"],
        ["Q5 怎么提升召回率？",
         "四个方向：查询改写加门控、调 RRF 参数（rrf_k/top_n）并给稀疏路加权、优化分块（chunk_size/overlap）、"
         "检索后加重排（cross-encoder）。"],
        ["Q6 gold 怎么来的？有什么坑？",
         "用标准答案在语料分块里做拆段/最长前缀匹配，命中分块的 md5 就是 gold。"
         "坑是自动派生对语义等价但字面不同的答案会漏判，需人工标注权威子集校准。"],
        ["Q7 召回率和生成质量什么关系？",
         "召回是生成的上界：检索漏了，大模型再强也答不出缺失信息。所以 RAG 先保召回，再用精确率和排序控制噪声。"],
    ]
    qa_table = Table(
        [[Paragraph(c, ST_CELL) for c in row] for row in qa],
        colWidths=[42 * mm, 128 * mm],
    )
    qa_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f8")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(qa_table)

    # ============ 六、优化方向 ============
    story.append(H1("六、召回率优化方向"))
    story.append(B("1. 修复评测可信度：去除语料重复分块（修 NDCG>1），加大 fault 组样本，先让数字可信再谈优化。"))
    story.append(B("2. 重排（rerank）：recall_full 0.95 vs recall@5 0.86 的落差，正好用 cross-encoder 把正确答案提到前排。"))
    story.append(B("3. 查询改写门控：只对口语化/指代/长尾问题改写，避免把“机器人”改成“计算机”这类噪声。"))
    story.append(B("4. RRF 参数与加权：调小 rrf_k、减小 top_n、给更强的 BM25 路加权，缓解短问答前排稀释。"))
    story.append(B("5. 分块优化：加大 chunk_overlap 或按语义切块，减少答案跨块漏召回。"))
    story.append(B("6. 查询扩展：利用项目已有的术语词典.txt 构建同义词扩展表，直接提升变体组召回。"))
    story.append(B("7. 按类型自适应策略：knowledge 类 hybrid 明显更好，短问答可优先 BM25，做策略分流。"))

    # ============ 七、附录 ============
    story.append(H1("七、附录：数据来源与复现命令"))
    story.append(P("评测集：eval/eval_set.json（build_eval_set.py 生成，2474 条）"))
    story.append(P("结果文件：eval/results/retrieval_20260828_144102.json（run_retrieval_eval.py --sample 100）"))
    story.append(P("复现：python eval/build_eval_set.py ; python eval/validate_gold.py ; python eval/run_retrieval_eval.py --sample 100"))
    return story


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="扫地机器人 RAG 项目 召回率评测报告",
        author="eval 评测体系",
    )
    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print("PDF 已生成:", OUT.resolve())


if __name__ == "__main__":
    main()
