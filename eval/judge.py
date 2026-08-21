# -*- coding: utf-8 -*-
"""LLM-as-Judge：用独立裁判模型给生成质量打分。

裁判默认 qwen-plus，与被测生成模型（qwen3-max）分离，减少自评偏差。
所有 judge_* 函数返回已解析的 dict，解析失败时返回兜底值。
"""
import json
import re

JUDGE_SYSTEM = "你是严格的 RAG 回答质量评测裁判，只输出 JSON，不要输出任何其他内容。"

FAITHFULNESS_PROMPT = """请判断"模型回答"是否忠实于"参考资料"，只输出 JSON：
{{"faithfulness": 1, "hallucination": false, "reason": "一句话理由"}}

评分标准（faithfulness 忠实度，1-5 整数）：
5 = 回答中每一条事实都能在参考资料中找到依据，无编造、无与资料矛盾；
3 = 大部分有依据，但存在少量超出资料的扩展或轻微不一致；
1 = 大量内容与资料无关，或直接与资料矛盾。
hallucination = true 表示回答中包含参考资料里不存在的关键事实（编造）。

参考资料（[] 内为编号）：
{context}

用户问题：{question}
模型回答：{answer}
"""

ANSWER_RELEVANCE_PROMPT = """请判断"模型回答"与"用户问题"的相关性，只输出 JSON：
{{"relevance": 1, "reason": "一句话理由"}}

评分标准（relevance 相关性，1-5 整数）：
5 = 直接、完整地回应问题核心，不跑题、不避答；
3 = 部分相关，或夹杂不相关内容；
1 = 基本没有回答用户问题、答非所问。

用户问题：{question}
模型回答：{answer}
"""

COMPLETENESS_PROMPT = """请判断"模型回答"是否覆盖"参考标准答案"的要点，只输出 JSON：
{{"completeness": 1, "reason": "一句话理由"}}

评分标准（completeness 完整度，1-5 整数）：
5 = 覆盖参考答案的全部核心要点；
3 = 覆盖部分核心要点，但有遗漏；
1 = 几乎未覆盖参考答案要点。

用户问题：{question}
参考标准答案：{reference}
模型回答：{answer}
"""


def parse_json(content, fallback):
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return fallback


def call_judge(prompt, judge_model):
    from dashscope import Generation

    resp = Generation.call(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        result_format="message",
        temperature=0.0,
    )
    return resp.output.choices[0].message.content


def judge_faithfulness(question, context, answer, judge_model):
    content = call_judge(
        FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer),
        judge_model,
    )
    return parse_json(
        content,
        {"faithfulness": 0, "hallucination": True, "reason": content[:100]},
    )


def judge_answer_relevance(question, answer, judge_model):
    content = call_judge(
        ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer),
        judge_model,
    )
    return parse_json(content, {"relevance": 0, "reason": content[:100]})


def judge_completeness(question, reference, answer, judge_model):
    content = call_judge(
        COMPLETENESS_PROMPT.format(question=question, reference=reference, answer=answer),
        judge_model,
    )
    return parse_json(content, {"completeness": 0, "reason": content[:100]})
