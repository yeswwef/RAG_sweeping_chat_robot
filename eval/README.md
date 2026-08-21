# 评测体系（eval）

目标：把 RAG 链路拆成「检索 → 生成」两个阶段分别量化，并用可复现的黄金集做锚点。
检索阶段评「检索精确率 / 召回 / 排序」；生成阶段评「忠实度 / 答案相关性 / 完整度」；
黄金集自身有独立的离线质量校验。

## 目录结构

```
eval/
  build_eval_set.py        # 从 data/ 语料解析出 eval_set.json（query/answer/kind）
  metrics.py               # 纯函数指标层：hit/recall/mrr/ndcg/context_precision
  gold.py                  # 黄金集派生与加载（答案 -> 相关 chunk 映射）
  judge.py                 # LLM-as-Judge（faithfulness / relevance / completeness）
  validate_gold.py         # 黄金集质量校验（离线，不调 API）
  run_retrieval_eval.py    # 检索指标（四基线 + context precision）
  run_generation_eval.py   # 生成质量（faithfulness / relevance / completeness）
  run_latency_eval.py      # 分阶段耗时 / token / 成本
  results/                 # 结果输出（运行后自动生成）
```

## 快速开始

```powershell
$env:NO_PROXY = "dashscope.aliyuncs.com"          # 若开着代理

python eval/build_eval_set.py                       # 1) 构造评测集（离线）
python eval/validate_gold.py                        # 2) 校验黄金集质量（离线）
python eval/run_retrieval_eval.py --sample 20       # 3) 检索指标（含 context precision）
python eval/run_generation_eval.py --limit 3        # 4) 生成质量（LLM 裁判）
python eval/run_latency_eval.py --limit 5           # 5) 耗时/成本
```

## 评测对象与口径

### 检索阶段（run_retrieval_eval.py）

四条基线：纯稠密 / 纯 BM25 / 混合 RRF / 混合+改写。指标：

| 指标 | 含义 |
|---|---|
| hit@k | top-k 是否命中至少一个相关 chunk |
| recall@k | top-k 命中相关 chunk 数 / 相关 chunk 总数 |
| mrr@k | 首个相关 chunk 排名倒数的均值 |
| ndcg@k | 二值相关性的折损累计增益 |
| context_precision@k | RAGAS 语境精确率（Average Precision），衡量相关 chunk 是否排前、无关 chunk 是否稀释上下文 |

### 生成阶段（run_generation_eval.py）

复用生产 RAG 服务做「改写→检索→拼接→生成」，并把**真实检索上下文**交给裁判：

| 指标 | 判据基准 |
|---|---|
| faithfulness 忠实度 | 回答 vs 检索到的参考资料 |
| relevance 相关性 | 回答 vs 用户问题 |
| completeness 完整度 | 回答 vs 标准答案 |

### 黄金集（gold.py + validate_gold.py）

- gold 派生口径：`eval_set.json` 中每条 answer 即"应被检索到的资料原文"，用答案最长前缀在语料分块中的出现映射为相关 chunk（chunk 正文 md5 作键，与生产 `_doc_key` 一致）。
- validate_gold.py 输出覆盖率、空 gold 明细、query/answer 重复率、平均相关 chunk 数，并可 `--save-gold` 落盘供人工复核。

## 已知边界

- 自动派生 gold 对"答案与语料原文高度同构"的 QA/故障语料有效；语义等价改写可能漏判，空 gold 率过高时应补充人工标注子集。
- 检索/生成脚本依赖向量库已灌入语料；若语料为空，validate_gold 会显式告警。
- 生成质量评测在组件级（RAG 服务）进行，便于精确拿到上下文；agent 工具调用/查询选择另属端到端行为，不在本评测范围。
