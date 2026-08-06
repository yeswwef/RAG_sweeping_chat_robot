"""查询改写：检索前用 LLM 生成改写查询 + 子查询，多路召回提升 recall。"""
import json
import re

from langchain_core.prompts import ChatPromptTemplate

from model.factory import chat_model
from utils.config_handler import rag_config
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_rewrite_prompts


class QueryRewriter:
    """返回 [原始查询, 改写查询, 子查询...] 的查询列表；任何异常回退原始查询，保证可用。"""

    def __init__(self):
        cfg = rag_config["query_rewrite"]
        self.enabled = cfg["enabled"]
        self.max_queries = cfg["max_queries"]
        self.prompt = ChatPromptTemplate.from_template(load_rag_rewrite_prompts())
        self.chain = self.prompt | chat_model

    def rewrite(self, query: str) -> list[str]:
        if not self.enabled:
            return [query]
        try:
            resp = self.chain.invoke({"question": query, "num": self.max_queries - 1})
            content = resp.content if hasattr(resp, "content") else str(resp)
            queries = self._parse(content)
            result = [query]
            for q in queries:
                q = q.strip().strip('"').strip("'")
                if q and q != query and q not in result:
                    result.append(q)
            result = result[: self.max_queries]
            logger.info(f"[QueryRewriter] '{query}' → {result}")
            return result
        except Exception as e:
            logger.error(f"[QueryRewriter] 改写失败，回退原始查询: {e}")
            return [query]

    @staticmethod
    def _parse(content: str) -> list[str]:
        """容错解析 LLM 输出的 JSON 数组：整体解析 → 提取中括号 → 按行兜底。"""
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
        m = re.search(r"\[.*\]", content, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    return [str(x) for x in data]
            except json.JSONDecodeError:
                pass
        return [ln.strip() for ln in content.splitlines() if ln.strip()]

if __name__ == '__main__':
    # 测试1: 实例化（验证配置读取）
    rw = QueryRewriter()
    print("实例化 OK, enabled =", rw.enabled, ", max_queries =", rw.max_queries)

    # 测试2: 真实改写（需要 API Key）
    print("\n=== 真实改写测试 ===")
    queries = rw.rewrite("小户型适合哪些扫地机器人")
    print(f"改写结果 ({len(queries)} 条):")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    # 测试3: _parse 容错测试（不调 API，纯函数）
    print("\n=== _parse 容错测试 ===")
    print("干净JSON:", QueryRewriter._parse('["a", "b", "c"]'))
    print("带markdown:", QueryRewriter._parse('```json\n["x", "y"]\n```'))
    print("带废话:", QueryRewriter._parse('好的，结果如下：["m", "n"] 请查收'))
    print("纯文本按行:", QueryRewriter._parse("第一行\n第二行"))
    print("非法输入:", QueryRewriter._parse("not json at all"))

    # 测试4: enabled=False 时回退
    print("\n=== 开关回退测试 ===")
    rw.enabled = False
    print("关闭后:", rw.rewrite("随便问问"))
    rw.enabled = True