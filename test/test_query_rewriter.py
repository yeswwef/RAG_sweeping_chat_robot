# -*- coding: utf-8 -*-
"""改造2 QueryRewriter 测试脚本"""
import sys
sys.path.insert(0, "..")

from rag.query_rewriter import QueryRewriter

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
