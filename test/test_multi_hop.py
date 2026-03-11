# -*- coding: utf-8 -*-
"""专门测试多跳扩展功能（使用 SimpleRetriever）"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DEBUG_PIPELINE"] = "1"

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from memory.stores import LongTermSemanticStore
from agent.simple_retriever import SimpleRetriever

print("=" * 60)
print("多跳扩展功能测试（SimpleRetriever）")
print("=" * 60)

# 初始化 LTSS
print("\n正在初始化 LTSS...")
ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=False)

# 测试1：检索包含多跳关系的问题
print("\n" + "=" * 60)
print("测试1：检索 Dr. Patel 相关信息")
print("=" * 60)

retriever = SimpleRetriever(ltss, agent_name="LongMemory")

query1 = "Where does Dr. Patel work?"
print(f"\n查询: {query1}")
result1 = retriever.search(
    query1,
    simple_fact_k=30,
    textunit_k=10,
    enable_multi_hop=True,
)
print(f"\n结果:\n{result1}")

# 测试2：检索博物馆访问记录
print("\n" + "=" * 60)
print("测试2：检索博物馆访问记录")
print("=" * 60)

query2 = "Which museums have I visited?"
print(f"\n查询: {query2}")
result2 = retriever.search(
    query2,
    simple_fact_k=40,
    textunit_k=15,
    enable_multi_hop=True,
)
print(f"\n结果:\n{result2}")

# 测试3：检索过去状态
print("\n" + "=" * 60)
print("测试3：检索过去状态（版本检测）")
print("=" * 60)

query3 = "What was my previous stance on spirituality?"
print(f"\n查询: {query3}")
result3 = retriever.search(
    query3,
    simple_fact_k=30,
    textunit_k=10,
    enable_multi_hop=True,
    enable_version_detection=True,
)
print(f"\n结果:\n{result3}")

ltss.close()
print("\n测试完成。")
