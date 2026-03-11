# -*- coding: utf-8 -*-
"""测试检索逻辑（使用 SimpleRetriever）"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 启用调试模式
os.environ["DEBUG_GRAPHRAG"] = "1"
os.environ["DEBUG_PIPELINE"] = "1"

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from memory.stores import LongTermSemanticStore
from agent.simple_retriever import SimpleRetriever

# 初始化 LTSS
ltss = LongTermSemanticStore(bootstrap_now=False)

# 测试查询
query = "What was my previous stance on spirituality?"
print(f"\n=== 测试查询: {query} ===\n")

# 使用 SimpleRetriever
retriever = SimpleRetriever(ltss, agent_name="LongMemory")
result = retriever.search(
    query,
    simple_fact_k=40,
    textunit_k=10,
    enable_multi_hop=True,
    enable_version_detection=True,
    current_turn=2,
)

print(f"\n=== 检索结果 ===\n")
print(result)

ltss.close()
print("\n测试完成。")
