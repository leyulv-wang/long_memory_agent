# -*- coding: utf-8 -*-
"""
测试 ASSERTS 写入时 embedding 是否正确传入
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.stores import LongTermSemanticStore
from memory.ltss_writer import write_consolidation_result
from utils.embedding import get_embedding_model

# 创建一个简单的测试
ltss = LongTermSemanticStore(bootstrap_now=False)
emb = get_embedding_model()

# 模拟一个简单的 structured_response
class MockStructuredResponse:
    def __init__(self):
        self.nodes = []
        self.relationships = []
        self.facts = ['Test fact about embedding write']
        self.insights = []
        self.claims = []

# 测试写入
write_consolidation_result(
    ltss=ltss,
    embedding_model=emb,
    agent_name='TestAgent',
    memories_str='Test memory about embedding',
    structured_response=MockStructuredResponse(),
    current_time='TURN_999',
    channel='consolidated',
)

# 检查写入结果
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
with driver.session() as session:
    result = session.run('''
        MATCH (n:__Node__)
        WHERE n.name CONTAINS 'Test fact about embedding'
        RETURN n.name AS name, size(n.embedding) AS emb_size
    ''')
    print('=== 测试写入结果 ===')
    for r in result:
        print('name:', r['name'])
        print('  embedding size:', r['emb_size'])
driver.close()
ltss.close()
