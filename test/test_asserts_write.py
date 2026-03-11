# -*- coding: utf-8 -*-
"""
测试 ASSERTS 关系写入
"""

import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from neo4j import GraphDatabase
from memory.cypher_templates import CYPHER_ASSERTS_EDGE
import json


def test_asserts_write():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    # 准备测试数据
    batch = [
        {
            "s_name": "test_unit_1",
            "s_label": "TextFact",
            "t_name": "User likes coffee",
            "t_label": "Fact",
            "type": "ASSERTS",
            "props": {
                "confidence": 0.9,
                "knowledge_type": "observed_fact",
                "source_of_belief": "user",
                "consolidated_at": "2024-01-01T00:00:00",
                "virtual_time": "TURN_1",
                "turn_id": 1,
                "event_step": 1,
                "event_timestamp": "unknown",
                "evidence_source_unit": "test_unit_1",
                "agent_name": "TestAgent",
                "channel": "consolidated",
                "is_current": True,
                "should_be_current": True,
                "source_rank": 1.0,
                "slot_key": "test_unit_1:ASSERTS:User likes coffee",
                "event_id": "evt:test:1",
                "belief_key": "1:consolidated:TestAgent:TextFact:test_unit_1:ASSERTS:Fact:User likes coffee",
            },
            "embedding": [0.1] * 1024,  # 简化的 embedding
        }
    ]
    
    with driver.session() as session:
        # 先创建 TextUnit
        session.run("""
            MERGE (u:TextUnit {name: 'test_unit_1'})
            SET u.content = 'Test content',
                u.turn_id = 1,
                u.channel = 'consolidated'
        """)
        print("✅ TextUnit 创建成功")
        
        # 执行 ASSERTS 写入
        try:
            result = session.run(CYPHER_ASSERTS_EDGE, {"batch": batch})
            summary = result.consume()
            print(f"✅ ASSERTS 写入成功: {summary.counters}")
        except Exception as e:
            print(f"❌ ASSERTS 写入失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 检查结果
        print("\n检查写入结果:")
        
        # 检查 __Node__
        result = session.run("MATCH (n:__Node__) RETURN count(*) AS count")
        count = result.single()["count"]
        print(f"  __Node__ 节点: {count}")
        
        # 检查 ASSERTS 关系
        result = session.run("MATCH ()-[r:ASSERTS]->() RETURN count(*) AS count")
        count = result.single()["count"]
        print(f"  ASSERTS 关系: {count}")
        
        # 检查 Fact 节点
        result = session.run("MATCH (f:Fact) RETURN count(*) AS count")
        count = result.single()["count"]
        print(f"  Fact 节点: {count}")
        
        # 清理测试数据
        session.run("MATCH (n) WHERE n.name STARTS WITH 'test_' DETACH DELETE n")
        session.run("MATCH (n:__Node__) WHERE n.name = 'User likes coffee' DETACH DELETE n")
        session.run("MATCH (f:Fact) WHERE f.agent_name = 'TestAgent' DETACH DELETE f")
        print("\n✅ 测试数据已清理")
    
    driver.close()


if __name__ == "__main__":
    test_asserts_write()
