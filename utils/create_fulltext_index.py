# -*- coding: utf-8 -*-
"""
创建Neo4j全文索引，用于混合检索（Hybrid Search）。

运行方式：
    python utils/create_fulltext_index.py

这会在Neo4j中创建一个全文索引 textunit_fulltext_index，
索引 TextUnit 节点的 content 属性。
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase


def create_fulltext_index(uri: str, username: str, password: str, index_name: str = "textunit_fulltext_index"):
    """创建TextUnit的全文索引"""
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    with driver.session() as session:
        # 检查索引是否已存在
        result = session.run("SHOW INDEXES YIELD name WHERE name = $name RETURN name", {"name": index_name})
        if result.single():
            print(f"[INFO] 全文索引 '{index_name}' 已存在，跳过创建。")
            driver.close()
            return
        
        # 创建全文索引
        # 索引 TextUnit 的 content 和 name 属性
        # eventually_consistent: false = 同步更新，写入后立即可查
        cypher = f"""
        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
        FOR (n:TextUnit)
        ON EACH [n.content, n.name]
        OPTIONS {{
            indexConfig: {{
                `fulltext.analyzer`: 'standard-no-stop-words',
                `fulltext.eventually_consistent`: false
            }}
        }}
        """
        try:
            session.run(cypher)
            print(f"[SUCCESS] 全文索引 '{index_name}' 创建成功！")
        except Exception as e:
            print(f"[ERROR] 创建全文索引失败: {e}")
    
    driver.close()


def main():
    # 从环境变量读取Neo4j连接信息
    # 默认使用 LOCAL 数据库
    uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    print(f"[INFO] 连接到 Neo4j: {uri}")
    create_fulltext_index(uri, username, password)
    
    # 如果有 AURA 数据库，也创建索引
    aura_uri = os.getenv("NEO4J_AURA_URI")
    if aura_uri:
        aura_username = os.getenv("NEO4J_AURA_USERNAME", "neo4j")
        aura_password = os.getenv("NEO4J_AURA_PASSWORD", "")
        print(f"\n[INFO] 连接到 Neo4j AURA: {aura_uri}")
        create_fulltext_index(aura_uri, aura_username, aura_password)


if __name__ == "__main__":
    main()
