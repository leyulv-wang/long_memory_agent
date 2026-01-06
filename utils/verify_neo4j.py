# -*- coding: utf-8 -*-
import logging
from neo4j import GraphDatabase
import json
import sys
import os

# 尝试导入配置，确保在项目根目录下运行
try:
    from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
except ImportError:
    # 如果找不到 config，尝试添加路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# 设置日志，屏蔽 Neo4j 的干扰信息
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


def verify_full_system():
    print(f"\n🚀 开始 Neo4j 数据库全方位体检...")
    print(f"🔗 连接地址: {NEO4J_URI}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        # =========================================================
        # 1. 基础数据量检查
        # =========================================================
        print("\n📊 [1. 基础数据统计]")
        node_count = session.run("MATCH (n) RETURN count(n) as c").single()['c']
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()['c']
        print(f"   - 节点总数: {node_count}")
        print(f"   - 关系总数: {rel_count}")

        if node_count == 0:
            print("❌ 严重警告: 数据库为空！没有写入任何数据。")
            return

        # =========================================================
        # 2. 核心关系与内容质量检查 (重点！)
        # =========================================================
        print("\n🧠 [2. 关系内容抽样 (Triples)]")
        print("   (检查是否只有实体而没有关系，或者关系是否太泛泛)")

        # 排除 FROM_SOURCE 这种元数据关系，只看业务逻辑关系
        triples = session.run("""
            MATCH (a)-[r]->(b) 
            WHERE type(r) <> 'FROM_SOURCE' 
            RETURN a.name, type(r), b.name, r.confidence
            LIMIT 10
        """).data()

        if not triples:
            print("⚠️  警告: 找到了节点，但没有发现任何‘业务逻辑关系’！(只有孤立点或仅有元数据连接)")
        else:
            for t in triples:
                print(f"   - ({t['a.name']}) -[{t['type(r)']}]-> ({t['b.name']}) [conf={t['r.confidence']}]")

        # =========================================================
        # 3. LoCoMo 时间逻辑检查 (event_timestamp)
        # =========================================================
        print("\n⏳ [3. 时间与时序逻辑检查]")

        # 检查是否成功写入了 event_timestamp
        has_ts_count = session.run("""
            MATCH ()-[r]->() 
            WHERE r.event_timestamp IS NOT NULL 
            RETURN count(r) as c
        """).single()['c']

        print(f"   - 拥有 event_timestamp 的关系数: {has_ts_count} / {rel_count}")

        if has_ts_count == 0 and rel_count > 0:
            print("❌ 失败: 没有任何关系包含 'event_timestamp' 属性！排序逻辑将失效。")
        elif has_ts_count < rel_count:
            print(
                f"⚠️  注意: 只有部分关系 ({has_ts_count}/{rel_count}) 有时间戳，可能是 FROM_SOURCE 关系没加(正常)或部分丢失。")
        else:
            print("✅ 成功: 所有关系都覆盖了时间戳字段。")

        # 检查具体的时间值样例
        ts_samples = session.run("""
            MATCH ()-[r]->() 
            WHERE r.event_timestamp IS NOT NULL AND type(r) <> 'FROM_SOURCE'
            RETURN r.event_timestamp as ts, r.consolidated_at as rec_time 
            LIMIT 5
        """).data()
        print(f"   - 时间戳样例 (Event vs Record):")
        for s in ts_samples:
            print(f"     > 事件发生: {s['ts']} | 记录时间: {s['rec_time']}")

        # =========================================================
        # 4. 证据链 (GraphRAG) 检查
        # =========================================================
        print("\n🔗 [4. 证据链完整性检查]")

        text_units = session.run("MATCH (n:TextUnit) RETURN count(n) as c").single()['c']
        print(f"   - 证据块 (TextUnit) 数量: {text_units}")

        source_links = session.run("MATCH ()-[r:FROM_SOURCE]->() RETURN count(r) as c").single()['c']
        print(f"   - 溯源连接 (FROM_SOURCE) 数量: {source_links}")

        if text_units > 0 and source_links == 0:
            print("❌ 失败: 有证据块但没连接！知识图谱是“悬空”的，无法溯源。")
        elif text_units > 0:
            print("✅ 成功: 知识图谱已正确挂载到原始证据上。")

        # =========================================================
        # 5. 向量 (Embedding) 检查
        # =========================================================
        print("\n📐 [5. 向量 Embedding 状态]")
        # 检查是否有 embedding 属性，且长度大约是 1024/1536/768 等
        vector_check = session.run("""
            MATCH (n) 
            WHERE n.embedding IS NOT NULL 
            RETURN count(n) as c, size(n.embedding) as dim 
            LIMIT 1
        """).single()

        if not vector_check or vector_check['c'] == 0:
            print("❌ 失败: 节点没有写入 'embedding' 向量属性！向量检索将无法工作。")
        else:
            print(f"✅ 成功: 检测到向量数据 (维度: {vector_check['dim']})，共 {vector_check['c']} 个节点已向量化。")

        # =========================================================
        # 6. 特定测试点验证 (GPS / Honda Civic)
        # =========================================================
        print("\n🎯 [6. 关键测试点验证 (LoCoMo)]")

        keywords = ["GPS", "Honda", "Service", "Car"]
        for kw in keywords:
            res = session.run(f"""
                MATCH (n) 
                WHERE toLower(n.name) CONTAINS toLower('{kw}') 
                RETURN n.name, labels(n) LIMIT 3
            """).data()
            if res:
                print(f"   - 关键词 '{kw}' 命中: {[r['n.name'] for r in res]}")
            else:
                print(f"   - 关键词 '{kw}' 未找到 (可能是提取失败或同义词差异)。")

        # =========================================================
        # 7. 节点类型分布
        # =========================================================
        print("\n🏷️  [7. 节点类型分布]")
        labels_dist = session.run("""
            MATCH (n) 
            UNWIND labels(n) as l 
            RETURN l, count(*) as c 
            ORDER BY c DESC
        """).data()
        for row in labels_dist:
            print(f"   - {row['l']}: {row['c']}")

        # 特别检查 Value 和 Date
        val_count = next((item['c'] for item in labels_dist if item['l'] == 'Value'), 0)
        date_count = next((item['c'] for item in labels_dist if item['l'] == 'Date'), 0)

        if val_count == 0 and date_count == 0:
            print("⚠️  警告: 没有发现 'Value' 或 'Date' 类型的节点。这对于 LoCoMo 这种细节推理任务很不利。")

    driver.close()
    print("\n✅ 体检结束。")


if __name__ == "__main__":
    verify_full_system()