# 这是一个独立的工具脚本，用于彻底清空智能体的长期记忆库 (Neo4j 数据库)。
# 功能：删除所有数据 + 删除所有约束 + 删除所有索引
# 运行方式: python utils/clear_long_term_memory.py

import sys
import os

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
from config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
)

NEO4J_AURA_URI = os.getenv("NEO4J_AURA_URI")
NEO4J_AURA_USERNAME = os.getenv("NEO4J_AURA_USERNAME", "neo4j")
NEO4J_AURA_PASSWORD = os.getenv("NEO4J_AURA_PASSWORD")


def clear_schema(driver):
    """
    暴力扫描并删除所有的约束和索引。
    """
    with driver.session() as session:
        # --- 1. 删除所有约束 (Constraints) ---
        print("正在扫描并删除所有约束...")
        constraints = []
        try:
            # 尝试 Neo4j 5.x 语法
            result = session.run("SHOW CONSTRAINTS YIELD name")
            constraints = [record["name"] for record in result]
        except Exception:
            try:
                # 回退到 Neo4j 4.x 语法
                result = session.run("CALL db.constraints() YIELD name")
                constraints = [record["name"] for record in result]
            except Exception as e:
                print(f"  警告: 无法获取约束列表 ({e})")

        for name in constraints:
            try:
                session.run(f"DROP CONSTRAINT {name}")
                print(f"  - 已删除约束: {name}")
            except Exception as e:
                print(f"  - 删除约束 {name} 失败: {e}")

        # --- 2. 删除所有索引 (Indexes) ---
        # 注意：保留全文索引和向量索引，只删除其他索引
        # 这些索引结构保留不影响数据清理，下次写入时自动更新
        print("正在扫描并删除索引（保留全文/向量索引）...")
        indexes = []
        preserved_indexes = {"textunit_fulltext_index", "textunit_vector_index"}  # 保留这些索引
        try:
            # 尝试 Neo4j 5.x 语法
            result = session.run("SHOW INDEXES YIELD name")
            indexes = [record["name"] for record in result]
        except Exception:
            try:
                # 回退到 Neo4j 4.x 语法
                result = session.run("CALL db.indexes() YIELD name")
                indexes = [record["name"] for record in result]
            except Exception as e:
                print(f"  警告: 无法获取索引列表 ({e})")

        for name in indexes:
            # 跳过未命名的或系统索引（通常 Drop 会报错，catch住即可）
            if not name:
                continue
            # 保留全文索引和向量索引
            if name in preserved_indexes:
                print(f"  - 保留索引: {name}")
                continue
            try:
                session.run(f"DROP INDEX {name}")
                print(f"  - 已删除索引: {name}")
            except Exception:
                pass  # 忽略无法删除的系统索引


def _clear_database(driver):
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def _targets():
    targets = []
    if NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD:
        targets.append(("local", NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD))
    if NEO4J_AURA_URI and NEO4J_AURA_USERNAME and NEO4J_AURA_PASSWORD:
        targets.append(("aura", NEO4J_AURA_URI, NEO4J_AURA_USERNAME, NEO4J_AURA_PASSWORD))
    return targets


def clear_ltss():
    """
    连接到长期记忆库并执行深度清理（本地 + Aura）。
    """
    print("\n--- Neo4j 深度清理工具 ---")
    print("警告：此操作将执行以下破坏性操作：")
    print("1. 删除所有节点和关系 (MATCH (n) DETACH DELETE n)")
    print("2. 删除所有数据库约束 (DROP CONSTRAINT ...)")
    print("3. 删除所有向量索引 (DROP INDEX ...)")

    targets = _targets()
    if not targets:
        print("\nNo Neo4j targets found in environment. Aborting.")
        return

    try:
        confirm = input("\nConfirm reset ALL configured databases? (type 'yes' to continue): ")

        if confirm.strip().lower() == "yes":
            for name, uri, user, pwd in targets:
                print(f"\n>>> Clearing target: {name} ({uri})")
                driver = GraphDatabase.driver(uri, auth=(user, pwd))
                try:
                    print("  - Step 1: clear graph data...")
                    _clear_database(driver)
                    print("  - Step 2: drop constraints/indexes...")
                    clear_schema(driver)
                finally:
                    driver.close()

            print("\nAll configured databases have been reset.")
        else:
            print("\n操作已取消。")

    except Exception as e:
        print(f"\n? 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n--- 结束 ---")


if __name__ == "__main__":
    clear_ltss()
