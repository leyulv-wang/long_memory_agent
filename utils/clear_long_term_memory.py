# 这是一个独立的工具脚本，用于彻底清空智能体的长期记忆库 (Neo4j 数据库)。
# 功能：删除所有数据 + 删除所有约束 + 删除所有索引
# 运行方式: python utils/clear_long_term_memory.py

import sys
import os

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.stores import LongTermSemanticStore


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
        except:
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
        print("正在扫描并删除所有索引...")
        indexes = []
        try:
            # 尝试 Neo4j 5.x 语法
            result = session.run("SHOW INDEXES YIELD name")
            indexes = [record["name"] for record in result]
        except:
            try:
                # 回退到 Neo4j 4.x 语法
                result = session.run("CALL db.indexes() YIELD name")
                indexes = [record["name"] for record in result]
            except Exception as e:
                print(f"  警告: 无法获取索引列表 ({e})")

        for name in indexes:
            # 跳过未命名的或系统索引（通常 Drop 会报错，catch住即可）
            if not name: continue
            try:
                session.run(f"DROP INDEX {name}")
                print(f"  - 已删除索引: {name}")
            except Exception:
                pass  # 忽略无法删除的系统索引


def clear_ltss():
    """
    连接到长期记忆库并执行深度清理。
    """
    print("\n--- 🗑️ Neo4j 深度清理工具 🗑️ ---")
    print("警告：此操作将执行以下破坏性操作：")
    print("1. 删除所有节点和关系 (MATCH (n) DETACH DELETE n)")
    print("2. 删除所有数据库约束 (DROP CONSTRAINT ...)")
    print("3. 删除所有向量索引 (DROP INDEX ...)")

    ltss = None
    try:
        # bootstrap_now=False 防止它尝试自动建立新索引，我们现在只想删东西
        ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=False)

        confirm = input("\n🔴 你确定要彻底重置数据库吗？ (输入 'yes' 继续): ")

        if confirm.strip().lower() == 'yes':
            print("\n>>> 第一步：清空图数据...")
            ltss.clear_database()

            print("\n>>> 第二步：清空架构 (约束与索引)...")
            if ltss.driver:
                clear_schema(ltss.driver)

            print("\n✅ 数据库已彻底重置为白板状态。")
            print("现在您可以运行 `utils/bootstrap_world_knowledge.py` 来应用新的 ID 结构了。")
        else:
            print("\n操作已取消。")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if ltss:
            ltss.close()
        print("\n--- 结束 ---")


if __name__ == "__main__":
    clear_ltss()