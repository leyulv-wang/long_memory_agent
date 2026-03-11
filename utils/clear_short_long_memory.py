import sys
import os
import shutil

# 确保能导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.stores import LongTermSemanticStore
from config import AGENTS_DATA_DIR


def clear_short_term_memory(target_agent=None):
    """
    清理短期记忆 (FAISS 索引文件)。

    :param target_agent: 如果指定了名字，只删该角色的记忆；否则删除所有。
    """
    print(f"\n>>> 正在清理短期记忆 (FAISS 数据)...")
    print(f"    数据目录: {AGENTS_DATA_DIR}")

    if not os.path.exists(AGENTS_DATA_DIR):
        print("    [提示] 短期记忆目录不存在，无需清理。")
        return

    # 遍历 agents 目录
    agents_list = os.listdir(AGENTS_DATA_DIR)

    for agent_dir in agents_list:
        # 如果指定了目标，且当前文件夹不是目标，则跳过
        if target_agent and agent_dir != target_agent:
            continue

        full_path = os.path.join(AGENTS_DATA_DIR, agent_dir)

        if os.path.isdir(full_path):
            try:
                # 删除整个角色目录 (包含 memory.faiss 和 memory.faiss.pkl)
                shutil.rmtree(full_path)
                print(f"    - 已删除角色目录: {agent_dir}")
            except Exception as e:
                print(f"    x 删除 {agent_dir} 失败: {e}")

def fast_clear_neo4j_data(ltss_instance):
    """
    【新增】极速清理 Neo4j 数据，但保留索引和约束 (Constraints)。
    直接复用传入的 ltss_instance 连接。
    """
    print(">>> 正在执行 Neo4j 极速清理 (DETACH DELETE)...")
    try:
        with ltss_instance.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("    - 所有节点与关系已清空，索引已保留。")
    except Exception as e:
        print(f"    x 极速清理失败: {e}")

def clear_long_term_memory():
    """
    调用 Neo4j 清理逻辑 (复用 clear_long_term_memory.py 的核心逻辑)
    """
    print("\n>>> 正在清理长期记忆 (Neo4j 数据库)...")
    try:
        # 1. 连接数据库
        ltss = LongTermSemanticStore(bootstrap_now=False)

        # 2. 执行清理
        ltss.clear_database()  # 清除节点和关系

        # 3. 清除 Schema (索引和约束) - 这里我们需要手动执行，因为 ltss.clear_database 通常只删数据
        # 参考 clear_long_term_memory.py 的逻辑
        with ltss.driver.session() as session:
            # 删除约束
            try:
                constraints = session.run("SHOW CONSTRAINTS YIELD name")
                for record in constraints:
                    name = record["name"]
                    session.run(f"DROP CONSTRAINT {name}")
                    print(f"    - 已删除约束: {name}")
            except Exception as e:
                print(f"    [跳过] 获取/删除约束时遇到问题 (可能是空数据库): {e}")

            # 删除索引
            try:
                indexes = session.run("SHOW INDEXES YIELD name, type")
                for record in indexes:
                    name = record["name"]
                    type_ = record["type"]
                    if type_ == "LOOKUP": continue  # 系统索引不能删
                    if name:
                        try:
                            session.run(f"DROP INDEX {name}")
                            print(f"    - 已删除索引: {name}")
                        except:
                            pass
            except Exception as e:
                print(f"    [跳过] 获取/删除索引时遇到问题: {e}")

        print("    长期记忆清理完成。")
        ltss.close()

    except Exception as e:
        print(f"    x 长期记忆清理失败: {e}")


def main():
    print("=== 🧹 智能体记忆深度清理工具 (长期 + 短期) ===")
    print("警告: 此操作将不可逆地删除以下数据:")
    print("1. Neo4j 中的所有知识图谱数据")
    print("2. data/agents/ 目录下的所有 FAISS 向量索引文件")

    confirm = input("\n请输入 'yes' 确认全部清除: ").strip().lower()

    if confirm == 'yes':
        # 1. 清理长期
        clear_long_term_memory()

        # 2. 清理短期
        clear_short_term_memory()

        print("\n✅ 所有记忆已重置完毕。下次运行时将重新初始化。")
    else:
        print("\n操作已取消。")


if __name__ == "__main__":
    main()