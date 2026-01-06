import os
import shutil
from memory.dual_memory_system import DualMemorySystem
from config import DATA_DIR


def cleanup_data_folder():
    """
    清理 data 文件夹，用于干净的测试。
    这个函数会清空 data 文件夹下的所有内容，但保留 data 文件夹本身。
    """
    if os.path.exists(DATA_DIR):
        print(f"为了进行干净的测试，正在清空 data 文件夹: {DATA_DIR}")
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'清理文件 {file_path} 时出错: {e}')
    else:
        print(f"data 文件夹不存在，正在创建: {DATA_DIR}")
        os.makedirs(DATA_DIR)


def test_memory_system():
    """
    一个独立的脚本，用于测试 DualMemorySystem 的各个组件。
    """
    cleanup_data_folder()

    print("\n--- 开始独立的记忆系统测试 ---")

    memory_system = None  # 初始化为 None 以便在 finally 中使用

    try:
        # 1. 创建一个 DualMemorySystem 实例
        memory_system = DualMemorySystem()

        # 2. 清理 Neo4j 数据库
        print("\n--- 准备 LTSS (Neo4j) 测试环境 ---")
        memory_system.ltss.clear_database()

        # 3. 测试 STES (短期情节记忆 - FAISS)
        print("\n--- 正在测试 STES (FAISS) ---")

        print("\n[步骤 3.1] 正在向 STES 添加三条记忆...")
        memories_to_add = [
            "Isabella 最喜欢的饮料是黑咖啡。",
            "Tom 是一个乐于助人的人。",
            "咖啡馆今天人很多。"
        ]
        for mem in memories_to_add:
            memory_system.add_episodic_memory(mem)

        print("\n[步骤 3.2] 正在从 STES 检索与 'Isabella' 相关的记忆...")
        query = "Isabella 喜欢喝什么饮料？"
        expected_memory = "Isabella 最喜欢的饮料是黑咖啡。"
        retrieved = memory_system.retrieve_episodic_memories(query, k=2)

        print(f"查询: '{query}'")
        print(f"预期最相关的记忆: '{expected_memory}'")
        print(f"实际检索结果: {retrieved}")

        if retrieved and retrieved[0] == expected_memory:
            print("✅ STES 测试成功: 成功检索到了最相关的记忆！")
        else:
            print("❌ STES 测试失败: 未能检索到预期的记忆作为首要结果。")

        # 4. 测试 LTSS (长期语义知识 - Neo4j)
        print("\n--- ---------------------------------- ---")
        print("\n--- 正在测试 LTSS (Neo4j) ---")

        print("\n[步骤 4.1] 正在向 LTSS 添加一个 'Person' 节点...")
        create_query = "CREATE (p:Person {name: 'Isabella', trait: 'loves black coffee'})"
        memory_system.ltss.update_graph(create_query)
        print("写入查询已执行。")

        print("\n[步骤 4.2] 正在从 LTSS 检索该节点以进行验证...")
        read_query = "MATCH (p:Person {name: 'Isabella'}) RETURN p.name AS name, p.trait AS trait"
        results = memory_system.ltss.query_graph(read_query)
        print(f"检索结果: {results}")

        if results and len(results) == 1 and results[0]['name'] == 'Isabella' and results[0][
            'trait'] == 'loves black coffee':
            print("✅ LTSS 测试成功: 成功写入并读回了节点！")
        else:
            print("❌ LTSS 测试失败: 未能正确读回写入的数据。")

    finally:
        # 5. 确保资源被释放
        if memory_system:
            memory_system.close()
        print("\n--- 记忆系统测试结束 ---")


if __name__ == "__main__":
    test_memory_system()

