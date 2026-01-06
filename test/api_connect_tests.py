# 这是一个独立的脚本，用于验证项目所需的所有外部服务连接是否正常。
# 在开始开发或部署前运行此脚本，可以快速定位环境配置问题。

# 【修改】导入新的 test_embedding_connection 函数
from utils.connection_tests import test_llm_connection, test_neo4j_connection, test_embedding_connection


def run_all_tests():
    """
    执行所有连接测试并打印总结报告。
    """
    print("--- 开始环境连接测试 ---")

    llm_ok = test_llm_connection()
    # 【新增】调用 embedding 测试
    embedding_ok = test_embedding_connection()
    neo4j_ok = test_neo4j_connection()

    print("\n--- 测试总结 ---")

    if llm_ok is True:
        print("🟢 LLM (Chat) 服务: 已连接")
    elif llm_ok is False:
        print("🔴 LLM (Chat) 服务: 连接失败")
    else:
        print("🟡 LLM (Chat) 服务: 测试已跳过")

    # 【新增】打印 embedding 测试结果
    if embedding_ok is True:
        print("🟢 Embedding 服务: 已连接")
    elif embedding_ok is False:
        print("🔴 Embedding 服务: 连接失败")
    else:
        print("🟡 Embedding 服务: 测试已跳过")

    if neo4j_ok is True:
        print("🟢 Neo4j 数据库: 已连接")
    elif neo4j_ok is False:
        print("🔴 Neo4j 数据库: 连接失败")
    else:
        print("🟡 Neo4j 数据库: 测试已跳过")

    print("------------------")

    # 【新增】将 embedding_ok 添加到最终检查
    if llm_ok is not False and neo4j_ok is not False and embedding_ok is not False:
        print("\n🎉 所有必要的连接均已配置成功！")
    else:
        print("\n⚠️ 请根据上面的错误提示检查您的 .env 配置文件或服务状态。")


if __name__ == "__main__":
    run_all_tests()

