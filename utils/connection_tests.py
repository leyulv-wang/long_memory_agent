import time
# 导入 LLM 类 (用于 Chat 测试)
from langchain_openai import ChatOpenAI
# 导入 Neo4j 驱动 (用于数据库测试)
from neo4j import GraphDatabase, basic_auth

# 导入配置
from config import (
    GRAPHRAG_API_BASE,
    GRAPHRAG_CHAT_API_KEY,
    GRAPHRAG_CHAT_MODEL,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD
)

# 【修改】直接导入我们在 utils/embedding.py 中写好的工厂函数
from utils.embedding import get_embedding_model


def test_llm_connection():
    """
    测试与大语言模型 (Chat) 的连接。
    """
    print("正在测试 LLM (Chat) 连接...")
    if not all([GRAPHRAG_API_BASE, GRAPHRAG_CHAT_API_KEY, GRAPHRAG_CHAT_MODEL]):
        print("🟡 LLM 连接跳过: 环境变量未完全配置。")
        return None

    try:
        llm = ChatOpenAI(
            model=GRAPHRAG_CHAT_MODEL,
            api_key=GRAPHRAG_CHAT_API_KEY,
            base_url=GRAPHRAG_API_BASE,
            timeout=60
        )

        print(f"--- 正在向 {GRAPHRAG_API_BASE} 发送请求... ---")
        start_time = time.time()
        response = llm.invoke("Say hello.")
        end_time = time.time()

        if response.content:
            print(f"✅ LLM 连接成功！(耗时: {end_time - start_time:.2f}s)")
            print(f"--- 回复: {response.content}")
            return True
        else:
            print("❌ LLM 连接失败: API 没有返回内容。")
            return False
    except Exception as e:
        print(f"❌ LLM 连接失败: {e}")
        return False


def test_neo4j_connection():
    """
    测试与 Neo4j 数据库的连接。
    """
    print("\n正在测试 Neo4j 数据库连接...")
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        print("🟡 Neo4j 连接跳过: 环境变量未完全配置。")
        return None

    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        driver.verify_connectivity()
        print("✅ Neo4j 连接成功！")
        driver.close()
        return True
    except Exception as e:
        print(f"❌ Neo4j 连接失败: {e}")
        return False


def test_embedding_connection():
    """
    测试与嵌入模型 (Embedding) 的连接。
    (支持自动切换 API / 本地模式)
    """
    print("\n正在测试 Embedding 模型连接...")

    try:
        # 【核心修改】直接调用 get_embedding_model()
        # 这个函数会自动根据 .env 配置决定是加载本地文件还是连 API
        embedding_model = get_embedding_model()

        test_text = "这是一个嵌入测试。"
        print(f"  [Embedding] 正在生成向量: '{test_text}'...")

        start_time = time.time()
        # 实际测试生成向量
        vector = embedding_model.embed_query(test_text)
        end_time = time.time()

        if vector and isinstance(vector, list) and len(vector) > 0:
            print(f"  [Embedding] 耗时: {end_time - start_time:.2f} 秒")
            print(f"  [Embedding] 成功获取 {len(vector)} 维向量。")
            print(f"✅ Embedding 模型工作正常！")
            return True
        else:
            print("❌ Embedding 模型失败: 未能返回有效的向量。")
            return False

    except Exception as e:
        print(f"❌ Embedding 模型错误: {e}")
        # 提示用户检查路径
        print("   提示: 如果是本地模式，请检查 .env 中的模型路径是否正确，且文件夹内包含 config.json 等文件。")
        return False