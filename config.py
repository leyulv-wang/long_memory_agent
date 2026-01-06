import os
from dotenv import load_dotenv
from datetime import timedelta, time as dt_time # 【新增】导入时间和时间差模块

from pathlib import Path

# --- 路径定义 ---
_current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(_current_file_path)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 【新增】为所有智能体的独立数据（如短期记忆）创建一个总目录
AGENTS_DATA_DIR = os.path.join(DATA_DIR, "agents")

WORLD_KNOWLEDGE_DIR = os.path.join(DATA_DIR, "world_knowledge")
CHARACTER_BOOKS_DIR = os.path.join(DATA_DIR, "books") # 【新增】角色书的目录

# 从 .env 文件加载环境变量
load_dotenv()

# --- LLM 与嵌入模型配置 ---
GRAPHRAG_API_BASE = os.getenv("GRAPHRAG_API_BASE")
GRAPHRAG_CHAT_API_KEY = os.getenv("GRAPHRAG_CHAT_API_KEY")
GRAPHRAG_CHAT_MODEL = os.getenv("GRAPHRAG_CHAT_MODEL")
#这个embedding我用的本地的
GRAPHRAG_EMBEDDING_API_BASE = os.getenv("GRAPHRAG_EMBEDDING_API_BASE")
GRAPHRAG_EMBEDDING_API_KEY = os.getenv("GRAPHRAG_EMBEDDING_API_KEY")
GRAPHRAG_EMBEDDING_MODEL = os.getenv("GRAPHRAG_EMBEDDING_MODEL")

#这个是便宜的，用这个
CHEAP_GRAPHRAG_API_BASE = os.getenv("CHEAP_GRAPHRAG_API_BASE")
CHEAP_GRAPHRAG_CHAT_API_KEY = os.getenv("CHEAP_GRAPHRAG_CHAT_API_KEY")
CHEAP_GRAPHRAG_CHAT_MODEL = os.getenv("CHEAP_GRAPHRAG_CHAT_MODEL")

# --- LlamaIndex (KG ingestion) 配置：沿用你现有 OpenAI-compatible 变量 ---
LLAMAINDEX_LLM_API_BASE = os.getenv("LLAMAINDEX_LLM_API_BASE", CHEAP_GRAPHRAG_API_BASE)
LLAMAINDEX_LLM_API_KEY  = os.getenv("LLAMAINDEX_LLM_API_KEY", CHEAP_GRAPHRAG_CHAT_API_KEY)
LLAMAINDEX_LLM_MODEL    = os.getenv("LLAMAINDEX_LLM_MODEL", CHEAP_GRAPHRAG_CHAT_MODEL)

LLAMAINDEX_EMB_API_BASE = os.getenv("LLAMAINDEX_EMB_API_BASE", GRAPHRAG_EMBEDDING_API_BASE)
LLAMAINDEX_EMB_API_KEY  = os.getenv("LLAMAINDEX_EMB_API_KEY", GRAPHRAG_EMBEDDING_API_KEY)
LLAMAINDEX_EMB_MODEL    = os.getenv("LLAMAINDEX_EMB_MODEL", GRAPHRAG_EMBEDDING_MODEL)

llamaindex_config = {
    "llm": {
        "base_url": LLAMAINDEX_LLM_API_BASE,
        "api_key": LLAMAINDEX_LLM_API_KEY,
        "model": LLAMAINDEX_LLM_MODEL,
    },
    "embeddings": {
        "base_url": LLAMAINDEX_EMB_API_BASE,
        "api_key": LLAMAINDEX_EMB_API_KEY,
        "model": LLAMAINDEX_EMB_MODEL,
    },
}

# --- 向量索引一致性策略（强鲁棒：不一致即退出） ---
VECTOR_INDEX_META_KEY = "vector_index_meta_v1"

# 只要 embedding 维度或模型名与数据库记录不一致：直接报错退出（评测推荐 True）
FAIL_FAST_ON_VECTOR_INDEX_MISMATCH = True

# 向量索引维度探测失败时的兜底（极少发生）
VECTOR_DIM_FALLBACK = 1024


# --- 长期记忆 (Neo4j) 配置 ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- 短期记忆 (FAISS) 配置 ---
# 【修改】旧的全局 FAISS_INDEX_PATH 已不再需要，因为每个智能体都有自己的路径。
# FAISS_INDEX_PATH = os.path.join(DATA_DIR, FAISS_INDEX_FILENAME)
FAISS_INDEX_FILENAME = os.getenv("FAISS_INDEX_FILENAME", "memory.faiss")


# --- 认知参数 ---
memory_consolidation_threshold = 10#阈值

related_memories = 12#和阈值数量的短期记忆抽象出来的主题相关的短期记忆，比阈值大一点，因为有可能检索出来的相关记忆里很多和最近的阈值数量的那几条记忆重复了
short_memory_number = 10#构建上下文，CFF中的短期记忆数量
long_memory_number = 10#构建上下文，CFF中的长期记忆数量，长期记忆数据库中检索旧的信念的数量
# 【新增】短期记忆窗口大小
SHORT_TERM_MEMORY_WINDOW = int(os.getenv("SHORT_TERM_MEMORY_WINDOW", 50))


# 针对 Turn/Step (按步数衰减)，建议 0.001 (约 1000 步衰减到 37%)
STEP_DECAY_RATE = 0.001
# --- GraphRAG 配置 ---
graphrag_config = {
    "llm": {
        "type": "openai",
        "api_key": GRAPHRAG_CHAT_API_KEY,
        "base_url": GRAPHRAG_API_BASE,
        "model": GRAPHRAG_CHAT_MODEL,
    },
    "embeddings": {
        "llm": {
            "type": "openai",
            "api_key": GRAPHRAG_EMBEDDING_API_KEY,
            "base_url": GRAPHRAG_API_BASE,
            "model": GRAPHRAG_EMBEDDING_MODEL,
        }
    },
    "community_reports": {
        "llm": {
            "type": "openai",
            "api_key": GRAPHRAG_CHAT_API_KEY,
            "base_url": GRAPHRAG_API_BASE,
            "model": GRAPHRAG_CHAT_MODEL,
        }
    },
    "entity_extraction": {
        "llm": {
            "type": "openai",
            "api_key": GRAPHRAG_CHAT_API_KEY,
            "base_url": GRAPHRAG_API_BASE,
            "model": GRAPHRAG_CHAT_MODEL,
        }
    },
    "input": {
        "input_type": "text",
        "input_dir": str(WORLD_KNOWLEDGE_DIR)
    },
    "graph_storage": {
        "type": "graphdb",
        "uri": NEO4J_URI,
        "username": NEO4J_USERNAME,
        "password": NEO4J_PASSWORD,
        "driver_auth_type": "basic",
        "database": "neo4j",
    },
    "cache": {
        "type": "graphdb",
        "uri": NEO4J_URI,
        "username": NEO4J_USERNAME,
        "password": NEO4J_PASSWORD,
        "driver_auth_type": "basic",
        "database": "graphrag_cache",
    },
    "reporting": {
        "type": "graphdb",
        "uri": NEO4J_URI,
        "username": NEO4J_USERNAME,
        "password": NEO4J_PASSWORD,
        "driver_auth_type": "basic",
        "database": "graphrag_reports",
    },
    "storage_type": "graphdb",
    "chunks": {"size": 1024, "overlap": 512},
    "root_dir": os.path.join(PROJECT_ROOT, "graphrag_output"),
    "encoding_model": "cl100k_base",
}

# --- 【新增】模拟世界配置 ---

# # 模拟开始时间 (时, 分)
# SIMULATION_START_TIME = {"hour": 9, "minute": 0}
#
# # 智能体默认出生点
# DEFAULT_AGENT_LOCATION = "houseZ"
#
# # 定义所有具有动态状态和定时事件的实体
#
# DYNAMIC_ENTITIES = [
#     {
#         "name": "dessert shop",      # 实体名称 (必须与 Neo4j/代码中的 key 一致)
#         "state_property": "status", # 哪个属性是动态的 (e.g., "status", "is_locked")
#         "events": [
#             # 规则列表：(触发时间, 变为的状态, 重复间隔)
#             {"trigger_time": dt_time(9, 0), "new_state": "open", "recurring_interval": timedelta(days=1)},
#             {"trigger_time": dt_time(18, 0), "new_state": "closed", "recurring_interval": timedelta(days=1)}
#         ]
#     },
#     {
#         "name": "gym",
#         "state_property": "status",
#         "events": [
#             {"trigger_time": dt_time(6, 0), "new_state": "open", "recurring_interval": timedelta(days=1)},
#             {"trigger_time": dt_time(22, 0), "new_state": "closed", "recurring_interval": timedelta(days=1)}
#         ]
#     },
#     {
#         "name": "park", # 公园本身
#         "state_property": "status", # 比如 "status" 可以是 "bright" 或 "dark"
#         "events": [
#             {"trigger_time": dt_time(7, 0), "new_state": "bright", "recurring_interval": timedelta(days=1)},
#             {"trigger_time": dt_time(19, 0), "new_state": "dark", "recurring_interval": timedelta(days=1)}
#         ]
#     },
#     {
#         "name": "password lock",     # 城堡地下室的密码锁
#         "state_property": "status",
#         "initial_state_at_start": "locked", # 这个没有定时事件，但有初始状态
#         "events": [
#             # 这个实体的状态只能通过智能体行动来改变
#         ]
#     },
#     {
#         "name": "statue",             # 公园的雕像
#         "state_property": "status",
#         "initial_state_at_start": "inactive", # 初始状态
#         "events": []
#     }
#     # ... 未来可以添加更多，比如 "coffee machine" (on/off) ...
# ]