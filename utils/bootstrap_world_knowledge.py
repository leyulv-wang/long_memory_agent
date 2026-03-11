# -*- coding: utf-8 -*-
"""
用途：
使用 LlamaIndex PropertyGraphIndex + Neo4jPropertyGraphStore
将 data/world_knowledge 下的世界书 txt 文件
“一键结构化 + 导入 Neo4j”

目标：
- 世界书作为基础信息层
- 不做衰减，不做双通道
- 只打世界知识标记，避免和后续记忆/巩固图冲突

关键点：
- 只对“本次世界书导入产生/关联”的节点与关系打标（绝不污染全库）
- 给世界书节点统一加 :World label
- 给世界书关系统一加 scope/source_type=world
- confidence 缺省补 1.0（世界书默认 ground truth）
"""

import os
import sys
import logging
from typing import List, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import (
    WORLD_KNOWLEDGE_DIR,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    CHEAP_GRAPHRAG_API_BASE,
    CHEAP_GRAPHRAG_CHAT_MODEL,
    CHEAP_GRAPHRAG_CHAT_API_KEY,
)
from utils.embedding import get_embedding_model

from llama_index.core.embeddings import BaseEmbedding


# ----------------------------
# Embedding 适配器：LangChain Embeddings -> LlamaIndex BaseEmbedding
# （支持单条 + 批量，避免 LlamaIndex 走 batch 接口时报错）
# ----------------------------
class LocalEmbeddingAdapter(BaseEmbedding):
    _lc_model: Any = None

    def __init__(self, lc_model, **kwargs):
        super().__init__(**kwargs)
        self._lc_model = lc_model

    # --- query embedding ---
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._lc_model.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    # --- text embedding (single) ---
    def _get_text_embedding(self, text: str) -> List[float]:
        # 对齐你的 embedding 客户端：query/text 都可以用 embed_query
        return self._lc_model.embed_query(text)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    # --- text embeddings (batch) ---
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        # 如果底层支持 embed_documents（你的 RemoteEmbeddingClient 支持），优先批量入口
        if hasattr(self._lc_model, "embed_documents"):
            return self._lc_model.embed_documents(texts)
        return [self._lc_model.embed_query(t) for t in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


# ----------------------------
# 日志配置（中文）
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("世界书导入")
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


def _should_skip_apoc_schema(uri: str) -> bool:
    flag = os.getenv("NEO4J_DISABLE_APOC_SCHEMA", "0").strip().lower() in ("1", "true", "yes")
    if flag:
        return True
    u = (uri or "").lower()
    if u.startswith("neo4j+s://") or u.startswith("neo4j+ssc://"):
        return True
    return os.getenv("USE_NEO4J_AURA", "0").strip().lower() in ("1", "true", "yes")


def _create_graph_store():
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

    if not _should_skip_apoc_schema(NEO4J_URI):
        return Neo4jPropertyGraphStore(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database="neo4j",
        )

    original_refresh = Neo4jPropertyGraphStore.refresh_schema

    def _noop_refresh(self):
        return None

    try:
        Neo4jPropertyGraphStore.refresh_schema = _noop_refresh
        return Neo4jPropertyGraphStore(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database="neo4j",
        )
    finally:
        Neo4jPropertyGraphStore.refresh_schema = original_refresh


def load_world_books(world_dir: str):
    """读取世界书目录下的 txt 文件，构造 LlamaIndex Document，写入 metadata 标记"""
    from llama_index.core import Document

    if not os.path.exists(world_dir):
        logger.warning(f"❌ 世界书目录不存在：{world_dir}")
        return []

    documents: List[Document] = []

    for filename in os.listdir(world_dir):
        if not filename.lower().endswith(".txt"):
            continue

        file_path = os.path.join(world_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    # 这些 metadata 最终会进入 Chunk/Entity 节点属性（你已经验证 book_name 存在）
                    "book_name": filename,
                    "source_path": file_path,
                    # 这里先不强依赖 source_type，后处理我们会更安全地加 :World label
                    "source_type": "world",
                },
            )
        )

    return documents


def bootstrap_world_knowledge():
    logger.info("🚀 开始使用 LlamaIndex 一键管线导入【世界书】到 Neo4j（防冲突版）")

    # 1) Embedding（你现在是远程服务/本地模型/API 三模式，均可）
    logger.info("🔧 初始化 Embedding（使用你的 utils.embedding.get_embedding_model）")
    my_embedding_model = get_embedding_model()

    # 2) 注入 LlamaIndex Settings
    from llama_index.core import Settings
    from llama_index.llms.openai_like import OpenAILike

    logger.info("🔧 初始化 LLM（OpenAILike，使用 CHEAP_GRAPHRAG_* 配置）")
    Settings.llm = OpenAILike(
        model=CHEAP_GRAPHRAG_CHAT_MODEL,
        api_base=CHEAP_GRAPHRAG_API_BASE,
        api_key=CHEAP_GRAPHRAG_CHAT_API_KEY,
        is_chat_model=True,
        context_window=32000,
        max_tokens=4096,
    )

    logger.info("🔧 注入 Embedding（自定义适配器，绕过 langchain embedding 依赖包）")
    Settings.embed_model = LocalEmbeddingAdapter(my_embedding_model)

    # 3) Neo4j Property Graph Store
    logger.info("Connecting to Neo4j Graph Store...")
    graph_store = _create_graph_store()

    # 4) 加载世界书
    logger.info(f"📚 加载世界书目录：{WORLD_KNOWLEDGE_DIR}")
    documents = load_world_books(WORLD_KNOWLEDGE_DIR)
    if not documents:
        logger.warning("⚠️ 未发现任何世界书 txt 文件，终止导入")
        graph_store.close()
        return

    logger.info(f"✅ 成功加载 {len(documents)} 本世界书")

    # 5) 构建抽取管线
    from llama_index.core.indices.property_graph import PropertyGraphIndex, SimpleLLMPathExtractor
    from llama_index.core.node_parser import SentenceSplitter

    logger.info("🧩 初始化文本切分器与 KG 抽取器")
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

    extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=20,
        num_workers=4,

    )

    # 6) 执行抽取 + 写入
    logger.info("🧠 正在调用 LLM 抽取知识图谱并写入 Neo4j（请耐心等待）")
    PropertyGraphIndex.from_documents(
        documents=documents,
        property_graph_store=graph_store,
        kg_extractors=[extractor],
        transformations=[splitter],
        show_progress=True,
    )

    # 7) 后处理（防冲突关键）
    # 只对 “book_name IS NOT NULL” 的节点（世界书导入产生/关联的 Chunk/Entity）进行打标与补值
    # 并且关系只标记“与这些 world 节点相连”的关系，避免污染全库
    logger.info("🏷️ 后处理：仅对世界书相关节点/关系打 :World 标记，并补默认置信度（不污染全库）")

    # 7.1 节点：加 :World label + scope/source_type + confidence 缺省补 1.0
    graph_store.structured_query(
        """
        MATCH (n)
        WHERE n.book_name IS NOT NULL
        SET n:World,
            n.scope = 'world',
            n.source_type = coalesce(n.source_type, 'world'),
            n.confidence = coalesce(n.confidence, 1.0)
        """
    )

    # 7.2 关系：仅标记与世界书节点相连的关系（Neo4j 5+ 用 id(n) 很稳）
    graph_store.structured_query(
        """
        MATCH (n:World)
        WITH collect(id(n)) AS world_ids
        MATCH (a)-[r]->(b)
        WHERE id(a) IN world_ids OR id(b) IN world_ids
        SET r.scope = 'world',
            r.source_type = coalesce(r.source_type, 'world'),
            r.confidence = coalesce(r.confidence, 1.0)
        """
    )

    logger.info("🎉 世界书导入完成：已加 :World 标签，后续可用 MATCH (n:World) 精确过滤")
    graph_store.close()


if __name__ == "__main__":
    bootstrap_world_knowledge()
