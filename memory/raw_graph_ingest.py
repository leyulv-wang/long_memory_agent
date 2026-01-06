# -*- coding: utf-8 -*-
"""
raw 通道导入：
- 输入：对话窗口原文（字符串）
- 输出：写入 Neo4j（LlamaIndex PropertyGraphIndex）
- 后处理：按 C 方案打标
    - 节点：n.channels += ['raw']
    - 关系：r.channel='raw' + agent_name + turn_id + virtual_time + confidence(缺省1.0)
    - 创建 TextUnit 证据节点，并把 doc_id 对应节点连接到 TextUnit（FROM_SOURCE）
    - ✅ 新增：创建 raw Event 节点，并连接证据 TextUnit；把本次 doc 相关关系挂 event_id
    - ✅ 新增：从文本标签解析 session_id/session_time，并写入 TextUnit/Event 节点
"""

from __future__ import annotations

import re
import datetime
import logging
from typing import Optional, Dict, Any, List, Tuple

from config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    CHEAP_GRAPHRAG_API_BASE,
    CHEAP_GRAPHRAG_CHAT_MODEL,
    CHEAP_GRAPHRAG_CHAT_API_KEY,
)
from utils.embedding import get_embedding_model

from memory.channels import (
    RAW,
    build_mark_params,
    cypher_mark_nodes_channels_by_doc_id,
    cypher_mark_rels_channel_connected_to_doc_nodes,
    cypher_create_textunit_for_raw,
    cypher_link_doc_nodes_to_textunit,
    cypher_create_event_and_link_textunit,
    cypher_attach_event_id_to_doc_rels,
)

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"\[(?P<k>[A-Za-z0-9_]+)=(?P<v>[^\]]+)\]")


def _extract_first_tag(text: str, key: str) -> Optional[str]:
    """从整段 text 里提取第一个 [key=xxx] 的值"""
    if not isinstance(text, str) or not text:
        return None
    for m in _TAG_RE.finditer(text):
        if m.group("k") == key:
            v = (m.group("v") or "").strip()
            return v if v else None
    return None


def _try_parse_time_to_iso(s: Optional[str]) -> Optional[str]:
    """
    兼容：
    - 数据集 LoCoMo 格式: 2023/05/21 (Sun) 19:50
    - ISO: 2023-05-21T19:50:00 或 2023-05-21 19:50:00
    解析失败：返回 None
    """
    if not s or not isinstance(s, str):
        return None
    ss = s.strip()

    # 1) 直接是 ISO
    try:
        # fromisoformat 兼容 "YYYY-MM-DDTHH:MM:SS" 和 "YYYY-MM-DD HH:MM:SS"
        dt = datetime.datetime.fromisoformat(ss.replace("Z", ""))
        return dt.isoformat()
    except Exception:
        pass

    # 2) LoCoMo 格式
    try:
        dt = datetime.datetime.strptime(ss, "%Y/%m/%d (%a) %H:%M")
        return dt.isoformat()
    except Exception:
        pass

    return None


def _extract_session_meta(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    从 raw window 文本中提取 session 元信息：
    - session_id
    - session_time_raw（原始字符串）
    - session_time_iso（可解析则给 ISO）
    """
    session_id = _extract_first_tag(text, "session_id")
    session_time_raw = _extract_first_tag(text, "session_time")
    session_time_iso = _try_parse_time_to_iso(session_time_raw)
    return session_id, session_time_raw, session_time_iso


def ingest_raw_dialogue_window(
    *,
    agent_name: str,
    text: str,
    virtual_time: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    channel: str = RAW,
    max_paths_per_chunk: int = 20,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    raw 一键入图 + 打标 + raw Event

    返回：
    {
      "doc_id": "...",          # 本次导入 doc_id（用它定位本次导入的节点）
      "textunit": "...",        # 本次 raw TextUnit 名称
      "event_id": "...",        # 本次 raw Event id
      "turn_id": int,
      "channel": "raw",
      "session_id": Optional[str],
      "session_time_raw": Optional[str],
      "session_time_iso": Optional[str],
    }
    """
    if not text or not str(text).strip():
        raise ValueError("raw 导入失败：text 为空")

    # 0) 从 text 提取 session 元信息（数据集提供的时间/会话ID）
    session_id, session_time_raw, session_time_iso = _extract_session_meta(str(text))

    # 1) 初始化 LlamaIndex Settings（OpenAILike + 你的 embedding）
    from llama_index.core import Settings, Document
    from llama_index.llms.openai_like import OpenAILike
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.indices.property_graph import PropertyGraphIndex, SimpleLLMPathExtractor
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

    Settings.llm = OpenAILike(
        model=CHEAP_GRAPHRAG_CHAT_MODEL,
        api_base=CHEAP_GRAPHRAG_API_BASE,
        api_key=CHEAP_GRAPHRAG_CHAT_API_KEY,
        is_chat_model=True,
        context_window=32000,
        max_tokens=4096,
    )

    # Embedding：走你的 utils.embedding（远程服务 / 本地 / API 三模式）
    embed_model = get_embedding_model()

    # 为了避免引入 llama-index-embeddings-langchain，内嵌最小适配器
    from llama_index.core.embeddings import BaseEmbedding

    class _LocalEmbeddingAdapter(BaseEmbedding):
        def __init__(self, lc_model, **kwargs):
            super().__init__(**kwargs)
            self._lc_model = lc_model

        def _get_query_embedding(self, query: str) -> List[float]:
            return self._lc_model.embed_query(query)

        async def _aget_query_embedding(self, query: str) -> List[float]:
            return self._get_query_embedding(query)

        def _get_text_embedding(self, text: str) -> List[float]:
            return self._lc_model.embed_query(text)

        async def _aget_text_embedding(self, text: str) -> List[float]:
            return self._get_text_embedding(text)

        def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            if hasattr(self._lc_model, "embed_documents"):
                return self._lc_model.embed_documents(texts)
            return [self._lc_model.embed_query(t) for t in texts]

        async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            return self._get_text_embeddings(texts)

    Settings.embed_model = _LocalEmbeddingAdapter(embed_model)

    # 2) 连接 Neo4j graph store
    graph_store = Neo4jPropertyGraphStore(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database="neo4j",
    )

    # 3) 生成本次导入 doc_id / TextUnit id
    now_iso = datetime.datetime.now().isoformat()
    safe_agent = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in (agent_name or "agent"))
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    doc_id = f"raw_{safe_agent}_{ts}"
    tu_name = f"raw_unit_{safe_agent}_{ts}"

    # 4) Document metadata（把 doc_id 写进去：保证入库节点能被定位）
    md = dict(metadata or {})
    md.update(
        {
            "doc_id": doc_id,
            "channel": RAW,
            "agent_name": agent_name,
            "virtual_time": virtual_time or "unknown",
            # ✅ session 元信息（来自数据集标签）
            "session_id": session_id or "",
            "session_time_raw": session_time_raw or "",
            "session_time_iso": session_time_iso or "",
        }
    )
    doc = Document(text=str(text), metadata=md)

    # 5) Pipeline：split + KG extractor + 写入 Neo4j
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        max_paths_per_chunk=max_paths_per_chunk,
        num_workers=num_workers,
    )

    logger.info(
        f"[raw_graph_ingest] 开始 raw 导入：agent={agent_name}, doc_id={doc_id}, "
        f"virtual_time={virtual_time}, session_id={session_id}, session_time={session_time_raw}"
    )

    PropertyGraphIndex.from_documents(
        documents=[doc],
        property_graph_store=graph_store,
        kg_extractors=[extractor],
        transformations=[splitter],
        show_progress=False,
    )

    # 6) 后处理：按 C 方案打标（只影响本次 doc_id 对应节点/关系）
    params = build_mark_params([doc_id], agent_name=agent_name, channel=RAW, virtual_time=virtual_time)

    # 6.1 创建 raw TextUnit（证据）并写 embedding
    try:
        tu_emb = embed_model.embed_query(str(text))
    except Exception:
        tu_emb = None

    graph_store.structured_query(
        cypher_create_textunit_for_raw(),
        {
            "name": tu_name,
            "content": str(text),
            "virtual_time": params["virtual_time"],
            "turn_id": params["turn_id"],
            "real_time": now_iso,
            "embedding": tu_emb,
            "agent_name": agent_name,
            "channel": RAW,
        },
    )

    # ✅ 6.1.1 额外补写 session 元信息到 TextUnit（不依赖 channels.py 的 cypher 是否支持）
    graph_store.structured_query(
        """
        MATCH (u:TextUnit {name:$name})
        SET u.session_id = $session_id,
            u.session_time_raw = $session_time_raw,
            u.session_time_iso = $session_time_iso
        """,
        {
            "name": tu_name,
            "session_id": session_id or "",
            "session_time_raw": session_time_raw or "",
            "session_time_iso": session_time_iso or "",
        },
    )

    # 6.2 节点 channels += raw
    graph_store.structured_query(cypher_mark_nodes_channels_by_doc_id(), params)

    # 6.3 关系写 channel=raw（仅标记连到本次 doc 节点的关系）
    graph_store.structured_query(cypher_mark_rels_channel_connected_to_doc_nodes(), params)

    # 6.4 连接证据（doc nodes -> TextUnit）
    graph_store.structured_query(
        cypher_link_doc_nodes_to_textunit(),
        {"textunit": tu_name, "doc_ids": [doc_id]},
    )

    # 6.5 ✅ 创建 raw Event，并连接证据 TextUnit
    # raw 阶段 event_timestamp 仍用 virtual_time（轮次），同时补写 session_time
    event_timestamp = params["virtual_time"]
    event_id = f"evt:{RAW}:{safe_agent}:{params['virtual_time']}:{tu_name}"

    graph_store.structured_query(
        cypher_create_event_and_link_textunit(),
        {
            "event_id": event_id,
            "channel": RAW,
            "agent_name": agent_name,
            "virtual_time": params["virtual_time"],
            "turn_id": params["turn_id"],
            "event_timestamp": event_timestamp,
            "updated_at": now_iso,
            "textunit": tu_name,
        },
    )

    # ✅ 6.5.1 额外补写 session 元信息到 Event，并写一个 event_time_iso（先等于 session_time_iso，否则用 raw）
    graph_store.structured_query(
        """
        MATCH (e:Event {event_id:$event_id})
        SET e.session_id = $session_id,
            e.session_time_raw = $session_time_raw,
            e.session_time_iso = $session_time_iso,
            e.event_time_iso = CASE
                WHEN $session_time_iso <> '' THEN $session_time_iso
                WHEN $session_time_raw <> '' THEN $session_time_raw
                ELSE ''
            END
        """,
        {
            "event_id": event_id,
            "session_id": session_id or "",
            "session_time_raw": session_time_raw or "",
            "session_time_iso": session_time_iso or "",
        },
    )

    # 6.6 把本次 doc 相关关系挂上 event_id（方便追踪/解释）
    graph_store.structured_query(
        cypher_attach_event_id_to_doc_rels(),
        {"doc_ids": [doc_id], "event_id": event_id},
    )

    # 6.7 GraphRAG-style evidence chain: __Document__ / __Chunk__ / HAS_ENTITY
    chunk_id = f"{doc_id}:{tu_name}"
    graph_store.structured_query(
        """
        MERGE (d:__Document__ {id: $doc_id})
        ON CREATE SET d.title = $title, d.raw_content = $raw
        SET d.agent_name = $agent_name,
            d.updated_at = $updated_at,
            d.session_id = $session_id,
            d.session_time_raw = $session_time_raw,
            d.session_time_iso = $session_time_iso
        """,
        {
            "doc_id": doc_id,
            "title": session_id or doc_id,
            "raw": str(text),
            "agent_name": agent_name,
            "updated_at": now_iso,
            "session_id": session_id or "",
            "session_time_raw": session_time_raw or "",
            "session_time_iso": session_time_iso or "",
        },
    )

    graph_store.structured_query(
        """
        MERGE (c:__Chunk__ {id: $chunk_id})
        SET c.text = $text,
            c.agent_name = $agent_name,
            c.channel = $channel,
            c.virtual_time = $virtual_time,
            c.turn_id = $turn_id,
            c.session_id = $session_id,
            c.session_time_raw = $session_time_raw,
            c.session_time_iso = $session_time_iso
        WITH c
        MATCH (d:__Document__ {id: $doc_id})
        MERGE (c)-[:PART_OF]->(d)
        WITH c
        MATCH (u:TextUnit {name: $textunit_id})
        MERGE (u)-[:DERIVED_FROM]->(c)
        """,
        {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": str(text),
            "agent_name": agent_name,
            "channel": RAW,
            "virtual_time": params["virtual_time"],
            "turn_id": params["turn_id"],
            "session_id": session_id or "",
            "session_time_raw": session_time_raw or "",
            "session_time_iso": session_time_iso or "",
            "textunit_id": tu_name,
        },
    )

    # Link chunk -> entities created by this doc (avoid Chunk/TextUnit nodes)
    graph_store.structured_query(
        """
        MATCH (c:__Chunk__ {id: $chunk_id})
        MATCH (n)
        WHERE n.doc_id = $doc_id OR n.document_id = $doc_id OR n.ref_doc_id = $doc_id
        WITH c, n, labels(n) AS labs
        WHERE NOT 'Chunk' IN labs
          AND NOT '__Chunk__' IN labs
          AND NOT 'TextUnit' IN labs
          AND NOT '__Document__' IN labs
          AND NOT '__Node__' IN labs
        MERGE (c)-[:HAS_ENTITY]->(n)
        """,
        {"chunk_id": chunk_id, "doc_id": doc_id},
    )

    graph_store.close()

    logger.info(
        f"[raw_graph_ingest] raw 导入完成：doc_id={doc_id}, textunit={tu_name}, event_id={event_id}, "
        f"turn_id={params['turn_id']}, session_id={session_id}"
    )

    return {
        "doc_id": doc_id,
        "textunit": tu_name,
        "event_id": event_id,
        "turn_id": params["turn_id"],
        "channel": RAW,
        "session_id": session_id,
        "session_time_raw": session_time_raw,
        "session_time_iso": session_time_iso,
    }
