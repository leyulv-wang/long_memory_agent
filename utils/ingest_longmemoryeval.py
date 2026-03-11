# -*- coding: utf-8 -*-
"""
utils/ingest_longmemoryeval.py  (FINAL COVER VERSION)

面向 LongMemoryEval / LoCoMo 数据的“特化注入”入口（双通道）：

核心特化点：
1) turn_id 固定为该 task 内“会话时间顺序序号”（优先按 haystack_dates 排序；失败用原顺序，稳定）
2) question_date -> question_turn_id（用于衰减/检索 current_time=TURN_{question_turn_id}）
3) RAW：批量窗口注入（每会话一个 doc），用 LlamaIndex KG 抽取写 Neo4j
4) CONSOLIDATED：并行跑 invoke_json() 得到核心事实（JSON），再按 turn_id 全 task 严格串行写回 Neo4j
5) RAW TextUnit 必写 content + embedding，保证后续 TextUnit 向量召回有效

返回：
{
  "question_turn_id": int,
  "num_sessions": int,
  "doc_infos": List[RawDocInfo]   # 全 task
}
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("ingest_longmemoryeval")
_DEBUG_INGEST = os.getenv("DEBUG_INGEST", "0") == "1"

# ========= 项目内依赖 =========
from config import (
    memory_consolidation_threshold,
    CHEAP_GRAPHRAG_API_BASE,
    CHEAP_GRAPHRAG_CHAT_MODEL,
    CHEAP_GRAPHRAG_CHAT_API_KEY,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    RAW_REL_CONFIDENCE,
)

from utils.embedding import get_embedding_model
from utils.llm import get_llm, invoke_json

from memory.channels import RAW, CONSOLIDATED
from memory.structured_memory import KnowledgeGraphExtraction
from memory.ltss_writer import write_consolidation_result
from utils.original_consolidation import consolidate_original_session

# ====== llama-index / neo4j store ======
from llama_index.core import Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.property_graph import PropertyGraphIndex, SimpleLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai_like import OpenAILike

from llama_index.core.embeddings import BaseEmbedding

_UPDATE_REL_TYPES = {
    "LIVES_IN",
    "RESIDES_IN",
    "HAS_LOCATION",
    "WORKS_AT",
    "HAS_JOB",
    "HAS_STATUS",
    "HAS_PHONE",
    "HAS_EMAIL",
}


def _should_skip_apoc_schema() -> bool:
    flag = os.getenv("NEO4J_DISABLE_APOC_SCHEMA", "0").strip().lower() in ("1", "true", "yes")
    if flag:
        return True
    uri = (NEO4J_URI or "").lower()
    if uri.startswith("neo4j+s://") or uri.startswith("neo4j+ssc://"):
        return True
    return os.getenv("USE_NEO4J_AURA", "0").strip().lower() in ("1", "true", "yes")


def _create_graph_store():
    if not _should_skip_apoc_schema():
        return Neo4jPropertyGraphStore(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database="neo4j",
        )

    # Aura often blocks apoc.meta.data; skip schema refresh to avoid failure.
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


# =========================
# 数据集时间解析
# =========================
_DATASET_TIME_FMT = "%Y/%m/%d (%a) %H:%M"


def _try_parse_dataset_dt(s: str) -> Optional[datetime.datetime]:
    raw = (s or "").strip()
    if not raw:
        return None
    try:
        return datetime.datetime.strptime(raw, _DATASET_TIME_FMT)
    except Exception:
        # 兼容：可能是 ISO 或 "YYYY-MM-DD HH:MM:SS"
        try:
            return datetime.datetime.fromisoformat(raw.replace("Z", ""))
        except Exception:
            return None


def _parse_dataset_time(s: str) -> Tuple[str, str]:
    """
    输入：2023/05/21 (Sun) 19:50
    输出：(raw_str, iso_str)；失败则 ("unknown","unknown")
    """
    raw = (s or "").strip()
    if not raw:
        return "unknown", "unknown"
    dt = _try_parse_dataset_dt(raw)
    if dt is None:
        return raw, "unknown"
    return raw, dt.isoformat()


# =========================
# Session 文本构建（每个 session 一个 doc）
# =========================
def build_session_text(
    session_turns: List[Dict[str, Any]],
    *,
    include_assistant: bool = True,
    max_turn_chars: int = 6000,
) -> str:
    """
    把一个 session 的 turns 格式化为单个字符串，给 LlamaIndex/Embedding/巩固用。
    turn dict 期望至少包含：role, content
    """
    if not session_turns:
        return ""

    buf: List[str] = []
    for t in session_turns:
        role = (t.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        if (not include_assistant) and role == "assistant":
            continue

        content = (t.get("content") or "").strip()
        if not content:
            continue

        if len(content) > max_turn_chars:
            content = content[: max_turn_chars - 3] + "..."

        buf.append(f"{role.upper()}: {content}")

    return "\n".join(buf).strip()


# =========================
# TextUnit 分块（提高检索粒度）
# =========================
# 配置开关：是否启用分块
ENABLE_TEXTUNIT_CHUNKING = os.getenv("ENABLE_TEXTUNIT_CHUNKING", "1").strip().lower() in ("1", "true", "yes")


@dataclass
class ChunkInfo:
    """分块信息"""
    chunk_id: int           # 块序号（从1开始）
    total_chunks: int       # 总块数
    content: str            # 块内容
    start_turn_idx: int     # 起始 turn 索引
    end_turn_idx: int       # 结束 turn 索引


def chunk_session_turns(
    session_turns: List[Dict[str, Any]],
    *,
    include_assistant: bool = True,
    max_turn_chars: int = 6000,
) -> List[ChunkInfo]:
    """
    将 session 的 turns 分块，提高检索粒度。
    
    分块规则：
    1. 基本单位：一个 USER + 对应的 ASSISTANT 回复 = 一个 chunk
    2. 每个 chunk 独立向量化
    
    Args:
        session_turns: session 的 turns 列表
        include_assistant: 是否包含 assistant 回复
        max_turn_chars: 单个 turn 最大字符数（截断保护）
    
    Returns:
        ChunkInfo 列表
    """
    if not session_turns:
        return []
    
    # 按 USER+ASSISTANT 对分组，每对一个 chunk
    chunks: List[ChunkInfo] = []
    current_pair_start = 0
    current_pair_text = []
    
    for i, t in enumerate(session_turns):
        role = (t.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        if (not include_assistant) and role == "assistant":
            continue
        
        content = (t.get("content") or "").strip()
        if not content:
            continue
        
        # 截断保护（防止单个 turn 过长）
        if len(content) > max_turn_chars:
            content = content[: max_turn_chars - 3] + "..."
        
        turn_text = f"{role.upper()}: {content}"
        
        # 如果是 USER 且已有内容，说明上一对结束了，保存为一个 chunk
        if role == "user" and current_pair_text:
            chunks.append(ChunkInfo(
                chunk_id=len(chunks) + 1,
                total_chunks=0,  # 稍后更新
                content="\n".join(current_pair_text),
                start_turn_idx=current_pair_start,
                end_turn_idx=i - 1,
            ))
            current_pair_text = []
            current_pair_start = i
        
        if not current_pair_text:
            current_pair_start = i
        
        current_pair_text.append(turn_text)
    
    # 保存最后一个 chunk
    if current_pair_text:
        chunks.append(ChunkInfo(
            chunk_id=len(chunks) + 1,
            total_chunks=0,
            content="\n".join(current_pair_text),
            start_turn_idx=current_pair_start,
            end_turn_idx=len(session_turns) - 1,
        ))
    
    # 更新 total_chunks
    total = len(chunks)
    for c in chunks:
        c.total_chunks = total
    
    return chunks


# =========================
# 时间线索抽取（帮助巩固恢复 yesterday/today）
# =========================
_TIME_HINT_RE = re.compile(
    r"\b("
    r"yesterday|today|tomorrow|tonight|last\s+night|this\s+morning|this\s+afternoon|this\s+evening|"
    r"last\s+week|next\s+week|this\s+week|last\s+month|next\s+month|this\s+month|last\s+year|next\s+year|"
    r"on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"in\s+\d{4}|"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\b\d{1,2}:\d{2}\b|"
    r"\b\d+\s+(days?|weeks?|months?|years?)\s+(ago|later)\b"
    r")\b",
    re.IGNORECASE,
)


def extract_time_snippets(text: str, *, max_lines: int = 20, max_chars: int = 2500) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    lines = []
    for ln in text.splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        if _TIME_HINT_RE.search(s):
            lines.append(s)
    # 去重
    uniq = []
    seen = set()
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    uniq = uniq[:max_lines]
    out = "\n".join(uniq)
    if len(out) > max_chars:
        out = out[:max_chars] + " ..."
    return out


# =========================
# turn_id 特化：按 haystack_dates 排序
# =========================
def build_task_turn_plan(
    sessions: List[List[Dict[str, Any]]],
    session_ids: List[str],
    session_dates: List[str],
) -> Dict[str, Any]:
    """
    输出：
    - ordered_indices: 按时间排序后的原 index 列表
    - turn_id_by_index: 原 index -> turn_id(1..N)
    - ordered_sessions/... : 按 turn_id 顺序排列后的列表
    """
    n = len(sessions)
    ids = session_ids if session_ids and len(session_ids) == n else [f"session_{i}" for i in range(n)]
    dates = session_dates if session_dates and len(session_dates) == n else ["unknown"] * n

    items = []
    for i in range(n):
        dt = _try_parse_dataset_dt(dates[i])  # 可能 None
        # stable sort：dt None 放最后，次关键字用原 index 保证稳定
        items.append((dt is None, dt or datetime.datetime.max, i))

    items.sort(key=lambda x: (x[0], x[1], x[2]))
    ordered_indices = [i for (_, __, i) in items]

    turn_id_by_index = {idx: (rank + 1) for rank, idx in enumerate(ordered_indices)}

    ordered_sessions = [sessions[i] for i in ordered_indices]
    ordered_ids = [ids[i] for i in ordered_indices]
    ordered_dates = [dates[i] for i in ordered_indices]

    return {
        "ordered_indices": ordered_indices,
        "turn_id_by_index": turn_id_by_index,
        "ordered_sessions": ordered_sessions,
        "ordered_session_ids": ordered_ids,
        "ordered_session_dates": ordered_dates,
    }


def compute_question_turn_id(question_date: str, haystack_dates: List[str]) -> int:
    """
    将 question_date 插入 haystack_dates 的时间轴上，得到 question_turn_id：
      question_turn_id = 1 + count(haystack_dt <= question_dt)  (仅统计可解析 dt)
    若 question_date 不可解析，则默认在最后：N+1
    """
    qdt = _try_parse_dataset_dt(question_date)
    if qdt is None:
        return len(haystack_dates) + 1

    dts = []
    for s in haystack_dates or []:
        dt = _try_parse_dataset_dt(s)
        if dt is not None:
            dts.append(dt)

    dts.sort()
    cnt = 0
    for dt in dts:
        if dt <= qdt:
            cnt += 1
        else:
            break
    return cnt + 1


# =========================
# RAW 入图：LlamaIndex 一次性处理多个 docs
# =========================
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
            return self._lc_model.embed_documents(texts) or []
        return [self._lc_model.embed_query(t) for t in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)


@dataclass
class RawDocInfo:
    doc_id: str
    textunit_id: str
    event_id: str
    agent_name: str
    channel: str
    virtual_time: str
    turn_id: int
    session_id: str
    session_time_raw: str
    session_time_iso: str
    session_text: str
    session_turns: List[Dict[str, Any]]


def _safe_agent(agent_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in (agent_name or "agent"))

def _debug_doc_stats(graph_store: Neo4jPropertyGraphStore, doc_id: str) -> None:
    if not _DEBUG_INGEST:
        return
    try:
        cy_nodes = """
        MATCH (n)
        WHERE n.doc_id = $doc_id OR n.document_id = $doc_id OR n.ref_doc_id = $doc_id
        RETURN count(n) AS node_cnt
        """
        cy_rels = """
        MATCH (n)
        WHERE n.doc_id = $doc_id OR n.document_id = $doc_id OR n.ref_doc_id = $doc_id
        WITH collect(elementId(n)) AS ids
        MATCH (a)-[r]->(b)
        WHERE elementId(a) IN ids OR elementId(b) IN ids
        RETURN count(r) AS rel_cnt
        """
        node_rows = graph_store.structured_query(cy_nodes, {"doc_id": doc_id}) or []
        rel_rows = graph_store.structured_query(cy_rels, {"doc_id": doc_id}) or []
        node_cnt = node_rows[0].get("node_cnt") if node_rows else 0
        rel_cnt = rel_rows[0].get("rel_cnt") if rel_rows else 0
        logger.info(f"[ingest_longmemoryeval][debug] doc_id={doc_id} nodes={node_cnt} rels={rel_cnt}")
    except Exception as e:
        logger.info(f"[ingest_longmemoryeval][debug] doc_id={doc_id} stats failed: {e}")


def _postprocess_raw_doc(graph_store: Neo4jPropertyGraphStore, info: RawDocInfo, *, embedding_model=None, textunit_embedding: Optional[List[float]] = None):
    """
    RAW 写库后处理（只影响本次 doc_id 子图）：
    - 给节点打 channels
    - 创建 raw TextUnit（含 content + embedding）
    - 将 doc_id 节点统一连到 TextUnit（FROM_SOURCE）
    - 创建 raw Event，并连 TextUnit（EVIDENCED_BY）
    - 给 doc 相关关系补齐 r.channel / r.event_id / r.session_time_iso
    """
    emb_model = embedding_model or get_embedding_model()
    real_time = datetime.datetime.now().isoformat()

    # 1) 给 doc_id 相关节点打 channels
    cy_nodes = """
    MATCH (n)
    WHERE n.doc_id = $doc_id OR n.document_id = $doc_id OR n.ref_doc_id = $doc_id
    WITH n, $ch AS ch
    SET n.channels =
      CASE
        WHEN ch IN coalesce(n.channels, []) THEN coalesce(n.channels, [])
        ELSE coalesce(n.channels, []) + [ch]
      END
    SET n.channel = coalesce(n.channel, ch)
    SET n.agent_name = coalesce(n.agent_name, $agent_name)
    RETURN count(n) AS updated
    """
    node_rows = graph_store.structured_query(
        cy_nodes, {"doc_id": info.doc_id, "ch": info.channel, "agent_name": info.agent_name}
    )

    # 1.1) Ensure doc-scoped nodes have stable names for downstream Fact linking.
    cy_fix_names = """
    MATCH (n)
    WHERE n.doc_id = $doc_id OR n.document_id = $doc_id OR n.ref_doc_id = $doc_id
      AND (n.name IS NULL OR trim(toString(n.name)) = '')
    SET n.name = coalesce(
      n.title, n.text, n.value, n.id, n.key, n.label, toString(elementId(n))
    )
    RETURN count(n) AS fixed
    """
    graph_store.structured_query(cy_fix_names, {"doc_id": info.doc_id})

    # 2) raw TextUnit：写 content + embedding
    # 2) raw TextUnit：写 content + embedding
    tu_emb = textunit_embedding  # ✅ 优先吃窗口级批量 embedding

    if tu_emb is None:
        # 兼容：未传入则降级为单条
        try:
            if info.session_text and info.session_text.strip():
                tu_emb = emb_model.embed_query(info.session_text)
        except Exception as e:
            logger.warning(f"[ingest_longmemoryeval] raw TextUnit embedding failed: {e}")
            tu_emb = None

    cy_textunit = """
    MERGE (u:TextUnit {name: $textunit})
    SET
      u.doc_id = $doc_id,
      u.agent_name = $agent_name,
      u.channel = $ch,
      u.virtual_time = $virtual_time,
      u.turn_id = $turn_id,
      u.real_time = $real_time,
      u.session_id = $session_id,
      u.session_time_raw = $session_time_raw,
      u.session_time_iso = $session_time_iso,
      u.content = $content,
      u.embedding = $embedding
    RETURN u.name AS name
    """
    tu_rows = graph_store.structured_query(
        cy_textunit,
        {
            "textunit": info.textunit_id,
            "doc_id": info.doc_id,
            "agent_name": info.agent_name,
            "ch": info.channel,
            "virtual_time": info.virtual_time,
            "turn_id": info.turn_id,
            "real_time": real_time,
            "session_id": info.session_id,
            "session_time_raw": info.session_time_raw,
            "session_time_iso": info.session_time_iso,
            "content": info.session_text or "",
            "embedding": tu_emb,
        },
    )

    # 2.1) doc_id 节点连接到 TextUnit（证据）
    cy_link_nodes = """
    MATCH (u:TextUnit {name: $textunit})
    MATCH (n)
    WHERE n.doc_id = $doc_id OR n.document_id = $doc_id OR n.ref_doc_id = $doc_id
    MERGE (n)-[:FROM_SOURCE]->(u)
    RETURN count(n) AS linked
    """
    link_rows = graph_store.structured_query(cy_link_nodes, {"textunit": info.textunit_id, "doc_id": info.doc_id})

    # 3) raw Event + EVIDENCED_BY(TextUnit)
    cy_event = """
    MERGE (e:Event {event_id: $event_id})
    SET e.channel = $ch,
        e.agent_name = $agent_name,
        e.virtual_time = $virtual_time,
        e.turn_id = $turn_id,
        e.session_id = $session_id,
        e.session_time_raw = $session_time_raw,
        e.session_time_iso = $session_time_iso,
        e.event_timestamp = coalesce(e.event_timestamp, $session_time_iso),
        e.updated_at = $real_time
    WITH e
    MATCH (u:TextUnit {name: $textunit})
    MERGE (e)-[:EVIDENCED_BY]->(u)
    RETURN e.event_id AS event_id
    """
    evt_rows = graph_store.structured_query(
        cy_event,
        {
            "event_id": info.event_id,
            "ch": info.channel,
            "agent_name": info.agent_name,
            "virtual_time": info.virtual_time,
            "turn_id": info.turn_id,
            "session_id": info.session_id,
            "session_time_raw": info.session_time_raw,
            "session_time_iso": info.session_time_iso,
            "real_time": real_time,
            "textunit": info.textunit_id,
        },
    )

    # 4) doc 相关关系补齐 channel/event_id/session_time_iso
    cy_rels = """
    MATCH (n)
    WHERE n.doc_id = $doc_id OR n.document_id = $doc_id OR n.ref_doc_id = $doc_id
    WITH collect(elementId(n)) AS ids
    MATCH (a)-[r]->(b)
    WHERE elementId(a) IN ids OR elementId(b) IN ids
    SET
        r.channel = coalesce(r.channel, $ch),
        r.agent_name = coalesce(r.agent_name, $agent_name),
        r.turn_id = coalesce(r.turn_id, $turn_id),
        r.virtual_time = coalesce(r.virtual_time, $virtual_time),
        r.confidence = coalesce(r.confidence, $raw_conf),
        r.event_id = coalesce(r.event_id, $event_id),
        r.session_time_iso = coalesce(r.session_time_iso, $session_time_iso),
        r.event_timestamp = coalesce(r.event_timestamp, $session_time_iso, $session_time_raw, $virtual_time),
        r.source_of_belief = coalesce(r.source_of_belief, 'raw_extraction'),
        r.knowledge_type = coalesce(r.knowledge_type, 'observed_fact'),
        r.belief_key = coalesce(
          r.belief_key,
          $ch + ':' + $agent_name + ':' +
          coalesce(labels(a)[0], 'Concept') + ':' + toString(a.name) + ':' + type(r) + ':' +
          coalesce(labels(b)[0], 'Concept') + ':' + toString(b.name)
        ),
        r.slot_key = coalesce(
          r.slot_key,
          CASE
            WHEN type(r) IN $update_types THEN coalesce(labels(a)[0], 'Concept') + ':' + toString(a.name) + ':' + type(r)
            ELSE toString(a.name) + ':' + type(r)
          END
        )
    RETURN count(r) AS updated
    """
    rel_rows = graph_store.structured_query(
        cy_rels,
        {
            "doc_id": info.doc_id,
            "ch": info.channel,
            "agent_name": info.agent_name,
            "turn_id": info.turn_id,
            "virtual_time": info.virtual_time,
            "event_id": info.event_id,
            "session_time_raw": info.session_time_raw,
            "session_time_iso": info.session_time_iso,
            "update_types": list(_UPDATE_REL_TYPES),
            "raw_conf": float(RAW_REL_CONFIDENCE),
        },
    )

    # 4.1) Normalize visit-like relations into VISITED for stable retrieval.
    cy_visit_rels = """
    MATCH (n)
    WHERE n.doc_id = $doc_id OR n.document_id = $doc_id OR n.ref_doc_id = $doc_id
    WITH collect(elementId(n)) AS ids
    MATCH (a)-[r]->(b)
    WHERE (elementId(a) IN ids OR elementId(b) IN ids)
      AND type(r) IN $visit_types
    MERGE (a)-[v:VISITED]->(b)
    SET
        v.rel_alias = coalesce(v.rel_alias, type(r)),
        v.channel = coalesce(v.channel, $ch),
        v.agent_name = coalesce(v.agent_name, $agent_name),
        v.turn_id = coalesce(v.turn_id, $turn_id),
        v.virtual_time = coalesce(v.virtual_time, $virtual_time),
        v.confidence = coalesce(v.confidence, coalesce(r.confidence, $raw_conf)),
        v.event_id = coalesce(v.event_id, coalesce(r.event_id, $event_id)),
        v.session_time_iso = coalesce(v.session_time_iso, $session_time_iso),
        v.event_timestamp = coalesce(v.event_timestamp, r.event_timestamp, $session_time_iso, $session_time_raw, $virtual_time),
        v.source_of_belief = coalesce(v.source_of_belief, 'raw_extraction'),
        v.knowledge_type = coalesce(v.knowledge_type, 'observed_fact'),
        v.belief_key = coalesce(
          v.belief_key,
          $ch + ':' + $agent_name + ':' +
          coalesce(labels(a)[0], 'Concept') + ':' + toString(a.name) + ':' + 'VISITED' + ':' +
          coalesce(labels(b)[0], 'Concept') + ':' + toString(b.name)
        ),
        v.slot_key = coalesce(v.slot_key, toString(a.name) + ':' + 'VISITED')
    RETURN count(v) AS visited_cnt
    """
    graph_store.structured_query(
        cy_visit_rels,
        {
            "doc_id": info.doc_id,
            "ch": info.channel,
            "agent_name": info.agent_name,
            "turn_id": info.turn_id,
            "virtual_time": info.virtual_time,
            "event_id": info.event_id,
            "session_time_raw": info.session_time_raw,
            "session_time_iso": info.session_time_iso,
            "raw_conf": float(RAW_REL_CONFIDENCE),
            "visit_types": ["VISIT", "VISITED", "WENT_TO", "ATTENDED", "TOURED", "TOUR", "GUIDED_TOUR", "WAS_AT"],
        },
    )

    # 5) Raw relationships -> Fact nodes (for V2 Fact retrieval)
    cy_raw_facts = """
    MATCH (n)
    WHERE n.doc_id = $doc_id OR n.document_id = $doc_id OR n.ref_doc_id = $doc_id
    WITH collect(elementId(n)) AS ids
    MATCH (a)-[r]->(b)
    WHERE elementId(a) IN ids OR elementId(b) IN ids
      AND type(r) <> 'FROM_SOURCE'
    WITH a, b, r,
         labels(a) AS la, labels(b) AS lb,
         coalesce(r.channel, $ch) AS ch,
         coalesce(r.agent_name, $agent_name) AS ag
    WITH a, b, r, la, lb, ch, ag,
         coalesce(la[0], 'Concept') AS a_label,
         coalesce(lb[0], 'Concept') AS b_label
    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
    WITH a, b, r, a_label, b_label, ch, ag,
         a_label + ':' + a.name + ':' + type(r) + ':' + b_label + ':' + b.name AS base_key
    WITH a, b, r, base_key, ch, ag,
         ch + ':' + ag + ':' + base_key AS belief_key

    MERGE (f:Fact {belief_key: belief_key})
    ON CREATE SET
        f.type = type(r),
        f.slot_key = coalesce(
          r.slot_key,
          CASE
            WHEN type(r) IN $update_types THEN coalesce(labels(a)[0], 'Concept') + ':' + toString(a.name) + ':' + type(r)
            ELSE toString(a.name) + ':' + type(r)
          END
        ),
        f.channel = ch,
        f.agent_name = ag,
        f.virtual_time = coalesce(r.virtual_time, $virtual_time),
        f.turn_id = coalesce(r.turn_id, $turn_id),
        f.recorded_turn_id = coalesce(r.turn_id, $turn_id),
        f.event_turn_offset = 0,
        f.event_turn_id = coalesce(r.turn_id, $turn_id),
        f.session_time = coalesce($session_time_iso, $session_time_raw),
        f.event_timestamp = coalesce(r.event_timestamp, $session_time_iso, $session_time_raw, $virtual_time),
        f.time_text = r.time_text,
        f.source_of_belief = coalesce(r.source_of_belief, 'raw_extraction'),
        f.knowledge_type = coalesce(r.knowledge_type, 'observed_fact'),
        f.confidence = coalesce(r.confidence, $raw_conf),
        f.created_at = $real_time,
        f.first_turn_id = coalesce(r.turn_id, $turn_id),
        f.first_event_time = coalesce(r.event_timestamp, $session_time_iso, $session_time_raw, $virtual_time),
        f.last_turn_id = coalesce(r.turn_id, $turn_id),
        f.last_event_time = coalesce(r.event_timestamp, $session_time_iso, $session_time_raw, $virtual_time),
        f.mention_count = 1,
        f.turn_history = [coalesce(r.turn_id, $turn_id)]
    ON MATCH SET
        f.confidence = CASE
            WHEN coalesce(r.confidence, 0.0) > coalesce(f.confidence, 0.0)
            THEN r.confidence ELSE f.confidence END,
        f.updated_at = $real_time,
        f.last_turn_id = coalesce(r.turn_id, $turn_id),
        f.last_event_time = coalesce(r.event_timestamp, $session_time_iso, $session_time_raw, $virtual_time),
        f.time_text = coalesce(r.time_text, f.time_text),
        f.mention_count = coalesce(f.mention_count, 0) + 1,
        f.turn_history = (coalesce(f.turn_history, []) + [coalesce(r.turn_id, $turn_id)])[-10..]
    WITH f, a, b, r
    MERGE (f)-[:SUBJECT]->(a)
    MERGE (f)-[:OBJECT]->(b)
    WITH f, r
    MATCH (e:Event {event_id: coalesce(r.event_id, $event_id)})
    MERGE (e)-[:HAS_FACT]->(f)
    RETURN count(f) AS fact_cnt
    """
    fact_rows = graph_store.structured_query(
        cy_raw_facts,
        {
            "doc_id": info.doc_id,
            "ch": info.channel,
            "agent_name": info.agent_name,
            "turn_id": info.turn_id,
            "virtual_time": info.virtual_time,
            "event_id": info.event_id,
            "session_time_raw": info.session_time_raw,
            "session_time_iso": info.session_time_iso,
            "real_time": real_time,
            "update_types": list(_UPDATE_REL_TYPES),
            "raw_conf": float(RAW_REL_CONFIDENCE),
        },
    )

    # 6) Debug: raw relation type stats (helps locate noisy relations)
    rel_stats = []
    if _DEBUG_INGEST:
        cy_rel_stats = """
        MATCH (n)
        WHERE n.doc_id = $doc_id OR n.document_id = $doc_id OR n.ref_doc_id = $doc_id
        WITH collect(elementId(n)) AS ids
        MATCH (a)-[r]->(b)
        WHERE elementId(a) IN ids OR elementId(b) IN ids
          AND type(r) <> 'FROM_SOURCE'
        RETURN type(r) AS rel_type, count(r) AS cnt
        ORDER BY cnt DESC
        """
        rel_stats = graph_store.structured_query(cy_rel_stats, {"doc_id": info.doc_id}) or []
    if _DEBUG_INGEST:
        node_cnt = node_rows[0].get("updated") if node_rows else 0
        tu_name = tu_rows[0].get("name") if tu_rows else "unknown"
        link_cnt = link_rows[0].get("linked") if link_rows else 0
        evt_id = evt_rows[0].get("event_id") if evt_rows else "unknown"
        rel_cnt = rel_rows[0].get("updated") if rel_rows else 0
        fact_cnt = fact_rows[0].get("fact_cnt") if fact_rows else 0
        logger.info(
            "[ingest_longmemoryeval][debug] postprocess "
            f"doc_id={info.doc_id} node_updates={node_cnt} textunit={tu_name} "
            f"link_cnt={link_cnt} event_id={evt_id} rel_updates={rel_cnt} fact_nodes={fact_cnt}"
        )
        if rel_stats:
            logger.info(
                "[ingest_longmemoryeval][debug] rel_type_counts "
                + ", ".join([f"{r.get('rel_type')}={r.get('cnt')}" for r in rel_stats[:12]])
            )


def iter_windows(total: int, batch_size: int, overlap: int) -> Iterable[Tuple[int, int]]:
    """
    total 个 session（按 turn_id 顺序排列后），生成窗口 [s:e)；
    overlap 表示相邻窗口共享的 session 数。
    """
    if total <= 0:
        return
    bs = max(1, int(batch_size))
    ov = max(0, int(overlap))
    step = max(1, bs - ov)
    s = 0
    while s < total:
        e = min(total, s + bs)
        yield s, e
        if e >= total:
            break
        s += step


def ingest_raw_sessions_window(
    *,
    agent_name: str,
    sessions_window: List[List[Dict[str, Any]]],
    session_ids_window: List[str],
    session_dates_window: List[str],
    turn_ids_window: List[int],
    include_assistant: bool = True,
    # 以下参数保留接口兼容性，但不再使用（去掉LlamaIndex提取）
    chunk_size: int = 900,
    chunk_overlap: int = 180,
    num_workers: int = 4,
    max_paths_per_chunk: int = 18,
    show_progress: bool = False,
) -> List[RawDocInfo]:
    """
    RAW通道入图（支持分块）：
    - 如果启用分块（ENABLE_TEXTUNIT_CHUNKING=1），将长 session 分成多个 TextUnit
    - 每个 chunk 独立向量化，提高检索粒度
    - 结构化提取由CONSOLIDATED通道负责
    """
    if not (len(sessions_window) == len(session_ids_window) == len(session_dates_window) == len(turn_ids_window)):
        raise ValueError("ingest_raw_sessions_window: window lists length mismatch")

    # 1) 获取 embedding model 和 Neo4j 连接
    embed_model = get_embedding_model()
    
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # 2) 构造 docs 信息
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    safe_agent = _safe_agent(agent_name)
    real_time = datetime.datetime.now().isoformat()

    infos: List[RawDocInfo] = []
    
    # 用于存储分块信息：textunit_id -> (content, chunk_id, total_chunks)
    chunk_meta: Dict[str, Tuple[str, int, int]] = {}

    for sess_turns, sid, sdate, turn_id in zip(sessions_window, session_ids_window, session_dates_window, turn_ids_window):
        turn_id = int(turn_id)
        virtual_time = f"TURN_{turn_id}"
        sraw, siso = _parse_dataset_time(sdate)
        
        # 完整的 session 文本（用于 __Document__）
        session_text = build_session_text(sess_turns, include_assistant=include_assistant)
        if not session_text:
            continue

        doc_id = f"raw_{safe_agent}_{ts}_{turn_id}"
        
        # 判断是否启用分块
        if ENABLE_TEXTUNIT_CHUNKING:
            # 分块处理
            chunks = chunk_session_turns(sess_turns, include_assistant=include_assistant)
            
            if not chunks:
                # 分块失败，回退到整个 session
                chunks = [ChunkInfo(
                    chunk_id=1,
                    total_chunks=1,
                    content=session_text,
                    start_turn_idx=0,
                    end_turn_idx=len(sess_turns) - 1,
                )]
            
            if _DEBUG_INGEST:
                logger.info(
                    f"[ingest_longmemoryeval][chunking] turn_id={turn_id} "
                    f"session_len={len(session_text)} chunks={len(chunks)} "
                    f"chunk_lens=[{', '.join(str(len(c.content)) for c in chunks)}]"
                )
            
            # 为每个 chunk 创建 RawDocInfo
            for chunk in chunks:
                if chunk.total_chunks > 1:
                    textunit_id = f"unit_TURN_{turn_id}_chunk_{chunk.chunk_id}"
                else:
                    textunit_id = f"unit_TURN_{turn_id}"
                
                event_id = f"evt:raw:{safe_agent}:{virtual_time}:{textunit_id}"
                
                infos.append(
                    RawDocInfo(
                        doc_id=doc_id,
                        textunit_id=textunit_id,
                        event_id=event_id,
                        agent_name=agent_name,
                        channel=RAW,
                        virtual_time=virtual_time,
                        turn_id=turn_id,
                        session_id=sid,
                        session_time_raw=sraw,
                        session_time_iso=siso,
                        session_text=chunk.content,  # 使用 chunk 内容
                        session_turns=sess_turns,
                    )
                )
                
                # 记录分块信息
                chunk_meta[textunit_id] = (chunk.content, chunk.chunk_id, chunk.total_chunks)
        else:
            # 不分块，使用整个 session
            textunit_id = f"unit_TURN_{turn_id}"
            event_id = f"evt:raw:{safe_agent}:{virtual_time}:{textunit_id}"
            
            if _DEBUG_INGEST:
                logger.info(
                    "[ingest_longmemoryeval][debug] session "
                    f"turn_id={turn_id} session_id={sid} text_len={len(session_text)} date_raw={sdate}"
                )
            
            infos.append(
                RawDocInfo(
                    doc_id=doc_id,
                    textunit_id=textunit_id,
                    event_id=event_id,
                    agent_name=agent_name,
                    channel=RAW,
                    virtual_time=virtual_time,
                    turn_id=turn_id,
                    session_id=sid,
                    session_time_raw=sraw,
                    session_time_iso=siso,
                    session_text=session_text,
                    session_turns=sess_turns,
                )
            )
            chunk_meta[textunit_id] = (session_text, 1, 1)

    if not infos:
        driver.close()
        return []

    # 统计分块信息
    total_textunits = len(infos)
    chunked_sessions = len(set(i.turn_id for i in infos))
    logger.info(
        f"[ingest_longmemoryeval] RAW 窗口入图：agent={agent_name} "
        f"sessions={chunked_sessions} textunits={total_textunits} "
        f"turn_id_range=[{min(turn_ids_window)}..{max(turn_ids_window)}] "
        f"chunking={'enabled' if ENABLE_TEXTUNIT_CHUNKING else 'disabled'}"
    )

    # 3) 批量计算 embedding
    emb_map: Dict[str, Optional[List[float]]] = {}
    try:
        texts = [(i.session_text or "").strip() for i in infos]
        non_empty_idx = [k for k, t in enumerate(texts) if t]
        if non_empty_idx:
            batch_texts = [texts[k] for k in non_empty_idx]
            if embed_model and hasattr(embed_model, "embed_documents"):
                batch_embs = embed_model.embed_documents(batch_texts) or []
            else:
                batch_embs = [embed_model.embed_query(x) for x in batch_texts] if embed_model else []

            # 验证 embedding 结果
            valid_count = 0
            for pos, k in enumerate(non_empty_idx):
                if pos < len(batch_embs) and batch_embs[pos] and len(batch_embs[pos]) > 0:
                    emb_map[infos[k].textunit_id] = batch_embs[pos]
                    valid_count += 1
                else:
                    emb_map[infos[k].textunit_id] = None
                    logger.warning(f"[ingest_longmemoryeval] embedding empty for textunit {infos[k].textunit_id}")
            
            logger.info(f"[ingest_longmemoryeval] RAW embedding: {valid_count}/{len(non_empty_idx)} valid")

        for k, t in enumerate(texts):
            if not t:
                emb_map[infos[k].textunit_id] = None

    except Exception as e:
        logger.warning(f"[ingest_longmemoryeval] raw TextUnit batch embedding failed: {e}")
        for i in infos:
            emb_map[i.textunit_id] = None

    # 4) 写入 Neo4j
    # 记录已创建的 Document（避免重复创建）
    created_docs: set = set()
    
    with driver.session(database="neo4j") as session:
        for info in infos:
            tu_emb = emb_map.get(info.textunit_id)
            chunk_content, chunk_id, total_chunks = chunk_meta.get(info.textunit_id, (info.session_text, 1, 1))

            # 4.1 创建 TextUnit（原文 + embedding + 分块信息）
            session.run(
                """
                MERGE (u:TextUnit {name: $name})
                SET u.content = $content,
                    u.doc_id = $doc_id,
                    u.virtual_time = $virtual_time,
                    u.turn_id = $turn_id,
                    u.chunk_id = $chunk_id,
                    u.total_chunks = $total_chunks,
                    u.real_time = $real_time,
                    u.embedding = $embedding,
                    u.agent_name = $agent_name,
                    u.channel = $channel,
                    u.session_id = $session_id,
                    u.session_time_raw = $session_time_raw,
                    u.session_time_iso = $session_time_iso
                """,
                {
                    "name": info.textunit_id,
                    "content": chunk_content,
                    "doc_id": info.doc_id,
                    "virtual_time": info.virtual_time,
                    "turn_id": info.turn_id,
                    "chunk_id": chunk_id,
                    "total_chunks": total_chunks,
                    "real_time": real_time,
                    "embedding": tu_emb,
                    "agent_name": agent_name,
                    "channel": RAW,
                    "session_id": info.session_id,
                    "session_time_raw": info.session_time_raw,
                    "session_time_iso": info.session_time_iso,
                },
            )

            # 4.2 创建 Event 并连接 TextUnit
            session.run(
                """
                MERGE (e:Event {event_id: $event_id})
                SET e.channel = $channel,
                    e.agent_name = $agent_name,
                    e.virtual_time = $virtual_time,
                    e.turn_id = $turn_id,
                    e.event_timestamp = $event_timestamp,
                    e.updated_at = $updated_at,
                    e.session_id = $session_id,
                    e.session_time_raw = $session_time_raw,
                    e.session_time_iso = $session_time_iso,
                    e.event_time_iso = CASE
                        WHEN $session_time_iso <> '' AND $session_time_iso <> 'unknown' THEN $session_time_iso
                        WHEN $session_time_raw <> '' AND $session_time_raw <> 'unknown' THEN $session_time_raw
                        ELSE $virtual_time
                    END
                WITH e
                MATCH (u:TextUnit {name: $textunit})
                MERGE (e)-[:EVIDENCED_BY]->(u)
                """,
                {
                    "event_id": info.event_id,
                    "channel": RAW,
                    "agent_name": agent_name,
                    "virtual_time": info.virtual_time,
                    "turn_id": info.turn_id,
                    "event_timestamp": info.session_time_iso if info.session_time_iso != "unknown" else info.virtual_time,
                    "updated_at": real_time,
                    "textunit": info.textunit_id,
                    "session_id": info.session_id,
                    "session_time_raw": info.session_time_raw,
                    "session_time_iso": info.session_time_iso,
                },
            )

            # 4.3 创建 __Document__（每个 session 只创建一次）
            if info.doc_id not in created_docs:
                # 获取完整的 session 文本
                full_session_text = build_session_text(info.session_turns, include_assistant=include_assistant)
                
                session.run(
                    """
                    MERGE (d:__Document__ {id: $doc_id})
                    ON CREATE SET d.title = $title, d.raw_content = $raw
                    SET d.agent_name = $agent_name,
                        d.updated_at = $updated_at,
                        d.name = coalesce(d.name, $title),
                        d.session_id = $session_id,
                        d.session_time_raw = $session_time_raw,
                        d.session_time_iso = $session_time_iso
                    """,
                    {
                        "doc_id": info.doc_id,
                        "title": info.session_id or info.doc_id,
                        "raw": full_session_text,
                        "agent_name": agent_name,
                        "updated_at": real_time,
                        "session_id": info.session_id,
                        "session_time_raw": info.session_time_raw,
                        "session_time_iso": info.session_time_iso,
                    },
                )
                created_docs.add(info.doc_id)

            # 4.4 创建 __Chunk__ 并建立关系
            neo4j_chunk_id = f"{info.doc_id}:{info.textunit_id}"
            session.run(
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
                    "chunk_id": neo4j_chunk_id,
                    "doc_id": info.doc_id,
                    "text": chunk_content,
                    "agent_name": agent_name,
                    "channel": RAW,
                    "virtual_time": info.virtual_time,
                    "turn_id": info.turn_id,
                    "session_id": info.session_id,
                    "session_time_raw": info.session_time_raw,
                    "session_time_iso": info.session_time_iso,
                    "textunit_id": info.textunit_id,
                },
            )

            if _DEBUG_INGEST:
                logger.info(
                    f"[ingest_longmemoryeval][debug] RAW written: turn_id={info.turn_id} "
                    f"textunit={info.textunit_id} chunk={chunk_id}/{total_chunks}"
                )

    driver.close()
    return infos


# =========================
# CONSOLIDATED：从 raw 子图轻量巩固
# =========================
def fetch_raw_triples_by_doc_id(ltss, doc_id: str, limit: int = 220) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (n)
    WHERE n.doc_id = $doc_id OR n.document_id = $doc_id OR n.ref_doc_id = $doc_id
    WITH collect(elementId(n)) AS ids
    MATCH (a)-[r]->(b)
    WHERE (elementId(a) IN ids OR elementId(b) IN ids)
      AND type(r) <> 'FROM_SOURCE'
    WITH a, r, b,
         labels(a) AS la, labels(b) AS lb
    WHERE NOT 'Chunk' IN la
      AND NOT 'TextUnit' IN la
      AND NOT 'Chunk' IN lb
      AND NOT 'TextUnit' IN lb
    RETURN
      la AS a_labels,
      a.name AS a_name,
      type(r) AS rel_type,
      r.confidence AS confidence,
      r.source_of_belief AS source_of_belief,
      r.event_timestamp AS event_timestamp,
      lb AS b_labels,
      b.name AS b_name
    LIMIT $limit
    """
    rows = ltss.query_graph(cypher, {"doc_id": doc_id, "limit": int(limit)})
    return rows or []


def format_raw_triples(rows: List[Dict[str, Any]]) -> str:
    lines = []
    for row in rows or []:
        a = (row.get("a_name") or "").strip()
        b = (row.get("b_name") or "").strip()
        rel = (row.get("rel_type") or "").strip()
        if not a or not b or not rel:
            continue

        la = row.get("a_labels") or []
        lb = row.get("b_labels") or []
        a_label = la[0] if isinstance(la, list) and la else "Concept"
        b_label = lb[0] if isinstance(lb, list) and lb else "Concept"

        conf = row.get("confidence", None)
        try:
            conf_s = f"{float(conf):.2f}" if conf is not None else "1.00"
        except Exception:
            conf_s = "1.00"

        src = row.get("source_of_belief") or ""
        ts = row.get("event_timestamp") or ""

        meta = []
        if conf_s:
            meta.append(f"conf={conf_s}")
        if src:
            meta.append(f"src={src}")
        if ts and str(ts).lower() != "unknown":
            meta.append(f"time={ts}")

        meta_str = (" | " + ",".join(meta)) if meta else ""
        lines.append(f"({a_label}:{a}) -[{rel}]-> ({b_label}:{b}){meta_str}")

    return "\n".join(lines) if lines else "(No usable RAW triples found.)"


def _parse_json_dict(raw: str) -> Dict[str, Any]:
    s = (raw or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        from json_repair import repair_json

        obj = repair_json(s, return_objects=True)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s2 = s[start : end + 1]
        obj2 = json.loads(s2)
        if isinstance(obj2, dict):
            return obj2

    raise ValueError("Failed to parse JSON dict from LLM output.")


def build_consolidation_prompt(*, virtual_time: str, raw_triples_str: str, time_snippets: str = "") -> str:
    """
    从 raw triples + time snippets 做轻量巩固。
    recorded_turn_id 从 TURN_x 解析；解析失败写 -1
    """
    recorded_turn_id = -1
    try:
        m = re.search(r"(\d+)", str(virtual_time or ""))
        recorded_turn_id = int(m.group(1)) if m else -1
    except Exception:
        recorded_turn_id = -1

    return f"""
You are consolidating a cognitive agent's memory based on:
1) RAW knowledge graph triples (may miss some time phrases)
2) TIME-SNIPPETS from original dialogue (to recover relative time like yesterday/today)

You MUST output a STRICT JSON object with keys:
- facts: array of HARD FACT strings (numbers, dates, durations, prices, proper names; keep exact wording)
- insights: array of LOGICAL INSIGHT strings (social relations, patterns, causal links, contradiction resolution)
- nodes: array of nodes: {{name: str, label: str, properties: object}}
- relationships: array of relationships:
  {{source_node_name: str, source_node_label: str, target_node_name: str, target_node_label: str, type: str, properties: object}}

CRITICAL TIME RULES (VERY IMPORTANT):
A) The reference "recorded time" is the current virtual turn: {virtual_time}
   You MUST set on EVERY Event node:
     properties.recorded_turn_id = {recorded_turn_id}
B) If you create an Event node AND the text implies an event time using relative words
   (today/yesterday/tomorrow/last week/next week/this morning/etc.),
   you MUST set on that Event node:
     - properties.event_time_text: string (e.g., "yesterday", "today", "last week")
     - properties.event_turn_offset: integer offset relative to recorded_turn_id.
C) If there is NO relative time info, OMIT event_time_text and event_turn_offset (do NOT guess).

GENERAL RULES:
1) Output ONLY valid JSON. No markdown.
2) DO NOT hallucinate. Only use information supported by RAW triples or TIME-SNIPPETS.
3) Node labels must be one of: Person, Organization, Location, Object, Event, Date, Value, Concept.
4) Relationship type MUST be UPPERCASE.
5) Keep the consolidated output SMALL: prefer <= 12 relationships, <= 15 nodes.

TIME-SNIPPETS:
{time_snippets if time_snippets else "(none)"}

RAW triples (evidence):
{raw_triples_str}
""".strip()


def _invoke_json_with_retry(
    llm,
    prompt: str,
    *,
    max_retries: int = 3,
    base_backoff_s: float = 1.0,
    max_backoff_s: float = 12.0,
    min_interval_s: float = 0.0,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            if min_interval_s and min_interval_s > 0:
                time.sleep(float(min_interval_s))
            return invoke_json(llm, prompt)
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            backoff = min(max_backoff_s, base_backoff_s * (2 ** (attempt - 1)))
            backoff = backoff * (0.75 + 0.5 * random.random())  # jitter
            logger.warning(
                f"[ingest_longmemoryeval] invoke_json failed (attempt={attempt}/{max_retries}): {e}; sleep {backoff:.2f}s"
            )
            time.sleep(backoff)
    raise RuntimeError(f"invoke_json failed after {max_retries} retries: {last_err}")


def consolidate_docs_to_ltss(
    *,
    ltss,
    agent_name: str,
    doc_infos: List[RawDocInfo],
    consolidation_llm=None,
    embedding_model=None,
    max_workers: int = 6,
    max_retries: int = 3,
    base_backoff_s: float = 1.0,
    min_interval_s: float = 0.0,
):
    """
    全 task 统一做：
      读取 raw triples -> 并行 LLM -> 按 turn_id 严格升序串行写入 consolidated
    """
    if not doc_infos:
        return

    llm = consolidation_llm or get_llm()
    emb = embedding_model or get_embedding_model()

    # 1) 先准备 prompt（串行）
    jobs: List[Tuple[int, RawDocInfo, str]] = []
    for info in doc_infos:
        try:
            rows = fetch_raw_triples_by_doc_id(ltss, doc_id=info.doc_id, limit=220)
            raw_triples_str = format_raw_triples(rows)
            time_snips = extract_time_snippets(info.session_text or "")
            prompt = build_consolidation_prompt(
                virtual_time=info.virtual_time,
                raw_triples_str=raw_triples_str,
                time_snippets=time_snips,
            )
            jobs.append((info.turn_id, info, prompt))
        except Exception as e:
            logger.error(
                f"[ingest_longmemoryeval] prepare consolidation prompt failed for doc_id={info.doc_id}: {e}",
                exc_info=True,
            )

    if not jobs:
        return

    # 2) 并行 invoke_json
    results: Dict[int, Tuple[RawDocInfo, KnowledgeGraphExtraction]] = {}

    def _worker(turn_id: int, info: RawDocInfo, prompt: str):
        kg_raw = _invoke_json_with_retry(
            llm,
            prompt,
            max_retries=max_retries,
            base_backoff_s=base_backoff_s,
            min_interval_s=min_interval_s,
        )
        obj = _parse_json_dict((kg_raw or "").strip())
        structured = KnowledgeGraphExtraction(**obj)
        return turn_id, info, structured

    use_workers = max(1, int(max_workers))
    logger.info(f"[ingest_longmemoryeval] CONSOLIDATED 并行巩固调用结构化处理：docs={len(jobs)} max_workers={use_workers}")

    with ThreadPoolExecutor(max_workers=use_workers) as ex:
        futs = [ex.submit(_worker, tid, info, prompt) for (tid, info, prompt) in jobs]
        for fut in as_completed(futs):
            try:
                tid, info, structured = fut.result()
                results[int(tid)] = (info, structured)
            except Exception as e:
                logger.error(f"[ingest_longmemoryeval] consolidation worker failed: {e}", exc_info=True)

    # 3) 严格按 turn_id 升序串行写回 Neo4j
    for tid in sorted(results.keys()):
        info, structured = results[tid]
        try:
            write_consolidation_result(
                ltss=ltss,
                embedding_model=emb,
                agent_name=agent_name,
                memories_str=info.session_text or "",
                structured_response=structured,
                current_time=info.virtual_time,
                channel=CONSOLIDATED,
                doc_id=info.doc_id,
                session_id=info.session_id,
                session_time_raw=info.session_time_raw,
                session_time_iso=info.session_time_iso,
            )
            logger.info(f"[ingest_longmemoryeval] CONSOLIDATED 写入完成：turn_id={tid} doc_id={info.doc_id}")
        except Exception as e:
            logger.error(
                f"[ingest_longmemoryeval] write_consolidation_result failed: turn_id={tid} doc_id={info.doc_id}: {e}",
                exc_info=True,
            )


def consolidate_docs_from_original_text(
    *,
    ltss,
    agent_name: str,
    doc_infos: List[RawDocInfo],
    consolidation_llm=None,
    embedding_model=None,
    max_workers: int = 6,
    include_assistant: bool = True,
    max_chars_per_chunk: int = 4000,
) -> None:
    """
    Directly read original session text for consolidation:
      - session-level parallelism (worker per session)
      - chunk within session (by character count)
      - merge chunk outputs, then write once per session (ordered by turn_id)
    """
    if not doc_infos:
        return

    llm = consolidation_llm or get_llm()
    emb = embedding_model or get_embedding_model()

    jobs: List[Tuple[int, RawDocInfo]] = []
    for info in doc_infos:
        jobs.append((int(info.turn_id), info))

    results: Dict[int, Tuple[RawDocInfo, KnowledgeGraphExtraction]] = {}

    def _worker(turn_id: int, info: RawDocInfo):
        if not info.session_turns:
            return None
        res = consolidate_original_session(
            session_turns=info.session_turns,
            virtual_time=info.virtual_time,
            session_time_raw=info.session_time_raw,
            session_time_iso=info.session_time_iso,
            include_assistant=include_assistant,
            llm=llm,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        return (turn_id, info, res)

    use_workers = max(1, int(max_workers))
    logger.info(
        f"[ingest_longmemoryeval] ORIGINAL_TEXT 并行巩固：docs={len(jobs)} max_workers={use_workers} "
        f"max_chars_per_chunk={max_chars_per_chunk}"
    )

    with ThreadPoolExecutor(max_workers=use_workers) as ex:
        futs = [ex.submit(_worker, tid, info) for (tid, info) in jobs]
        for fut in as_completed(futs):
            try:
                out = fut.result()
                if not out:
                    continue
                tid, info, res = out
                if res and res.structured:
                    results[int(tid)] = (info, res.structured)
            except Exception as e:
                logger.error(f"[ingest_longmemoryeval] original_text worker failed: {e}", exc_info=True)

    for tid in sorted(results.keys()):
        info, structured = results[tid]
        try:
            write_consolidation_result(
                ltss=ltss,
                embedding_model=emb,
                agent_name=agent_name,
                memories_str=info.session_text or "",
                structured_response=structured,
                current_time=info.virtual_time,
                channel=CONSOLIDATED,
                doc_id=info.doc_id,
                session_id=info.session_id,
                session_time_raw=info.session_time_raw,
                session_time_iso=info.session_time_iso,
            )
            logger.info(f"[ingest_longmemoryeval] ORIGINAL_TEXT 写入完成：turn_id={tid} doc_id={info.doc_id}")
        except Exception as e:
            logger.error(
                f"[ingest_longmemoryeval] original_text write failed: turn_id={tid} doc_id={info.doc_id}: {e}",
                exc_info=True,
            )


# =========================
# 对外入口：注入一个 task 的 haystack_sessions
# =========================
def ingest_longmemoryeval_sample(
    *,
    agent,
    case: Dict[str, Any],
    batch_size: Optional[int] = None,
    overlap_sessions: int = 0,  # ✅ 特化测试默认不 overlap，避免重复写入
    include_assistant: bool = True,
    show_progress: bool = False,
    chunk_size: int = 900,
    chunk_overlap: int = 180,
    num_workers: int = 4,
    # consolidated 并行参数
    consolidation_max_workers: int = 6,
    consolidation_max_retries: int = 3,
    consolidation_base_backoff_s: float = 1.0,
    consolidation_min_interval_s: float = 0.0,
    # consolidation mode
    consolidation_mode: str = "raw_triples",  # raw_triples | original_text
    # original-text chunking
    original_max_chars_per_chunk: int = 4000,
) -> Dict[str, Any]:
    """
    注入一个 task：
    - 根据 haystack_dates 排序，定义 turn_id=1..N（数据集特化，稳定）
    - question_date 映射到 question_turn_id（建议你回答时用 TURN_{question_turn_id}）
    - RAW：按窗口批量入图
    - CONSOLIDATED：全 task 并行生成 + 严格 turn_id 串行写回

    返回：question_turn_id、doc_infos 等信息，方便你的测试脚本使用。
    """
    sessions: List[List[Dict[str, Any]]] = case.get("haystack_sessions", []) or []
    session_ids: List[str] = case.get("haystack_session_ids", []) or []
    session_dates: List[str] = case.get("haystack_dates", []) or []
    question_date: str = (case.get("question_date") or "").strip()

    if not sessions:
        logger.warning("[ingest_longmemoryeval] 样本没有 haystack_sessions，跳过注入。")
        return {"question_turn_id": 1, "num_sessions": 0, "doc_infos": []}

    ltss = getattr(getattr(agent, "memory", None), "ltss", None) or getattr(agent, "ltss", None)
    if ltss is None:
        raise RuntimeError("agent.memory.ltss 不存在，无法写 Neo4j。")

    agent_name = getattr(agent, "name", None) or getattr(agent, "character_name", None) or "agent"

    # 1) turn_id 规划（按 haystack_dates 排序）
    plan = build_task_turn_plan(sessions, session_ids, session_dates)
    ordered_sessions = plan["ordered_sessions"]
    ordered_ids = plan["ordered_session_ids"]
    ordered_dates = plan["ordered_session_dates"]
    n = len(ordered_sessions)
    ordered_turn_ids = list(range(1, n + 1))  # ✅ turn_id=1..N

    # 2) question_turn_id
    question_turn_id = compute_question_turn_id(question_date, ordered_dates) if question_date else (n + 1)
    if _DEBUG_INGEST:
        parsed_cnt = sum(1 for d in ordered_dates if _try_parse_dataset_dt(d) is not None)
        logger.info(
            "[ingest_longmemoryeval][debug] time_plan "
            f"sessions={n} parsed_dates={parsed_cnt} question_date={question_date} "
            f"question_turn_id={question_turn_id}"
        )
        for i in range(min(5, n)):
            logger.info(
                "[ingest_longmemoryeval][debug] ordered_session "
                f"idx={i} session_id={ordered_ids[i]} date={ordered_dates[i]}"
            )

    bs = int(batch_size or memory_consolidation_threshold or 6)
    overlap_sessions = max(0, int(overlap_sessions))

    logger.info(
        f"[ingest_longmemoryeval] 开始注入 task: sessions={n}, batch_size={bs}, overlap={overlap_sessions}, "
        f"question_turn_id={question_turn_id}, consolidation_mode={consolidation_mode}"
    )

    # 3) RAW：按窗口批量注入
    all_doc_infos: List[RawDocInfo] = []
    for w_idx, (s, e) in enumerate(iter_windows(n, bs, overlap_sessions), 1):
        win_sessions = ordered_sessions[s:e]
        win_ids = ordered_ids[s:e]
        win_dates = ordered_dates[s:e]
        win_turn_ids = ordered_turn_ids[s:e]

        logger.info(f"[ingest_longmemoryeval] RAW 窗口 {w_idx}: sessions[{s}:{e}] turn_id[{win_turn_ids[0]}..{win_turn_ids[-1]}]")

        doc_infos = ingest_raw_sessions_window(
            agent_name=agent_name,
            sessions_window=win_sessions,
            session_ids_window=win_ids,
            session_dates_window=win_dates,
            turn_ids_window=win_turn_ids,
            include_assistant=include_assistant,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            num_workers=num_workers,
            show_progress=show_progress,
        )
        all_doc_infos.extend(doc_infos)

    # 去重（如果 overlap>0 可能重复）
    uniq_by_turn: Dict[int, RawDocInfo] = {}
    for d in all_doc_infos:
        # 同 turn_id 重复时，保留第一个（更稳定）
        uniq_by_turn.setdefault(int(d.turn_id), d)
    all_doc_infos = [uniq_by_turn[k] for k in sorted(uniq_by_turn.keys())]

    # 4) CONSOLIDATED
    if str(consolidation_mode or "").lower() == "original_text":
        logger.info("[ingest_longmemoryeval] CONSOLIDATED 模式: original_text")
        consolidate_docs_from_original_text(
            ltss=ltss,
            agent_name=agent_name,
            doc_infos=all_doc_infos,
            consolidation_llm=getattr(getattr(agent, "memory", None), "consolidation_llm", None),
            embedding_model=getattr(getattr(agent, "memory", None), "embedding_model", None),
            max_workers=consolidation_max_workers,
            include_assistant=include_assistant,
            max_chars_per_chunk=original_max_chars_per_chunk,
        )
    else:
        logger.info("[ingest_longmemoryeval] CONSOLIDATED 模式: raw_triples")
        consolidate_docs_to_ltss(
            ltss=ltss,
            agent_name=agent_name,
            doc_infos=all_doc_infos,
            consolidation_llm=getattr(getattr(agent, "memory", None), "consolidation_llm", None),
            embedding_model=getattr(getattr(agent, "memory", None), "embedding_model", None),
            max_workers=consolidation_max_workers,
            max_retries=consolidation_max_retries,
            base_backoff_s=consolidation_base_backoff_s,
            min_interval_s=consolidation_min_interval_s,
        )

    logger.info("[ingest_longmemoryeval] ✅ task 注入完成（RAW + CONSOLIDATED）")

    return {
        "question_turn_id": int(question_turn_id),
        "question_date": question_date,  # ✅ 返回实际的问题日期，用于时间计算
        "num_sessions": int(n),
        "doc_infos": all_doc_infos,
    }
