# -*- coding: utf-8 -*-
"""
RAW 通道导入（简化版 - 方案B）：
- 输入：对话窗口原文（字符串）
- 输出：写入 Neo4j
- 职责：只存储原文 + embedding，作为证据来源和向量检索入口
- 不做：LlamaIndex自动提取三元组（结构化提取由CONSOLIDATED通道负责）

写入的节点：
- TextUnit: 原文内容 + embedding + 时间戳
- Event: 事件节点，连接TextUnit
- __Document__: 文档节点（GraphRAG风格）
- __Chunk__: 文本块节点（GraphRAG风格）

写入的关系：
- Event -[:EVIDENCED_BY]-> TextUnit
- __Chunk__ -[:PART_OF]-> __Document__
- TextUnit -[:DERIVED_FROM]-> __Chunk__
"""

from __future__ import annotations

import re
import datetime
import logging
from typing import Optional, Dict, Any, Tuple

from neo4j import GraphDatabase

from config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
)
from utils.embedding import get_embedding_model
from memory.channels import RAW, build_mark_params

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
    # 以下参数保留接口兼容性，但不再使用
    max_paths_per_chunk: int = 20,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    RAW 通道入图（简化版）：只存储原文 + embedding

    返回：
    {
      "doc_id": "...",
      "textunit": "...",
      "event_id": "...",
      "turn_id": int,
      "channel": "raw",
      "session_id": Optional[str],
      "session_time_raw": Optional[str],
      "session_time_iso": Optional[str],
    }
    """
    if not text or not str(text).strip():
        raise ValueError("raw 导入失败：text 为空")

    # 0) 从 text 提取 session 元信息
    session_id, session_time_raw, session_time_iso = _extract_session_meta(str(text))

    # 1) 获取 embedding model
    embed_model = get_embedding_model()

    # 2) 连接 Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # 3) 生成 ID
    now_iso = datetime.datetime.now().isoformat()
    safe_agent = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in (agent_name or "agent"))
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    doc_id = f"raw_{safe_agent}_{ts}"
    tu_name = f"raw_unit_{safe_agent}_{ts}"

    # 4) 构建参数
    params = build_mark_params(
        [doc_id],
        agent_name=agent_name,
        channel=RAW,
        virtual_time=virtual_time,
        session_time_iso=session_time_iso,
    )

    # 5) 计算 embedding
    try:
        tu_emb = embed_model.embed_query(str(text))
    except Exception as e:
        logger.warning(f"[raw_graph_ingest] embedding failed: {e}")
        tu_emb = None

    logger.info(
        f"[raw_graph_ingest] 开始 raw 导入：agent={agent_name}, doc_id={doc_id}, "
        f"virtual_time={virtual_time}, session_id={session_id}, session_time={session_time_raw}"
    )

    # 6) 写入 Neo4j
    with driver.session(database="neo4j") as session:
        # 6.1 创建 TextUnit（原文 + embedding）
        session.run(
            """
            MERGE (u:TextUnit {name: $name})
            SET u.content = $content,
                u.virtual_time = $virtual_time,
                u.turn_id = $turn_id,
                u.real_time = $real_time,
                u.embedding = $embedding,
                u.agent_name = $agent_name,
                u.channel = $channel,
                u.session_id = $session_id,
                u.session_time_raw = $session_time_raw,
                u.session_time_iso = $session_time_iso
            """,
            {
                "name": tu_name,
                "content": str(text),
                "virtual_time": params["virtual_time"],
                "turn_id": params["turn_id"],
                "real_time": now_iso,
                "embedding": tu_emb,
                "agent_name": agent_name,
                "channel": RAW,
                "session_id": session_id or "",
                "session_time_raw": session_time_raw or "",
                "session_time_iso": session_time_iso or "",
            },
        )

        # 6.2 创建 Event 并连接 TextUnit
        event_id = f"evt:{RAW}:{safe_agent}:{params['virtual_time']}:{tu_name}"
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
                    WHEN $session_time_iso <> '' THEN $session_time_iso
                    WHEN $session_time_raw <> '' THEN $session_time_raw
                    ELSE $virtual_time
                END
            WITH e
            MATCH (u:TextUnit {name: $textunit})
            MERGE (e)-[:EVIDENCED_BY]->(u)
            """,
            {
                "event_id": event_id,
                "channel": RAW,
                "agent_name": agent_name,
                "virtual_time": params["virtual_time"],
                "turn_id": params["turn_id"],
                "event_timestamp": session_time_iso or params["virtual_time"],
                "updated_at": now_iso,
                "textunit": tu_name,
                "session_id": session_id or "",
                "session_time_raw": session_time_raw or "",
                "session_time_iso": session_time_iso or "",
            },
        )

        # 6.3 创建 __Document__（GraphRAG风格）
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

        # 6.4 创建 __Chunk__ 并建立关系
        chunk_id = f"{doc_id}:{tu_name}"
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

    driver.close()

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
