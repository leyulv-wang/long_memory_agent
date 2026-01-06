# -*- coding: utf-8 -*-
"""
统一双通道相关常量与工具函数：
- channel: 'raw' / 'consolidated'
- 节点：维护 n.channels (list) 作为集合（apoc.toSet 去重）
- 关系：写 r.channel / r.agent_name / r.turn_id
"""

from __future__ import annotations
import re
from typing import Optional, Tuple, Dict, Any

RAW = "raw"
CONSOLIDATED = "consolidated"

_TURN_RE = re.compile(r"(?i)\bturn[_\-\s]*([0-9]+)\b")


def parse_turn_id(virtual_time: Optional[str]) -> int:
    """把 TURN_12 / turn12 / '12' 解析成 int；失败返回 0。"""
    if not virtual_time:
        return 0
    s = str(virtual_time).strip()
    if not s:
        return 0
    m = _TURN_RE.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return 0
    return 0


def normalize_channel(channel: str) -> str:
    c = (channel or "").strip().lower()
    if c in (RAW, CONSOLIDATED):
        return c
    # 兜底：未知都当 raw
    return RAW


def cypher_mark_nodes_channels_by_doc_id() -> str:
    """
    给本次导入（doc_id 在 doc_ids 中）的节点打 channels + agent_name
    注意：LlamaIndex 导入的 Chunk/Entity 节点都带 doc_id 或 triplet_source_id 等属性，
    我们这里主要用 doc_id 作为“本次导入”的锚点（更稳定）。
    """
    return """
    MATCH (n)
    WHERE n.doc_id IN $doc_ids OR n.document_id IN $doc_ids
    SET n.agent_name = coalesce(n.agent_name, $agent_name),
        n.channels = apoc.coll.toSet(coalesce(n.channels, []) + [$channel])
    RETURN count(n) AS updated_nodes
    """.strip()


def cypher_mark_rels_channel_connected_to_doc_nodes() -> str:
    """
    标记关系：只标记“连接到本次导入 doc_id 节点”的关系，避免污染全库。
    """
    return """
    MATCH (n)
    WHERE n.doc_id IN $doc_ids OR n.document_id IN $doc_ids
    WITH collect(id(n)) AS ids
    MATCH (a)-[r]->(b)
    WHERE id(a) IN ids OR id(b) IN ids
    SET r.channel = $channel,
        r.agent_name = $agent_name,
        r.turn_id = $turn_id,
        r.virtual_time = $virtual_time,
        r.confidence = coalesce(r.confidence, 1.0)
    RETURN count(r) AS updated_rels
    """.strip()


def cypher_create_textunit_for_raw() -> str:
    """
    raw 通道也建议创建 TextUnit 作为证据载体（和 consolidated 一致）。
    这样后续你可以把 raw->consolidated 的证据追溯做得很干净。
    """
    return """
    MERGE (u:TextUnit {name: $name})
    SET u.content = $content,
        u.virtual_time = $virtual_time,
        u.turn_id = $turn_id,
        u.real_time = $real_time,
        u.embedding = $embedding,
        u.agent_name = $agent_name,
        u.channel = $channel
    RETURN u.name AS name
    """.strip()


def cypher_link_doc_nodes_to_textunit() -> str:
    """
    把本次导入 doc_id 对应的节点，统一连接到本次 raw TextUnit（证据）。
    用 FROM_SOURCE 和你 consolidated 的写法一致。
    """
    return """
    MATCH (u:TextUnit {name: $textunit})
    MATCH (n)
    WHERE n.doc_id IN $doc_ids OR n.document_id IN $doc_ids
    MERGE (n)-[:FROM_SOURCE]->(u)
    RETURN count(n) AS linked
    """.strip()


def build_mark_params(
    doc_ids,
    agent_name: str,
    channel: str,
    virtual_time: Optional[str],
) -> Dict[str, Any]:
    c = normalize_channel(channel)
    turn_id = parse_turn_id(virtual_time)
    return {
        "doc_ids": list(doc_ids),
        "agent_name": agent_name,
        "channel": c,
        "virtual_time": virtual_time or "unknown",
        "turn_id": turn_id,
    }
def cypher_create_event_and_link_textunit() -> str:
    return """
    MERGE (e:Event {event_id: $event_id})
    SET e.channel = $channel,
        e.agent_name = $agent_name,
        e.virtual_time = $virtual_time,
        e.turn_id = $turn_id,
        e.event_timestamp = $event_timestamp,
        e.updated_at = $updated_at
    WITH e
    MATCH (u:TextUnit {name: $textunit})
    MERGE (e)-[:EVIDENCED_BY]->(u)
    RETURN e.event_id AS event_id
    """.strip()


def cypher_attach_event_id_to_doc_rels() -> str:
    return """
    MATCH (n)
    WHERE n.doc_id IN $doc_ids OR n.document_id IN $doc_ids
    WITH collect(id(n)) AS ids
    MATCH (a)-[r]->(b)
    WHERE id(a) IN ids OR id(b) IN ids
    SET r.event_id = $event_id
    RETURN count(r) AS updated
    """.strip()
