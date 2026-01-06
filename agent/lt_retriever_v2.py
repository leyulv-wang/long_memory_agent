# -*- coding: utf-8 -*-
"""
V2 长期记忆检索（软更新 + consolidated 优先 + TURN 衰减 + Neo4j 向量索引）

核心：
1) 用 Neo4j 向量索引 textunit_vector_index 对 TextUnit.embedding 做相似度召回（raw + consolidated）
2) seed TextUnit -> Event -> Fact -> SUBJECT/OBJECT 取结构化事实
3) score = sim * confidence * exp(-lambda * age) * channel_bonus
4) 软更新：按 slot_key（若存在）或 belief_key 分组，只输出每组的最佳候选，并保留 alternatives

依赖：
- ltss.query_graph(cypher, params)
- Neo4j 已建向量索引：textunit_vector_index（stores.py 里就是这个名字）
- config.STEP_DECAY_RATE: 按轮次衰减系数（空/<=0 则不衰减）
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import List, Dict, Any, Optional, Tuple

from config import STEP_DECAY_RATE
from utils.embedding import get_embedding_model


_TEXTUNIT_VECTOR_INDEX = "textunit_vector_index"
_TURN_RE = re.compile(r"(?:TURN|Turn|STEP|Step)[^0-9]*(\d+)")
logger = logging.getLogger(__name__)
_DEBUG_GRAPHRAG = os.getenv("DEBUG_GRAPHRAG", "0") == "1"
_DEBUG_PIPELINE = os.getenv("DEBUG_PIPELINE", "0") == "1"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _parse_turn_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if not s:
        return default
    m = _TURN_RE.search(s)
    if m:
        return _safe_int(m.group(1), default)
    # 纯数字兜底
    try:
        return int(s)
    except Exception:
        return default


def _channel_bonus(channel: str) -> float:
    if (channel or "").lower() == "consolidated":
        return 1.25
    return 1.0


def _decay(age: int) -> float:
    if STEP_DECAY_RATE is None:
        return 1.0
    try:
        lam = float(STEP_DECAY_RATE)
    except Exception:
        lam = 0.0
    if lam <= 0.0:
        return 1.0
    return float(math.exp(-lam * float(max(0, age))))


def _score(sim: float, conf: float, fact_turn: int, current_turn: int, channel: str) -> float:
    age = max(0, int(current_turn) - int(fact_turn))
    return float(sim) * float(conf) * _decay(age) * _channel_bonus(channel)


def _better_tie_break(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    True 表示 a 比 b 更优（用于同分或近似同分的 tie-break）：
    1) consolidated > raw
    2) turn_id 更大（更近）
    3) confidence 更大
    """
    ach = (a.get("channel") or "").lower()
    bch = (b.get("channel") or "").lower()
    if ach != bch:
        return ach == "consolidated"

    at = _safe_int(a.get("turn_id"), 0)
    bt = _safe_int(b.get("turn_id"), 0)
    if at != bt:
        return at > bt

    ac = _safe_float(a.get("confidence"), 0.0)
    bc = _safe_float(b.get("confidence"), 0.0)
    return ac >= bc


def _vector_retrieve_textunits(
    ltss,
    *,
    query: str,
    agent_name: str,
    k: int,
    channels: List[str],
) -> List[Dict[str, Any]]:
    """
    直接用 Neo4j 向量索引召回 TextUnit，避免把全库 embedding 拉回 Python。
    返回：name/content/channel/turn_id/sim(Neo4j score)
    """
    emb_model = get_embedding_model()
    q_emb = emb_model.embed_query(query)

    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $embedding) YIELD node, score
    WHERE
      (node.accessible_by IS NULL OR 'all' IN node.accessible_by OR $agent_name IN node.accessible_by)
      AND (node.agent_name IS NULL OR node.agent_name = $agent_name)
      AND (node.channel IS NULL OR node.channel IN $channels)
    RETURN
      node.name AS name,
      node.content AS content,
      node.channel AS channel,
      node.turn_id AS turn_id,
      score AS sim
    """
    try:
        rows = ltss.query_graph(
            cypher,
            {
                "index_name": _TEXTUNIT_VECTOR_INDEX,
                "k": int(k),
                "embedding": q_emb,
                "agent_name": agent_name,
                "channels": [c.lower() for c in (channels or [])],
            },
        ) or []
        out = []
        for r in rows:
            n = r.get("name")
            if not n:
                continue
            out.append(
                {
                    "name": n,
                    "content": r.get("content", "") or "",
                    "channel": (r.get("channel") or "") or "",
                    "turn_id": _parse_turn_int(r.get("turn_id"), 0),
                    "sim": _safe_float(r.get("sim"), 0.0),
                }
            )
        return out
    except Exception as e:
        if _DEBUG_GRAPHRAG or _DEBUG_PIPELINE:
            logger.info(f"[GraphRAG-V2][debug] vector query failed: {e}")
        return []


def retrieve_long_term_facts_v2(
    ltss,
    query: str,
    current_turn: int,
    k: int = 8,
    seed_textunit_top_m: int = 12,
    channels: Optional[List[str]] = None,
    max_alternatives_per_slot: int = 3,
    agent_name: str = "User",
) -> List[Dict[str, Any]]:
    """
    返回 list[dict]，每个 dict 表示一个“软更新后的最佳事实”：
    {
      "slot_id": "...",                 # slot_key 或 belief_key（用于软更新分组）
      "best": {...事实...},             # 最佳候选
      "alternatives": [{...}, ...]      # 其它候选（最多 max_alternatives_per_slot 个）
    }

    best/alternatives 的字段：
      belief_key, slot_key(若有), rel_type, confidence, channel, turn_id, event_id,
      evidence_textunit, evidence_content, subject, object, sim, score
    """
    if not query or not str(query).strip():
        return []

    cur_turn_int = _parse_turn_int(current_turn, 0)

    # Keyword bonus: simple lexical match against subject/object to avoid
    # long-session noise dominating retrieval.
    ql = re.sub(r"[^a-z0-9\\s]", " ", str(query or "").lower())
    tokens = [t for t in ql.split() if len(t) >= 4]
    stop = {"what", "which", "when", "where", "would", "could", "about", "from", "your", "with", "that", "this", "have"}
    keywords = [t for t in tokens if t not in stop]

    if channels is None:
        channels = ["consolidated", "raw"]
    channels_norm = [str(c).strip().lower() for c in channels if str(c).strip()]
    if not channels_norm:
        channels_norm = ["consolidated", "raw"]

    # 1) 先用 Neo4j 向量索引召回 TextUnit
    seed_units = _vector_retrieve_textunits(
        ltss,
        query=query,
        agent_name=agent_name,
        k=max(1, int(seed_textunit_top_m)),
        channels=channels_norm,
    )
    if not seed_units:
        if _DEBUG_GRAPHRAG or _DEBUG_PIPELINE:
            logger.info("[GraphRAG-V2][debug] no seed TextUnit hits")
        return []

    # seed 相似度映射：后面给 fact 打分要乘 sim
    sim_map: Dict[str, float] = {}
    for u in seed_units:
        nm = u.get("name")
        if nm:
            sim_map[nm] = float(u.get("sim") or 0.0)

    seed_names = [u["name"] for u in seed_units if u.get("name")]
    if not seed_names:
        return []

    # 2) seed TextUnit -> Event -> Fact
    # 兼容：Fact 不一定有 slot_key，所以都返回，Python 侧分组时兜底 belief_key
    cypher_facts = """
    UNWIND $seed_names AS tu
    MATCH (e:Event)-[:EVIDENCED_BY]->(u:TextUnit {name: tu})
    MATCH (e)-[:HAS_FACT]->(f:Fact)
    OPTIONAL MATCH (f)-[:SUBJECT]->(s)
    OPTIONAL MATCH (f)-[:OBJECT]->(o)
    RETURN
      u.name AS evidence_textunit,
      u.content AS evidence_content,
      u.channel AS tu_channel,
      u.turn_id AS tu_turn_id,

      e.event_id AS event_id,
      e.channel AS event_channel,

      f.belief_key AS belief_key,
      f.slot_key AS slot_key,
      f.type AS rel_type,
      f.confidence AS confidence,
      f.turn_id AS turn_id,
      f.channel AS channel,

      s.name AS subject,
      o.name AS object
    """
    fact_rows = ltss.query_graph(cypher_facts, {"seed_names": seed_names}) or []
    if not fact_rows:
        if _DEBUG_GRAPHRAG or _DEBUG_PIPELINE:
            logger.info("[GraphRAG-V2][debug] no Fact rows for seed TextUnit hits")
        return []

    # 3) 打分候选
    candidates: List[Dict[str, Any]] = []
    for r in fact_rows:
        belief_key = r.get("belief_key")
        if not belief_key:
            continue

        ev_unit = r.get("evidence_textunit")
        sim = float(sim_map.get(ev_unit, 0.0))

        # channel 优先级：Fact.channel > Event.channel > TextUnit.channel
        ch = (r.get("channel") or r.get("event_channel") or r.get("tu_channel") or "raw")
        ch = str(ch).lower()

        conf = _safe_float(r.get("confidence"), 0.9)
        f_turn = _parse_turn_int(r.get("turn_id"), _parse_turn_int(r.get("tu_turn_id"), 0))

        sc = _score(sim, conf, f_turn, cur_turn_int, ch)
        if keywords:
            subj = (r.get("subject") or "").lower()
            obj = (r.get("object") or "").lower()
            hit = 0
            for kw in keywords:
                if kw in subj or kw in obj:
                    hit += 1
            if hit:
                bonus = min(0.6, 0.15 * hit)
                sc = sc * (1.0 + bonus)

        candidates.append(
            {
                "belief_key": str(belief_key),
                "slot_key": r.get("slot_key"),
                "rel_type": r.get("rel_type"),
                "confidence": conf,
                "channel": ch,
                "turn_id": f_turn,
                "event_id": r.get("event_id"),
                "evidence_textunit": ev_unit,
                "evidence_content": r.get("evidence_content", "") or "",
                "subject": r.get("subject"),
                "object": r.get("object"),
                "sim": sim,
                "score": sc,
            }
        )

    if not candidates:
        return []

    # 4) 软更新分组：优先 slot_key，否则 belief_key
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for c in candidates:
        slot_id = c.get("slot_key") or c.get("belief_key")
        groups.setdefault(str(slot_id), []).append(c)

    # 组内排序：score desc；极近似同分时强制 consolidated 优先
    def _group_sort_key(x: Dict[str, Any]) -> Tuple[float, float, int, float]:
        # 主：score；次：sim；再：turn_id；再：confidence
        return (
            float(x.get("score") or 0.0),
            float(x.get("sim") or 0.0),
            _safe_int(x.get("turn_id"), 0),
            _safe_float(x.get("confidence"), 0.0),
        )

    results: List[Dict[str, Any]] = []
    for slot_id, items in groups.items():
        items.sort(key=_group_sort_key, reverse=True)

        if len(items) >= 2:
            top = items[0]
            for j in range(1, len(items)):
                cand = items[j]
                if abs(float(cand["score"]) - float(top["score"])) < 1e-9:
                    if _better_tie_break(cand, top):
                        items[0], items[j] = items[j], items[0]
                        top = items[0]
                else:
                    break

        best = items[0]
        alts = items[1 : 1 + max(0, int(max_alternatives_per_slot))]

        results.append(
            {
                "slot_id": slot_id,
                "best": best,
                "alternatives": alts,
            }
        )

    # 5) 组间排序：按 best.score 排，取 top-k
    results.sort(key=lambda g: float(g["best"].get("score") or 0.0), reverse=True)
    return results[: max(1, int(k))]
