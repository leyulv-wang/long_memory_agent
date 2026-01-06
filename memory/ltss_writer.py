# -*- coding: utf-8 -*-
import datetime
import logging
import os
from collections import defaultdict
from typing import Any, Dict, Optional, List
import json
import re
import time

logger = logging.getLogger(__name__)
_DEBUG_LTSS = os.getenv("DEBUG_LTSS", "0") == "1" or os.getenv("DEBUG_PIPELINE", "0") == "1"
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

# 兼容 TURN_12 / TURN12 / TURN_001
_TURN_RE = re.compile(r"^TURN_?(\d+)$", re.IGNORECASE)

# 仅允许 Cypher 标识符安全字符：字母数字下划线
_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_turn(s: str):
    """返回 turn 的 int（用于 event_step 等），解析失败返回 None。"""
    if not s:
        return None
    m = _TURN_RE.match(str(s).strip())
    return int(m.group(1)) if m else None


def _parse_turn_id(s: str) -> int:
    """返回 turn 的 int（用于 turn_id/recorded_turn_id），解析失败返回 -1。"""
    v = _parse_turn(s)
    return int(v) if v is not None else -1


def _safe_ident(s: Any, default: str) -> str:
    """
    label/rel_type 会动态拼进 Cypher：必须保证是合法标识符，否则会报错/有注入风险。
    """
    try:
        ss = str(s or "").strip()
    except Exception:
        ss = ""
    if not ss:
        return default
    if _SAFE_IDENT_RE.match(ss):
        return ss
    return default


def serialize_props(props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Neo4j 属性不接受 dict；list 里如果有 dict/复杂对象也会失败；
    这里统一把 dict/list（embedding 除外）序列化成 JSON 字符串。
    """
    new_props: Dict[str, Any] = {}
    for k, v in (props or {}).items():
        if k == "embedding":
            new_props[k] = v
            continue

        if isinstance(v, (dict, list)):
            try:
                new_props[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                new_props[k] = str(v)
            continue

        new_props[k] = v
    return new_props


def _make_event_id(agent_name: str, channel: str, virtual_time: str, event_timestamp: str, evidence_unit: str) -> str:
    """
    Event 节点 ID：优先用 event_timestamp（如果抽出来了），否则用 turn_id+evidence_unit。
    稳定且不会和 raw 冲突（包含 channel + agent_name）。
    """
    vt = str(virtual_time or "unknown")
    et = str(event_timestamp or "unknown")
    agent = str(agent_name or "agent")
    ch = str(channel or "consolidated")

    if et and et.lower() != "unknown":
        return f"evt:{ch}:{agent}:{et}"
    return f"evt:{ch}:{agent}:{vt}:{evidence_unit}"


# ----------------------------
# turn-offset 推断（轻量规则）
# ----------------------------
_REL_TIME_MAP = {
    "yesterday": -1,
    "today": 0,
    "tomorrow": 1,
    "last week": -7,
    "next week": 7,
    "last month": -30,
    "next month": 30,
    "last year": -365,
    "next year": 365,
}


def _infer_event_turn_offset(props: Dict[str, Any]) -> int:
    """
    从 props 里推断 event_turn_offset：
    1) 若 LLM/上游已经给了 event_turn_offset/event_offset：优先用
    2) 否则尝试读 event_time_text/event_time 里的常见相对时间（英文）
    3) 否则返回 0
    """
    # 1) 显式 offset
    for k in ["event_turn_offset", "event_offset", "turn_offset", "time_offset"]:
        if k in props:
            try:
                return int(float(props.get(k)))
            except Exception:
                pass

    # 2) 相对时间文本
    cand = ""
    for k in ["event_time_text", "event_time", "event_timestamp_text", "time_text"]:
        v = props.get(k)
        if isinstance(v, str) and v.strip():
            cand = v.strip().lower()
            break

    if cand:
        # 直接命中短语
        for phrase, off in _REL_TIME_MAP.items():
            if phrase in cand:
                return int(off)

        # “X days ago / in X days”
        m = re.search(r"(\d+)\s+days?\s+ago", cand)
        if m:
            return -int(m.group(1))
        m = re.search(r"in\s+(\d+)\s+days?", cand)
        if m:
            return int(m.group(1))

        # “X weeks ago / in X weeks”
        m = re.search(r"(\d+)\s+weeks?\s+ago", cand)
        if m:
            return -7 * int(m.group(1))
        m = re.search(r"in\s+(\d+)\s+weeks?", cand)
        if m:
            return 7 * int(m.group(1))

    return 0


def _compute_event_turn_fields(recorded_turn_id: int, props: Dict[str, Any]) -> Dict[str, int]:
    """
    输出：
      recorded_turn_id
      event_turn_offset
      event_turn_id = recorded_turn_id + offset（下限 0）
    """
    offset = _infer_event_turn_offset(props or {})
    try:
        rid = int(recorded_turn_id)
    except Exception:
        rid = -1

    etid = rid + int(offset)
    if etid < 0:
        etid = 0

    return {
        "recorded_turn_id": rid,
        "event_turn_offset": int(offset),
        "event_turn_id": int(etid),
    }


def _upsert_document_chunk(
    ltss,
    *,
    doc_id: str,
    chunk_id: str,
    textunit_id: str,
    memories_str: str,
    agent_name: str,
    channel: str,
    virtual_time: str,
    turn_id: int,
    session_id: Optional[str] = None,
    session_time_raw: Optional[str] = None,
    session_time_iso: Optional[str] = None,
) -> None:
    if not doc_id or not chunk_id:
        return

    cy_doc = """
    MERGE (d:__Document__ {id: $doc_id})
    ON CREATE SET d.title = $title, d.raw_content = $raw
    SET d.agent_name = $agent_name,
        d.updated_at = $updated_at,
        d.session_id = $session_id,
        d.session_time_raw = $session_time_raw,
        d.session_time_iso = $session_time_iso
    """
    ltss.update_graph(
        cy_doc,
        parameters={
            "doc_id": doc_id,
            "title": session_id or doc_id,
            "raw": memories_str or "",
            "agent_name": agent_name,
            "updated_at": datetime.datetime.now().isoformat(),
            "session_id": session_id,
            "session_time_raw": session_time_raw,
            "session_time_iso": session_time_iso,
        },
    )

    cy_chunk = """
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
    """
    ltss.update_graph(
        cy_chunk,
        parameters={
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": memories_str or "",
            "agent_name": agent_name,
            "channel": channel,
            "virtual_time": virtual_time,
            "turn_id": int(turn_id),
            "session_id": session_id,
            "session_time_raw": session_time_raw,
            "session_time_iso": session_time_iso,
            "textunit_id": textunit_id,
        },
    )


def write_consolidation_result(
    ltss,
    embedding_model,
    agent_name: str,
    memories_str: str,
    structured_response,
    current_time: Optional[str],
    channel: str = "consolidated",
    *,
    doc_id: Optional[str] = None,
    session_id: Optional[str] = None,
    session_time_raw: Optional[str] = None,
    session_time_iso: Optional[str] = None,
):
    """
    写入 Neo4j：
    1) TextUnit 证据节点（含 channel / virtual_time / turn_id）
    2) 节点：维护 n.channels 集合（apoc.toSet 去重），不写单值 channel
    3) 关系：写 r.channel（raw/consolidated），belief_key 含 channel+agent 避免冲突
    4) Event 节点 + Fact 节点（reify 关系），用于事件来源、解释、以及后续衰减/检索排序
    """
    if not ltss or not getattr(ltss, "driver", None):
        logger.warning("[ltss_writer] LTSS 不可用，跳过写入数据库。")
        return

    agent_name_log_prefix = f"[智能体 '{agent_name}' 的记忆巩固引擎]"
    real_timestamp = datetime.datetime.now().isoformat()

    # --- embedding profiling (no logic change) ---
    _emb_calls = 0
    _emb_total_s = 0.0
    _emb_tu_s = 0.0
    _emb_node_calls = 0
    _emb_node_s = 0.0

    # 1) virtual_time / TextUnit ID
    virtual_timestamp = "unknown"
    if current_time and str(current_time).strip().lower() not in ["none", "null"]:
        virtual_timestamp = str(current_time).strip()

    text_unit_id = f"unit_{virtual_timestamp}" if virtual_timestamp != "unknown" else "unit_unknown"
    logger.info(f"{agent_name_log_prefix} 生成 TextUnit ID: {text_unit_id} (virtual_time: {virtual_timestamp})")
    if _DEBUG_LTSS:
        logger.info(
            f"[ltss_writer][debug] structured nodes={len(getattr(structured_response, 'nodes', []) or [])} "
            f"rels={len(getattr(structured_response, 'relationships', []) or [])} "
            f"facts={len(getattr(structured_response, 'facts', []) or [])} "
            f"insights={len(getattr(structured_response, 'insights', []) or [])}"
        )

    # 当前写入发生的 turn（recorded）
    recorded_turn_id = _parse_turn_id(virtual_timestamp)
    turn_step = _parse_turn(virtual_timestamp)

    # A) TextUnit embedding（原逻辑：单条 embed_query）
    tu_emb = None
    if isinstance(memories_str, str) and memories_str.strip():
        try:
            _t0 = time.time()
            tu_emb = embedding_model.embed_query(memories_str)
            _dt = time.time() - _t0
            _emb_calls += 1
            _emb_total_s += _dt
            _emb_tu_s += _dt
            logger.info(f"[ltss_writer][embed] mode=single kind=textunit n=1 sec={_dt:.3f}")
        except Exception as e:
            logger.warning(f"{agent_name_log_prefix} TextUnit embedding failed: {e}")
            tu_emb = None

    # A) 写 TextUnit
    ltss.update_graph(
        "MERGE (u:TextUnit {name: $name}) "
        "SET u.content = $content, "
        "    u.virtual_time = $vts, "
        "    u.turn_id = $turn_id, "
        "    u.real_time = $rts, "
        "    u.channel = $channel, "
        "    u.agent_name = $agent_name, "
        "    u.embedding = $embedding ",
        parameters={
            "name": text_unit_id,
            "content": memories_str,
            "vts": virtual_timestamp,
            "turn_id": recorded_turn_id,
            "rts": real_timestamp,
            "channel": channel,
            "agent_name": agent_name,
            "embedding": tu_emb,
        },
    )

    # A.1) Document/Chunk evidence chain (GraphRAG-style)
    if doc_id:
        chunk_id = f"{doc_id}:{text_unit_id}"
        _upsert_document_chunk(
            ltss,
            doc_id=doc_id,
            chunk_id=chunk_id,
            textunit_id=text_unit_id,
            memories_str=memories_str,
            agent_name=agent_name,
            channel=channel,
            virtual_time=virtual_timestamp,
            turn_id=recorded_turn_id,
            session_id=session_id,
            session_time_raw=session_time_raw,
            session_time_iso=session_time_iso,
        )
    else:
        chunk_id = None

    # B) Nodes
    if getattr(structured_response, "nodes", None):
        nodes_to_write = []
        node_names: List[str] = []

        for node in structured_response.nodes:
            nlabel = _safe_ident(getattr(node, "label", None) or "Concept", "Concept")
            nname = str(getattr(node, "name", "") or "").strip()
            if not nname:
                continue

            props = node.properties if isinstance(getattr(node, "properties", None), dict) else {}
            props = dict(props or {})
            props.update(
                {
                    "agent_name": agent_name,
                    "updated_at": real_timestamp,
                    "virtual_time": virtual_timestamp,
                    "turn_id": recorded_turn_id,
                }
            )

            nodes_to_write.append({"label": nlabel, "name": nname, "props": props})
            node_names.append(nname)

        # Batch embeddings for nodes (faster on remote server/local batch)
        node_embs: List[Optional[List[float]]] = []
        if embedding_model and node_names:
            try:
                _t0 = time.time()
                if hasattr(embedding_model, "embed_documents"):
                    node_embs = embedding_model.embed_documents(node_names) or []
                else:
                    node_embs = [embedding_model.embed_query(n) for n in node_names]
                _dt = time.time() - _t0
                _emb_calls += 1
                _emb_total_s += _dt
                _emb_node_calls += len(node_embs)
                _emb_node_s += _dt
                logger.debug(
                    f"[ltss_writer][embed] mode=batch kind=node n={len(node_embs)} sec={_dt:.3f}"
                )
            except Exception as e:
                logger.warning(f"[ltss_writer][embed] node batch failed, fallback to single: {e}")
                node_embs = []

        # Fill embeddings (fallback single if batch missing/mismatch)
        for i, it in enumerate(nodes_to_write):
            emb = None
            if node_embs and i < len(node_embs):
                emb = node_embs[i]
            elif embedding_model:
                try:
                    _t0 = time.time()
                    emb = embedding_model.embed_query(it["name"])
                    _dt = time.time() - _t0
                    _emb_calls += 1
                    _emb_total_s += _dt
                    _emb_node_calls += 1
                    _emb_node_s += _dt
                    logger.debug(
                        f"[ltss_writer][embed] mode=single kind=node n=1 sec={_dt:.3f} name={it['name'][:60]}"
                    )
                except Exception:
                    emb = None

            it["props"]["embedding"] = emb
            it["props"] = serialize_props(it["props"])

        groups = defaultdict(list)
        for it in nodes_to_write:
            groups[it["label"]].append(it)

        for label, batch in groups.items():
            label_safe = _safe_ident(label, "Concept")
            cypher_nodes = f"""
UNWIND $batch AS row
MERGE (n:`{label_safe}` {{name: row.name}})
SET n += row.props
WITH n, $channel AS ch
SET n.channels = apoc.coll.toSet(coalesce(n.channels, []) + [ch])
RETURN count(*)
""".strip()
            ltss.update_graph(cypher_nodes, parameters={"batch": batch, "channel": channel})

        # Link entities to chunk for evidence tracing
        if chunk_id:
            for label, batch in groups.items():
                label_safe = _safe_ident(label, "Concept")
                rows = [{"name": it["name"]} for it in batch]
                cy_link = f"""
UNWIND $rows AS row
MATCH (c:__Chunk__ {{id: $chunk_id}})
MATCH (e:`{label_safe}` {{name: row.name}})
MERGE (c)-[:HAS_ENTITY]->(e)
""".strip()
                ltss.update_graph(cy_link, parameters={"rows": rows, "chunk_id": chunk_id})

    # ----------------------------
    # C1) Facts / Insights（strings）→ Property-as-edge
    # ----------------------------
    props_edge_rows = []

    if getattr(structured_response, "facts", None):
        for fact in structured_response.facts:
            if not isinstance(fact, str) or not fact.strip():
                continue
            props_edge_rows.append(("Fact", fact.strip(), 1.0))

    if getattr(structured_response, "insights", None):
        for ins in structured_response.insights:
            if not isinstance(ins, str) or not ins.strip():
                continue
            props_edge_rows.append(("Insight", ins.strip(), 0.9))

    if props_edge_rows:
        batch = []
        for lbl, text, conf in props_edge_rows:
            s_label = "TextFact"
            t_label = lbl
            s_name = text_unit_id
            t_name = text

            props = {
                "confidence": float(conf),
                "knowledge_type": "observed_fact" if lbl == "Fact" else "logical_inference",
                "source_of_belief": "reflection",
                "consolidated_at": real_timestamp,
                "virtual_time": virtual_timestamp,
                "turn_id": recorded_turn_id,
                "event_step": turn_step,
                "event_timestamp": "unknown",
                "evidence_source_unit": text_unit_id,
                "agent_name": agent_name,
                "channel": channel,
                "is_current": True,
                "should_be_current": True,
                "source_rank": 1.0,
            }
            # Ensure TextFact facts don't collapse into one slot during soft-update grouping.
            props["slot_key"] = f"{s_name}:ASSERTS:{t_name}"

            turn_fields = _compute_event_turn_fields(recorded_turn_id, props)
            props.update(turn_fields)

            event_id = _make_event_id(agent_name, channel, virtual_timestamp, "unknown", text_unit_id)
            props["event_id"] = event_id

            base_key = f"{s_label}:{s_name}:ASSERTS:{t_label}:{t_name}"
            props["belief_key"] = f"{channel}:{agent_name}:{base_key}"

            batch.append(
                {
                    "s_name": s_name,
                    "s_label": s_label,
                    "t_name": t_name,
                    "t_label": t_label,
                    "type": "ASSERTS",
                    "props": serialize_props(props),
                }
            )

        cypher_props_as_edges = """
UNWIND $batch AS row
MERGE (source:TextUnit {name: row.s_name})
MERGE (target:`__Node__` {name: row.t_name})
SET target._node_type = row.t_label

WITH source, target, row
MATCH (u:TextUnit {name: row.props.evidence_source_unit})
MERGE (source)-[:FROM_SOURCE]->(u)
MERGE (target)-[:FROM_SOURCE]->(u)

WITH source, target, row, row.props AS props
OPTIONAL MATCH (source)-[old:ASSERTS]->(other)
WHERE old.belief_key = props.belief_key
  AND coalesce(old.is_current,false) = true

WITH source, target, old, other, row, props,
CASE
  WHEN old IS NULL THEN true
  WHEN coalesce(props.source_rank, 1.0) > coalesce(old.source_rank, 1.0) THEN true
  WHEN coalesce(props.source_rank, 1.0) < coalesce(old.source_rank, 1.0) THEN false
  WHEN coalesce(props.event_step, -1) > coalesce(old.event_step, -1) THEN true
  WHEN coalesce(props.event_step, -1) < coalesce(old.event_step, -1) THEN false
  WHEN props.event_timestamp <> 'unknown' AND old.event_timestamp <> 'unknown' AND props.event_timestamp > old.event_timestamp THEN true
  WHEN coalesce(props.confidence, 0.0) >= coalesce(old.confidence, 0.0) THEN true
  ELSE false
END AS new_wins

FOREACH (_ IN CASE WHEN old IS NOT NULL AND new_wins = true AND other.name <> row.t_name THEN [1] ELSE [] END |
  SET old.is_current = false,
      old.deprecated_at = props.consolidated_at
)

MERGE (source)-[r:ASSERTS {belief_key: props.belief_key}]->(target)
ON CREATE SET
  r += props,
  r.created_at = props.consolidated_at,
  r.is_current = new_wins
ON MATCH SET
  r.confidence = CASE WHEN new_wins = true THEN props.confidence ELSE r.confidence END,
  r.consolidated_at = props.consolidated_at,
  r.virtual_time = props.virtual_time,
  r.turn_id = props.turn_id,
  r.recorded_turn_id = props.recorded_turn_id,
  r.event_turn_offset = props.event_turn_offset,
  r.event_turn_id = props.event_turn_id,
  r.event_step = props.event_step,
  r.event_timestamp = props.event_timestamp,
  r.event_id = props.event_id,
  r.is_current = CASE WHEN new_wins = true THEN true ELSE r.is_current END

// --- Fact + Event ---
WITH source, target, row, props, r,
     toString(props.belief_key) AS bk

MERGE (f:Fact {belief_key: bk})
ON CREATE SET
  f.type = type(r),
  f.slot_key = coalesce(props.slot_key, toString(source.name) + ':' + type(r)),
  f.channel = props.channel,
  f.agent_name = props.agent_name,
  f.virtual_time = props.virtual_time,
  f.turn_id = props.turn_id,
  f.recorded_turn_id = props.recorded_turn_id,
  f.event_turn_offset = props.event_turn_offset,
  f.event_turn_id = props.event_turn_id,
  f.event_timestamp = props.event_timestamp,
  f.source_of_belief = props.source_of_belief,
  f.knowledge_type = props.knowledge_type,
  f.confidence = props.confidence,
  f.created_at = props.consolidated_at
ON MATCH SET
  f.confidence = CASE
      WHEN coalesce(props.confidence,0.0) >= coalesce(f.confidence,0.0)
      THEN props.confidence
      ELSE f.confidence
  END,
  f.slot_key = coalesce(f.slot_key, props.slot_key, toString(source.name) + ':' + type(r)),
  f.turn_id = props.turn_id,
  f.recorded_turn_id = props.recorded_turn_id,
  f.event_turn_offset = props.event_turn_offset,
  f.event_turn_id = props.event_turn_id,
  f.virtual_time = props.virtual_time,
  f.event_timestamp = props.event_timestamp,
  f.updated_at = props.consolidated_at

MERGE (f)-[:SUBJECT]->(source)
MERGE (f)-[:OBJECT]->(target)

MERGE (e:Event {event_id: props.event_id})
SET e.channel = props.channel,
    e.agent_name = props.agent_name,
    e.virtual_time = props.virtual_time,
    e.turn_id = props.turn_id,
    e.recorded_turn_id = props.recorded_turn_id,
    e.event_turn_offset = props.event_turn_offset,
    e.event_turn_id = props.event_turn_id,
    e.event_timestamp = props.event_timestamp,
    e.updated_at = props.consolidated_at

MERGE (e)-[:HAS_FACT]->(f)
WITH e, props
MATCH (u2:TextUnit {name: props.evidence_source_unit})
MERGE (e)-[:EVIDENCED_BY]->(u2)

RETURN count(*)
""".strip()
        ltss.update_graph(cypher_props_as_edges, parameters={"batch": batch})

    # ----------------------------
    # C2) Relationships（structured_response.relationships）
    # ----------------------------
    if getattr(structured_response, "relationships", None):
        rels_to_write = []
        for rel in structured_response.relationships:
            props = rel.properties if isinstance(getattr(rel, "properties", None), dict) else {}
            props = dict(props or {})

            extracted_time = props.get("event_timestamp")
            if extracted_time and str(extracted_time).lower() not in ["", "none", "null", "unknown"]:
                final_event_time = extracted_time
            else:
                final_event_time = "unknown"

            rel_type = _safe_ident(getattr(rel, "type", None) or "RELATED_TO", "RELATED_TO")
            s_label = _safe_ident(getattr(rel, "source_node_label", None) or "Concept", "Concept")
            t_label = _safe_ident(getattr(rel, "target_node_label", None) or "Concept", "Concept")

            s_name = str(getattr(rel, "source_node_name", "") or "").strip()
            t_name = str(getattr(rel, "target_node_name", "") or "").strip()
            if not s_name or not t_name:
                continue

            source_of_belief = props.get("source_of_belief", "reflection")
            try:
                props["confidence"] = float(props.get("confidence", 0.9))
            except Exception:
                props["confidence"] = 0.9

            if "slot_key" not in props and rel_type in _UPDATE_REL_TYPES:
                props["slot_key"] = f"{s_label}:{s_name}:{rel_type}"

            event_id = _make_event_id(agent_name, channel, virtual_timestamp, final_event_time, text_unit_id)
            turn_fields = _compute_event_turn_fields(recorded_turn_id, props)

            props.update(
                {
                    "consolidated_at": real_timestamp,
                    "virtual_time": virtual_timestamp,
                    "turn_id": recorded_turn_id,
                    "recorded_turn_id": turn_fields["recorded_turn_id"],
                    "event_turn_offset": turn_fields["event_turn_offset"],
                    "event_turn_id": turn_fields["event_turn_id"],
                    "event_step": turn_step,
                    "event_timestamp": final_event_time,
                    "event_id": event_id,
                    "evidence_source_unit": text_unit_id,
                    "source_of_belief": source_of_belief,
                    "is_current": True,
                    "should_be_current": True,
                    "source_rank": 2.0 if source_of_belief == "ground_truth" else 1.0,
                    "agent_name": agent_name,
                    "channel": channel,
                }
            )

            base_key = f"{s_label}:{s_name}:{rel_type}:{t_label}:{t_name}"
            props["belief_key"] = f"{channel}:{agent_name}:{base_key}"

            rels_to_write.append(
                {
                    "s_name": s_name,
                    "s_label": s_label,
                    "t_name": t_name,
                    "t_label": t_label,
                    "type": rel_type,
                    "props": serialize_props(props),
                }
            )

        groups = defaultdict(list)
        for r in rels_to_write:
            groups[(r["s_label"], r["t_label"], r["type"])].append(r)

        for (s_label, t_label, rel_type), batch in groups.items():
            s_label_safe = _safe_ident(s_label, "Concept")
            t_label_safe = _safe_ident(t_label, "Concept")
            rel_type_safe = _safe_ident(rel_type, "RELATED_TO")

            cypher_rels = f"""
UNWIND $batch AS row
MERGE (source:`{s_label_safe}` {{name: row.s_name}})
MERGE (target:`{t_label_safe}` {{name: row.t_name}})
WITH source, target, row
MATCH (u:TextUnit {{name: row.props.evidence_source_unit}})
MERGE (source)-[:FROM_SOURCE]->(u)
MERGE (target)-[:FROM_SOURCE]->(u)

WITH source, target, row, row.props AS props
OPTIONAL MATCH (source)-[old:`{rel_type_safe}`]->(other)
WHERE old.belief_key = props.belief_key
  AND coalesce(old.is_current,false) = true

WITH source, target, old, other, row, props,
CASE
  WHEN old IS NULL THEN true
  WHEN coalesce(props.source_rank, 1.0) > coalesce(old.source_rank, 1.0) THEN true
  WHEN coalesce(props.source_rank, 1.0) < coalesce(old.source_rank, 1.0) THEN false
  WHEN coalesce(props.event_step, -1) > coalesce(old.event_step, -1) THEN true
  WHEN coalesce(props.event_step, -1) < coalesce(old.event_step, -1) THEN false
  WHEN props.event_timestamp <> 'unknown' AND old.event_timestamp <> 'unknown' AND props.event_timestamp > old.event_timestamp THEN true
  WHEN coalesce(props.confidence, 0.0) >= coalesce(old.confidence, 0.0) THEN true
  ELSE false
END AS new_wins

FOREACH (_ IN CASE WHEN old IS NOT NULL AND new_wins = true AND other.name <> row.t_name THEN [1] ELSE [] END |
  SET old.is_current = false,
      old.deprecated_at = props.consolidated_at
)

MERGE (source)-[r:`{rel_type_safe}` {{belief_key: props.belief_key}}]->(target)
ON CREATE SET
  r += props,
  r.created_at = props.consolidated_at,
  r.is_current = new_wins
ON MATCH SET
  r.confidence = CASE WHEN new_wins = true THEN props.confidence ELSE r.confidence END,
  r.consolidated_at = props.consolidated_at,
  r.virtual_time = props.virtual_time,
  r.turn_id = props.turn_id,
  r.recorded_turn_id = props.recorded_turn_id,
  r.event_turn_offset = props.event_turn_offset,
  r.event_turn_id = props.event_turn_id,
  r.event_step = props.event_step,
  r.event_timestamp = props.event_timestamp,
  r.event_id = props.event_id,
  r.is_current = CASE WHEN new_wins = true THEN true ELSE r.is_current END

WITH source, target, row, props, r,
     toString(props.belief_key) AS bk

MERGE (f:Fact {{belief_key: bk}})
ON CREATE SET
  f.type = type(r),
  f.slot_key = coalesce(props.slot_key, toString(source.name) + ':' + type(r)),
  f.channel = props.channel,
  f.agent_name = props.agent_name,
  f.virtual_time = props.virtual_time,
  f.turn_id = props.turn_id,
  f.recorded_turn_id = props.recorded_turn_id,
  f.event_turn_offset = props.event_turn_offset,
  f.event_turn_id = props.event_turn_id,
  f.event_timestamp = props.event_timestamp,
  f.source_of_belief = props.source_of_belief,
  f.knowledge_type = props.knowledge_type,
  f.confidence = props.confidence,
  f.created_at = props.consolidated_at
ON MATCH SET
  f.confidence = CASE
      WHEN coalesce(props.confidence,0.0) >= coalesce(f.confidence,0.0)
      THEN props.confidence
      ELSE f.confidence
  END,
  f.slot_key = coalesce(f.slot_key, props.slot_key, toString(source.name) + ':' + type(r)),
  f.turn_id = props.turn_id,
  f.recorded_turn_id = props.recorded_turn_id,
  f.event_turn_offset = props.event_turn_offset,
  f.event_turn_id = props.event_turn_id,
  f.virtual_time = props.virtual_time,
  f.event_timestamp = props.event_timestamp,
  f.updated_at = props.consolidated_at

MERGE (f)-[:SUBJECT]->(source)
MERGE (f)-[:OBJECT]->(target)

MERGE (e:Event {{event_id: props.event_id}})
SET e.channel = props.channel,
    e.agent_name = props.agent_name,
    e.virtual_time = props.virtual_time,
    e.turn_id = props.turn_id,
    e.recorded_turn_id = props.recorded_turn_id,
    e.event_turn_offset = props.event_turn_offset,
    e.event_turn_id = props.event_turn_id,
    e.event_timestamp = props.event_timestamp,
    e.updated_at = props.consolidated_at

MERGE (e)-[:HAS_FACT]->(f)
WITH e, props
MATCH (u2:TextUnit {{name: props.evidence_source_unit}})
MERGE (e)-[:EVIDENCED_BY]->(u2)

RETURN count(*)
""".strip()
            ltss.update_graph(cypher_rels, parameters={"batch": batch, "agent_name": agent_name, "channel": channel})

    # --- embedding profiling summary ---
    if _emb_calls > 0:
        logger.info(
            f"[ltss_writer][embed] summary calls={_emb_calls} total_sec={_emb_total_s:.3f} "
            f"textunit_sec={_emb_tu_s:.3f} node_calls={_emb_node_calls} node_sec={_emb_node_s:.3f}"
        )
