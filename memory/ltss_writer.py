# -*- coding: utf-8 -*-
import datetime
import logging
import os
from collections import defaultdict
from typing import Any, Dict, Optional, List
import json
import re
import time

from config import CONSOLIDATED_REL_CONFIDENCE, CONSOLIDATED_ASSERTS_CONFIDENCE

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

# 事件型关系：保留最早的 turn_id（因为事件只发生一次，后续提到的都是回忆）
_EVENT_REL_TYPES = {
    "PARTICIPATED_IN",
    "ATTENDED",
    "WITNESSED",
    "EXPERIENCED",
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


def _validate_structured_response(structured_response) -> Dict[str, int]:
    """
    Lightweight guard before writing to Neo4j.
    Drops invalid nodes/relationships/claims and normalizes empty fields.
    """
    stats = {
        "nodes_dropped": 0,
        "rels_dropped": 0,
        "claims_dropped": 0,
        "facts_dropped": 0,
        "insights_dropped": 0,
    }
    if not structured_response:
        return stats

    # Nodes
    cleaned_nodes = []
    for n in getattr(structured_response, "nodes", []) or []:
        name = str(getattr(n, "name", "") or "").strip()
        if not name:
            stats["nodes_dropped"] += 1
            continue
        label = getattr(n, "label", None)
        if not isinstance(label, str) or not label.strip():
            try:
                setattr(n, "label", "Concept")
            except Exception:
                pass
        props = getattr(n, "properties", None)
        if not isinstance(props, dict):
            try:
                setattr(n, "properties", {})
            except Exception:
                pass
        cleaned_nodes.append(n)
    try:
        structured_response.nodes = cleaned_nodes
    except Exception:
        pass

    # Relationships
    cleaned_rels = []
    for r in getattr(structured_response, "relationships", []) or []:
        rel_type = str(getattr(r, "type", "") or "").strip()
        s_name = str(getattr(r, "source_node_name", "") or "").strip()
        t_name = str(getattr(r, "target_node_name", "") or "").strip()
        if not (rel_type and s_name and t_name):
            stats["rels_dropped"] += 1
            continue
        s_label = getattr(r, "source_node_label", None)
        if not isinstance(s_label, str) or not s_label.strip():
            try:
                setattr(r, "source_node_label", "Concept")
            except Exception:
                pass
        t_label = getattr(r, "target_node_label", None)
        if not isinstance(t_label, str) or not t_label.strip():
            try:
                setattr(r, "target_node_label", "Concept")
            except Exception:
                pass
        props = getattr(r, "properties", None)
        if not isinstance(props, dict):
            try:
                setattr(r, "properties", {})
            except Exception:
                pass
        cleaned_rels.append(r)
    try:
        structured_response.relationships = cleaned_rels
    except Exception:
        pass

    # Claims
    cleaned_claims = []
    for c in getattr(structured_response, "claims", []) or []:
        c_dict = c.model_dump() if hasattr(c, "model_dump") else (c if isinstance(c, dict) else None)
        if not isinstance(c_dict, dict):
            stats["claims_dropped"] += 1
            continue
        text = str(c_dict.get("text") or "").strip()
        if not text:
            stats["claims_dropped"] += 1
            continue
        c_dict["text"] = text
        if c_dict.get("event_turn_offset") is not None:
            try:
                c_dict["event_turn_offset"] = int(float(c_dict.get("event_turn_offset")))
            except Exception:
                c_dict["event_turn_offset"] = None
        cleaned_claims.append(c_dict)
    try:
        structured_response.claims = cleaned_claims
    except Exception:
        pass

    # Facts / insights
    cleaned_facts = []
    for f in getattr(structured_response, "facts", []) or []:
        if isinstance(f, str) and f.strip():
            cleaned_facts.append(f.strip())
        else:
            stats["facts_dropped"] += 1
    try:
        structured_response.facts = cleaned_facts
    except Exception:
        pass

    cleaned_insights = []
    for ins in getattr(structured_response, "insights", []) or []:
        if isinstance(ins, str) and ins.strip():
            cleaned_insights.append(ins.strip())
        else:
            stats["insights_dropped"] += 1
    try:
        structured_response.insights = cleaned_insights
    except Exception:
        pass

    return stats


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
# 实体消歧（Entity Disambiguation）
# ----------------------------
# 常见别名映射（可扩展）
_ENTITY_ALIASES = {
    # 人称代词 -> 统一
    "i": "User",
    "me": "User",
    "my": "User",
    "myself": "User",
    "we": "User",
    "us": "User",
    "our": "User",
    # 常见缩写
    "nyc": "New York City",
    "la": "Los Angeles",
    "sf": "San Francisco",
    "uk": "United Kingdom",
    "usa": "United States",
    "us": "United States",
}

# 需要保持原样的实体（不做标准化）
_PRESERVE_CASE_PATTERNS = [
    r"^[A-Z]{2,}$",  # 全大写缩写如 NASA, FBI
    r"^[A-Z][a-z]+[A-Z]",  # 驼峰如 iPhone, eBay
]


def normalize_entity_name(name: str, *, preserve_case: bool = False) -> str:
    """
    实体名称标准化/消歧：
    1. 去除首尾空格和多余空格
    2. 别名映射
    3. 标题化（首字母大写）
    """
    if not isinstance(name, str):
        return str(name or "").strip()
    
    s = " ".join(name.split()).strip()
    if not s:
        return s
    
    # 检查别名映射
    s_lower = s.lower()
    if s_lower in _ENTITY_ALIASES:
        return _ENTITY_ALIASES[s_lower]
    
    # 检查是否需要保持原样
    import re
    for pattern in _PRESERVE_CASE_PATTERNS:
        if re.match(pattern, s):
            return s
    
    # 标题化（但保留全大写词）
    if preserve_case:
        return s
    
    words = s.split()
    result = []
    for w in words:
        if w.isupper() and len(w) > 1:
            result.append(w)  # 保留全大写
        else:
            result.append(w.capitalize())
    
    return " ".join(result)


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

    # 确保 embedding_model 可用
    if embedding_model is None:
        from utils.embedding import get_embedding_model
        embedding_model = get_embedding_model()
        logger.info(f"[ltss_writer] embedding_model was None, loaded: {type(embedding_model).__name__}")

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
            f"insights={len(getattr(structured_response, 'insights', []) or [])} "
            f"claims={len(getattr(structured_response, 'claims', []) or [])}"
        )

    validate_stats = _validate_structured_response(structured_response)
    strict_validation = os.getenv("STRICT_WRITE_VALIDATION", "0").strip() in ("1", "true", "yes")
    if (strict_validation or _DEBUG_LTSS) and any(v > 0 for v in validate_stats.values()):
        logger.info(
            "[ltss_writer][debug] validate_drops "
            + ", ".join([f"{k}={v}" for k, v in validate_stats.items() if v])
        )
        if strict_validation:
            raise ValueError(f"[ltss_writer] strict validation failed: {validate_stats}")

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
            # 验证 embedding 是否有效
            if tu_emb and len(tu_emb) > 0:
                logger.info(f"[ltss_writer][embed] mode=single kind=textunit n=1 sec={_dt:.3f} dim={len(tu_emb)}")
            else:
                logger.warning(f"[ltss_writer][embed] TextUnit embedding returned empty! sec={_dt:.3f}")
                tu_emb = None
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

            # ✅ 实体消歧：标准化实体名称
            nname = normalize_entity_name(nname)
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
    # C1) Facts / Insights / Claims → Property-as-edge
    # ----------------------------
    props_edge_rows = []

    def _push_prop_fact(
        label: str,
        text: str,
        *,
        confidence: float,
        knowledge_type: str,
        source_of_belief: str,
        event_time_text: Optional[str] = None,
        event_turn_offset: Optional[int] = None,
        event_timestamp: Optional[str] = None,
    ) -> None:
        s = (text or "").strip()
        if not s:
            return
        capped_conf = float(confidence)
        if label == "Fact":
            try:
                capped_conf = min(capped_conf, float(CONSOLIDATED_ASSERTS_CONFIDENCE))
            except Exception:
                capped_conf = float(CONSOLIDATED_ASSERTS_CONFIDENCE)

        props_edge_rows.append(
            {
                "label": label,
                "text": s,
                "confidence": float(capped_conf),
                "knowledge_type": knowledge_type,
                "source_of_belief": source_of_belief,
                "event_time_text": event_time_text,
                "event_turn_offset": event_turn_offset,
                "event_timestamp": event_timestamp,
            }
        )

    # Claims (structured objects from LLM schema)
    claims = getattr(structured_response, "claims", None)
    if isinstance(claims, list):
        for c in claims:
            if hasattr(c, "model_dump"):
                c = c.model_dump()
            if not isinstance(c, dict):
                continue
            event_time_text = c.get("event_time_text")
            _push_prop_fact(
                "Fact",
                c.get("text", ""),
                confidence=float(c.get("confidence", 0.9)),
                knowledge_type=str(c.get("knowledge_type") or "observed_fact"),
                source_of_belief=str(c.get("source_of_belief") or "user_statement"),
                event_time_text=event_time_text,
                event_turn_offset=c.get("event_turn_offset"),
                event_timestamp=(session_time_iso or None) if not event_time_text else None,
            )

    # Backward compatible facts/insights (strings)
    if getattr(structured_response, "facts", None):
        for fact in structured_response.facts:
            # 支持多种格式：
            # 1. 字符串 "fact text"
            # 2. 带来源前缀 "[user] fact text" 或 "[assistant] fact text"
            # 3. JSON 编码的字典 '{"text": "...", "source": "user/assistant", "event_time": "YYYY-MM-DD"}'
            fact_event_time = None  # LLM 计算的事件时间
            
            if isinstance(fact, str):
                fact_text = fact
                fact_source = "user"  # 默认来源
                
                # 尝试解析 JSON 格式
                if fact_text.startswith("{") and fact_text.endswith("}"):
                    try:
                        fact_obj = json.loads(fact_text)
                        fact_text = fact_obj.get("text", "")
                        fact_source = fact_obj.get("source", "user")
                        fact_event_time = fact_obj.get("event_time", "")
                    except json.JSONDecodeError:
                        pass  # 不是有效的 JSON，当作普通字符串处理
                
                # 解析 [source] 前缀
                if fact_text.startswith("[user]"):
                    fact_text = fact_text[6:].strip()
                    fact_source = "user"
                elif fact_text.startswith("[assistant]"):
                    fact_text = fact_text[11:].strip()
                    fact_source = "assistant"
            elif isinstance(fact, dict):
                fact_text = fact.get("text", "")
                fact_source = fact.get("source", "user")
                fact_event_time = fact.get("event_time", "")  # ✅ 获取 LLM 计算的事件时间
            else:
                continue
            
            if not fact_text:
                continue
            
            # ✅ 使用 LLM 计算的 event_time，如果没有则使用 session_time
            final_event_timestamp = fact_event_time if fact_event_time else (session_time_iso or None)
            
            _push_prop_fact(
                "Fact",
                fact_text,
                confidence=1.0,
                knowledge_type="observed_fact",
                source_of_belief=fact_source,  # ✅ 使用 source 字段区分来源
                event_timestamp=final_event_timestamp,
            )

    if getattr(structured_response, "insights", None):
        for ins in structured_response.insights:
            if not isinstance(ins, str):
                continue
            _push_prop_fact(
                "Insight",
                ins,
                confidence=0.9,
                knowledge_type="logical_inference",
                source_of_belief="reflection",
                event_timestamp=session_time_iso or None,
            )

    if props_edge_rows:
        batch = []
        
        # ✅ 为简单事实生成 embedding（批量处理，分批避免 API 限制）
        fact_texts = [row["text"] for row in props_edge_rows if row.get("text")]
        fact_embeddings: List[Optional[List[float]]] = []
        
        # 分批处理，每批最多 60 个（留一些余量，API 限制是 64）
        BATCH_SIZE = 60
        
        if embedding_model and fact_texts:
            try:
                _t0 = time.time()
                all_embeddings = []
                
                for i in range(0, len(fact_texts), BATCH_SIZE):
                    batch_texts = fact_texts[i:i + BATCH_SIZE]
                    if hasattr(embedding_model, "embed_documents"):
                        batch_embs = embedding_model.embed_documents(batch_texts) or []
                    else:
                        batch_embs = [embedding_model.embed_query(t) for t in batch_texts]
                    all_embeddings.extend(batch_embs)
                
                fact_embeddings = all_embeddings
                _dt = time.time() - _t0
                _emb_calls += 1
                _emb_total_s += _dt
                
                # 验证 embedding 是否有效
                valid_count = sum(1 for e in fact_embeddings if e and len(e) > 0)
                logger.info(f"[ltss_writer][embed] mode=batch kind=simple_fact n={len(fact_embeddings)} valid={valid_count} sec={_dt:.3f}")
            except Exception as e:
                logger.warning(f"[ltss_writer][embed] simple_fact batch failed: {e}")
                fact_embeddings = []
        
        for idx, row in enumerate(props_edge_rows):
            lbl = row["label"]
            text = row["text"]
            conf = row["confidence"]
            s_label = "TextFact"
            t_label = lbl
            s_name = text_unit_id
            t_name = text
            
            # ✅ 获取该简单事实的 embedding
            fact_emb = fact_embeddings[idx] if idx < len(fact_embeddings) else None

            props = {
                "confidence": float(conf),
                "knowledge_type": row.get("knowledge_type") or ("observed_fact" if lbl == "Fact" else "logical_inference"),
                "source_of_belief": row.get("source_of_belief") or "reflection",
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
            if row.get("event_time_text"):
                props["event_time_text"] = row.get("event_time_text")
            if row.get("event_turn_offset") is not None:
                props["event_turn_offset"] = row.get("event_turn_offset")
            if row.get("event_timestamp") and str(row.get("event_timestamp")).lower() not in ["", "none", "null", "unknown"]:
                props["event_timestamp"] = row.get("event_timestamp")

            # Ensure TextFact facts don't collapse into one slot during soft-update grouping.
            props["slot_key"] = f"{s_name}:ASSERTS:{t_name}"

            turn_fields = _compute_event_turn_fields(recorded_turn_id, props)
            props.update(turn_fields)

            event_id = _make_event_id(agent_name, channel, virtual_timestamp, "unknown", text_unit_id)
            props["event_id"] = event_id

            base_key = f"{s_label}:{s_name}:ASSERTS:{t_label}:{t_name}"
            # 方案A：belief_key 加入 turn_id，每次提及创建独立记录
            props["belief_key"] = f"{recorded_turn_id}:{channel}:{agent_name}:{base_key}"

            batch.append(
                {
                    "s_name": s_name,
                    "s_label": s_label,
                    "t_name": t_name,
                    "t_label": t_label,
                    "type": "ASSERTS",
                    "props": serialize_props(props),
                    "embedding": fact_emb,  # ✅ 添加 embedding
                }
            )

        # ✅ 调试：检查 batch 中的 embedding
        if _DEBUG_LTSS:
            emb_stats = [len(b.get("embedding") or []) for b in batch]
            logger.info(f"[ltss_writer][debug] ASSERTS batch size={len(batch)} embedding_sizes={emb_stats[:5]}...")

        # 使用模板文件中的 Cypher
        from memory.cypher_templates import CYPHER_ASSERTS_EDGE
        ltss.update_graph(CYPHER_ASSERTS_EDGE, parameters={"batch": batch})

    # ----------------------------
    # C2) Relationships（structured_response.relationships）
    # ----------------------------
    if getattr(structured_response, "relationships", None):
        rels_to_write = []
        for rel in structured_response.relationships:
            props = rel.properties if isinstance(getattr(rel, "properties", None), dict) else {}
            props = dict(props or {})

            extracted_time = props.get("event_timestamp")
            # ✅ 时间戳标准化：如果LLM提取的是相对时间（today/yesterday等），用session_time_iso替换
            if extracted_time and str(extracted_time).lower() not in ["", "none", "null", "unknown"]:
                extracted_lower = str(extracted_time).lower().strip()
                # 检测相对时间词
                relative_time_words = ["today", "yesterday", "tomorrow", "now", "just", "recently", 
                                       "last week", "this week", "last month", "this month"]
                is_relative = any(word in extracted_lower for word in relative_time_words)
                
                if is_relative and session_time_iso:
                    # 相对时间 -> 用session的绝对时间
                    final_event_time = session_time_iso
                    if _DEBUG_LTSS:
                        logger.info(f"[ltss_writer][time_normalize] '{extracted_time}' -> '{session_time_iso}'")
                else:
                    final_event_time = extracted_time
            elif session_time_iso:
                # 没有提取到时间，用session时间作为默认值
                final_event_time = session_time_iso
            else:
                final_event_time = "unknown"

            rel_type = _safe_ident(getattr(rel, "type", None) or "RELATED_TO", "RELATED_TO")
            s_label = _safe_ident(getattr(rel, "source_node_label", None) or "Concept", "Concept")
            t_label = _safe_ident(getattr(rel, "target_node_label", None) or "Concept", "Concept")

            s_name = str(getattr(rel, "source_node_name", "") or "").strip()
            t_name = str(getattr(rel, "target_node_name", "") or "").strip()
            if not s_name or not t_name:
                continue

            # ✅ 实体消歧：标准化实体名称
            s_name = normalize_entity_name(s_name)
            t_name = normalize_entity_name(t_name)
            if not s_name or not t_name:
                continue

            source_of_belief = props.get("source_of_belief", "reflection")
            try:
                props["confidence"] = float(props.get("confidence", CONSOLIDATED_REL_CONFIDENCE))
            except Exception:
                props["confidence"] = float(CONSOLIDATED_REL_CONFIDENCE)

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
                    # 添加 session_time 字段（对话发生时间）
                    "session_time": session_time_iso or session_time_raw or None,
                }
            )

            base_key = f"{s_label}:{s_name}:{rel_type}:{t_label}:{t_name}"
            # 方案A：belief_key 加入 turn_id，每次提及创建独立记录
            # 这样同一事实在不同 turn 提及会创建不同的 Fact 节点，避免信息丢失
            props["belief_key"] = f"{recorded_turn_id}:{channel}:{agent_name}:{base_key}"
            
            # ✅ 生成简单事实文本（自然语言）
            from utils.consolidated_extractor import _relationship_to_natural_language
            fact_text = _relationship_to_natural_language(
                source=s_name,
                target=t_name,
                rel_type=rel_type,
                source_type=s_label,
                target_type=t_label,
                description=props.get("description", ""),
            )
            props["text"] = fact_text  # 添加简单事实文本

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

            # 使用模板文件中的 Cypher
            from memory.cypher_templates import cypher_upsert_relationship
            cypher_rels = cypher_upsert_relationship(s_label_safe, t_label_safe, rel_type_safe)
            ltss.update_graph(cypher_rels, parameters={"batch": batch, "agent_name": agent_name, "channel": channel})

    # --- embedding profiling summary ---
    if _emb_calls > 0:
        logger.info(
            f"[ltss_writer][embed] summary calls={_emb_calls} total_sec={_emb_total_s:.3f} "
            f"textunit_sec={_emb_tu_s:.3f} node_calls={_emb_node_calls} node_sec={_emb_node_s:.3f}"
        )
