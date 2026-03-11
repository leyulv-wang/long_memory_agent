# -*- coding: utf-8 -*-
"""
Cypher 模板集合 - 用于 LTSS 写入操作
包含软更新逻辑：保留历史信息，控制膨胀
"""

# ============================================================
# 软更新配置
# ============================================================
SOFT_UPDATE_MAX_HISTORY = 10  # turn_history 最多保留条数
SOFT_UPDATE_MAX_EVENT_TIMES = 10  # event_time_history 最多保留条数


# ============================================================
# Document / Chunk 写入
# ============================================================
CYPHER_UPSERT_DOCUMENT = """
MERGE (d:__Document__ {id: $doc_id})
ON CREATE SET d.title = $title, d.raw_content = $raw
SET d.agent_name = $agent_name,
    d.updated_at = $updated_at,
    d.session_id = $session_id,
    d.session_time_raw = $session_time_raw,
    d.session_time_iso = $session_time_iso
"""

CYPHER_UPSERT_CHUNK = """
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


# ============================================================
# TextUnit 写入
# ============================================================
CYPHER_UPSERT_TEXTUNIT = """
MERGE (u:TextUnit {name: $name})
SET u.content = $content,
    u.virtual_time = $vts,
    u.turn_id = $turn_id,
    u.real_time = $rts,
    u.channel = $channel,
    u.agent_name = $agent_name,
    u.embedding = $embedding
"""


# ============================================================
# Node 批量写入（带 channels 集合）
# ============================================================
def cypher_upsert_nodes(label: str) -> str:
    """生成节点 MERGE Cypher，label 动态传入"""
    return f"""
UNWIND $batch AS row
MERGE (n:`{label}` {{name: row.name}})
SET n += row.props
WITH n, $channel AS ch
SET n.channels = apoc.coll.toSet(coalesce(n.channels, []) + [ch])
RETURN count(*)
""".strip()


def cypher_link_chunk_entity(label: str) -> str:
    """链接 Chunk 到实体"""
    return f"""
UNWIND $rows AS row
MATCH (c:__Chunk__ {{id: $chunk_id}})
MATCH (e:`{label}` {{name: row.name}})
MERGE (c)-[:HAS_ENTITY]->(e)
""".strip()


# ============================================================
# ASSERTS 关系写入（Facts/Insights/Claims）- 带软更新
# ============================================================
CYPHER_ASSERTS_EDGE = """
UNWIND $batch AS row
MERGE (source:TextUnit {name: row.s_name})
MERGE (target:`__Node__` {name: row.t_name})
SET target._node_type = row.t_label,
    target.embedding = row.embedding,
    target.event_timestamp = coalesce(target.event_timestamp, row.props.event_timestamp, row.props.session_time),
    target.session_time = coalesce(target.session_time, row.props.session_time),
    target.turn_id = coalesce(target.turn_id, row.props.turn_id),
    target.virtual_time = coalesce(target.virtual_time, row.props.virtual_time)

// ✅ 修复：使用 OPTIONAL MATCH 避免因找不到节点而中断
WITH source, target, row
OPTIONAL MATCH (u:TextUnit {name: row.props.evidence_source_unit})
FOREACH (_ IN CASE WHEN u IS NOT NULL THEN [1] ELSE [] END |
  MERGE (source)-[:FROM_SOURCE]->(u)
  MERGE (target)-[:FROM_SOURCE]->(u)
)

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
  r.is_current = new_wins,
  r.first_turn_id = props.turn_id,
  r.first_event_timestamp = props.event_timestamp,
  r.first_virtual_time = props.virtual_time,
  r.last_turn_id = props.turn_id,
  r.last_event_timestamp = props.event_timestamp,
  r.mention_count = 1
ON MATCH SET
  r.confidence = CASE WHEN new_wins = true THEN props.confidence ELSE r.confidence END,
  r.consolidated_at = props.consolidated_at,
  r.first_turn_id = coalesce(r.first_turn_id, props.turn_id),
  r.first_event_timestamp = CASE 
    WHEN r.first_event_timestamp IS NULL OR r.first_event_timestamp = 'unknown' THEN props.event_timestamp 
    ELSE r.first_event_timestamp 
  END,
  r.first_virtual_time = coalesce(r.first_virtual_time, props.virtual_time),
  r.last_turn_id = props.turn_id,
  r.last_event_timestamp = props.event_timestamp,
  r.mention_count = coalesce(r.mention_count, 0) + 1,
  r.turn_id = coalesce(r.first_turn_id, props.turn_id),
  r.recorded_turn_id = props.recorded_turn_id,
  r.event_turn_offset = props.event_turn_offset,
  r.event_turn_id = props.event_turn_id,
  r.event_step = props.event_step,
  r.event_timestamp = props.event_timestamp,
  r.event_id = props.event_id,
  r.is_current = CASE WHEN new_wins = true THEN true ELSE r.is_current END

WITH source, target, row, props, r,
     toString(props.belief_key) AS bk

MERGE (f:Fact {belief_key: bk})
ON CREATE SET
  f.type = type(r),
  f.text = props.text,
  f.slot_key = coalesce(props.slot_key, toString(source.name) + ':' + type(r)),
  f.channel = props.channel,
  f.agent_name = props.agent_name,
  f.virtual_time = props.virtual_time,
  f.turn_id = props.turn_id,
  f.recorded_turn_id = props.recorded_turn_id,
  f.event_turn_offset = props.event_turn_offset,
  f.event_turn_id = props.event_turn_id,
  f.event_timestamp = props.event_timestamp,
  f.session_time = props.session_time,
  f.source_of_belief = props.source_of_belief,
  f.knowledge_type = props.knowledge_type,
  f.confidence = props.confidence,
  f.created_at = props.consolidated_at,
  f.first_turn_id = props.turn_id,
  f.first_event_time = props.event_timestamp,
  f.last_turn_id = props.turn_id,
  f.last_event_time = props.event_timestamp,
  f.mention_count = 1,
  f.turn_history = [props.turn_id]
ON MATCH SET
  f.text = coalesce(f.text, props.text),
  f.confidence = CASE
      WHEN coalesce(props.confidence,0.0) > coalesce(f.confidence,0.0)
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
  f.updated_at = props.consolidated_at,
  f.last_turn_id = props.turn_id,
  f.last_event_time = props.event_timestamp,
  f.mention_count = coalesce(f.mention_count, 0) + 1,
  f.turn_history = (coalesce(f.turn_history, []) + [props.turn_id])[-10..]

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
WITH e, f, props
MATCH (u2:TextUnit {name: props.evidence_source_unit})
MERGE (e)-[:EVIDENCED_BY]->(u2)

// ✅ 建立 Fact 与简单事实的关联（通过 turn_id 和 channel）
// 注意：这里的 f 是 ASSERTS 关系创建的 Fact 节点（用于 reify 简单事实）
// HAS_SIMPLE_FACT 关系应该在三元组写入时创建，而不是在简单事实写入时
// 因此这里暂时移除这个逻辑，改为在 cypher_upsert_relationship 中处理

RETURN count(*)
"""


# ============================================================
# Relationship 写入 - 带软更新
# ============================================================
def cypher_upsert_relationship(s_label: str, t_label: str, rel_type: str) -> str:
    """生成关系 MERGE Cypher，带软更新逻辑"""
    return f"""
UNWIND $batch AS row
MERGE (source:`{s_label}` {{name: row.s_name}})
MERGE (target:`{t_label}` {{name: row.t_name}})
WITH source, target, row
MATCH (u:TextUnit {{name: row.props.evidence_source_unit}})
MERGE (source)-[:FROM_SOURCE]->(u)
MERGE (target)-[:FROM_SOURCE]->(u)

WITH source, target, row, row.props AS props
OPTIONAL MATCH (source)-[old:`{rel_type}`]->(other)
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

MERGE (source)-[r:`{rel_type}` {{belief_key: props.belief_key}}]->(target)
ON CREATE SET
  r += props,
  r.created_at = props.consolidated_at,
  r.is_current = new_wins,
  r.first_turn_id = props.turn_id,
  r.first_event_timestamp = props.event_timestamp,
  r.first_virtual_time = props.virtual_time,
  r.last_turn_id = props.turn_id,
  r.last_event_timestamp = props.event_timestamp,
  r.mention_count = 1
ON MATCH SET
  r.confidence = CASE WHEN new_wins = true THEN props.confidence ELSE r.confidence END,
  r.consolidated_at = props.consolidated_at,
  r.first_turn_id = coalesce(r.first_turn_id, props.turn_id),
  r.first_event_timestamp = CASE 
    WHEN r.first_event_timestamp IS NULL OR r.first_event_timestamp = 'unknown' THEN props.event_timestamp 
    ELSE r.first_event_timestamp 
  END,
  r.first_virtual_time = coalesce(r.first_virtual_time, props.virtual_time),
  r.last_turn_id = props.turn_id,
  r.last_event_timestamp = props.event_timestamp,
  r.mention_count = coalesce(r.mention_count, 0) + 1,
  r.turn_id = coalesce(r.first_turn_id, props.turn_id)

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
  f.session_time = props.session_time,
  f.source_of_belief = props.source_of_belief,
  f.knowledge_type = props.knowledge_type,
  f.confidence = props.confidence,
  f.created_at = props.consolidated_at,
  f.first_turn_id = props.turn_id,
  f.first_event_time = props.event_timestamp,
  f.last_turn_id = props.turn_id,
  f.last_event_time = props.event_timestamp,
  f.mention_count = 1,
  f.turn_history = [props.turn_id]
ON MATCH SET
  f.confidence = CASE
      WHEN coalesce(props.confidence,0.0) > coalesce(f.confidence,0.0)
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
  f.session_time = coalesce(f.session_time, props.session_time),
  f.updated_at = props.consolidated_at,
  f.last_turn_id = props.turn_id,
  f.last_event_time = props.event_timestamp,
  f.mention_count = coalesce(f.mention_count, 0) + 1,
  f.turn_history = (coalesce(f.turn_history, []) + [props.turn_id])[-10..]

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
WITH e, f, props, source, target
MATCH (u2:TextUnit {{name: props.evidence_source_unit}})
MERGE (e)-[:EVIDENCED_BY]->(u2)

// ✅ 建立 Fact（三元组）与简单事实的关联
// 方案：基于文本匹配，检查简单事实是否包含三元组的主语或宾语
// 宽松匹配：只要包含主语或宾语之一即可（因为简单事实可能用不同的词描述同一概念）
WITH f, props, source, target
CALL (f, props, source, target) {{
  // 查找同一 turn 中的简单事实
  MATCH (tu:TextUnit {{turn_id: props.turn_id, channel: props.channel}})-[:ASSERTS]->(simple_fact:__Node__)
  WHERE simple_fact._node_type = 'Fact'
    // 宽松匹配：包含主语或宾语之一
    AND (
      toLower(simple_fact.name) CONTAINS toLower(source.name)
      OR toLower(simple_fact.name) CONTAINS toLower(target.name)
    )
  MERGE (f)-[:HAS_SIMPLE_FACT]->(simple_fact)
  RETURN count(*) AS linked
}}

RETURN count(*)
""".strip()
