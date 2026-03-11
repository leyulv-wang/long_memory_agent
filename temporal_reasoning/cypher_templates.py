# temporal_reasoning/cypher_templates.py
# -*- coding: utf-8 -*-

"""
Cypher templates for temporal reasoning.

Design goals:
- Parameter names are统一 snake_case:
  - anchor_keyword
  - keywords
  - limit_val
- Returned column names are统一给 executor.run_template 使用:
  - src, rel, tgt
  - event_time
  - source_of_belief
  - confidence
  - knowledge_type
  - evidence_source_unit
"""

# -----------------------------
# Common snippet: choose person
# -----------------------------
CHOOSE_PERSON = r"""
MATCH (u:Person)
WITH u
ORDER BY CASE
  WHEN toLower(u.name) = toLower($agent_name) THEN 0
  WHEN u.name = 'User' THEN 1
  ELSE 2
END
LIMIT 1
"""



# -----------------------------------
# FIRST_EVENT_AFTER_ANCHOR
# -----------------------------------
FIRST_EVENT_AFTER_ANCHOR = r"""
// 0) 选择一个 Person：优先 User，其次任意 Person
MATCH (u:Person)
WITH u
ORDER BY CASE WHEN u.name = 'User' THEN 0 ELSE 1 END
LIMIT 1

// 1) 找到最早的 anchor（按 event_step / TURN / ISO 排序）
MATCH (u)-[ra:EXPERIENCED|RECEIVED|PERFORMED]->(anchor:Event)
WHERE $anchor_keyword IS NOT NULL
  AND toLower(anchor.name) CONTAINS toLower($anchor_keyword)
WITH u, anchor, ra,
     coalesce(
       CASE
         WHEN coalesce(ra.virtual_time, '') STARTS WITH 'TURN'
         THEN toInteger(replace(coalesce(ra.virtual_time, ''), 'TURN', ''))
       END,
       CASE
         WHEN coalesce(ra.event_timestamp, '') =~ '\\d{4}-\\d{2}-\\d{2}.*'
         THEN date(left(coalesce(ra.event_timestamp, ''), 10)).epochDays
       END
     ) AS anchor_order
WHERE anchor_order IS NOT NULL
ORDER BY anchor_order ASC
LIMIT 1

// 2) 找到 anchor 之后最早发生的“问题/故障/维修类”事件
MATCH (u)-[r:EXPERIENCED|RECEIVED|PERFORMED]->(e:Event)
WITH u, r, e, anchor_order,
     coalesce(
       CASE
         WHEN coalesce(r.virtual_time, '') STARTS WITH 'TURN'
         THEN toInteger(replace(coalesce(r.virtual_time, ''), 'TURN', ''))
       END,
       CASE
         WHEN coalesce(r.event_timestamp, '') =~ '\\d{4}-\\d{2}-\\d{2}.*'
         THEN date(left(coalesce(r.event_timestamp, ''), 10)).epochDays
       END
     ) AS e_order,
     coalesce(r.event_timestamp, r.virtual_time, 'unknown') AS event_time
WITH u, r, e, anchor_order, e_order, event_time
WHERE e_order IS NOT NULL AND e_order > anchor_order

WITH u, r, e, e_order, event_time,
     CASE
       WHEN toLower(e.name) CONTAINS 'issue' THEN 0
       WHEN toLower(e.name) CONTAINS 'problem' THEN 1
       WHEN toLower(e.name) CONTAINS 'repair' THEN 2
       WHEN toLower(e.name) CONTAINS 'gps' THEN 3
       WHEN toLower(e.name) CONTAINS 'replacement' THEN 4
       ELSE 10
     END AS issue_rank

RETURN
  u.name AS src,
  type(r) AS rel,
  e.name AS tgt,
  event_time AS event_time,
  coalesce(r.source_of_belief, 'ground_truth') AS source_of_belief,
  coalesce(r.confidence, 1.0) AS confidence,
  coalesce(r.knowledge_type, 'observed_fact') AS knowledge_type,
  coalesce(r.evidence_source_unit, 'unknown') AS evidence_source_unit
ORDER BY issue_rank ASC, e_order ASC
LIMIT 1
"""


# -----------------------------------
# EVENT_DATES_BY_KEYWORDS
# -----------------------------------
EVENT_DATES_BY_KEYWORDS = rf"""
{CHOOSE_PERSON}
WITH u, $keywords AS kws
MATCH (u)-[r]->(t)
WHERE NOT 'TextUnit' IN labels(t)
  AND any(k IN kws WHERE
        toLower(coalesce(t.name, '')) CONTAINS toLower(k)
        OR toLower(type(r)) CONTAINS toLower(k)
  )
WITH t, r,
     coalesce(r.event_timestamp, r.virtual_time, 'unknown') AS event_time,
     coalesce(
       CASE WHEN coalesce(r.virtual_time, '') STARTS WITH 'TURN'
            THEN toInteger(replace(coalesce(r.virtual_time, ''), 'TURN', ''))
       END,
       CASE WHEN coalesce(r.event_timestamp, '') =~ '\\d{{4}}-\\d{{2}}-\\d{{2}}.*'
            THEN date(left(coalesce(r.event_timestamp, ''), 10)).epochDays
       END
     ) AS order_key
RETURN
  t.name AS src,
  type(r) AS rel,
  coalesce(event_time, 'unknown') AS tgt,
  event_time AS event_time,
  coalesce(r.source_of_belief, 'ground_truth') AS source_of_belief,
  coalesce(r.confidence, 1.0) AS confidence,
  coalesce(r.knowledge_type, 'observed_fact') AS knowledge_type,
  coalesce(r.evidence_source_unit, 'unknown') AS evidence_source_unit
ORDER BY order_key ASC
LIMIT $limit_val
"""


# -----------------------------------
# COUNT_DISTINCT_TARGETS_BY_KEYWORDS
# -----------------------------------
COUNT_DISTINCT_TARGETS_BY_KEYWORDS = rf"""
{CHOOSE_PERSON}
WITH u, $keywords AS kws
MATCH (u)-[r]->(t)
WHERE NOT 'TextUnit' IN labels(t)
  AND any(k IN kws WHERE
        toLower(coalesce(t.name, '')) CONTAINS toLower(k)
        OR toLower(type(r)) CONTAINS toLower(k)
  )
RETURN count(DISTINCT coalesce(t.name, '')) AS cnt
"""


# -----------------------------------
# COUNT_DISTINCT_TARGETS_TRIPLES
# -----------------------------------
COUNT_DISTINCT_TARGETS_TRIPLES = rf"""
{CHOOSE_PERSON}
WITH u, $keywords AS kws
MATCH (u)-[r]->(t)
WHERE NOT 'TextUnit' IN labels(t)
  AND any(k IN kws WHERE
        toLower(coalesce(t.name, '')) CONTAINS toLower(k)
        OR toLower(type(r)) CONTAINS toLower(k)
  )
RETURN
  u.name AS src,
  type(r) AS rel,
  t.name AS tgt,
  coalesce(r.source_of_belief, 'ground_truth') AS source_of_belief,
  coalesce(r.confidence, 1.0) AS confidence,
  coalesce(r.knowledge_type, 'observed_fact') AS knowledge_type,
  coalesce(r.evidence_source_unit, 'unknown') AS evidence_source_unit,
  coalesce(r.event_timestamp, r.virtual_time, 'unknown') AS event_time
LIMIT $limit_val
"""
