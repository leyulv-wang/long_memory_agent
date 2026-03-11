# -*- coding: utf-8 -*-
"""
巩固流程主模块

此模块负责将对话内容提取为结构化知识图谱。

提取器架构：
- 使用 consolidated_extractor.py（两阶段提取：实体→关系）
- 借鉴 GraphRAG + Instructor 的设计
"""
from __future__ import annotations

import json
import logging
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils.llm import invoke_json
from memory.structured_memory import KnowledgeGraphExtraction, Node, Relationship, Claim

# ============================================================================
# 提取器导入（统一使用 consolidated_extractor）
# ============================================================================

# 两阶段提取器（借鉴 GraphRAG + Instructor）
try:
    from utils.consolidated_extractor import (
        extract_to_kg_format as extract_kg_v2,
        consolidate_session_v2,
        _INSTRUCTOR_AVAILABLE,
    )
    _EXTRACTOR_AVAILABLE = _INSTRUCTOR_AVAILABLE
except ImportError:
    _EXTRACTOR_AVAILABLE = False
    extract_kg_v2 = None
    consolidate_session_v2 = None



logger = logging.getLogger(__name__)


# ----------------------------
# Helpers
# ----------------------------
_TURN_RE = re.compile(r"(\d+)")
_TURN_TOKEN_RE = re.compile(r"^TURN_?\d+$", re.IGNORECASE)

# 注意：以下正则表达式保留用于其他功能（如偏好提取），但不再用于规则提取 facts
_PREFERENCE_RE = re.compile(
    r"\bI\s+(?:really\s+)?(like|love|enjoy|prefer)\s+([^\.!\n\?]+)",
    re.IGNORECASE,
)
_DISLIKE_RE = re.compile(
    r"\bI\s+(?:really\s+)?(?:do not|don't|dont)\s+like\s+([^\.!\n\?]+)",
    re.IGNORECASE,
)
_MOVE_RE = re.compile(
    r"\bI\s+(?:just\s+)?(?:moved\s+to|now\s+live\s+in|live\s+in|am\s+in)\s+([^\.!\n\?]+)",
    re.IGNORECASE,
)
_USED_TO_LIVE_RE = re.compile(
    r"\bI\s+used\s+to\s+live\s+in\s+([^\.!\n\?]+)",
    re.IGNORECASE,
)

# ============================================================
# 泛化设计原则：
# 1. 实体提取完全依赖 LLM + few-shot 示例
# 2. 规则只做最基本的格式验证（如实体名称不能是句子片段）
# 3. 不针对特定领域（如博物馆）做硬编码规则
# ============================================================

# 以下正则仅用于调试/日志，不用于实际提取
_VISIT_CUE_RE = re.compile(
    r"\b(visited|went\s+to|been\s+to|was\s+at|attended)\b",
    re.IGNORECASE,
)

_FROM_BRAND_RE = re.compile(
    r"\bfrom\s+([A-Z][A-Za-z0-9&'\\-]+(?:\s+[A-Z][A-Za-z0-9&'\\-]+){0,3})",
    re.IGNORECASE,
)


def _parse_turn_id(virtual_time: str) -> int:
    try:
        m = _TURN_RE.search(str(virtual_time or ""))
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


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


# ----------------------------
# LLM 验证 + 规则兜底（增强版巩固流程）
# ----------------------------

# 是否启用 LLM 验证（可通过环境变量控制）
_ENABLE_LLM_VALIDATION = os.getenv("ENABLE_LLM_VALIDATION", "1").strip().lower() in ("1", "true", "yes")
_ENABLE_RULE_VALIDATION = os.getenv("ENABLE_RULE_VALIDATION", "1").strip().lower() in ("1", "true", "yes")


def _build_validation_prompt(relationships: List[Dict], nodes: List[Dict]) -> str:
    """构建 LLM 验证 prompt"""
    nodes_text = "\n".join([
        f"  - {n.get('name', '')} ({n.get('label', 'Unknown')})"
        for n in nodes[:50]  # 限制数量
    ])
    
    rels_text = "\n".join([
        f"  [{i}] ({r.get('source_node_name', '')}) -[{r.get('type', '')}]-> ({r.get('target_node_name', '')})"
        for i, r in enumerate(relationships[:30])  # 限制数量
    ])
    
    return f"""You are a knowledge graph validator. Check if each relationship is semantically correct.

NODES (partial list, may not include all nodes):
{nodes_text}

RELATIONSHIPS TO VALIDATE:
{rels_text}

VALIDATION RULES:
1. VISITED/WENT_TO/TOURED: target must be a real place (Organization/Location), NOT an Event
   - WRONG: User -[VISITED]-> "Space Exploration Exhibition Visit" (contains "Visit", this is an Event)
   - WRONG: User -[VISITED]-> "Dinosaur Fossils exhibition" (contains "exhibition", this is an Event)
   - CORRECT: User -[VISITED]-> "Science Museum" (this is an Organization)
   - CORRECT: User -[VISITED]-> "Museum of Contemporary Art" (this is an Organization)
2. LIVES_IN/RESIDES_IN: target must be a Location
3. WORKS_AT/EMPLOYED_BY: target must be an Organization
4. Event nodes (exhibitions, tours, lectures, visits) should NOT be targets of VISITED relations
5. If target name contains "visit", "tour", "lecture", "reception", "exhibition", it's likely an Event

IMPORTANT:
- DO NOT remove a relationship just because the target is not in the NODES list above
- The NODES list is partial and may not include all valid nodes
- Only remove relationships that violate the semantic rules above
- Museums, galleries, and other real places are VALID targets for VISITED

For each relationship, return:
- "keep": relationship is correct OR target is a real place (museum, gallery, etc.)
- "remove": relationship violates semantic rules (explain why)

Output JSON only:
{{
  "validations": [
    {{"index": 0, "action": "keep"}},
    {{"index": 1, "action": "remove", "reason": "target contains 'exhibition', this is an Event not a place"}}
  ]
}}
"""


def _invoke_llm_validation(llm, relationships: List[Dict], nodes: List[Dict]) -> List[int]:
    """
    调用 LLM 验证关系，返回需要删除的关系索引列表
    """
    if not relationships:
        return []
    
    prompt = _build_validation_prompt(relationships, nodes)
    
    try:
        raw = invoke_json(llm, prompt)
        obj = json.loads(raw) if isinstance(raw, str) else raw
        
        validations = obj.get("validations", [])
        remove_indices = []
        
        for v in validations:
            if v.get("action") == "remove":
                idx = v.get("index")
                if isinstance(idx, int) and 0 <= idx < len(relationships):
                    remove_indices.append(idx)
                    if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
                        reason = v.get("reason", "unknown")
                        rel = relationships[idx]
                        logger.info(
                            f"[llm_validation] remove rel[{idx}]: "
                            f"({rel.get('source_node_name')}) -[{rel.get('type')}]-> ({rel.get('target_node_name')}) "
                            f"reason={reason}"
                        )
        
        return remove_indices
    
    except Exception as e:
        logger.warning(f"[llm_validation] failed: {e}")
        return []


def _rule_based_validation(
    relationships: List[Relationship],
    nodes_map: Dict[str, str],  # {name: label}
) -> List[Relationship]:
    """
    规则兜底验证：基于节点标签和关系类型的通用规则
    
    设计原则：
    1. 基于节点标签（LLM 已提取），不硬编码具体名称
    2. 规则尽量通用，覆盖大类而不是具体关系
    3. 宁可漏过，不可错杀（保守过滤）
    """
    validated = []
    
    for rel in relationships:
        target_name = str(getattr(rel, "target_node_name", "") or "").strip()
        source_name = str(getattr(rel, "source_node_name", "") or "").strip()
        # 优先使用 nodes_map，其次使用 relationship 中的 label
        target_label = nodes_map.get(target_name) or str(getattr(rel, "target_node_label", "") or "").strip() or "Unknown"
        source_label = nodes_map.get(source_name) or str(getattr(rel, "source_node_label", "") or "").strip() or "Unknown"
        rel_type = str(getattr(rel, "type", "") or "").upper()
        
        should_remove = False
        reason = ""
        
        # ========== 规则1：Event 节点约束（极度放宽版）==========
        # 问题：LLM 经常把 Concept 错误标记为 Event
        # 解决：几乎允许所有关系类型指向 Event，只排除极少数明显不合理的
        if target_label == "Event":
            # 极度放宽：只排除极少数明显不合理的关系
            # 原则：宁可漏过，不可错杀
            forbidden_for_event_target = {
                # 只排除物理位置关系（Event 不是物理地点）
                "LIVES_IN", "BORN_IN", "DIED_IN", "LOCATED_IN",
                # 只排除所有权关系（Event 不能被拥有）
                "OWNS", "HAS_PROPERTY",
            }
            if rel_type in forbidden_for_event_target:
                should_remove = True
                reason = f"Event cannot be target of {rel_type}"
        
        # ========== 规则2：Concept 节点约束 ==========
        # Concept 不能作为物理动作关系的 target
        if target_label == "Concept":
            physical_action_rels = {
                "VISITED", "WENT_TO", "LIVES_IN", "WORKS_AT",
                "LOCATED_AT", "BORN_IN", "DIED_IN", "TOURED"
            }
            if rel_type in physical_action_rels:
                should_remove = True
                reason = f"Concept cannot be target of {rel_type}"
        
        # ========== 规则3：target 名称模式检查 ==========
        # 即使标签错误，名称也能暴露问题
        if rel_type in {"VISITED", "WENT_TO", "TOURED"}:
            target_lower = target_name.lower()
            invalid_patterns = [
                r"\bvisit\b", r"\btour\b", r"\btrip\b",
                r"\blecture\s*(series)?\b", r"\breception\b", 
                r"\bexhibition\s+visit\b", r"\bmuseum\s+visit\b",
                r"\bguided\s+tour\b", r"\bbehind.the.scenes\b"
            ]
            for pattern in invalid_patterns:
                if re.search(pattern, target_lower):
                    should_remove = True
                    reason = f"target name matches Event pattern: {pattern}"
                    break
        
        # ========== 规则4：source 类型约束 ==========
        # 某些关系的 source 必须是 Person
        person_source_rels = {
            "VISITED", "LIVES_IN", "WORKS_AT", "BORN_IN",
            "MARRIED_TO", "FRIEND_OF", "KNOWS", "TOURED"
        }
        if rel_type in person_source_rels:
            if source_label not in {"Person", "Unknown"}:
                should_remove = True
                reason = f"{rel_type} requires Person as source, got {source_label}"
        
        # ========== 规则5：自引用检查 ==========
        if source_name and source_name == target_name:
            should_remove = True
            reason = "self-reference"
        
        # 记录过滤原因
        if should_remove:
            if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
                logger.info(
                    f"[rule_validation] remove: ({source_name}) -[{rel_type}]-> ({target_name}) "
                    f"reason={reason}"
                )
            continue
        
        validated.append(rel)
    
    return validated


def validate_consolidated_relationships(
    structured: KnowledgeGraphExtraction,
    llm=None,
) -> KnowledgeGraphExtraction:
    """
    验证巩固结果中的关系（LLM 验证 + 规则兜底）
    
    流程：
    1. LLM 验证（如果启用）
    2. 规则兜底验证
    3. 返回清理后的结果
    """
    if not structured:
        return structured
    
    relationships = list(getattr(structured, "relationships", []) or [])
    nodes = list(getattr(structured, "nodes", []) or [])
    
    if not relationships:
        return structured
    
    # 构建 nodes_map: {name: label}
    nodes_map = {}
    for n in nodes:
        name = str(getattr(n, "name", "") or "").strip()
        label = str(getattr(n, "label", "") or "").strip()
        if name:
            nodes_map[name] = label
    
    # 1. LLM 验证
    if _ENABLE_LLM_VALIDATION and llm is not None:
        # 转换为 dict 格式供 LLM 验证
        rels_dict = [
            {
                "source_node_name": getattr(r, "source_node_name", ""),
                "source_node_label": getattr(r, "source_node_label", ""),
                "target_node_name": getattr(r, "target_node_name", ""),
                "target_node_label": getattr(r, "target_node_label", ""),
                "type": getattr(r, "type", ""),
            }
            for r in relationships
        ]
        nodes_dict = [
            {"name": getattr(n, "name", ""), "label": getattr(n, "label", "")}
            for n in nodes
        ]
        
        remove_indices = _invoke_llm_validation(llm, rels_dict, nodes_dict)
        
        if remove_indices:
            relationships = [
                r for i, r in enumerate(relationships) 
                if i not in remove_indices
            ]
            if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
                logger.info(f"[llm_validation] removed {len(remove_indices)} relationships")
    
    # 2. 规则兜底验证
    if _ENABLE_RULE_VALIDATION:
        before_count = len(relationships)
        relationships = _rule_based_validation(relationships, nodes_map)
        after_count = len(relationships)
        
        if before_count != after_count:
            if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
                logger.info(
                    f"[rule_validation] removed {before_count - after_count} relationships"
                )
    
    # 更新 structured
    structured.relationships = relationships
    return structured


def _sanitize_kg_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Defensive cleanup for LLM output:
    - drop relationships missing required fields (type/source/target)
    - ensure list types for nodes/relationships/facts/insights
    """
    if not isinstance(obj, dict):
        return {}

    nodes = obj.get("nodes") or []
    rels = obj.get("relationships") or []
    facts = obj.get("facts") or []
    claims = obj.get("claims") or []
    insights = obj.get("insights") or []

    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(rels, list):
        rels = []
    if not isinstance(facts, list):
        facts = []
    if not isinstance(claims, list):
        claims = []
    if not isinstance(insights, list):
        insights = []

    cleaned_rels = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        if not (r.get("type") and r.get("source_node_name") and r.get("target_node_name")):
            continue
        if not isinstance(r.get("source_node_label"), str) or not r.get("source_node_label"):
            r["source_node_label"] = "Concept"
        if not isinstance(r.get("target_node_label"), str) or not r.get("target_node_label"):
            r["target_node_label"] = "Concept"
        cleaned_rels.append(r)

    obj["nodes"] = nodes
    obj["relationships"] = cleaned_rels
    cleaned_facts: List[str] = []
    cleaned_claims: List[Dict[str, Any]] = []
    max_claim_chars = 240
    sentence_split_re = re.compile(r"[.!?]\s+")

    def _normalize_claim_dict(claim: Dict[str, Any]) -> Dict[str, Any]:
        c = dict(claim or {})
        text = str(c.get("text") or "").strip()
        c["text"] = text

        # Normalize event_time_text
        if not c.get("event_time_text"):
            for k in ("eventTimeText", "event_time", "event_time_text", "event_timestamp_text", "time_text"):
                v = c.get(k)
                if isinstance(v, str) and v.strip():
                    c["event_time_text"] = v.strip()
                    break
        if isinstance(c.get("event_time_text"), str) and not c.get("event_time_text").strip():
            c["event_time_text"] = None

        # Normalize event_turn_offset
        if c.get("event_turn_offset") is None:
            for k in ("event_turn", "turn_offset", "time_offset", "event_offset"):
                if k in c and c.get(k) is not None:
                    c["event_turn_offset"] = c.get(k)
                    break
        if c.get("event_turn_offset") is not None:
            try:
                c["event_turn_offset"] = int(float(c.get("event_turn_offset")))
            except Exception:
                c["event_turn_offset"] = None

        # Normalize confidence to [0, 1]
        try:
            conf = float(c.get("confidence", 1.0))
        except Exception:
            conf = 1.0
        if conf > 1.0 and conf <= 100.0:
            conf = conf / 100.0
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0:
            conf = 1.0
        c["confidence"] = float(conf)

        # Normalize knowledge_type
        kt = str(c.get("knowledge_type") or c.get("knowledgeType") or "").strip().lower()
        if kt in ("observed", "observation", "fact"):
            kt = "observed_fact"
        elif kt in ("inferred", "inference", "infer"):
            kt = "inferred"
        elif kt in ("reported", "report"):
            kt = "reported"
        if not kt:
            kt = "observed_fact"
        c["knowledge_type"] = kt

        # Normalize source_of_belief
        sob = str(c.get("source_of_belief") or c.get("sourceOfBelief") or c.get("source") or "").strip().lower()
        if not sob:
            sob = "user_statement"
        elif "assistant" in sob:
            sob = "assistant_statement"
        elif "user" in sob:
            sob = "user_statement"
        elif sob in ("tool", "system"):
            sob = "tool"
        elif sob in ("derived", "reflection", "inference", "reasoning"):
            sob = "derived"
        c["source_of_belief"] = sob

        return c

    def _push_claim(text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        s = (text or "").strip()
        if not s or _TURN_TOKEN_RE.match(s):
            return
        texts = [s]
        if len(s) > max_claim_chars:
            parts = [p.strip() for p in sentence_split_re.split(s) if p.strip()]
            texts = [p for p in parts if 8 <= len(p) <= max_claim_chars]
        for t in texts:
            cleaned_facts.append(t)
            claim = {
                "text": t,
                "confidence": 1.0,
                "knowledge_type": "observed_fact",
                "source_of_belief": "user_statement",
                "event_time_text": None,
                "event_turn_offset": None,
            }
            if meta is not None and isinstance(meta, dict):
                claim.update({k: v for k, v in meta.items() if v is not None})
            cleaned_claims.append(_normalize_claim_dict(claim))

    for c in claims:
        if isinstance(c, dict):
            _push_claim(c.get("text", ""), c)
        elif hasattr(c, "model_dump"):
            c_dict = c.model_dump()
            _push_claim(c_dict.get("text", ""), c_dict)

    for f in facts:
        if isinstance(f, dict):
            _push_claim(f.get("text", ""), f)
            continue
        if isinstance(f, str):
            _push_claim(f, None)

    # de-dup facts
    seen_f = set()
    uniq_f = []
    for f in cleaned_facts:
        if f in seen_f:
            continue
        seen_f.add(f)
        uniq_f.append(f)

    obj["facts"] = uniq_f
    # De-dup claims by text, keep first occurrence.
    seen_c = set()
    uniq_c = []
    for c in cleaned_claims:
        text = str(c.get("text") or "").strip()
        if not text or text in seen_c:
            continue
        seen_c.add(text)
        uniq_c.append(c)
    obj["claims"] = uniq_c
    obj["insights"] = insights
    return obj


def _normalize_claims_in_structured(structured: KnowledgeGraphExtraction) -> None:
    claims = list(getattr(structured, "claims", []) or [])
    normalized: List[Claim] = []
    seen = set()
    for c in claims:
        c_dict = c.model_dump() if hasattr(c, "model_dump") else (c if isinstance(c, dict) else {})
        if not isinstance(c_dict, dict):
            continue
        c_norm = _sanitize_kg_dict({"claims": [c_dict]}).get("claims", [])
        if not c_norm:
            continue
        c_norm = c_norm[0]
        text = str(c_norm.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(Claim(**c_norm))
    structured.claims = normalized


# 注意：_extract_rule_facts_from_turns 已删除，完全依赖 LLM 提取 facts


def _add_node(nodes: List[Node], *, label: str, name: str, properties: Optional[Dict[str, Any]] = None) -> None:
    key = (label.strip(), name.strip())
    for n in nodes:
        if (getattr(n, "label", "") or "").strip() == key[0] and (getattr(n, "name", "") or "").strip() == key[1]:
            return
    nodes.append(Node(label=key[0], name=key[1], properties=properties or {}))


def _add_rel(
    rels: List[Relationship],
    *,
    s_name: str,
    s_label: str,
    t_name: str,
    t_label: str,
    rel_type: str,
    properties: Optional[Dict[str, Any]] = None,
) -> None:
    key = (s_label.strip(), s_name.strip(), rel_type.strip().upper(), t_label.strip(), t_name.strip())
    for r in rels:
        if (
            (getattr(r, "source_node_label", "") or "").strip() == key[0]
            and (getattr(r, "source_node_name", "") or "").strip() == key[1]
            and (getattr(r, "type", "") or "").strip().upper() == key[2]
            and (getattr(r, "target_node_label", "") or "").strip() == key[3]
            and (getattr(r, "target_node_name", "") or "").strip() == key[4]
        ):
            return
    rels.append(
        Relationship(
            source_node_name=key[1],
            source_node_label=key[0],
            target_node_name=key[4],
            target_node_label=key[3],
            type=key[2],
            properties=properties or {},
        )
    )


def _clean_entity_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return ""
    s = re.sub(r"(?:'s)\b", "", s)
    s = re.sub(r"\\s+", " ", s).strip()
    s = re.sub(r"^the\s+", "", s, flags=re.IGNORECASE)
    return s.strip(" ,.;:!?\"'()")


def _is_valid_entity_name(name: str) -> bool:
    """
    检查实体名称是否有效（不是句子片段）
    
    这是一个轻量级的过滤器，只过滤最明显的错误。
    主要依赖 LLM prompt 中的指导来确保输出质量。
    """
    if not name:
        return False
    
    name = name.strip()
    words = name.split()
    
    # 太长的名称可能是句子片段
    if len(words) > 8:
        return False
    
    name_lower = name.lower()
    
    # 只过滤最明显的句子开头
    obvious_sentence_starters = [
        "i ", "i'm ", "i've ", "i'd ",
        "we ", "you ", "he ", "she ", "they ",
    ]
    for starter in obvious_sentence_starters:
        if name_lower.startswith(starter):
            return False
    
    return True


def _filter_invalid_entities(structured) -> None:
    """
    过滤掉无效的实体名称（句子片段等）
    """
    if not structured:
        return
    
    # 过滤节点
    valid_nodes = []
    invalid_names = set()
    for node in (getattr(structured, "nodes", []) or []):
        name = str(getattr(node, "name", "") or "").strip()
        if _is_valid_entity_name(name):
            valid_nodes.append(node)
        else:
            invalid_names.add(name)
            if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
                logger.info(f"[_filter_invalid_entities] 过滤无效实体: {name}")
    
    try:
        structured.nodes = valid_nodes
    except Exception:
        pass
    
    # 过滤关系（如果 source 或 target 被过滤掉了）
    valid_rels = []
    for rel in (getattr(structured, "relationships", []) or []):
        s_name = str(getattr(rel, "source_node_name", "") or "").strip()
        t_name = str(getattr(rel, "target_node_name", "") or "").strip()
        if s_name not in invalid_names and t_name not in invalid_names:
            if _is_valid_entity_name(s_name) and _is_valid_entity_name(t_name):
                valid_rels.append(rel)
    
    try:
        structured.relationships = valid_rels
    except Exception:
        pass


_EXHIBITION_RE = re.compile(
    r"\b(exhibition|exhibit|show|installation|lecture|tour|series|reception)\b",
    re.IGNORECASE,
)

# 泛化的无效 VISITED 目标模式（Event 类型的活动/事件）
_INVALID_VISIT_TARGET_RE = re.compile(
    r"\b("
    r"visit|trip|attendance|outing|"  # 访问类动作名词
    r"tour|guided\s+tour|behind.the.scenes|"  # 导览类
    r"lecture|series|seminar|workshop|class|"  # 教育类
    r"reception|event|meeting|session|gathering|"  # 活动类
    r"exhibition\s+visit|museum\s+visit|gallery\s+visit"  # 复合访问
    r")\b",
    re.IGNORECASE,
)

# 有效的地点类型关键词（泛化）
_VALID_LOCATION_KEYWORDS = {
    "museum", "restaurant", "hotel", "cafe", "bar", "store", "shop",
    "park", "beach", "mountain", "lake", "airport", "station",
    "hospital", "school", "university", "college", "library",
    "theater", "theatre", "cinema", "gallery", "center", "centre",
    "mall", "market", "plaza", "square", "building", "tower",
    "church", "temple", "mosque", "cathedral", "palace", "castle",
}


def _is_valid_visit_target(target: str) -> bool:
    """
    检查 VISITED 关系的目标是否是有效的地点（泛化）
    
    有效：实际地点（Museum, Restaurant, Hotel 等）
    无效：Event 类型（Exhibition Visit, Guided Tour, Lecture Series 等）
    """
    if not target:
        return False
    
    target_lower = target.lower()
    
    # 1. 如果包含有效地点关键词，优先认为有效
    has_location_keyword = any(kw in target_lower for kw in _VALID_LOCATION_KEYWORDS)
    
    # 2. 检查是否是无效的 Event 类型目标
    is_event_target = bool(_INVALID_VISIT_TARGET_RE.search(target))
    
    # 3. 如果同时包含地点关键词和 Event 模式，检查格式
    if has_location_keyword and is_event_target:
        # "Science Museum Visit" -> 无效（地点 + visit）
        # "Museum of History" -> 有效（地点，没有 visit）
        for kw in _VALID_LOCATION_KEYWORDS:
            if kw in target_lower:
                # 检查是否是 "XXX Location Visit/Tour" 格式
                if re.search(rf"{kw}\s+(visit|tour|trip)\b", target_lower, re.IGNORECASE):
                    return False
        return True
    
    # 4. 有地点关键词且没有 Event 模式 -> 有效
    if has_location_keyword:
        return True
    
    # 5. 没有地点关键词但有 Event 模式 -> 无效
    if is_event_target:
        return False
    
    # 6. 默认：没有明确信号，保守地认为有效
    return True


# ============================================================
# 以下博物馆特定函数已废弃，保留空实现以兼容旧代码
# ============================================================

def _normalize_museum_target(name: str) -> str:
    """已废弃：直接返回清理后的名称"""
    return _clean_entity_name(name)


def _extract_museums_from_text(content: str) -> List[str]:
    """已废弃：返回空列表，完全依赖 LLM 提取"""
    return []


def _filter_unverified_visit_claims(
    structured: KnowledgeGraphExtraction, session_turns: List[Dict[str, Any]]
) -> None:
    """
    简化版：不再做博物馆特定的过滤，完全信任 LLM 的提取结果。
    
    泛化原则：LLM 应该能够正确判断哪些是实际访问，哪些是计划/推荐。
    如果 LLM 提取错误，应该通过改进 prompt 来修复，而不是添加规则。
    """
    # 不做任何过滤，保留 LLM 提取的所有关系
    pass


# 注意：_augment_structured_with_rules 已删除，完全依赖 LLM 提取


_VISIT_REL_ALIASES = {
    "VISITED",
    "VISITS",
    "VISIT",
    "WENT_TO",
    "ATTENDED",
    "TOURED",
    "TOUR",
    "GUIDED_TOUR",
    "WAS_AT",
}


def _normalize_visit_relationships(structured: KnowledgeGraphExtraction) -> None:
    """
    规范化 VISITED 关系，过滤无效目标（泛化）
    
    1. 统一关系类型为 VISITED
    2. 规范化目标名称
    3. 过滤 Event 类型的无效目标
    4. 设置正确的目标标签
    """
    rels = list(getattr(structured, "relationships", []) or [])
    cleaned: List[Relationship] = []
    for r in rels:
        rel_type = str(getattr(r, "type", "") or "").upper()
        if rel_type in _VISIT_REL_ALIASES:
            r.type = "VISITED"
        if str(getattr(r, "type", "") or "").upper() != "VISITED":
            cleaned.append(r)
            continue
        
        target_name = str(getattr(r, "target_node_name", "") or "").strip()
        normalized = _normalize_museum_target(target_name)
        if normalized:
            r.target_node_name = normalized
        
        final_target = normalized or target_name
        
        # 泛化的有效性检查
        if not _is_valid_visit_target(final_target):
            continue
        
        # 设置正确的标签
        target_lc = final_target.lower()
        if any(kw in target_lc for kw in _VALID_LOCATION_KEYWORDS):
            r.target_node_label = "Organization"
        
        cleaned.append(r)
    structured.relationships = cleaned


@dataclass
class OriginalConsolidationResult:
    structured: KnowledgeGraphExtraction
    raw_chunks: int


def _ensure_claims_from_facts(structured: KnowledgeGraphExtraction) -> None:
    if not structured:
        return
    claims = list(getattr(structured, "claims", []) or [])
    claim_texts = set()
    for c in claims:
        c_dict = c.model_dump() if hasattr(c, "model_dump") else (c if isinstance(c, dict) else {})
        text = str(c_dict.get("text") or "").strip()
        if text:
            claim_texts.add(text)
    for f in getattr(structured, "facts", []) or []:
        if not isinstance(f, str):
            continue
        text = f.strip()
        if not text or text in claim_texts or _TURN_TOKEN_RE.match(text):
            continue
        claims.append(
            {
                "text": text,
                "confidence": 1.0,
                "knowledge_type": "observed_fact",
                "source_of_belief": "reflection",
                "event_time_text": None,
                "event_turn_offset": None,
            }
        )
        claim_texts.add(text)
    structured.claims = claims


def consolidate_original_session(
    *,
    session_turns: List[Dict[str, Any]],
    virtual_time: str,
    session_time_raw: str,
    session_time_iso: str,
    include_assistant: bool,
    llm,
    max_chars_per_chunk: int = 5000,
) -> Optional[OriginalConsolidationResult]:
    """
    巩固会话：从对话中提取结构化知识图谱
    
    使用两阶段提取器（借鉴 GraphRAG + Instructor）
    
    Args:
        session_turns: 会话轮次列表
        virtual_time: 虚拟时间（如 TURN_1）
        session_time_raw: 原始时间字符串
        session_time_iso: ISO 格式时间戳
        include_assistant: 是否包含助手回复
        llm: LLM 实例
        max_chars_per_chunk: 每个 chunk 最大字符数
    
    Returns:
        OriginalConsolidationResult 或 None
    """
    if not session_turns:
        return None
    
    # 检查提取器是否可用
    if not _EXTRACTOR_AVAILABLE or consolidate_session_v2 is None:
        logger.error("[original_consolidation] extractor not available (instructor not installed?)")
        return None
    
    if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
        logger.info(
            f"[original_consolidation] using two-phase extractor "
            f"virtual_time={virtual_time}"
        )
    
    try:
        result = consolidate_session_v2(
            session_turns=session_turns,
            virtual_time=virtual_time,
            session_time_iso=session_time_iso or session_time_raw or "",
            include_assistant=include_assistant,
            max_chars=max_chars_per_chunk,
        )
        
        if not result or not result.get("kg_extraction"):
            logger.warning("[original_consolidation] extractor returned empty result")
            return None
        
        merged = result["kg_extraction"]
        
        # 注意：规则提取已删除，完全依赖 LLM 提取
        _normalize_visit_relationships(merged)
        
        # 规则兜底验证
        if _ENABLE_RULE_VALIDATION:
            nodes_map = {
                str(getattr(n, "name", "") or "").strip(): str(getattr(n, "label", "") or "").strip()
                for n in (getattr(merged, "nodes", []) or [])
            }
            merged.relationships = _rule_based_validation(
                list(getattr(merged, "relationships", []) or []),
                nodes_map,
            )
        
        _filter_unverified_visit_claims(merged, session_turns)
        _ensure_claims_from_facts(merged)
        _normalize_claims_in_structured(merged)
        
        if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
            logger.info(
                "[original_consolidation] merged "
                f"nodes={len(getattr(merged, 'nodes', []) or [])} "
                f"rels={len(getattr(merged, 'relationships', []) or [])} "
                f"facts={len(getattr(merged, 'facts', []) or [])} "
                f"claims={len(getattr(merged, 'claims', []) or [])}"
            )
        
        return OriginalConsolidationResult(structured=merged, raw_chunks=1)
    
    except Exception as e:
        logger.error(f"[original_consolidation] extractor failed: {e}")
        return None
