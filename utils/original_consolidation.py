# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils.llm import invoke_json
from utils.relation_extractor import extract_relations_from_text
from memory.structured_memory import KnowledgeGraphExtraction, Node, Relationship

logger = logging.getLogger(__name__)


# ----------------------------
# Helpers
# ----------------------------
_TURN_RE = re.compile(r"(\d+)")
_PERCENT_RE = re.compile(r"\b(\d{1,3})\s*%\b")
_WORD_DURATION_RE = re.compile(
    r"\b(?:(over|more than|less than|about|around)\s+)?"
    r"(a|an|one|\d+)\s+"
    r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

_KNOWN_BRANDS = [
    "hellofresh",
    "ubereats",
    "grubhub",
    "doordash",
    "postmates",
]

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
_MUSEUM_PATTERNS = [
    re.compile(
        r"\bvisited\s+(?:the\s+)?([A-Z][A-Za-z0-9&'\\-\\s]+?Museum(?:\s+of\s+[A-Z][A-Za-z0-9&'\\-\\s]+)?(?:'s)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwent\s+to\s+(?:the\s+)?([A-Z][A-Za-z0-9&'\\-\\s]+?Museum(?:\s+of\s+[A-Z][A-Za-z0-9&'\\-\\s]+)?(?:'s)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\battended\b.*?\bat\s+(?:the\s+)?([A-Z][A-Za-z0-9&'\\-\\s]+?Museum(?:\s+of\s+[A-Z][A-Za-z0-9&'\\-\\s]+)?(?:'s)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btook\s+(?:my|our|his|her)?\s*[^\.!\n\?]{0,40}?\s+to\s+(?:the\s+)?([A-Z][A-Za-z0-9&'\\-\\s]+?Museum(?:\s+of\s+[A-Z][A-Za-z0-9&'\\-\\s]+)?(?:'s)?)\b",
        re.IGNORECASE,
    ),
]
_MUSEUM_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&'\\-\\s]+?Museum(?:\s+of\s+[A-Z][A-Za-z0-9&'\\-\\s]+)?(?:'s)?)\b",
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
    insights = obj.get("insights") or []

    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(rels, list):
        rels = []
    if not isinstance(facts, list):
        facts = []
    if not isinstance(insights, list):
        insights = []

    cleaned_rels = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        if not (r.get("type") and r.get("source_node_name") and r.get("target_node_name")):
            continue
        cleaned_rels.append(r)

    obj["nodes"] = nodes
    obj["relationships"] = cleaned_rels
    obj["facts"] = facts
    obj["insights"] = insights
    return obj


def build_original_consolidation_prompt(
    *,
    session_text: str,
    virtual_time: str,
    session_time_raw: str,
    session_time_iso: str,
    chunk_index: int,
    chunk_total: int,
) -> str:
    recorded_turn_id = _parse_turn_id(virtual_time)
    session_time_hint = session_time_iso or session_time_raw or "unknown"

    return f"""
You are a knowledge graph extractor for a cognitive agent.
You MUST output a STRICT JSON object with keys:
- facts: array of HARD FACT strings (numbers, dates, durations, prices, proper names; keep exact wording)
- insights: array of LOGICAL INSIGHT strings (social relations, patterns, causal links)
- nodes: array of nodes: {{name: str, label: str, properties: object}}
- relationships: array of relationships:
  {{source_node_name: str, source_node_label: str, target_node_name: str, target_node_label: str, type: str, properties: object}}

CRITICAL TIME RULES:
1) The reference "recorded time" is the current virtual turn: {virtual_time}
   You MUST set on EVERY Event node:
     properties.recorded_turn_id = {recorded_turn_id}
2) If the text explicitly mentions time ("yesterday", "last week", specific dates),
   you MUST set on that Event node:
     - properties.event_time_text (string)
     - properties.event_turn_offset (integer relative to recorded_turn_id)
3) If the text does NOT mention explicit time, you may set:
     properties.event_timestamp = "{session_time_hint}"
   only if it is a dataset-provided session time. Do NOT guess dates.

GENERAL RULES:
- Output ONLY valid JSON. No markdown.
- NO hallucinations.
- Node labels must be one of: Person, Organization, Location, Object, Event, Date, Value, Concept.
- Relationship type MUST be UPPERCASE.
- Keep output concise (prefer <= 12 relationships, <= 18 nodes).

CHUNK INFO:
This is chunk {chunk_index} / {chunk_total} of the same session.

SESSION TEXT:
{session_text}
""".strip()


def _build_chunk_text(
    turns: List[Dict[str, Any]],
    *,
    include_assistant: bool,
    max_chars: int,
) -> str:
    lines: List[str] = []
    for t in turns:
        role = (t.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        if (not include_assistant) and role == "assistant":
            continue
        content = (t.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")

    text = "\n".join(lines).strip()
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def _extract_rule_facts_from_turns(session_turns: List[Dict[str, Any]]) -> List[str]:
    facts: List[str] = []
    for t in session_turns or []:
        role = (t.get("role") or "").strip().lower()
        if role != "user":
            continue
        content = (t.get("content") or "").strip()
        if not content:
            continue

        low = content.lower()

        # Duration-like facts
        for m in _WORD_DURATION_RE.finditer(content):
            frag = m.group(0).strip()
            if frag:
                facts.append(frag)

        # Percent discount facts
        pct_matches = _PERCENT_RE.findall(content)
        if pct_matches:
            for pct in pct_matches:
                pct_text = f"{pct}%"
                brand = None
                for b in _KNOWN_BRANDS:
                    if b in low:
                        brand = b
                        break
                if brand:
                    brand_name = brand.title() if brand != "ubereats" else "UberEats"
                    if "first order" in low:
                        facts.append(f"{brand_name} first order discount {pct_text}")
                    else:
                        facts.append(f"{brand_name} discount {pct_text}")
                else:
                    facts.append(f"{pct_text} discount")

        # Direct keep: strong sentence with key terms
        if "asylum" in low and ("year" in low or "month" in low or "week" in low or "day" in low):
            facts.append(content)

        # Museum visit facts (user only)
        for museum in _extract_museums_from_text(content):
            facts.append(f"User visited {museum}")

    # de-dup
    uniq: List[str] = []
    seen = set()
    for f in facts:
        s = f.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


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
    s = re.sub(r"['’]s\\b", "", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s.strip(" ,.;:!?\"'()")


def _extract_museums_from_text(content: str) -> List[str]:
    if not content:
        return []
    hits: List[str] = []
    for pat in _MUSEUM_PATTERNS:
        for m in pat.finditer(content):
            name = _clean_entity_name(m.group(1))
            if name:
                hits.append(name)

    # Fallback: if user says "I visited ... museum" with odd punctuation
    if not hits:
        low = content.lower()
        if "visited" in low or "went to" in low or "attended" in low or "took" in low:
            for m in _MUSEUM_NAME_RE.finditer(content):
                name = _clean_entity_name(m.group(1))
                if name:
                    hits.append(name)

    # de-dup
    seen = set()
    uniq = []
    for h in hits:
        if h in seen:
            continue
        seen.add(h)
        uniq.append(h)
    return uniq


def _augment_structured_with_rules(
    structured: KnowledgeGraphExtraction,
    session_turns: List[Dict[str, Any]],
    *,
    session_time_iso: str,
) -> None:
    if not structured:
        return
    nodes = list(getattr(structured, "nodes", []) or [])
    rels = list(getattr(structured, "relationships", []) or [])
    extra_facts: List[str] = []

    for t in session_turns or []:
        role = (t.get("role") or "").strip().lower()
        if role != "user":
            continue
        content = (t.get("content") or "").strip()
        if not content:
            continue
        low = content.lower()

        # Duration linked to asylum waiting
        if "asylum" in low:
            for m in _WORD_DURATION_RE.finditer(content):
                dur = m.group(0).strip()
                if not dur:
                    continue
                _add_node(nodes, label="Concept", name="Asylum application")
                _add_node(nodes, label="Value", name=dur)
                props = {
                    "confidence": 0.9,
                    "source_of_belief": "observation",
                }
                if session_time_iso:
                    props["event_timestamp"] = session_time_iso
                _add_rel(
                    rels,
                    s_name="Asylum application",
                    s_label="Concept",
                    t_name=dur,
                    t_label="Value",
                    rel_type="WAIT_TIME",
                    properties=props,
                )

        # Discount percent by brand
        pct_matches = _PERCENT_RE.findall(content)
        if pct_matches:
            for pct in pct_matches:
                pct_text = f"{pct}%"
                brand = None
                for b in _KNOWN_BRANDS:
                    if b in low:
                        brand = b
                        break
                if brand:
                    brand_name = brand.title() if brand != "ubereats" else "UberEats"
                    rel_type = "FIRST_ORDER_DISCOUNT_PERCENT" if "first order" in low else "DISCOUNT_PERCENT"
                    _add_node(nodes, label="Organization", name=brand_name)
                    _add_node(nodes, label="Value", name=pct_text)
                    props = {
                        "confidence": 0.9,
                        "source_of_belief": "observation",
                    }
                    if session_time_iso:
                        props["event_timestamp"] = session_time_iso
                    _add_rel(
                        rels,
                        s_name=brand_name,
                        s_label="Organization",
                        t_name=pct_text,
                        t_label="Value",
                        rel_type=rel_type,
                        properties=props,
                    )
                    extra_facts.append(f"{brand_name} {rel_type.replace('_', ' ').lower()} {pct_text}")

        # Preferences
        m = _PREFERENCE_RE.search(content)
        if m:
            pref_obj = m.group(2).strip()
            pref_obj = pref_obj[:120].strip()
            if pref_obj:
                _add_node(nodes, label="Person", name="User")
                _add_node(nodes, label="Concept", name=pref_obj)
                _add_rel(
                    rels,
                    s_name="User",
                    s_label="Person",
                    t_name=pref_obj,
                    t_label="Concept",
                    rel_type="PREFERS",
                    properties={"confidence": 0.85, "source_of_belief": "observation"},
                )
        m = _DISLIKE_RE.search(content)
        if m:
            pref_obj = m.group(1).strip()
            pref_obj = pref_obj[:120].strip()
            if pref_obj:
                _add_node(nodes, label="Person", name="User")
                _add_node(nodes, label="Concept", name=pref_obj)
                _add_rel(
                    rels,
                    s_name="User",
                    s_label="Person",
                    t_name=pref_obj,
                    t_label="Concept",
                    rel_type="DISLIKES",
                    properties={"confidence": 0.85, "source_of_belief": "observation"},
                )

        # Knowledge updates (location)
        m = _USED_TO_LIVE_RE.search(content)
        if m:
            loc = m.group(1).strip()[:120].strip()
            if loc:
                _add_node(nodes, label="Person", name="User")
                _add_node(nodes, label="Location", name=loc)
                _add_rel(
                    rels,
                    s_name="User",
                    s_label="Person",
                    t_name=loc,
                    t_label="Location",
                    rel_type="PREVIOUSLY_LIVED_IN",
                    properties={"confidence": 0.8, "source_of_belief": "observation"},
                )
        m = _MOVE_RE.search(content)
        if m:
            loc = m.group(1).strip()[:120].strip()
            if loc:
                _add_node(nodes, label="Person", name="User")
                _add_node(nodes, label="Location", name=loc)
                _add_rel(
                    rels,
                    s_name="User",
                    s_label="Person",
                    t_name=loc,
                    t_label="Location",
                    rel_type="LIVES_IN",
                    properties={"confidence": 0.9, "source_of_belief": "observation", "slot_key": "User:LIVES_IN"},
                )

        # Museum visits (order-sensitive by turn_id)
        museum_hits = _extract_museums_from_text(content)
        for museum in museum_hits:
            _add_node(nodes, label="Person", name="User")
            _add_node(nodes, label="Organization", name=museum)
            props = {
                "confidence": 0.9,
                "source_of_belief": "observation",
            }
            if session_time_iso:
                props["event_timestamp"] = session_time_iso
            _add_rel(
                rels,
                s_name="User",
                s_label="Person",
                t_name=museum,
                t_label="Organization",
                rel_type="VISITED",
                properties=props,
            )
            extra_facts.append(f"User visited {museum}")

    structured.nodes = nodes
    structured.relationships = rels
    if extra_facts:
        merged = list(dict.fromkeys((structured.facts or []) + extra_facts))
        structured.facts = merged


def chunk_session_turns(
    session_turns: List[Dict[str, Any]],
    *,
    max_turns_per_chunk: int = 8,
    overlap_turns: int = 1,
) -> List[List[Dict[str, Any]]]:
    if not session_turns:
        return []
    mt = max(1, int(max_turns_per_chunk))
    ov = max(0, int(overlap_turns))
    step = max(1, mt - ov)
    chunks: List[List[Dict[str, Any]]] = []
    s = 0
    while s < len(session_turns):
        e = min(len(session_turns), s + mt)
        chunks.append(session_turns[s:e])
        if e >= len(session_turns):
            break
        s += step
    return chunks


@dataclass
class OriginalConsolidationResult:
    structured: KnowledgeGraphExtraction
    raw_chunks: int


def _merge_structured(results: List[KnowledgeGraphExtraction]) -> KnowledgeGraphExtraction:
    nodes_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    rel_map: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    facts_set: Dict[str, None] = {}
    insights_set: Dict[str, None] = {}

    for sr in results:
        # Nodes
        for n in getattr(sr, "nodes", []) or []:
            label = str(getattr(n, "label", "") or "").strip()
            name = str(getattr(n, "name", "") or "").strip()
            if not label or not name:
                continue
            key = (label, name)
            props = dict(getattr(n, "properties", None) or {})
            if key not in nodes_map:
                nodes_map[key] = {"name": name, "label": label, "properties": props}
            else:
                # Fill missing properties only
                existing = nodes_map[key]["properties"]
                for k, v in props.items():
                    if k not in existing or existing.get(k) in (None, "", "unknown"):
                        existing[k] = v

        # Relationships
        for r in getattr(sr, "relationships", []) or []:
            s_name = str(getattr(r, "source_node_name", "") or "").strip()
            s_label = str(getattr(r, "source_node_label", "") or "").strip()
            t_name = str(getattr(r, "target_node_name", "") or "").strip()
            t_label = str(getattr(r, "target_node_label", "") or "").strip()
            rel_type = str(getattr(r, "type", "") or "").strip().upper()
            if not (s_name and t_name and rel_type):
                continue
            key = (s_label, s_name, rel_type, t_label, t_name)
            props = dict(getattr(r, "properties", None) or {})
            if key not in rel_map:
                rel_map[key] = {
                    "source_node_name": s_name,
                    "source_node_label": s_label or "Concept",
                    "target_node_name": t_name,
                    "target_node_label": t_label or "Concept",
                    "type": rel_type,
                    "properties": props,
                }
            else:
                existing = rel_map[key]["properties"]
                # Keep higher confidence when present
                try:
                    new_conf = float(props.get("confidence", 0.0))
                    old_conf = float(existing.get("confidence", 0.0))
                    if new_conf > old_conf:
                        existing["confidence"] = new_conf
                except Exception:
                    pass
                for k, v in props.items():
                    if k not in existing or existing.get(k) in (None, "", "unknown"):
                        existing[k] = v

        for f in getattr(sr, "facts", []) or []:
            if isinstance(f, str) and f.strip():
                facts_set[f.strip()] = None
        for ins in getattr(sr, "insights", []) or []:
            if isinstance(ins, str) and ins.strip():
                insights_set[ins.strip()] = None

    merged = {
        "nodes": list(nodes_map.values()),
        "relationships": list(rel_map.values()),
        "facts": list(facts_set.keys()),
        "insights": list(insights_set.keys()),
    }
    return KnowledgeGraphExtraction(**merged)


def consolidate_original_session(
    *,
    session_turns: List[Dict[str, Any]],
    virtual_time: str,
    session_time_raw: str,
    session_time_iso: str,
    include_assistant: bool,
    llm,
    max_turns_per_chunk: int = 8,
    overlap_turns: int = 1,
    max_chars_per_chunk: int = 6000,
    min_confidence: float = 0.0,
) -> Optional[OriginalConsolidationResult]:
    if not session_turns:
        return None

    chunks = chunk_session_turns(
        session_turns,
        max_turns_per_chunk=max_turns_per_chunk,
        overlap_turns=overlap_turns,
    )
    if not chunks:
        return None

    structured_chunks: List[KnowledgeGraphExtraction] = []
    if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
        logger.info(
            f"[original_consolidation] chunks={len(chunks)} virtual_time={virtual_time} "
            f"session_time_iso={session_time_iso or session_time_raw}"
        )

    for idx, turns in enumerate(chunks, 1):
        text = _build_chunk_text(turns, include_assistant=include_assistant, max_chars=max_chars_per_chunk)
        if not text:
            continue
        if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
            snippet = text.replace("\n", " ")[:200]
            logger.info(
                f"[original_consolidation] chunk={idx}/{len(chunks)} text_len={len(text)} "
                f"snippet={snippet}"
            )
        prompt = build_original_consolidation_prompt(
            session_text=text,
            virtual_time=virtual_time,
            session_time_raw=session_time_raw,
            session_time_iso=session_time_iso,
            chunk_index=idx,
            chunk_total=len(chunks),
        )
        kg_raw = invoke_json(llm, prompt)
        if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
            logger.info(
                f"[original_consolidation] chunk={idx}/{len(chunks)} raw_json={str(kg_raw)[:1200]}"
            )
        obj = _parse_json_dict(kg_raw)
        obj = _sanitize_kg_dict(obj)
        structured = KnowledgeGraphExtraction(**obj)

        # Light post-process: if no explicit event_timestamp on relationships, attach session_time_iso
        if session_time_iso:
            for rel in getattr(structured, "relationships", []) or []:
                props = rel.properties if isinstance(getattr(rel, "properties", None), dict) else {}
                props = dict(props or {})
                if not props.get("event_timestamp"):
                    props["event_timestamp"] = session_time_iso
                if "confidence" in props:
                    try:
                        if float(props.get("confidence", 0.0)) < float(min_confidence):
                            props["confidence"] = float(min_confidence)
                    except Exception:
                        pass
                rel.properties = props

        structured_chunks.append(structured)

        if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
            logger.info(
                f"[original_consolidation] chunk={idx}/{len(chunks)} "
                f"nodes={len(getattr(structured, 'nodes', []) or [])} "
                f"rels={len(getattr(structured, 'relationships', []) or [])} "
                f"facts={len(getattr(structured, 'facts', []) or [])}"
            )

    if not structured_chunks:
        return None

    merged = _merge_structured(structured_chunks)

    # Additional generic relation extraction (single pass per session)
    try:
        session_text = _build_chunk_text(
            session_turns,
            include_assistant=include_assistant,
            max_chars=max_chars_per_chunk,
        )
        extra_structured = extract_relations_from_text(
            llm=llm,
            text=session_text,
            virtual_time=virtual_time,
            session_time_iso=session_time_iso or "",
        )
        if extra_structured:
            merged = _merge_structured([merged, extra_structured])
    except Exception as e:
        if os.getenv("DEBUG_REL_EXTRACTOR", "0") == "1":
            logger.info(f"[original_consolidation] relation_extractor failed: {e}")
    # Rule-based augmentation (user text only)
    rule_facts = _extract_rule_facts_from_turns(session_turns)
    if rule_facts:
        merged_facts = list(dict.fromkeys((merged.facts or []) + rule_facts))
        merged.facts = merged_facts
    _augment_structured_with_rules(merged, session_turns, session_time_iso=session_time_iso or "")
    if os.getenv("DEBUG_ORIGINAL_CONSOLIDATION", "0") == "1":
        logger.info(
            "[original_consolidation] merged "
            f"nodes={len(getattr(merged, 'nodes', []) or [])} "
            f"rels={len(getattr(merged, 'relationships', []) or [])} "
            f"facts={len(getattr(merged, 'facts', []) or [])}"
        )
    return OriginalConsolidationResult(structured=merged, raw_chunks=len(structured_chunks))
