# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
from typing import Dict, Any, List, Optional

from utils.llm import invoke_json
from memory.structured_memory import KnowledgeGraphExtraction

logger = logging.getLogger(__name__)

_ALLOWED_LABELS = {
    "Person",
    "Organization",
    "Location",
    "Object",
    "Event",
    "Date",
    "Value",
    "Concept",
}


def _normalize_label(label: str) -> str:
    lab = (label or "").strip()
    if not lab:
        return "Concept"
    if lab not in _ALLOWED_LABELS:
        return "Concept"
    return lab


def _sanitize_kg_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
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

    cleaned_nodes = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        name = (n.get("name") or "").strip()
        if not name:
            continue
        n["label"] = _normalize_label(n.get("label") or "Concept")
        cleaned_nodes.append(n)

    cleaned_rels = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        if not (r.get("type") and r.get("source_node_name") and r.get("target_node_name")):
            continue
        r["source_node_label"] = _normalize_label(r.get("source_node_label") or "Concept")
        r["target_node_label"] = _normalize_label(r.get("target_node_label") or "Concept")
        cleaned_rels.append(r)

    obj["nodes"] = cleaned_nodes
    obj["relationships"] = cleaned_rels
    obj["facts"] = facts
    obj["insights"] = insights
    return obj


def build_relation_extraction_prompt(
    *,
    text: str,
    virtual_time: str,
    session_time_iso: str,
) -> str:
    recorded_turn_id = -1
    try:
        import re

        m = re.search(r"(\d+)", str(virtual_time or ""))
        recorded_turn_id = int(m.group(1)) if m else -1
    except Exception:
        recorded_turn_id = -1

    time_hint = session_time_iso or "unknown"

    return f"""
You are extracting a compact knowledge graph from dialogue text.
Output a STRICT JSON object with keys:
- nodes: array of nodes: {{name: str, label: str, properties: object}}
- relationships: array of relationships:
  {{source_node_name: str, source_node_label: str, target_node_name: str, target_node_label: str, type: str, properties: object}}
- facts: []
- insights: []

RULES:
1) Allowed labels: Person, Organization, Location, Object, Event, Date, Value, Concept.
2) Relationship type MUST be UPPERCASE.
3) Use concise, literal names (no long sentences).
4) Only extract facts clearly stated in the text; no hallucinations.
5) For relationship properties, include:
   - confidence (0.0-1.0)
   - source_of_belief ("observation")
   - event_timestamp if the text gives an explicit time, otherwise use "{time_hint}" only when it is a dataset session time.
6) If you create Event nodes, set properties.recorded_turn_id = {recorded_turn_id}.

TEXT:
{text}
""".strip()


def extract_relations_from_text(
    *,
    llm,
    text: str,
    virtual_time: str,
    session_time_iso: str,
) -> Optional[KnowledgeGraphExtraction]:
    if not text or not text.strip():
        return None

    prompt = build_relation_extraction_prompt(
        text=text,
        virtual_time=virtual_time,
        session_time_iso=session_time_iso,
    )
    kg_raw = invoke_json(llm, prompt)
    try:
        import json

        obj = json.loads(kg_raw) if isinstance(kg_raw, str) else kg_raw
    except Exception:
        obj = {}

    obj = _sanitize_kg_dict(obj)
    if not obj:
        return None

    if os.getenv("DEBUG_REL_EXTRACTOR", "0") == "1":
        logger.info(
            f"[relation_extractor] nodes={len(obj.get('nodes') or [])} "
            f"rels={len(obj.get('relationships') or [])}"
        )

    return KnowledgeGraphExtraction(**obj)
