from __future__ import annotations

import re
import datetime
import logging
from typing import List, Tuple, Dict, Any, Optional

from config import short_memory_number, long_memory_number

logger = logging.getLogger(__name__)

# ===== 新格式检测：[Fact N] / [Match N] / [Turn N] ... =====
_NEW_FACT_FORMAT_RE = re.compile(r"^\[(Fact|Match|Turn) \d+\]")
_NEW_SECTION_HEADER_RE = re.compile(r"^=== (KEYWORD MATCHED FACTS|LONG-TERM MEMORY FACTS|ORIGINAL TEXT)")

# ===== 旧格式：Canonicalize triples to one stable evidence format =====
_TRIPLE_CANON_RE_1 = re.compile(
    r"^\s*(?:-\s*)?\(\s*(?P<a>[^)]+?)\s*\)\s*-\[\s*(?P<rel>[A-Za-z0-9_:\-]+)\s*\]->\s*\(\s*(?P<b>[^)]+?)\s*\)\s*(?P<meta>\[.*\])?\s*$"
)
_TRIPLE_CANON_RE_2 = re.compile(
    r"^\s*(?:-\s*)?(?P<a>.+?)\s*-\s*(?P<rel>[A-Za-z0-9_:\-]+)\s*-\s*(?P<b>.+?)\s*(?P<trailing>.*?)\s*$"
)

_META_KV_BRACKET_RE = re.compile(r"(\w+)\s*=\s*([^;\]]+)")
_META_KV_STUCK_RE = re.compile(
    r"(source|confidence|knowledge_type|evidence_unit|event_time|eventtime|channel|score|alt|slot_id|event_id|turn_id)\s*([A-Za-z0-9_.:\-]+)",
    re.IGNORECASE,
)

# 解析 canonical meta 的 key=value
_CANON_META_RE = re.compile(r"\[(?P<meta>.*)\]$")
_CANON_META_KV_RE = re.compile(r"(\w+)\s*=\s*([^;]+)")


def _canonicalize_meta(meta: dict) -> str:
    key_order = [
        "channel", "score", "confidence", "source", "knowledge_type",
        "evidence_unit", "event_time", "event_id", "slot_id", "turn_id", "alt",
    ]
    parts = []
    for k in key_order:
        if k in meta and str(meta[k]).strip():
            parts.append(f"{k}={str(meta[k]).strip()}")
    for k in sorted(meta.keys()):
        if k in key_order:
            continue
        v = str(meta[k]).strip()
        if v:
            parts.append(f"{k}={v}")
    return "; ".join(parts)


def _looks_like_triple(s: str) -> bool:
    if not s:
        return False
    ss = s.strip()
    if re.search(r"\)\s*-\[\s*[A-Za-z0-9_:\-]+\s*\]->\s*\(", ss):
        return True
    if re.search(r"\s-\s*[A-Za-z0-9_:\-]+\s-\s", ss):
        return True
    return False


def canonicalize_triple_line(line: str) -> str:
    if not isinstance(line, str):
        return line
    raw = line.strip()
    if not raw:
        return line
    s = raw
    if s.startswith("•"):
        s = "- " + s.lstrip("•").strip()
    if (not s.startswith("-")) and _looks_like_triple(s):
        s = "- " + s
    if not s.startswith("-"):
        return line

    m1 = _TRIPLE_CANON_RE_1.match(s)
    if m1:
        a = m1.group("a").strip()
        rel = m1.group("rel").strip()
        b = m1.group("b").strip()
        meta_raw = (m1.group("meta") or "").strip()
        meta = {}
        if meta_raw.startswith("[") and meta_raw.endswith("]"):
            for k, v in _META_KV_BRACKET_RE.findall(meta_raw):
                meta[k.strip()] = v.strip()
        meta_str = _canonicalize_meta(meta)
        return f"- ({a}) -[{rel}]-> ({b})" + (f" [{meta_str}]" if meta_str else "")

    m2 = _TRIPLE_CANON_RE_2.match(s)
    if not m2:
        return s
    a = m2.group("a").strip()
    rel = m2.group("rel").strip()
    b = m2.group("b").strip()
    trailing = (m2.group("trailing") or "").strip()
    meta = {}
    bracket = None
    if "[" in trailing and "]" in trailing:
        lb = trailing.find("[")
        rb = trailing.rfind("]")
        if 0 <= lb < rb:
            bracket = trailing[lb: rb + 1]
    if bracket:
        for k, v in _META_KV_BRACKET_RE.findall(bracket):
            meta[k.strip()] = v.strip()
        trailing_wo = (trailing.replace(bracket, " ")).strip()
    else:
        trailing_wo = trailing
    for k, v in _META_KV_STUCK_RE.findall(trailing_wo):
        kk = k.lower()
        vv = v.strip()
        if kk == "eventtime":
            kk = "event_time"
        meta[kk] = vv
    meta_str = _canonicalize_meta(meta)
    return f"- ({a}) -[{rel}]-> ({b})" + (f" [{meta_str}]" if meta_str else "")


def canonicalize_triple_block(text: str) -> str:
    """
    对三元组块进行规范化处理。
    
    ✅ 新格式检测：如果是 [Fact N] 格式，直接返回原文，不做处理。
    新格式已经是规范化的，不需要再处理。
    """
    if not isinstance(text, str) or not text.strip():
        return text
    
    # ✅ 检测新格式：如果包含 [Fact N] 开头的行，直接返回
    if _is_new_fact_format(text):
        return text
    
    return "\n".join([canonicalize_triple_line(ln) for ln in text.splitlines()])


def _parse_canon_meta(line: str) -> Dict[str, str]:
    m = _CANON_META_RE.search(line.strip())
    if not m:
        return {}
    meta_str = (m.group("meta") or "").strip()
    out: Dict[str, str] = {}
    for k, v in _CANON_META_KV_RE.findall(meta_str):
        out[k.strip()] = v.strip()
    return out


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
        return int(float(x))
    except Exception:
        return default


def _extract_query_entities(query: str) -> List[str]:
    if not query:
        return []
    stop_words = {
        "what", "where", "when", "which", "who", "how", "why",
        "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
        "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her",
        "this", "that", "these", "those", "in", "on", "at", "to", "for", "of",
        "and", "or", "but", "not", "with", "from", "by", "about",
        "first", "last", "before", "after", "many", "much", "all", "any",
        "tell", "show", "give", "find", "get", "list", "describe", "explain",
    }
    title_prefixes = {"dr.", "mr.", "mrs.", "ms.", "prof.", "dr", "mr", "mrs", "ms", "prof"}
    words = query.split()
    entities = []
    current_entity = []
    for word in words:
        word_lower = word.lower().rstrip(",?!;:'\"()[]")
        if word_lower in title_prefixes:
            if word_lower.endswith('.'):
                current_entity.append(word_lower.capitalize())
            else:
                current_entity.append(word_lower.capitalize() + ".")
            continue
        clean_word = word.strip(",?!;:'\"()[]")
        if clean_word.endswith('.') and clean_word.lower() not in title_prefixes:
            clean_word = clean_word.rstrip('.')
        if clean_word and clean_word[0].isupper() and clean_word.lower() not in stop_words:
            current_entity.append(clean_word)
        else:
            if current_entity:
                entity = " ".join(current_entity)
                if len(entity) > 1:
                    entities.append(entity)
                current_entity = []
    if current_entity:
        entity = " ".join(current_entity)
        if len(entity) > 1:
            entities.append(entity)
    return entities


def _ltss_line_score(line: str, query_entities: Optional[List[str]] = None) -> float:
    meta = _parse_canon_meta(line)
    ch = (meta.get("channel") or "").lower()
    alt = (meta.get("alt") or "").lower() == "true"
    score = _safe_float(meta.get("score"), default=0.0)
    conf = _safe_float(meta.get("confidence"), default=0.0)
    base = score if score > 0 else conf
    bonus = 1.25 if ch == "consolidated" else 1.0
    penalty = 0.65 if alt else 1.0
    entity_bonus = 1.0
    if query_entities:
        line_lower = line.lower()
        for entity in query_entities:
            if entity.lower() in line_lower:
                entity_bonus += 0.8
    return base * bonus * penalty * entity_bonus


def _ltss_line_key(line: str) -> str:
    meta = _parse_canon_meta(line)
    if meta.get("slot_id"):
        return f"slot:{meta.get('slot_id')}"
    head = line
    if "[" in head:
        head = head[: head.rfind("[")].strip()
    eid = meta.get("event_id", "")
    return f"{eid}::{head}" if eid else head


# ===== 新格式处理函数 =====

def _is_new_fact_format(ltss_context: str) -> bool:
    """
    检测是否是新的 SimpleRetriever 格式
    
    新格式特征：
    - Section headers: === KEYWORD MATCHED FACTS ===, === LONG-TERM MEMORY FACTS ===, etc.
    - Evidence lines: [Fact N], [Match N], [Turn N]
    """
    if not ltss_context:
        return False
    
    # 检查前 20 行（增加检查范围，因为可能有 section headers）
    for line in ltss_context.splitlines()[:20]:
        line_stripped = line.strip()
        # 检查是否有新格式的 section header
        if _NEW_SECTION_HEADER_RE.match(line_stripped):
            return True
        # 检查是否有新格式的 evidence line
        if _NEW_FACT_FORMAT_RE.match(line_stripped):
            return True
    
    return False


def _prune_ltss_triples(
    ltss_context: str,
    *,
    max_lines: int = 48,
    max_chars: int = 6500,
    max_alt_lines: int = 8,
    query_entities: Optional[List[str]] = None,
) -> str:
    """对 Graph triples 做裁剪，自动检测新旧格式"""
    if not isinstance(ltss_context, str) or not ltss_context.strip():
        return "None"
    
    # ✅ 新格式：SimpleRetriever 已经完成评分、去重、排序，直接返回
    if _is_new_fact_format(ltss_context):
        return ltss_context
    
    # 旧格式处理
    canon = canonicalize_triple_block(ltss_context)
    lines = [ln.strip() for ln in canon.splitlines() if ln.strip().startswith("-")]
    if not lines:
        return "None"

    scored = []
    for ln in lines:
        meta = _parse_canon_meta(ln)
        scored.append((_ltss_line_score(ln, query_entities), _safe_int(meta.get("turn_id"), 0), ln))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    kept: List[str] = []
    seen = set()
    alt_cnt = 0
    total_chars = 0

    for _, __, ln in scored:
        k = _ltss_line_key(ln)
        if k in seen:
            continue
        meta = _parse_canon_meta(ln)
        if (meta.get("alt") or "").lower() == "true":
            if alt_cnt >= max_alt_lines:
                continue
            alt_cnt += 1
        new_chars = total_chars + len(ln) + 1
        if len(kept) >= max_lines or new_chars > max_chars:
            continue
        seen.add(k)
        kept.append(ln)
        total_chars = new_chars

    return "\n".join(kept) if kept else "None"


class ContextBuilder:
    """构建最终给 LLM 的上下文"""

    @staticmethod
    def _is_good_episodic(m: str) -> bool:
        if not isinstance(m, str):
            return False
        s = m.strip()
        if not s or len(s) > 280:
            return False
        bad_markers = [
            "Evidence", "evidence", "Final Answer", "json", "{", "}",
            "You should", "recommend", "tips", "consider", "compare prices",
        ]
        low = s.lower()
        if any(b.lower() in low for b in bad_markers):
            return False
        if s.count("-") > 2:
            return False
        return True

    @classmethod
    def build_stes_context(cls, stes_memories: List[str]) -> str:
        filtered = [m.strip() for m in (stes_memories or []) if cls._is_good_episodic(m)]
        if not filtered:
            return "None"
        return "\n".join([f"- {m}" for m in filtered])

    @staticmethod
    def build_temporal_triples(observation: str, graphrag_retriever, agent_name: str) -> Tuple[List[str], str]:
        temporal_triples: List[str] = []
        reason = "not triggered"
        try:
            from temporal_reasoning.intent_router import detect_intent
            from temporal_reasoning.executor import run_template
            from temporal_reasoning.cypher_templates import FIRST_EVENT_AFTER_ANCHOR

            intent_obj = detect_intent(observation) or {}
            intent = intent_obj.get("intent")
            driver = getattr(graphrag_retriever, "driver", None)
            if intent == "FIRST_AFTER" and driver:
                kws = intent_obj.get("keywords", []) or []
                anchor_keyword = ""
                for k in kws:
                    if isinstance(k, str) and k.strip():
                        anchor_keyword = k.strip()
                        break
                if not anchor_keyword:
                    return [], "intent FIRST_AFTER matched but no anchor_keyword extracted"
                temporal_triples = run_template(
                    driver, FIRST_EVENT_AFTER_ANCHOR,
                    {"agent_name": agent_name or "User", "anchor_keyword": anchor_keyword, "limit_val": 10},
                ) or []
                reason = f"intent FIRST_AFTER matched (anchor={anchor_keyword})"
            else:
                reason = f"intent not matched ({intent})"
        except Exception as e:
            reason = f"exception: {e}"

        uniq, seen = [], set()
        for t in temporal_triples:
            if not isinstance(t, str):
                continue
            k = t.strip()
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(k)
        return uniq[:3], reason

    @staticmethod
    def build_derived_facts(observation: str, graphrag_retriever, agent_name: str = "User") -> Tuple[List[str], List[str]]:
        derived_facts: List[str] = []
        derived_evidence: List[str] = []
        try:
            from temporal_reasoning.intent_router import detect_intent
            from temporal_reasoning.executor import run_template, run_query
            from temporal_reasoning import cypher_templates as T

            intent_obj = detect_intent(observation) or {}
            intent = intent_obj.get("intent", "NONE")
            keywords = intent_obj.get("keywords", []) or []
            driver = getattr(graphrag_retriever, "driver", None)
            if not driver:
                return [], []

            if intent == "COUNT" and keywords:
                rows = run_query(driver, T.COUNT_DISTINCT_TARGETS_BY_KEYWORDS,
                    {"keywords": keywords, "agent_name": agent_name, "limit_val": 50})
                cnt = rows[0].get("cnt") if rows else None
                triples = run_template(driver, T.COUNT_DISTINCT_TARGETS_TRIPLES, {"keywords": keywords, "limit_val": 5}) or []
                if cnt is not None:
                    derived_facts.append(f"[DERIVED] count={cnt} (distinct targets matched by keywords={keywords})")
                    derived_evidence.extend(triples)

            if intent == "TIME_DIFF":
                clause_a = intent_obj.get("clause_a", "") or ""
                clause_b = intent_obj.get("clause_b", "") or ""
                kw_a_obj = detect_intent(clause_a) if clause_a else {}
                kw_b_obj = detect_intent(clause_b) if clause_b else {}
                kw_a = (kw_a_obj or {}).get("keywords", []) or []
                kw_b = (kw_b_obj or {}).get("keywords", []) or []

                def _pick_date(triples: List[str]) -> str:
                    for t in triples:
                        m = re.search(r"event_time=(\d{4}-\d{2}-\d{2})", t)
                        if m:
                            return m.group(1)
                    return "unknown"

                t1 = run_template(driver, T.EVENT_DATES_BY_KEYWORDS, {"keywords": kw_a or keywords, "limit_val": 3}) or []
                t2 = run_template(driver, T.EVENT_DATES_BY_KEYWORDS, {"keywords": kw_b or keywords, "limit_val": 3}) or []
                d1, d2 = _pick_date(t1), _pick_date(t2)
                if d1 != "unknown" and d2 != "unknown":
                    dt1 = datetime.date.fromisoformat(d1)
                    dt2 = datetime.date.fromisoformat(d2)
                    days = abs((dt2 - dt1).days)
                    derived_facts.append(f"[DERIVED] days_between={days} (from {d1} and {d2})")
                    derived_evidence.extend((t1[:1] if t1 else []) + (t2[:1] if t2 else []))
        except Exception as e:
            logger.info(f"[ContextBuilder] Derived facts injection skipped: {e}")
        return derived_facts, derived_evidence

    @staticmethod
    def build_final_context(
        stes_context: str,
        ltss_context: str = "",
        temporal_triples: List[str] | None = None,
        derived_facts: List[str] | None = None,
        derived_evidence: List[str] | None = None,
        consolidated_ltss_context: str | None = None,
        raw_ltss_context: str | None = None,
        query: str | None = None,
    ) -> str:
        temporal_triples = temporal_triples or []
        derived_facts = derived_facts or []
        derived_evidence = derived_evidence or []

        query_entities = _extract_query_entities(query) if query else None

        main_ltss = (
            consolidated_ltss_context
            if isinstance(consolidated_ltss_context, str) and consolidated_ltss_context.strip()
            else ltss_context
        )

        main_ltss_pruned = _prune_ltss_triples(
            main_ltss,
            max_lines=max(48, int(long_memory_number) * 6),
            max_chars=15000,
            max_alt_lines=10,
            query_entities=query_entities,
        )

        raw_block = "None"
        if isinstance(raw_ltss_context, str) and raw_ltss_context.strip():
            raw_pruned = _prune_ltss_triples(
                raw_ltss_context,
                max_lines=12,
                max_chars=2800,
                max_alt_lines=8,
                query_entities=query_entities,
            )
            raw_block = raw_pruned or "None"

        temporal_triples = [canonicalize_triple_line(t) for t in temporal_triples]
        derived_evidence = [canonicalize_triple_line(t) for t in derived_evidence]

        temporal_block = "\n".join([t for t in temporal_triples if isinstance(t, str) and t.strip()]) if temporal_triples else "None"
        derived_block = "\n".join(derived_facts + derived_evidence) if (derived_facts or derived_evidence) else "None"

        # ⚠️ 不要添加额外的 header，直接使用 SimpleRetriever 的输出
        # SimpleRetriever 已经包含了正确的 headers 和顺序：
        # 1. === KEYWORD MATCHED FACTS (关键词检索) ===
        # 2. === LONG-TERM MEMORY FACTS (语义检索) ===
        # 3. === ORIGINAL TEXT (原文兜底) ===
        return (
            f"{main_ltss_pruned}\n\n"
            "=== EPISODIC MEMORY (RECENT CONTEXT) ===\n"
            f"{stes_context}\n\n"
            "=== RAW SUPPLEMENT (ALLOWED EVIDENCE) ===\n"
            f"{raw_block}\n\n"
            "=== TIMELINE (ALLOWED EVIDENCE) ===\n"
            f"{temporal_block}\n\n"
            "=== DERIVED FACTS (ALLOWED EVIDENCE) ===\n"
            f"{derived_block}\n"
        )
