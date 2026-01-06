from __future__ import annotations

import re
import datetime
import logging
from typing import List, Tuple, Dict, Any, Optional

from config import short_memory_number, long_memory_number  # 保持原有导入（即使当前文件未使用）

logger = logging.getLogger(__name__)

# ===== Canonicalize triples to one stable evidence format =====
# 允许前面没有 "-"，因为很多检索/打印会输出 "(A) -[R]-> (B) ..."
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
    """
    meta dict -> 统一排序的 key=value; key=value
    """
    key_order = [
        "channel",
        "score",
        "confidence",
        "source",
        "knowledge_type",
        "evidence_unit",
        "event_time",
        "event_id",
        "slot_id",
        "turn_id",
        "alt",
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
    """
    尽量稳：只把“明显像 triple”的行前置成 '-'，避免把普通文本误当证据。
    """
    if not s:
        return False
    ss = s.strip()
    # 典型形态：(A) -[REL]-> (B)
    if re.search(r"\)\s*-\[\s*[A-Za-z0-9_:\-]+\s*\]->\s*\(", ss):
        return True
    # 简化形态：A - REL - B
    if re.search(r"\s-\s*[A-Za-z0-9_:\-]+\s-\s", ss):
        return True
    return False


def canonicalize_triple_line(line: str) -> str:
    """
    把各种“类三元组文本”统一成：
    - (A) -[REL]-> (B) [k=v; k=v]

    关键稳定性修补：
    - 允许输入不是 '-' 开头，只要看起来像 triple，就自动补 '- '
    - 允许输入 bullet '•' 开头
    """
    if not isinstance(line, str):
        return line

    raw = line.strip()
    if not raw:
        return line

    s = raw

    # 统一 bullet
    if s.startswith("•"):
        s = "- " + s.lstrip("•").strip()

    # 关键：如果不是 '-' 开头但看起来像 triple，就补 '-'
    if (not s.startswith("-")) and _looks_like_triple(s):
        s = "- " + s

    # 如果仍不是 '-' 开头，就不碰（避免误伤普通文本）
    if not s.startswith("-"):
        return line

    # 形态 1：- (A) -[REL]-> (B) [meta]
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

    # 形态 2：- A - REL - B  (后面可能粘 meta)
    m2 = _TRIPLE_CANON_RE_2.match(s)
    if not m2:
        # 至少保证它是 '-' 开头的 evidence 行（稳定进入 catalog/prune）
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
    if not isinstance(text, str) or not text.strip():
        return text
    return "\n".join([canonicalize_triple_line(ln) for ln in text.splitlines()])


# =========================
# LTSS triple pruning/sorting
# =========================
def _parse_canon_meta(line: str) -> Dict[str, str]:
    """
    从 canonical triple 行里解析 [] 内部的 key=value
    """
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


def _ltss_line_score(line: str) -> float:
    """
    给 canonical triple 行打一个“保留优先级”分数：
    - 优先用 meta.score（来自 V2 检索/软更新）
    - 没有 score 就用 confidence
    - consolidated 给 bonus
    - alt=true 给惩罚（减少占用）
    """
    meta = _parse_canon_meta(line)
    ch = (meta.get("channel") or "").lower()
    alt = (meta.get("alt") or "").lower() == "true"
    score = _safe_float(meta.get("score"), default=0.0)
    conf = _safe_float(meta.get("confidence"), default=0.0)

    base = score if score > 0 else conf
    bonus = 1.25 if ch == "consolidated" else 1.0
    penalty = 0.65 if alt else 1.0
    return base * bonus * penalty


def _ltss_line_key(line: str) -> str:
    """
    去重 key：优先 slot_id，其次 event_id + head（A-rel-B）
    """
    meta = _parse_canon_meta(line)
    if meta.get("slot_id"):
        return f"slot:{meta.get('slot_id')}"

    head = line
    if "[" in head:
        head = head[: head.rfind("[")].strip()

    eid = meta.get("event_id", "")
    return f"{eid}::{head}" if eid else head


def _prune_ltss_triples(
    ltss_context: str,
    *,
    max_lines: int = 48,
    max_chars: int = 6500,
    max_alt_lines: int = 8,
) -> str:
    """
    对 Graph triples 做：
    - canonicalize（关键：现在会把非 '-' 开头的 triple 也变成 '-' evidence）
    - 逐行打分排序
    - 去重（同 slot_id 或同 head）
    - 控制 alt=true 的数量
    - 限制总行数 & 总字符数
    """
    if not isinstance(ltss_context, str) or not ltss_context.strip():
        return "None"

    canon = canonicalize_triple_block(ltss_context)

    # 只保留 '-' 开头 evidence 行；canonicalize 已保证“像 triple 的行”都会变成 '- ...'
    lines = [ln.strip() for ln in canon.splitlines() if ln.strip().startswith("-")]

    if not lines:
        return "None"

    scored = []
    for ln in lines:
        meta = _parse_canon_meta(ln)
        scored.append((_ltss_line_score(ln), _safe_int(meta.get("turn_id"), 0), ln))
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


def merge_graph_context(
    *,
    consolidated_context: str,
    raw_context: str,
) -> str:
    """
    consolidated 优先 + raw 补充（最终还会 prune）
    """
    parts = []
    if isinstance(consolidated_context, str) and consolidated_context.strip():
        parts.append(consolidated_context.strip())
    if isinstance(raw_context, str) and raw_context.strip():
        parts.append(raw_context.strip())

    merged = "\n".join(parts).strip()
    if not merged:
        return "None"
    return _prune_ltss_triples(merged)


class ContextBuilder:
    """
    构建最终给 LLM 的上下文：
    - episodic(STES) 可开关
    - graph: consolidated 优先 + raw 补充
    - timeline/derived: 仍保留
    """

    @staticmethod
    def _is_good_episodic(m: str) -> bool:
        if not isinstance(m, str):
            return False
        s = m.strip()
        if not s:
            return False
        if len(s) > 280:
            return False

        bad_markers = [
            "Evidence",
            "evidence",
            "Final Answer",
            "json",
            "{",
            "}",
            "You should",
            "recommend",
            "tips",
            "consider",
            "compare prices",
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
                    driver,
                    FIRST_EVENT_AFTER_ANCHOR,
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
        uniq = uniq[:3]

        return uniq, reason

    @staticmethod
    def build_derived_facts(
        observation: str,
        graphrag_retriever,
        agent_name: str = "User",
    ) -> Tuple[List[str], List[str]]:
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
                rows = run_query(
                    driver,
                    T.COUNT_DISTINCT_TARGETS_BY_KEYWORDS,
                    {"keywords": keywords, "agent_name": agent_name, "limit_val": 50},
                )
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
        # ===== 新签名兼容：consolidated/raw 双通道 =====
        consolidated_ltss_context: str | None = None,
        raw_ltss_context: str | None = None,
    ) -> str:
        temporal_triples = temporal_triples or []
        derived_facts = derived_facts or []
        derived_evidence = derived_evidence or []

        main_ltss = (
            consolidated_ltss_context
            if isinstance(consolidated_ltss_context, str) and consolidated_ltss_context.strip()
            else ltss_context
        )

        main_ltss_pruned = _prune_ltss_triples(
            main_ltss,
            max_lines=max(24, int(long_memory_number) * 4),
            max_chars=6500,
            max_alt_lines=6,
        )

        raw_block = "None"
        if isinstance(raw_ltss_context, str) and raw_ltss_context.strip():
            raw_pruned = _prune_ltss_triples(
                raw_ltss_context,
                max_lines=12,
                max_chars=2800,
                max_alt_lines=8,
            )
            raw_block = raw_pruned or "None"

        # 再做一次 canonicalize（稳）：保证 temporal/derived 的三元组也进 evidence
        temporal_triples = [canonicalize_triple_line(t) for t in temporal_triples]
        derived_evidence = [canonicalize_triple_line(t) for t in derived_evidence]

        temporal_block = "\n".join([t for t in temporal_triples if isinstance(t, str) and t.strip()]) if temporal_triples else "None"
        derived_block = "\n".join(derived_facts + derived_evidence) if (derived_facts or derived_evidence) else "None"

        return (
            "=== EPISODIC (DO NOT USE AS EVIDENCE) ===\n"
            f"{stes_context}\n\n"
            "=== GRAPH TRIPLES (ALLOWED EVIDENCE) ===\n"
            f"{main_ltss_pruned}\n\n"
            "=== RAW GRAPH TRIPLES (SUPPLEMENT, ALLOWED EVIDENCE) ===\n"
            f"{raw_block}\n\n"
            "=== TIMELINE TRIPLES (ALLOWED EVIDENCE) ===\n"
            f"{temporal_block}\n\n"
            "=== DERIVED FACTS (ALLOWED EVIDENCE) ===\n"
            f"{derived_block}\n"
        )
