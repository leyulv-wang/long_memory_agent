# -*- coding: utf-8 -*-
"""
Deterministic Derived Answer Layer (Robust, Conservative)

Goal:
- For VALUE / DURATION / BEST / LIST style questions, try to derive the answer
  deterministically from retrieved_context (Graph/Timeline triples + Derived Facts block).
- Be conservative: only answer when evidence is strong and unambiguous.
- Otherwise return None and let LLM handle it.

This file is intended to be the SINGLE source of truth.
Use memory/derived_answer.py as a thin re-export shim if needed.
"""

from typing import List, Optional, Dict, Any, Tuple
import logging
import os
import re
# 引入新工具
from utils.triple_parser import parse_triple, EvidenceTriple
from dataclasses import dataclass

# ----------------------------
# Utilities: parsing evidence lines
# ----------------------------

TRIPLE_RE = re.compile(
    r"^\s*-\s*\(\s*(?P<a>.*?)\s*\)\s*-\[\s*(?P<rel>[A-Za-z0-9_:\-]+)\s*\]->\s*\(\s*(?P<b>.*?)\s*\)\s*(?P<meta>\[.*\])?\s*$"
)

META_KV_RE = re.compile(r"(\w+)\s*=\s*([^\];]+)")

TIME_VALUE_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

# Word-number durations like "a year", "an hour", "one day", "over a year"
TIME_VALUE_WORD_RE = re.compile(
    r"\b(?:(over|more than|less than|about|around)\s+)?"
    r"(a|an|one)\s+"
    r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _lower(s: str) -> str:
    return _norm(s).lower()


@dataclass
class EvidenceLine:
    raw: str
    a: str
    rel: str
    b: str
    meta: Dict[str, str]

    @property
    def rel_u(self) -> str:
        return self.rel.upper()

    @property
    def a_l(self) -> str:
        return _lower(self.a)

    @property
    def b_l(self) -> str:
        return _lower(self.b)


def parse_evidence_lines(retrieved_context: str) -> List[EvidenceLine]:
    """
    Parse ONLY evidence-like triple lines from the context.
    We intentionally do NOT parse episodic lines.
    """
    if not retrieved_context:
        return []

    lines: List[EvidenceLine] = []
    for raw in (retrieved_context.splitlines() or []):
        raw = raw.rstrip("\n")
        m = TRIPLE_RE.match(raw.strip())
        if not m:
            continue

        meta_text = (m.group("meta") or "").strip()
        meta: Dict[str, str] = {}
        for k, v in META_KV_RE.findall(meta_text):
            meta[k.strip()] = v.strip()

        lines.append(
            EvidenceLine(
                raw=raw.strip(),
                a=_norm(m.group("a")),
                rel=_norm(m.group("rel")),
                b=_norm(m.group("b")),
                meta=meta,
            )
        )
    if os.getenv("DEBUG_DERIVED", "0") == "1":
        logger = logging.getLogger(__name__)
        logger.info(f"[derived_answer] parsed evidence lines={len(lines)}")
        for i, ln in enumerate(lines[:5], 1):
            logger.info(f"[derived_answer] line[{i}] {ln.raw}")
    return lines


def slice_allowed_evidence(retrieved_context: str) -> str:
    """
    Keep only:
      - === GRAPH TRIPLES (ALLOWED EVIDENCE) ===
      - === TIMELINE TRIPLES (ALLOWED EVIDENCE) ===
      - === DERIVED FACTS (ALLOWED EVIDENCE) ===
    If headings missing, return original (fail open but still safe because our parser is strict).
    """
    if not retrieved_context:
        return ""

    # Prefer keeping blocks if present
    patterns = [
        r"(=== GRAPH TRIPLES \(ALLOWED EVIDENCE\) ===.*?)(?:(=== [A-Z ].*? ===)|\Z)",
        r"(=== RAW GRAPH TRIPLES \(SUPPLEMENT, ALLOWED EVIDENCE\) ===.*?)(?:(=== [A-Z ].*? ===)|\Z)",
        r"(=== TIMELINE TRIPLES \(ALLOWED EVIDENCE\) ===.*?)(?:(=== [A-Z ].*? ===)|\Z)",
        r"(=== DERIVED FACTS \(ALLOWED EVIDENCE\) ===.*?)(?:(=== [A-Z ].*? ===)|\Z)",
    ]

    kept: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, retrieved_context, re.DOTALL):
            kept.append(m.group(1))

    if kept:
        return "\n".join(kept)

    return retrieved_context


# ----------------------------
# Question typing (lightweight, robust)
# ----------------------------

@dataclass
class QuestionIntent:
    kind: str  # "DURATION" | "BEST" | "VALUE" | "LIST" | "NONE"
    ask_n: Optional[int] = None


def detect_question_intent(question: str) -> QuestionIntent:
    q = _lower(question)

    # LIST detection (two hobbies, three items, list X)
    if any(w in q for w in ["two ", "three ", "four ", "list ", "which ", "what are the"]):
        # Explicit "two/three/four"
        if "two " in q:
            return QuestionIntent(kind="LIST", ask_n=2)
        if "three " in q:
            return QuestionIntent(kind="LIST", ask_n=3)
        if "four " in q:
            return QuestionIntent(kind="LIST", ask_n=4)

    # BEST / personal best
    if any(w in q for w in ["personal best", "pb", "best time", "fastest time", "previous personal best"]):
        return QuestionIntent(kind="BEST")

    # DURATION (how long, minutes, duration)
    if any(w in q for w in ["how long", "duration", "minutes", "hours", "days", "per day", "daily"]):
        return QuestionIntent(kind="DURATION")

    # VALUE (how much, number)
    if any(w in q for w in ["how much", "amount", "value", "cost", "price", "percent", "%"]):
        return QuestionIntent(kind="VALUE")

    return QuestionIntent(kind="NONE")


_WORD_TO_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _extract_requested_count(question: str) -> Optional[int]:
    ql = _lower(question)
    for w, n in _WORD_TO_NUM.items():
        if f"{w} " in ql:
            return n
    m = re.search(r"\b(\d+)\b", ql)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def _extract_event_phrases(question: str) -> Optional[Tuple[str, str]]:
    q = _lower(question)
    if " or " not in q:
        return None
    parts = q.split(" or ", 1)
    if len(parts) != 2:
        return None
    left = parts[0]
    right = parts[1]
    for junk in ["did i", "was i", "do i", "am i", "first", "before", "after", "mention", "say", "tell", "ask", "?"]:
        left = left.replace(junk, " ")
        right = right.replace(junk, " ")
    left = _norm(left)
    right = _norm(right)
    if not left or not right:
        return None
    return left, right


_MUSEUM_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&'\\-\\s]+?Museum(?:\s+of\s+[A-Z][A-Za-z0-9&'\\-\\s]+)?(?:'s)?)\b",
    re.IGNORECASE,
)


def _extract_museums_from_text(text: str) -> List[str]:
    if not text:
        return []
    hits: List[str] = []
    for m in _MUSEUM_NAME_RE.finditer(text):
        name = _norm(m.group(1)).replace("'s", "").strip()
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


# ----------------------------
# Core derivation engine
# ----------------------------

REL_DURATION_HINTS = {
    "HAS_DURATION", "LASTED", "TOOK", "DURATION", "PRACTICES_FOR", "SPENT",
    "HAS_VALUE", "HAS_TIME", "TIME_SPENT",
    "WAIT_TIME",
    # Consolidated TextFact edges are ASSERTS; allow duration parsing on these.
    "ASSERTS",
}

REL_VALUE_HINTS = {
    "HAS_VALUE", "AMOUNT", "COST", "PRICE", "SCORE", "HAS_PERCENT", "PERCENT",
    "HAS_NUMBER", "HAS_RATE"
}

REL_DISCOUNT_HINTS = {
    "FIRST_ORDER_DISCOUNT_PERCENT",
    "DISCOUNT_PERCENT",
    "HAS_PERCENT",
    "PERCENT",
    "HAS_VALUE",
    "ASSERTS",
}

REL_VISIT_HINTS = {
    "VISITED",
    "VISITS",
    "WENT_TO",
    "ATTENDED",
}

KNOWN_BRANDS = [
    "hellofresh",
    "ubereats",
    "grubhub",
    "doordash",
    "postmates",
]

# HOBBY-like relations: split into strong vs weak to reduce noise.
REL_HOBBY_STRONG = {
    "HOBBY", "HOBBIES",
    "INTERESTED_IN", "ENJOYS", "LIKES",
    # 可按你的图谱再加： "HAS_HOBBY", "PREFERS", "IS_HOBBY_OF"
}

# Weak signals: may describe platform/community participation rather than the hobby itself.
REL_HOBBY_WEAK = {
    "PARTICIPATES_IN",
    # 可按你的图谱再加： "JOINS", "MEMBER_OF"
}

REL_BEST_HINTS = {
    "PERSONAL_BEST", "HAS_PERSONAL_BEST", "BEST_TIME", "FASTEST_TIME", "HAS_VALUE"
}


def _extract_time_like(text: str) -> Optional[str]:
    """
    Extract '30 minutes' / '27 minutes and 45 seconds' variants.
    Conservative: require at least one time unit.
    """
    if not text:
        return None
    m = TIME_VALUE_RE.search(text)
    if not m:
        m = TIME_VALUE_WORD_RE.search(text)
        if not m:
            return None
    # If text contains a richer phrase like "27 minutes and 45 seconds", keep it as-is if it has units.
    # Otherwise just normalize the matched portion.
    return _norm(text)


def _extract_percent_value(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _score_line(line: EvidenceLine) -> float:
    """
    Score by meta fields if present:
    - source=ground_truth higher
    - confidence higher
    - event_time / consolidated_at presence slightly higher
    """
    score = 0.0
    src = (line.meta.get("source") or line.meta.get("source_of_belief") or "").lower()
    if src == "ground_truth":
        score += 5.0
    elif src:
        score += 1.0

    conf_s = line.meta.get("confidence", "")
    try:
        score += float(conf_s)
    except Exception:
        pass

    if line.meta.get("event_time") or line.meta.get("event_timestamp"):
        score += 0.5
    if line.meta.get("consolidated_at"):
        score += 0.2
    return score


class DerivedAnswerEngine:
    """
    Given question + retrieved_context (already built by CFF),
    try to produce (final_answer, evidence_lines) deterministically.
    """

    def derive(self, question: str, retrieved_context: str) -> Optional[Dict[str, Any]]:
        if not question or not retrieved_context:
            return None

        allowed_ctx = slice_allowed_evidence(retrieved_context)
        lines = parse_evidence_lines(allowed_ctx)
        if not lines:
            return None

        ql = _lower(question)
        if any(k in ql for k in ["first", "before", "after"]):
            out = self._derive_temporal_order(question, lines)
            if out:
                return out
        if "discount" in ql and ("higher" in ql or "compare" in ql or "percent" in ql):
            out = self._derive_discount_compare(question, lines)
            if out:
                return out
        if "order" in ql and "museum" in ql:
            out = self._derive_museum_order(question, lines)
            if out:
                return out

        intent = detect_question_intent(question)

        if intent.kind == "DURATION":
            return self._derive_duration(question, lines)

        if intent.kind == "BEST":
            return self._derive_best(question, lines)

        if intent.kind == "VALUE":
            return self._derive_value(question, lines)

        if intent.kind == "LIST":
            return self._derive_list(question, lines, ask_n=intent.ask_n)

        return None

    def _derive_temporal_order(self, question: str, lines: List[EvidenceLine]) -> Optional[Dict[str, Any]]:
        ql = _lower(question)
        phrases = _extract_event_phrases(question)
        if not phrases:
            return None
        left, right = phrases
        left_tokens = [t for t in re.split(r"[^a-z0-9]+", left) if len(t) >= 4]
        right_tokens = [t for t in re.split(r"[^a-z0-9]+", right) if len(t) >= 4]
        if not left_tokens or not right_tokens:
            return None

        def _pick_best(tokens: List[str]) -> Optional[Tuple[int, EvidenceLine]]:
            best: Optional[Tuple[int, EvidenceLine]] = None
            for ln in lines:
                if not any(t in ln.a_l or t in ln.b_l for t in tokens):
                    continue
                turn_id = ln.meta.get("turn_id")
                if turn_id is None:
                    continue
                try:
                    tid = int(float(turn_id))
                except Exception:
                    continue
                if best is None or tid < best[0]:
                    best = (tid, ln)
            return best

        left_best = _pick_best(left_tokens)
        right_best = _pick_best(right_tokens)
        if not left_best or not right_best:
            return None

        left_tid, left_ln = left_best
        right_tid, right_ln = right_best

        if "before" in ql:
            final_answer = "Yes." if left_tid < right_tid else "No."
        elif "after" in ql:
            final_answer = "Yes." if left_tid > right_tid else "No."
        else:
            final_answer = left if left_tid < right_tid else right

        return {
            "final_answer": final_answer,
            "evidence_triples": [left_ln.raw, right_ln.raw],
            "reason": "temporal_order",
        }

    def _derive_discount_compare(self, question: str, lines: List[EvidenceLine]) -> Optional[Dict[str, Any]]:
        ql = _lower(question)
        brand_hits = []
        for b in KNOWN_BRANDS:
            if b in ql:
                brand_hits.append(b)
        if len(brand_hits) < 2:
            return None

        # Map brand -> best (score, percent, evidence)
        best: Dict[str, Tuple[float, float, EvidenceLine]] = {}

        for ln in lines:
            if ln.rel_u not in REL_DISCOUNT_HINTS:
                continue
            percent = _extract_percent_value(ln.b) or _extract_percent_value(ln.a)
            if percent is None:
                continue
            for b in brand_hits:
                if b in ln.a_l or b in ln.b_l:
                    sc = _score_line(ln)
                    prev = best.get(b)
                    if not prev or sc > prev[0]:
                        best[b] = (sc, percent, ln)

        if len(best) < 2:
            return None

        b1, b2 = brand_hits[0], brand_hits[1]
        if b1 not in best or b2 not in best:
            return None

        _, p1, l1 = best[b1]
        _, p2, l2 = best[b2]

        if abs(p1 - p2) < 1e-6:
            final_answer = "No."
        else:
            final_answer = "Yes." if p1 > p2 else "No."

        return {
            "final_answer": final_answer,
            "evidence_triples": [l1.raw, l2.raw],
            "reason": "discount_compare",
        }

    def _derive_museum_order(self, question: str, lines: List[EvidenceLine]) -> Optional[Dict[str, Any]]:
        ql = _lower(question)
        if "museum" not in ql:
            return None

        requested = _extract_requested_count(question)

        museums: Dict[str, Tuple[int, EvidenceLine]] = {}
        for ln in lines:
            is_visit = ln.rel_u in REL_VISIT_HINTS
            if not is_visit and ln.rel_u == "ASSERTS":
                if "visited" in ln.a_l or "visited" in ln.b_l:
                    is_visit = True
            if not is_visit:
                continue

            text_blob = f"{ln.a} {ln.b}"
            names = _extract_museums_from_text(text_blob)
            if not names:
                continue

            tid_raw = ln.meta.get("turn_id")
            if tid_raw is None:
                continue
            try:
                tid = int(float(tid_raw))
            except Exception:
                continue

            for name in names:
                if "museum" not in _lower(name):
                    continue
                prev = museums.get(name)
                if prev is None or tid < prev[0]:
                    museums[name] = (tid, ln)

        if not museums:
            return None

        ordered = sorted(museums.items(), key=lambda kv: kv[1][0])
        if requested and len(ordered) < requested:
            return None

        names = [name for (name, _) in ordered[: requested or len(ordered)]]
        evidence = [museums[name][1].raw for name in names]
        final_answer = ", ".join(names)

        return {
            "final_answer": final_answer,
            "evidence_triples": evidence,
            "reason": "museum_order",
        }

    # -------- DURATION --------
    def _derive_duration(self, question: str, lines: List[EvidenceLine]) -> Optional[Dict[str, Any]]:
        ql = _lower(question)

        # Find time-like nodes or time-like objects linked via duration-ish relations
        cands: List[Tuple[float, EvidenceLine, str]] = []

        for ln in lines:
            # Direct object is time-like and relation suggests duration/value
            if ln.rel_u in REL_DURATION_HINTS or "DURATION" in ln.rel_u:
                t = _extract_time_like(ln.b)
                if t:
                    cands.append((_score_line(ln), ln, t))
                    continue

            # Sometimes duration is stored in node.name or node.value-like text
            # We accept time-like anywhere in a/b but only if relation implies duration-ish
            if ln.rel_u in REL_DURATION_HINTS:
                t2 = _extract_time_like(ln.a)
                if t2:
                    cands.append((_score_line(ln), ln, t2))

        # Extra fallback: if question mentions daily practice, look for any line containing the activity
        # and a separate line that gives duration for that activity (two-hop via common node name).
        # Conservative: require shared anchor node string.
        if not cands:
            # anchors: nouns from question (very rough)
            anchors = [w for w in re.split(r"[^a-z0-9]+", ql) if w and len(w) >= 4]
            anchors = anchors[:8]
            for ln in lines:
                if any(a in ln.a_l or a in ln.b_l for a in anchors):
                    # if this line references an anchor activity, try find another duration line mentioning same b
                    anchor_entity = ln.b_l
                    for ln2 in lines:
                        if ln2.rel_u in REL_DURATION_HINTS and (ln2.a_l == anchor_entity or ln2.b_l == anchor_entity):
                            t = _extract_time_like(ln2.a if ln2.a_l != anchor_entity else ln2.b)
                            if t:
                                cands.append((_score_line(ln2) - 0.2, ln2, t))

        if not cands:
            return None

        cands.sort(key=lambda x: x[0], reverse=True)
        best_score, best_line, duration_text = cands[0]

        # Conservative gating: require decent score OR ground_truth
        src = (best_line.meta.get("source") or best_line.meta.get("source_of_belief") or "").lower()
        if best_score < 1.2 and src != "ground_truth":
            return None

        return {
            "final_answer": duration_text,
            "evidence_triples": [best_line.raw],
            "reason": "duration_match",
        }

    # -------- BEST --------
    def _derive_best(self, question: str, lines: List[EvidenceLine]) -> Optional[Dict[str, Any]]:
        ql = _lower(question)

        # Look for best-related relations and time-like values
        cands: List[Tuple[float, EvidenceLine, str]] = []
        for ln in lines:
            if ln.rel_u in REL_BEST_HINTS or "BEST" in ln.rel_u:
                t = _extract_time_like(ln.b)
                if t:
                    cands.append((_score_line(ln) + 0.3, ln, t))

        # If none, fallback: if question mentions "5k" / "run" etc, accept HAS_VALUE with time-like b
        if not cands:
            anchors = [w for w in ["5k", "run", "race", "charity"] if w in ql]
            for ln in lines:
                if ln.rel_u == "HAS_VALUE":
                    t = _extract_time_like(ln.b)
                    if t and (any(a in ln.a_l or a in ln.b_l for a in anchors) or "personal best" in ql):
                        cands.append((_score_line(ln), ln, t))

        if not cands:
            return None

        cands.sort(key=lambda x: x[0], reverse=True)
        best_score, best_line, best_time = cands[0]

        src = (best_line.meta.get("source") or best_line.meta.get("source_of_belief") or "").lower()
        if best_score < 1.2 and src != "ground_truth":
            return None

        return {
            "final_answer": best_time,
            "evidence_triples": [best_line.raw],
            "reason": "best_match",
        }

    # -------- VALUE --------
    def _derive_value(self, question: str, lines: List[EvidenceLine]) -> Optional[Dict[str, Any]]:
        # Conservative: only answer if we find a single strong numeric/value candidate.
        cands: List[Tuple[float, EvidenceLine, str]] = []

        for ln in lines:
            if ln.rel_u in REL_VALUE_HINTS:
                # value could be in (B) or in meta or in node.value text embedded in name
                val_text = ln.b
                if NUMBER_RE.search(val_text):
                    cands.append((_score_line(ln), ln, _norm(val_text)))

        if not cands:
            return None

        cands.sort(key=lambda x: x[0], reverse=True)
        best_score, best_line, val = cands[0]

        # If multiple close candidates, avoid answering
        if len(cands) >= 2 and abs(cands[0][0] - cands[1][0]) < 0.3:
            return None

        src = (best_line.meta.get("source") or best_line.meta.get("source_of_belief") or "").lower()
        if best_score < 1.2 and src != "ground_truth":
            return None

        return {
            "final_answer": val,
            "evidence_triples": [best_line.raw],
            "reason": "value_match",
        }

    # -------- LIST --------
    def _derive_list(self, question: str, lines: List[EvidenceLine], ask_n: Optional[int]) -> Optional[Dict[str, Any]]:
        """
        LIST is the hardest without strict schema.
        Robust strategy:
        1) Use strong hobby relations first (HOBBY/ENJOYS/LIKES/INTERESTED_IN)
        2) If still < N, then allow weak relations (PARTICIPATES_IN) to fill the gap
        3) Apply self-reference exclusion + meta-concept exclusion to reduce noise
        """
        if not ask_n or ask_n <= 0:
            return None

        ql = _lower(question)

        def is_self_reference(obj_l: str) -> bool:
            # self-reference: candidate appears in question text
            return bool(obj_l) and obj_l in ql

        def is_meta_concept(obj_l: str) -> bool:
            meta_blacklist = {
                "community", "communities", "online community", "online communities",
                "platform", "forum", "group",
                "online", "internet",
            }
            return obj_l in meta_blacklist

        def add_candidate(items: Dict[str, Tuple[float, EvidenceLine]], ln: EvidenceLine, obj: str, bonus: float = 0.0):
            obj = _norm(obj)
            if not obj:
                return
            obj_l = _lower(obj)

            # 1) 自指排除（核心）：题干里出现的对象不作为列表答案项
            if is_self_reference(obj_l):
                return

            # 2) 元概念排除：community/platform 等不算“爱好本体”
            if is_meta_concept(obj_l):
                return

            sc = _score_line(ln) + bonus
            # keep best evidence per item
            if obj_l not in items or sc > items[obj_l][0]:
                items[obj_l] = (sc, ln)

        # Pass 1: strong relations
        items: Dict[str, Tuple[float, EvidenceLine]] = {}
        for ln in lines:
            if ln.rel_u in REL_HOBBY_STRONG:
                add_candidate(items, ln, ln.b, bonus=0.6)

        # If enough, we can stop early
        if len(items) < ask_n:
            # Pass 2: weak relations (fill only)
            for ln in lines:
                if ln.rel_u in REL_HOBBY_WEAK:
                    add_candidate(items, ln, ln.b, bonus=0.0)

        if len(items) < ask_n:
            return None

        # rank by score, take top N
        sorted_items = sorted(items.items(), key=lambda kv: kv[1][0], reverse=True)
        top = sorted_items[:ask_n]

        # Conservative gating: ensure bottom item not too weak unless ground_truth
        bottom_score, bottom_ln = top[-1][1]
        src = (bottom_ln.meta.get("source") or bottom_ln.meta.get("source_of_belief") or "").lower()
        if bottom_score < 1.2 and src != "ground_truth":
            return None

        # Recover original casing from evidence line
        final_texts = [items[k][1].b for (k, _) in top]
        if ask_n == 2:
            final_answer = f"{final_texts[0]} and {final_texts[1]}"
        else:
            final_answer = ", ".join(final_texts)

        evidence = [items[k][1].raw for (k, _) in top]

        return {
            "final_answer": final_answer,
            "evidence_triples": evidence,
            "reason": "list_match_strong_then_weak",
        }


# ---- Public API expected by agent/answer_selector.py ----
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DerivedAnswer:
    final_answer: str
    evidence_triples: List[str]
    reason: str

_ENGINE = None  # lazy singleton (robust, avoid repeated regex compile cost)

def derive_answer(question: str, retrieved_context: str) -> Optional[DerivedAnswer]:
    """
    Public entrypoint.
    - Returns DerivedAnswer only when evidence is strong and unambiguous.
    - Otherwise returns None (let LLM handle it).
    """
    global _ENGINE
    if not question or not retrieved_context:
        return None

    if _ENGINE is None:
        _ENGINE = DerivedAnswerEngine()

    out = _ENGINE.derive(question, retrieved_context)
    if not out:
        return None

    final_answer = (out.get("final_answer") or "").strip()
    ev = out.get("evidence_triples") or []
    ev = [e for e in ev if isinstance(e, str) and e.strip()]

    # Hard gate: no evidence, no short-circuit.
    if not final_answer or not ev:
        return None

    return DerivedAnswer(
        final_answer=final_answer,
        evidence_triples=ev,
        reason=str(out.get("reason", "derived")),
    )


