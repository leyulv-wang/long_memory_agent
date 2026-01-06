# temporal_reasoning/intent_router.py
# -*- coding: utf-8 -*-

import re
from typing import Optional, Dict, Any, List

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "from", "and", "or",
    "did", "do", "does", "have", "has", "had", "is", "are", "was", "were",
    "you", "your", "i", "me", "my", "we", "our", "they", "their",
    "what", "when", "where", "who", "why", "how", "many", "much", "between",
    "different", "types", "type", "kind", "kinds", "recently", "last", "first",
}

# 长日期放前面，避免 03/22/2023 同时命中 03/22
_DATE_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",          # 2023-03-18
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",    # 03/22/2023
    r"\b\d{1,2}/\d{1,2}\b",            # 3/22
]


def _tokenize_keywords(text: str, max_k: int = 10) -> List[str]:
    t = re.sub(r"[^a-zA-Z0-9\s'-]", " ", (text or "").lower())
    parts = [p.strip() for p in t.split() if p.strip()]

    kws: List[str] = []
    for p in parts:
        if p in _STOPWORDS:
            continue
        if len(p) <= 2:
            continue
        kws.append(p)

    # 去重保序
    uniq: List[str] = []
    seen = set()
    for k in kws:
        if k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    return uniq[:max_k]


def _extract_explicit_dates(q: str) -> List[str]:
    q = q or ""
    found: List[str] = []
    for pat in _DATE_PATTERNS:
        found += re.findall(pat, q)

    uniq: List[str] = []
    seen = set()
    for d in found:
        if d in seen:
            continue
        seen.add(d)
        uniq.append(d)
    return uniq


def detect_intent(question: str) -> Dict[str, Any]:
    """
    Return an intent dict (always dict):
      - intent: COUNT / TIME_DIFF / FIRST_AFTER / LAST_BEFORE / WHEN / NONE
      - route: alias of intent (compat)
      - keywords: list[str]
      - dates: list[str]
      - clause_a/clause_b: for TIME_DIFF between A and B (raw text)
      - keywords_a/keywords_b: tokenized keywords for each clause (optional but useful)
    """
    if not question or not isinstance(question, str):
        return {"intent": "NONE", "route": "NONE", "keywords": [], "dates": []}

    q = question.strip()
    ql = q.lower()

    base = {
        "intent": "NONE",
        "route": "NONE",
        "keywords": _tokenize_keywords(q),
        "dates": _extract_explicit_dates(q),
    }

    # TIME_DIFF: between A and B + (how many days / how long)
    if ("between" in ql and "and" in ql) and (
        ("how many" in ql and ("day" in ql or "days" in ql))
        or ("how long" in ql)
    ):
        # 用 q（原始字符串）提取子句，避免全小写导致后续显示/匹配怪异
        m = re.search(r"between\s+(.*?)\s+and\s+(.*?)(\?|$)", q, re.IGNORECASE | re.DOTALL)
        clause_a = m.group(1).strip() if m else ""
        clause_b = m.group(2).strip() if m else ""

        out = dict(base)
        out.update(
            {
                "intent": "TIME_DIFF",
                "route": "TIME_DIFF",
                "clause_a": clause_a,
                "clause_b": clause_b,
                "keywords_a": _tokenize_keywords(clause_a) if clause_a else [],
                "keywords_b": _tokenize_keywords(clause_b) if clause_b else [],
            }
        )
        return out

    # COUNT
    if "how many" in ql or "number of" in ql or "count" in ql:
        out = dict(base)
        out.update({"intent": "COUNT", "route": "COUNT"})
        return out

    # FIRST_AFTER / LAST_BEFORE
    if ("first" in ql and "after" in ql) or ("earliest" in ql and "after" in ql):
        out = dict(base)
        out.update({"intent": "FIRST_AFTER", "route": "FIRST_AFTER"})
        return out

    if ("last" in ql and "before" in ql) or ("latest" in ql and "before" in ql):
        out = dict(base)
        out.update({"intent": "LAST_BEFORE", "route": "LAST_BEFORE"})
        return out

    # WHEN
    if ql.startswith("when ") or "what date" in ql or "what day" in ql:
        out = dict(base)
        out.update({"intent": "WHEN", "route": "WHEN"})
        return out

    return base
