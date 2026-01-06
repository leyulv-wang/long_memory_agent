# temporal_reasoning/executor.py
# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional
import logging
import re
import datetime

logger = logging.getLogger(__name__)


def _get_session(driver):
    # driver 可能是 neo4j.Driver，也可能是 ltss_instance（有 .driver）
    if driver is None:
        return None
    if hasattr(driver, "session"):
        return driver.session()
    if hasattr(driver, "driver") and hasattr(driver.driver, "session"):
        return driver.driver.session()
    return None


def _norm_iso_date(s: str) -> str:
    """
    Normalize date-like string to YYYY-MM-DD.
    More robust handling for ISO timestamps (e.g., 2023-05-20T10:00:00).
    """
    if not s:
        return "unknown"

    s_str = str(s).strip()
    s_lower = s_str.lower()

    if s_lower in ("unknown", "none", "null", "", "n/a"):
        return "unknown"

    # 1) only truncate on real ISO timestamp, avoid harming TURNxxx
    if re.match(r"^\d{4}-\d{2}-\d{2}T", s_str):
        s_str = s_str.split("T", 1)[0]
    elif " " in s_str:
        parts = s_str.split(" ")
        if parts and ("-" in parts[0] or "/" in parts[0]):
            s_str = parts[0]

    # 2) YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s_str):
        return s_str

    # 3) MM/DD/YYYY or MM/DD/YY
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", s_str)
    if m:
        mm = int(m.group(1))
        dd = int(m.group(2))
        yy = m.group(3)
        if not yy:
            return "unknown"

        y = int(yy)
        if y < 100:
            y += 2000

        try:
            return datetime.date(y, mm, dd).isoformat()
        except Exception:
            return "unknown"

    return "unknown"


def _is_turn(s: Any) -> bool:
    try:
        return bool(re.fullmatch(r"TURN\d+", str(s).strip(), flags=re.IGNORECASE))
    except Exception:
        return False


def _normalize_event_time(value: Any) -> str:
    """
    Normalize event time:
    - Keep TURNxxx as TURNxxx (do not convert to unknown).
    - Normalize ISO-like time to YYYY-MM-DD.
    """
    if value is None:
        return "unknown"
    s = str(value).strip()
    if not s or s.lower() in ("unknown", "none", "null", "n/a"):
        return "unknown"
    if _is_turn(s):
        return s.upper()
    return _norm_iso_date(s)


def run_query(driver, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return raw rows as dicts."""
    rows: List[Dict[str, Any]] = []
    sess = _get_session(driver)
    if sess is None:
        return rows

    try:
        with sess as session:
            # 🔥 关键修复：自动补齐 agent_name 参数，避免 ParameterMissing 报错
            safe_params = dict(params or {})  # 复制避免修改原dict
            safe_params.setdefault("agent_name", "User")  # 兜底默认值
            safe_params.setdefault("limit_val", 10)  # 常用参数兜底

            res = session.run(cypher, safe_params)
            for rec in res:
                rows.append(rec.data())
    except Exception as e:
        logger.error(f"[temporal_reasoning] run_query failed: {e}", exc_info=True)
    return rows


def run_template(driver, cypher: str, params: Dict[str, Any]) -> List[str]:
    """
    Execute a Cypher template and return standardized evidence triple lines.
    Standard format (English, parseable):
    - A -REL- B [source=...; confidence=...; knowledgetype=...; evidenceunit=...; eventtime=...]
    """
    triples: List[str] = []
    rows = run_query(driver, cypher, params)

    for row in rows:
        if not isinstance(row, dict):
            continue

        src = row.get("src", "User")
        rel = row.get("rel", "RELATED_TO")
        tgt = row.get("tgt", "Unknown")

        # Prefer template-returned time fields; keep TURNxxx; normalize ISO to YYYY-MM-DD
        event_time_raw = row.get("time", row.get("event_time", row.get("eventtime", "unknown")))
        event_time = _normalize_event_time(event_time_raw)

        # IMPORTANT: don't fallback to row["source"] (can collide with other meanings like subject)
        sob = row.get("source_of_belief", row.get("sourceofbelief", "ground_truth")) or "ground_truth"

        conf = row.get("confidence", 1.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 1.0

        k_type = row.get("knowledge_type", row.get("knowledgetype", "observed_fact")) or "observed_fact"

        # Prefer evidence_source_unit (snake_case) then old keys
        evidence_unit = row.get(
            "evidence_source_unit",
            row.get("evidencesourceunit", row.get("evidenceunit", "unknown")),
        ) or "unknown"

        meta_str = (
            f"source={sob}; "
            f"confidence={conf:.2f}; "
            f"knowledgetype={k_type}; "
            f"evidenceunit={evidence_unit}; "
            f"eventtime={event_time}"
        )
        triples.append(f"- {src} -{rel}- {tgt} [{meta_str}]")

    return triples


def safe_run_query(driver, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Fail-safe query runner.
    - Never raises (returns []).
    - Always returns List[Dict[str, Any]].
    """
    sess = _get_session(driver)
    if sess is None:
        return []

    params = params or {}
    try:
        with sess as session:
            rs = session.run(cypher, params)
            out: List[Dict[str, Any]] = []
            for r in rs:
                try:
                    out.append(dict(r.data()))
                except Exception:
                    out.append({"_raw": str(r)})
            return out
    except Exception:
        return []


def safe_records_to_triples(records: List[Dict[str, Any]]) -> List[str]:
    """
    Convert records to evidence-like strings, robustly.
    Prefer keys: a/rel/b, source/confidence/event_time if present.
    Otherwise stringify.
    """
    triples: List[str] = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue

        a = rec.get("a") or rec.get("source") or rec.get("from") or rec.get("subject")
        rel = rec.get("rel") or rec.get("relation") or rec.get("type")
        b = rec.get("b") or rec.get("target") or rec.get("to") or rec.get("object")

        if a and rel and b:
            meta_parts = []

            # Prefer snake_case, but keep some compatibility keys.
            # Also normalize event_time and event_timestamp if present.
            for k in [
                "source_of_belief",
                "sourceofbelief",
                "confidence",
                "knowledge_type",
                "knowledgetype",
                "event_time",
                "eventtime",
                "event_timestamp",
                "eventtimestamp",
                "consolidated_at",
                "consolidatedat",
            ]:
                if k in rec and rec.get(k) is not None:
                    val = rec.get(k)
                    if k in ("event_time", "eventtime", "event_timestamp", "eventtimestamp"):
                        val = _normalize_event_time(val)
                    meta_parts.append(f"{k}={val}")

            meta = (" [" + "; ".join(meta_parts) + "]") if meta_parts else ""
            triples.append(f"- ({a}) -[{rel}]-> ({b}){meta}")
        else:
            triples.append(f"- {str(rec)}")

    return triples
