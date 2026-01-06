# utils/evidence_formatter.py

from typing import Dict, Any

def format_evidence_triple(
    subject: str,
    relation: str,
    obj: str,
    props: Dict[str, Any]
) -> str:
    """
    Standardized English-only evidence triple formatter.
    """

    source = props.get("source_of_belief") or props.get("source") or "unknown"
    confidence = props.get("confidence", "unknown")

    event_time = (
        props.get("event_timestamp")
        or props.get("event_time")
        or "unknown"
    )

    record_time = (
        props.get("consolidated_at")
        or props.get("record_time")
        or "unknown"
    )

    # 强制字符串化 + 安全
    def norm(x):
        if x is None:
            return "unknown"
        return str(x)

    return (
        f"- ({subject}) -[{relation}]-> ({obj}) "
        f"[source={norm(source)}; "
        f"confidence={norm(confidence)}; "
        f"event_time={norm(event_time)}; "
        f"record_time={norm(record_time)}]"
    )
