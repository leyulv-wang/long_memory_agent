import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# 统一的正则：支持标准格式，宽容空格，支持末尾的可选 [meta]
# 核心模式：- (A) -[REL]-> (B) [meta...]
UNIFIED_TRIPLE_RE = re.compile(
    r"^\s*-\s*\(\s*(?P<a>.+?)\s*\)\s*-\[\s*(?P<rel>.+?)\s*\]->\s*\(\s*(?P<b>.+?)\s*\)\s*(?P<meta>\[.*\])?",
    re.DOTALL
)


@dataclass
class EvidenceTriple:
    head: str
    relation: str
    tail: str
    meta: Dict[str, Any]
    raw_text: str

    @property
    def a(self): return self.head

    @property
    def b(self): return self.tail

    @property
    def a_l(self): return self.head.lower()

    @property
    def b_l(self): return self.tail.lower()

    @property
    def rel_u(self): return self.relation.upper()


def parse_triple(line: str) -> Optional[EvidenceTriple]:
    """
    统一解析入口。失败返回 None。
    """
    if not line or not isinstance(line, str):
        return None

    line = line.strip()
    m = UNIFIED_TRIPLE_RE.match(line)

    if not m:
        # 尝试容错：有时候可能是 - A -REL-> B (没有括号)
        # 这里可以根据需要添加备用正则，暂时保持严格以规范输出
        return None

    head = m.group("a").strip()
    rel = m.group("rel").strip()
    tail = m.group("b").strip()
    meta_str = (m.group("meta") or "").strip()

    meta_dict = {}
    if meta_str.startswith("[") and meta_str.endswith("]"):
        content = meta_str[1:-1]
        # 解析 key=value; key2=value2
        pairs = content.split(";")
        for p in pairs:
            if "=" in p:
                k, v = p.split("=", 1)
                meta_dict[k.strip().lower()] = v.strip()
            else:
                # 处理没有值的标记
                if p.strip():
                    meta_dict[p.strip().lower()] = True

    return EvidenceTriple(
        head=head,
        relation=rel,
        tail=tail,
        meta=meta_dict,
        raw_text=line
    )


def format_triple(head: str, rel: str, tail: str, meta: Dict[str, Any] = None) -> str:
    """
    统一生成入口。
    """
    base = f"- ({head}) -[{rel}]-> ({tail})"
    if not meta:
        return base

    # 规范化 meta 字符串
    meta_parts = []
    # 优先顺序
    priority_keys = ["source", "event_time", "confidence", "knowledge_type"]

    for k in priority_keys:
        if k in meta and meta[k] is not None:
            meta_parts.append(f"{k}={meta[k]}")

    # 添加剩余的 keys
    if meta:
        for k, v in meta.items():
            if k not in priority_keys and v is not None:
                meta_parts.append(f"{k}={v}")

    return f"{base} [{'; '.join(meta_parts)}]"


def safe_records_to_triples(records: List[Dict[str, Any]]) -> List[str]:
    """
    Convert records (from Neo4j or dicts) to evidence-like strings, robustly.
    Prefer keys: a/rel/b, source/confidence/event_time if present.
    """
    triples = []
    for rec in records or []:
        if not isinstance(rec, dict):
            continue

        a = rec.get("a") or rec.get("source") or rec.get("from") or rec.get("subject")
        rel = rec.get("rel") or rec.get("relation") or rec.get("type")
        b = rec.get("b") or rec.get("target") or rec.get("to") or rec.get("object")

        if a and rel and b:
            # 收集 meta
            meta = {}
            # 自动迁移常见的 meta 字段
            for k in ["source", "confidence", "event_time", "event_timestamp", "consolidated_at", "knowledge_type",
                      "evidence_unit"]:
                if k in rec and rec.get(k) is not None:
                    # 标准化 key 名称 (比如 event_timestamp -> event_time)
                    out_k = "event_time" if k == "event_timestamp" else k
                    meta[out_k] = rec.get(k)

            triples.append(format_triple(a, rel, b, meta))
        else:
            # 兜底
            triples.append(f"- {str(rec)}")
    return triples
