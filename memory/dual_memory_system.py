# -*- coding: utf-8 -*-

import json
import logging
from typing import List, Optional, Any, Dict
from datetime import datetime, timezone
import re
logger = logging.getLogger(__name__)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

MAX_EMBEDDING_INPUT_CHARS = 700

# ----------------------------
# Imports
# ----------------------------
from utils.embedding import get_embedding_model
from utils.llm import get_llm, invoke_json

from config import memory_consolidation_threshold, related_memories

from memory.stores import ShortTermEpisodicStore
from memory.structured_memory import KnowledgeGraphExtraction
from memory.ltss_writer import write_consolidation_result

# ✅ 新增：raw 通道一键入图
from memory.raw_graph_ingest import ingest_raw_dialogue_window
from memory.channels import RAW, CONSOLIDATED

# --- 时间片段抽取（用于让 LLM 更稳定解析 yesterday/today 等）---
_TIME_HINT_RE = re.compile(
    r"\b("
    r"yesterday|today|tomorrow|tonight|last\s+night|this\s+morning|this\s+afternoon|this\s+evening|"
    r"last\s+week|next\s+week|this\s+week|last\s+month|next\s+month|this\s+month|last\s+year|next\s+year|"
    r"on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"in\s+\d{4}|"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\b\d{1,2}:\d{2}\b|"
    r"\b\d+\s+(days?|weeks?|months?|years?)\s+(ago|later)\b"
    r")\b",
    re.IGNORECASE,
)

def extract_time_snippets(memories_str: str, *, max_lines: int = 20, max_chars: int = 2500) -> str:
    """
    从窗口文本里抽取含时间线索的行（yesterday/today/日期/星期/xx days ago...）
    只取最相关的片段，避免 prompt 爆长。
    """
    if not isinstance(memories_str, str) or not memories_str.strip():
        return ""

    lines = []
    for ln in memories_str.splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        if _TIME_HINT_RE.search(s):
            lines.append(s)

    # 去重并截断
    uniq = []
    seen = set()
    for s in lines:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)

    uniq = uniq[:max_lines]
    out = "\n".join(uniq)
    if len(out) > max_chars:
        out = out[:max_chars] + " ..."
    return out

# ----------------------------
# JSON parse helpers（保留你的鲁棒解析）
# ----------------------------
def bind_llm_low_temp(llm):
    try:
        return llm.bind(temperature=0.0)
    except Exception:
        pass
    try:
        return llm.bind_temperature(0.0)
    except Exception:
        pass
    return llm


def parse_json_dict(raw: str) -> Dict[str, Any]:
    s = (raw or "").strip()
    # 直接 JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # repair_json（如果装了）
    try:
        from json_repair import repair_json

        obj = repair_json(s, return_objects=True)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 截取最外层 {}
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s2 = s[start : end + 1]
        obj2 = json.loads(s2)
        if isinstance(obj2, dict):
            return obj2

    raise ValueError("Failed to parse JSON dict from LLM output.")


# ----------------------------
# Prompt（现在走：读 raw triples → 输出 consolidated KG）
# ----------------------------
def build_consolidation_from_raw_prompt(*, current_time: Optional[str], raw_triples_str: str, time_snippets: str) -> str:
    time_context = current_time or "TURN_UNKNOWN"

    # recorded_turn_id 只从 TURN_x 里解析数字，不可解析就写 -1
    recorded_turn_id = -1
    try:
        if isinstance(time_context, str) and time_context.upper().startswith("TURN"):
            import re as _re
            m = _re.search(r"(\d+)", time_context)
            if m:
                recorded_turn_id = int(m.group(1))
    except Exception:
        recorded_turn_id = -1

    return f"""
You are consolidating a cognitive agent's memory based on:
1) RAW knowledge graph triples (may miss some time phrases)
2) TIME-SNIPPETS from original dialogue (to recover relative time like yesterday/today)

You MUST output a STRICT JSON object with keys:
- facts: array of HARD FACT strings (numbers, dates, durations, prices, proper names; keep exact wording)
- insights: array of LOGICAL INSIGHT strings (social relations, patterns, causal links, contradiction resolution)
- nodes: array of nodes: {{name: str, label: str, properties: object}}
- relationships: array of relationships:
  {{source_node_name: str, source_node_label: str, target_node_name: str, target_node_label: str, type: str, properties: object}}

CRITICAL TIME RULES (VERY IMPORTANT):
A) The reference "recorded time" is the current virtual turn: {time_context}
   You MUST set on EVERY Event node:
     properties.recorded_turn_id = {recorded_turn_id}
B) If you create an Event node AND the text implies an event time using relative words
   (today/yesterday/tomorrow/last week/next week/this morning/etc.),
   you MUST set on that Event node:
     - properties.event_time_text: string (e.g., "yesterday", "today", "last week")
     - properties.event_turn_offset: integer offset relative to recorded_turn_id.
       Examples:
         "today" => 0
         "yesterday" => -1
         "tomorrow" => +1
         "last week" => -7   (approx ok)
         "next week" => +7   (approx ok)
C) If there is NO relative time info, OMIT event_time_text and event_turn_offset (do NOT guess).

GENERAL RULES:
1) Output ONLY valid JSON. No markdown.
2) DO NOT hallucinate. Only use information supported by RAW triples or TIME-SNIPPETS.
3) Node labels must be one of: Person, Organization, Location, Object, Event, Date, Value, Concept.
4) Relationship type MUST be UPPERCASE.
5) Keep the consolidated output SMALL: prefer <= 12 relationships, <= 15 nodes.
6) Relative time normalization:
   - You MUST set event_turn_offset as an integer relative to recorded_turn_id.
   - If event_time_text is "today" -> offset 0
   - "yesterday" -> offset -1
   - "tomorrow" -> offset +1
   - "last night" -> offset -1 (default)
   - "this morning" -> offset 0
   - "this afternoon" -> offset 0
   - "this evening" -> offset 0
   - "tonight" -> offset 0
   Example:
     recorded_turn_id=10, event_time_text="yesterday" => event_turn_offset=-1
     recorded_turn_id=10, event_time_text="today" => event_turn_offset=0

TIME-SNIPPETS (original dialogue lines that contain time cues):
{time_snippets if time_snippets else "(none)"}

RAW triples (evidence):
{raw_triples_str}
""".strip()


# ----------------------------
# Neo4j: 拉取本次 raw 导入的子图 triples（按 doc_id 锚定）
# ----------------------------
def fetch_raw_triples_by_doc_id(ltss, doc_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    """
    只取和本次 doc_id 相关的边，过滤掉 Chunk/TextUnit/FROM_SOURCE 等噪声。
    """
    cypher = """
    MATCH (n)
    WHERE n.doc_id = $doc_id OR n.document_id = $doc_id
    WITH collect(id(n)) AS ids
    MATCH (a)-[r]->(b)
    WHERE (id(a) IN ids OR id(b) IN ids)
      AND type(r) <> 'FROM_SOURCE'
    WITH a, r, b,
         labels(a) AS la, labels(b) AS lb
    WHERE NOT 'Chunk' IN la
      AND NOT '__Node__' IN la
      AND NOT 'TextUnit' IN la
      AND NOT 'Chunk' IN lb
      AND NOT '__Node__' IN lb
      AND NOT 'TextUnit' IN lb
    RETURN
      la AS a_labels,
      a.name AS a_name,
      type(r) AS rel_type,
      r.confidence AS confidence,
      r.source_of_belief AS source_of_belief,
      r.event_timestamp AS event_timestamp,
      lb AS b_labels,
      b.name AS b_name
    LIMIT $limit
    """
    rows = ltss.query_graph(cypher, {"doc_id": doc_id, "limit": int(limit)})
    return rows or []


def format_raw_triples(rows: List[Dict[str, Any]]) -> str:
    """
    把 Neo4j 子图转成紧凑文本，喂给 LLM 做轻量巩固。
    """
    lines = []
    for row in rows:
        a = (row.get("a_name") or "").strip()
        b = (row.get("b_name") or "").strip()
        rel = (row.get("rel_type") or "").strip()
        if not a or not b or not rel:
            continue

        # label 取第一个非空
        la = row.get("a_labels") or []
        lb = row.get("b_labels") or []
        a_label = la[0] if isinstance(la, list) and la else "Concept"
        b_label = lb[0] if isinstance(lb, list) and lb else "Concept"

        conf = row.get("confidence", None)
        try:
            conf_s = f"{float(conf):.2f}" if conf is not None else "1.00"
        except Exception:
            conf_s = "1.00"

        src = row.get("source_of_belief") or ""
        ts = row.get("event_timestamp") or ""

        meta = []
        if conf_s:
            meta.append(f"conf={conf_s}")
        if src:
            meta.append(f"src={src}")
        if ts and str(ts).lower() != "unknown":
            meta.append(f"time={ts}")

        meta_str = (" | " + ",".join(meta)) if meta else ""
        lines.append(f"({a_label}:{a}) -[{rel}]-> ({b_label}:{b}){meta_str}")

    if not lines:
        return "(No usable RAW triples found.)"
    return "\n".join(lines)

def _normalize_event_turn_offset(structured_response, recorded_turn_id: int):
    """
    规则兜底：修正 Event 节点 properties 里的 event_turn_offset
    - 优先处理最常见的相对时间词，保证 today/yesterday/tomorrow 不会错
    - 只在 Event 节点上生效
    """
    if not structured_response or not hasattr(structured_response, "nodes"):
        return

    # 常见相对时间词（LongMemoryEval/LoCoMo 常见）
    rel_map = {
        "today": 0,
        "yesterday": -1,
        "tomorrow": 1,
        "last night": -1,
        "tonight": 0,
        "this morning": 0,
        "this afternoon": 0,
        "this evening": 0,
        "yesterday morning": -1,
        "yesterday afternoon": -1,
        "yesterday evening": -1,
        "two days ago": -2,
        "3 days ago": -3,
        "three days ago": -3,
        "a week ago": -7,
        "last week": -7,   # 粗略
        "next week": 7,    # 粗略
    }

    # 额外：解析 “N days ago”
    import re

    for n in structured_response.nodes:
        try:
            if (getattr(n, "label", "") or "").lower() != "event":
                continue
            props = getattr(n, "properties", None) or {}
            # recorded_turn_id：如果没给就补上
            props.setdefault("recorded_turn_id", int(recorded_turn_id))

            t = (props.get("event_time_text") or "").strip().lower()
            if not t:
                n.properties = props
                continue

            # 1) 直接命中词典
            if t in rel_map:
                props["event_turn_offset"] = int(rel_map[t])
                n.properties = props
                continue

            # 2) N days ago
            m = re.match(r"^\s*(\d+)\s+days?\s+ago\s*$", t)
            if m:
                props["event_turn_offset"] = -int(m.group(1))
                n.properties = props
                continue

            # 3) in N days
            m2 = re.match(r"^\s*in\s+(\d+)\s+days?\s*$", t)
            if m2:
                props["event_turn_offset"] = int(m2.group(1))
                n.properties = props
                continue

            # 4) fallback：如果 LLM 没给 offset，就设为 0（保守）
            if "event_turn_offset" not in props or props["event_turn_offset"] is None:
                props["event_turn_offset"] = 0

            n.properties = props
        except Exception:
            # 任何异常都不要让巩固链路崩
            continue

class DualMemorySystem:
    """
    现在的 DualMemorySystem 负责调度：
    - STES（短期）：add/search/most_recent
    - consolidation（巩固）：先 raw 入图，再读 raw 子图做 consolidated
    """

    def __init__(
        self,
        agent_name: str,
        ltss_instance,
        initial_memories: Optional[List[str]] = None,
        consolidation_llm=None,
    ):
        self.agent_name = agent_name
        self.ltss = ltss_instance
        self.consolidation_llm = consolidation_llm or get_llm()

        # STES
        self.stes = ShortTermEpisodicStore(agent_name=agent_name)

        # Embedding（失败也不崩）
        try:
            self.embedding_model = get_embedding_model()
        except Exception as e:
            logger.error(f"Embedding model init failed (will degrade to None): {e}", exc_info=True)
            self.embedding_model = None

        # initial_memories 不写进 STES（保持你原逻辑）
        self._stes_save_if_possible()

    def close(self):
        self._stes_save_if_possible()

    def _stes_save_if_possible(self):
        try:
            auto_save_flag = getattr(self.stes, "auto_save", None)
            if auto_save_flag is None:
                auto_save_flag = getattr(self.stes, "autosave", True)
            if auto_save_flag is False:
                return
            if hasattr(self.stes, "save"):
                self.stes.save()
        except Exception as e:
            logger.error(f"STES for {self.agent_name}: save failed: {e}", exc_info=True)

    def add_episodic_memory(self, observation: str):
        if not isinstance(observation, str) or not observation.strip():
            return
        try:
            self.stes.add([observation.strip()])
        except Exception as e:
            logger.error(f"STES for {self.agent_name}: add episodic memory failed: {e}", exc_info=True)

    def retrieve_episodic_memories(self, query: str, k: int) -> List[str]:
        if not isinstance(query, str) or not query.strip():
            return []
        try:
            kk = int(k or 0)
        except Exception:
            kk = 0
        if kk <= 0:
            return []
        try:
            return self.stes.search(query, k=kk)
        except Exception as e:
            logger.error(f"STES for {self.agent_name}: search failed: {e}", exc_info=True)
        return []

    def _get_most_recent_k(self, k: int) -> List[str]:
        try:
            kk = int(k or 0)
        except Exception:
            kk = 0
        if kk <= 0:
            return []
        try:
            if hasattr(self.stes, "get_most_recent_k"):
                return self.stes.get_most_recent_k(kk)  # type: ignore
            return self.stes.get_most_recent(kk)
        except Exception as e:
            logger.error(f"STES for {self.agent_name}: get most recent failed: {e}", exc_info=True)
        return []

    def trigger_consolidation(self, current_time: Optional[str] = None, k: Optional[int] = None):
        agent_prefix = f"[智能体 '{self.agent_name}' 的记忆巩固引擎]"

        # 1) 抓取短期窗口
        try:
            num_to_fetch = int(k) if k is not None else int(memory_consolidation_threshold)
        except Exception:
            num_to_fetch = int(memory_consolidation_threshold)

        if num_to_fetch <= 0:
            num_to_fetch = 1

        logger.info(f"{agent_prefix} - 1 开始巩固，抓取最近 {num_to_fetch} 条 STES 记忆。")
        most_recent = self._get_most_recent_k(num_to_fetch)
        if not most_recent:
            logger.warning(f"{agent_prefix} STES 没有可巩固的记忆，跳过。")
            return

        # 2) 召回更多相关短期（可选）
        try:
            rel_k = int(related_memories or 0)
        except Exception:
            rel_k = 0
        if rel_k < 0:
            rel_k = 0

        # 用 most_recent 拼一个粗 query（避免额外 LLM call）
        rough_query = " ".join(most_recent[:3])
        rough_query = rough_query[:MAX_EMBEDDING_INPUT_CHARS]
        relevant = self.retrieve_episodic_memories(rough_query, k=rel_k)

        combined, seen = [], set()
        for m in (most_recent + (relevant or [])):
            if not isinstance(m, str):
                continue
            mm = m.strip()
            if not mm or mm in seen:
                continue
            seen.add(mm)
            combined.append(mm)

        memories_str = "\n".join(f"- {m}" for m in combined)

        # 3) ✅ raw 通道：一键入图（成熟工具）
        logger.info(f"{agent_prefix} - 2 raw 通道：开始一键结构化入图 (LlamaIndex -> Neo4j)")
        try:
            raw_result = ingest_raw_dialogue_window(
                agent_name=self.agent_name,
                text=memories_str,
                virtual_time=current_time or "TURN_UNKNOWN",
                metadata={"source_type": "dialogue"},
                channel=RAW,
            )
            doc_id = raw_result["doc_id"]
            logger.info(f"{agent_prefix} raw 入图完成 doc_id={doc_id}")
        except Exception as e:
            logger.error(f"{agent_prefix} raw 入图失败: {e}", exc_info=True)
            return

        # 4) 从 Neo4j 拉回本次 raw 子图 triples
        logger.info(f"{agent_prefix} - 3 从 Neo4j 读取 raw 子图 triples，准备轻量巩固")
        try:
            rows = fetch_raw_triples_by_doc_id(self.ltss, doc_id=doc_id, limit=200)
            raw_triples_str = format_raw_triples(rows)
        except Exception as e:
            logger.error(f"{agent_prefix} 读取 raw 子图失败: {e}", exc_info=True)
            return

        # 5) ✅ consolidated：让 LLM 只读 triples 输出核心事实（更快、更稳）
        llm_low = bind_llm_low_temp(self.consolidation_llm)
        time_snippets = extract_time_snippets(memories_str, max_lines=20, max_chars=2500)

        prompt = build_consolidation_from_raw_prompt(
            current_time=current_time or "TURN_UNKNOWN",
            raw_triples_str=raw_triples_str,
            time_snippets=time_snippets,
        )

        logger.info(f"{agent_prefix} - 4 开始轻量巩固（读取 raw triples）")
        if time_snippets:
            logger.info(f"{agent_prefix} - 时间片段抽取命中 {len(time_snippets.splitlines())} 行，用于提升相对时间解析")

        try:
            kg_raw = invoke_json(llm_low, prompt).strip()
            obj = parse_json_dict(kg_raw)
            # ===== DEBUG：打印 consolidated JSON（只打印 Event 节点，避免太长）=====
            try:
                nodes = obj.get("nodes", []) if isinstance(obj, dict) else []
                event_nodes = []
                for n in nodes:
                    if isinstance(n, dict) and str(n.get("label", "")).lower() == "event":
                        event_nodes.append(n)
                logger.info(
                    f"{agent_prefix} [DEBUG] consolidated Event nodes = {json.dumps(event_nodes, ensure_ascii=False)[:2000]}")
            except Exception as e:
                logger.info(f"{agent_prefix} [DEBUG] 打印 Event nodes 失败: {e}")

            structured_response = KnowledgeGraphExtraction(**obj)
            # ✅ 规则兜底：修正 Event offset（确保 today/yesterday 等永远正确）
            try:
                # 1. 先从 current_time (例如 "TURN_12") 中解析出整数 ID
                current_turn_int = 0
                if current_time and isinstance(current_time, str):
                    import re
                    # 尝试匹配 "TURN_12" 或纯数字
                    m = re.search(r"(\d+)", current_time)
                    if m:
                        current_turn_int = int(m.group(1))

                # 2. 传入解析出来的 current_turn_int
                _normalize_event_turn_offset(
                    structured_response,
                    recorded_turn_id=current_turn_int
                )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"{agent_prefix} consolidated LLM 输出解析失败: {e}", exc_info=True)
            return

        # 6) 写入 consolidated 通道（你改过的 ltss_writer 会打通道标记）
        logger.info(f"{agent_prefix} - 5 写入 consolidated 通道 LTSS")
        try:
            write_consolidation_result(
                ltss=self.ltss,
                embedding_model=self.embedding_model,
                agent_name=self.agent_name,
                memories_str=memories_str,
                structured_response=structured_response,
                current_time=current_time or "TURN_UNKNOWN",
                channel=CONSOLIDATED,
                doc_id=doc_id,
            )
        except Exception as e:
            logger.error(f"{agent_prefix} consolidated 写入失败: {e}", exc_info=True)
            return

        self._stes_save_if_possible()
        logger.info(f"{agent_prefix} ✅ 本次巩固结束：raw + consolidated 双通道已写入")
