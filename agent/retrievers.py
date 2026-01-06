# -*- coding: utf-8 -*-
"""
agent/retrievers.py

当前主干目标：
- 长期记忆检索优先走 V2（Fact/Event 链 + 软更新 + consolidated 优先）
- 保留旧逻辑作为 fallback（不删，便于对比/回滚）
- 输出统一为 ContextBuilder 可解析的 triple block：
  - (A) -[REL]-> (B) [k=v; ...]

依赖：
- ltss_instance：你的长期记忆系统实例（需要 query_graph；并且通常也有 driver）
- V2 检索：agent/lt_retriever_v2.py
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional

import os
from config import long_memory_number
from .lt_retriever_v2 import retrieve_long_term_facts_v2

logger = logging.getLogger(__name__)

def _safe_meta_value(val: Any, max_len: int = 180) -> str:
    """把证据片段安全地塞进 [k=v; ...] meta，避免 ';' ']' 破坏解析。"""
    if val is None:
        return ""
    x = str(val)
    x = " ".join(x.split())
    x = x.replace(";", ",").replace("]", ")")
    if len(x) > max_len:
        x = x[: max_len - 3] + "..."
    return x


# 匹配 TURN_123, Step 50, Turn: 10 等格式
_TURN_RE = re.compile(r"(?:TURN|Turn|STEP|Step)[^0-9]*(\d+)")
_TURN_RE_2 = re.compile(r"^TURN_(\d+)$", re.IGNORECASE)


class GraphRAGRetriever:
    """
    长期记忆检索器：
    - search() 默认优先 V2：TextUnit -> Event -> Fact -> (SUBJECT/OBJECT)
    - V2 的“软更新”发生在检索排序阶段，不会覆盖/删除图中旧信息
    """

    def __init__(self, ltss_instance, llm=None, agent_name: Optional[str] = None):
        if not ltss_instance:
            raise ValueError("LTSS 实例未就绪")

        # LTSS 至少需要 query_graph；driver 用于兼容旧代码
        if not hasattr(ltss_instance, "query_graph"):
            raise ValueError("LTSS 缺少 query_graph()，GraphRAGRetriever 无法工作")

        self.ltss = ltss_instance
        self.driver = getattr(ltss_instance, "driver", None)  # optional
        self.llm = llm
        self.agent_name = agent_name or "unknown"

    # -------------------------
    # Turn/Step 解析（用于衰减）
    # -------------------------
    def _parse_turn_step(self, current_time: Optional[str]) -> Optional[int]:
        """
        解析 current_time 中的轮次/步数：
        - "TURN_12" / "Turn 12" / "Step: 12" -> 12
        """
        if not current_time:
            return None

        s = str(current_time).strip()
        m2 = _TURN_RE_2.match(s)
        if m2:
            try:
                return int(m2.group(1))
            except Exception:
                return None

        m = _TURN_RE.search(s)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    # -------------------------
    # V2 输出格式化（triple block）
    # -------------------------
    def _format_v2_groups_to_triple_block(self, groups: List[Dict[str, Any]]) -> str:
        """
        把 V2 的 groups（slot_id -> best/alternatives）格式化为 triple block。
        """
        if not groups:
            return "无相关长期知识。"

        lines: List[str] = []
        for g in groups:
            best = (g or {}).get("best") or {}
            subj = (best.get("subject") or "").strip()
            obj = (best.get("object") or "").strip()
            rel = (best.get("rel_type") or "RELATED_TO").strip()

            if not subj or not obj:
                # 兜底：避免空
                bk = (best.get("belief_key") or "").strip()
                subj = subj or "Unknown"
                obj = obj or (bk if bk else "Unknown")

            meta_parts = []
            meta_parts.append(f"channel={best.get('channel')}")
            meta_parts.append(f"confidence={best.get('confidence')}")
            meta_parts.append(f"turn_id={best.get('turn_id')}")
            meta_parts.append(f"score={best.get('score')}")
            if best.get("event_id"):
                meta_parts.append(f"event_id={best.get('event_id')}")
            if best.get("evidence_textunit"):
                meta_parts.append(f"evidence={best.get('evidence_textunit')}")
            evc = _safe_meta_value(best.get('evidence_content') or '')
            if evc:
                meta_parts.append(f"evidence_snip={evc}")
            if g.get("slot_id"):
                meta_parts.append(f"slot_id={g.get('slot_id')}")

            meta = "; ".join([p for p in meta_parts if p and "None" not in str(p)])
            lines.append(f"- ({subj}) -[{rel}]-> ({obj}) [{meta}]")

            # 把 alternatives 作为冲突/备选证据（默认最多 2 条，避免上下文膨胀）
            alts = (g or {}).get("alternatives") or []
            for a in alts[:2]:
                asub = (a.get("subject") or subj or "Unknown").strip()
                aobj = (a.get("object") or a.get("belief_key") or obj).strip()
                arel = (a.get("rel_type") or rel).strip()

                ameta_parts = [
                    f"channel={a.get('channel')}",
                    f"confidence={a.get('confidence')}",
                    f"turn_id={a.get('turn_id')}",
                    f"score={a.get('score')}",
                ]
                if a.get("event_id"):
                    ameta_parts.append(f"event_id={a.get('event_id')}")
                if a.get("evidence_textunit"):
                    ameta_parts.append(f"evidence={a.get('evidence_textunit')}")
                aevc = _safe_meta_value(a.get('evidence_content') or '')
                if aevc:
                    ameta_parts.append(f"evidence_snip={aevc}")
                ameta_parts.append("alt=true")
                ameta = "; ".join([p for p in ameta_parts if p and "None" not in str(p)])
                lines.append(f"- ({asub}) -[{arel}]-> ({aobj}) [{ameta}]")

        return "\n".join(lines) if lines else "无相关长期知识。"

    # -------------------------
    # Fallback（保留旧接口/能力）
    # -------------------------
    def _search_fallback_old(self, observation: str, current_time: str = None) -> str:
        """
        旧逻辑保底：
        - 如果 ltss_instance 有 retrieve_textunits()，优先取 TextUnit（更接近原文证据）
        - 如果 ltss_instance 有 retrieve_knowledge()，就用它拿一些“证据 TextUnit”
        - 输出仍然是 triple 风格，但更弱（仅 evidence 行）
        """
        if not self.ltss:
            return "无相关长期知识。"

        max_k = max(8, long_memory_number)

        # 1) TextUnit fallback（优先）
        if hasattr(self.ltss, "retrieve_textunits"):
            try:
                hits = self.ltss.retrieve_textunits(
                    observation,
                    top_k=max_k,
                    agent_name=self.agent_name,
                )
                lines = self._format_textunit_hits(hits, max_k=max_k, tag="textunit_fallback")
                if lines:
                    return lines
            except Exception as e:
                logger.error(f"[fallback] retrieve_textunits 失败: {e}", exc_info=True)

        # 2) 旧的实体索引 fallback
        if not hasattr(self.ltss, "retrieve_knowledge"):
            return "无相关长期知识。"

        try:
            hits = self.ltss.retrieve_knowledge(observation, top_k=max_k)
        except Exception as e:
            logger.error(f"[fallback] retrieve_knowledge 失败: {e}", exc_info=True)
            return "无相关长期知识。"

        lines = self._format_textunit_hits(hits, max_k=max_k, tag="entity_fallback")

        return "\n".join(lines) if lines else "无相关长期知识。"

    def _format_textunit_hits(self, hits: List[Dict[str, Any]], *, max_k: int, tag: str) -> str:
        if not hits:
            return ""

        lines: List[str] = []
        max_chars = 220
        for h in hits[: max_k]:
            name = h.get("name") or h.get("id") or "unknown_unit"
            content = (h.get("content") or h.get("text") or "").strip()
            if not content:
                continue
            snippet = " ".join(content.split())
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 3] + "..."
            score = h.get("score", None)
            meta = [f"fallback={tag}"]
            if h.get("channel"):
                meta.append(f"channel={h.get('channel')}")
            if h.get("turn_id") is not None:
                meta.append(f"turn_id={h.get('turn_id')}")
            if score is not None:
                meta.append(f"score={score}")
            lines.append(f"- (TextUnit:{name}) -[EVIDENCE]-> ({snippet}) [{'; '.join(meta)}]")

        return "\n".join(lines)

    # -------------------------
    # Public API
    # -------------------------
    def search(self, observation: str, current_time: str = None) -> str:
        """
        对外主入口：返回 triple block 字符串。
        """
        obs = (observation or "").strip()
        if not obs:
            return "无相关长期知识。"

        cur_turn = self._parse_turn_step(current_time)
        if cur_turn is None:
            cur_turn = 0

        # 1) V2 优先
        try:
            groups = retrieve_long_term_facts_v2(
                ltss=self.ltss,
                query=obs,
                current_turn=int(cur_turn),
                k=long_memory_number,
                seed_textunit_top_m=12,
                channels=["consolidated", "raw"],
                max_alternatives_per_slot=3,
                agent_name=self.agent_name,

            )
            if groups:
                logger.info(f"[GraphRAG-V2] 命中事实组数: {len(groups)} | current_turn={cur_turn}")
                block = self._format_v2_groups_to_triple_block(groups)
                if os.getenv("DEBUG_GRAPHRAG", "0") == "1":
                    lines = [ln for ln in (block.splitlines() if isinstance(block, str) else []) if ln.strip()]
                    logger.info(f"[GraphRAG-V2] triple_lines={len(lines)} sample={lines[:3]}")
                return block

            logger.info("[GraphRAG-V2] 未命中事实，转 fallback。")
        except Exception as e:
            logger.error(f"[GraphRAG-V2] 检索失败，转 fallback：{e}", exc_info=True)

        # 2) fallback
        return self._search_fallback_old(observation=obs, current_time=current_time)
