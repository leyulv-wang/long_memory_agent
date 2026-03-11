# -*- coding: utf-8 -*-
"""
agent/retrievers.py

检索器入口模块（已简化）

当前架构：
- 统一使用 SimpleRetriever（simple_retriever.py）
- 包含综合评分、动态版本检测、多跳扩展等功能

存储模式：全量保留（Append-Only）
- 每次提及都创建独立记录（belief_key 包含 turn_id）
- 所有历史版本都保留，不会被覆盖
- 通过动态版本检测识别同一主题的不同版本
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from config import long_memory_number
from .simple_retriever import SimpleRetriever

logger = logging.getLogger(__name__)

_DEBUG_GRAPHRAG = os.getenv("DEBUG_GRAPHRAG", "0") == "1"

# 匹配 TURN_123, Step 50, Turn: 10 等格式
_TURN_RE = re.compile(r"(?:TURN|Turn|STEP|Step)[^0-9]*(\d+)")
_TURN_RE_2 = re.compile(r"^TURN_(\d+)$", re.IGNORECASE)


class GraphRAGRetriever:
    """
    长期记忆检索器入口
    
    内部使用 SimpleRetriever 执行实际检索，包含：
    - 综合评分机制
    - 动态版本检测
    - 多跳扩展
    - 原文兜底
    """

    def __init__(self, ltss_instance, llm=None, agent_name: Optional[str] = None):
        if not ltss_instance:
            raise ValueError("LTSS 实例未就绪")

        if not hasattr(ltss_instance, "query_graph"):
            raise ValueError("LTSS 缺少 query_graph()，GraphRAGRetriever 无法工作")

        self.ltss = ltss_instance
        self.driver = getattr(ltss_instance, "driver", None)
        self.llm = llm
        self.agent_name = agent_name or "unknown"

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

    def search(self, observation: str, current_time: str = None) -> str:
        """
        对外主入口：返回格式化的上下文字符串
        
        使用 SimpleRetriever 执行检索，包含：
        - 综合评分机制
        - 动态版本检测
        - 双维度时间衰减
        - 多跳扩展
        - 原文兜底
        """
        obs = (observation or "").strip()
        if not obs:
            return "无相关长期知识。"

        cur_turn = self._parse_turn_step(current_time) or 0

        try:
            retriever = SimpleRetriever(self.ltss, self.agent_name)
            
            # 根据问题类型调整检索参数
            ql = obs.lower()
            is_order_question = any(k in ql for k in ["order", "earliest", "latest", "first", "last", "before", "after"])
            
            if is_order_question:
                # 排序问题需要更多召回
                simple_fact_k = max(long_memory_number, 100)
                textunit_k = 15
            else:
                # ✅ 修复：从 80 提高到 100，避免重要信息被挤出 top-k
                simple_fact_k = max(long_memory_number, 100)
                textunit_k = 10
            
            result = retriever.search(
                obs,
                simple_fact_k=simple_fact_k,
                textunit_k=textunit_k,
                enable_multi_hop=True,
                enable_version_detection=True,
                current_turn=cur_turn,
            )
            
            if _DEBUG_GRAPHRAG or os.getenv("DEBUG_PIPELINE", "0") == "1":
                lines = [ln for ln in result.splitlines() if ln.strip()]
                logger.info(f"[GraphRAGRetriever] output_lines={len(lines)}")
            
            return result
            
        except Exception as e:
            logger.error(f"[GraphRAGRetriever] search failed: {e}", exc_info=True)
            return "无相关长期知识。"
