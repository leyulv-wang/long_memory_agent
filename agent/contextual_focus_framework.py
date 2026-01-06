# agent/contextual_focus_framework.py
import logging
import os
from typing import Dict, Any, Optional

from memory.dual_memory_system import DualMemorySystem
from utils.llm import get_llm
from config import short_memory_number

from .retrievers import GraphRAGRetriever
from .context_builder import ContextBuilder, canonicalize_triple_block

logger = logging.getLogger(__name__)
_DEBUG_PIPELINE = os.getenv("DEBUG_PIPELINE", "0") == "1"


class ContextualFocusFramework:
    """
    CFF（Contextual Focus Framework）调度器：
    - 负责把 episodic / long-term / temporal / derived 组织成最终 prompt context
    - Graph 检索逻辑：GraphRAGRetriever（agent/retrievers.py）
    - 上下文拼装：ContextBuilder（agent/context_builder.py）
    """

    def __init__(self, memory_system: DualMemorySystem):
        self.memory_system = memory_system
        self.llm = get_llm()

        self.graphrag_retriever: Optional[GraphRAGRetriever] = None
        try:
            ltss = getattr(self.memory_system, "ltss", None)
            if ltss and getattr(ltss, "driver", None):
                agent_name = (
                    getattr(self.memory_system, "agent_name", None)
                    or getattr(self.memory_system, "agentname", None)
                    or "User"
                )
                self.graphrag_retriever = GraphRAGRetriever(
                    ltss_instance=ltss,
                    llm=self.llm,
                    agent_name=agent_name,
                )
            else:
                logger.warning("[CFF] LTSS 不可用（或 driver 为空），GraphRAGRetriever 未初始化。")
        except Exception as e:
            logger.error(f"[CFF] GraphRAGRetriever 初始化失败: {e}", exc_info=True)
            self.graphrag_retriever = None

    def build_context(
        self,
        observation: str,
        current_time: Optional[str] = None,
        *,
        use_stes: bool = True,
    ) -> Dict[str, Any]:
        """
        构建最终上下文（供 Agent 传给 LLM）。
        返回: {"retrieved_context": "..."}
        """
        logger.info(f"--- CFF: 正在构建上下文 (Time={current_time}) ---")

        # 0) agent_name
        agent_name = (
            getattr(self.memory_system, "agent_name", None)
            or getattr(self.memory_system, "agentname", None)
            or "User"
        )

        # 1) STES episodic（短期情景记忆）——可关闭（longmemoryeval 建议关）
        stes_context = "None"
        if use_stes:
            stes_memories = []
            try:
                stes_memories = self.memory_system.retrieve_episodic_memories(
                    query=observation, k=short_memory_number
                )
            except TypeError:
                stes_memories = self.memory_system.retrieve_episodic_memories(observation, short_memory_number)
            except Exception as e:
                logger.error(f"[CFF] retrieve_episodic_memories 失败: {e}", exc_info=True)
                stes_memories = []

            stes_context = ContextBuilder.build_stes_context(stes_memories)

            if _DEBUG_PIPELINE:
                logger.info(f"[CFF][debug] STES memories={len(stes_memories)}")

        # 2) LTSS Graph triples（长期语义记忆 / 图谱证据）
        # 你现在的策略是：ContextBuilder 内部做 “consolidated 优先 + raw 补充”
        ltss_context = "None"
        if self.graphrag_retriever:
            try:
                ltss_context = self.graphrag_retriever.search(observation, current_time=current_time)
            except Exception as e:
                logger.error(f"[CFF] GraphRAGRetriever.search 失败: {e}", exc_info=True)
                ltss_context = "None"
        else:
            logger.warning("[CFF] GraphRAGRetriever 未就绪，跳过长期图谱检索。")

        # 统一三元组格式（避免下游解析变形）
        ltss_context = canonicalize_triple_block(ltss_context)

        if _DEBUG_PIPELINE:
            ctx_len = len(ltss_context or "")
            logger.info(f"[CFF][debug] LTSS context len={ctx_len}")

        # 3) Timeline triples（时间序列补充：仅在命中 intent 时触发）
        temporal_triples, temporal_reason = ContextBuilder.build_temporal_triples(
            observation=observation,
            graphrag_retriever=self.graphrag_retriever,
            agent_name=agent_name,
        )

        if _DEBUG_PIPELINE:
            logger.info(f"[CFF][debug] temporal_triples={len(temporal_triples)} reason={temporal_reason}")

        # 4) Derived facts（计数/时间差等派生事实：仅在命中 intent 时触发）
        derived_facts, derived_evidence = ContextBuilder.build_derived_facts(
            observation=observation,
            graphrag_retriever=self.graphrag_retriever,
        )

        if _DEBUG_PIPELINE:
            logger.info(f"[CFF][debug] derived_facts={len(derived_facts)} derived_evidence={len(derived_evidence)}")

        # 5) Assemble final context
        final_context = ContextBuilder.build_final_context(
            stes_context=stes_context,
            ltss_context=ltss_context,
            temporal_triples=temporal_triples,
            derived_facts=derived_facts,
            derived_evidence=derived_evidence,
        )

        if _DEBUG_PIPELINE:
            logger.info(f"[CFF][debug] final_context len={len(final_context or '')}")

        logger.info("[CFF] 上下文构建完成。")
        return {"retrieved_context": final_context}

    # 兼容旧代码里可能存在的 buildcontext 调用方式
    def buildcontext(self, observation: str, currenttime: Optional[str] = None) -> Dict[str, Any]:
        return self.build_context(observation=observation, current_time=currenttime)
