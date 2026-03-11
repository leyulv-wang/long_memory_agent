import json
import logging
import os
import re
from pathlib import Path
from typing import TypedDict, Dict, Any, Optional, List

from langgraph.graph import StateGraph, END

from memory.dual_memory_system import DualMemorySystem
from memory.stores import LongTermSemanticStore
from utils.llm import get_llm, get_expensive_llm
from utils.file_parsers import parse_book_to_dict
from config import memory_consolidation_threshold, CHARACTER_BOOKS_DIR

from .contextual_focus_framework import ContextualFocusFramework

# 导入新的提示词模块
from prompts import get_answer_prompt, classify_question_type

logger = logging.getLogger(__name__)
_DEBUG_PIPELINE = os.getenv("DEBUG_PIPELINE", "0") == "1"

class AgentState(TypedDict, total=False):
    """定义智能体在认知循环中传递的状态。"""
    observation: str
    retrieved_context: str
    action: str
    steps_since_last_consolidation: int
    error: Optional[str]
    current_time: Optional[str]
    final_answer: Optional[str]
    evidence_triples: Optional[List[str]]
    evidence_ids: Optional[List[int]]

class CognitiveAgent:
    """智能体核心类：感知 -> 构建上下文 -> 生成回答 -> 执行 -> (可选)巩固"""

    def __init__(self, character_name: str, ltss_instance: LongTermSemanticStore):
        logger.info(f"正在为 '{character_name}' 构建认知智能体...")

        self.llm = get_llm()
        self.expensive_llm = get_expensive_llm()

        self.character_name = character_name
        character_book_path = self._get_book_path("character", self.character_name)
        self.character_profile = parse_book_to_dict(character_book_path)
        self.character_anchor = self._format_anchor_from_profile(self.character_profile)

        initial_mems_raw = self.character_profile.get("initial_memories")
        if initial_mems_raw is None:
            initial_mems_raw = self.character_profile.get("Initial memories and tasks")
        if initial_mems_raw is None:
            initial_mems_raw = self.character_profile.get("最初的记忆和任务")
        if initial_mems_raw is None:
            initial_mems_raw = []

        initial_mems = [initial_mems_raw] if isinstance(initial_mems_raw, str) else initial_mems_raw

        # 关键：参数名与 dual_memory_system.py 对齐
        self.memory = DualMemorySystem(
            agent_name=self.character_name,
            ltss_instance=ltss_instance,
            initial_memories=initial_mems,
            consolidation_llm=self.llm,
        )

        self.cff = ContextualFocusFramework(self.memory)
        self.graph = self._build_graph()

        logger.info(f"认知智能体 '{self.character_name}' 构建完成。")

    def _get_book_path(self, book_type: str, book_name: str) -> Path:
        book_path = Path(CHARACTER_BOOKS_DIR) / f"{book_type}_book_{book_name}.txt"
        if not book_path.exists():
            raise FileNotFoundError(f"找不到书籍文件: {book_path}")
        return book_path

    def _format_anchor_from_profile(self, profile: dict) -> str:
        parts = [f"你是 {profile.get('name', '未知')}。"]

        core_id = profile.get("core_identity")
        if core_id:
            core_id_str = " ".join(core_id) if isinstance(core_id, list) else str(core_id)
            parts.append(f"- 核心身份: {core_id_str}")

        traits_raw = profile.get("personality_traits", [])
        traits = [traits_raw] if isinstance(traits_raw, str) else traits_raw
        if traits:
            parts.append("- 性格特质:")
            parts.extend([f" - {str(trait).lstrip('* ')}" for trait in traits])

        goals_raw = profile.get("goals", [])
        goals = [goals_raw] if isinstance(goals_raw, str) else goals_raw
        if goals:
            parts.append("- 当前目标:")
            parts.extend([f" - {str(goal).lstrip('* ')}" for goal in goals])

        return "\n".join(parts)

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("perceive", self.perceive)
        workflow.add_node("build_context", self.build_context_node)
        workflow.add_node("generate_action", self.generate_action)
        workflow.add_node("execute_action", self.execute_action)
        workflow.add_node("trigger_consolidation", self.trigger_consolidation)

        workflow.set_entry_point("perceive")
        workflow.add_edge("perceive", "build_context")
        workflow.add_edge("build_context", "generate_action")
        workflow.add_edge("generate_action", "execute_action")

        workflow.add_conditional_edges(
            "execute_action",
            self.should_consolidate,
            {"consolidate": "trigger_consolidation", "continue": END},
        )

        workflow.add_edge("trigger_consolidation", END)
        return workflow.compile()

    def should_consolidate(self, state: AgentState) -> str:
        if state.get("steps_since_last_consolidation", 0) >= memory_consolidation_threshold:
            return "consolidate"
        return "continue"

    def perceive(self, state: AgentState) -> Dict[str, Any]:
        """感知节点：把 observation 写入短期记忆，并累计步数。"""
        observation = (state.get("observation") or "").strip()
        if not observation:
            return {"error": "Empty observation"}

        logger.info(f"\n========== 1. {self.character_name} 感知阶段 ==========")
        logger.info(f'观测到: "{observation}"')

        try:
            self.memory.add_episodic_memory(observation)
        except Exception as e:
            logger.error(f"[{self.character_name}] 添加短期记忆出错: {e}", exc_info=True)
            return {
                "error": f"Failed to add episodic memory: {e}",
                                "steps_since_last_consolidation": state.get("steps_since_last_consolidation", 0) + 1,
            }

        return {
                        "steps_since_last_consolidation": state.get("steps_since_last_consolidation", 0) + 1,
        }

    def build_context_node(self, state: AgentState) -> Dict[str, Any]:
        """构建上下文节点：STES + LTSS 联合检索，拼接 retrieved_context。"""
        logger.info(f"\n========== 2. {self.character_name} 构建上下文阶段 ==========")

        observation = (state.get("observation") or "").strip()
        current_time = state.get("current_time")

        try:
            context_data = self.cff.build_context(observation, current_time=current_time)
            return {
                "retrieved_context": context_data.get("retrieved_context", "") or "",
                "error": None,
            }
        except Exception as e:
            logger.error(f"[{self.character_name}] 构建上下文出错: {e}", exc_info=True)
            return {
                "retrieved_context": "",
                "error": f"Failed to build context: {e}",
            }

    @staticmethod
    def _extract_json(text: str) -> str:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return m.group(1)
        return text.replace("```json", "").replace("```", "").strip()

    @staticmethod
    def _slice_allowed(ctx: str) -> str:
        """
        提取允许作为证据的上下文区块。
        
        ✅ 支持多种格式：
        - 旧格式：=== GRAPH TRIPLES (ALLOWED EVIDENCE) ===
        - 中间格式：=== LONG-TERM MEMORY FACTS (USE AS EVIDENCE) ===
        - 新格式（SimpleRetriever）：=== LONG-TERM MEMORY FACTS (语义检索) ===
        """
        ctx = ctx or ""
        headers = [
            # 旧格式
            "=== GRAPH TRIPLES (ALLOWED EVIDENCE) ===",
            "=== RAW GRAPH TRIPLES (SUPPLEMENT, ALLOWED EVIDENCE) ===",
            "=== ORIGINAL TEXT EVIDENCE (USE FOR DETAILED ANSWERS) ===",
            "=== TIMELINE TRIPLES (ALLOWED EVIDENCE) ===",
            "=== DERIVED FACTS (ALLOWED EVIDENCE) ===",
            # 中间格式
            "=== LONG-TERM MEMORY FACTS (USE AS EVIDENCE) ===",
            "=== RAW SUPPLEMENT (ALLOWED EVIDENCE) ===",
            "=== TIMELINE (ALLOWED EVIDENCE) ===",
            # 新格式（SimpleRetriever 输出）
            "=== LONG-TERM MEMORY FACTS (语义检索) ===",
            "=== KEYWORD MATCHED FACTS (关键词检索) ===",
            "=== ORIGINAL TEXT (原文兜底) ===",
        ]
        lines = ctx.splitlines()
        kept = []
        in_block = False

        for line in lines:
            if line.strip() in headers:
                in_block = True
                kept.append(line)
                continue
            if line.strip().startswith("===") and line.strip().endswith("===") and line.strip() not in headers:
                in_block = False
            if in_block:
                kept.append(line)

        if not any(h in "\n".join(kept) for h in headers):
            return ctx
        return "\n".join(kept)

    @staticmethod
    def _parse_meta(line: str) -> Dict[str, str]:
        """解析证据行尾部的 [k=v; k=v]，解析不到就返回 {}。"""
        if not isinstance(line, str):
            return {}
        m = re.search(r"\[(.*?)\]\s*$", line.strip())
        if not m:
            return {}
        blob = m.group(1).strip()
        if not blob:
            return {}
        out: Dict[str, str] = {}
        for part in [p.strip() for p in blob.split(";") if p.strip()]:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                out[k] = v
        return out

    def _build_evidence_catalog(self, retrieved_context: str, max_items: int = 120) -> list[str]:
        """
        构建证据目录，支持多种格式：
        
        旧格式：- (A) -[REL]-> (B) [k=v; ...]
        新格式（SimpleRetriever）：
            [Fact N] 主要内容
              时间: session_time=xxx, turn_id=xxx
              来源: source=user, score=0.8
            [Match N] 主要内容（关键词检索）
              时间: session_time=xxx, turn_id=xxx
              来源: source=user
            [Turn N] (时间) 原文内容
        
        ✅ 处理逻辑：
        - 识别 [Fact N]、[Match N]、[Turn N] 开头的行作为主要证据
        - 将 Fact/Match 块的信息合并成一行，便于 LLM 引用
        - 保留时间和来源信息用于排序
        """
        allowed_ctx = self._slice_allowed(retrieved_context or "")
        
        # ✅ 调试：打印 allowed_ctx 的前 2000 个字符
        if _DEBUG_PIPELINE:
            logger.info(f"[_build_evidence_catalog] allowed_ctx length={len(allowed_ctx)}")
            logger.info(f"[_build_evidence_catalog] allowed_ctx preview:\n{allowed_ctx[:2000]}")
            # 检查是否包含 KEYWORD MATCHED FACTS
            if "=== KEYWORD MATCHED FACTS" in allowed_ctx:
                logger.info(f"[_build_evidence_catalog] ✅ Contains KEYWORD MATCHED FACTS section")
                # 找到 KEYWORD 部分的位置
                keyword_pos = allowed_ctx.find("=== KEYWORD MATCHED FACTS")
                logger.info(f"[_build_evidence_catalog] KEYWORD section starts at position {keyword_pos}")
                logger.info(f"[_build_evidence_catalog] KEYWORD section preview:\n{allowed_ctx[keyword_pos:keyword_pos+1000]}")
            else:
                logger.info(f"[_build_evidence_catalog] ❌ KEYWORD MATCHED FACTS section NOT FOUND")
        
        catalog: list[str] = []
        
        lines = allowed_ctx.splitlines()
        i = 0
        while i < len(lines):
            s = (lines[i] or "").strip()
            
            if not s:
                i += 1
                continue
            if s.startswith("===") and s.endswith("==="):
                i += 1
                continue
            if s.lower() in ("none", "无", "无相关长期知识。", "无相关长期记忆。", "无关键词匹配结果。", "无相关原文。"):
                i += 1
                continue
            # 跳过匹配关键词说明行
            if s.startswith("匹配关键词:") or s.startswith("（关键词匹配结果已包含在语义检索中）"):
                i += 1
                continue
            
            # ✅ 新格式：[Fact N] 或 [Match N] 开头
            if s.startswith("[Fact ") or s.startswith("[Match "):
                # ✅ 调试：记录解析的 Fact/Match
                is_match = s.startswith("[Match ")
                if _DEBUG_PIPELINE and is_match:
                    logger.info(f"[_build_evidence_catalog] Parsing {s[:80]}")
                
                # 解析 Fact/Match 块
                fact_line = s
                meta_parts = []
                session_time = ""
                
                # 读取后续的元数据行
                j = i + 1
                while j < len(lines):
                    next_line = (lines[j] or "").strip()
                    if not next_line:
                        j += 1
                        continue
                    # 遇到下一个证据块或分隔符时停止
                    if (next_line.startswith("[Fact ") or 
                        next_line.startswith("[Match ") or 
                        next_line.startswith("[Turn ") or 
                        next_line.startswith("===")):
                        break
                    
                    # 解析元数据（注意：元数据行可能有缩进）
                    next_line_stripped = next_line.lstrip()
                    if next_line_stripped.startswith("时间:") or next_line_stripped.startswith("时间："):
                        time_part = next_line_stripped[3:].strip()
                        meta_parts.append(time_part)
                        # 提取 session_time 用于排序
                        st_match = re.search(r'session_time=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', time_part)
                        if st_match:
                            session_time = st_match.group(1)
                    elif next_line_stripped.startswith("结构:") or next_line_stripped.startswith("结构："):
                        meta_parts.append(f"结构: {next_line_stripped[3:].strip()}")
                    elif next_line_stripped.startswith("来源:") or next_line_stripped.startswith("来源："):
                        meta_parts.append(next_line_stripped[3:].strip())
                    elif next_line_stripped.startswith("原文:") or next_line_stripped.startswith("原文："):
                        meta_parts.append(f"原文: {next_line_stripped[3:].strip()}")
                    
                    j += 1
                
                # 构建合并后的证据行
                meta_str = "; ".join([p for p in meta_parts if p])
                if meta_str:
                    catalog_line = f"{fact_line} [{meta_str}]"
                else:
                    catalog_line = fact_line
                
                # ✅ 调试：记录构建的 catalog_line
                if _DEBUG_PIPELINE and is_match:
                    logger.info(f"[_build_evidence_catalog] Built catalog_line: {catalog_line[:120]}")
                
                # 存储 session_time 用于排序
                catalog.append((catalog_line, session_time))
                i = j
                continue
            
            # ✅ 原文记录格式：[Turn N] xxx
            # 注意：原文可能是多行的，需要合并
            if s.startswith("[Turn "):
                # 读取后续的原文内容行
                content_lines = [s]
                j = i + 1
                while j < len(lines):
                    next_line = (lines[j] or "").strip()
                    if not next_line:
                        j += 1
                        continue
                    # 遇到下一个证据块或分隔符时停止
                    if (next_line.startswith("[Fact ") or 
                        next_line.startswith("[Match ") or 
                        next_line.startswith("[Turn ") or 
                        next_line.startswith("===")):
                        break
                    # 合并原文内容（截取前 500 字符避免过长）
                    content_lines.append(next_line[:500])
                    j += 1
                    # 最多读取 5 行原文
                    if len(content_lines) > 5:
                        break
                
                # 合并成一行
                merged_line = " ".join(content_lines)
                catalog.append((merged_line, "9999-99-99T99:99:99"))  # 原文记录放最后
                i = j
                continue
            
            # 统一 bullet
            if s.startswith("•"):
                s = "- " + s.lstrip("•").strip()

            # 原文片段（[原文1] xxx 或 [unit_TURN_1] xxx）
            is_original_text = s.startswith("[原文") or s.startswith("[unit_")
            
            is_evidence_like = (
                    s.startswith("-")
                    or s.startswith("[DERIVED]")
                    or s.startswith("DERIVED")
                    or is_original_text
                    or (
                        # 兜底：典型 triple 形态 (A) -[REL]-> (B)
                            s.startswith("(") and (") -[" in s or ")-[" in s) and "]->" in s
                    )
            )

            if is_evidence_like:
                # 如果是 "(A) -[R]-> (B)"，为了后续一致性，补个 "- "
                if s.startswith("("):
                    s = "- " + s

                meta = self._parse_meta(s)
                kt = (meta.get("knowledge_type") or meta.get("knowledgeType") or "").strip().lower()
                if kt == "logical_inference":
                    i += 1
                    continue

                # 提取 session_time 用于排序
                session_time = meta.get("session_time", "9999-99-99T99:99:99")
                catalog.append((s, session_time))

            i += 1
            
            if len(catalog) >= max_items:
                break

        # ✅ 去重：完全相同的行只保留一个
        seen = set()
        unique_catalog = []
        
        if _DEBUG_PIPELINE:
            logger.info(f"[_build_evidence_catalog] Before dedup: catalog size={len(catalog)}")
            # 统计 Match 的数量
            match_count = sum(1 for item in catalog if "[Match " in (item[0] if isinstance(item, tuple) else item))
            logger.info(f"[_build_evidence_catalog] Before dedup: Match count={match_count}")
        
        for item in catalog:
            line = item[0] if isinstance(item, tuple) else item
            if line not in seen:
                seen.add(line)
                unique_catalog.append(item)
            elif _DEBUG_PIPELINE and "[Match " in line:
                logger.info(f"[_build_evidence_catalog] DUPLICATE REMOVED: {line[:120]}")
        
        catalog = unique_catalog
        
        if _DEBUG_PIPELINE:
            logger.info(f"[_build_evidence_catalog] After dedup: catalog size={len(catalog)}")
            # 统计 Match 的数量
            match_count = sum(1 for item in catalog if "[Match " in (item[0] if isinstance(item, tuple) else item))
            logger.info(f"[_build_evidence_catalog] After dedup: Match count={match_count}")

        # ✅ 排序策略：
        # - 对于时间顺序问题（order, earliest, latest, first, last），按 session_time 排序
        # - 对于其他问题，保持原有顺序（相关性排序）
        # 注意：简化检索器已经按相关性排序，这里不应该打乱
        # 只有当明确需要时间排序时才重新排序
        # catalog.sort(key=lambda x: x[1] if isinstance(x, tuple) else "9999-99-99T99:99:99")
        
        # 提取最终的行列表
        result = [item[0] if isinstance(item, tuple) else item for item in catalog]
        
        if _DEBUG_PIPELINE:
            logger.info(f"[_build_evidence_catalog] Final result size={len(result)}")
            # 统计 Match 的数量
            match_count = sum(1 for line in result if "[Match " in line)
            logger.info(f"[_build_evidence_catalog] Final Match count={match_count}")
            # 输出前 5 条
            for i, line in enumerate(result[:5], 1):
                logger.info(f"[_build_evidence_catalog] result[{i}] {line[:120]}")
        
        return result

    @staticmethod
    def _render_evidence_catalog(catalog: list[str]) -> str:
        if not catalog:
            return "(empty)"
        return "\n".join([f"[{i + 1}] {catalog[i]}" for i in range(len(catalog))])

    @staticmethod
    def _resolve_evidence_ids(raw_ids: Any, catalog: list[str]) -> list[str]:
        if isinstance(raw_ids, int):
            raw_ids = [raw_ids]
        if isinstance(raw_ids, str):
            raw_ids = [x.strip() for x in raw_ids.replace("，", ",").split(",") if x.strip()]
        if not isinstance(raw_ids, list):
            return []

        ids: list[int] = []
        for x in raw_ids:
            try:
                ids.append(int(x))
            except Exception:
                continue

        valid_lines: list[str] = []
        seen = set()
        for i in ids:
            if i in seen:
                continue
            seen.add(i)
            if 1 <= i <= len(catalog):
                valid_lines.append(catalog[i - 1])
            if len(valid_lines) >= 8:
                break

        return valid_lines

    def generate_action(self, state: AgentState) -> Dict[str, Any]:
        """生成行动节点：使用 LLM 生成回答。"""
        logger.info(f"\n========== 3. {self.character_name} 生成行动阶段 ==========")

        curr_time_str = state.get("current_time", "Unknown Time")
        observation = (state.get("observation") or "").strip()
        retrieved_context = (state.get("retrieved_context") or "").strip()

        # LLM 回答（闭域）
        catalog = self._build_evidence_catalog(retrieved_context, max_items=180)

        if _DEBUG_PIPELINE:
            logger.info(f"[{self.character_name}][debug] evidence catalog size={len(catalog)}")
            if not catalog:
                logger.info(f"[{self.character_name}][debug] evidence catalog empty")
            else:
                # 输出前 5 条证据，帮助调试
                for i, line in enumerate(catalog[:5], 1):
                    logger.info(f"[{self.character_name}][debug] catalog[{i}] {line[:150]}...")
        if os.getenv("DEBUG_EVIDENCE", "0") == "1":
            logger.info(
                f"[{self.character_name}] evidence catalog size={len(catalog)}"
            )
            for i, line in enumerate(catalog[:10], 1):
                logger.info(f"[{self.character_name}] catalog[{i}] {line}")
        catalog_text = self._render_evidence_catalog(catalog)

        example_json = json.dumps(
            {"final_answer": "I don't have enough information to answer that.", "evidence_ids": []},
            ensure_ascii=False,
            indent=2,
        )

        # ✅ 使用新的多类型提示词模块
        # 根据问题类型自动选择对应的提示词模板
        question_type = classify_question_type(observation)
        if _DEBUG_PIPELINE:
            logger.info(f"[{self.character_name}][debug] question_type={question_type.value}")
        
        prompt = get_answer_prompt(
            question=observation,
            current_time=curr_time_str,
            character_anchor=self.character_anchor,
            catalog_text=catalog_text,
            example_json=example_json,
            question_type=question_type,
        )

        try:
            # 最终回答使用昂贵模型 + 低温，确保回答质量和稳定性
            response = self.expensive_llm.bind(temperature=0.0).invoke(prompt)
            llm_output_text = (response.content or "").strip()
            parsed = json.loads(self._extract_json(llm_output_text))
            if not isinstance(parsed, dict):
                raise ValueError("LLM output JSON is not an object")

            final_answer = (parsed.get("final_answer") or "").strip()
            evidence_triples = self._resolve_evidence_ids(parsed.get("evidence_ids", []), catalog)
            if os.getenv("DEBUG_EVIDENCE", "0") == "1":
                logger.info(
                    f"[{self.character_name}] LLM evidence_ids={parsed.get('evidence_ids', [])} "
                    f"resolved={len(evidence_triples)}"
                )

            # ✅ 移除代码强制弃权逻辑，完全依靠 LLM 自主判断
            # 只在回答为空时才强制弃权（这是异常情况）
            if not final_answer:
                final_answer = "I don't have enough information to answer that."
                evidence_triples = []

            if evidence_triples:
                pretty_lines = [s if s.strip().startswith("-") else f"- {s.strip()}" for s in evidence_triples]
                action_text = "Final Answer: " + final_answer + "\nEvidence:\n" + "\n".join(pretty_lines)
            else:
                action_text = "Final Answer: " + final_answer + "\nEvidence:"

            return {
                "action": action_text,
                "final_answer": final_answer,
                "evidence_triples": evidence_triples,
                                "error": None,
            }

        except Exception as e:
            logger.error(f"[{self.character_name}] 解析/生成回答出错: {e}", exc_info=True)
            return {
                "action": "Final Answer: I don't have enough information to answer that.\nEvidence:",
                "final_answer": "I don't have enough information to answer that.",
                "evidence_triples": [],
                "error": f"Failed to generate answer: {e}",
            }

    def execute_action(self, state: AgentState) -> Dict[str, Any]:
        """执行节点：不再重复写 observation 到 STES（perceive 已写过）。"""
        logger.info(f"\n========== 4. {self.character_name} 执行行动阶段 ==========")

        action = state.get("action", "")

        logger.info(f"Action: {action[:120]}{'...' if len(action) > 120 else ''}")
        return {}

    def trigger_consolidation(self, state: AgentState) -> Dict[str, Any]:
        """巩固节点：把 STES 最近 K 条进行反思抽取，写入 Neo4j。"""
        logger.info(f"\n========== 5. {self.character_name} 记忆巩固阶段 ==========")

        steps = int(state.get("steps_since_last_consolidation", memory_consolidation_threshold) or memory_consolidation_threshold)
        current_virtual_time = state.get("current_time")

        try:
            self.memory.trigger_consolidation(current_time=current_virtual_time, k=steps)
            return {"steps_since_last_consolidation": 0, "error": None}
        except Exception as e:
            logger.error(f"[{self.character_name}] 记忆巩固出错: {e}", exc_info=True)
            return {"steps_since_last_consolidation": 0, "error": f"Failed during consolidation: {e}"}

    def run(self, observation: str, initial_state: Optional[Dict[str, Any]] = None, current_time: Optional[str] = None) -> Dict[str, Any]:
        """对外入口：运行一次完整认知循环。"""
        if initial_state is None:
            current_state: AgentState = {
                "observation": observation,
                "retrieved_context": "",
                "action": "",
                "final_answer": "",
                "evidence_triples": [],
                                "steps_since_last_consolidation": 0,
                "error": None,
                "current_time": current_time,
                "evidence_ids": [],
            }
        else:
            current_state = {
                "observation": observation,
                "retrieved_context": initial_state.get("retrieved_context", ""),
                "action": initial_state.get("action", ""),
                "final_answer": initial_state.get("final_answer", ""),
                "evidence_triples": initial_state.get("evidence_triples", []) or [],
                                "steps_since_last_consolidation": int(initial_state.get("steps_since_last_consolidation", 0) or 0),
                "error": initial_state.get("error"),
                "current_time": current_time,
                "evidence_ids": [],
            }

        final_state = self.graph.invoke(current_state)
        logger.info(f"\n--- {self.character_name} 的一次完整认知循环结束 ---")
        return final_state
