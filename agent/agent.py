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
from .answer_selector import AnswerSelector

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
        self.answer_selector = AnswerSelector()
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
    def _is_advicey(text: str) -> bool:
        if not text:
            return True
        lower = text.lower()
        bad_markers = [
            "consider", "recommend", "popular", "suitable", "you can", "you should",
            "check reviews", "compare prices", "tips",
        ]
        return any(m in lower for m in bad_markers)

    @staticmethod
    def _extract_json(text: str) -> str:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return m.group(1)
        return text.replace("```json", "").replace("```", "").strip()

    @staticmethod
    def _slice_allowed(ctx: str) -> str:
        ctx = ctx or ""
        headers = [
            "=== GRAPH TRIPLES (ALLOWED EVIDENCE) ===",
            "=== RAW GRAPH TRIPLES (SUPPLEMENT, ALLOWED EVIDENCE) ===",
            "=== TIMELINE TRIPLES (ALLOWED EVIDENCE) ===",
            "=== DERIVED FACTS (ALLOWED EVIDENCE) ===",
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
        稳定性修补（不新增能力，只避免因为格式问题丢证据）：
        - 过滤 knowledge_type=logical_inference（保持你原逻辑）
        - 允许三元组行不是 '-' 开头（例如 '(A) -[R]-> (B)'），也能进入 catalog
        - 允许 bullet '•' 行进入（会自动转成 '- '）
        """
        allowed_ctx = self._slice_allowed(retrieved_context or "")
        catalog: list[str] = []

        for ln in allowed_ctx.splitlines():
            s = (ln or "").strip()
            if not s:
                continue
            if s.startswith("===") and s.endswith("==="):
                continue
            if s.lower() in ("none", "无", "无相关长期知识。"):
                continue

            # 统一 bullet
            if s.startswith("•"):
                s = "- " + s.lstrip("•").strip()

            is_evidence_like = (
                    s.startswith("-")
                    or s.startswith("[DERIVED]")
                    or s.startswith("DERIVED")
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
                    continue

                catalog.append(s)

            if len(catalog) >= max_items:
                break

        return catalog

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
        """生成行动节点：先 deterministic derive_answer，失败再调用 LLM。"""
        logger.info(f"\n========== 3. {self.character_name} 生成行动阶段 ==========")

        curr_time_str = state.get("current_time", "Unknown Time")
        observation = (state.get("observation") or "").strip()
        retrieved_context = (state.get("retrieved_context") or "").strip()

        # 1) 尝试确定性回答
        selected = None
        try:
            selected = self.answer_selector.try_select({"observation": observation, "retrieved_context": retrieved_context})
        except Exception as e:
            logger.warning(f"[{self.character_name}] AnswerSelector 异常，转 LLM: {e}")

        if selected:
            return {
                "action": selected.action,
                "final_answer": selected.final_answer,
                "evidence_triples": selected.evidence_triples,
                                "error": None,
            }

        if _DEBUG_PIPELINE:
            logger.info(f"[{self.character_name}][debug] AnswerSelector skipped, using LLM")

        # 2) LLM 回答（闭域）
        catalog = self._build_evidence_catalog(retrieved_context, max_items=180)

        if _DEBUG_PIPELINE:
            logger.info(f"[{self.character_name}][debug] evidence catalog size={len(catalog)}")
            if not catalog:
                logger.info(f"[{self.character_name}][debug] evidence catalog empty")
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

        prompt = f"""
You must answer in English!

# 0. Current Status
- Current Virtual Time: {curr_time_str}

# 1. Core Identity
{self.character_anchor}

CRITICAL RESTRICTION:
- You are a CLOSED-DOMAIN memory retrieval agent.
- You must NOT give advice, tips, plans, or recommendations.
- You must NOT infer actions, resolutions, or outcomes unless they are EXPLICITLY stated in memory.

# 2. Current Question
{observation}

# 3. Evidence Catalog (ALLOWED EVIDENCE ONLY)
Select evidence by ID only. Do NOT copy lines verbatim.
Catalog:
{catalog_text}

# 4. Output Requirements
Return STRICT JSON. Required keys:
- "final_answer": one or two sentences answering the user's question.
- "evidence_ids": an array of integers selecting from the catalog above.

Example JSON:
{example_json}
""".strip()

        try:
            response = self.llm.invoke(prompt)
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

            insufficient = (
                "don't have enough information" in final_answer.lower()
                or "do not have enough information" in final_answer.lower()
            )

            # 禁止建议式回答
            if (not final_answer) or (self._is_advicey(final_answer) and not insufficient):
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
