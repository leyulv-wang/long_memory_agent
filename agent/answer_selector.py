# agent/answer_selector.py

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, Optional

from utils.derived_answer import derive_answer, DerivedAnswer

logger = logging.getLogger(__name__)
_DEBUG_DERIVED = os.getenv("DEBUG_DERIVED", "0") == "1"


@dataclass
class SelectedResult:
    final_answer: str
    evidence_triples: list[str]
    action: str
    selector_reason: str


class AnswerSelector:
    """
    Deterministic answer layer (pre-LLM).

    - If we can derive a strong answer from allowed triples, return it.
    - Else return None and let the LLM handle it.
    """

    def __init__(self, *, max_evidence: int = 10):
        try:
            self.max_evidence = int(max_evidence or 10)
        except Exception:
            self.max_evidence = 10
        if self.max_evidence < 1:
            self.max_evidence = 1

    def try_select(self, state: Dict[str, Any]) -> Optional[SelectedResult]:
        observation = (state.get("observation") or "").strip()

        # 兼容历史命名：retrieved_context / retrievedcontext
        retrieved_context = (
            (state.get("retrieved_context") or "")
            or (state.get("retrievedcontext") or "")
        ).strip()

        if not observation or not retrieved_context:
            return None

        # 关键修复：derive_answer 异常必须回退到 LLM（不能炸整条链路）
        try:
            da: Optional[DerivedAnswer] = derive_answer(observation, retrieved_context)
        except Exception:
            return None

        if not da and _DEBUG_DERIVED:
            logger.info("[AnswerSelector][debug] derive_answer returned None")

        if not da:
            return None

        final_answer = (getattr(da, "final_answer", "") or "").strip()
        raw_evidence = getattr(da, "evidence_triples", None) or []

        # 过滤无意义证据行（例如 "-" 这类），避免“有证据但显示为空”
        evidence_triples: list[str] = []
        for e in raw_evidence:
            if not isinstance(e, str):
                continue
            s = e.strip()
            if not s:
                continue
            if s in ("-", "- ", "—"):
                continue
            evidence_triples.append(s)

        # Hard safety: if we "derived" but don't have evidence lines, do NOT short-circuit.
        if not final_answer or not evidence_triples:
            if _DEBUG_DERIVED:
                logger.info("[AnswerSelector][debug] derived answer missing evidence or final_answer")
            return None


        # Normalize evidence lines for display, but keep semantics unchanged.
        # (We only remove leading "- " if it exists, then re-add "- " uniformly.)
        pretty_evidence: list[str] = []
        for line in evidence_triples[: self.max_evidence]:
            core = line.strip()
            core = core[2:].strip() if core.startswith("- ") else core.lstrip("-").strip()
            if not core:
                continue
            pretty_evidence.append(f"- {core}")

        # 若清洗后为空，说明证据行本身不合格，直接回退到 LLM
        if not pretty_evidence:
            return None

        action = "Final Answer: " + final_answer + "\n" + "Evidence:\n" + "\n".join(pretty_evidence)

        return SelectedResult(
            final_answer=final_answer,
            evidence_triples=evidence_triples[: self.max_evidence],
            action=action,
            selector_reason=(getattr(da, "reason", None) or "derived_answer"),
        )
