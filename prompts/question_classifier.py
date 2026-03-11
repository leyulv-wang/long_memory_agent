# -*- coding: utf-8 -*-
"""
问题类型分类器

使用 LLM 根据问题内容自动识别问题类型，选择对应的提示词模板。
"""

import json
import logging
import os
from enum import Enum
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """问题类型枚举"""
    COUNTING = "counting"           # 数量统计：how many, total, count
    TEMPORAL = "temporal"           # 时间推理：when, how long ago, order, first/last
    COMPARISON = "comparison"       # 比较推理：more, less, difference, compared to
    FACTUAL = "factual"             # 事实检索：what, where, who, which
    PREFERENCE = "preference"       # 偏好检索：prefer, like, favorite
    INSUFFICIENT = "insufficient"   # 信息不足判断：did I mention, have I ever


# LLM 分类提示词
_CLASSIFICATION_PROMPT = """Classify this question into ONE of these categories:

Categories:
1. COUNTING - Questions about quantities, totals, or counts (how many, total number, count)
2. TEMPORAL - Questions about time, dates, order, duration, OR calculations involving time/age (when, how long ago, what order, first/last, days ago, how old was I when, how long have I been)
3. COMPARISON - Questions comparing two things OR requiring mathematical calculations (more than, less than, difference, compared to, percentage, ratio, discount, how much X is Y)
4. FACTUAL - Questions asking for specific facts, including past states or beliefs (what, where, who, which, what was my previous X)
5. PREFERENCE - Questions about preferences, favorites, OR asking what kind of responses/suggestions user would prefer (favorite, prefer, like most, what kind of X would I prefer, suggestions based on my preferences)
6. INSUFFICIENT - Questions EXPLICITLY verifying if something was mentioned (did I mention, have I ever told you, did I say)

IMPORTANT: 
- "What was my previous X?" is FACTUAL (asking for a fact about past state)
- "Did I mention X?" is INSUFFICIENT (verifying if something was said)
- "How old was I when X?" is TEMPORAL (requires time-based calculation)
- "What kind of X would I prefer?" is PREFERENCE (asking about preference patterns)
- "Any suggestions for X?" when user has established preferences is PREFERENCE
- "What percentage discount did I get?" is COMPARISON (requires calculation: (original - discounted) / original)
- "How much did I save?" is COMPARISON (requires calculation)
- "What is the ratio of X to Y?" is COMPARISON (requires calculation)

Question: "{question}"

Reply with ONLY the category name in uppercase (e.g., "TEMPORAL" or "COUNTING").
"""

# 类型映射
_TYPE_MAP = {
    "COUNTING": QuestionType.COUNTING,
    "TEMPORAL": QuestionType.TEMPORAL,
    "COMPARISON": QuestionType.COMPARISON,
    "FACTUAL": QuestionType.FACTUAL,
    "PREFERENCE": QuestionType.PREFERENCE,
    "INSUFFICIENT": QuestionType.INSUFFICIENT,
}


def _get_cheap_llm():
    """获取便宜的 LLM 用于分类"""
    try:
        from utils.llm import get_llm
        return get_llm()
    except Exception:
        return None


def classify_question_with_llm(question: str) -> Tuple[QuestionType, float]:
    """
    使用 LLM 分类问题类型
    
    Args:
        question: 用户问题
    
    Returns:
        (问题类型, 置信度)
    
    ✅ 重要：分类失败时必须回退到最通用的 FACTUAL 类型，宁愿找不到类型也不能用错误的提示词
    """
    if not question:
        return QuestionType.FACTUAL, 0.5
    
    llm = _get_cheap_llm()
    if not llm:
        logger.warning("[classifier] LLM not available, falling back to FACTUAL (safest)")
        return QuestionType.FACTUAL, 0.5  # 直接返回 FACTUAL，不用规则分类
    
    try:
        prompt = _CLASSIFICATION_PROMPT.format(question=question)
        response = llm.invoke(prompt)
        result = (response.content or "").strip().upper()
        
        # 清理结果（去掉可能的引号、句号等）
        result = result.replace('"', '').replace("'", "").replace(".", "").strip()
        
        if result in _TYPE_MAP:
            if os.getenv("DEBUG_CLASSIFIER", "0") == "1":
                logger.info(f"[classifier] LLM classified '{question[:50]}...' as {result}")
            return _TYPE_MAP[result], 0.9
        else:
            # ✅ 分类失败时使用最通用的 FACTUAL 类型
            logger.warning(f"[classifier] LLM returned unknown type: {result}, falling back to FACTUAL (safest)")
            return QuestionType.FACTUAL, 0.5
            
    except Exception as e:
        # ✅ 异常时使用最通用的 FACTUAL 类型
        logger.warning(f"[classifier] LLM classification failed: {e}, falling back to FACTUAL (safest)")
        return QuestionType.FACTUAL, 0.5


def _classify_question_rule_based(question: str) -> Tuple[QuestionType, float]:
    """
    规则兜底分类（当 LLM 不可用时）
    """
    import re
    
    if not question:
        return QuestionType.FACTUAL, 0.5
    
    q_lower = question.lower().strip()
    
    # 时间类优先（更具体）
    temporal_patterns = [
        r"\bdays? ago\b", r"\bweeks? ago\b", r"\bmonths? ago\b",
        r"\border\b", r"\bfirst\b.*\blast\b", r"\bearliest\b.*\blatest\b",
        r"\bbefore\b", r"\bafter\b", r"\bwhen did\b",
        r"\bhow old was i when\b", r"\bhow long have i been\b",
    ]
    for pattern in temporal_patterns:
        if re.search(pattern, q_lower):
            return QuestionType.TEMPORAL, 0.8
    
    # 数量类
    counting_patterns = [r"\bhow many\b", r"\bhow much\b", r"\btotal\b", r"\bcount\b"]
    for pattern in counting_patterns:
        if re.search(pattern, q_lower):
            return QuestionType.COUNTING, 0.8
    
    # 比较类（包括百分比计算）
    comparison_patterns = [
        r"\bmore\b.*\bthan\b", r"\bless\b.*\bthan\b", r"\bdifference\b", r"\bcompared to\b",
        r"\bpercentage\b", r"\bdiscount\b", r"\bratio\b", r"\bhow much.*save\b",
    ]
    for pattern in comparison_patterns:
        if re.search(pattern, q_lower):
            return QuestionType.COMPARISON, 0.8
    
    # 偏好类（扩展识别）
    preference_patterns = [
        r"\bfavorite\b", r"\bprefer\b", r"\bwould i prefer\b",
        r"\bwhat kind of\b.*\bwould\b", r"\bsuggestions?\b.*\bbased on\b",
        r"\bany suggestions\b", r"\brecommendations?\b.*\bfor me\b",
    ]
    for pattern in preference_patterns:
        if re.search(pattern, q_lower):
            return QuestionType.PREFERENCE, 0.8
    
    # 信息验证类
    insufficient_patterns = [r"\bdid i mention\b", r"\bhave i ever\b", r"\bdid i tell\b"]
    for pattern in insufficient_patterns:
        if re.search(pattern, q_lower):
            return QuestionType.INSUFFICIENT, 0.8
    
    # 默认
    return QuestionType.FACTUAL, 0.6


def classify_question(question: str) -> Tuple[QuestionType, float]:
    """
    分类问题类型（主入口）
    
    默认使用 LLM 分类，失败时回退到 FACTUAL（最通用的类型）。
    可通过环境变量 USE_RULE_CLASSIFIER=1 强制使用规则分类。
    
    ✅ 重要：宁愿使用最通用的 FACTUAL 类型，也不能用错误的提示词导致回答错误
    
    Args:
        question: 用户问题
    
    Returns:
        (问题类型, 置信度)
    """
    if os.getenv("USE_RULE_CLASSIFIER", "0") == "1":
        return _classify_question_rule_based(question)
    
    return classify_question_with_llm(question)
