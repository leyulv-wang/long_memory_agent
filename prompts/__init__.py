# -*- coding: utf-8 -*-
"""
提示词模板管理模块

包含：
- 巩固阶段提示词 (consolidation_prompts.py)
- 问答阶段提示词 (answer_prompts.py)
- 问题分类器 (question_classifier.py)
"""

from .answer_prompts import get_answer_prompt, classify_question_type
from .consolidation_prompts import get_consolidation_prompt

__all__ = [
    "get_answer_prompt",
    "classify_question_type", 
    "get_consolidation_prompt",
]
