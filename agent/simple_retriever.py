# -*- coding: utf-8 -*-
"""
统一检索器 (Simple Retriever)

设计目标：
1. 简化检索流程，减少中间处理步骤，避免丢失正确答案
2. 保留核心能力：时间推理、多跳扩展、原文兜底
3. 综合评分机制：多因素加权排序（核心创新）
4. 动态版本检测：自动识别同一主题的多个版本（核心创新）
5. 三部分独立输出：语义检索、关键词检索、原文兜底

存储模式：全量保留（Append-Only）
- 每次提及都创建独立记录（belief_key 包含 turn_id）
- 所有历史版本都保留，不会被覆盖
- 通过动态版本检测识别同一主题的不同版本

检索流程：
┌─────────────────────────────────────────────────────────────────┐
│ 1. 简单事实向量检索（主路径，k=100）                             │
│    ↓                                                            │
│ 2. 查询扩展（LLM 扩展关键词）                                    │
│    ↓                                                            │
│ 3. 全文检索（BM25关键词匹配，独立输出，k=20）                    │
│    ↓                                                            │
│ 4. 多跳扩展（通过三元组关联更多简单事实，limit=20）              │
│    ↓                                                            │
│ 5. TextUnit 原文检索（兜底，k=10）                               │
│    ↓                                                            │
│ 6. 动态版本检测（识别同一主题的多个版本）                        │
│    ↓                                                            │
│ 7. 综合评分（sim×0.7 + conf×0.2 + chan×0.1）                    │
│    ↓                                                            │
│ 8. 三部分独立输出：                                              │
│    - 语义检索结果（向量检索 + 多跳扩展，top-60）                 │
│    - 关键词检索结果（全文检索，top-15）                          │
│    - 原文兜底（TextUnit，top-10）                                │
└─────────────────────────────────────────────────────────────────┘

核心创新：
1. 综合评分机制：多因素加权，平衡相关性、可靠性和时效性
2. 动态版本检测：不依赖硬编码的关系类型列表，通过语义相似度自动识别
3. 双维度时间衰减：session_time + turn_id
4. 全量保留：保留所有历史版本，支持时间线分析
5. 三部分独立输出：语义检索和关键词检索不竞争，各自独立展示
"""

import logging
import os
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import json

from utils.embedding import get_embedding_model
from utils.llm import get_llm, invoke_json

logger = logging.getLogger(__name__)
_DEBUG = os.getenv("DEBUG_PIPELINE", "0") == "1"

# ============================================================================
# 消融实验开关（通过环境变量控制）
# ============================================================================
# ABLATION_NO_MULTI_HOP=1  禁用多跳扩展（结构化三元组推理）
_ABLATION_NO_MULTI_HOP = os.getenv("ABLATION_NO_MULTI_HOP", "0") == "1"
# ABLATION_NO_RAW_FALLBACK=1  禁用RAW通道兜底（TextUnit原文检索）
_ABLATION_NO_RAW_FALLBACK = os.getenv("ABLATION_NO_RAW_FALLBACK", "0") == "1"
# ABLATION_NO_VERSION_HISTORY=1  禁用历史版本保留（只保留最新版本）
_ABLATION_NO_VERSION_HISTORY = os.getenv("ABLATION_NO_VERSION_HISTORY", "0") == "1"
# ABLATION_NO_HYBRID_SEARCH=1  禁用混合检索（查询扩展 + BM25），只用向量检索
_ABLATION_NO_HYBRID_SEARCH = os.getenv("ABLATION_NO_HYBRID_SEARCH", "0") == "1"
# ABLATION_NO_SCORING=1  禁用综合评分机制，只用向量相似度排序
_ABLATION_NO_SCORING = os.getenv("ABLATION_NO_SCORING", "0") == "1"

# 配置
_SIMPLE_FACT_INDEX = "__node___vector_index"
_TEXTUNIT_INDEX = "textunit_vector_index"
_DEFAULT_SIMPLE_FACT_K = 100  # 简单事实检索数量（从80增加到100，提高召回率）
_DEFAULT_TEXTUNIT_K = 10     # 原文检索数量（加强兜底）
_DEFAULT_MULTI_HOP_LIMIT = 20  # 多跳扩展限制
_DEFAULT_FULLTEXT_K = 20      # 全文检索数量（从15增加到20，混合检索）
_ENABLE_SNIPPET_EXTRACTION = True  # 是否启用LLM提取关键片段
_ENABLE_QUERY_EXPANSION = True  # 是否启用查询扩展（用LLM扩展查询关键词）
_SNIPPET_EXTRACTION_PARALLEL_WORKERS = int(os.getenv("SNIPPET_EXTRACTION_PARALLEL_WORKERS", "3"))  # 片段提取并行 worker 数

# ============================================================================
# 综合评分机制配置（核心创新点）
# ============================================================================
# 评分公式: final_score = base_score × time_decay
# base_score = sim × 0.7 + confidence × 0.2 + channel_bonus × 0.1
# 注意：多跳惩罚已内置在 sim 中（seed_sim × 0.85^hop），不需要额外权重
_WEIGHT_SIM = 0.7        # 向量相似度权重（多跳惩罚已内置）
_WEIGHT_CONFIDENCE = 0.2  # 置信度权重
_WEIGHT_CHANNEL = 0.1     # 通道优先级权重

# 通道优先级
_CHANNEL_BONUS = {
    "consolidated": 1.0,   # 巩固后的信息最可靠
    "raw": 0.8,            # 原始信息次之
}

# 全文检索配置
# 注意：全文检索结果独立输出，不参与综合评分竞争
_DEFAULT_FULLTEXT_K = 20      # 全文检索数量（从15增加到20）

# 时间衰减配置
_SESSION_DECAY_RATE = 0.005   # 每天衰减 0.5%
_SESSION_DECAY_MIN = 0.85     # 最低衰减到 85%
_TURN_DECAY_RATE = 0.001      # 每轮衰减 0.1%
_TURN_DECAY_MIN = 0.95        # 最低衰减到 95%

# 动态版本检测配置
_VERSION_SIMILARITY_THRESHOLD = 0.75  # 语义相似度阈值，高于此值认为是同一主题
_MIN_TIME_DIFF_FOR_VERSION = 1  # 最小时间差（天），用于判断是否是不同版本


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except:
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        if val is None:
            return default
        return int(float(val))
    except:
        return default


def _parse_turn_id(val: Any) -> int:
    """解析 turn_id，支持 TURN_1、1、'1' 等格式"""
    if val is None:
        return 0
    s = str(val).strip()
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else 0


# 历史版本查询关键词
# 注意：当前使用全量保留模式，所有版本都是 is_current=true
# 这个关键词列表保留用于未来可能的软更新模式
_HISTORY_KEYWORDS = [
    # 中文
    "之前", "以前", "曾经", "过去", "原来", "原先", "以往", "先前",
    # 英文
    "previous", "previously", "used to", "before", "formerly", "earlier",
    "original", "originally", "past", "prior", "old", "former",
    # 短语
    "in the past", "at first", "at the beginning", "initially",
]


def _should_include_history(query: str) -> bool:
    """
    检测问题是否询问历史版本
    
    注意：当前使用全量保留模式，此函数仅用于调试日志。
    所有版本都会被检索，不会被过滤。
    
    例如：
    - "What was my previous stance on spirituality?" → True
    - "How often did I used to play tennis?" → True
    - "What is my current job?" → False
    """
    query_lower = query.lower()
    return any(kw in query_lower for kw in _HISTORY_KEYWORDS)


# ============================================================
# 综合评分机制（核心创新）
# ============================================================

def _compute_comprehensive_score(
    fact: Dict[str, Any],
    current_turn: int = 0,
    current_session_time: str = "",
) -> float:
    """
    计算事实的综合评分
    
    公式: final_score = base_score × time_decay
    base_score = sim × 0.7 + confidence × 0.2 + channel_bonus × 0.1
    time_decay = session_decay × turn_decay
    
    注意：
    1. 全文检索结果独立输出，不参与综合评分
    2. 多跳惩罚已内置在 sim 中（seed_sim × 0.85^hop）
    
    Args:
        fact: 事实字典，包含 sim, confidence, channel, hop, turn_id, session_time
        current_turn: 当前轮次
        current_session_time: 当前会话时间（ISO 格式）
    
    Returns:
        综合评分 (0-1)
    """
    # 1. 基础分数
    sim = _safe_float(fact.get("sim"), 0.0)
    confidence = _safe_float(fact.get("confidence"), 1.0)
    channel = (fact.get("channel") or "consolidated").lower()
    
    channel_bonus = _CHANNEL_BONUS.get(channel, 0.8)
    
    # 综合评分（多跳惩罚已内置在 sim 中）
    base_score = (
        sim * 0.7 +
        confidence * 0.2 +
        channel_bonus * 0.1
    )
    
    # 2. 时间衰减
    time_decay = _compute_time_decay(fact, current_turn, current_session_time)
    
    # 3. 最终分数
    final_score = base_score * time_decay
    
    return final_score


def _compute_time_decay(
    fact: Dict[str, Any],
    current_turn: int = 0,
    current_session_time: str = "",
) -> float:
    """
    计算双维度时间衰减
    
    1. 跨 Session 衰减（基于 session_time）：每天衰减 0.5%，最低 85%
    2. Session 内衰减（基于 turn_id）：每轮衰减 0.1%，最低 95%
    
    设计原则：以准确性为主，时间衰减只作为轻微调整因子
    """
    session_decay = 1.0
    turn_decay = 1.0
    
    # 1. 跨 Session 衰减
    fact_session_time = fact.get("session_time") or ""
    if fact_session_time and current_session_time:
        try:
            from datetime import datetime
            # 解析时间（支持多种格式）
            for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    fact_dt = datetime.strptime(fact_session_time[:19], fmt)
                    current_dt = datetime.strptime(current_session_time[:19], fmt)
                    days_diff = abs((current_dt - fact_dt).days)
                    session_decay = max(_SESSION_DECAY_MIN, 1.0 - days_diff * _SESSION_DECAY_RATE)
                    break
                except:
                    continue
        except:
            pass
    
    # 2. Session 内衰减
    fact_turn = _safe_int(fact.get("turn_id"), 0)
    if current_turn > 0 and fact_turn > 0:
        turn_diff = abs(current_turn - fact_turn)
        turn_decay = max(_TURN_DECAY_MIN, 1.0 - turn_diff * _TURN_DECAY_RATE)
    
    return session_decay * turn_decay


def _apply_comprehensive_scoring(
    facts: List[Dict[str, Any]],
    current_turn: int = 0,
    current_session_time: str = "",
    query: str = "",
) -> List[Dict[str, Any]]:
    """
    对事实列表应用综合评分，并按评分排序
    
    Args:
        facts: 事实列表
        current_turn: 当前轮次
        current_session_time: 当前会话时间
        query: 用户问题（保留参数，未来可能用于更复杂的相关性判断）
    
    Returns:
        按综合评分排序的事实列表（添加了 score 字段）
    """
    for fact in facts:
        fact["score"] = _compute_comprehensive_score(fact, current_turn, current_session_time)
    
    # 按综合评分降序排序
    facts.sort(key=lambda f: -f.get("score", 0))
    
    if _DEBUG and facts:
        logger.info(f"[SimpleRetriever] 综合评分 top 5:")
        for i, f in enumerate(facts[:5], 1):
            logger.info(f"  [{i}] score={f.get('score', 0):.4f} sim={f.get('sim', 0):.4f} {f['name'][:50]}...")
    
    return facts


# ============================================================
# 动态版本检测（核心创新）
# ============================================================

def _extract_subject_pattern(fact_text: str) -> str:
    """
    从事实文本中提取主语模式，用于分组
    
    例如：
    - "[user] User plays tennis every week" → "user_plays_tennis"
    - "[user] User plays tennis every other week" → "user_plays_tennis"
    - "[user] User lives in New York" → "user_lives"
    """
    text = fact_text.lower().strip()
    
    # 移除来源标记
    text = re.sub(r"^\[(?:user|assistant)\]\s*", "", text)
    
    # 提取主语（通常是 User 或人名）
    subject_match = re.match(r"^(\w+)\s+", text)
    subject = subject_match.group(1) if subject_match else "unknown"
    
    # 提取动词/谓语（前几个词）
    words = text.split()[:4]  # 取前4个词
    
    # 移除数字和频率词，保留核心动作
    core_words = []
    skip_words = {"every", "the", "a", "an", "to", "for", "with", "at", "in", "on", "is", "are", "was", "were"}
    for w in words:
        w_clean = re.sub(r"[^a-z]", "", w)
        if w_clean and w_clean not in skip_words and not w_clean.isdigit():
            core_words.append(w_clean)
            if len(core_words) >= 3:
                break
    
    return "_".join(core_words) if core_words else "unknown"


def _compute_text_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的相似度（基于词重叠）
    这是一个轻量级的相似度计算，不需要调用 embedding 模型
    """
    # 预处理
    def normalize(text):
        text = text.lower()
        text = re.sub(r"^\[(?:user|assistant)\]\s*", "", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return set(text.split())
    
    words1 = normalize(text1)
    words2 = normalize(text2)
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard 相似度
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def _parse_session_time(time_str: str) -> Optional[str]:
    """解析 session_time，返回日期部分（YYYY-MM-DD）"""
    if not time_str:
        return None
    # 支持 ISO 格式：2023-05-24T08:13:00 或 2023-05-24
    match = re.match(r"(\d{4}-\d{2}-\d{2})", time_str)
    return match.group(1) if match else None


def _detect_version_groups(facts: List[Dict], similarity_threshold: float = _VERSION_SIMILARITY_THRESHOLD) -> List[Dict]:
    """
    动态检测同一主题的多个版本
    
    核心逻辑：
    1. 按主语模式分组
    2. 在同一组内，检测高相似度但不同时间的事实
    3. 标记为版本组，按时间排序标记版本号（v1, v2, v3, ...）
    
    改进：支持多个版本（不限于2个）
    
    Args:
        facts: 事实列表
        similarity_threshold: 相似度阈值
    
    Returns:
        标记了版本信息的事实列表
    """
    if not facts or len(facts) < 2:
        return facts
    
    # 1. 按主语模式分组
    pattern_groups: Dict[str, List[Dict]] = defaultdict(list)
    for f in facts:
        pattern = _extract_subject_pattern(f.get("name", ""))
        pattern_groups[pattern].append(f)
    
    # 2. 在每个组内检测版本
    version_id = 0
    for pattern, group_facts in pattern_groups.items():
        if len(group_facts) < 2:
            continue
        
        # 按时间排序
        group_facts_sorted = sorted(
            group_facts,
            key=lambda f: (f.get("session_time") or "", f.get("turn_id", 0))
        )
        
        # 使用并查集找出所有相似的事实对
        # 构建相似度矩阵
        n = len(group_facts_sorted)
        similar_pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                fact_i = group_facts_sorted[i]
                fact_j = group_facts_sorted[j]
                
                # 计算相似度
                sim = _compute_text_similarity(fact_i.get("name", ""), fact_j.get("name", ""))
                
                if sim >= similarity_threshold:
                    # 检查时间是否不同
                    time_i = _parse_session_time(fact_i.get("session_time", ""))
                    time_j = _parse_session_time(fact_j.get("session_time", ""))
                    
                    # 如果时间不同，或者内容不完全相同，标记为版本对
                    if (time_i != time_j) or (fact_i.get("name", "").lower() != fact_j.get("name", "").lower()):
                        similar_pairs.append((i, j, sim))
        
        if not similar_pairs:
            continue
        
        # 使用并查集合并相似的事实
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i, j, sim in similar_pairs:
            union(i, j)
        
        # 按连通分量分组
        components = defaultdict(list)
        for i in range(n):
            root = find(i)
            components[root].append(i)
        
        # 为每个连通分量（版本组）分配版本号
        for root, indices in components.items():
            if len(indices) < 2:
                continue
            
            version_id += 1
            
            # 按时间排序，分配版本号
            sorted_indices = sorted(indices, key=lambda i: (
                group_facts_sorted[i].get("session_time") or "",
                group_facts_sorted[i].get("turn_id", 0)
            ))
            
            for version_num, idx in enumerate(sorted_indices, 1):
                fact = group_facts_sorted[idx]
                fact["version_group"] = version_id
                fact["version_number"] = version_num
                fact["total_versions"] = len(sorted_indices)
                
                if _DEBUG:
                    time_str = fact.get("session_time", "")[:10] if fact.get("session_time") else "unknown"
                    logger.info(
                        f"[VersionDetect] group={version_id} v{version_num}/{len(sorted_indices)} "
                        f"time={time_str} text={fact.get('name', '')[:60]}..."
                    )
    
    return facts


class SimpleRetriever:
    """
    简化检索器（全量保留模式）
    
    核心能力：
    - 简单事实向量检索（主路径）
    - 时间信息关联（session_time）
    - 多跳扩展（通过三元组）
    - 原文兜底（TextUnit）
    - 动态版本检测
    """
    
    def __init__(self, ltss_instance, agent_name: str = "User"):
        if not ltss_instance:
            raise ValueError("LTSS 实例不能为空")
        self.ltss = ltss_instance
        self.agent_name = agent_name
        self.emb_model = get_embedding_model()
    
    def search(
        self,
        query: str,
        *,
        simple_fact_k: int = _DEFAULT_SIMPLE_FACT_K,
        textunit_k: int = _DEFAULT_TEXTUNIT_K,
        enable_multi_hop: bool = True,
        enable_version_detection: bool = True,
        enable_snippet_extraction: bool = _ENABLE_SNIPPET_EXTRACTION,
        current_turn: Optional[int] = None,
    ) -> str:
        """
        执行检索，返回格式化的上下文字符串
        
        Args:
            query: 用户问题
            simple_fact_k: 简单事实检索数量
            textunit_k: 原文检索数量
            enable_multi_hop: 是否启用多跳扩展
            enable_version_detection: 是否启用动态版本检测
            enable_snippet_extraction: 是否启用LLM提取关键片段（减少原文噪音）
            current_turn: 当前轮次（用于时间衰减，可选）
        
        Returns:
            格式化的上下文字符串，供 LLM 使用
        """
        if _DEBUG:
            logger.info(f"[SimpleRetriever] query={query[:50]}...")
        
        # 检测是否询问历史版本（仅用于调试日志）
        is_history_query = _should_include_history(query)
        if _DEBUG and is_history_query:
            logger.info(f"[SimpleRetriever] 检测到历史查询关键词")
        
        # 1. 简单事实向量检索（全量保留模式，不过滤历史版本）
        simple_facts = self._retrieve_simple_facts(query, k=simple_fact_k)
        if _DEBUG:
            logger.info(f"[SimpleRetriever] simple_facts={len(simple_facts)}")
        
        # 1.4 查询扩展（用LLM扩展查询关键词，弥补语义鸿沟）
        # 消融实验：ABLATION_NO_HYBRID_SEARCH=1 时禁用查询扩展
        if _ABLATION_NO_HYBRID_SEARCH:
            expanded_keywords = []
            if _DEBUG:
                logger.info(f"[SimpleRetriever][ABLATION] 查询扩展已禁用 (ABLATION_NO_HYBRID_SEARCH=1)")
        else:
            expanded_keywords = self._expand_query(query) if _ENABLE_QUERY_EXPANSION else []
        
        # 1.5 全文检索（BM25关键词匹配，独立输出，不参与综合评分竞争）
        # 消融实验：ABLATION_NO_HYBRID_SEARCH=1 时禁用全文检索
        if _ABLATION_NO_HYBRID_SEARCH:
            fulltext_facts = []
            if _DEBUG:
                logger.info(f"[SimpleRetriever][ABLATION] 全文检索已禁用 (ABLATION_NO_HYBRID_SEARCH=1)")
        else:
            fulltext_facts = self._fulltext_search_facts(query, expanded_keywords=expanded_keywords)
            if _DEBUG and fulltext_facts:
                logger.info(f"[SimpleRetriever] fulltext_facts={len(fulltext_facts)}")
        
        # 2. 多跳扩展（通过三元组关联）
        # 消融实验：ABLATION_NO_MULTI_HOP=1 时禁用多跳扩展
        if _ABLATION_NO_MULTI_HOP:
            if _DEBUG:
                logger.info(f"[SimpleRetriever][ABLATION] 多跳扩展已禁用 (ABLATION_NO_MULTI_HOP=1)")
        elif enable_multi_hop and simple_facts:
            multi_hop_facts = self._multi_hop_expand(simple_facts)
            if _DEBUG:
                logger.info(f"[SimpleRetriever] multi_hop_facts={len(multi_hop_facts)}")
            # 合并去重（只合并多跳结果，不合并全文检索结果）
            simple_facts = self._merge_facts(simple_facts, multi_hop_facts)
        
        # 3. 原文检索（加强兜底）
        # 消融实验：ABLATION_NO_RAW_FALLBACK=1 时禁用原文检索
        if _ABLATION_NO_RAW_FALLBACK:
            textunits = []
            if _DEBUG:
                logger.info(f"[SimpleRetriever][ABLATION] 原文检索已禁用 (ABLATION_NO_RAW_FALLBACK=1)")
        else:
            textunits = self._retrieve_textunits(query, k=textunit_k)
            if _DEBUG:
                logger.info(f"[SimpleRetriever] textunits={len(textunits)}")
        
        # 3.5 LLM提取关键片段（减少原文噪音）
        # 如果 TextUnit 已经是分块的（长度较短），跳过 LLM 提取
        if enable_snippet_extraction and textunits:
            # 检查是否需要片段提取：如果平均长度超过阈值才提取
            avg_len = sum(len(tu.get("content", "")) for tu in textunits) / len(textunits) if textunits else 0
            SNIPPET_EXTRACTION_THRESHOLD = 1500  # 平均长度超过1500字符才提取（从3000降低）
            
            if avg_len > SNIPPET_EXTRACTION_THRESHOLD:
                textunits = self._extract_key_snippets(textunits, query)
                if _DEBUG:
                    logger.info(f"[SimpleRetriever] snippet extraction completed (avg_len={avg_len:.0f})")
            elif _DEBUG:
                logger.info(f"[SimpleRetriever] skip snippet extraction (avg_len={avg_len:.0f} < {SNIPPET_EXTRACTION_THRESHOLD})")
        
        # 4. ✅ 动态版本检测（核心创新）
        if enable_version_detection and simple_facts:
            simple_facts = _detect_version_groups(simple_facts)
            version_count = sum(1 for f in simple_facts if f.get("version_group"))
            if _DEBUG and version_count > 0:
                logger.info(f"[SimpleRetriever] 检测到 {version_count} 个事实属于版本组")
            
            # 消融实验：ABLATION_NO_VERSION_HISTORY=1 时只保留最新版本
            if _ABLATION_NO_VERSION_HISTORY and version_count > 0:
                before_count = len(simple_facts)
                # 过滤：只保留每个版本组的最新版本（version_number == total_versions）
                # 或者不属于任何版本组的事实
                simple_facts = [
                    f for f in simple_facts
                    if not f.get("version_group") or f.get("version_number") == f.get("total_versions")
                ]
                after_count = len(simple_facts)
                if _DEBUG:
                    logger.info(f"[SimpleRetriever][ABLATION] 历史版本已过滤 (ABLATION_NO_VERSION_HISTORY=1): {before_count} -> {after_count}")
        
        # 5. ✅ 综合评分（只对向量检索+多跳结果评分，全文检索独立输出）
        # 消融实验：ABLATION_NO_SCORING=1 时禁用综合评分，只用向量相似度排序
        if _ABLATION_NO_SCORING:
            # 只按向量相似度排序，不考虑置信度、时间衰减、通道优先级
            simple_facts.sort(key=lambda f: -f.get("sim", 0))
            for f in simple_facts:
                f["score"] = f.get("sim", 0)  # score 直接等于 sim
            if _DEBUG:
                logger.info(f"[SimpleRetriever][ABLATION] 综合评分已禁用 (ABLATION_NO_SCORING=1)，只用向量相似度排序")
        else:
            # 获取当前会话时间（从 textunits 中提取，如果有的话）
            current_session_time = ""
            if textunits:
                current_session_time = textunits[0].get("session_time", "")
            simple_facts = _apply_comprehensive_scoring(
                simple_facts,
                current_turn=current_turn or 0,
                current_session_time=current_session_time,
                query=query,
            )
        
        # 6. 格式化输出（三部分独立输出）
        return self._format_output(simple_facts, fulltext_facts, textunits, query, expanded_keywords)
    
    def _retrieve_simple_facts(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        简单事实向量检索（全量保留模式）
        
        Args:
            query: 用户问题
            k: 检索数量
        
        返回字段：
        - name: 事实文本
        - sim: 向量相似度
        - turn_id: 轮次
        - session_time: 对话时间
        - event_time: 事件发生时间（LLM 计算）
        - source: 来源（user/assistant）
        - evidence_content: 原文片段
        """
        q_emb = self.emb_model.embed_query(query)
        
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $k, $embedding) YIELD node, score
        WHERE node._node_type = 'Fact'
        
        // 关联 ASSERTS 边获取元数据
        OPTIONAL MATCH (tu:TextUnit)-[a:ASSERTS]->(node)
        
        RETURN
            node.name AS name,
            score AS sim,
            a.turn_id AS turn_id,
            a.source_of_belief AS source,
            a.channel AS channel,
            a.confidence AS confidence,
            // ✅ 分别获取 event_timestamp 和 session_time
            a.event_timestamp AS event_time,
            coalesce(a.session_time, tu.session_time_iso) AS session_time,
            tu.content AS evidence_content
        ORDER BY score DESC
        """
        
        if _DEBUG:
            logger.info(f"[SimpleRetriever] query embedding dim={len(q_emb)}")
        
        try:
            rows = self.ltss.query_graph(cypher, {
                "index_name": _SIMPLE_FACT_INDEX,
                "k": int(k),
                "embedding": q_emb,
            }) or []
        except Exception as e:
            logger.error(f"[SimpleRetriever] simple_fact query failed: {e}")
            return []
        
        # 处理结果
        # ✅ 使用字典存储，key 是文本，value 是 fact 对象
        # 这样可以在去重时优先保留有 event_time 的 fact
        facts_dict = {}
        seen_raw = set()  # 用于跟踪原始 name，避免重复处理
        
        for r in rows:
            name = r.get("name")
            if not name or name in seen_raw:
                continue
            
            seen_raw.add(name)
            sim = _safe_float(r.get("sim"), 0.0)
            
            # ✅ 解析 JSON 格式的 facts
            # 格式：'{"text": "...", "source": "user/assistant", "event_time": "YYYY-MM-DD"}'
            fact_text = name
            fact_source = r.get("source") or "user"
            fact_event_time = r.get("event_time") or ""
            
            if name.startswith("{") and name.endswith("}"):
                try:
                    fact_obj = json.loads(name)
                    fact_text = fact_obj.get("text", name)
                    fact_source = fact_obj.get("source", fact_source)
                    # ✅ 优先使用 JSON 中的 event_time（LLM 计算的）
                    json_event_time = fact_obj.get("event_time", "")
                    if json_event_time and json_event_time.strip():
                        fact_event_time = json_event_time
                except json.JSONDecodeError:
                    pass  # 不是有效的 JSON，当作普通字符串处理
            
            # 调试：检查是否包含 ratio 关键词
            if _DEBUG and ("ratio" in fact_text.lower() or "1:10" in fact_text):
                logger.info(f"[SimpleRetriever][debug] FOUND ratio fact: sim={sim:.4f} name={fact_text[:80]}...")
            
            # 清理 event_time（可能是 "unknown" 或空）
            if fact_event_time.lower() in ["unknown", "none", "null", ""]:
                fact_event_time = ""
            
            # ✅ 使用解析后的 fact_text 作为去重 key
            text_key = fact_text.strip().lower()
            
            # ✅ 去重逻辑：优先保留有 event_time 的 fact
            if text_key in facts_dict:
                existing = facts_dict[text_key]
                existing_event_time = existing.get("event_time", "")
                
                # 如果现有的没有 event_time，但新的有，则替换
                if not existing_event_time and fact_event_time:
                    if _DEBUG:
                        logger.info(f"[SimpleRetriever][dedup] Replacing fact with event_time: {fact_text[:60]}...")
                    facts_dict[text_key] = {
                        "name": fact_text,
                        "sim": max(sim, existing.get("sim", 0)),  # 保留较高的相似度
                        "turn_id": _parse_turn_id(r.get("turn_id")),
                        "session_time": r.get("session_time") or "",
                        "event_time": fact_event_time,
                        "source": fact_source,
                        "channel": (r.get("channel") or "consolidated").lower(),
                        "confidence": _safe_float(r.get("confidence"), 1.0),
                        "evidence_content": r.get("evidence_content") or "",
                        "fact_type": "simple_fact",
                    }
                # 否则保留现有的（可能有更好的 event_time 或更高的相似度）
                continue
            
            facts_dict[text_key] = {
                "name": fact_text,
                "sim": sim,
                "turn_id": _parse_turn_id(r.get("turn_id")),
                "session_time": r.get("session_time") or "",
                "event_time": fact_event_time,
                "source": fact_source,
                "channel": (r.get("channel") or "consolidated").lower(),
                "confidence": _safe_float(r.get("confidence"), 1.0),
                "evidence_content": r.get("evidence_content") or "",
                "fact_type": "simple_fact",
            }
        
        # 转换为列表并按相似度排序
        facts = list(facts_dict.values())
        facts.sort(key=lambda f: -f.get("sim", 0))
        
        # 调试：输出前 10 条事实
        if _DEBUG and facts:
            logger.info(f"[SimpleRetriever][debug] top 10 facts:")
            for i, f in enumerate(facts[:10], 1):
                event_str = f" event_time={f['event_time']}" if f.get('event_time') else ""
                logger.info(f"  [{i}] sim={f['sim']:.4f}{event_str} {f['name'][:60]}...")
        
        return facts
    
    def _multi_hop_expand(self, seed_facts: List[Dict[str, Any]], limit: int = _DEFAULT_MULTI_HOP_LIMIT) -> List[Dict[str, Any]]:
        """
        多跳扩展：从种子事实出发，通过三元组关联找到更多相关事实
        
        Args:
            seed_facts: 种子事实列表
            limit: 最大返回数量
        
        路径：简单事实 ← HAS_SIMPLE_FACT ← Fact(三元组) → SUBJECT/OBJECT → Entity
              → 其他 Fact → HAS_SIMPLE_FACT → 其他简单事实
        
        ⚠️ 关键改进：
        1. 排除 User/Person 类型的实体，避免通过 User 把所有事实都关联起来
        2. 多跳分数 = 种子分数 × (衰减系数 ^ 跳跃步数)，步数越多衰减越大
        3. 使用 Cypher 的路径长度计算跳跃步数
        """
        if not seed_facts:
            return []
        
        # 提取种子事实的文本和分数
        seed_data = []
        for f in seed_facts[:10]:  # 只用前 10 个种子
            name = f.get("name")
            sim = f.get("sim", 0.75)
            if name:
                seed_data.append({"name": name, "sim": sim})
        
        if not seed_data:
            return []
        
        cypher = """
        // 从种子简单事实出发
        UNWIND $seed_data AS seed
        MATCH (seed_node:__Node__ {name: seed.name})
        WHERE seed_node._node_type = 'Fact'
        
        // 找到关联的三元组
        MATCH (triple:Fact)-[:HAS_SIMPLE_FACT]->(seed_node)
        
        // 找到三元组的主语/宾语实体
        // ⚠️ 关键改进：排除 User/Person 类型的实体，避免通过 User 把所有事实都关联起来
        MATCH (triple)-[:SUBJECT|OBJECT]->(entity)
        WHERE NOT (entity:Person OR entity:User OR toLower(entity.name) IN ['user', 'i', 'me', 'my'])
          AND NOT toLower(coalesce(entity.type, '')) IN ['person', 'user']
        
        // 找到实体关联的其他三元组（计算路径长度）
        MATCH path = (triple)-[:SUBJECT|OBJECT]->(entity)<-[:SUBJECT|OBJECT]-(other_triple:Fact)
        WHERE other_triple <> triple
        
        // 找到其他三元组关联的简单事实
        MATCH (other_triple)-[:HAS_SIMPLE_FACT]->(other_fact:__Node__)
        WHERE other_fact._node_type = 'Fact' AND other_fact.name <> seed.name
        
        // 获取元数据
        OPTIONAL MATCH (tu:TextUnit)-[a:ASSERTS]->(other_fact)
        
        // 计算跳跃步数：种子 → 三元组 → 实体 → 其他三元组 = 3 步
        // 简化为：通过实体的跳跃 = 1 跳
        WITH other_fact, seed, entity, a, tu, 
             1 AS hop_count
        
        RETURN DISTINCT
            other_fact.name AS name,
            seed.sim AS seed_sim,  // 返回种子分数
            seed.name AS seed_name,  // 返回种子名称（调试用）
            hop_count AS hop,  // 跳跃步数
            a.turn_id AS turn_id,
            a.source_of_belief AS source,
            coalesce(other_fact.event_timestamp, other_fact.session_time, a.session_time, tu.session_time_iso) AS session_time,
            a.channel AS channel,
            tu.content AS evidence_content,
            entity.name AS via_entity  // 记录通过哪个实体扩展的
        LIMIT $limit
        """
        
        try:
            rows = self.ltss.query_graph(cypher, {
                "seed_data": seed_data,
                "limit": int(limit),
            }) or []
        except Exception as e:
            logger.error(f"[SimpleRetriever] multi_hop query failed: {e}")
            return []
        
        # 多跳衰减系数：每跳保留 85% 的分数
        # 1 跳：0.85，2 跳：0.72，3 跳：0.61
        HOP_DECAY_FACTOR = 0.85
        
        facts = []
        for r in rows:
            name = r.get("name")
            if not name:
                continue
            
            # 计算多跳分数：种子分数 × (衰减系数 ^ 跳跃步数)
            seed_sim = _safe_float(r.get("seed_sim"), 0.75)
            hop_count = int(r.get("hop", 1))
            multi_hop_sim = seed_sim * (HOP_DECAY_FACTOR ** hop_count)
            
            via_entity = r.get("via_entity", "")
            seed_name = r.get("seed_name", "")
            
            if _DEBUG:
                logger.info(
                    f"[MultiHop] via '{via_entity}': {name[:60]}... "
                    f"(seed_sim={seed_sim:.4f}, hop={hop_count} → final_sim={multi_hop_sim:.4f})"
                )
            
            facts.append({
                "name": name,
                "sim": multi_hop_sim,  # 使用计算后的多跳分数
                "turn_id": _parse_turn_id(r.get("turn_id")),
                "session_time": r.get("session_time") or "",
                "source": r.get("source") or "user",
                "channel": (r.get("channel") or "consolidated").lower(),
                "evidence_content": r.get("evidence_content") or "",
                "hop": hop_count,
                "fact_type": "multi_hop",
                "via_entity": via_entity,  # 记录扩展路径
                "seed_name": seed_name,  # 记录种子名称（调试用）
                "seed_sim": seed_sim,  # 记录种子分数（调试用）
            })
        
        return facts
    
    def _expand_query(self, query: str) -> List[str]:
        """
        用 LLM 扩展查询关键词
        
        目标：弥补语义鸿沟，找到问题中隐含的相关概念
        例如：
        - "getting around Tokyo" → ["Suica", "TripIt", "transit", "subway", "train"]
        - "anxious about the trip" → ["nervous", "preparation", "planning", "organized"]
        
        Args:
            query: 用户问题
        
        Returns:
            扩展的关键词列表
        """
        if not _ENABLE_QUERY_EXPANSION:
            return []
        
        prompt = f"""You are searching a personal memory database. Your task is to generate keywords that will help find relevant information.

User's question: "{query}"

Generate 10-15 SPECIFIC keywords to search. Focus on:

1. **SPECIFIC PRODUCT/BRAND NAMES** (MOST IMPORTANT):
   - Shoes: Veja, Nike, Adidas, Allbirds, New Balance, Converse
   - Apps: TripIt, Spotify, Netflix, Headspace, Duolingo
   - Brands: Patagonia, IKEA, Samsung, Apple, Sony
   - Services: Uber, Airbnb, DoorDash, Instacart
   
2. **SPECIFIC ACTIVITY/HOBBY NAMES**:
   - Dances: Hoop Dance, Salsa, Ballet, Hip Hop, Breakdancing
   - Sports: Tennis, Golf, Yoga, Pilates, CrossFit
   - Crafts: Knitting, Pottery, Woodworking, Origami
   - Jobs: Transcriptionist, Freelancer, Developer, Designer

3. **KEY NOUNS from the question**:
   - Extract important nouns that might appear in stored facts
   - Include synonyms and related terms

4. **NUMBERS and MEASUREMENTS** (if relevant):
   - Percentages, amounts, durations, counts

CRITICAL RULES:
- Prioritize SPECIFIC names over generic words
- Include brand names, product names, activity names
- Think: "What specific things might have been mentioned in the conversation?"
- For "What brand of shoes" → include: Veja, Nike, Adidas, Allbirds, etc.
- For "What dance" → include: Hoop Dance, Salsa, Ballet, etc.
- For "What job" → include: Transcriptionist, Freelancer, etc.

Output ONLY a JSON array. Example: ["Veja", "Nike", "Adidas", "shoes", "sneakers", "brand", "sustainable"]

Keywords:"""
        
        try:
            llm = get_llm().bind(temperature=0.0)  # 使用低温，确保结果稳定
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 解析 JSON 数组
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                keywords = json.loads(match.group())
                if isinstance(keywords, list):
                    # 过滤无效关键词
                    keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
                    if _DEBUG:
                        logger.info(f"[QueryExpansion] expanded keywords: {keywords}")
                    return keywords[:12]  # 增加到12个关键词
        except Exception as e:
            if _DEBUG:
                logger.warning(f"[QueryExpansion] failed: {e}")
        
        return []
    
    def _fulltext_search_facts(self, query: str, k: int = _DEFAULT_FULLTEXT_K, expanded_keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        全文检索简单事实（BM25关键词匹配）
        
        使用 LLM 扩展的关键词 + 原始问题的关键词进行全文检索。
        
        Args:
            query: 用户问题
            k: 检索数量
            expanded_keywords: LLM扩展的关键词列表
        
        Returns:
            事实列表
        """
        # 收集所有关键词
        all_keywords = []
        
        # 1. 添加 LLM 扩展的关键词
        if expanded_keywords:
            all_keywords.extend(expanded_keywords)
        
        # 2. 从原始问题中提取关键词（名词、专有名词）
        query_keywords = self._extract_query_keywords(query)
        all_keywords.extend(query_keywords)
        
        if not all_keywords:
            return []
        
        # 过滤掉太长的短语和太通用的词
        GENERIC_WORDS = {
            # 通用动词/形容词
            "navigate", "commute", "travel", "organized", "planned", "prepared",
            "good", "great", "nice", "best", "better", "important", "useful",
            # 通用名词
            "tips", "advice", "help", "information", "guide", "recommendation",
            # 疑问词
            "what", "which", "who", "where", "when", "how", "why",
            # 常见动词
            "did", "does", "was", "were", "have", "has", "had", "been",
            "mention", "mentioned", "recommend", "recommended", "suggest", "suggested",
            # 代词
            "you", "your", "i", "my", "me", "we", "our", "they", "their",
        }
        
        keywords = []
        seen = set()
        for kw in all_keywords:
            kw_clean = kw.strip()
            if not kw_clean:
                continue
            kw_lower = kw_clean.lower()
            # 跳过已添加的
            if kw_lower in seen:
                continue
            # 跳过太长的短语（超过3个词）
            word_count = len(kw_clean.split())
            if word_count > 3:
                continue
            # 跳过太短的词（少于2个字符）
            if len(kw_clean) < 2:
                continue
            # 跳过太通用的词
            if kw_lower in GENERIC_WORDS:
                if _DEBUG:
                    logger.info(f"[FulltextSearch] skip generic word: {kw_clean}")
                continue
            seen.add(kw_lower)
            keywords.append(kw_clean)
        
        if not keywords:
            return []
        
        # 构建全文搜索查询（OR连接关键词）
        search_terms = [f'"{kw}"' if ' ' in kw else kw for kw in keywords]  # 短语用引号包裹
        search_query = " OR ".join(search_terms)
        
        if _DEBUG:
            logger.info(f"[FulltextSearch] keywords (expanded + query): {keywords}")
            logger.info(f"[FulltextSearch] search_query: {search_query}")
        
        cypher = """
        CALL db.index.fulltext.queryNodes('fact_name_fulltext_index', $search_query) 
        YIELD node, score
        WHERE node._node_type = 'Fact'
        
        // 获取元数据
        OPTIONAL MATCH (tu:TextUnit)-[a:ASSERTS]->(node)
        
        RETURN
            node.name AS name,
            score AS fulltext_score,
            a.turn_id AS turn_id,
            a.source_of_belief AS source,
            a.channel AS channel,
            a.confidence AS confidence,
            coalesce(node.event_timestamp, node.session_time, a.session_time, tu.session_time_iso) AS session_time,
            tu.content AS evidence_content
        ORDER BY score DESC
        LIMIT $k
        """
        
        try:
            rows = self.ltss.query_graph(cypher, {
                "search_query": search_query,
                "k": int(k),
            }) or []
        except Exception as e:
            logger.error(f"[SimpleRetriever] fulltext search failed: {e}")
            return []
        
        facts = []
        seen = set()
        for r in rows:
            name = r.get("name")
            if not name or name in seen:
                continue
            
            seen.add(name)
            # 全文检索结果独立输出，不需要设置固定分数
            # 保留原始的 fulltext_score 供参考
            
            facts.append({
                "name": name,
                "fulltext_score": _safe_float(r.get("fulltext_score"), 0.0),
                "turn_id": _parse_turn_id(r.get("turn_id")),
                "session_time": r.get("session_time") or "",
                "source": r.get("source") or "user",
                "channel": (r.get("channel") or "consolidated").lower(),
                "confidence": _safe_float(r.get("confidence"), 1.0),
                "evidence_content": r.get("evidence_content") or "",
            })
        
        if _DEBUG and facts:
            logger.info(f"[SimpleRetriever] fulltext search found {len(facts)} facts")
            for i, f in enumerate(facts[:5], 1):
                logger.info(f"  [{i}] score={f['fulltext_score']:.4f} {f['name'][:60]}...")
        
        return facts
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """
        从原始问题中提取关键词
        
        提取规则：
        1. 提取大写开头的词（可能是专有名词）
        2. 提取引号内的内容
        3. 提取数字和百分比
        4. 提取较长的词（可能是具体名词）
        
        Args:
            query: 用户问题
        
        Returns:
            关键词列表
        """
        keywords = []
        
        # 1. 提取引号内的内容（最重要，通常是具体名称）
        quoted = re.findall(r'["\']([^"\']+)["\']', query)
        keywords.extend(quoted)
        
        # 2. 提取大写开头的词（可能是专有名词，排除句首）
        words = query.split()
        for i, word in enumerate(words):
            # 清理标点
            word_clean = re.sub(r'[^\w\s-]', '', word)
            if not word_clean:
                continue
            # 检查是否大写开头（排除句首）
            if i > 0 and word_clean[0].isupper() and len(word_clean) > 2:
                keywords.append(word_clean)
        
        # 3. 提取数字和百分比
        numbers = re.findall(r'\d+%?', query)
        keywords.extend(numbers)
        
        # 4. 提取较长的词（5个字符以上，可能是具体名词）
        for word in words:
            word_clean = re.sub(r'[^\w\s-]', '', word).lower()
            if len(word_clean) >= 5 and word_clean not in keywords:
                keywords.append(word_clean)
        
        if _DEBUG and keywords:
            logger.info(f"[QueryKeywords] extracted from query: {keywords}")
        
        return keywords
    
    def _retrieve_textunits(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        TextUnit 原文检索（加强兜底）
        
        简单事实可能丢失信息，原文是最完整的信息来源
        """
        q_emb = self.emb_model.embed_query(query)
        
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $k, $embedding) YIELD node, score
        WHERE node:TextUnit
        
        RETURN
            node.name AS name,
            score AS sim,
            node.turn_id AS turn_id,
            node.channel AS channel,
            node.session_time_iso AS session_time,
            node.content AS content
        ORDER BY score DESC
        """
        
        try:
            rows = self.ltss.query_graph(cypher, {
                "index_name": _TEXTUNIT_INDEX,
                "k": int(k),
                "embedding": q_emb,
            }) or []
        except Exception as e:
            logger.error(f"[SimpleRetriever] textunit query failed: {e}")
            return []
        
        textunits = []
        for r in rows:
            content = r.get("content")
            if not content or len(content) < 20:
                continue
            textunits.append({
                "name": r.get("name") or "",
                "sim": _safe_float(r.get("sim"), 0.0),
                "turn_id": _parse_turn_id(r.get("turn_id")),
                "session_time": r.get("session_time") or "",
                "channel": (r.get("channel") or "raw").lower(),
                "content": content,
            })
        
        return textunits
    
    # TextUnit 片段提取的分批配置
    _SNIPPET_BATCH_MAX_CHARS = 8000  # 每批最多 8000 字符
    
    def _extract_key_snippets(self, textunits: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        用LLM从TextUnit原文中提取与问题最相关的片段（分批并行处理）
        
        目标：减少原文噪音，提高信息密度
        - 原文可能很长（几十到上百行），包含大量无关信息
        - 按长度动态分批，每批不超过 8000 字符
        - 多批并行调用 LLM，加速处理
        
        Args:
            textunits: 原文列表
            query: 用户问题
        
        Returns:
            更新后的textunits（content字段替换为提取的片段）
        """
        if not textunits:
            return textunits
        
        # 按长度动态分批
        batches = self._batch_textunits_by_length(textunits)
        
        if _DEBUG:
            logger.info(f"[SnippetExtraction] split {len(textunits)} textunits into {len(batches)} batches")
        
        # 定义单个 batch 的处理函数
        def process_single_batch(batch_idx: int, batch: List[Dict]) -> List[Tuple[int, str, int]]:
            """
            处理单个 batch，返回提取结果列表
            每个结果是 (original_idx, extracted_content, snippet_count)
            """
            batch_indices = [item["original_index"] for item in batch]
            batch_textunits = [item["textunit"] for item in batch]
            
            if _DEBUG:
                total_chars = sum(len(tu.get("content", "")) for tu in batch_textunits)
                logger.info(f"[SnippetExtraction] batch {batch_idx + 1}/{len(batches)}: {len(batch_textunits)} textunits, {total_chars} chars")
            
            # 构建提示词（只包含当前批次的 textunits）
            prompt = self._build_snippet_extraction_prompt(batch_textunits, query)
            
            results = []
            
            # 重试逻辑：最多重试 2 次
            max_retries = 2
            for retry in range(max_retries + 1):
                try:
                    llm = get_llm().bind(temperature=0.0)  # 低温确保结果稳定
                    response_text = invoke_json(llm, prompt)
                    
                    # 解析JSON响应
                    response_text = response_text.strip()
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.startswith("```"):
                        response_text = response_text[3:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                    
                    response = json.loads(response_text)
                    extractions = response.get("extractions", [])
                    
                    # 收集提取结果
                    for item in extractions:
                        batch_local_idx = item.get("index")
                        snippets = item.get("snippets", [])
                        
                        if batch_local_idx is not None and 0 <= batch_local_idx < len(batch_indices) and snippets:
                            original_idx = batch_indices[batch_local_idx]
                            extracted_content = "\n".join(snippets)
                            results.append((original_idx, extracted_content, len(snippets)))
                    
                    break  # 成功，跳出重试循环
                    
                except Exception as e:
                    if retry < max_retries:
                        logger.warning(f"[SnippetExtraction] batch {batch_idx + 1} retry {retry + 1}/{max_retries}: {e}")
                        continue
                    else:
                        logger.warning(f"[SnippetExtraction] batch {batch_idx + 1} failed after {max_retries} retries: {e}")
            
            return results
        
        # 并行处理所有 batches
        all_results = []
        workers = min(_SNIPPET_EXTRACTION_PARALLEL_WORKERS, len(batches))
        
        if workers > 1 and len(batches) > 1:
            # 并行处理
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            if _DEBUG:
                logger.info(f"[SnippetExtraction] parallel processing with {workers} workers")
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_single_batch, idx, batch): idx
                    for idx, batch in enumerate(batches)
                }
                
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        all_results.extend(results)
                    except Exception as e:
                        batch_idx = futures[future]
                        logger.error(f"[SnippetExtraction] batch {batch_idx} worker failed: {e}")
        else:
            # 串行处理（单个 batch 或 workers=1）
            for idx, batch in enumerate(batches):
                results = process_single_batch(idx, batch)
                all_results.extend(results)
        
        # 更新 textunits
        for original_idx, extracted_content, snippet_count in all_results:
            textunits[original_idx]["content"] = extracted_content
            textunits[original_idx]["extracted"] = True
            
            if _DEBUG:
                logger.info(f"[SnippetExtraction] TextUnit {original_idx}: extracted {snippet_count} snippets")
        
        return textunits
    
    def _batch_textunits_by_length(self, textunits: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        按长度动态分批 TextUnit
        
        规则：
        - 每批累计字符数不超过 _SNIPPET_BATCH_MAX_CHARS (8000)
        - 如果单个 TextUnit 超过限制，单独成一批
        
        Returns:
            批次列表，每批包含 {"original_index": int, "textunit": dict}
        """
        batches = []
        current_batch = []
        current_len = 0
        
        for i, tu in enumerate(textunits):
            tu_len = len(tu.get("content", ""))
            
            # 如果当前批次加上这个 TextUnit 会超过限制
            if current_len + tu_len > self._SNIPPET_BATCH_MAX_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_len = 0
            
            current_batch.append({"original_index": i, "textunit": tu})
            current_len += tu_len
        
        # 添加最后一批
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _build_snippet_extraction_prompt(self, textunits: List[Dict[str, Any]], query: str) -> str:
        """
        构建片段提取提示词
        
        要求：
        1. 提取最相关的5-10个片段
        2. 保持原文不变（不要改写、总结、翻译）
        3. 如果没有相关信息，返回空数组
        """
        # 格式化textunits
        textunit_blocks = []
        for i, tu in enumerate(textunits):
            content = tu.get("content", "").strip()
            turn_id = tu.get("turn_id", "?")
            session_time = tu.get("session_time", "")
            
            header = f"[TextUnit {i}] Turn {turn_id}"
            if session_time:
                header += f" ({session_time})"
            
            textunit_blocks.append(f"{header}\n{content}")
        
        textunits_text = "\n\n---\n\n".join(textunit_blocks)
        
        prompt = f"""You are a snippet extraction assistant for a personal memory system. Your task is to extract the most relevant snippets from conversation history to help answer a user's question.

**CRITICAL RULES:**
1. Extract EXACT text snippets from the original text - DO NOT modify, paraphrase, or summarize
2. Extract 5-10 most relevant snippets per TextUnit (if relevant information exists)
3. Each snippet should be 1-3 sentences that directly relate to the question
4. If a TextUnit has NO relevant information, return empty snippets array for that index
5. Preserve the original wording EXACTLY - copy-paste from the source text

**PRIORITY ORDER for extraction (MOST IMPORTANT FIRST):**
1. USER'S PERSONAL INFORMATION: What the user owns, has, uses, or has done (e.g., "I have a Suica card", "I downloaded TripIt app", "I'm staying at Park Hyatt")
2. USER'S PREFERENCES: What the user likes, prefers, or wants (e.g., "I prefer walking", "I'm nervous about...")
3. USER'S PLANS: What the user is planning to do (e.g., "I'm heading to Tokyo", "I want to visit...")
4. SPECIFIC TOOLS/APPS/CARDS: Any mention of specific tools, apps, cards, or resources the user has
5. RELEVANT ADVICE: Advice or recommendations that directly address the user's situation

**IMPORTANT:** When the question asks about tips or recommendations, prioritize extracting:
- What resources/tools the USER already has (cards, apps, reservations)
- What the USER has already prepared or done
- Specific advice given to the USER about their situation

**Question:**
{query}

**Conversation Texts:**
{textunits_text}

**Output Format:**
Return a JSON object with this structure:
{{
  "extractions": [
    {{
      "index": 0,
      "snippets": [
        "Exact sentence 1 from the text",
        "Exact sentence 2 from the text",
        "Exact sentence 3 from the text"
      ]
    }},
    {{
      "index": 1,
      "snippets": []  // Empty if no relevant information
    }}
  ]
}}

**Example:**
Question: "I'm anxious about getting around Tokyo. Any tips?"
Text: "USER: I'm heading to Tokyo soon. ASSISTANT: Great! USER: I just got a Suica card for transit. I also downloaded the TripIt app to stay organized. ASSISTANT: That's excellent preparation! The Suica card will make navigating Tokyo's trains very easy. USER: I'm a bit nervous about the trip. ASSISTANT: Don't worry, Tokyo's public transportation is very tourist-friendly."
Output:
{{
  "extractions": [
    {{
      "index": 0,
      "snippets": [
        "I just got a Suica card for transit.",
        "I also downloaded the TripIt app to stay organized.",
        "The Suica card will make navigating Tokyo's trains very easy.",
        "Tokyo's public transportation is very tourist-friendly."
      ]
    }}
  ]
}}

Now extract snippets for the given question and texts:"""
        
        return prompt
    
    def _merge_facts(self, facts1: List[Dict], facts2: List[Dict]) -> List[Dict]:
        """
        合并两个事实列表，去重，保留分数更高的版本
        
        改进：当同一事实在两个列表中都存在时，保留 sim 分数更高的版本
        """
        # 用字典存储，key 是事实名称，value 是事实对象
        fact_map = {}
        
        # 先添加 facts1
        for f in facts1:
            name = f["name"]
            fact_map[name] = f
        
        # 再添加 facts2，如果分数更高则替换
        for f in facts2:
            name = f["name"]
            if name not in fact_map:
                fact_map[name] = f
            else:
                # 比较 sim 分数，保留更高的
                existing_sim = fact_map[name].get("sim", 0)
                new_sim = f.get("sim", 0)
                if new_sim > existing_sim:
                    fact_map[name] = f
        
        merged = list(fact_map.values())
        # 按综合评分排序（如果有 score 字段），否则按 sim 排序
        merged.sort(key=lambda f: -(f.get("score") or f.get("sim", 0)))
        return merged
    
    def _format_output(
        self,
        simple_facts: List[Dict],
        fulltext_facts: List[Dict],
        textunits: List[Dict],
        query: str,
        expanded_keywords: List[str] = None,
        sort_by_time: bool = True,
        max_facts: int = 80,
        max_fulltext: int = 20,
    ) -> str:
        """
        格式化输出，供 LLM 使用
        
        三部分独立输出（优先级从高到低）：
        1. 关键词检索结果（全文检索，最相关，优先展示）
        2. 语义检索结果（向量检索 + 多跳扩展）
        3. 原文兜底（TextUnit）
        
        ⚠️ 关键改进：把 KEYWORD MATCHED FACTS 放在最前面
        原因：LLM 有位置偏好，更容易注意到前面的信息
        
        Args:
            simple_facts: 简单事实列表（向量检索 + 多跳扩展，已按综合评分排序）
            fulltext_facts: 全文检索结果（独立输出）
            textunits: 原文列表
            query: 用户问题
            expanded_keywords: LLM 扩展的关键词列表
            sort_by_time: 是否按时间排序输出（默认 True）
            max_facts: 语义检索最大输出数量
            max_fulltext: 关键词检索最大输出数量
        """
        lines = []
        
        # ============================================================
        # 第一部分：关键词检索结果（全文检索，优先展示）
        # ============================================================
        lines.append("=== KEYWORD MATCHED FACTS (关键词检索) ===")
        
        # 强制输出调试信息
        if _DEBUG:
            logger.info(f"[_format_output] fulltext_facts count: {len(fulltext_facts) if fulltext_facts else 0}")
            if fulltext_facts:
                for i, f in enumerate(fulltext_facts[:3], 1):
                    logger.info(f"[_format_output] fulltext[{i}]: {f.get('name', '')[:80]}")
        
        if expanded_keywords:
            lines.append(f"匹配关键词: {', '.join(expanded_keywords)}")
            lines.append("")
        
        if not fulltext_facts:
            lines.append("无关键词匹配结果。")
        else:
            # 去重，排除已在语义检索中出现的
            seen_in_semantic = {f["name"].strip().lower() for f in simple_facts} if simple_facts else set()
            seen_texts = set()
            filtered_fulltext = []
            
            if _DEBUG:
                logger.info(f"[FulltextFilter] total fulltext facts: {len(fulltext_facts)}")
                logger.info(f"[FulltextFilter] seen_in_semantic size: {len(seen_in_semantic)}")
            
            for f in fulltext_facts:
                text_key = f["name"].strip().lower()
                # 排除已在语义检索中出现的
                if text_key in seen_in_semantic:
                    if _DEBUG:
                        logger.info(f"[FulltextFilter] SKIP (in semantic): {f['name'][:80]}")
                    continue
                if text_key in seen_texts:
                    if _DEBUG:
                        logger.info(f"[FulltextFilter] SKIP (duplicate): {f['name'][:80]}")
                    continue
                seen_texts.add(text_key)
                filtered_fulltext.append(f)
                if _DEBUG:
                    logger.info(f"[FulltextFilter] KEEP: {f['name'][:80]}")
                if len(filtered_fulltext) >= max_fulltext:
                    break
            
            if _DEBUG:
                logger.info(f"[FulltextFilter] filtered_fulltext size: {len(filtered_fulltext)}")
            
            if not filtered_fulltext:
                if _DEBUG:
                    logger.info(f"[FulltextFilter] No facts after filtering, adding placeholder")
                lines.append("（关键词匹配结果已包含在语义检索中）")
            else:
                try:
                    if _DEBUG:
                        logger.info(f"[FulltextFilter] Formatting {len(filtered_fulltext)} facts for output")
                    
                    # 按时间排序
                    if sort_by_time and filtered_fulltext:
                        filtered_fulltext = sorted(
                            filtered_fulltext,
                            key=lambda f: (f.get("session_time") or "", f.get("turn_id", 0))
                        )
                    
                    if _DEBUG:
                        logger.info(f"[FulltextFilter] Starting to append facts to lines")
                    
                    for fact_idx, f in enumerate(filtered_fulltext, 1):
                        line = f"[Match {fact_idx}] {f['name']}"
                        lines.append(line)
                        if _DEBUG and fact_idx <= 2:
                            logger.info(f"[FulltextFilter] Appended: {line[:80]}")
                        
                        # 时间信息
                        time_parts = []
                        if f.get("session_time"):
                            time_parts.append(f"session_time={f['session_time']}")
                        time_parts.append(f"turn_id={f.get('turn_id', 0)}")
                        lines.append(f"  时间: {', '.join(time_parts)}")
                        
                        # 来源信息
                        lines.append(f"  来源: source={f.get('source', 'user')}")
                        
                        lines.append("")
                    
                    if _DEBUG:
                        logger.info(f"[FulltextFilter] Finished appending {len(filtered_fulltext)} facts")
                
                except Exception as e:
                    logger.error(f"[FulltextFilter] ERROR formatting fulltext facts: {e}", exc_info=True)
                    lines.append(f"（关键词检索结果格式化失败：{e}）")
        
        # ============================================================
        # 第二部分：语义检索结果（向量检索 + 多跳扩展）
        # ============================================================
        lines.append("")
        lines.append("=== LONG-TERM MEMORY FACTS (语义检索) ===")
        
        if not simple_facts:
            lines.append("无相关长期记忆。")
        else:
            # 去重，保留 top-K
            seen_texts = set()
            filtered_facts = []
            for f in simple_facts:
                text_key = f["name"].strip().lower()
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                filtered_facts.append(f)
                if len(filtered_facts) >= max_facts:
                    break
            
            # 按时间排序
            if sort_by_time and filtered_facts:
                filtered_facts = sorted(
                    filtered_facts,
                    key=lambda f: (f.get("session_time") or "", f.get("turn_id", 0))
                )
                if _DEBUG:
                    logger.info(f"[SimpleRetriever] facts sorted by time, first={filtered_facts[0].get('session_time')}, last={filtered_facts[-1].get('session_time')}")
            
            # 输出格式化
            for fact_idx, f in enumerate(filtered_facts, 1):
                # 构建事实标签
                labels = []
                version_group = f.get("version_group")
                version_number = f.get("version_number")
                total_versions = f.get("total_versions")
                
                if version_group and version_number:
                    labels.append(f"VERSION {version_number}/{total_versions}")
                
                # ✅ 使用 LLM 计算的 event_time（如果有）
                event_time = f.get("event_time", "")
                if event_time:
                    labels.append(f"event_time={event_time}")
                
                fact_text = f['name']
                label_str = " ".join(f"[{l}]" for l in labels)
                if label_str:
                    lines.append(f"[Fact {fact_idx}] {label_str} {fact_text}")
                else:
                    lines.append(f"[Fact {fact_idx}] {fact_text}")
                
                # 时间信息
                time_parts = []
                if f.get("session_time"):
                    time_parts.append(f"session_time={f['session_time']}")
                time_parts.append(f"turn_id={f.get('turn_id', 0)}")
                lines.append(f"  时间: {', '.join(time_parts)}")
                
                # 来源信息
                source_parts = [f"source={f.get('source', 'user')}"]
                if f.get("score") is not None:
                    source_parts.append(f"score={f.get('score', 0.0):.4f}")
                if f.get("hop"):
                    source_parts.append(f"hop={f.get('hop')}")
                lines.append(f"  来源: {', '.join(source_parts)}")
                
                lines.append("")
        
        # ============================================================
        # 第三部分：原文兜底
        # ============================================================
        lines.append("")
        lines.append("=== ORIGINAL TEXT (原文兜底) ===")
        
        if not textunits:
            lines.append("无相关原文。")
        else:
            seen_content = set()
            for tu in textunits:
                content = tu.get("content", "").strip()
                content_key = content[:100].lower()
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)
                
                turn_id = tu.get("turn_id", "?")
                session_time = tu.get("session_time", "")
                
                header = f"[Turn {turn_id}]"
                if session_time:
                    header += f" ({session_time})"
                lines.append(f"{header}")
                lines.append(content)
                lines.append("")
        
        result = "\n".join(lines)
        
        if _DEBUG:
            logger.info(f"[_format_output] Total lines: {len(lines)}")
            logger.info(f"[_format_output] Result length: {len(result)}")
            # 检查是否包含关键部分
            has_keyword = "KEYWORD MATCHED FACTS" in result or "关键词检索" in result
            logger.info(f"[_format_output] Contains KEYWORD section: {has_keyword}")
            if not has_keyword:
                logger.warning(f"[_format_output] WARNING: KEYWORD section missing!")
                # 打印 lines 的最后几个元素，看看是什么
                logger.info(f"[_format_output] Last 5 lines:")
                for i, line in enumerate(lines[-5:], len(lines)-4):
                    logger.info(f"  [{i}] {line[:80]}")
        
        return result


# ============================================================
# 便捷函数
# ============================================================

def simple_search(
    ltss_instance,
    query: str,
    agent_name: str = "User",
    **kwargs
) -> str:
    """
    便捷函数：执行简化检索
    
    Args:
        ltss_instance: LTSS 实例
        query: 用户问题
        agent_name: 智能体名称
        **kwargs: 传递给 SimpleRetriever.search 的其他参数
    
    Returns:
        格式化的上下文字符串
    """
    retriever = SimpleRetriever(ltss_instance, agent_name)
    return retriever.search(query, **kwargs)
