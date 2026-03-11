# -*- coding: utf-8 -*-
"""
测试新版两阶段提取器（consolidated_extractor.py）
"""

import os
import sys
import logging

# 添加项目根目录到 sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 启用调试
os.environ["DEBUG_CONSOLIDATED_EXTRACTOR"] = "1"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_extract_entities():
    """测试实体提取"""
    from utils.consolidated_extractor import extract_entities, _INSTRUCTOR_AVAILABLE
    
    if not _INSTRUCTOR_AVAILABLE:
        logger.warning("Instructor not available, skipping test")
        return
    
    text = """
    USER: I visited the Science Museum today with my colleague David. 
    The Space Exploration exhibition was amazing!
    """
    
    entities = extract_entities(text, session_time_iso="2024-01-15T10:00:00")
    
    logger.info(f"\n=== 实体提取结果 ===")
    for e in entities:
        logger.info(f"  - {e.name} ({e.type}): {e.description}")
    
    # 验证
    entity_names = {e.name for e in entities}
    entity_types = {e.name: e.type for e in entities}
    
    assert "User" in entity_names, "Should extract User"
    assert "Science Museum" in entity_names, "Should extract Science Museum"
    assert entity_types.get("Science Museum") == "Organization", "Science Museum should be Organization"
    
    logger.info("✅ 实体提取测试通过")


def test_extract_relationships():
    """测试关系提取"""
    from utils.consolidated_extractor import extract_entities, extract_relationships, _INSTRUCTOR_AVAILABLE
    
    if not _INSTRUCTOR_AVAILABLE:
        logger.warning("Instructor not available, skipping test")
        return
    
    text = """
    USER: I visited the Science Museum today with my colleague David. 
    The Space Exploration exhibition was amazing!
    """
    
    # 先提取实体
    entities = extract_entities(text, session_time_iso="2024-01-15T10:00:00")
    
    # 再提取关系
    relationships = extract_relationships(text, entities, session_time_iso="2024-01-15T10:00:00")
    
    logger.info(f"\n=== 关系提取结果 ===")
    for r in relationships:
        logger.info(f"  - {r.source} -[{r.type}]-> {r.target}: {r.description}")
    
    # 验证
    rel_types = {(r.source, r.type, r.target) for r in relationships}
    
    # 应该有 User VISITED Science Museum
    visited_museum = any(
        r.source == "User" and r.type == "VISITED" and "Museum" in r.target
        for r in relationships
    )
    assert visited_museum, "Should have User VISITED Science Museum"
    
    # 不应该有 User VISITED exhibition（Event）
    visited_event = any(
        r.type == "VISITED" and "exhibition" in r.target.lower()
        for r in relationships
    )
    assert not visited_event, "Should NOT have User VISITED exhibition (Event)"
    
    logger.info("✅ 关系提取测试通过")


def test_extract_knowledge_graph():
    """测试完整的知识图谱提取（含验证补漏）"""
    from utils.consolidated_extractor import extract_knowledge_graph, _INSTRUCTOR_AVAILABLE
    
    if not _INSTRUCTOR_AVAILABLE:
        logger.warning("Instructor not available, skipping test")
        return
    
    # 这个文本故意包含多个博物馆，测试是否能全部提取
    text = """
    USER: I visited the Metropolitan Museum of Art yesterday. 
    I also went to the Natural History Museum last week.
    The Dinosaur Fossils exhibition was incredible!
    Then I visited the Modern Art Museum and the Museum of Contemporary Art.
    """
    
    result = extract_knowledge_graph(text, session_time_iso="2024-01-15T10:00:00", turn_id=1)
    
    logger.info(f"\n=== 知识图谱提取结果（含验证补漏）===")
    logger.info(f"实体数: {len(result['entities'])}")
    logger.info(f"关系数: {len(result['relationships'])}")
    
    for e in result['entities']:
        logger.info(f"  实体: {e.name} ({e.type})")
    
    for r in result['relationships']:
        logger.info(f"  关系: {r.source} -[{r.type}]-> {r.target}")
    
    # 验证
    entity_names = {e.name for e in result['entities']}
    entity_names_lower = {e.name.lower() for e in result['entities']}
    
    # 应该提取到 4 个博物馆
    expected_museums = [
        "Metropolitan Museum of Art",
        "Natural History Museum", 
        "Modern Art Museum",
        "Museum of Contemporary Art",
    ]
    
    found_museums = []
    for museum in expected_museums:
        if museum in entity_names or museum.lower() in entity_names_lower:
            found_museums.append(museum)
    
    logger.info(f"\n期望博物馆: {expected_museums}")
    logger.info(f"找到博物馆: {found_museums}")
    
    # 验证 VISITED 关系的 target 都是 Organization/Location
    for r in result['relationships']:
        if r.type == "VISITED":
            target_entity = next((e for e in result['entities'] if e.name == r.target or e.name.lower() == r.target.lower()), None)
            if target_entity:
                assert target_entity.type in ("Organization", "Location"), \
                    f"VISITED target should be Organization/Location, got {target_entity.type}"
    
    # 至少应该找到 3 个博物馆（允许一定误差）
    assert len(found_museums) >= 3, f"Should find at least 3 museums, got {len(found_museums)}: {found_museums}"
    
    logger.info("✅ 知识图谱提取测试通过")


def test_consolidate_session_v2():
    """测试简化版巩固函数"""
    from utils.consolidated_extractor import consolidate_session_v2, _INSTRUCTOR_AVAILABLE
    
    if not _INSTRUCTOR_AVAILABLE:
        logger.warning("Instructor not available, skipping test")
        return
    
    session_turns = [
        {"role": "user", "content": "I visited the Science Museum today."},
        {"role": "assistant", "content": "That sounds great! What did you see?"},
        {"role": "user", "content": "I saw the Space Exploration exhibition. It was amazing!"},
    ]
    
    result = consolidate_session_v2(
        session_turns=session_turns,
        virtual_time="TURN_1",
        session_time_iso="2024-01-15T10:00:00",
        include_assistant=True,
    )
    
    logger.info(f"\n=== 巩固结果 ===")
    logger.info(f"节点数: {len(result['nodes'])}")
    logger.info(f"关系数: {len(result['relationships'])}")
    
    for n in result['nodes']:
        logger.info(f"  节点: {n['name']} ({n['label']})")
    
    for r in result['relationships']:
        logger.info(f"  关系: {r['source_node_name']} -[{r['type']}]-> {r['target_node_name']}")
    
    # 验证
    assert result['nodes'], "Should have nodes"
    assert result['relationships'], "Should have relationships"
    assert result['kg_extraction'] is not None, "Should have kg_extraction"
    
    logger.info("✅ 巩固函数测试通过")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("测试新版两阶段提取器")
    logger.info("=" * 60)
    
    try:
        test_extract_entities()
        test_extract_relationships()
        test_extract_knowledge_graph()
        test_consolidate_session_v2()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 所有测试通过！")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
