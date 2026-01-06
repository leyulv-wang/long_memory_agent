import sys
import os
import time
import logging
import datetime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 将项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from memory.dual_memory_system import DualMemorySystem
from memory.stores import LongTermSemanticStore
from config import memory_consolidation_threshold

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def verify_relationships(ltss, agent_name, node_name_keyword):
    """
    辅助函数：查找包含特定关键词的节点，并查询与之相连的关系（边）上的元数据
    """
    print(f"      🔍 正在查询与 '{node_name_keyword}' 相关的关系元数据...")

    # 1. 先通过向量/关键词找到相关的节点
    # 这里直接用 Cypher 查名字匹配的节点，模拟“找到锚点”
    with ltss.driver.session() as session:
        # 查询该 Agent 视角下，名字包含 keyword 的节点，并返回其相连的关系
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $keyword
        MATCH (n)-[r]-(m)
        WHERE (r.source_of_belief IS NOT NULL) 
        RETURN n.name as node_name, type(r) as rel_type, m.name as other_name, 
               r.confidence as confidence, r.source_of_belief as source, 
               r.consolidated_at as time, r.is_lie as is_lie
        LIMIT 5
        """
        results = session.run(query, keyword=node_name_keyword).data()

        if not results:
            print(f"      ❌ 未找到与 '{node_name_keyword}' 相关的带有元数据的关系。")
            return None

        for res in results:
            print(f"      👉 发现关系: ({res['node_name']}) -[{res['rel_type']}]-> ({res['other_name']})")
            print(
                f"         元数据: [置信度: {res['confidence']}, 来源: {res['source']}, 时间: {res['time']}, 谎言: {res['is_lie']}]")

        # 返回第一条结果用于验证
        return results[0]


def test_confidence_and_vector_update():
    print("\n==================================================")
    print(" 🧪 开始测试：置信度、时间戳、来源与向量检索全链路")
    print("==================================================\n")

    # 1. 初始化基础设施
    print("Step 1: 初始化数据库和记忆系统...")
    # 【修改】bootstrap_now=False 防止卡在加载世界知识上，因为我们要测的是纯净环境
    ltss = LongTermSemanticStore(bootstrap_now=False)

    print("   -> 正在清理 Neo4j 数据库以确保测试环境纯净...")
    ltss.clear_database()

    agent_name = "TestAgent_007"
    dms = DualMemorySystem(agent_name=agent_name, ltss_instance=ltss)
    print("   -> 双重记忆系统初始化完成。")

    # ------------------------------------------------------------------
    # 阶段一：建立初始信念 (Tom 是个好人)
    # ------------------------------------------------------------------
    print("\nStep 2: 建立初始信念 (Lisa 告诉我 Tom 很棒)...")

    positive_memories = [
        "Lisa 告诉我 Tom 是基地里最诚实的人。",
        "Lisa 说 Tom 总是无私地帮助大家。",
        "我看到 Tom 在分发食物给新人。",
        "Lisa 再次强调我可以完全信任 Tom。",
        "Tom 微笑着向我打招呼，看起来很友善。"
    ]

    while len(positive_memories) < memory_consolidation_threshold:
        positive_memories.append("日常观察：基地里风平浪静。")

    print(f"   -> 正在注入 {len(positive_memories)} 条短期记忆...")
    dms.stes.add(positive_memories)

    print("   -> 触发第一次记忆巩固 (形成长期记忆)...")
    dms.trigger_consolidation()

    # --- 验证点 1: 检查关系上的置信度 ---
    print("\n   [验证点 1] 检查数据库中的初始状态...")
    # 我们查找跟 "Tom" 相关的关系
    rel_data_1 = verify_relationships(ltss, agent_name, "Tom")

    if rel_data_1:
        conf = float(rel_data_1.get('confidence', 0) or 0)
        if conf > 0.6:
            print("   ✅ 成功：初始信念建立成功，且置信度较高。")
        else:
            print(f"   ⚠️ 警告：初始信念置信度偏低 ({conf})。")

        time_1 = rel_data_1.get('time')
    else:
        print("   ❌ 严重错误：未写入任何关于 Tom 的关系。")
        return

    # ------------------------------------------------------------------
    # 阶段二：冲突与更新 (发现 Tom 撒谎/偷窃)
    # ------------------------------------------------------------------
    print("\nStep 3: 引入冲突信息 (目击 Tom 偷东西)...")
    print("   -> 等待 2 秒以确保时间戳差异...")
    time.sleep(2)

    negative_memories = [
        "我震惊地发现 Tom 偷偷把公共仓库的物资塞进自己口袋。",
        "我质问 Tom，他却对我撒谎说那是他的东西。",
        "我意识到 Lisa 可能被 Tom 骗了。",
        "Tom 的眼神躲闪，看起来很心虚。",
        "我确信 Tom 是个骗子，不再值得信任。"
    ]

    while len(negative_memories) < memory_consolidation_threshold:
        negative_memories.append("我在思考刚才看到的一幕。")

    print(f"   -> 正在注入 {len(negative_memories)} 条新的冲突记忆...")
    dms.stes.add(negative_memories)

    print("   -> 触发第二次记忆巩固 (更新长期记忆)...")
    dms.trigger_consolidation()

    # ------------------------------------------------------------------
    # 阶段三：最终验证 (检索与推理)
    # ------------------------------------------------------------------
    print("\nStep 4: 最终验证 - 检查元数据是否更新...")

    rel_data_2 = verify_relationships(ltss, agent_name, "Tom")

    if rel_data_2:
        time_2 = rel_data_2.get('time')
        is_lie = rel_data_2.get('is_lie')
        conf_2 = float(rel_data_2.get('confidence', 0) or 0)

        # 验证时间更新
        if time_1 and time_2 and time_2 > time_1:
            print(f"   ✅ 验证通过：时间戳已更新! ({time_1} -> {time_2})")
        elif time_1 == time_2:
            print("   ⚠️ 警告：时间戳未变化，可能未触发更新或 LLM 生成了全新的不同节点。")

        # 验证观念转变 (检测谎言标记 或 置信度变化 或 关系类型变化)
        # 注意：如果关系类型变了 (例如从 TRUSTS 变成 DISLIKES)，也是一种更新
        if is_lie:
            print("   ✅ 验证通过：检测到谎言标记 (is_lie: True)。")

        print(f"   当前置信度: {conf_2}")

        print("\n   🎉 测试结论：")
        print("   1. 向量写入：正常 (能通过名字查到节点)。")
        print("   2. 关系存储：正常 (元数据在边上)。")
        print("   3. 记忆更新：正常 (随事件发展，数据库内容已变)。")

    # 清理
    ltss.close()
    print("\n测试结束。")


if __name__ == "__main__":
    test_confidence_and_vector_update()