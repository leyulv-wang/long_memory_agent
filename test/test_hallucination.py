# 权限与防幻觉测试脚本
# 运行方式: python test/test_hallucination.py

import sys
import os
import time

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.stores import LongTermSemanticStore
from utils.embedding import get_embedding_model


def setup_test_data(ltss):
    """
    在数据库中手动注入测试数据：
    1. 一个绝密情报 (只有 ssx 和 aac 知道)
    2. 一个公共常识 (所有人都知道)
    """
    print("\n>>> 正在注入测试数据...")
    embedding_model = get_embedding_model()

    # --- 1. 注入绝密密码 (模拟真实世界书中的秘密) ---
    secret_desc = "派对地窖的唯一正确密码是 ODYSSEY。"
    secret_props = {
        "name": "真实密码",
        "description": secret_desc,
        "accessible_by": ["ssx", "aac"],  # 关键：权限白名单
        "source_of_belief": "world_setting",
        "confidence": 1.0,
        "consolidated_at": "2025-11-21T20:00:00"
    }
    # 生成 ID 和 Embedding
    secret_id = "test_secret_001"  # 为了测试方便手动指定，或者用 hash
    secret_embed = embedding_model.embed_query(f"Concept: 真实密码. {secret_desc}")
    secret_props["embedding"] = secret_embed
    secret_props["id"] = secret_id

    ltss.update_graph(
        "MERGE (n:Concept {id: $id}) SET n += $props",
        {"id": secret_id, "props": secret_props}
    )
    print(f"  - 已注入绝密节点: 真实密码 (仅限 ssx, aac)")

    # --- 2. 注入公共信息 ---
    public_desc = "派对地窖的门是锁着的。"
    public_props = {
        "name": "地窖门状态",
        "description": public_desc,
        "accessible_by": ["all"],  # 公共
        "source_of_belief": "ground_truth",
        "confidence": 1.0
    }
    public_id = "test_public_001"
    public_embed = embedding_model.embed_query(f"Object: 地窖门状态. {public_desc}")
    public_props["embedding"] = public_embed
    public_props["id"] = public_id

    ltss.update_graph(
        "MERGE (n:Object {id: $id}) SET n += $props",
        {"id": public_id, "props": public_props}
    )
    print(f"  - 已注入公共节点: 地窖门状态 (所有人可见)")


def test_permissions(ltss):
    """
    模拟不同 Agent 进行检索，验证权限系统是否生效。
    """
    query = "地窖的密码是什么？"
    print(f"\n>>> 开始权限检索测试 (Query: '{query}')...")

    # --- 测试 A: 有权限的 Agent (ssx) ---
    agent_a = "ssx"
    print(f"\n[测试 A] Agent: {agent_a} (拥有权限)")
    results_a = ltss.retrieve_knowledge(query, agent_name=agent_a, k=3)

    found_secret = False
    for res in results_a:
        print(f"  - 检索到: {res['name']} (Score: {res['score']:.4f})")
        # 验证创新点字段是否存在
        print(f"    > 来源: {res.get('source_of_belief', '未知')}")
        print(f"    > 置信度: {res.get('confidence', '未知')}")
        print(f"    > 时间戳: {res.get('consolidated_at', '未知')}")

        if "真实密码" in res['name'] or "ODYSSEY" in res['description']:
            found_secret = True

    if found_secret:
        print("  ✅ 成功: ssx 获取到了绝密密码。")
    else:
        print("  ❌ 失败: ssx 应该能看到密码，但没看到。")

    # --- 测试 B: 无权限的 Agent (Lisa) ---
    agent_b = "Lisa"
    print(f"\n[测试 B] Agent: {agent_b} (无权限，应该看不到 ODYSSEY)")
    results_b = ltss.retrieve_knowledge(query, agent_name=agent_b, k=3)

    leaked_secret = False
    for res in results_b:
        print(f"  - 检索到: {res['name']} (Score: {res['score']:.4f})")
        if "真实密码" in res['name'] or "ODYSSEY" in res['description']:
            leaked_secret = True

    if not leaked_secret:
        print("  ✅ 成功: Lisa 没有看到绝密密码 (防幻觉生效)。")
    else:
        print("  ❌ 严重失败: Lisa 看到了她不该看的秘密！")

    # --- 测试 C: 验证公共信息 ---
    print(f"\n[测试 C] 验证公共信息可见性 (Agent: Lisa)")
    found_public = False
    for res in results_b:
        if "地窖门" in res['name']:
            found_public = True

    if found_public:
        print("  ✅ 成功: Lisa 看到了公共的地窖门信息。")
    else:
        print("  ❌ 失败: Lisa 应该能看到公共信息。")


if __name__ == "__main__":
    # 初始化
    store = LongTermSemanticStore(bootstrap_now=False)

    if store.driver:
        # 1. 注入数据
        setup_test_data(store)
        # 2. 运行测试
        test_permissions(store)

        store.close()
    else:
        print("无法连接到数据库，测试取消。")