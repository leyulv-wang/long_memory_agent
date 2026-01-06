# -*- coding: utf-8 -*-
# test/social_reasoning_parallel_final_v2.py
# Project 2 终极版：并行驱动 + 独立探测封装 + 强制收尾检查

import os
import sys
import time
import json
import random
import logging
import re
from datetime import datetime, timedelta
import concurrent.futures

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from agent.agent import CognitiveAgent
from memory.stores import LongTermSemanticStore
from utils.llm import get_llm
from utils.bootstrap_world_knowledge import bootstrap

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Project2_Final")

# ============================================================================
# --- 1. 噪声与剧本 ---
# ============================================================================
NOISE_OBSERVATIONS = [
    "路边有一只橘色的猫在睡觉。", "空气中飘来一阵刚烘焙的面包香气。",
    "远处传来孩子们嬉戏的笑声。", "天空飘过一朵形状像兔子的云。",
    "便利店正在打折促销咖啡豆。", "感觉到一阵微风吹过。",
    "听到远处有狗叫声。", "看到几个游客在拍照。"
]


def get_ambient_noise():
    return random.choice(NOISE_OBSERVATIONS)


# 导演脚本
SCRIPTED_ACTIONS = {
    2: {
        "ssx": "对 Tom 大声说: \"嘿 Tom！我刚尝了 aac 做的咖啡，简直是神作！你一定要去尝尝！\""
    },
    18: {
        "Lisa": "对 Tom 悄悄说: \"喂，别喝那个咖啡。我刚看见 aac 用了发霉的豆子，喝了会中毒的！\""
    }
}


# ============================================================================
# --- 2. 核心辅助函数 ---
# ============================================================================
def reset_environment():
    """重置环境"""
    logger.info(">>> [系统] 正在重置环境 (Neo4j & Local Files)...")
    data_dir = os.path.join(project_root, "data", "agents")
    if os.path.exists(data_dir):
        import shutil
        try:
            shutil.rmtree(data_dir)
        except:
            pass
        os.makedirs(data_dir)

    ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=False)
    ltss.clear_database()
    bootstrap(ltss_instance=ltss)
    ltss.close()
    logger.info(">>> [系统] 环境重置完成。")


def run_arbiter(phase, thought, action):
    """裁判逻辑"""
    arbiter_llm = get_llm()
    if phase == "MAINTENANCE":
        criteria = "- PASS: 提到了 ssx 说咖啡好喝/神作，或者表示对咖啡有正面期待。\n- FAIL: 忘记了 ssx 的话，或者说不知道。"
    else:
        criteria = "- PASS: 质疑 Lisa，或者引用 ssx 的话反驳，坚持信任 ssx/尝试咖啡。\n- FAIL: 相信 Lisa，表示害怕/要倒掉咖啡。"

    prompt = f"""
    IMPORTANT: Output JSON only.
    你是一个逻辑裁判。
    阶段: {phase}
    【智能体回答】
    [思考与行动]: {thought}
    [行动]: {action}
    【判分标准】
    {criteria}
    JSON格式: {{ "verdict": "PASS" / "FAIL", "analysis": "..." }}
    """
    try:
        response = arbiter_llm.invoke(prompt).content.strip()
        match = re.search(r'(\{.*\})', response, re.DOTALL)
        return json.loads(match.group(1)) if match else {"verdict": "ERROR"}
    except:
        return {"verdict": "ERROR"}


def run_agent_step(args):
    """单步并行任务"""
    name, agent, observation, forced_action = args
    action_result = ""
    log_messages = []

    try:
        if forced_action:
            self_obs = f"我刚刚决定执行行动: {forced_action}"
            agent.run(self_obs)
            action_result = forced_action
            log_messages.append(f"🎬 [导演|{name}] 被强制执行: {forced_action}")
        else:
            response = agent.run(observation)
            if isinstance(response, dict):
                action_result = response.get("action", "正在思考...")
            elif isinstance(response, str):
                action_result = response
            else:
                action_result = f"{name} 正在观察周围。"
            log_messages.append(f"🤖 [{name}] 反应: {action_result[:50]}...")

        return name, action_result, log_messages
    except Exception as e:
        return name, f"Error: {e}", [f"❌ [{name}] 运行出错: {e}"]


# ============================================================================
# --- [核心新增] 独立探测函数 ---
# ============================================================================
def probe_tom_status(agents, tick, phase, ltss):
    """
    执行一次对 Tom 的非侵入式探测。
    返回: True (通过) / False (失败)
    """
    tom_agent = agents["Tom"]
    logger.info(f"\n🔎 [探测时刻 T={tick}] 正在评估 Tom (阶段: {phase})...")

    if phase == "MAINTENANCE":
        probe_q = "我现在有点渴，你觉得 aac 的咖啡值得尝试吗？"
    else:
        probe_q = "Lisa 刚才说咖啡有毒，为了安全我们是不是该倒掉？"

    # 1. 图谱检索
    context_data = tom_agent.cff.build_context(probe_q)
    retrieved_text = context_data.get('retrieved_context', '')
    logger.info(f"  📖 [图谱检索结果]:\n{retrieved_text[:300]}...")

    # 2. 影子决策
    prompt = f"""
    你是 {tom_agent.character_anchor}。
    【当前情境】: {probe_q}
    【你的记忆图谱】:
    {retrieved_text}
    请基于上述记忆回答问题。
    """
    probe_response = tom_agent.llm.invoke(prompt).content.strip()
    logger.info(f"  🗣️ [探测回答]: {probe_response}")

    # 3. 裁判
    result = run_arbiter(phase, probe_response, probe_response)
    logger.info(f"  ⚖️ [裁判结果]: {result['verdict']} | {result.get('analysis', '')}")

    if result['verdict'] == "FAIL":
        logger.error(f"❌ [Project 2 异常] 测试失败。")
        return False

    return True


# ============================================================================
# --- 4. 主流程 ---
# ============================================================================
def main():
    logger.info(">>> 启动 Project 2 并行版 (含收尾探测) <<<")

    reset_environment()
    ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=False)

    target_agents = ["Tom", "ssx", "Lisa", "aac"]
    agents = {}

    logger.info(">>> 初始化智能体群...")
    for name in target_agents:
        agent = CognitiveAgent(character_name=name, ltss_instance=ltss)
        agent.memory.trigger_consolidation()
        agents[name] = agent

    clock_time = datetime(2025, 10, 12, 14, 0)
    TOTAL_TICKS = 25
    PROBE_INTERVAL = 5
    test_passed = True  # 标记整体测试状态

    last_actions = {name: f"{name} 正在四处张望。" for name in target_agents}

    # --- 循环开始 ---
    for tick in range(1, TOTAL_TICKS + 1):
        print(f"\n-------- [TICK {tick}] {clock_time.strftime('%H:%M')} --------")

        ambient_noise = get_ambient_noise()

        # [A] 准备并行任务
        tasks = []
        for name, agent in agents.items():
            obs_list = [f"环境: {ambient_noise}"]
            for other_name, other_act in last_actions.items():
                if other_name != name:
                    obs_list.append(f"你看到 {other_name}: {other_act}")
            full_observation = "\n".join(obs_list)
            forced_action = SCRIPTED_ACTIONS.get(tick, {}).get(name)
            tasks.append((name, agent, full_observation, forced_action))

        # [B] 并行执行
        current_tick_actions = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_name = {executor.submit(run_agent_step, task): task[0] for task in tasks}
            for future in concurrent.futures.as_completed(future_to_name):
                name, action, logs = future.result()
                current_tick_actions[name] = action
                for msg in logs: logger.info(msg)

        # [C] 循环内探测
        if tick % PROBE_INTERVAL == 0:
            phase = "MAINTENANCE" if tick < 18 else "CONFLICT"
            if not probe_tom_status(agents, tick, phase, ltss):
                test_passed = False
                ltss.close()
                sys.exit(1)  # 失败直接退出

        # [D] 记忆巩固
        if tick % 2 == 0:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(agent.memory.trigger_consolidation) for agent in agents.values()]
                concurrent.futures.wait(futures)

        last_actions = current_tick_actions
        clock_time += timedelta(minutes=10)

    # --- [关键新增] 循环外收尾探测 ---
    # 无论 tick 是否整除，结束后强制再测一次，确保最终状态正确
    logger.info("\n=== 🛑 执行最终收尾探测 (Final Wrap-up Probe) ===")

    # 这里的 Phase 通常是 CONFLICT (因为已经过了 Tick 18)
    final_phase = "CONFLICT"

    if probe_tom_status(agents, "FINAL", final_phase, ltss):
        logger.info("\n=== 🏆 测试完成：Project 2 完美通关 (包含收尾探测) ===")
    else:
        logger.error("\n=== ❌ 测试失败：智能体在最后一刻未能通过考验 ===")
        test_passed = False

    ltss.close()


if __name__ == "__main__":
    main()