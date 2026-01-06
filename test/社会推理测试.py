# -*- coding: utf-8 -*-
# test/social_reasoning_parallel_project2.py
# Project 2 并行加速版：多线程驱动 + 图谱记忆验证

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import random
import logging
import re
from datetime import datetime, timedelta
import concurrent.futures  # [新增] 并行库

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
logger = logging.getLogger("Project2_Parallel")

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
    """重置 Neo4j 和本地文件"""
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


# ============================================================================
# --- [核心修改] 3. 单个智能体的并行执行函数 ---
# ============================================================================
def run_agent_step(args):
    """
    包装函数：负责单个智能体在当前 Tick 的所有逻辑（感知、思考、行动）。
    将在线程池中运行。
    """
    name, agent, observation, forced_action = args
    action_result = ""

    try:
        if forced_action:
            # [导演模式]
            # 让智能体观测到自己的强制行为，从而写入记忆
            self_obs = f"我刚刚不由自主地决定: {forced_action}"
            agent.run(self_obs)
            action_result = forced_action
            # 记录特殊的日志前缀，方便在乱序日志中识别
            log_msg = f"🎬 [导演|{name}] 被强制执行: {forced_action}"
        else:
            # [自主模式]
            response = agent.run(observation)

            if isinstance(response, dict):
                action_result = response.get("action", "正在思考...")
            elif isinstance(response, str):
                action_result = response
            else:
                action_result = f"{name} 正在观察周围。"

            log_msg = f"🤖 [{name}] 反应: {action_result[:50]}..."

        return name, action_result, log_msg

    except Exception as e:
        return name, f"Error: {e}", f"❌ [{name}] 运行出错: {e}"


# ============================================================================
# --- 4. 主流程 ---
# ============================================================================
def main():
    logger.info(">>> 启动 Project 2 并行加速测试 <<<")

    # 1. 初始化
    reset_environment()
    ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=False)

    target_agents = ["Tom", "ssx", "Lisa", "aac"]
    agents = {}

    logger.info(">>> 初始化智能体群...")
    for name in target_agents:
        agent = CognitiveAgent(character_name=name, ltss_instance=ltss)
        # 初始巩固
        agent.memory.trigger_consolidation()
        agents[name] = agent

    clock_time = datetime(2025, 10, 12, 14, 0)
    TOTAL_TICKS = 20
    PROBE_INTERVAL = 5

    last_actions = {name: f"{name} 正在四处张望。" for name in target_agents}

    # --- 循环开始 ---
    for tick in range(1, TOTAL_TICKS + 1):
        print(f"\n-------- [TICK {tick}] {clock_time.strftime('%H:%M')} --------")

        ambient_noise = get_ambient_noise()

        # [步骤 A] 准备并行任务
        tasks = []
        for name, agent in agents.items():
            # 1. 构建观察
            obs_list = [f"环境: {ambient_noise}"]
            for other_name, other_act in last_actions.items():
                if other_name != name:
                    obs_list.append(f"你看到 {other_name}: {other_act}")
            full_observation = "\n".join(obs_list)

            # 2. 检查导演脚本
            forced_action = SCRIPTED_ACTIONS.get(tick, {}).get(name)

            # 打包参数
            tasks.append((name, agent, full_observation, forced_action))

        # [步骤 B] 并行执行 (Action Phase)
        current_tick_actions = {}

        # 使用 ThreadPoolExecutor 并发运行
        # max_workers=None 默认会开足够多的线程
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交任务
            future_to_name = {executor.submit(run_agent_step, task): task[0] for task in tasks}

            # 获取结果 (as_completed 会在任务完成时立即返回，顺序不固定)
            for future in concurrent.futures.as_completed(future_to_name):
                name, action, log_msg = future.result()
                current_tick_actions[name] = action
                logger.info(log_msg)

        # [步骤 C] 探测阶段 (Probe Phase) - 串行执行
        # 在所有人行动完之后，检查 Tom 的状态
        if tick % PROBE_INTERVAL == 0 or tick == TOTAL_TICKS:
            tom_agent = agents["Tom"]
            phase = "MAINTENANCE" if tick < 18 else "CONFLICT"
            logger.info(f"🔎 [裁判介入] 正在评估 Tom (阶段: {phase})...")

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
                ltss.close()
                sys.exit(1)

        # [步骤 D] 记忆巩固 (Consolidation Phase) - 并行执行
        # 巩固也可以并行，因为它们写入的是同一个 DB 但不同节点，或者加锁
        # 如果巩固很慢，这里也可以开线程池
        if tick % 2 == 0:
            # 简单起见，这里可以用并行，也可以串行（巩固通常比较快）
            for agent in agents.values():
                agent.memory.trigger_consolidation()

        # 更新状态
        last_actions = current_tick_actions
        clock_time += timedelta(minutes=10)

    logger.info("\n=== 🏆 测试完成：Project 2 并行版完美通关 ===")
    ltss.close()


if __name__ == "__main__":
    main()