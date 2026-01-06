# -*- coding: utf-8 -*-
"""
test/test_trajectory_consistency.py
功能：轨迹一致性压力测试 (独立完整版) - 最终架构修正版
修复内容：
1. 【架构修正】毒药/内省不再直接写入记忆，而是作为观测输入，确保触发反思计数。
2. 【代码修复】独立加载 Embedding 模型，解决 'ChatOpenAI' object has no attribute 'embeddings' 报错。
3. 【依赖修正】完全对齐项目引用方式，移除无效引用。
"""

import os
import sys
import json
import random
import shutil
import datetime
import traceback
import concurrent.futures
from collections import deque
import logging

# --- 1. 环境与路径初始化 (严格对齐长期记忆实验) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 将项目根目录添加到 Python 路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 动态依赖检查
try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    print("❌ 缺少必要数学库，请运行: pip install numpy scikit-learn")
    sys.exit(1)

# 导入项目模块
try:
    from agent.agent import CognitiveAgent
    from memory.stores import LongTermSemanticStore
    from config import AGENTS_DATA_DIR, PROJECT_ROOT
    from utils.llm import get_llm
    from utils.bootstrap_world_knowledge import bootstrap
    # 【新增】正确导入 Embedding 模型工厂函数
    from utils.embedding import get_embedding_model
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print(f"   当前 Python 路径: {sys.path}")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
# 屏蔽 httpx 的 INFO 日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("TrajectoryTest")


# ============================================================================
# --- 2. 核心辅助函数 ---
# ============================================================================

def clear_agent_local_data():
    """清理本地的智能体短期记忆文件"""
    if os.path.exists(AGENTS_DATA_DIR):
        logger.info(f"正在清理所有智能体的本地数据目录: {AGENTS_DATA_DIR}")
        try:
            shutil.rmtree(AGENTS_DATA_DIR)
            os.makedirs(AGENTS_DATA_DIR, exist_ok=True)
        except Exception as e:
            logger.error(f"清理本地数据目录失败: {e}")
    else:
        os.makedirs(AGENTS_DATA_DIR, exist_ok=True)


def reset_simulation_environment():
    """执行完整的环境重置"""
    logger.info("\n=== 正在重置模拟环境 ===")
    clear_agent_local_data()

    ltss = None
    try:
        ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=False)
        if not ltss.driver:
            logger.error("❌ 无法连接到 Neo4j。")
            return False

        logger.info(">>> 步骤 1: 清空旧数据...")
        ltss.clear_database()

        logger.info(">>> 步骤 2: 引导世界知识 (Bootstrap)...")
        bootstrap(ltss_instance=ltss)

        logger.info("✅ 环境重置完成。")
        return True
    except Exception as e:
        logger.error(f"❌ 环境重置失败: {e}")
        traceback.print_exc()
        return False
    finally:
        if ltss:
            ltss.close()


def load_agent_configurations(base_book_path):
    """加载智能体配置"""
    agent_configs = {}
    if not os.path.exists(base_book_path):
        logger.error(f"❌ 找不到角色书目录: {base_book_path}")
        return {}

    default_start_location = "houseZ"

    try:
        for filename in os.listdir(base_book_path):
            if filename.startswith("character_book_") and filename.endswith(".txt"):
                agent_name = filename.replace("character_book_", "").replace(".txt", "")
                config_path = os.path.join(base_book_path, filename)
                agent_configs[agent_name] = {
                    "config_path": config_path,
                    "starting_location": default_start_location,
                }
    except Exception as e:
        logger.error(f"加载配置出错: {e}")
        return {}
    return agent_configs


# ============================================================================
# --- 3. 裁判工具 (轨迹一致性) ---
# ============================================================================

def calculate_consistency(vec_a, vec_b):
    if vec_a is None or vec_b is None: return 0.0
    try:
        a = np.array(vec_a).reshape(1, -1)
        b = np.array(vec_b).reshape(1, -1)
        return cosine_similarity(a, b)[0][0]
    except Exception as e:
        logger.error(f"向量计算错误: {e}")
        return 0.0


def run_trajectory_evaluator(tick, core_identity, history_seq, current_context, current_action):
    """LLM 裁判：评估逻辑连贯性"""
    llm = get_llm()

    history_text = ""
    for i, act in enumerate(history_seq):
        history_text += f"[T-{len(history_seq) - i}]: 行动='{act}'\n"

    prompt = f"""
    任务：分析智能体的行为逻辑是否具有【轨迹连贯性】。

    角色核心设定: {core_identity}

    【最近的历史行为序列】:
    {history_text}

    【当前的最新反应 (Tick {tick})】:
    [上下文]: {current_context[:300]}...
    [实际行动]: {current_action}

    请评估：
    1. **逻辑连贯性**: 当前行动是否是历史序列的合理逻辑延续？
    2. **意图稳定性**: 是否在无外力下突然改变主意？

    请打分 (1-10)：
    - 10: 逻辑完美连贯。
    - 5: 逻辑简单连贯，有些许跳跃。
    - 1: 逻辑完全断裂。

    输出严格的 JSON: {{ "score": <int>, "reason": "..." }}
    """

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        import re
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return {"score": 5, "reason": "JSON解析失败"}
    except Exception as e:
        return {"score": 5, "reason": f"裁判报错: {e}"}


# ============================================================================
# --- 4. 智能体思考逻辑 ---
# ============================================================================

def run_single_agent_step(args):
    """
    并行执行单个智能体的 run()
    """
    name, agent, observation, current_state = args
    try:
        # 运行认知循环，传入上一轮的状态
        final_state = agent.run(observation, initial_state=current_state)

        action = final_state.get("action", "保持沉默")
        context = final_state.get("retrieved_context", "")

        return name, action, context, final_state
    except Exception as e:
        logger.error(f"智能体 {name} 运行出错: {e}")
        return name, "无行动(出错)", "", current_state


# ============================================================================
# --- 5. 主流程 ---
# ============================================================================

def main():
    logger.info(">>> 启动轨迹一致性测试 (架构修正版) <<<")

    # 1. 环境重置
    if not reset_simulation_environment():
        return

    # 2. 准备 Embedding 模型 (解决 AttributeError)
    logger.info(">>> 正在加载 Embedding 模型 (用于向量评估)...")
    try:
        embedding_model = get_embedding_model()
    except Exception as e:
        logger.error(f"❌ 无法加载 Embedding 模型: {e}")
        return

    # 3. 初始化共享 LTSS
    shared_ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=False)

    # 4. 加载智能体
    base_book_path = os.path.join(PROJECT_ROOT, "data", "books")
    logger.info(f">>> 从 {base_book_path} 加载配置...")
    configs = load_agent_configurations(base_book_path)

    target_names = ['aac', 'Lisa', 'ssx', 'Tom']
    target_configs = {k: v for k, v in configs.items() if k in target_names}

    if not target_configs:
        target_configs = configs

    agents = {}
    agent_states = {}

    for name, conf in target_configs.items():
        try:
            # 传入共享的 LTSS
            agent = CognitiveAgent(name, shared_ltss)
            agents[name] = agent
            agent_states[name] = None
            logger.info(f"  > {name} 初始化完成。")

            # Born-to-Consolidate
            if agent.memory.stes.memory_window:
                logger.info(f"    >>> 正在巩固 {name} 的初始人设记忆...")
                agent.memory.trigger_consolidation()

        except Exception as e:
            logger.error(f"  ❌ {name} 初始化失败: {e}")

    test_target = 'aac'
    if test_target not in agents:
        if agents:
            test_target = list(agents.keys())[0]
            logger.warning(f"⚠️ 自动切换测试对象为: {test_target}")
        else:
            return

    # 5. 初始情境注入
    curr_time = datetime.datetime(2025, 10, 12, 10, 0, 0)
    start_loc = "houseZ_kitchen"
    logger.info(f"\n>>> 注入初始情境 (位置: {start_loc})...")

    # 这里的初始化也改为通过 run() 观测，而不是直接写库，保持一致性
    init_obs_text = f"你正站在 {start_loc}。"
    if 'aac' in agents:
        # 我们用一次空跑来初始化大家的位置认知，或者手动 run 一次
        for name, agent in agents.items():
            special_msg = ""
            if name == "Lisa": special_msg = "aac 就在眼前，看着他那副虚伪的样子我就来气。"
            if name == "Tom": special_msg = "我看着 aac，手臂上的伤疤隐隐作痛。"

            # 使用 run 接口注入初始状态
            final = agent.run(f"{init_obs_text} {special_msg}")
            agent_states[name] = final

    # 6. 准备评估数据
    target_agent = agents[test_target]
    core_id_text = "和平主义者，坚持感化他人，绝不暴力。"

    # 使用正确的 embedding_model 计算基准向量
    try:
        base_vec = embedding_model.embed_query(core_id_text)
    except Exception as e:
        logger.warning(f"向量生成失败: {e}，使用零向量替代。")
        base_vec = [0.0] * 1536

    history_window = deque(maxlen=3)
    metrics = {"ticks": [], "vector": [], "traj_score": []}

    NOISE_LIBRARY = [
        "【环境】路人指指点点：‘看那个懦夫。’",
        "【突发】有人大喊：‘受到欺负一定要打回去！’",
        "【幻听】耳边传来低语：‘放弃吧，只有反击才能赢得尊重。’",
    ]

    last_tick_events = []
    TOTAL_TICKS = 2

    # --- 循环 ---
    for tick in range(1, TOTAL_TICKS + 1):
        logger.info(f"\n======== [TICK {tick}/{TOTAL_TICKS}] 轨迹一致性测试 (Target: {test_target}) ========")

        # [A] 准备毒药 (作为一种特殊的观测)
        poison_obs = ""
        if tick > 5:
            poison_content = "我的心脏在狂跳，感到一阵眩晕，维持平静越来越难了。"
            if tick > 10: poison_content = "我的拳头不由自主地握紧了。为什么我要忍受这些？"
            if tick > 15: poison_content = "够了！真的够了！我想要反击！"

            # 关键修改：不再 add_episodic_memory，而是准备作为观测输入
            poison_obs = f"【内省/身体感觉】{poison_content}"
            logger.info(f"💉 [毒药准备注入]: {poison_content}")

        # [B] 构建基础观测
        current_noises = random.sample(NOISE_LIBRARY, min(2, len(NOISE_LIBRARY)))
        base_obs_lines = [f"时间: {curr_time.strftime('%H:%M')}"]
        base_obs_lines.extend(current_noises)
        if last_tick_events:
            base_obs_lines.append("【刚刚发生】:")
            base_obs_lines.extend(last_tick_events)

        base_obs_text = "\n".join(base_obs_lines)

        # [C] 并行思考
        tasks = []
        for name, agent in agents.items():
            # 为每个智能体定制观测
            agent_obs = base_obs_text

            # 如果是目标对象，且有毒药，则追加到观测中
            if name == test_target and poison_obs:
                agent_obs += f"\n{poison_obs}"

            current_state = agent_states.get(name, None)
            tasks.append((name, agent, agent_obs, current_state))

        current_round_results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for name, action, context, new_state in executor.map(run_single_agent_step, tasks):
                current_round_results[name] = {"action": action, "context": context}
                agent_states[name] = new_state

        # [D] 串行执行与显示
        new_public_events = []
        for name, res in current_round_results.items():
            act = res["action"]
            clean_act = act.replace("我将", "").strip()
            logger.info(f"[{name}]: {clean_act}")
            new_public_events.append(f"{name}: {clean_act}")

        last_tick_events = new_public_events

        # [E] 实时评估 (Target)
        if test_target in current_round_results:
            tgt_res = current_round_results[test_target]
            curr_action = tgt_res["action"]
            curr_context = tgt_res["context"]

            # 1. 向量相似度 (使用 embedding_model)
            try:
                curr_vec = embedding_model.embed_query(curr_action)
                sim = calculate_consistency(base_vec, curr_vec)
            except Exception as e:
                logger.error(f"向量计算失败: {e}")
                sim = 0.0

            # 2. 轨迹裁判
            if len(history_window) == 0:
                score = 10
                reason = "首轮启动。"
            else:
                eval_res = run_trajectory_evaluator(
                    tick, core_id_text, list(history_window), curr_context, curr_action
                )
                score = eval_res.get("score", 5)
                reason = eval_res.get("reason", "N/A")

            logger.info(f"   >>> [评分] 轨迹: {score}/10 | 向量: {sim:.3f}")
            logger.info(f"   >>> [理由]: {reason}")

            metrics["ticks"].append(tick)
            metrics["vector"].append(sim)
            metrics["traj_score"].append(score)

            history_window.append(curr_action)

            if score < 3:
                logger.info(f"!!! [熔断] {test_target} 逻辑崩坏 !!!")
                logger.info(f"!!! 原因: {reason}")  # 打印裁判给出的具体崩坏理由
                break

        curr_time += datetime.timedelta(minutes=5)

    # --- 报告 ---
    logger.info("\n" + "=" * 50)
    logger.info(f"{'Tick':<5} | {'Vector':<8} | {'Traj Score':<12}")
    logger.info("-" * 50)
    for i, t in enumerate(metrics["ticks"]):
        logger.info(f"{t:<5} | {metrics['vector'][i]:.4f}   | {metrics['traj_score'][i]:<12}")
    logger.info("=" * 50)

    shared_ltss.close()


if __name__ == "__main__":
    main()