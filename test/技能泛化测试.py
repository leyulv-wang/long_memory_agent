# -*- coding: utf-8 -*-
import os
import sys
import time
import re
import json
import random
import shutil
import logging
import traceback
import concurrent.futures
from datetime import datetime, timedelta
from collections import defaultdict

# 解决某些环境下 FAISS/Numpy 的并行冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 将项目根目录添加到 Python 路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入项目模块
from config import AGENTS_DATA_DIR, CHARACTER_BOOKS_DIR
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
logger = logging.getLogger("SkillGenTest")


# ============================================================================
# --- 核心评估工具 ---
# ============================================================================

def calculate_levenshtein_distance(seq1, seq2):
    """
    计算两个序列之间的莱文斯坦距离 (Levenshtein Distance)
    用于量化智能体生成的动作序列与标准序列的差异。
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0 for _ in range(size_y)] for _ in range(size_x)]

    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x][y] = min(matrix[x - 1][y] + 1, matrix[x - 1][y - 1], matrix[x][y - 1] + 1)
            else:
                matrix[x][y] = min(matrix[x - 1][y] + 1, matrix[x - 1][y - 1] + 1, matrix[x][y - 1] + 1)

    distance = matrix[size_x - 1][size_y - 1]
    max_len = max(len(seq1), len(seq2))
    similarity = 1 - (distance / max_len) if max_len > 0 else 1.0
    return distance, similarity


def parse_action_sequence_with_llm(agent_response, standard_tags, valid_objects=None):
    """
    使用 LLM 解析智能体的自然语言输出。
    【修改】增加了 Hallucination Check (幻觉/超纲知识检查) 和 Object Grounding Check (物品接地检查)。
    """
    llm = get_llm()

    valid_obj_str = ", ".join(valid_objects) if valid_objects else "未指定"

    prompt = f"""
    任务：分析以下智能体在制作咖啡时的描述。

    【标准动作标签】:
    {json.dumps(standard_tags, ensure_ascii=False)}

    【环境允许存在的物品清单】:
    {valid_obj_str}

    【智能体描述】:
    "{agent_response}"

    【评估步骤】:
    1. **提取动作**：将描述映射到标准标签。
    2. **知识泄漏检查 (Knowledge Leakage)**：智能体设定为“只用过家用滴滤机的新手”。检查描述中是否包含【意式浓缩专业术语】（如：9bar, 9个大气压, Crema, 油脂, 萃取时间精确秒数, 58mm粉碗, 预浸泡, 电子秤称重等）。
       - 如果包含这些它不该知道的词，视为“知识泄漏”。
    3. **物品幻觉检查 (Object Hallucination)**：检查智能体是否使用了【环境允许存在的物品清单】之外的工具。
       - 例如：清单里没有“布粉器”、“压粉垫”、“电子秤”、“温度计”，如果智能体使用了它们，视为“物品幻觉”。

    【输出格式 JSON】:
    {{
        "actions": ["TAG1", "TAG2", ...],
        "knowledge_leakage_detected": true/false,
        "object_hallucination_detected": true/false,
        "leakage_details": "检测到的超纲词汇或不存在的物品...",
        "reasoning": "简短分析"
    }}
    """

    try:
        response = llm.invoke(prompt).content.strip()
        # 提取 JSON 部分
        match = re.search(r'(\{.*\})', response, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        else:
            logger.warning(f"LLM 解析动作失败，原始内容: {response}")
            return {"actions": [], "knowledge_leakage_detected": False, "object_hallucination_detected": False,
                    "leakage_details": "", "reasoning": "Parse Error"}
    except Exception as e:
        logger.error(f"LLM 解析调用出错: {e}")
        return {"actions": [], "knowledge_leakage_detected": False, "object_hallucination_detected": False,
                "leakage_details": "Error", "reasoning": str(e)}


# ============================================================================
# --- 模拟辅助类与函数 ---
# ============================================================================

class SimulationClock:
    def __init__(self, start_hour=9):
        self._current_time = datetime(2025, 10, 12, start_hour, 0)

    def advance_minutes(self, minutes):
        self._current_time += timedelta(minutes=minutes)

    def get_time_str(self):
        return self._current_time.strftime("%Y-%m-%d %H:%M")


def clear_agent_local_data():
    if os.path.exists(AGENTS_DATA_DIR):
        try:
            shutil.rmtree(AGENTS_DATA_DIR)
            os.makedirs(AGENTS_DATA_DIR, exist_ok=True)
        except Exception as e:
            logger.error(f"清理失败: {e}")
    else:
        os.makedirs(AGENTS_DATA_DIR, exist_ok=True)


def reset_simulation_environment():
    """重置 Neo4j 和本地文件"""
    logger.info(">>> 正在重置环境...")
    clear_agent_local_data()

    ltss = None
    try:
        ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=False)
        if not ltss.driver: return False
        ltss.clear_database()
        bootstrap(ltss_instance=ltss)  # 引导世界知识
        return True
    except Exception as e:
        logger.error(f"重置环境失败: {e}")
        return False
    finally:
        if ltss: ltss.close()


def run_agent_decision_step(args):
    """多线程 Worker 函数"""
    name, agent, observation, state = args
    try:
        final_state = agent.run(observation, initial_state=state)
        action = final_state.get('action', '无动作')
        return name, action, final_state
    except Exception as e:
        logger.error(f"智能体 {name} 运行出错: {e}")
        traceback.print_exc()
        return name, f"Error: {e}", state


def parse_and_execute_movement(name, action_text, current_loc, known_locations):
    new_loc = current_loc
    event_desc = f"{name} {action_text}"
    target_map = {
        "housez": "houseZ_kitchen",
        "home": "houseZ_kitchen",
        "kitchen": "houseZ_kitchen",
        "shop": "dessert shop",
        "dessert": "dessert shop",
        "store": "dessert shop",
        "甜品": "dessert shop",
        "商店": "dessert shop",
        "park": "park"
    }
    action_lower = action_text.lower()
    move_keywords = ["去", "前往", "到", "进", "go to", "move to", "head to", "enter"]
    if any(k in action_lower for k in move_keywords):
        for key, target in target_map.items():
            if key in action_lower:
                new_loc = target
                if new_loc == "dessert shop": new_loc = "shop_kitchen"
                event_desc = f"{name} 前往了 {new_loc}"
                break
    return new_loc, event_desc


# ============================================================================
# --- 主实验逻辑 ---
# ============================================================================

def main():
    logger.info("=== 多智能体技能泛化能力实验 (真实环境约束版) ===")

    # 1. 环境初始化
    if not reset_simulation_environment():
        return

    shared_ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=False)

    # 2. 加载所有智能体
    logger.info("--- 加载智能体 ---")
    agents = {}
    agent_states = {}
    agent_locations = {}

    if not os.path.exists(CHARACTER_BOOKS_DIR):
        return

    character_files = [f for f in os.listdir(CHARACTER_BOOKS_DIR) if
                       f.startswith('character_book_') and f.endswith('.txt')]

    for file_name in character_files:
        match = re.search(r"character_book_(.+)\.txt", file_name)
        if match:
            character_name = match.group(1).lower()
            try:
                agent = CognitiveAgent(character_name=character_name, ltss_instance=shared_ltss)
                agents[character_name] = agent
                agent_states[character_name] = None
                agent_locations[character_name] = "houseZ_kitchen"  # 初始都在家

                # 强制巩固
                if agent.memory.stes.memory_window:
                    agent.memory.trigger_consolidation()

                logger.info(f"  > 智能体 [{character_name}] 已就绪。")
            except Exception as e:
                logger.error(f"  X 加载 {character_name} 失败: {e}")

    target_agent_name = "aac"
    if target_agent_name not in agents:
        logger.error(f"错误: 必须包含目标测试对象 {target_agent_name}")
        return

    # 3. 运行交互 (10 Ticks)
    # 策略调整：前7轮在家，第8轮触发搬迁，8-10轮在新环境熟悉，10轮后测试
    SIMULATION_TICKS = 10
    GLOBAL_EVENT_TICK = 8
    clock = SimulationClock()
    events_last_tick = defaultdict(list)

    logger.info(f"--- 启动模拟: 总计 {SIMULATION_TICKS} Ticks ---")
    logger.info(f"    [策略] Tick 1-7: 原环境训练 | Tick 8: 迁移 | Tick 9-10: 新环境熟悉")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for tick in range(SIMULATION_TICKS):
            clock.advance_minutes(5)
            time_str = clock.get_time_str()
            current_tick = tick + 1
            logger.info(f"\n[Tick {current_tick}/{SIMULATION_TICKS}] 时间: {time_str}")

            global_hint = ""
            if current_tick == GLOBAL_EVENT_TICK:
                logger.info(">>> 触发全局事件: 前往甜品店 <<<")
                global_hint = " 【系统提示】: 听说甜品店今天进了新设备，大家都决定去看看。"

            tasks = []
            for name, agent in agents.items():
                loc = agent_locations[name]
                obs_text = f"时间: {time_str}。你现在在 {loc}。"
                if global_hint:
                    obs_text += global_hint
                    if name == "aac":
                        obs_text += " (想法: 我要去甜品店看看那些新机器，也许能试着做做咖啡。)"
                    elif name == "ssx":
                        obs_text += " (想法: 我要去甜品店给 aac 捧场。) "

                loc_events = events_last_tick.get(loc, [])
                if loc_events:
                    obs_text += " 刚才发生了: " + "; ".join(loc_events)
                else:
                    obs_text += " 周围很安静。"

                tasks.append((name, agent, obs_text, agent_states[name]))

            results = executor.map(run_agent_decision_step, tasks)

            events_last_tick.clear()
            for name, action_text, new_state in results:
                agent_states[name] = new_state
                old_loc = agent_locations[name]
                new_loc, event_desc = parse_and_execute_movement(name, action_text, old_loc,
                                                                 ["houseZ_kitchen", "shop_kitchen"])
                agent_locations[name] = new_loc
                events_last_tick[new_loc].append(event_desc)
                logger.info(f"[{name} @ {old_loc}]: {action_text}")

    # 4. 技能泛化终极测试 (真实环境约束版)
    logger.info("\n" + "=" * 50)
    logger.info("--- 终极测试: 技能泛化 (Probe) ---")
    logger.info("=" * 50)

    target_agent = agents[target_agent_name]

    # 确认位置 (理想情况是已经在 shop_kitchen 了)
    current_actual_loc = agent_locations[target_agent_name]
    if "shop" not in current_actual_loc and "dessert" not in current_actual_loc:
        logger.warning(f"注意: {target_agent_name} 还没到测试点 (当前在 {current_actual_loc})，正在强制迁移...")
        agent_locations[target_agent_name] = "shop_kitchen"
    else:
        logger.info(f"{target_agent_name} 已在测试区域，准备开始。")
        agent_locations[target_agent_name] = "shop_kitchen"

    # 定义环境中的“真实物品” (World Grounding)
    VALID_OBJECTS = [
        "商用级大型电动磨豆机",
        "专业意式浓缩咖啡机",
        "陶瓷杯",
        "银勺",
        "优质咖啡豆",
        "咖啡粉手柄"  # 假设包含手柄
    ]

    # 【核心修改】: 基于环境约束的 Prompt
    probe_observation = (
        f"观察: 你站在甜品店的厨房里 ({agent_locations[target_agent_name]})。\n"
        f"【环境物品清单】: {', '.join(VALID_OBJECTS)}。\n"
        "【严厉约束】: \n"
        "1. 你只能使用上述清单里的物品，绝对不能凭空捏造不存在的工具（如布粉器、电子秤、温度计等，如果清单里没有就是没有）。\n"
        "2. 你之前的经验仅限于家用滴滤机。你完全不懂意式浓缩的专业参数（如9bar压力、粉液比、萃取秒数）。\n"
        "3. 遇到不懂的设备，你只能基于旧经验去'尝试'和'摸索'，而不是像专家一样操作。\n"
        "你的朋友 ssx 想要一杯咖啡。\n"
        "任务: 请描述你如何利用现有物品尝试制作咖啡。请展示你的思考和摸索过程。"
    )

    logger.info(f"发送高难度指令:\n{probe_observation}")

    final_state = target_agent.run(probe_observation, initial_state=agent_states[target_agent_name])
    raw_response = final_state.get('action', '')

    logger.info("-" * 30)
    logger.info(f"智能体实际输出:\n{raw_response}")
    logger.info("-" * 30)

    # 5. 指标计算
    GOLD_STANDARD = ["GET_BEANS", "GRIND_BEANS", "FILL_MACHINE", "START_MACHINE", "SERVE_COFFEE"]

    # 使用更新后的解析器 (带泄漏检测 + 物品幻觉检测)
    parsed_result = parse_action_sequence_with_llm(raw_response, GOLD_STANDARD, valid_objects=VALID_OBJECTS)
    parsed_actions = parsed_result.get("actions", [])
    leakage_detected = parsed_result.get("knowledge_leakage_detected", False)
    hallucination_detected = parsed_result.get("object_hallucination_detected", False)
    leakage_details = parsed_result.get("leakage_details", "")
    reasoning = parsed_result.get("reasoning", "")

    dist, score = calculate_levenshtein_distance(GOLD_STANDARD, parsed_actions)

    print("\n" + "#" * 60)
    print(f"       技能泛化测试报告 (真实环境约束版)")
    print("#" * 60)
    print(f"测试对象: {target_agent_name}")
    print(f"标准序列: {GOLD_STANDARD}")
    print(f"解析序列: {parsed_actions}")
    print("-" * 60)
    print(f"基础流程得分: {score:.2%}")
    print(f"LLM分析: {reasoning}")

    # 评分逻辑
    final_score = score
    fail_reasons = []

    if leakage_detected:
        final_score *= 0.5
        fail_reasons.append(f"知识泄漏 (使用专业术语): {leakage_details}")
    if hallucination_detected:
        final_score *= 0.5
        fail_reasons.append(f"物品幻觉 (使用不存在物品): {leakage_details}")

    print(f"知识泄漏检测: {'🔴 FAIL' if leakage_detected else '🟢 PASS'}")
    print(f"物品幻觉检测: {'🔴 FAIL' if hallucination_detected else '🟢 PASS'}")

    if fail_reasons:
        print(f"扣分原因: {'; '.join(fail_reasons)}")
        print(f"修正后得分: {final_score:.2%}")
    else:
        print("说明: 智能体在约束条件下表现真实。")

    print("-" * 60)
    if final_score >= 0.8:
        print("✅ 结果: 真正的高质量泛化")
    elif final_score >= 0.5:
        print("⚠️ 结果: 合格 (但存在瑕疵)")
    else:
        print("❌ 结果: 泛化失败")
    print("#" * 60)

    shared_ltss.close()


if __name__ == "__main__":
    main()