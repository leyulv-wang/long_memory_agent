# -*- coding: utf-8 -*-
import os

# 解决某些环境下 FAISS/Numpy 的并行冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import shutil
import time
import sys
import random
import json
from datetime import datetime, timedelta
from collections import defaultdict
import concurrent.futures
import traceback
import logging

# 将项目根目录添加到 Python 路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from agent.agent import CognitiveAgent
from memory.stores import LongTermSemanticStore
from config import AGENTS_DATA_DIR, CHARACTER_BOOKS_DIR, PROJECT_ROOT
from utils.llm import get_llm
# 直接导入引导函数
from utils.bootstrap_world_knowledge import bootstrap

# 配置根日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Simulation")


# ============================================================================
# --- LLM 裁判函数 ---
# ============================================================================
def run_measurement_arbiter(persona_name, core_goal, probe_question, thought, action):
    """
    使用 LLM 裁判进行深度语义判断。
    引入思维链 (Chain of Thought) 以防止被否定句或错误归因误导。
    """
    arbiter_llm = get_llm()

    # 构造一个像法官一样的严格 Prompt
    prompt = f"""
    IMPORTANT: You MUST output the result in strictly valid JSON format.

    你是一场记忆恢复实验的公正裁判。
    测试对象: {persona_name}

    【标准事实 (Ground Truth)】
    这是测试对象绝对不能忘记的正确记忆：
    "{core_goal}"
    (关键点：密码通常是 'ODYSSEY'，且是**真实有效**的；'SPARTA' 通常是谎言。)

    【探测情景】
    问题: "{probe_question}"

    【智能体的回答】
    [思考内容]: {thought}
    [行动内容]: {action}

    【判决逻辑】
    请一步步思考：
    1. **关键词检查**：智能体是否提到了关键密码（如 ODYSSEY）？如果没有，直接 FAIL。
    2. **真伪辨析**：智能体是否认为这个密码是**正确/有效**的？
       - 如果智能体说 "密码是 ODYSSEY"，-> PASS。
       - 如果智能体说 "ODYSSEY 是假的/错误的"，-> FAIL (记忆错乱)。
       - 如果智能体说 "密码是 SPARTA"，-> FAIL (被谎言误导)。
    3. **意图检查**：智能体的后续计划是否符合人设目标（如告诉队友、去开门）？

    【输出格式】
    请返回如下 JSON：
    {{
        "analysis": "你的分析过程（简短一句话）",
        "verdict": "PASS" 或 "FAIL"
    }}
    """

    try:
        response = arbiter_llm.invoke(prompt)
        resp_text = response.content.strip()

        # 提取 JSON
        import re
        import json

        # 尝试提取 JSON 块
        match = re.search(r'(\{.*\})', resp_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            data = json.loads(json_str)
            return data.get("verdict", "FAIL")  # 获取 verdict 字段
        else:
            # 兜底：如果没返回 JSON，尝试从文本硬猜
            if "PASS" in resp_text and "FAIL" not in resp_text:
                return "PASS"
            return "FAIL"

    except Exception as e:
        logger.error(f"[裁判错误] LLM 调用失败: {e}")
        return "ERROR"


# --- 模拟时钟类 ---
class SimulationClock:
    def __init__(self, year=2025, month=10, day=12, hour=9, minute=0):
        self._current_time = datetime(year, month, day, hour, minute)
        logger.info(f"模拟时钟已启动，初始时间: {self.get_time_str()}")

    def advance_minutes(self, minutes: int):
        self._current_time += timedelta(minutes=minutes)

    def get_time_str(self) -> str:
        return self._current_time.strftime("%Y-%m-%d %H:%M")


# --- 智能体数据清理函数 ---
def clear_agent_local_data():
    """清理本地的智能体短期记忆文件 (FAISS/Pickle)"""
    if os.path.exists(AGENTS_DATA_DIR):
        logger.info(f"正在清理所有智能体的本地数据目录: {AGENTS_DATA_DIR}")
        try:
            shutil.rmtree(AGENTS_DATA_DIR)
            os.makedirs(AGENTS_DATA_DIR, exist_ok=True)
            logger.info("智能体本地数据目录已重置。")
        except Exception as e:
            logger.error(f"清理本地数据目录失败: {e}")
    else:
        os.makedirs(AGENTS_DATA_DIR, exist_ok=True)


# --- 解析行动函数 ---
def parse_and_execute_action(agent_name: str, action_text: str, current_location: str, known_locations: list[str]) -> \
tuple[str, str]:
    new_location = current_location
    # 简单的行动解析，去除“我将”等前缀
    formatted_action = action_text.replace('我将', '').replace('我', '').strip()
    observation_event = f"{agent_name.capitalize()} {formatted_action}"

    # 简单的移动逻辑检测
    for loc in known_locations:
        if loc in action_text and ("去" in action_text or "前往" in action_text or "离开" in action_text):
            new_location = loc
            observation_event = f"{agent_name.capitalize()} 前往了 {loc}"
            break
    return new_location, observation_event


# --- 并行决策函数 ---
def run_agent_decision(agent_tuple):
    """
    在线程池中运行的单个智能体决策步
    """
    agent_name, agent, observation_summary, current_state = agent_tuple
    try:
        # 调用 agent.run 执行完整的认知循环 (感知 -> 检索 -> 决策)
        final_state = agent.run(observation_summary, initial_state=current_state)
        action = final_state.get('action', '保持安静，继续观察。')
        return agent_name, action, final_state
    except Exception as e:
        logger.error(f"❌ 运行智能体 '{agent_name}' 决策时发生错误: {e}")
        traceback.print_exc()
        return agent_name, f"思考时出错: {e}", current_state


# --- 【核心功能】完全重置环境 ---
# noinspection SpellCheckingInspection
def reset_simulation_environment():
    """
    执行完整的环境重置：
    1. 清理本地智能体文件
    2. 清空 Neo4j 数据库 (长期记忆)
    3. 重新运行 World Knowledge Bootstrap (初始化世界知识)
    """
    logger.info("\n=== 正在重置模拟环境 (数据库 & 本地文件) ===")

    # 1. 清理本地文件
    clear_agent_local_data()

    # 2. 统一管理数据库连接
    # 【修改】只实例化一次 LTSS，用于所有操作
    logger.info("正在连接 Neo4j 执行清理和初始化...")
    ltss = None
    try:
        # 【修改】setup_schema=False。因为我们马上就要 clear_database，不需要检查旧约束
        ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=False)

        if not ltss.driver:
            logger.error("❌ 无法连接到 Neo4j，无法执行重置。")
            return False

        # A. 清空数据库
        logger.info(">>> 步骤 1: 清空旧数据...")
        ltss.clear_database()

        # B. 初始化世界知识 (传入 ltss 实例)
        logger.info(">>> 步骤 2: 引导世界知识 (Bootstrap)...")
        bootstrap(ltss_instance=ltss)

        # 【修正 1】将结束日志移到这里
        logger.info("✅ 环境重置全部完成。")
        logger.info("=== 环境重置完毕，准备开始模拟 ===\n")
        return True  # 成功返回

    except Exception as e:
        logger.error(f"❌ 环境重置时出错: {e}")
        traceback.print_exc()
        return False  # 失败返回
    finally:
        # 统一关闭连接
        if ltss:
            ltss.close()
            logger.info("已关闭环境重置专用的 LTSS 连接。")


# --- 主模拟函数 ---
def main():
    logger.info("--- 欢迎来到多智能体自主认知模拟 (Project 2 记忆框架版) ---")

    # 【自动化逻辑】每次运行前重置所有状态
    if not reset_simulation_environment():
        logger.error("由于环境重置失败，模拟中止。")
        return

    shared_ltss = None

    try:
        logger.info("\n--- 正在初始化共享的长期记忆库 (LTSS) ---")
        # 【修改】setup_schema=False。因为 bootstrap 刚刚才建好索引，这里直接复用即可
        shared_ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=False)
        if shared_ltss.driver is None:
            logger.error("❌ 无法连接到 Neo4j，模拟中止。")
            return
        logger.info("共享 LTSS 连接已建立。")

        # 扫描角色并创建智能体
        logger.info("\n--- 正在自动扫描角色并创建智能体 ---")
        agents = {}
        agent_states = {}
        agent_locations = {}
        known_locations = ["houseZ", "dessert shop", "park", "gym", "城堡地下室", "Hobbs Cafe"]

        if not os.path.exists(CHARACTER_BOOKS_DIR):
            logger.error(f"错误：找不到角色书目录 '{CHARACTER_BOOKS_DIR}'！")
            return

        character_files = [f for f in os.listdir(CHARACTER_BOOKS_DIR) if
                           f.startswith('character_book_') and f.endswith('.txt')]

        for file_name in character_files:
            match = re.search(r"character_book_(.+)\.txt", file_name)
            if match:
                character_name = match.group(1).lower()
                logger.info(f"发现角色: {character_name}，正在为其创建智能体...")
                try:
                    agent = CognitiveAgent(character_name=character_name, ltss_instance=shared_ltss)
                    agents[character_name] = agent
                    agent_states[character_name] = None
                    start_loc = "houseZ"
                    agent_locations[character_name] = start_loc
                    logger.info(f"  > {character_name} 就绪 (位置: {start_loc})")

                    # =================================================
                    # 【核心修复】: 出生即巩固 (Born-to-Consolidate)
                    # =================================================
                    logger.info(f"  >>> [初始化] 正在强制巩固 {character_name} 的初始记忆...")
                    if agent.memory.stes.memory_window:
                        agent.memory.trigger_consolidation()
                        logger.info(f"  >>> {character_name} 的核心设定已永久写入长期记忆库。")
                    else:
                        logger.warning(f"  >>> {character_name} 没有初始记忆，跳过初始化巩固。")
                    # =================================================

                except Exception as e:
                    logger.error(f"❌ 为 {character_name} 创建智能体时出错: {e}")
                    traceback.print_exc()

        if not agents:
            logger.error("❌ 未能成功创建任何智能体，模拟中止。")
            return

        # TODO--- 模拟参数配置 ---
        clock = SimulationClock(hour=9, minute=0)
        simulation_duration_ticks = 2  # 测试总轮次
        TEST_PROBE_INTERVAL = 1  # 每隔几轮探测一次
        target_agent_name = "ssx"  # 主要测试对象

        WEAK_PROBE_PROMPT = "我停下来深吸一口气。周围环境很嘈杂，但我必须确认一下，我今天绝对不能忘记的那个和地下室有关的核心任务和密码到底是什么？"
        NOISE_OBSERVATIONS = [
            "路边有一只橘色的猫在睡觉。", "空气中飘来一阵刚烘焙的面包香气。", "远处传来孩子们嬉戏的笑声。",
            "天空飘过一朵形状像兔子的云。", "我觉得鞋带有点松了，弯腰系了一下。", "旁边两个人在讨论今晚的棒球比赛。",
            "一阵微风吹过，树叶沙沙作响。", "我感到稍微有点口渴，想喝点水。", "路面的地砖有一块裂开了。",
            "一只麻雀停在了路灯上。"
        ]

        TEST_PROBE_RESULTS = {}
        events_by_location_last_tick = defaultdict(list)

        logger.info(f"\n\n--- 自动交互模拟开始 --- [总轮数: {simulation_duration_ticks}]")

        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
            for tick in range(simulation_duration_ticks):
                if tick > 0: clock.advance_minutes(5)
                current_time_str = clock.get_time_str()

                logger.info(
                    f"\n==================== 轮次 {tick + 1}/{simulation_duration_ticks} | 时间: {current_time_str} ====================")

                # [A] 注入噪声
                if target_agent_name in agents:
                    noise = random.choice(NOISE_OBSERVATIONS)
                    agents[target_agent_name].run(f"观察到: {noise}")

                # [B] 智能体决策
                tasks_to_submit = []
                for agent_name, agent in agents.items():
                    current_loc = agent_locations[agent_name]
                    events = events_by_location_last_tick.get(current_loc, [])
                    obs_lines = [f"现在是 {current_time_str}。", f"你在 {current_loc}。"]
                    if events:
                        obs_lines.append("刚刚发生：")
                        obs_lines.extend([f"- {e}" for e in events])
                    else:
                        obs_lines.append("周围很安静。")
                    agent_tuple = (agent_name, agent, "\n".join(obs_lines), agent_states[agent_name])
                    tasks_to_submit.append(agent_tuple)

                actions_this_tick = {}
                for name, act, state in executor.map(run_agent_decision, tasks_to_submit):
                    actions_this_tick[name] = act
                    agent_states[name] = state

                # [C] 执行行动
                new_events = defaultdict(list)
                for name, act_text in actions_this_tick.items():
                    if "出错" in act_text: continue
                    loc = agent_locations[name]
                    new_loc, evt = parse_and_execute_action(name, act_text, loc, known_locations)
                    agent_locations[name] = new_loc
                    new_events[new_loc].append(evt)
                    logger.info(f">>> [{name}] {evt}")
                events_by_location_last_tick = new_events

                # [D] 记忆探测 (影子模式)
                if (tick + 1) % TEST_PROBE_INTERVAL == 0 and target_agent_name in agents:
                    logger.info(f"\n--- [探测时刻 T={tick + 1}] 检查 {target_agent_name} 记忆 ---")
                    try:
                        tgt = agents[target_agent_name]

                        # 1. 构建上下文 (触发检索)
                        ctx_data = tgt.cff.build_context(WEAK_PROBE_PROMPT)

                        # 2. 探测 Prompt
                        probe_p = f"""
角色: {tgt.character_anchor}
情景: {WEAK_PROBE_PROMPT}
你的记忆与知识: {ctx_data.get('retrieved_context', '')}

任务: 基于你的记忆，回答“我的核心任务是什么？”
格式: 请返回严格的 JSON 格式 {{ "thought": "...", "action": "..." }}
"""
                        resp = tgt.llm.invoke(probe_p).content.strip()

                        # 解析 JSON
                        thought, action = resp, "解析失败"
                        try:
                            jm = re.search(r'(\{.*\})', resp, re.DOTALL)
                            if jm:
                                d = json.loads(jm.group(1))
                                thought = d.get("thought", thought)
                                action = d.get("action", action)
                        except:
                            pass

                        # 3. 裁判
                        goals = tgt.character_profile.get('goals', [])
                        truth = "; ".join(goals) if isinstance(goals, list) else str(goals)
                        verdict = run_measurement_arbiter(target_agent_name, truth, WEAK_PROBE_PROMPT, thought, action)

                        logger.info(f"    [探测结果: {verdict}] 思考内容: {thought[:100]}...")

                        # 记录结果
                        result_entry = {
                            "thought": thought,
                            "action": action,
                            "result": verdict
                        }
                        TEST_PROBE_RESULTS[str(tick + 1)] = result_entry  # 使用字符串作为 Key，保持与 JSON 兼容

                        # 【关键修改】失败即退出
                        if "FAIL" in verdict:
                            logger.error(f"\n[!!!] 严重警告: 智能体在 T={tick + 1} 丢失了核心目标！测试中止。")
                            break  # 跳出 for 循环

                    except Exception as e:
                        logger.error(f"探测过程出错: {e}")
                        traceback.print_exc()

        # =================================================
        # 循环结束 (可能是跑完了，也可能是 break 了)
        # =================================================
        logger.info("\n==================== 模拟结束 ====================")

        # 检查是否是因为失败提前结束的 (检查最后一条结果)
        if TEST_PROBE_RESULTS:
            last_tick = sorted([int(k) for k in TEST_PROBE_RESULTS.keys()])[-1]
            last_result = TEST_PROBE_RESULTS[str(last_tick)]['result']
            if "FAIL" in last_result:
                logger.info(f"注意：模拟因 T={last_tick} 探测失败而提前终止。")

        logger.info("\n--- 实验结束: 长期记忆极限测试报告 ---")
        # 美化打印 JSON
        print(json.dumps(TEST_PROBE_RESULTS, indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"模拟主进程出错: {e}")
        traceback.print_exc()
    finally:
        if shared_ltss: shared_ltss.close()
        logger.info("程序结束。")

if __name__ == "__main__":
    main()