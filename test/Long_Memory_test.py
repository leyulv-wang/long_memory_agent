# -*- coding: utf-8 -*-
"""
LongMemoryEval 测试脚本（FINAL，可覆盖）

改动点（相对你旧版）：
1) 不再 turn_stream/perceive/trigger_consolidation（旧版会让 TURN 轴变成 batch_counter）:contentReference[oaicite:2]{index=2}
2) 直接调用 utils.ingest_longmemoryeval.ingest_longmemoryeval_sample：
   - turn_id = task 内 session 时间顺序(1..N)
   - question_time = TURN_{question_turn_id}（由 question_date 映射）
   - RAW（成熟工具抽取）+ CONSOLIDATED（LLM 轻推理）全部写 Neo4j
3) 巩固阶段始终使用便宜模型：强制 agent.memory.consolidation_llm = agent.llm
4) 只有最终问答时临时切 expensive_llm（保留你原逻辑）:contentReference[oaicite:3]{index=3}
"""

import json
import logging
import os
import sys
from typing import Optional, List, Dict, Any

from tqdm import tqdm

# ---- 路径：确保能导入项目模块 ----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 尽量把“项目根目录”加进 sys.path：test/.. -> root
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import memory_consolidation_threshold
from 智能体初始化实例 import AgentManager
from memory.stores import LongTermSemanticStore
from utils.清理短期长期记忆 import clear_short_term_memory, fast_clear_neo4j_data
from utils.ingest_longmemoryeval import ingest_longmemoryeval_sample

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("LongMemoryEval")

# =========================
# 数据路径（自动适配）
# =========================
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "long_memory_eval", "sampled_test_questions.json")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "test", "long_memory_results.json")

CHARACTER_NAME = "LongMemory"
INCLUDE_ASSISTANT_TURNS = True


def _pick_existing_path(*candidates: str) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0] if candidates else ""


def run_longmemoryeval(
    data_path: str = DEFAULT_DATA_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    character_name: str = CHARACTER_NAME,
    limit_cases: Optional[int] = None,
    # consolidated 并行参数（可按需调）
    consolidation_max_workers: int = 6,
    consolidation_max_retries: int = 3,
    consolidation_base_backoff_s: float = 1.0,
    consolidation_min_interval_s: float = 0.0,
):
    # 兼容：如果用户从不同工作目录运行，兜底再找一次
    data_path = _pick_existing_path(
        data_path,
        os.path.join(THIS_DIR, "data", "long_memory_eval", "sampled_test_questions.json"),
        os.path.join(PROJECT_ROOT, "data", "long_memory_eval", "sampled_test_questions.json"),
    )

    if not os.path.exists(data_path):
        logger.error(f"ERROR: data file not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        test_data: List[Dict[str, Any]] = json.load(f)

    if limit_cases is not None:
        test_data = test_data[: int(limit_cases)]

    # test_data = test_data[0:1]
    # 仅跑第 2、3 题（0-based 索引）
    test_data = test_data[1:3]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"\n开始测试 | 样本数: {len(test_data)} | session_batch_size={memory_consolidation_threshold}")
    logger.info(f"INCLUDE_ASSISTANT_TURNS={INCLUDE_ASSISTANT_TURNS}")
    logger.info(f"数据集: {data_path}")
    logger.info(f"输出文件: {output_path}\n")

    results: List[Dict[str, Any]] = []

    for idx, case in enumerate(tqdm(test_data, desc="LongMemoryEval", unit="case"), 1):
        q_id = case.get("question_id", f"case_{idx:04d}")
        question = case.get("question", "")
        gt = case.get("answer", "")

        logger.info(f"\n{'=' * 30} 样本 [{idx}/{len(test_data)}] ID: {q_id} {'=' * 30}")

        # ========== A) 每个样本独立：清 Neo4j（保留索引约束） ==========
        try:
            temp_ltss = LongTermSemanticStore(bootstrap_now=False)
            fast_clear_neo4j_data(temp_ltss)  # DETACH DELETE n :contentReference[oaicite:4]{index=4}
            temp_ltss.close()
        except Exception as e:
            logger.error(f"ERROR: failed to clear Neo4j: {e}", exc_info=True)
            continue

        # ========== B) 实例化智能体（不导入世界书） ==========
        bot = AgentManager(character_name, auto_bootstrap=False)

        # 评测时：关闭 STES autosave（避免写盘影响速度）
        try:
            if hasattr(bot.agent.memory.stes, "autosave"):
                bot.agent.memory.stes.autosave = False
            if hasattr(bot.agent.memory.stes, "auto_save"):
                bot.agent.memory.stes.auto_save = False
        except Exception:
            pass

        agent = bot.agent

        # ✅ 关键：保证“巩固阶段”始终用便宜模型
        # 你的旧脚本只有最终问答才切 expensive_llm :contentReference[oaicite:5]{index=5}
        # 这里显式把 consolidation_llm 固定为当前 agent.llm（通常是 cheap）。
        try:
            if getattr(agent, "memory", None) is not None:
                agent.memory.consolidation_llm = agent.llm
        except Exception:
            pass

        # ========== C) 直接注入（RAW + CONSOLIDATED，数据集特化 turn_id/session 时间轴） ==========
        try:
            ingest_res = ingest_longmemoryeval_sample(
                agent=agent,
                case=case,
                batch_size=memory_consolidation_threshold,  # 一次处理 n 个 session 的窗口
                overlap_sessions=0,  # 特化测试：默认不 overlap，避免重复写入
                include_assistant=INCLUDE_ASSISTANT_TURNS,
                show_progress=False,
                # consolidated 并行参数
                consolidation_max_workers=consolidation_max_workers,
                consolidation_max_retries=consolidation_max_retries,
                consolidation_base_backoff_s=consolidation_base_backoff_s,
                consolidation_min_interval_s=consolidation_min_interval_s,
                consolidation_mode="original_text",
                original_max_turns_per_chunk=8,
                original_overlap_turns=1,
                original_max_chars_per_chunk=6000,
            )

            # import importlib.util
            # # 动态加载 test/各种简单测试.py，做测试
            # diag_path = os.path.join(PROJECT_ROOT, "test", "各种简单测试.py")
            # spec = importlib.util.spec_from_file_location("diag_module", diag_path)
            # diag_module = importlib.util.module_from_spec(spec)
            # spec.loader.exec_module(diag_module)
            #
            # rep = diag_module.diagnose_neo4j_graph(agent_name=character_name, limit_samples=5)
            # diag_module.print_report(rep)
            # raise SystemExit("Stop after diagnosis")

            question_turn_id = int(ingest_res.get("question_turn_id", 1) or 1)
            question_time = f"TURN_{question_turn_id}"
            logger.info(f"[q_id={q_id}] 注入完成：num_sessions={ingest_res.get('num_sessions')} question_time={question_time}")
        except Exception as e:
            logger.error(f"ERROR: ingest failed: q_id={q_id} err={e}", exc_info=True)
            try:
                bot.close()
            except Exception:
                pass
            continue

        # ========== D) 最终问答（只在这里切贵模型） ==========
        logger.info(f"\n问题: {question}")
        prediction = ""
        try:
            original_llm = agent.llm
            # 只在最后问答用贵模型（保持你原逻辑）:contentReference[oaicite:6]{index=6}
            if hasattr(agent, "expensive_llm") and agent.expensive_llm is not None:
                agent.llm = agent.expensive_llm

            prediction = agent.run(question, current_time=question_time).get("action", "")
            agent.llm = original_llm
        except Exception as e:
            logger.error(f"ERROR: QA failed: {e}", exc_info=True)
            try:
                agent.llm = original_llm
            except Exception:
                pass
            prediction = ""

        logger.info(f"回答(截断): {prediction[:160]}{'...' if len(prediction) > 160 else ''}")

        results.append(
            {
                "question_id": q_id,
                "hypothesis": prediction,
                "ground_truth": gt,
                "question_time": question_time,
            }
        )

        # ========== E) 关闭 + 清短期（FAISS） ==========
        try:
            bot.close()
        except Exception:
            pass

        try:
            clear_short_term_memory(target_agent=character_name)  # :contentReference[oaicite:7]{index=7}
        except Exception:
            pass

    # ========== 保存结果 ==========
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"\nTest finished. Results written to: {output_path}")
    except Exception as e:
        logger.error(f"ERROR: failed to write results: {e}", exc_info=True)

    # ========== 额外保存：官方评测格式（jsonl） ==========
    hypo_path = output_path.replace(".json", ".jsonl")

    with open(hypo_path, "w", encoding="utf-8") as f:
        for r in results:
            line = {
                "question_id": r["question_id"],
                "hypothesis": r["hypothesis"],
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info(f"Official format written: {hypo_path}")


if __name__ == "__main__":
    run_longmemoryeval()
