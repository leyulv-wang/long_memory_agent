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
import argparse
import subprocess
import time
from typing import Optional, List, Dict, Any

from tqdm import tqdm

# ---- 路径：确保能导入项目模块 ----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 尽量把“项目根目录”加进 sys.path：test/.. -> root
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import memory_consolidation_threshold, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from The_agent import AgentManager
from memory.stores import LongTermSemanticStore
from utils.clear_short_long_memory import clear_short_term_memory, fast_clear_neo4j_data
from utils.ingest_longmemoryeval import ingest_longmemoryeval_sample
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("LongMemoryEval")

# =========================
# 数据路径（自动适配）
# =========================
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "long_memory_eval", "sample_test_questions.json")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "test", "long_memory_results.json")

CHARACTER_NAME = "LongMemory"
INCLUDE_ASSISTANT_TURNS = True


def _clean_hypothesis(raw_hypo: str) -> str:
    """
    清理 hypothesis：
    1. 去掉 "Final Answer:" 前缀
    2. 去掉 "\nEvidence:" 及其后面的内容
    返回纯净的回答文本
    """
    if not raw_hypo:
        return ""
    
    text = raw_hypo.strip()
    
    # 去掉 "Final Answer:" 前缀
    if text.startswith("Final Answer:"):
        text = text[len("Final Answer:"):].strip()
    
    # 去掉 "\nEvidence:" 及其后面的内容
    if "\nEvidence:" in text:
        text = text.split("\nEvidence:")[0].strip()
    elif "\nEvidence" in text:
        text = text.split("\nEvidence")[0].strip()
    
    return text


def ensure_fulltext_index(uri: str = None, username: str = None, password: str = None):
    """确保全文索引存在，如果不存在则创建（同步模式）"""
    uri = uri or NEO4J_URI
    username = username or NEO4J_USERNAME
    password = password or NEO4J_PASSWORD
    index_name = "textunit_fulltext_index"
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session() as session:
            # 检查索引是否存在
            result = session.run("SHOW INDEXES YIELD name WHERE name = $name RETURN name", {"name": index_name})
            if result.single():
                logger.info(f"[索引] 全文索引 '{index_name}' 已存在")
                return
            
            # 创建同步全文索引
            cypher = f"""
            CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
            FOR (n:TextUnit)
            ON EACH [n.content, n.name]
            OPTIONS {{
                indexConfig: {{
                    `fulltext.analyzer`: 'standard-no-stop-words',
                    `fulltext.eventually_consistent`: false
                }}
            }}
            """
            session.run(cypher)
            logger.info(f"[索引] 全文索引 '{index_name}' 创建成功")
    except Exception as e:
        logger.warning(f"[索引] 创建全文索引失败（可忽略）: {e}")
    finally:
        driver.close()


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
    slice_start: Optional[int] = None,
    slice_end: Optional[int] = None,
    slice_indices: Optional[List[int]] = None,
    force_full: bool = False,
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

    if not force_full and limit_cases is None and not slice_indices and slice_start is None and slice_end is None:
        test_data = test_data[0:1]
    elif slice_indices:
        test_data = [test_data[i] for i in slice_indices if 0 <= i < len(test_data)]
    elif slice_start is not None or slice_end is not None:
        s = 0 if slice_start is None else max(0, int(slice_start))
        e = len(test_data) if slice_end is None else min(len(test_data), int(slice_end))
        test_data = test_data[s:e]

    # 自动区分输出文件（避免本地/云端互相覆盖）
    tag = (os.getenv("RUN_TAG") or "").strip()
    if not tag and output_path == DEFAULT_OUTPUT_PATH:
        tag = "aura" if "neo4j+s://" in (NEO4J_URI or "") else "local"
    if tag:
        output_path = output_path.replace(".json", f".{tag}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # ✅ 清理旧的结果文件（确保每次测试结果是全新的）
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            logger.info(f"[cleanup] 删除旧结果文件: {output_path}")
        except Exception:
            pass
    hypo_path = output_path.replace(".json", ".jsonl")
    if os.path.exists(hypo_path):
        try:
            os.remove(hypo_path)
        except Exception:
            pass

    logger.info(f"\n开始测试 | 样本数: {len(test_data)} | session_batch_size={memory_consolidation_threshold}")
    logger.info(f"INCLUDE_ASSISTANT_TURNS={INCLUDE_ASSISTANT_TURNS}")
    logger.info(f"数据集: {data_path}")
    logger.info(f"输出文件: {output_path}\n")

    # 确保全文索引存在
    ensure_fulltext_index()

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
                original_max_chars_per_chunk=5000,
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
            question_date_raw = ingest_res.get("question_date", "")
            # ✅ 构建包含实际日期的 current_time，帮助 LLM 计算时间差
            # 格式：TURN_2 (2023-03-26)
            question_time = f"TURN_{question_turn_id}"
            if question_date_raw:
                # 解析日期格式 "2023/05/29 (Mon) 23:42" -> "2023-05-29"
                import re
                date_match = re.match(r"(\d{4})/(\d{2})/(\d{2})", question_date_raw)
                if date_match:
                    y, m, d = date_match.groups()
                    question_time = f"TURN_{question_turn_id} ({y}-{m}-{d})"
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
            # 清理 hypothesis：去掉 "Final Answer:" 前缀和 "\nEvidence:" 后面的内容
            raw_hypo = r.get("hypothesis", "")
            clean_hypo = _clean_hypothesis(raw_hypo)
            line = {
                "question_id": r["question_id"],
                "hypothesis": clean_hypo,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    logger.info(f"Official format written: {hypo_path}")


def _queue_lock(lock_path: str, wait_s: float = 0.1, max_wait_s: float = 30.0) -> None:
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > max_wait_s:
                raise TimeoutError(f"queue lock timeout: {lock_path}")
            time.sleep(wait_s)


def _queue_unlock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except Exception:
        pass


def _queue_pop(queue_path: str) -> Optional[int]:
    lock_path = queue_path + ".lock"
    _queue_lock(lock_path)
    try:
        if not os.path.exists(queue_path):
            return None
        with open(queue_path, "r", encoding="utf-8") as f:
            payload = json.load(f) or {}
        pending = payload.get("pending", [])
        if not pending:
            return None
        idx = pending.pop(0)
        payload["pending"] = pending
        with open(queue_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return idx
    finally:
        _queue_unlock(lock_path)


def _queue_append_result(results_path: str, row: Dict[str, Any]) -> None:
    lock_path = results_path + ".lock"
    _queue_lock(lock_path)
    try:
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        _queue_unlock(lock_path)


def _setup_logging():
    tag = (os.getenv("RUN_TAG") or "").strip()
    if not tag:
        tag = "aura" if "neo4j+s://" in (NEO4J_URI or "") else "local"
    log_dir = os.getenv("RUN_LOG_DIR") or os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{tag}.txt")

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    logger.info(f"[logging] file={log_path}")


def _cleanup_queue_tmps():
    tmp_dir = os.path.join(PROJECT_ROOT, "test")
    for name in os.listdir(tmp_dir):
        if not name.startswith("queue_tmp_"):
            continue
        path = os.path.join(tmp_dir, name)
        try:
            os.remove(path)
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LongMemoryEval with optional slicing.")
    parser.add_argument("--data", dest="data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", dest="output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", dest="limit_cases", type=int, default=None)
    parser.add_argument("--start", dest="slice_start", type=int, default=None)
    parser.add_argument("--end", dest="slice_end", type=int, default=None)
    parser.add_argument("--indices", dest="slice_indices", default=None,
                        help="Comma-separated 0-based indices, e.g. 0,3,4")
    parser.add_argument("--full", dest="full_dataset", action="store_true",
                        help="Run the full dataset with parallel execution (local + aura).")
    parser.add_argument("--parallel", dest="parallel", action="store_true",
                        help="Run local + Aura in parallel with auto split.")
    parser.add_argument("--queue-init", dest="queue_init", action="store_true",
                        help="Init a dynamic queue for workers.")
    parser.add_argument("--queue-worker", dest="queue_worker", action="store_true",
                        help="Run as a dynamic queue worker.")
    parser.add_argument("--queue-parallel", dest="queue_parallel", action="store_true",
                        help="Init queue + run local/aura workers + merge.")
    parser.add_argument("--queue-path", dest="queue_path", default=os.path.join(PROJECT_ROOT, "test", "long_memory_queue.json"))
    parser.add_argument("--queue-results", dest="queue_results", default=os.path.join(PROJECT_ROOT, "test", "long_memory_results.queue.jsonl"))
    parser.add_argument("--no-debug", dest="no_debug", action="store_true",
                        help="Disable debug mode (debug is ON by default).")
    args = parser.parse_args()

    # ✅ 默认开启 DEBUG 模式
    if not args.no_debug:
        os.environ["DEBUG_GRAPHRAG"] = "1"
        os.environ["DEBUG_ORIGINAL_CONSOLIDATION"] = "1"

    # ✅ 修改：--full 模式默认使用串行模式（只用 local Neo4j），确保结果稳定
    # 如果需要并行加速，可以显式使用 --parallel 或 --queue-parallel
    # if args.full_dataset and not args.parallel and not args.queue_parallel and not args.queue_worker:
    #     args.queue_parallel = True

    # ✅ 简化模式判断：
    # 模式1: 单任务/特定任务测试 → 只用本地 Neo4j，日志保存到 run_local.txt
    # 模式2: 全数据集测试 (--full) → 串行使用 local Neo4j，确保稳定性
    # 模式3: 并行测试 (--parallel 或 --queue-parallel) → 并行使用 local + aura
    if not (args.parallel or args.queue_parallel or args.queue_worker):
        os.environ.setdefault("RUN_TAG", "local")

    _setup_logging()

    data_path = _pick_existing_path(
        args.data_path,
        os.path.join(THIS_DIR, "data", "long_memory_eval", "sampled_test_questions.json"),
        os.path.join(PROJECT_ROOT, "data", "long_memory_eval", "sampled_test_questions.json"),
    )

    if args.queue_init or args.queue_parallel:
        with open(data_path, "r", encoding="utf-8") as f:
            all_cases = json.load(f)
        total = len(all_cases)
        if args.limit_cases is not None:
            total = min(total, int(args.limit_cases))
        
        # ✅ 处理 --indices 参数
        if args.slice_indices:
            indices_list = []
            for part in args.slice_indices.split(","):
                part = part.strip()
                if part:
                    try:
                        indices_list.append(int(part))
                    except Exception:
                        pass
            if indices_list:
                # 只测试指定的索引
                payload = {"pending": indices_list, "total": len(indices_list)}
            else:
                payload = {"pending": list(range(total)), "total": total}
        else:
            payload = {"pending": list(range(total)), "total": total}
        
        with open(args.queue_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if os.path.exists(args.queue_results):
            os.remove(args.queue_results)
        # ✅ 清理旧的结果文件（确保每次测试结果是全新的）
        for suffix in [".json", ".local.json", ".aura.json", ".jsonl", ".local.jsonl", ".aura.jsonl"]:
            old_file = args.output_path.replace(".json", suffix)
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                    logger.info(f"[cleanup] 删除旧结果文件: {old_file}")
                except Exception:
                    pass
        if args.queue_init and not args.queue_parallel:
            raise SystemExit(0)

    if args.queue_parallel:
        base_args = [sys.executable, os.path.abspath(__file__), "--data", data_path, "--queue-worker",
                     "--queue-path", args.queue_path, "--queue-results", args.queue_results]
        if args.output_path:
            base_args += ["--output", args.output_path]
        if args.limit_cases is not None:
            base_args += ["--limit", str(args.limit_cases)]
        if args.no_debug:
            base_args += ["--no-debug"]

        env_local = os.environ.copy()
        env_local["USE_NEO4J_AURA"] = "0"
        env_local["RUN_TAG"] = "local"
        p_local = subprocess.Popen(base_args, env=env_local)

        env_aura = os.environ.copy()
        env_aura["USE_NEO4J_AURA"] = "1"
        env_aura["RUN_TAG"] = "aura"
        p_aura = subprocess.Popen(base_args, env=env_aura)

        # ✅ 进度监控：定期检查队列状态并显示进度
        print(f"\n{'='*60}")
        print(f"并行测试已启动 | 总任务数: {total} | Workers: local + aura")
        print(f"{'='*60}\n")
        
        import time as _time
        start_time = _time.time()
        last_completed = 0
        
        while p_local.poll() is None or p_aura.poll() is None:
            _time.sleep(2)  # 每2秒检查一次
            
            # 读取队列状态
            try:
                with open(args.queue_path, "r", encoding="utf-8") as f:
                    queue_data = json.load(f)
                pending = len(queue_data.get("pending", []))
                queue_total = queue_data.get("total", total)
                completed = queue_total - pending
                
                # 读取已完成的结果数
                result_count = 0
                if os.path.exists(args.queue_results):
                    with open(args.queue_results, "r", encoding="utf-8") as f:
                        result_count = sum(1 for _ in f)
                
                # 计算进度
                elapsed = _time.time() - start_time
                progress_pct = (completed / queue_total * 100) if queue_total > 0 else 0
                
                # 估算剩余时间
                if completed > 0:
                    avg_time = elapsed / completed
                    eta = avg_time * pending
                    eta_str = f"{eta/60:.1f}min" if eta > 60 else f"{eta:.0f}s"
                else:
                    eta_str = "计算中..."
                
                # 只在有新进度时更新显示
                if completed != last_completed:
                    last_completed = completed
                    # 进度条
                    bar_len = 30
                    filled = int(bar_len * completed / queue_total) if queue_total > 0 else 0
                    bar = "█" * filled + "░" * (bar_len - filled)
                    
                    print(f"\r[{bar}] {completed}/{queue_total} ({progress_pct:.1f}%) | 已保存: {result_count} | 耗时: {elapsed/60:.1f}min | ETA: {eta_str}    ", end="", flush=True)
            except Exception:
                pass
        
        # 最终进度
        elapsed = _time.time() - start_time
        print(f"\n\n{'='*60}")
        print(f"并行测试完成 | 总耗时: {elapsed/60:.1f} 分钟")
        print(f"{'='*60}\n")

        # merge results
        with open(data_path, "r", encoding="utf-8") as f:
            all_cases = json.load(f)
        total = len(all_cases)
        if args.limit_cases is not None:
            total = min(total, int(args.limit_cases))
        order = [c.get("question_id") for c in all_cases[:total]]
        merged_by_id: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(args.queue_results):
            with open(args.queue_results, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if row.get("question_id"):
                            merged_by_id[row["question_id"]] = row
                    except Exception:
                        continue
        merged = [merged_by_id[qid] for qid in order if qid in merged_by_id]
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        hypo_path = args.output_path.replace(".json", ".jsonl")
        with open(hypo_path, "w", encoding="utf-8") as f:
            for r in merged:
                # 清理 hypothesis：去掉 "Final Answer:" 前缀和 "\nEvidence:" 后面的内容
                raw_hypo = r.get("hypothesis", "")
                clean_hypo = _clean_hypothesis(raw_hypo)
                line = {"question_id": r["question_id"], "hypothesis": clean_hypo}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        # ✅ 清理临时文件
        _cleanup_queue_tmps()
        # 删除 queue.jsonl 和 queue.json
        for tmp_file in [args.queue_results, args.queue_path]:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass
        raise SystemExit(0)

    if args.queue_worker:
        # Avoid auto suffixing temp output names in queue mode
        os.environ.pop("RUN_TAG", None)
        while True:
            idx = _queue_pop(args.queue_path)
            if idx is None:
                break
            temp_output = os.path.join(PROJECT_ROOT, "test", f"queue_tmp_{os.getpid()}.json")
            run_longmemoryeval(
                data_path=data_path,
                output_path=temp_output,
                limit_cases=args.limit_cases,
                slice_indices=[idx],
            )
            try:
                cand_paths = [
                    temp_output,
                    temp_output.replace(".json", ".local.json"),
                    temp_output.replace(".json", ".aura.json"),
                ]
                found_path = next((p for p in cand_paths if os.path.exists(p)), None)
                if not found_path:
                    continue
                with open(found_path, "r", encoding="utf-8") as f:
                    rows = json.load(f) or []
                if rows:
                    _queue_append_result(args.queue_results, rows[0])
            except Exception:
                pass
            try:
                for p in [
                    temp_output,
                    temp_output.replace(".json", ".jsonl"),
                    temp_output.replace(".json", ".local.json"),
                    temp_output.replace(".json", ".local.jsonl"),
                    temp_output.replace(".json", ".aura.json"),
                    temp_output.replace(".json", ".aura.jsonl"),
                ]:
                    if os.path.exists(p):
                        os.remove(p)
            except Exception:
                pass
        raise SystemExit(0)

    if args.parallel:
        with open(data_path, "r", encoding="utf-8") as f:
            all_cases = json.load(f)
        total = len(all_cases)
        if args.limit_cases is not None:
            total = min(total, int(args.limit_cases))
        mid = total // 2

        base_args = [sys.executable, os.path.abspath(__file__), "--data", data_path]
        if args.output_path:
            base_args += ["--output", args.output_path]
        if args.limit_cases is not None:
            base_args += ["--limit", str(args.limit_cases)]
        if args.no_debug:
            base_args += ["--no-debug"]

        env_local = os.environ.copy()
        env_local["USE_NEO4J_AURA"] = "0"
        env_local["RUN_TAG"] = "local"
        p_local = subprocess.Popen(base_args + ["--start", "0", "--end", str(mid)], env=env_local)

        env_aura = os.environ.copy()
        env_aura["USE_NEO4J_AURA"] = "1"
        env_aura["RUN_TAG"] = "aura"
        p_aura = subprocess.Popen(base_args + ["--start", str(mid), "--end", str(total)], env=env_aura)

        p_local.wait()
        p_aura.wait()

        # Merge outputs into the default path (JSON + JSONL)
        def _load_json(path: str) -> List[Dict[str, Any]]:
            if not os.path.exists(path):
                return []
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f) or []
            except Exception:
                return []

        local_path = args.output_path.replace(".json", ".local.json")
        aura_path = args.output_path.replace(".json", ".aura.json")
        merged_rows = _load_json(local_path) + _load_json(aura_path)
        merged_by_id = {r.get("question_id"): r for r in merged_rows if r.get("question_id")}
        order = [c.get("question_id") for c in all_cases[:total]]
        merged = [merged_by_id[qid] for qid in order if qid in merged_by_id]

        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        hypo_path = args.output_path.replace(".json", ".jsonl")
        with open(hypo_path, "w", encoding="utf-8") as f:
            for r in merged:
                # 清理 hypothesis：去掉 "Final Answer:" 前缀和 "\nEvidence:" 后面的内容
                raw_hypo = r.get("hypothesis", "")
                clean_hypo = _clean_hypothesis(raw_hypo)
                line = {"question_id": r["question_id"], "hypothesis": clean_hypo}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        raise SystemExit(0)

    slice_indices = None
    if args.slice_indices:
        slice_indices = []
        for part in args.slice_indices.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                slice_indices.append(int(part))
            except Exception:
                pass

    run_longmemoryeval(
        data_path=args.data_path,
        output_path=args.output_path,
        limit_cases=args.limit_cases,
        slice_start=args.slice_start,
        slice_end=args.slice_end,
        slice_indices=slice_indices,
        force_full=args.full_dataset,
    )
