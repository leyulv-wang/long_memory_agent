# -*- coding: utf-8 -*-
"""
并行测试脚本 - 双数据库并行

使用两个独立的 Neo4j 实例并行测试：
- 进程 A：使用 NEO4J_URI (7687)
- 进程 B：使用 NEO4J_AURA_URI (7690)

用法：
    # 双进程并行测试全部数据
    python test/parallel_test.py
    
    # 限制测试样本数
    python test/parallel_test.py --limit 10
    
    # 单进程测试（使用默认数据库）
    python test/parallel_test.py --single
"""

import argparse
import json
import logging
import os
import sys
import time
import multiprocessing as mp
from typing import List, Dict, Any

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("parallel_test")

DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "long_memory_eval", "sampled_test_questions.json")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "test", "long_memory_results.parallel.json")


def worker_process(
    cases: List[Dict[str, Any]],
    worker_id: int,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pass: str,
    result_queue: mp.Queue,
    start_idx: int,
    total_cases: int,
    log_dir: str = None,
):
    """
    工作进程：处理分配的测试样本
    
    Args:
        cases: 分配给该进程的测试样本
        worker_id: 进程 ID (0 或 1)
        neo4j_uri: Neo4j 连接 URI
        neo4j_user: Neo4j 用户名
        neo4j_pass: Neo4j 密码
        result_queue: 结果队列
        start_idx: 起始索引（用于日志）
        total_cases: 总样本数（用于日志）
        log_dir: 日志目录
    """
    # 设置独立的日志文件
    proc_name = f"Worker-{worker_id}"
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"worker_{worker_id}.txt")
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"[{proc_name}] 日志文件: {log_file}")
    
    # 设置环境变量（覆盖默认配置）
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USERNAME"] = neo4j_user
    os.environ["NEO4J_PASSWORD"] = neo4j_pass
    
    # 重新导入模块（使用新的环境变量）
    import importlib
    import config
    importlib.reload(config)
    
    from The_agent import AgentManager
    from memory.stores import LongTermSemanticStore
    from utils.clear_short_long_memory import clear_short_term_memory, fast_clear_neo4j_data
    from utils.ingest_longmemoryeval import ingest_longmemoryeval_sample
    
    proc_name = f"Worker-{worker_id}"
    character_name = "LongMemory"
    
    logger.info(f"[{proc_name}] 启动，使用数据库: {neo4j_uri}")
    logger.info(f"[{proc_name}] 分配样本数: {len(cases)}")
    
    results = []
    
    for i, case in enumerate(cases):
        global_idx = start_idx + i + 1
        q_id = case.get("question_id", f"case_{global_idx:04d}")
        question = case.get("question", "")
        gt = case.get("answer", "")
        
        logger.info(f"[{proc_name}] [{global_idx}/{total_cases}] 开始: {q_id}")
        
        result = {
            "question_id": q_id,
            "hypothesis": "",
            "ground_truth": gt,
            "question_time": "TURN_1",
            "worker_id": worker_id,
            "error": None,
        }
        
        try:
            # A) 清 Neo4j
            temp_ltss = LongTermSemanticStore(bootstrap_now=False)
            fast_clear_neo4j_data(temp_ltss)
            temp_ltss.close()
            
            # B) 实例化智能体
            bot = AgentManager(character_name, auto_bootstrap=False)
            agent = bot.agent
            
            try:
                if hasattr(agent.memory.stes, "autosave"):
                    agent.memory.stes.autosave = False
            except Exception:
                pass
            
            try:
                agent.memory.consolidation_llm = agent.llm
            except Exception:
                pass
            
            # C) 注入数据
            ingest_res = ingest_longmemoryeval_sample(
                agent=agent,
                case=case,
                batch_size=10,
                overlap_sessions=0,
                include_assistant=True,
                show_progress=False,
                consolidation_max_workers=4,
                consolidation_mode="original_text",
            )
            
            question_turn_id = int(ingest_res.get("question_turn_id", 1) or 1)
            question_time = f"TURN_{question_turn_id}"
            result["question_time"] = question_time
            
            # D) 问答
            original_llm = agent.llm
            if hasattr(agent, "expensive_llm") and agent.expensive_llm is not None:
                agent.llm = agent.expensive_llm
            
            prediction = agent.run(question, current_time=question_time).get("action", "")
            agent.llm = original_llm
            
            result["hypothesis"] = prediction
            
            # E) 关闭
            bot.close()
            clear_short_term_memory(target_agent=character_name)
            
            logger.info(f"[{proc_name}] [{global_idx}/{total_cases}] 完成: {q_id}")
            
        except Exception as e:
            logger.error(f"[{proc_name}] [{global_idx}/{total_cases}] 失败: {q_id} - {e}")
            result["error"] = str(e)
            try:
                bot.close()
            except Exception:
                pass
        
        results.append(result)
        result_queue.put(result)
    
    logger.info(f"[{proc_name}] 完成所有样本")


def main():
    parser = argparse.ArgumentParser(description="双数据库并行测试 LongMemoryEval")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="数据集路径")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="输出路径")
    parser.add_argument("--limit", type=int, default=None, help="限制测试样本数")
    parser.add_argument("--single", action="store_true", help="单进程模式（不并行）")
    args = parser.parse_args()
    
    # 加载数据
    if not os.path.exists(args.data):
        logger.error(f"数据文件不存在: {args.data}")
        return
    
    with open(args.data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    if args.limit:
        test_data = test_data[:args.limit]
    
    total = len(test_data)
    
    # 读取两个数据库配置
    from dotenv import load_dotenv
    load_dotenv()
    
    db1_uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    db1_user = os.getenv("NEO4J_USERNAME", "neo4j")
    db1_pass = os.getenv("NEO4J_PASSWORD", "")
    
    db2_uri = os.getenv("NEO4J_AURA_URI", "neo4j://127.0.0.1:7690")
    db2_user = os.getenv("NEO4J_AURA_USERNAME", "neo4j")
    db2_pass = os.getenv("NEO4J_AURA_PASSWORD", "")
    
    logger.info("=" * 60)
    logger.info("并行测试 LongMemoryEval")
    logger.info(f"样本数: {total}")
    logger.info(f"数据库 1: {db1_uri}")
    logger.info(f"数据库 2: {db2_uri}")
    logger.info(f"输出: {args.output}")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    if args.single:
        # 单进程模式
        logger.info("单进程模式")
        result_queue = mp.Queue()
        worker_process(test_data, 0, db1_uri, db1_user, db1_pass, result_queue, 0, total)
        
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
    else:
        # 双进程并行模式
        mid = total // 2
        cases_1 = test_data[:mid]
        cases_2 = test_data[mid:]
        
        logger.info(f"进程 0: 样本 1-{mid}")
        logger.info(f"进程 1: 样本 {mid+1}-{total}")
        
        result_queue = mp.Queue()
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        
        p1 = mp.Process(
            target=worker_process,
            args=(cases_1, 0, db1_uri, db1_user, db1_pass, result_queue, 0, total, log_dir),
            name="Worker-0",
        )
        p2 = mp.Process(
            target=worker_process,
            args=(cases_2, 1, db2_uri, db2_user, db2_pass, result_queue, mid, total, log_dir),
            name="Worker-1",
        )
        
        p1.start()
        p2.start()
        
        # 收集结果
        results = []
        completed = 0
        
        while completed < total:
            try:
                result = result_queue.get(timeout=300)  # 5分钟超时
                results.append(result)
                completed += 1
                
                # 定期保存中间结果
                if completed % 5 == 0:
                    # 按 question_id 排序
                    sorted_results = sorted(results, key=lambda x: x.get("question_id", ""))
                    with open(args.output, "w", encoding="utf-8") as f:
                        json.dump(sorted_results, f, ensure_ascii=False, indent=2)
                    logger.info(f"已保存 {completed}/{total} 个结果")
                    
            except Exception as e:
                logger.warning(f"等待结果超时: {e}")
                break
        
        p1.join(timeout=60)
        p2.join(timeout=60)
        
        if p1.is_alive():
            p1.terminate()
        if p2.is_alive():
            p2.terminate()
    
    elapsed = time.time() - start_time
    
    # 按 question_id 排序
    results = sorted(results, key=lambda x: x.get("question_id", ""))
    
    # 保存最终结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 统计
    success = sum(1 for r in results if not r.get("error"))
    failed = len(results) - success
    
    logger.info("=" * 60)
    logger.info("测试完成")
    logger.info(f"完成: {len(results)}/{total}")
    logger.info(f"成功: {success}")
    logger.info(f"失败: {failed}")
    logger.info(f"耗时: {elapsed:.1f}s ({elapsed/max(len(results),1):.1f}s/样本)")
    logger.info(f"结果: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
