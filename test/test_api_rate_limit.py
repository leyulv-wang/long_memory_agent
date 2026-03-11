# -*- coding: utf-8 -*-
"""
测试 API 速率限制

测试方法：
1. 逐步增加并发数，观察是否触发限流
2. 记录每个并发级别的成功率和延迟
"""
import os
import sys
import time
import concurrent.futures
from typing import List, Tuple

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm import get_llm

def single_call(call_id: int) -> Tuple[int, bool, float, str]:
    """
    单次 API 调用
    
    Returns:
        (call_id, success, latency, error_msg)
    """
    llm = get_llm()
    start = time.time()
    try:
        response = llm.invoke("Say 'ok'")
        latency = time.time() - start
        return (call_id, True, latency, "")
    except Exception as e:
        latency = time.time() - start
        error_msg = str(e)
        # 检查是否是速率限制错误
        if "rate" in error_msg.lower() or "429" in error_msg or "limit" in error_msg.lower():
            return (call_id, False, latency, f"RATE_LIMIT: {error_msg[:100]}")
        return (call_id, False, latency, error_msg[:100])


def test_concurrent(n_workers: int, n_calls: int = None) -> dict:
    """
    测试指定并发数
    
    Args:
        n_workers: 并发数
        n_calls: 总调用次数（默认等于并发数）
    """
    if n_calls is None:
        n_calls = n_workers
    
    print(f"\n{'='*60}")
    print(f"测试并发数: {n_workers}, 总调用数: {n_calls}")
    print(f"{'='*60}")
    
    results: List[Tuple[int, bool, float, str]] = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        start = time.time()
        futures = [ex.submit(single_call, i) for i in range(n_calls)]
        
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            results.append(result)
            call_id, success, latency, error = result
            status = "✓" if success else "✗"
            if error:
                print(f"  [{call_id:2d}] {status} {latency:.2f}s - {error}")
            else:
                print(f"  [{call_id:2d}] {status} {latency:.2f}s")
        
        total_time = time.time() - start
    
    # 统计
    successes = [r for r in results if r[1]]
    failures = [r for r in results if not r[1]]
    rate_limits = [r for r in failures if "RATE_LIMIT" in r[3]]
    
    stats = {
        "n_workers": n_workers,
        "n_calls": n_calls,
        "total_time": total_time,
        "success_count": len(successes),
        "failure_count": len(failures),
        "rate_limit_count": len(rate_limits),
        "success_rate": len(successes) / n_calls * 100,
        "avg_latency": sum(r[2] for r in successes) / len(successes) if successes else 0,
        "throughput": len(successes) / total_time if total_time > 0 else 0,
    }
    
    print(f"\n统计:")
    print(f"  成功: {stats['success_count']}/{n_calls} ({stats['success_rate']:.1f}%)")
    print(f"  失败: {stats['failure_count']} (其中速率限制: {stats['rate_limit_count']})")
    print(f"  总耗时: {stats['total_time']:.2f}s")
    print(f"  平均延迟: {stats['avg_latency']:.2f}s")
    print(f"  吞吐量: {stats['throughput']:.1f} req/s")
    
    return stats


def run_progressive_test():
    """
    逐步增加并发数测试
    """
    print("=" * 60)
    print("API 速率限制测试")
    print("=" * 60)
    
    # 先测试单个请求，确保 API 可用
    print("\n1. 测试单个请求...")
    call_id, success, latency, error = single_call(0)
    if not success:
        print(f"❌ API 不可用: {error}")
        return
    print(f"✓ API 可用，延迟: {latency:.2f}s")
    
    # 逐步增加并发数
    print("\n2. 逐步增加并发数测试...")
    
    concurrency_levels = [5, 10, 15, 20, 30, 50]
    all_stats = []
    
    for n in concurrency_levels:
        stats = test_concurrent(n)
        all_stats.append(stats)
        
        # 如果出现大量速率限制错误，停止测试
        if stats["rate_limit_count"] > n * 0.3:  # 超过 30% 被限流
            print(f"\n⚠️ 并发数 {n} 触发大量速率限制，停止测试")
            break
        
        # 等待一下，避免累积触发限流
        time.sleep(2)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"{'并发数':>8} | {'成功率':>8} | {'吞吐量':>10} | {'平均延迟':>10} | {'限流数':>6}")
    print("-" * 60)
    for s in all_stats:
        print(f"{s['n_workers']:>8} | {s['success_rate']:>7.1f}% | {s['throughput']:>8.1f}/s | {s['avg_latency']:>9.2f}s | {s['rate_limit_count']:>6}")
    
    # 推荐并发数
    best = max([s for s in all_stats if s['rate_limit_count'] == 0], 
               key=lambda x: x['throughput'], default=None)
    if best:
        print(f"\n推荐并发数: {best['n_workers']} (吞吐量: {best['throughput']:.1f} req/s)")


if __name__ == "__main__":
    run_progressive_test()
