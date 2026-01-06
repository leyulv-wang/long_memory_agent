import json
import random
import os

# 1. 定义路径
# SOURCE_PATH = r"D:\python\longmemeval\LongMemEval\data\longmemeval_oracle.json"#最小的
# SOURCE_PATH = r"D:\python\longmemeval\LongMemEval\data\longmemeval_s_cleaned.json"#中等的
SOURCE_PATH = r"D:\python\longmemeval\LongMemEval\data\longmemeval_m_cleaned.json"#最大的
TARGET_DIR = r"D:\python\智能体记忆框架\data\long_memory_eval"
# TARGET_PATH = os.path.join(TARGET_DIR, "sampled_test_questions.json")#最小的
# TARGET_PATH = os.path.join(TARGET_DIR, "medium_test_questions.json")#中等的
TARGET_PATH = os.path.join(TARGET_DIR, "max_test_questions.json")#最大的
def sample_longmem_tasks():
    # 确保目标目录存在
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"已创建目录: {TARGET_DIR}")

    # 2. 加载数据集
    if not os.path.exists(SOURCE_PATH):
        print(f"错误：找不到源文件 {SOURCE_PATH}")
        return

    with open(SOURCE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 3. 按 LongMemEval 的五个维度分组
    categories = {
        'IE (信息提取)': [d for d in data if d['question_type'] in ['single-session-user', 'single-session-assistant'] and '_abs' not in d['question_id']],
        'MR (多会话推理)': [d for d in data if d['question_type'] == 'multi-session' and '_abs' not in d['question_id']],
        'TR (时序推理)': [d for d in data if d['question_type'] == 'temporal-reasoning' and '_abs' not in d['question_id']],
        'KU (知识更新)': [d for d in data if d['question_type'] == 'knowledge-update'],
        'ABS (拒答/幻觉)': [d for d in data if '_abs' in d['question_id']]
    }

    # 4. 抽取样本
    sampled_tasks = []
    print("--- 抽取进度 ---")
    for cat_name, items in categories.items():
        # 如果某类样本不足4个，则取全部
        count = min(len(items), 1)
        sample = random.sample(items, count)
        sampled_tasks.extend(sample)
        print(f"维度 {cat_name}: 已随机抽取 {len(sample)}/{len(items)} 个题目")

    # 5. 保存结果
    with open(TARGET_PATH, 'w', encoding='utf-8') as f:
        json.dump(sampled_tasks, f, indent=4, ensure_ascii=False)

    print("-" * 20)
    print(f"成功！抽取的 20 个样本已保存至:\n{TARGET_PATH}")

if __name__ == "__main__":
    sample_longmem_tasks()