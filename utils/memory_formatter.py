# utils/memory_formatter.py
# 工具脚本：将检索到的知识转化为 LLM 可理解的 Prompt 格式
# 这里的逻辑代替了测试脚本中的“数学公式”，让 LLM 负责推理。

import datetime
from typing import List, Dict, Any


def format_knowledge_for_prompt(memories: List[Dict[str, Any]], agent_name: str) -> str:
    """
    将从 LongTermSemanticStore.retrieve_knowledge 获取的原始数据
    转换为包含丰富元数据的自然语言文本块。

    Args:
        memories: stores.py 返回的字典列表
        agent_name: 当前智能体的名字（用于定制化提示）

    Returns:
        str: 格式化后的字符串，可直接拼接到 Agent 的 Prompt 中
    """
    if not memories:
        return f"当前没有与 {agent_name} 的观察或意图相关的长期知识。"

    formatted_blocks = []

    for i, mem in enumerate(memories):
        # 1. 提取核心内容
        name = mem.get('name', '未知概念')
        desc = mem.get('description', '无详细描述')

        # 2. 提取元数据 (你的创新点)
        source = mem.get('source_of_belief', '未知来源')
        confidence = mem.get('confidence', 0.0)
        timestamp = mem.get('consolidated_at', '未知时间')
        score = mem.get('score', 0.0)

        # 尝试格式化时间戳
        date_str = "未知时间"
        try:
            if timestamp:
                # 简化时间显示，只保留到分钟
                if 'T' in str(timestamp):
                    # 处理 ISO 格式
                    dt_obj = datetime.datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                    date_str = dt_obj.strftime("%Y-%m-%d %H:%M")
                else:
                    date_str = str(timestamp)
        except:
            date_str = str(timestamp)  # 解析失败则直接显示

        # 3. 构建 Prompt 块
        # 使用自然语言和结构化格式，引导 LLM 注意这些细节
        block = (
            f"\n--- 长期记忆片段 #{i + 1} (主题: {name}) ---\n"
            f"**记录时间**: {date_str}\n"
            f"**内容描述**: {desc}\n"
            f"**信息来源**: {source}\n"
            f"**置信度**: {confidence} / 1.0 (高分意味着更可靠)\n"
            f"**检索相似度**: {score:.4f} (越接近1.0越相关)"
        )
        formatted_blocks.append(block)

    # 添加给 LLM 的指导语 (System Note)
    header = (
        "### 长期记忆检索结果\n"
        "以下是与你当前观察和目标相关的历史记忆片段。你必须结合这些信息进行判断：\n"
        "1. **批判性评估**: 优先采信**置信度高**且**记录时间最新**的信息。\n"
        "2. **来源鉴别**: 评估信息来源的可靠性（例如 'ground_truth' 优于 'rumor'）。\n"
    )

    return header + "\n".join(formatted_blocks)