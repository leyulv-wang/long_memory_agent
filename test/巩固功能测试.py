
import sys
import os
import logging
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 将项目根目录添加到 Python 路径中

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.stores import LongTermSemanticStore
from memory.dual_memory_system import DualMemorySystem
from config import AGENTS_DATA_DIR

# 配置日志显示到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ConsolidationTest")


def test_consolidation():
    logger.info("=== 开始记忆巩固专项测试 ===")

    agent_name = "test_dummy_agent"

    # 1. 清理旧的测试数据
    agent_dir = os.path.join(AGENTS_DATA_DIR, agent_name)
    if os.path.exists(agent_dir):
        shutil.rmtree(agent_dir)
        logger.info(f"清理了旧的测试目录: {agent_dir}")

    # 2. 初始化数据库连接
    ltss = LongTermSemanticStore(bootstrap_now=False)

    try:
        # 3. 初始化双重记忆系统
        # 我们直接测试 DualMemorySystem，不加载整个 Agent，这样更快
        memory_system = DualMemorySystem(agent_name=agent_name, ltss_instance=ltss)

        # 4. 注入足够的短期记忆以触发阈值 (假设阈值是 5)
        logger.info(">>> 正在注入 10 条测试记忆...")
        test_memories = [
            "今天早上阳光明媚，我在花园里散步。",
            "我看到邻居 Tom 在修理他的红色卡车。",
            "Tom 告诉我，他觉得镇上的面包店最近换了面粉。",
            "我在面包店门口遇到了 Lisa，她看起来很匆忙。",
            "Lisa 说她要把一份重要文件送到市政厅。",
            "我觉得 Lisa 有点神神秘秘的。",
            "中午我在公园长椅上吃了一个三明治。",
            "我在喷泉旁捡到了一把生锈的钥匙。",
            "我怀疑这把钥匙能打开城堡的地下室。",
            "我决定晚上去试一试这把钥匙。"
        ]

        for mem in test_memories:
            memory_system.add_episodic_memory(mem)

        logger.info(f"成功注入 {len(test_memories)} 条记忆。")

        # 5. 手动触发巩固
        logger.info("\n>>> ⚡ 正在触发记忆巩固 (Trigger Consolidation)...")
        memory_system.trigger_consolidation()

        logger.info("\n=== ✅ 测试结束：如果没有报错，说明巩固功能修复成功！===")

    except Exception as e:
        logger.error(f"\n=== ❌ 测试失败: {e} ===", exc_info=True)
    finally:
        if ltss:
            ltss.close()


if __name__ == "__main__":
    test_consolidation()