# -*- coding: utf-8 -*-
import sys
import os
import logging

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = _THIS_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.agent import CognitiveAgent
from memory.stores import LongTermSemanticStore

# ✅ 改：导入你新的世界书一键入图函数
# 你需要确保 utils/bootstrap_world_knowledge.py 里暴露了这个函数
from utils.bootstrap_world_knowledge import bootstrap_world_knowledge

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AgentInstance")
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


class AgentManager:
    """
    智能体实例管理器：负责 LTSS 连接、可选的（一次性）世界书/角色书写库、以及智能体实例化。
    """

    def __init__(
        self,
        character_name: str,
        auto_bootstrap: bool = False,
        allow_write_world_knowledge: bool = False,
    ):
        self.character_name = character_name
        self.ltss = None
        self.agent = None

        logger.info(f"正在唤醒智能体: {character_name}")

        # 1) 连接长期记忆数据库（不自动写入/不强制初始化）
        self.ltss = LongTermSemanticStore(bootstrap_now=False)

        # 2) 可选：世界书导入（一次性写库）
        if auto_bootstrap:
            if not allow_write_world_knowledge:
                logger.warning(">>> auto_bootstrap=True 但 allow_write_world_knowledge=False：已阻止写入世界知识。")
                logger.warning(">>> 如确认要写入，请将 allow_write_world_knowledge=True。")
            else:
                logger.warning(">>> 正在执行【世界书】一键结构化导入 Neo4j（仅建议运行一次）...")
                try:
                    # ✅ 你的世界书导入脚本里如果能复用 ltss 的 driver 也行
                    # 但最简单是 bootstrap_world_knowledge() 自己连 Neo4j 写入
                    bootstrap_world_knowledge()
                    logger.warning(">>> 世界书导入完成。")
                except Exception as e:
                    logger.error(f">>> 世界书导入失败: {e}", exc_info=True)

        else:
            logger.info(">>> 跳过世界知识引导，直接使用现有长期记忆。")

        # 3) 实例化认知智能体
        try:
            self.agent = CognitiveAgent(
                character_name=self.character_name,
                ltss_instance=self.ltss
            )
            logger.info(f"角色 '{character_name}' 加载完毕，记忆系统就绪。")
        except Exception as e:
            logger.error(f"智能体实例化失败: {e}", exc_info=True)
            try:
                if self.ltss:
                    self.ltss.close()
            finally:
                raise

    def chat(self, user_input: str) -> str:
        observation = f"User says: {user_input}"
        try:
            final_state = self.agent.run(observation)

            action_text = (
                final_state.get("action")
                or final_state.get("response")
                or final_state.get("output")
                or final_state.get("text")
                or ""
            )
            response_text = self._clean_response(str(action_text))
            self.save_state()
            return response_text
        except Exception as e:
            logger.error(f"对话处理出错: {e}", exc_info=True)
            return "（智能体似乎走神了...）"

    def _clean_response(self, text: str) -> str:
        if not text:
            return ""
        clean_text = text.strip()
        prefixes = [
            f"{self.character_name} says:",
            f"{self.character_name} says：",
            f"{self.character_name} 说:",
            f"{self.character_name} 说：",
            "says:", "says：",
            "Answer:", "Answer：",
            "Response:", "Response：",
            "Result:", "Result：",
        ]
        for prefix in prefixes:
            if clean_text.lower().startswith(prefix.lower()):
                clean_text = clean_text[len(prefix):].strip()
                break
        return clean_text.strip().strip('"').strip("'").strip("`").strip()

    def save_state(self):
        try:
            if self.agent and hasattr(self.agent, "memory") and self.agent.memory:
                if hasattr(self.agent.memory, "save_memory_system"):
                    self.agent.memory.save_memory_system()
        except Exception as e:
            logger.warning(f"保存记忆状态时出错: {e}", exc_info=True)

    def force_save(self):
        try:
            if self.agent and hasattr(self.agent, "memory") and self.agent.memory:
                if hasattr(self.agent.memory, "trigger_consolidation"):
                    self.agent.memory.trigger_consolidation()
                    logger.info("已强制执行末尾记忆巩固。")
        except Exception as e:
            logger.warning(f"强制巩固失败: {e}", exc_info=True)

    def close(self):
        logger.info("正在关闭智能体管理器...")
        try:
            self.save_state()
        finally:
            try:
                if self.ltss:
                    self.ltss.close()
            finally:
                logger.info("再见。")


if __name__ == "__main__":
    # 默认不写世界书，避免误触发
    bot = AgentManager("LoCoMoTester", auto_bootstrap=False)
    try:
        while True:
            q = input("\nUser: ")
            if q.lower() in ["exit", "quit"]:
                break
            res = bot.chat(q)
            print(f"Agent: {res}")
    except KeyboardInterrupt:
        pass
    finally:
        bot.close()
