认知架构智能体项目
本项目是《认知架构智能体的工程蓝图：从理论到实现》规划文档的完整Python代码实现。它构建了一个基于双记忆系统（情节性记忆和语义知识图谱）的智能体，并使用LangGraph编排其认知循环。

项目结构
/cognitive_agent         # 智能体核心逻辑
    /stes.py                 # 短期情节性记忆存储 (STES) - FAISS
    /ltss.py                 # 长期语义知识存储 (LTSS) - Neo4j
    /consolidation_engine.py # 记忆巩固引擎
    /agent_state.py          # LangGraph 状态定义
    /agent_nodes.py          # LangGraph 认知节点
    /agent_graph.py          # LangGraph 图构建与编译
/evaluation                # 自动化评估沙盒
    # ...
/verification              # 部署验证脚本
    # ...
main.py                    # 项目主入口/演示脚本
requirements.txt           # Python依赖
.env                       # 环境变量 (需自行创建)
AURA_SETUP_GUIDE.md      # [新] Neo4j AuraDB 设置指南
README.md                  # 本文档

设置步骤
1. 安装先决条件
Python 3.10+

一个 Neo4j AuraDB 云数据库实例 (推荐使用免费套餐)

2. 获取云数据库凭证
您需要从您的 Neo4j AuraDB 控制面板获取连接信息。

请参考 AURA_SETUP_GUIDE.md 文件获取详细的图文指导。

3. 设置Python环境
建议使用虚拟环境以避免包冲突。

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .\.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

4. 创建.env文件
在项目根目录创建一个名为.env的文件，并将您从云数据库获取的凭证填入其中。请不要将此文件提交到版本控制中。

.env 文件示例:

# OpenAI API Key
OPENAI_API_KEY="sk-..."

# Neo4j AuraDB Database Credentials
NEO4J_URI="neo4j+s://xxxxxxxx.databases.neo4j.io" # 替换成你的云数据库URI
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="YOUR_AURA_PASSWORD" # 替换成你的云数据库密码

# (可选) LangSmith API Key for Tracing
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="ls__..."

如何运行
1. 运行连接验证脚本
在开始构建复杂逻辑前，请先运行验证脚本，确保所有核心组件都已正确配置。

python verification/verify_llm.py
python verification/verify_neo4j.py

您应该会看到每个脚本都输出“验证成功”的信息。

2. 运行主程序演示
main.py 脚本提供了一个完整的端到端演示。

python main.py

观察终端输出，您可以看到智能体从感知、聚焦上下文、规划到行动的完整认知流程。

许可证
本项目采用 MIT 许可证。