# -*- coding: utf-8 -*-
import os
import pickle
import subprocess
import collections
import sys
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union

import faiss
import numpy as np
from neo4j import GraphDatabase

from utils.embedding import get_embedding_model

# 保持原有配置导入（这些常量不一定都用到，但保留以避免外部依赖报错）
from config import (
    AGENTS_DATA_DIR,
    FAISS_INDEX_FILENAME,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    SHORT_TERM_MEMORY_WINDOW,
    VECTOR_INDEX_META_KEY,
    FAIL_FAST_ON_VECTOR_INDEX_MISMATCH,
    VECTOR_DIM_FALLBACK,
)



class SimpleDocstore:
    """一个简单的内存键值存储..."""

    def __init__(self):
        self.mapping: Dict[int, str] = {}

    def add(self, documents: Dict[int, str]):
        self.mapping.update(documents)

    def search(self, index: int) -> Optional[str]:
        return self.mapping.get(index)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.mapping, f)

    @classmethod
    def load(cls, path: str):
        instance = cls()
        with open(path, "rb") as f:
            instance.mapping = pickle.load(f)
        return instance


class ShortTermEpisodicStore:
    """
    【已优化】短期情节记忆存储 (STES)。
    保留滑动窗口 (Window) 机制，但采用增量 Embedding 策略 + dirty 延迟重建索引。
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.embedding_model = get_embedding_model()

        # 兼容：dual_memory_system.py 里有的版本检查 autosave，有的检查 auto_save
        self.auto_save = True
        self.autosave = True  # alias

        self._dirty = False

        agent_data_dir = os.path.join(AGENTS_DATA_DIR, self.agent_name)
        os.makedirs(agent_data_dir, exist_ok=True)
        self.index_path = os.path.join(agent_data_dir, "memory.faiss")

        self.memory_window = collections.deque(maxlen=SHORT_TERM_MEMORY_WINDOW)
        self.window_embeddings = collections.deque(maxlen=SHORT_TERM_MEMORY_WINDOW)

        self.index = None
        self._load_or_initialize()

    def _initialize_empty_index(self):
        """初始化一个空索引。"""
        try:
            dummy_embedding = self.embedding_model.embed_query("get dimension")
            dimension = int(len(dummy_embedding))
            if dimension <= 0:
                raise ValueError("Bad embedding dimension")
        except Exception:
            dimension = int(VECTOR_DIM_FALLBACK) if VECTOR_DIM_FALLBACK else 1024
        self.index = faiss.IndexFlatIP(dimension)

    def _load_or_initialize(self):
        """加载现有记忆或创建一个新的空记忆库。"""
        memory_pkl_path = f"{self.index_path}.pkl"
        if os.path.exists(memory_pkl_path):
            print(f"STES for {self.agent_name}: Loading existing memories from '{memory_pkl_path}'...")
            try:
                with open(memory_pkl_path, "rb") as f:
                    loaded_memories = pickle.load(f) or []
                if isinstance(loaded_memories, list):
                    self.memory_window.extend([m for m in loaded_memories if isinstance(m, str) and m.strip()])
                self._rebuild_cache_and_index_startup()
                print(f"STES for {self.agent_name}: Loaded {len(self.memory_window)} memories.")
            except Exception as e:
                print(f"STES: 加载失败 ({e})，将创建新索引。")
                self._initialize_empty_index()
                self.memory_window.clear()
                self.window_embeddings.clear()
        else:
            print(f"STES for {self.agent_name}: Creating a new empty store.")
            self._initialize_empty_index()

    def _rebuild_cache_and_index_startup(self):
        """仅在启动时调用：根据加载的文本，全量计算一次向量并填充缓存。"""
        if not self.memory_window:
            self._initialize_empty_index()
            return

        print(f"STES for {self.agent_name}: Initializing embeddings cache (startup)...")
        documents = list(self.memory_window)
        embeddings = self.embedding_model.embed_documents(documents) or []

        self.window_embeddings.clear()
        self.window_embeddings.extend(embeddings)
        self._refresh_faiss_index()
        self._dirty = False

    def _refresh_faiss_index(self):
        """从 window_embeddings 刷新 FAISS 索引（纯内存操作）。"""
        if not self.window_embeddings:
            self._initialize_empty_index()
            return

        embeddings_np = np.array(list(self.window_embeddings), dtype=np.float32)

        if self.index is None or getattr(self.index, "d", None) != embeddings_np.shape[1]:
            self.index = faiss.IndexFlatIP(int(embeddings_np.shape[1]))
        else:
            self.index.reset()

        self.index.add(embeddings_np)

    # ---------- 兼容入口：dual_memory_system 可能传 str ----------
    def add(self, documents: Union[str, List[str]]):
        """
        添加新记忆：支持传入单条 str 或 list[str]。
        """
        if documents is None:
            return

        if isinstance(documents, str):
            docs = [documents]
        else:
            docs = list(documents)

        docs = [d.strip() for d in docs if isinstance(d, str) and d.strip()]
        if not docs:
            return

        new_embeddings = self.embedding_model.embed_documents(docs) or []
        for doc, emb in zip(docs, new_embeddings):
            self.memory_window.append(doc)
            self.window_embeddings.append(emb)

        self._dirty = True

        if getattr(self, "auto_save", True) and getattr(self, "autosave", True):
            self.save()

    # 旧名兼容
    def add_observation(self, observation: str):
        self.add(observation)

    def search(self, query: str, k: int) -> List[str]:
        """从 STES 中检索 k 个最相关的情节性记忆。"""
        if self._dirty or self.index is None:
            self._refresh_faiss_index()
            self._dirty = False

        if not self.memory_window or self.index is None:
            return []

        query_embedding = self.embedding_model.embed_query(query)
        query_embedding_np = np.array([query_embedding], dtype=np.float32)

        k = min(int(k), len(self.memory_window))
        if k <= 0:
            return []

        _, indices = self.index.search(query_embedding_np, k)
        current_memories = list(self.memory_window)

        results: List[str] = []
        for idx in indices[0]:
            if idx != -1 and idx < len(current_memories):
                results.append(current_memories[idx])
        return results

    def save(self):
        """只保存文本到磁盘（保持兼容性）。"""
        memory_pkl_path = f"{self.index_path}.pkl"
        with open(memory_pkl_path, "wb") as f:
            pickle.dump(list(self.memory_window), f)

    def clear(self):
        self.memory_window.clear()
        self.window_embeddings.clear()
        self._initialize_empty_index()
        self._dirty = False
        self.save()
        print(f"STES for {self.agent_name} has been cleared.")

    def get_most_recent(self, k: int) -> List[str]:
        if not self.memory_window:
            return []
        k = min(int(k), len(self.memory_window))
        return list(self.memory_window)[-k:]

    # 旧名兼容：dual_memory_system 可能会找 get_most_recent_k
    def get_most_recent_k(self, k: int) -> List[str]:
        return self.get_most_recent(k)


class LongTermSemanticStore:
    """
    长期语义知识存储 (LTSS)。
    """

    def __init__(self, bootstrap_now: bool = True, setup_schema: bool = True):
        print("LTSS: 正在初始化 Neo4j 连接...")
        self.embedding_model = get_embedding_model()

        self.embedding_dim, self.embedding_model_name = self._detect_embedding_dim_and_model()
        print(f"LTSS: embedding probe => dim={self.embedding_dim}, model={self.embedding_model_name}")

        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
                keep_alive=True,
                max_connection_lifetime=3600,
                liveness_check_timeout=2,
            )
            self.driver.verify_connectivity()
            print("LTSS: Neo4j 连接成功。")

            if setup_schema:
                self._setup_schema()
                self._ensure_vector_index_meta_consistency_or_fail(self.embedding_dim, self.embedding_model_name)
                self._setup_vector_indices()
                self._setup_fulltext_indices()

            if bootstrap_now:
                self._bootstrap_world_knowledge()

        except RuntimeError:
            raise
        except Exception as e:
            print(f"LTSS: 无法连接到 Neo4j: {e}")
            self.driver = None

    def _detect_embedding_dim_and_model(self):
        model_name = (
            getattr(self.embedding_model, "model", None)
            or getattr(self.embedding_model, "model_name", None)
            or getattr(self.embedding_model, "deployment", None)
            or "unknown"
        )
        try:
            v = self.embedding_model.embed_query("dim probe")
            dim = len(v)
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"Bad embedding dim: {dim}")
            return dim, str(model_name)
        except Exception as e:
            print(f"LTSS: embedding dim probe failed, fallback to {VECTOR_DIM_FALLBACK}. err={e}")
            return int(VECTOR_DIM_FALLBACK) if VECTOR_DIM_FALLBACK else 1024, str(model_name)

    def _get_vector_index_meta(self):
        cypher = """
        MATCH (m:SchemaMeta {key: $key})
        RETURN m.embedding_dim AS dim,
               m.embedding_model AS model,
               m.created_at AS created_at
        """
        with self.driver.session() as session:
            rec = session.run(cypher, key=VECTOR_INDEX_META_KEY).single()
            if not rec:
                return None
            return {"dim": rec.get("dim"), "model": rec.get("model"), "created_at": rec.get("created_at")}

    def _set_vector_index_meta(self, dim: int, model: str):
        cypher = """
        MERGE (m:SchemaMeta {key: $key})
        ON CREATE SET m.embedding_dim = $dim,
                      m.embedding_model = $model,
                      m.created_at = $now
        ON MATCH SET  m.embedding_dim = $dim,
                      m.embedding_model = $model
        RETURN m.embedding_dim AS dim, m.embedding_model AS model
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.driver.session() as session:
            session.run(cypher, key=VECTOR_INDEX_META_KEY, dim=int(dim), model=str(model), now=now)

    def _ensure_vector_index_meta_consistency_or_fail(self, dim: int, model: str):
        meta = self._get_vector_index_meta()
        if meta is None:
            print(f"LTSS: SchemaMeta not found, creating meta: dim={dim}, model={model}")
            self._set_vector_index_meta(dim, model)
            return

        meta_dim = meta.get("dim")
        meta_model = meta.get("model")

        dim_mismatch = (meta_dim is not None) and (int(meta_dim) != int(dim))
        model_mismatch = (meta_model is not None) and (str(meta_model) != str(model))

        if dim_mismatch or model_mismatch:
            msg = (
                "LTSS: ❌ Vector index meta mismatch detected.\n"
                f"- DB meta: dim={meta_dim}, model={meta_model}\n"
                f"- Current: dim={dim}, model={model}\n"
                "Action required: you must rebuild vector indexes and (recommended) re-embed stored nodes.\n"
                "Because FAIL_FAST_ON_VECTOR_INDEX_MISMATCH=True, the program will exit now."
            )
            print(msg)
            if FAIL_FAST_ON_VECTOR_INDEX_MISMATCH:
                raise RuntimeError(msg)
            print("LTSS: WARNING: mismatch detected but FAIL_FAST disabled; continuing (may degrade retrieval).")

    def _setup_fulltext_indices(self):
        print("LTSS: 正在检查并创建全文索引...")
        with self.driver.session() as session:
            session.run(
                """
                CREATE FULLTEXT INDEX global_name_index IF NOT EXISTS
                FOR (n:Agent|Location|Object|Concept|Event|Value|Date)
                ON EACH [n.name]
                """
            )

    def _setup_vector_indices(self):
        print("LTSS: 正在检查并创建向量索引...")

        target_labels = [
            "Agent", "Location", "Object", "Concept", "Event", "Value", "Date",
            "Duration", "Action", "Trait", "Organization", "Skill", "TextUnit",
        ]
        embedding_dimension = int(self.embedding_dim)

        with self.driver.session() as session:
            for label in target_labels:
                index_name = f"{label.lower()}_vector_index"
                cypher = f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:`{label}`)
                ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {embedding_dimension},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
                try:
                    session.run(cypher)
                except Exception as e:
                    print(f"LTSS: 创建索引 {index_name} 警告 (可能是版本兼容问题): {e}")

        print("LTSS: 向量索引检查完毕。")

    def _bootstrap_world_knowledge(self):
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count").single()
                node_count = int(result["count"]) if result and "count" in result else 0
        except Exception:
            node_count = 0

        if node_count != 0:
            print(f"\n--- LTSS: 检测到数据库中已存在 {node_count} 个节点，跳过知识构建。 ---")
            return

        print("\n--- LTSS: 数据库为空, 正在自动运行引导程序... ---")
        try:
            current_dir = os.path.dirname(__file__)
            bootstrap_script_path = os.path.abspath(
                os.path.join(current_dir, "..", "utils", "bootstrap_world_knowledge.py")
            )
            if not os.path.exists(bootstrap_script_path):
                print(f"--- 错误: 无法找到引导脚本 at {bootstrap_script_path} ---")
                return

            result = subprocess.run(
                [sys.executable, bootstrap_script_path],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
            )
            print("--- 引导程序输出 ---")
            print(result.stdout)
            if result.stderr:
                print("--- 引导程序错误 ---")
                print(result.stderr)
            print("--- ✅ 世界知识自动引导完成。 ---")

        except subprocess.CalledProcessError as e:
            print("--- ❌ 世界知识自动引导失败 ---")
            print(e.stdout)
            print(e.stderr)
        except FileNotFoundError:
            print("--- 错误: 无法找到 python 命令来执行引导脚本。 ---")

    def _setup_schema(self):
        self._setup_constraints()

    def _setup_constraints(self):
        print("LTSS: 正在检查并创建数据库约束...")

        agent_semantic_labels = [
            "Entity", "Agent", "Location", "Object", "Concept", "Event",
            "Procedure", "SecretArea", "ResidentialBuilding",
            "CommercialBuilding", "PublicSpace", "Mechanism",
            "Room", "Activity", "FacilityType", "BuildingType",
            "SpaceType", "StructureType", "Door", "Kitchen",
            "SeatingArea", "SocialHub",
            "Value", "Date", "Duration", "Action", "Trait", "Organization", "Skill",
        ]

        with self.driver.session() as session:
            for label in agent_semantic_labels:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.id IS UNIQUE"
                )
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.name IS UNIQUE"
                )

            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (u:TextUnit) REQUIRE u.name IS UNIQUE"
            )

        print("LTSS: 智能体语义节点约束设置完成 (基于 ID)。")

    def retrieve_textunits(
            self,
            query: str,
            agent_name: str = "all",  # ✅ 允许不传
            k: int = 5,
            top_k: int = None,  # ✅ 兼容 top_k
            **kwargs
    ):
        if top_k is not None:
            k = top_k
        if not agent_name:
            agent_name = "all"
        if not self.driver:
            return []
        # ---- 下面保持你原本函数体不变 ----
        embedding = self.embedding_model.embed_query(query)
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $k, $embedding) YIELD node, score
        WHERE
            node.accessible_by IS NULL
            OR 'all' IN node.accessible_by
            OR $agent_name IN node.accessible_by
        RETURN
            node.name AS name,
            node.content AS content,
            node.channel AS channel,
            node.turn_id AS turn_id,
            node.virtual_time AS virtual_time,
            score
        """
        try:
            return self.query_graph(
                cypher,
                {
                    "index_name": "textunit_vector_index",
                    "k": int(k),
                    "embedding": embedding,
                    "agent_name": agent_name,
                },
            ) or []
        except Exception as e:
            if os.getenv("DEBUG_VECTOR_QUERY", "0") == "1":
                print(f"LTSS: retrieve_textunits failed: {e}")
            return []

    def retrieve_knowledge(
            self,
            query: str,
            agent_name: str = "all",  # ✅ 允许不传
            k: int = 5,
            top_k: int = None,  # ✅ 兼容 top_k
            **kwargs
    ):
        if top_k is not None:
            k = top_k
        if not agent_name:
            agent_name = "all"
        if not self.driver:
            return []
        # ---- 下面保持你原本函数体不变 ----
        embedding = self.embedding_model.embed_query(query)
        results = []

        target_labels = [
            "Agent", "Location", "Object", "Concept", "Event",
            "Value", "Date", "Duration", "Action", "Trait", "Organization", "Skill",
        ]

        for label in target_labels:
            index_name = f"{label.lower()}_vector_index"
            cypher = """
            CALL db.index.vector.queryNodes($index_name, $k, $embedding) YIELD node, score
            WHERE
                node.accessible_by IS NULL
                OR 'all' IN node.accessible_by
                OR $agent_name IN node.accessible_by
            RETURN
                node.name AS name,
                labels(node) AS labels,
                node.description AS description,
                node.value AS value,
                node.id AS id,
                node.source_of_belief AS source_of_belief,
                node.confidence AS confidence,
                score
            """
            try:
                data = self.query_graph(
                    cypher,
                    {"index_name": index_name, "k": int(k), "embedding": embedding, "agent_name": agent_name},
                ) or []
                for item in data:
                    item["source_label"] = label
                results.extend(data)
            except Exception as e:
                if os.getenv("DEBUG_VECTOR_QUERY", "0") == "1":
                    print(f"LTSS: vector query failed on index={index_name}: {e}")

        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[: int(k)]

    def close(self):
        if self.driver:
            self.driver.close()
            print("LTSS: Neo4j 连接已关闭。")

    def _execute_query(self, query: str, parameters=None, db=None, write: bool = False):
        if not self.driver:
            print("LTSS: 驱动未初始化，无法执行查询。")
            return None

        with self.driver.session(database=db) as session:
            def _run(tx):
                return list(tx.run(query, parameters))

            if write:
                if hasattr(session, "execute_write"):
                    result = session.execute_write(_run)
                else:
                    result = session.write_transaction(_run)
            else:
                if hasattr(session, "execute_read"):
                    result = session.execute_read(_run)
                else:
                    result = session.read_transaction(_run)

        return [record.data() for record in result]

    def update_graph(self, cypher_query: str, parameters=None):
        """写入图谱前，自动处理 parameters['props'] 里的嵌套 dict。"""
        if parameters and isinstance(parameters, dict) and "props" in parameters and isinstance(parameters["props"], dict):
            processed_props = {}
            for k, v in parameters["props"].items():
                if isinstance(v, dict):
                    processed_props[k] = json.dumps(v, ensure_ascii=False)
                else:
                    processed_props[k] = v
            parameters["props"] = processed_props

        return self._execute_query(cypher_query, parameters=parameters, write=True)

    def query_graph(self, cypher_query: str, parameters=None):
        return self._execute_query(cypher_query, parameters=parameters, write=False)

    def clear_database(self):
        print("LTSS: 正在执行深度清理 (删除数据 + 架构)...")
        self.update_graph("MATCH (n) DETACH DELETE n")
        print("LTSS: 所有节点和关系已删除。")
        self._clear_schema()
        print("LTSS: 数据库架构已重置 (约束与索引已清除)。")

    def _clear_schema(self):
        with self.driver.session() as session:
            constraints = []
            try:
                result = session.run("SHOW CONSTRAINTS YIELD name")
                constraints = [record["name"] for record in result]
            except Exception:
                try:
                    result = session.run("CALL db.constraints() YIELD name")
                    constraints = [record["name"] for record in result]
                except Exception as e:
                    print(f"LTSS: 获取约束列表失败: {e}")

            for name in constraints:
                try:
                    session.run(f"DROP CONSTRAINT {name}")
                except Exception as e:
                    print(f"LTSS: 删除约束 {name} 失败: {e}")

            indexes = []
            try:
                result = session.run("SHOW INDEXES YIELD name")
                indexes = [record["name"] for record in result]
            except Exception:
                try:
                    result = session.run("CALL db.indexes() YIELD name")
                    indexes = [record["name"] for record in result]
                except Exception as e:
                    print(f"LTSS: 获取索引列表失败: {e}")

            for name in indexes:
                if not name:
                    continue
                try:
                    session.run(f"DROP INDEX {name}")
                except Exception:
                    pass
