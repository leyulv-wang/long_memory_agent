# -*- coding: utf-8 -*-
import os
import json
import time
import argparse
from typing import Dict, Any, List, Optional

import pandas as pd

from memory.stores import LongTermSemanticStore
from utils.embedding import get_embedding_model


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _to_json_safe(v: Any) -> Any:
    # Neo4j 属性不接受 dict；list 里如果嵌套 dict 也不稳，统一 JSON 化
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return v


def _batched(df: pd.DataFrame, batch_size: int):
    n = len(df)
    for i in range(0, n, batch_size):
        yield df.iloc[i : i + batch_size]


def import_text_units_as_chunks(
    ltss: LongTermSemanticStore,
    embedder,
    artifacts_dir: str,
    batch_size: int = 200,
    virtual_time: str = "OFFLINE",
    event_step: int = -1,
    agent_name: str = "__world__",
):
    path = os.path.join(artifacts_dir, "create_final_text_units.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_parquet(path, columns=["id", "text"])
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    cypher = """
    UNWIND $rows AS row
    MERGE (c:__Chunk__ {id: row.id})
    SET c.text = row.text,
        c.text_embedding = row.text_embedding,
        c.virtual_time = $vt,
        c.event_step = $step,
        c.real_time = $rt,
        c.agent_name = $agent
    RETURN count(*) AS n
    """.strip()

    for batch in _batched(df, batch_size):
        texts = batch["text"].tolist()
        embs = embedder.embed_documents(texts)  # 与在线一致的 embedding
        rows = []
        for (cid, text, emb) in zip(batch["id"].tolist(), texts, embs):
            if not cid or not text.strip():
                continue
            rows.append({"id": cid, "text": text, "text_embedding": emb})
        if not rows:
            continue
        ltss.update_graph(
            cypher,
            parameters={"rows": rows, "vt": virtual_time, "step": int(event_step), "rt": _now_iso(), "agent": agent_name},
        )


def import_entities(
    ltss: LongTermSemanticStore,
    embedder,
    artifacts_dir: str,
    batch_size: int = 200,
) -> Dict[str, str]:
    """
    返回 name(lower) -> entity_id 的映射，供 relationships 用
    """
    path = os.path.join(artifacts_dir, "create_final_entities.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # graphrag 常见列：name/type/description/id/text_unit_ids（你那个导入脚本就是这么读的）
    df = pd.read_parquet(path)
    for col in ["id", "name"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")

    # 允许字段缺失：description/type/text_unit_ids
    if "description" not in df.columns:
        df["description"] = ""
    if "type" not in df.columns:
        df["type"] = "Concept"
    if "text_unit_ids" not in df.columns:
        df["text_unit_ids"] = [[] for _ in range(len(df))]

    df["id"] = df["id"].astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["type"] = df["type"].fillna("Concept").astype(str)

    # 统一字段：type -> type_label
    df["type_label"] = df["type"].replace("", "Concept")

    # 为了保证与 SchemaMeta/向量索引一致：在你智能体项目里重算 description_embedding
    cypher_upsert = """
    UNWIND $rows AS row
    MERGE (e:__Entity__ {id: row.id})
    SET e.name = row.name,
        e.type_label = row.type_label,
        e.description = row.description,
        e.description_embedding = row.description_embedding,
        e.attrs_json = row.attrs_json
    RETURN count(*) AS n
    """.strip()

    name2id: Dict[str, str] = {}
    for batch in _batched(df, batch_size):
        descs = batch["description"].tolist()
        desc_embs = embedder.embed_documents(descs)

        rows = []
        for (_id, name, type_label, desc, emb, tu_ids) in zip(
            batch["id"].tolist(),
            batch["name"].tolist(),
            batch["type_label"].tolist(),
            descs,
            desc_embs,
            batch["text_unit_ids"].tolist(),
        ):
            if not _id or not name.strip():
                continue
            attrs = {"source": "graphrag_offline", "text_unit_ids": tu_ids}
            rows.append(
                {
                    "id": _id,
                    "name": name,
                    "type_label": type_label or "Concept",
                    "description": desc,
                    "description_embedding": emb,
                    "attrs_json": json.dumps(attrs, ensure_ascii=False),
                }
            )
            name2id[name.strip().lower()] = _id

        if rows:
            ltss.update_graph(cypher_upsert, parameters={"rows": rows})

    return name2id


def link_chunks_to_entities_by_text_unit_ids(
    ltss: LongTermSemanticStore,
    artifacts_dir: str,
    batch_size: int = 300,
):
    """
    用 entity.text_unit_ids 建立 (chunk)-[:HAS_ENTITY]->(entity)
    """
    path = os.path.join(artifacts_dir, "create_final_entities.parquet")
    df = pd.read_parquet(path, columns=["id", "text_unit_ids"])
    df["id"] = df["id"].astype(str)
    if "text_unit_ids" not in df.columns:
        return

    cypher = """
    UNWIND $rows AS row
    MATCH (e:__Entity__ {id: row.id})
    UNWIND row.text_unit_ids AS cid
    MATCH (c:__Chunk__ {id: toString(cid)})
    MERGE (c)-[:HAS_ENTITY]->(e)
    RETURN count(*) AS n
    """.strip()

    for batch in _batched(df, batch_size):
        rows = []
        for _id, tu_ids in zip(batch["id"].tolist(), batch["text_unit_ids"].tolist()):
            if not _id:
                continue
            if not isinstance(tu_ids, list) or not tu_ids:
                continue
            rows.append({"id": _id, "text_unit_ids": [str(x) for x in tu_ids if str(x).strip()]})
        if rows:
            ltss.update_graph(cypher, parameters={"rows": rows})


def import_relationships(
    ltss: LongTermSemanticStore,
    artifacts_dir: str,
    name2id: Dict[str, str],
    batch_size: int = 500,
    virtual_time: str = "OFFLINE",
    event_step: int = -1,
):
    path = os.path.join(artifacts_dir, "create_final_relationships.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    for col in ["id", "source", "target"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")

    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "text_unit_ids" not in df.columns:
        df["text_unit_ids"] = [[] for _ in range(len(df))]
    if "description" not in df.columns:
        df["description"] = ""

    df["id"] = df["id"].astype(str)
    df["source"] = df["source"].fillna("").astype(str)
    df["target"] = df["target"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    cypher = """
    UNWIND $rows AS row
    MATCH (s:__Entity__ {id: row.s_id})
    MATCH (t:__Entity__ {id: row.t_id})

    MERGE (s)-[r:RELATED {belief_key: row.belief_key}]->(t)
    ON CREATE SET r.created_at = row.now

    SET r.rel_type = row.rel_type,
        r.confidence = row.confidence,
        r.source_of_belief = row.source_of_belief,
        r.source_rank = row.source_rank,
        r.knowledge_type = row.knowledge_type,
        r.virtual_time = row.virtual_time,
        r.event_step = row.event_step,
        r.event_timestamp = row.event_timestamp,
        r.evidence_chunk_id = row.evidence_chunk_id,
        r.is_current = true,
        r.description = row.description

    RETURN count(*) AS n
    """.strip()

    now = _now_iso()
    for batch in _batched(df, batch_size):
        rows = []
        for rid, s_name, t_name, w, desc, tu_ids in zip(
            batch["id"].tolist(),
            batch["source"].tolist(),
            batch["target"].tolist(),
            batch["weight"].tolist(),
            batch["description"].tolist(),
            batch["text_unit_ids"].tolist(),
        ):
            s_id = name2id.get(s_name.strip().lower())
            t_id = name2id.get(t_name.strip().lower())
            if not rid or not s_id or not t_id:
                continue

            evidence_chunk_id = "unknown"
            if isinstance(tu_ids, list) and tu_ids:
                evidence_chunk_id = str(tu_ids[0])

            try:
                conf = float(w)
            except Exception:
                conf = 1.0
            # 保守裁剪到 (0, 1]，避免权重很大导致排序失真
            conf = max(0.01, min(1.0, conf))

            rows.append(
                {
                    "s_id": s_id,
                    "t_id": t_id,
                    "belief_key": f"GRAPHRAG_REL:{rid}",
                    "rel_type": "GRAPHRAG_RELATED",
                    "confidence": conf,
                    "source_of_belief": "graphrag_offline",
                    "source_rank": 2.0,
                    "knowledge_type": "observed_fact",
                    "virtual_time": virtual_time,
                    "event_step": int(event_step),
                    "event_timestamp": "unknown",
                    "evidence_chunk_id": evidence_chunk_id,
                    "description": desc,
                    "now": now,
                }
            )

        if rows:
            ltss.update_graph(cypher, parameters={"rows": rows})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", required=True, help="GraphRAG artifacts directory (contains create_final_*.parquet)")
    parser.add_argument("--clear_db", action="store_true", help="Danger: wipe Neo4j then rebuild schema")
    parser.add_argument("--batch_size", type=int, default=300)
    args = parser.parse_args()

    ltss = LongTermSemanticStore(bootstrap_now=False, setup_schema=True)
    if args.clear_db:
        ltss.clear_database(rebuild_schema=True, drop_schema=True)

    embedder = get_embedding_model()

    import_text_units_as_chunks(ltss, embedder, args.artifacts_dir, batch_size=args.batch_size)
    name2id = import_entities(ltss, embedder, args.artifacts_dir, batch_size=args.batch_size)
    link_chunks_to_entities_by_text_unit_ids(ltss, args.artifacts_dir, batch_size=args.batch_size)
    import_relationships(ltss, args.artifacts_dir, name2id, batch_size=max(500, args.batch_size))

    ltss.close()


if __name__ == "__main__":
    main()
