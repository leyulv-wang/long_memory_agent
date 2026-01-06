# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Dict, List, Optional

from memory.stores import LongTermSemanticStore


def _q(ltss: LongTermSemanticStore, cypher: str, params: Optional[Dict[str, Any]] = None):
    return ltss.query_graph(cypher, params or {}) or []


def diagnose_neo4j_graph(agent_name: str = "LongMemory", limit_samples: int = 5) -> Dict[str, Any]:
    """
    在注入/巩固完成后调用，检查 Neo4j 图结构是否满足 GraphRAG-V2 的检索链路。

    重点检查：
    - RAW: TextUnit / Event 是否存在，Event->TextUnit 关系是否有 EVIDENCED_BY（V2 查询依赖）
    - CONSOLIDATED: Fact / HAS_FACT 是否存在
    - 属性：turn_id/confidence/belief_key/slot_key/embedding/channel 等
    """
    ltss = LongTermSemanticStore(bootstrap_now=False)
    report: Dict[str, Any] = {"agent_name": agent_name, "checks": {}, "samples": {}}

    # 1) 基础计数：看有没有写入
    report["checks"]["count_TextUnit"] = _q(ltss, "MATCH (u:TextUnit) RETURN count(u) AS c")[0]["c"]
    report["checks"]["count_Event"] = _q(ltss, "MATCH (e:Event) RETURN count(e) AS c")[0]["c"]
    report["checks"]["count_Fact"] = _q(ltss, "MATCH (f:Fact) RETURN count(f) AS c")[0]["c"]
    report["checks"]["count_HAS_FACT"] = _q(ltss, "MATCH (:Event)-[r:HAS_FACT]->(:Fact) RETURN count(r) AS c")[0]["c"]

    # 2) 检查 Event-TextUnit 的关系名是否对齐 V2（V2 用 EVIDENCED_BY）
    report["checks"]["count_EVIDENCED_BY"] = _q(
        ltss, "MATCH (:Event)-[r:EVIDENCED_BY]->(:TextUnit) RETURN count(r) AS c"
    )[0]["c"]
    report["checks"]["count_HAS_EVIDENCE"] = _q(
        ltss, "MATCH (:Event)-[r:HAS_EVIDENCE]->(:TextUnit) RETURN count(r) AS c"
    )[0]["c"]
    report["checks"]["count_EVIDENCE_OF"] = _q(
        ltss, "MATCH (:TextUnit)-[r:EVIDENCE_OF]->(:Event) RETURN count(r) AS c"
    )[0]["c"]

    # 3) 检查 TextUnit embedding/turn_id/channel 是否齐全（向量召回必须）
    report["checks"]["textunit_missing_embedding"] = _q(
        ltss,
        """
        MATCH (u:TextUnit)
        WHERE u.embedding IS NULL
        RETURN count(u) AS c
        """,
    )[0]["c"]

    report["checks"]["textunit_missing_turn_id"] = _q(
        ltss,
        """
        MATCH (u:TextUnit)
        WHERE u.turn_id IS NULL
        RETURN count(u) AS c
        """,
    )[0]["c"]

    report["checks"]["textunit_channel_distribution"] = _q(
        ltss,
        """
        MATCH (u:TextUnit)
        RETURN u.channel AS channel, count(*) AS c
        ORDER BY c DESC
        """,
    )

    # 4) 检查 Fact 关键属性（你日志里出现 slot_key warning）
    report["checks"]["fact_missing_turn_id"] = _q(
        ltss,
        """
        MATCH (f:Fact)
        WHERE f.turn_id IS NULL
        RETURN count(f) AS c
        """,
    )[0]["c"]

    report["checks"]["fact_missing_confidence"] = _q(
        ltss,
        """
        MATCH (f:Fact)
        WHERE f.confidence IS NULL
        RETURN count(f) AS c
        """,
    )[0]["c"]

    report["checks"]["fact_missing_belief_key"] = _q(
        ltss,
        """
        MATCH (f:Fact)
        WHERE f.belief_key IS NULL
        RETURN count(f) AS c
        """,
    )[0]["c"]

    report["checks"]["fact_missing_slot_key"] = _q(
        ltss,
        """
        MATCH (f:Fact)
        WHERE f.slot_key IS NULL
        RETURN count(f) AS c
        """,
    )[0]["c"]

    # 5) agent_name 对齐检查（V2 里会过滤 node.agent_name = $agent_name）
    # 注意：你的 TextUnit 可能没写 agent_name，这里只是帮助你排查。
    report["checks"]["textunit_agent_name_distribution"] = _q(
        ltss,
        """
        MATCH (u:TextUnit)
        RETURN coalesce(u.agent_name, "<NULL>") AS agent_name, count(*) AS c
        ORDER BY c DESC
        """,
    )

    # 6) 抽样看看图是否满足 V2 的那条链：Event -EVIDENCED_BY-> TextUnit; Event -HAS_FACT-> Fact
    report["samples"]["event_textunit_edges"] = _q(
        ltss,
        f"""
        MATCH (e:Event)-[r]->(u:TextUnit)
        RETURN type(r) AS rel, e.event_id AS event_id, e.channel AS event_channel,
               u.name AS textunit_name, u.channel AS tu_channel, u.turn_id AS tu_turn_id
        LIMIT {int(limit_samples)}
        """,
    )

    report["samples"]["event_fact_edges"] = _q(
        ltss,
        f"""
        MATCH (e:Event)-[r:HAS_FACT]->(f:Fact)
        OPTIONAL MATCH (f)-[:SUBJECT]->(s)
        OPTIONAL MATCH (f)-[:OBJECT]->(o)
        RETURN e.event_id AS event_id, e.channel AS event_channel,
               f.belief_key AS belief_key, f.slot_key AS slot_key,
               f.type AS rel_type, f.confidence AS confidence,
               f.turn_id AS turn_id, f.channel AS fact_channel,
               s.name AS subject, o.name AS object
        LIMIT {int(limit_samples)}
        """,
    )

    # 7) 模拟 V2 核心查询：用现有 TextUnit.name 去查 (Event)-[:EVIDENCED_BY]->(TextUnit) 再接 HAS_FACT
    # 如果这里返回 0 行，说明你的关系名/Fact 写入不对齐。
    report["checks"]["v2_fact_chain_rows"] = _q(
        ltss,
        """
        MATCH (u:TextUnit)
        WITH collect(u.name)[0..10] AS seed_names
        UNWIND seed_names AS tu
        MATCH (e:Event)-[:EVIDENCED_BY]->(u2:TextUnit {name: tu})
        MATCH (e)-[:HAS_FACT]->(f:Fact)
        RETURN count(f) AS c
        """,
    )[0]["c"] if report["checks"]["count_TextUnit"] else 0

    ltss.close()
    return report


def print_report(report: Dict[str, Any]):
    print("\n================= Neo4j 诊断报告 =================")
    for k, v in report["checks"].items():
        print(f"{k}: {v}")

    print("\n--- samples: event_textunit_edges ---")
    for row in report["samples"]["event_textunit_edges"]:
        print(row)

    print("\n--- samples: event_fact_edges ---")
    for row in report["samples"]["event_fact_edges"]:
        print(row)

    print("\n✅ 结论提示：")
    if report["checks"]["count_Fact"] == 0 or report["checks"]["count_HAS_FACT"] == 0:
        print(" - 你的 CONSOLIDATED 侧没有写出 Fact/HAS_FACT（巩固写入链路或 writer 出问题）。")
    if report["checks"]["count_EVIDENCED_BY"] == 0 and (report["checks"]["count_HAS_EVIDENCE"] > 0 or report["checks"]["count_EVIDENCE_OF"] > 0):
        print(" - 你有 Event<->TextUnit 关系，但缺少 EVIDENCED_BY；V2 查询会匹配不到（需要补写或改查询）。")
    if report["checks"]["textunit_missing_embedding"] > 0:
        print(" - 有 TextUnit 没 embedding；向量召回会弱/空（检查 embedding 写入）。")
    if report["checks"]["v2_fact_chain_rows"] == 0 and report["checks"]["count_Fact"] > 0:
        print(" - 有 Fact，但 V2 的链路查询仍为 0：大概率是 EVIDENCED_BY 缺失，或 Event 没连 HAS_FACT。")
    print("=================================================\n")


if __name__ == "__main__":
    # 你在跑完一次 ingest + consolidation 后，直接运行这个脚本即可。
    rep = diagnose_neo4j_graph(agent_name="LongMemory", limit_samples=5)
    print_report(rep)
