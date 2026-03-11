# -*- coding: utf-8 -*-
import os
import time
from typing import Dict, Optional

from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase, basic_auth

from config import (
    GRAPHRAG_API_BASE,
    GRAPHRAG_CHAT_API_KEY,
    GRAPHRAG_CHAT_MODEL,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
)

from utils.embedding import get_embedding_model


def test_llm_connection() -> Optional[bool]:
    """Test LLM chat API connectivity."""
    print("Testing LLM (chat) connection...")
    if not all([GRAPHRAG_API_BASE, GRAPHRAG_CHAT_API_KEY, GRAPHRAG_CHAT_MODEL]):
        print("SKIP LLM: missing env vars.")
        return None

    try:
        llm = ChatOpenAI(
            model=GRAPHRAG_CHAT_MODEL,
            api_key=GRAPHRAG_CHAT_API_KEY,
            base_url=GRAPHRAG_API_BASE,
            timeout=60,
        )
        start_time = time.time()
        response = llm.invoke("Say hello.")
        elapsed = time.time() - start_time
        if response.content:
            print(f"OK LLM: {elapsed:.2f}s")
            return True
        print("FAIL LLM: empty response.")
        return False
    except Exception as e:
        print(f"FAIL LLM: {e}")
        return False


def _test_neo4j(uri: str, username: str, password: str, label: str) -> Optional[bool]:
    if not all([uri, username, password]):
        print(f"SKIP Neo4j [{label}]: missing env vars.")
        return None
    try:
        driver = GraphDatabase.driver(uri, auth=basic_auth(username, password))
        driver.verify_connectivity()
        driver.close()
        print(f"OK Neo4j [{label}].")
        return True
    except Exception as e:
        print(f"FAIL Neo4j [{label}]: {e}")
        return False


def test_neo4j_connection() -> Optional[bool]:
    """Test the active Neo4j connection from config."""
    print("Testing Neo4j connection (active config)...")
    return _test_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, "active")


def test_all_neo4j_connections() -> Dict[str, Optional[bool]]:
    """Test all available Neo4j connections (local + aura if configured)."""
    results: Dict[str, Optional[bool]] = {}

    print("Testing Neo4j connections (all)...")
    results["active"] = _test_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, "active")

    aura_uri = os.getenv("NEO4J_AURA_URI")
    aura_user = os.getenv("NEO4J_AURA_USERNAME", "neo4j")
    aura_pass = os.getenv("NEO4J_AURA_PASSWORD")
    if aura_uri or aura_pass:
        results["aura"] = _test_neo4j(aura_uri or "", aura_user, aura_pass or "", "aura")
    else:
        results["aura"] = None
        print("SKIP Neo4j [aura]: missing env vars.")

    return results


def test_embedding_connection() -> Optional[bool]:
    """Test embedding model connectivity."""
    print("Testing Embedding connection...")
    try:
        embedding_model = get_embedding_model()
        test_text = "embedding connectivity test"
        start_time = time.time()
        vector = embedding_model.embed_query(test_text)
        elapsed = time.time() - start_time
        if vector and isinstance(vector, list) and len(vector) > 0:
            print(f"OK Embedding: {elapsed:.2f}s, dim={len(vector)}")
            return True
        print("FAIL Embedding: empty vector.")
        return False
    except Exception as e:
        print(f"FAIL Embedding: {e}")
        return False
