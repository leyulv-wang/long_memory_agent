# -*- coding: utf-8 -*-
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.connection_tests import (
    test_llm_connection,
    test_embedding_connection,
    test_neo4j_connection,
    test_all_neo4j_connections,
)


def run_all_tests():
    print("--- Connectivity checks ---")

    llm_ok = test_llm_connection()
    emb_ok = test_embedding_connection()
    neo_active_ok = test_neo4j_connection()
    neo_all = test_all_neo4j_connections()

    print("\n--- Summary ---")
    def _fmt(name, ok):
        if ok is True:
            return f"OK {name}"
        if ok is False:
            return f"FAIL {name}"
        return f"SKIP {name}"

    print(_fmt("LLM (chat)", llm_ok))
    print(_fmt("Embedding", emb_ok))
    print(_fmt("Neo4j (active)", neo_active_ok))

    for label, ok in neo_all.items():
        print(_fmt(f"Neo4j ({label})", ok))

    print("---------------------------")

    all_ok = all(x is not False for x in [llm_ok, emb_ok, neo_active_ok] + list(neo_all.values()))
    if all_ok:
        print("\nAll required connections look good.")
    else:
        print("\nSome connections failed. Check your .env and services.")


if __name__ == "__main__":
    run_all_tests()
