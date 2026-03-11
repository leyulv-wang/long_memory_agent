import os

from memory.stores import LongTermSemanticStore


def main() -> None:
    print("Initializing Neo4j constraints and indexes...")
    store = LongTermSemanticStore(bootstrap_now=False, setup_schema=True)
    try:
        if not store or not getattr(store, "driver", None):
            print("Neo4j connection failed; schema init skipped.")
            return
        print("Schema init complete.")
    finally:
        try:
            store.close()
        except Exception:
            pass


if __name__ == "__main__":
    if os.getenv("USE_NEO4J_AURA", "0").strip().lower() in ("1", "true", "yes"):
        print("USE_NEO4J_AURA=1 (Aura)")
    else:
        print("USE_NEO4J_AURA=0 (Local)")
    main()
