from temporal_reasoning.cypher_templates import FIRST_EVENT_AFTER_ANCHOR
from temporal_reasoning.executor import run_template
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PWD = "wyx15757569582"
AGENT = "LoCoMoTester"

driver = GraphDatabase.driver(URI, auth=(USER, PWD))

triples = run_template(
    driver,
    FIRST_EVENT_AFTER_ANCHOR,
    {"agent": AGENT, "anchor_keyword": "service"}
)

print("\n".join(triples) if triples else "No triples returned.")
driver.close()
