# Neo4j Quick Reference

Spin up the local database:

```bash
docker-compose up -d neo4j
```

Ensure the credentials match the defaults used by `src/graph/neo4j_connector.py`:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
```

After building a graph via the builder CLI, persist it with the new flag:

```bash
python -m src.graph.builder path/to/prediction.json --persist-neo4j --neo4j-clear
```

## Example Cypher Queries

List every step with its text and optional order:

```cypher
MATCH (s:Step)
RETURN s.id AS step_id, s.text AS description, s.order AS order
ORDER BY order
```

Inspect constraint-to-step edges:

```cypher
MATCH (c:Condition)-[r:CONDITION_ON]->(s:Step)
RETURN c.id AS constraint_id, c.type AS type, s.id AS guarded_step
ORDER BY constraint_id
```

Show NEXT edges as an ordered path:

```cypher
MATCH path=(s:Step)-[:NEXT*]->(t:Step)
RETURN nodes(path) AS steps
LIMIT 5
```

Count node types to verify ingestion volume:

```cypher
MATCH (n)
RETURN labels(n) AS labels, count(*) AS count
ORDER BY count DESC
```

Clear the graph directly via Cypher if needed:

```cypher
MATCH (n) DETACH DELETE n
```
