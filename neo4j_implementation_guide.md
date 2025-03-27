# Neo4j Knowledge Graph Implementation Guide

## Overview

This guide provides detailed instructions for implementing the Neo4j Knowledge Graph for the Regional Emission Factor system. The knowledge graph stores emission factors with their relationships to regions, entity types, and sources, enabling efficient querying and analysis.

## Setup Requirements

- Docker and Docker Compose
- Python 3.8+
- Pandas, NumPy, and UUID Python packages
- Harmonized emission factor dataset (CSV)

## Directory Structure

```
/
├── docker-compose.yml           # Docker Compose configuration
├── neo4j/                       # Neo4j related files
│   ├── data/                    # Neo4j data directory
│   ├── logs/                    # Neo4j logs
│   ├── import/                  # Import directory for CSV files
│   │   ├── schema.cypher        # Schema definition
│   │   ├── *.csv                # CSV files for import
│   ├── plugins/                 # Neo4j plugins
│   ├── sample_queries.cypher    # Sample Cypher queries
├── scripts/
│   ├── prepare_neo4j_data.py    # Script to prepare CSV files
│   ├── import_neo4j_data.sh     # Import automation script
```

## Implementation Steps

### 1. Environment Setup

1. Create the necessary directories:

   ```bash
   mkdir -p neo4j/data neo4j/logs neo4j/import neo4j/plugins
   ```

2. Download required Neo4j plugins:

   ```bash
   curl -L https://github.com/neo4j/apoc/releases/download/5.11.0/apoc-5.11.0-core.jar -o neo4j/plugins/apoc-5.11.0-core.jar
   curl -L https://github.com/neo4j/graph-data-science/releases/download/2.4.0/neo4j-graph-data-science-2.4.0.jar -o neo4j/plugins/neo4j-graph-data-science-2.4.0.jar
   ```

3. Start Neo4j using Docker Compose:
   ```bash
   docker-compose up -d
   ```

### 2. Data Preparation

Run the data preparation script to convert the harmonized dataset into CSV files for Neo4j import:

```bash
python scripts/prepare_neo4j_data.py
```

This script will:

- Load the harmonized dataset from `data/processed/harmonized_global_ef_dataset.csv`
- Extract and create separate CSV files for:
  - Region nodes (regions.csv)
  - EntityType nodes (entity_types.csv)
  - Source nodes (sources.csv)
  - EmissionFactor nodes (emission_factors.csv)
  - APPLIES_TO_REGION relationships (ef_to_region.csv)
  - HAS_ENTITY_TYPE relationships (ef_to_entity_type.csv)
  - SOURCED_FROM relationships (ef_to_source.csv)
  - PART_OF relationships (region_hierarchy.csv)

### 3. Schema Creation

The schema defines constraints and indexes for the Neo4j database, improving data integrity and query performance.

Key constraints:

- Unique `ef_id` for EmissionFactor nodes
- Unique `region_code` for Region nodes
- Unique `type_id` for EntityType nodes
- Unique `source_id` for Source nodes

Indexes for performance:

- EmissionFactor nodes: ef_value, is_outlier, multiplier_applied, confidence
- Region nodes: name
- EntityType nodes: type_name
- Source nodes: name

### 4. Data Import

Run the import script to load the data into Neo4j:

```bash
bash scripts/import_neo4j_data.sh
```

This script will:

1. Wait for Neo4j to be available
2. Apply schema constraints and indexes
3. Import all node types (Region, EntityType, Source, EmissionFactor)
4. Create all relationships (APPLIES_TO_REGION, HAS_ENTITY_TYPE, SOURCED_FROM, PART_OF)
5. Run validation queries to confirm successful import

### 5. Knowledge Graph Structure

#### Node Types

1. **EmissionFactor**

   - Key properties: ef_id, entity_id, entity_name, ef_value, ef_unit, confidence, is_outlier, multiplier_applied, timestamp
   - Represents emission factor records with their values and metadata

2. **Region**

   - Key properties: region_code, name, continent, is_global
   - Represents geographic regions or countries

3. **EntityType**

   - Key properties: type_id, type_name, description
   - Represents categories like product, sector, or process

4. **Source**
   - Key properties: source_id, name, version, url
   - Represents original dataset sources

#### Relationship Types

1. **APPLIES_TO_REGION**

   - From EmissionFactor to Region
   - Properties: confidence

2. **HAS_ENTITY_TYPE**

   - From EmissionFactor to EntityType
   - Properties: confidence

3. **SOURCED_FROM**

   - From EmissionFactor to Source
   - Properties: timestamp

4. **PART_OF**
   - From Region to Region
   - Properties: relationship_type
   - Represents hierarchical relationships between regions

## Query Patterns

See `neo4j/sample_queries.cypher` for a comprehensive set of example queries. Key query patterns include:

### Regional Comparison

```cypher
MATCH (ef:EmissionFactor)-[:HAS_ENTITY_TYPE]->(et:EntityType {type_name: 'energy'})
MATCH (ef)-[:APPLIES_TO_REGION]->(r:Region)
WHERE r.region_code IN ['USA', 'FR', 'GLB']
RETURN r.name AS Region, avg(ef.ef_value) AS AvgEmissionFactor, count(ef) AS Count
ORDER BY AvgEmissionFactor DESC;
```

### Entity Type Analysis

```cypher
MATCH (ef:EmissionFactor)-[:HAS_ENTITY_TYPE]->(et:EntityType)
WHERE ef.is_outlier = false
RETURN et.type_name AS EntityType,
       count(ef) AS Count,
       avg(ef.ef_value) AS AvgValue,
       min(ef.ef_value) AS MinValue,
       max(ef.ef_value) AS MaxValue
ORDER BY Count DESC;
```

### Source Reliability

```cypher
MATCH (ef:EmissionFactor)-[:SOURCED_FROM]->(s:Source)
RETURN s.name AS Source,
       avg(ef.confidence) AS AvgConfidence,
       count(ef) AS RecordCount
ORDER BY AvgConfidence DESC;
```

## Performance Optimization

### Query Strategy

- Use parameterized queries to improve caching
- Apply filters early in query patterns
- Leverage indexes for frequent query patterns
- Use `PROFILE` and `EXPLAIN` to analyze query execution plans

### Database Configuration

For production environments, consider these optimizations:

1. Memory settings:

   ```
   NEO4J_dbms_memory_heap_max__size=8G
   NEO4J_dbms_memory_pagecache_size=4G
   ```

2. Parallel query execution:

   ```
   NEO4J_dbms_cypher_parallel_execution_enabled=true
   ```

3. Query logging for performance monitoring:
   ```
   NEO4J_dbms_logs_query_enabled=true
   NEO4J_dbms_logs_query_threshold=1000ms
   ```

## Monitoring and Maintenance

### Key Metrics to Monitor

1. Memory usage
2. Query execution times
3. Cache hit ratio
4. Disk usage
5. Connection count

### Backup Procedure

```bash
docker exec neo4j-ef-kg neo4j-admin dump --database=neo4j --to=/backups/neo4j-backup.dump
```

### Common Issues and Solutions

1. **Memory issues**: Increase heap and page cache settings
2. **Slow queries**: Add indexes, optimize Cypher queries
3. **Import failures**: Check CSV formatting, column names
4. **Plugin errors**: Verify plugin compatibility with Neo4j version

## Next Steps

1. Connect to API layer
2. Implement automated updates
3. Add visualization components
4. Set up monitoring and alerting
5. Configure backup and recovery procedures
