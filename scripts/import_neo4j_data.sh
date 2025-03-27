#!/bin/bash

# Neo4j Data Import Script
# This script automates the process of importing data into Neo4j

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Return value of a pipeline is the value of the last command to exit with non-zero status

# Configuration
DOCKER_BIN="/Applications/Docker.app/Contents/Resources/bin/docker"
CONTAINER_NAME="neo4j-ef-kg"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="complex-password-here"  # Should match docker-compose.yml

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Neo4j data import process...${NC}"

# Step 1: Prepare the CSV files from the harmonized dataset
echo -e "${YELLOW}Preparing CSV files from harmonized dataset...${NC}"
python scripts/prepare_neo4j_data.py

# Step 2: Wait for Neo4j to be available
echo -e "${YELLOW}Waiting for Neo4j to be available...${NC}"
until $DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD "RETURN 1;" > /dev/null 2>&1; do
  echo "Waiting for Neo4j to start..."
  sleep 5
done
echo -e "${GREEN}Neo4j is now available!${NC}"

# Clear existing data
echo -e "${YELLOW}Clearing existing data from the database...${NC}"
$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD \
"MATCH (n) DETACH DELETE n;"
echo -e "${GREEN}Database cleared!${NC}"

# Step 3: Apply schema constraints and indexes
echo -e "${YELLOW}Applying schema constraints and indexes...${NC}"
$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD -f /import/schema.cypher
echo -e "${GREEN}Schema created successfully!${NC}"

# Step 4: Import the data using cypher-shell with LOAD CSV
echo -e "${YELLOW}Importing data into Neo4j...${NC}"

# Import Region nodes
echo -e "${YELLOW}Importing Region nodes...${NC}"
$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD \
"LOAD CSV WITH HEADERS FROM 'file:///regions.csv' AS row
CREATE (r:Region {
  region_code: row.region_code,
  name: row.name,
  continent: row.continent,
  is_global: row.is_global = 'True'
});"
echo -e "${GREEN}Region nodes imported!${NC}"

# Import EntityType nodes
echo -e "${YELLOW}Importing EntityType nodes...${NC}"
$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD \
"LOAD CSV WITH HEADERS FROM 'file:///entity_types.csv' AS row
CREATE (et:EntityType {
  type_id: row.type_id,
  type_name: row.type_name,
  description: row.description
});"
echo -e "${GREEN}EntityType nodes imported!${NC}"

# Import Source nodes
echo -e "${YELLOW}Importing Source nodes...${NC}"
$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD \
"LOAD CSV WITH HEADERS FROM 'file:///sources.csv' AS row
CREATE (s:Source {
  source_id: row.source_id,
  name: row.name,
  version: row.version,
  url: row.url
});"
echo -e "${GREEN}Source nodes imported!${NC}"

# Import EmissionFactor nodes
echo -e "${YELLOW}Importing EmissionFactor nodes...${NC}"
$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD \
"LOAD CSV WITH HEADERS FROM 'file:///emission_factors.csv' AS row
CREATE (ef:EmissionFactor {
  ef_id: row.ef_id,
  entity_id: row.entity_id,
  entity_name: row.entity_name,
  ef_value: toFloat(row.ef_value),
  ef_unit: row.ef_unit,
  confidence: toFloat(row.confidence),
  is_outlier: row.is_outlier = 'True',
  multiplier_applied: row.multiplier_applied = 'True',
  timestamp: row.timestamp
});"
echo -e "${GREEN}EmissionFactor nodes imported!${NC}"

# Import APPLIES_TO_REGION relationships
echo -e "${YELLOW}Creating APPLIES_TO_REGION relationships...${NC}"
$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD \
"LOAD CSV WITH HEADERS FROM 'file:///ef_to_region.csv' AS row
MATCH (ef:EmissionFactor {ef_id: row.ef_id})
MATCH (r:Region {region_code: row.region_code})
CREATE (ef)-[:APPLIES_TO_REGION {confidence: toFloat(row.confidence)}]->(r);"
echo -e "${GREEN}APPLIES_TO_REGION relationships created!${NC}"

# Import HAS_ENTITY_TYPE relationships
echo -e "${YELLOW}Creating HAS_ENTITY_TYPE relationships...${NC}"
$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD \
"LOAD CSV WITH HEADERS FROM 'file:///ef_to_entity_type.csv' AS row
MATCH (ef:EmissionFactor {ef_id: row.ef_id})
MATCH (et:EntityType {type_id: row.type_id})
CREATE (ef)-[:HAS_ENTITY_TYPE {confidence: toFloat(row.confidence)}]->(et);"
echo -e "${GREEN}HAS_ENTITY_TYPE relationships created!${NC}"

# Import SOURCED_FROM relationships
echo -e "${YELLOW}Creating SOURCED_FROM relationships...${NC}"
$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD \
"LOAD CSV WITH HEADERS FROM 'file:///ef_to_source.csv' AS row
MATCH (ef:EmissionFactor {ef_id: row.ef_id})
MATCH (s:Source {source_id: row.source_id})
CREATE (ef)-[:SOURCED_FROM {timestamp: row.timestamp}]->(s);"
echo -e "${GREEN}SOURCED_FROM relationships created!${NC}"

# Import PART_OF relationships (region hierarchy)
echo -e "${YELLOW}Creating PART_OF relationships for region hierarchy...${NC}"
$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD \
"LOAD CSV WITH HEADERS FROM 'file:///region_hierarchy.csv' AS row
MATCH (child:Region {region_code: row.child_region_code})
MATCH (parent:Region {region_code: row.parent_region_code})
CREATE (child)-[:PART_OF {relationship_type: row.relationship_type}]->(parent);"
echo -e "${GREEN}PART_OF relationships created!${NC}"

# Step 5: Run validation queries
echo -e "${YELLOW}Running validation queries...${NC}"
$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD \
"MATCH (n) RETURN labels(n) AS Label, count(n) AS Count ORDER BY Count DESC;"

$DOCKER_BIN exec $CONTAINER_NAME cypher-shell -u $NEO4J_USER -p $NEO4J_PASSWORD \
"MATCH ()-[r]->() RETURN type(r) AS RelationshipType, count(r) AS Count ORDER BY Count DESC;"

echo -e "${GREEN}Neo4j data import completed successfully!${NC}"

# Done
echo -e "\n${GREEN}Knowledge Graph setup is complete. You can access Neo4j Browser at http://localhost:7474${NC}" 
