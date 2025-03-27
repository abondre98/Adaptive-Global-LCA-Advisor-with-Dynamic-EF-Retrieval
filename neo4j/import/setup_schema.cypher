// First, let's create the constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (ef:EmissionFactor) REQUIRE ef.ef_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (r:Region) REQUIRE r.region_code IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (et:EntityType) REQUIRE et.type_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.source_id IS UNIQUE;

// Create indexes for performance
CREATE INDEX IF NOT EXISTS FOR (ef:EmissionFactor) ON (ef.ef_value);
CREATE INDEX IF NOT EXISTS FOR (ef:EmissionFactor) ON (ef.is_outlier);
CREATE INDEX IF NOT EXISTS FOR (ef:EmissionFactor) ON (ef.multiplier_applied);
CREATE INDEX IF NOT EXISTS FOR (r:Region) ON (r.name);
CREATE INDEX IF NOT EXISTS FOR (et:EntityType) ON (et.type_name);
CREATE INDEX IF NOT EXISTS FOR (s:Source) ON (s.name);
CREATE INDEX IF NOT EXISTS FOR (ef:EmissionFactor) ON (ef.confidence);

// Create test nodes for relationship verification
CREATE (ef:EmissionFactor {
    ef_id: 'test_ef_001',
    entity_name: 'Test Emission Factor',
    ef_value: 1.0,
    ef_unit: 'kg CO2e/kg',
    confidence: 0.9,
    is_outlier: false,
    multiplier_applied: false
});

CREATE (r:Region {
    region_code: 'TEST_REG',
    name: 'Test Region'
});

CREATE (et:EntityType {
    type_id: 'TEST_TYPE',
    type_name: 'Test Type'
});

CREATE (s:Source {
    source_id: 'TEST_SRC',
    name: 'Test Source'
});

// Create test relationships
MATCH (ef:EmissionFactor {ef_id: 'test_ef_001'})
MATCH (r:Region {region_code: 'TEST_REG'})
MATCH (et:EntityType {type_id: 'TEST_TYPE'})
MATCH (s:Source {source_id: 'TEST_SRC'})
CREATE (ef)-[:APPLIES_TO_REGION {confidence: 0.9}]->(r)
CREATE (ef)-[:HAS_ENTITY_TYPE {confidence: 1.0}]->(et)
CREATE (ef)-[:SOURCED_FROM {timestamp: datetime()}]->(s)
CREATE (r)-[:PART_OF {relationship_type: 'contains'}]->(r)
CREATE (et)-[:RELATED_TO {relationship_strength: 0.8}]->(et);

// Create indexes on relationship properties
CREATE INDEX IF NOT EXISTS FOR ()-[r:APPLIES_TO_REGION]-() ON (r.confidence);
CREATE INDEX IF NOT EXISTS FOR ()-[r:HAS_ENTITY_TYPE]-() ON (r.confidence);
CREATE INDEX IF NOT EXISTS FOR ()-[r:SOURCED_FROM]-() ON (r.timestamp);
CREATE INDEX IF NOT EXISTS FOR ()-[r:PART_OF]-() ON (r.relationship_type);
CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.relationship_strength);

// Verify constraints
SHOW CONSTRAINTS;

// Verify indexes
SHOW INDEXES;

// Verify node labels
CALL db.labels();

// Verify relationship types
CALL db.relationshipTypes();

// Verify property keys
CALL db.propertyKeys();

// Clean up test data
MATCH (n) 
WHERE n.ef_id = 'test_ef_001' 
   OR n.region_code = 'TEST_REG' 
   OR n.type_id = 'TEST_TYPE' 
   OR n.source_id = 'TEST_SRC'
DETACH DELETE n; 
