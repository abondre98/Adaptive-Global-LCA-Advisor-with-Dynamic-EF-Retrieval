// Sample Queries for Emission Factor Knowledge Graph
// These queries demonstrate the capabilities of the knowledge graph for various analyses

// Basic Node Counting
// ------------------
// Count all nodes by type
MATCH (n)
RETURN labels(n) AS NodeType, count(n) AS Count
ORDER BY Count DESC;

// Regional Comparison Queries
// --------------------------
// Compare emission factors for the same entity type across regions
MATCH (ef:EmissionFactor)-[:HAS_ENTITY_TYPE]->(et:EntityType {type_name: 'energy'})
MATCH (ef)-[:APPLIES_TO_REGION]->(r:Region)
WHERE r.region_code IN ['USA', 'FR', 'GLB']
RETURN r.name AS Region, avg(ef.ef_value) AS AvgEmissionFactor, count(ef) AS Count
ORDER BY AvgEmissionFactor DESC;

// Find differences between regional and global emission factors
MATCH (ef1:EmissionFactor)-[:APPLIES_TO_REGION]->(r1:Region {region_code: 'USA'})
MATCH (ef1)-[:HAS_ENTITY_TYPE]->(et:EntityType)
MATCH (ef2:EmissionFactor)-[:APPLIES_TO_REGION]->(r2:Region {region_code: 'GLB'})
MATCH (ef2)-[:HAS_ENTITY_TYPE]->(et)
WITH et.type_name AS EntityType, 
     avg(ef1.ef_value) AS USAAvg, 
     avg(ef2.ef_value) AS GlobalAvg,
     count(ef1) AS USACount
RETURN EntityType, USAAvg, GlobalAvg, USAAvg/GlobalAvg AS Ratio, USACount
ORDER BY abs(USAAvg/GlobalAvg - 1) DESC
LIMIT 10;

// Entity Type Analysis
// -------------------
// Analyze emission factors by entity type
MATCH (ef:EmissionFactor)-[:HAS_ENTITY_TYPE]->(et:EntityType)
WHERE ef.is_outlier = false
RETURN et.type_name AS EntityType,
       count(ef) AS Count,
       avg(ef.ef_value) AS AvgValue,
       min(ef.ef_value) AS MinValue,
       max(ef.ef_value) AS MaxValue
ORDER BY Count DESC;

// Distribution of emission factors by entity type
MATCH (ef:EmissionFactor)-[:HAS_ENTITY_TYPE]->(et:EntityType)
RETURN et.type_name AS EntityType,
      CASE 
        WHEN ef.ef_value < 1 THEN 'Very Low (<1)'
        WHEN ef.ef_value < 10 THEN 'Low (1-10)'
        WHEN ef.ef_value < 100 THEN 'Medium (10-100)'
        WHEN ef.ef_value < 1000 THEN 'High (100-1000)'
        ELSE 'Very High (>1000)'
      END AS ValueRange,
      count(ef) AS Count
ORDER BY EntityType, ValueRange;

// Source Reliability
// -----------------
// Analyze confidence by source
MATCH (ef:EmissionFactor)-[:SOURCED_FROM]->(s:Source)
RETURN s.name AS Source,
       avg(ef.confidence) AS AvgConfidence,
       count(ef) AS RecordCount
ORDER BY AvgConfidence DESC;

// Outlier Analysis
// ---------------
// Distribution of outliers by source and entity type
MATCH (ef:EmissionFactor)-[:HAS_ENTITY_TYPE]->(et:EntityType)
MATCH (ef)-[:SOURCED_FROM]->(s:Source)
WHERE ef.is_outlier = true
RETURN s.name AS Source, et.type_name AS EntityType, count(ef) AS OutlierCount
ORDER BY OutlierCount DESC;

// Regional Multiplier Analysis
// ---------------------------
// Distribution of multiplier application by entity type
MATCH (ef:EmissionFactor)-[:HAS_ENTITY_TYPE]->(et:EntityType)
WHERE ef.multiplier_applied = true
RETURN et.type_name AS EntityType, count(ef) AS Count
ORDER BY Count DESC;

// Regional multiplier distribution by source
MATCH (ef:EmissionFactor)-[:SOURCED_FROM]->(s:Source)
WHERE ef.multiplier_applied = true
RETURN s.name AS Source, count(ef) AS Count
ORDER BY Count DESC;

// Path Analysis
// ------------
// Find all paths between emission factors and regions with a specific entity type
MATCH path = (ef:EmissionFactor)-[:HAS_ENTITY_TYPE]->(et:EntityType {type_name: 'energy'}),
             (ef)-[:APPLIES_TO_REGION]->(r:Region)
RETURN path
LIMIT 10;

// Find emission factors that apply to regions that are part of Europe
MATCH (ef:EmissionFactor)-[:APPLIES_TO_REGION]->(r1:Region)-[:PART_OF]->
      (r2:Region {region_code: 'EUR'})
RETURN ef.entity_name, ef.ef_value, ef.ef_unit, r1.name AS Country
LIMIT 20;

// Performance Query (For Benchmarking)
// ----------------
// Query for regional emission factors with multiple conditions
MATCH (ef:EmissionFactor)-[:HAS_ENTITY_TYPE]->(et:EntityType)
MATCH (ef)-[:APPLIES_TO_REGION]->(r:Region)
MATCH (ef)-[:SOURCED_FROM]->(s:Source)
WHERE r.region_code = 'USA'
  AND ef.is_outlier = false
  AND ef.ef_value > 0
  AND ef.ef_value < 100
RETURN ef.entity_name, ef.ef_value, ef.ef_unit, 
       et.type_name AS EntityType, s.name AS Source,
       ef.confidence
ORDER BY ef.ef_value DESC
LIMIT 20; 
