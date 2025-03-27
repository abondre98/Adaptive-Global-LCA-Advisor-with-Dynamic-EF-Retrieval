# Milestone 1: Regional EF Knowledge Graph

This document outlines the implementation plan for Milestone 1 of the Adaptive Global LCA Advisor project, focusing on data collection, processing, and knowledge graph creation.

## Data Collection and Preparation

### 1. Gather Existing Datasets

#### A. Agribalyse
- **Overview**: Developed by INRAE, focusing on agricultural and food product emission factors in the EU.
- **Source**: [INRAE Portal](https://agribalyse.ademe.fr/) or [ADEME Data](https://data.ademe.fr/datasets/agribalyse)
- **Download**: Choose the most recent version (Agribalyse 3.1) in CSV/Excel format.

#### B. USEEIO (U.S. Environmentally-Extended Input-Output Model)
- **Overview**: Published by the U.S. EPA, providing industrial emission factors for the U.S.
- **Source**: [EPA GitHub Repository](https://github.com/USEPA/USEEIO)
- **Download**: Clone or download the repository, focusing on files in "data" or "model" subdirectories.

#### C. OpenLCA Databases
- **Overview**: Process-based emission factors for various industries and regions.
- **Source**: [OpenLCA Nexus](https://nexus.openlca.org/)
- **Download**: Select relevant databases (Agriculture, Energy, Industry) and export in CSV or JSON format.

#### D. Additional Sources
- **EXIOBASE**: Multi-regional input-output data ([EXIOBASE Portal](https://exiobase.eu/))
- **Climate TRACE**: Real-time emissions data ([Climate TRACE](https://climatetrace.org/))

### 2. Structure and Standardize Data Formats

#### A. Organize Files and Directories
```
data/
  agribalyse/
  useeio/
  openlca/
  ...
```
- Keep each dataset in its own subfolder.
- Include dataset name and version in filenames (e.g., `agribalyse_v3.1.csv`).

#### B. Inspect and Clean Each Dataset
1. **Inspect Columns**
   - Identify common fields (ProductName, Region, EmissionFactorValue, Unit, etc.)
   - Note differences in terminology between datasets.

2. **Rename or Map Columns**
   - Standardize column names (e.g., "GHG_emissions" → "emission_factor")
   - Document mapping in a data dictionary.

3. **Remove Duplicates**
   - Use unique keys (product_name + region + data_source) to identify and remove duplicates.

4. **Handle Missing Values**
   - Mark incomplete records for further investigation.
   - Consider dropping or flagging rows with missing critical data.

#### C. Merge Data into a Unified Format
1. **Combine CSVs** into a master dataset with standardized columns.
2. **Add Data Source Column** to track the origin of each record.
3. **Conduct Sanity Checks** on random sample rows.

### 3. Identify and Apply IPCC AR6 Regional Multipliers

#### A. Obtain IPCC AR6 Region Definitions
- Source: [IPCC AR6 Reports](https://www.ipcc.ch/report/ar6/)
- Extract region-specific multipliers for emission adjustments.

#### B. Map Dataset Regions to IPCC AR6
1. Create a mapping table linking countries to IPCC regions.
2. Standardize country codes (ISO 3166-1) for consistent lookups.

#### C. Apply Regional Multipliers
1. Use formula: `Adjusted_EF = Base_EF × IPCC_multiplier(region)`
2. Implement in code and validate results.

### 4. Data Processing with Regional Adjustments

#### A. Review Multiplier Reference Table
- Compile region → multiplier mappings.
- Ensure every region in your dataset has a corresponding multiplier.

#### B. Implement Adjustment
```python
# Example pseudocode
df['adjusted_ef'] = df.apply(
    lambda row: row['emission_factor'] * ipcc_multiplier[row['ipcc_region']],
    axis=1
)
```

#### C. Log Results
- Track which multiplier was applied to each record.
- Flag records with missing multipliers.

#### D. Validate Adjusted Emission Factors
1. Sample 1-5% of records across regions and products.
2. Verify calculations and check for outliers.
3. Document any errors or anomalies.

#### E. Clean and Finalize Dataset
1. Check for incomplete data (null values in critical fields).
2. Flag or remove extreme outliers.
3. Consolidate final fields:
   - product_name
   - region
   - emission_factor (original)
   - adjusted_ef
   - multiplier_used
   - source_dataset
4. Export to CSV/JSON for Neo4j import.

## Neo4j Knowledge Graph Implementation

### 1. Define Schema / Data Model

#### A. Key Entities (Nodes)
1. **EF (Emission Factor) Node**
   - Attributes: ef_id, value, unit, data_source, adjusted_ef
   - Purpose: Stores emission factor data for specific products/processes

2. **Region Node**
   - Attributes: region_id/country_code, continent, ipcc_region
   - Purpose: Represents geographical location

3. **Industry Node**
   - Attributes: industry_id, industry_name
   - Purpose: Groups EFs by standard industry classification

#### B. Relationships (Edges)
1. **PRODUCED_IN**
   - Links EF node to Region node: (EF)-[:PRODUCED_IN]->(Region)

2. **HAS_IMPACT**
   - Links Industry to EF: (Industry)-[:HAS_IMPACT]->(EF)

### 2. Set Up Neo4j Environment

#### A. Deployment Options
1. **Neo4j Aura Free Tier**
   - [Neo4j Aura](https://neo4j.com/cloud/aura/)
   - Create free instance (50K nodes limit)
   - Note connection URI, username, and password

2. **Local Docker-based Neo4j**
   ```bash
   docker run -d --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password \
     neo4j:latest
   ```

#### B. Configure Indexes / Constraints
```cypher
// Indexes for quick lookups
CREATE INDEX region_code_index FOR (r:Region) ON (r.country_code);
CREATE INDEX industry_id_index FOR (i:Industry) ON (i.industry_id);
CREATE INDEX ef_id_index FOR (e:EF) ON (e.ef_id);

// Uniqueness constraints
CREATE CONSTRAINT unique_region_id IF NOT EXISTS
FOR (r:Region) REQUIRE r.region_id IS UNIQUE;
```

### 3. Import Data into Neo4j

#### A. Prepare Data Files
- Export processed data as CSV/JSON.
- Create separate files for nodes and relationships if needed.

#### B. Import Options
1. **Using neo4j-admin import (Offline Bulk Import)**
   ```bash
   neo4j-admin import \
     --nodes=Region=regionnodes.csv \
     --nodes=Industry=industrynodes.csv \
     --nodes=EF=efnodes.csv \
     --relationships=PRODUCED_IN=produced_in.csv \
     --relationships=HAS_IMPACT=has_impact.csv \
     --database=<your-database-name>
   ```

2. **Using Cypher LOAD CSV (For Aura or Running Instance)**
   ```cypher
   // Loading Region nodes
   LOAD CSV WITH HEADERS FROM "file:///regionnodes.csv" AS row
   CREATE (r:Region {
     region_id: row.region_id,
     country_code: row.country_code,
     continent: row.continent,
     ipcc_region: row.ipcc_region
   });
   
   // Loading relationships
   LOAD CSV WITH HEADERS FROM "file:///produced_in.csv" AS row
   MATCH (e:EF {ef_id: row.ef_id}), (r:Region {region_id: row.region_id})
   CREATE (e)-[:PRODUCED_IN]->(r);
   ```

#### C. Verify Node Count
```cypher
MATCH (n) RETURN labels(n)[0], count(*) ORDER BY labels(n)[0];
```
- Confirm presence of 50K+ EF nodes.

### 4. Test Knowledge Graph

#### A. Data Integrity Queries
```cypher
// Count nodes
MATCH (e:EF) RETURN count(e) AS EFCount;
MATCH (r:Region) RETURN count(r) AS RegionCount;
MATCH (i:Industry) RETURN count(i) AS IndustryCount;

// Count relationships
MATCH ()-[rel:PRODUCED_IN]->() RETURN count(rel) AS ProducedInCount;
MATCH ()-[rel:HAS_IMPACT]->() RETURN count(rel) AS HasImpactCount;
```

#### B. Sample Queries
```cypher
// Find EF by Region
MATCH (r:Region {country_code:"FR"})<-[:PRODUCED_IN]-(e:EF)
RETURN e.value, e.unit, e.data_source
LIMIT 10;

// Find EF by Industry
MATCH (i:Industry {industry_id:"1111"})-[:HAS_IMPACT]->(e:EF)
RETURN e.ef_id, e.value, e.adjusted_ef
LIMIT 10;

// Cross-Reference
MATCH (i:Industry {industry_id:"1111"})-[:HAS_IMPACT]->(e:EF)-[:PRODUCED_IN]->(r:Region)
WHERE r.country_code = "US"
RETURN e.value, r.country_code, i.industry_name;
```

## Deliverables

1. **Unified Dataset**: CSV/JSON containing harmonized emission factors with regional adjustments
2. **Neo4j Knowledge Graph**: Graph database with 50K+ EF nodes mapped to regions and industries
3. **Documentation**:
   - Data dictionaries for all datasets
   - Mapping tables for regions and industries
   - IPCC multiplier reference
   - Neo4j schema diagram
4. **Scripts**:
   - Data extraction and cleaning scripts
   - Neo4j import scripts
   - Testing and validation queries

## Success Criteria

- Neo4j knowledge graph with 50K+ EF nodes
- Complete coverage of 44+ countries
- EF retrieval time <200ms
- MAPE <5% against EXIOBASE regional data
- Documentation of all regional adjustments with IPCC AR6 compliance
