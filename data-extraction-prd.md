# Product Requirements Document:

# Data Extraction and Cleaning for Adaptive Global LCA Advisor

## 1. Introduction and Project Context

### 1.1 Purpose

This document outlines the requirements for the data extraction and cleaning phase of the Adaptive Global LCA Advisor project, focusing on Milestone 1: Regional EF Knowledge Graph. This phase will establish the foundation for a unified dataset with regional adjustments that will power the AI-driven emission factor recommendation system.

### 1.2 Project Overview

The Adaptive Global LCA Advisor aims to develop an AI system that recommends region-specific emission factors (EFs) for accurate carbon accounting. This system addresses limitations in existing solutions like Parakeet's static datasets and LEAF's single region focus. The project combines LLM fine-tuning with dynamic RAG for real-time EF updates.

### 1.3 Scope

This PRD covers:

- Data sources identification and prioritization
- Data extraction methodologies
- Data cleaning procedures
- Data harmonization and standardization
- Storage architecture
- Quality assurance and validation

## 2. Business Requirements

### 2.1 Problem Statement

- 73% of food products lack environmental labels
- Manual EF selection causes 15-30% error rates in carbon reporting
- EU's Carbon Border Adjustment Mechanism (CBAM) requires precise regional EF mapping
- Existing solutions provide either static datasets or focus on single regions

### 2.2 Success Criteria

- Creation of a unified dataset containing ~34,220 EF nodes (23,520 after harmonization)
- Coverage of EFs for 101 regions including major economies (54.5% global coverage)
- Database structure enabling <200ms retrieval times
- Harmonized data schema across 8 distinct source datasets
- Documented regional adjustment factors with 1,233 records (5.2%) having regional multipliers applied
- Reduction of error rates from 15-30% to <5% MAPE against EXIOBASE regional data

## 3. Data Sources

### 3.1 Primary Datasets

| Dataset        | Purpose                                 | Source URL                                               | Priority | Actual Record Count |
| -------------- | --------------------------------------- | -------------------------------------------------------- | -------- | ------------------- |
| Agribalyse 3.1 | Base agricultural EFs (EU focus)        | https://agribalyse.ademe.fr                              | High     | 2,792               |
| USEEIO v2.1    | US industrial emissions                 | https://github.com/USEPA/USEEIO                          | High     | 13,548              |
| EXIOBASE 3.8   | Multi-regional EEIO data                | https://www.exiobase.eu                                  | High     | 1,029               |
| Climate TRACE  | Real-time emission updates              | https://climatetrace.org                                 | Medium   | 4,680               |
| IPCC AR6       | Regional multipliers                    | https://www.ipcc.ch/report/sixth-assessment-report-cycle | High     | 88                  |
| OpenLCA        | Process-based LCA data                  | https://nexus.openlca.org/                               | Medium   | 960                 |
| IPCC EFDB      | Emission factors by sector and gas      | https://www.ipcc-nggip.iges.or.jp/EFDB/main.php          | Medium   | 130                 |
| GREET Model    | Transportation fuel lifecycle emissions | https://greet.anl.gov/                                   | Medium   | 234                 |

### 3.2 Dataset Selection Criteria

- **Coverage**: Prioritize datasets with broad product and regional coverage
- **Recency**: Favor most recent versions (e.g., Agribalyse 3.1 over 3.0)
- **Granularity**: Prefer sources with product-specific rather than category-level data
- **Format**: Prioritize structured formats (CSV, JSON) over PDFs or web scraping requirements
- **Update Frequency**: Consider how often the source is refreshed (especially Climate TRACE)

### 3.3 Dataset Values

- **Agribalyse 3.1**: Critical for EU context, especially for comparing French products with other EU regions
- **USEEIO v2.1**: Provides robust baseline for US products and cross-comparison with EU data
- **EXIOBASE 3.8**: Essential for cross-border comparisons (e.g., French vs. Indian wheat) critical for CBAM compliance
- **Climate TRACE**: Provides dynamic updates to reduce errors from outdated data
- **IPCC AR6**: Supplies scientifically-backed multipliers to adjust generic EFs for regional specificity

## 4. Detailed Extraction Workflows

### 4.1 Pre-Extraction Setup

1. Create a standardized directory structure:

```
/data
  /raw                # Original unmodified datasets
  /interim            # Intermediate processing results
  /processed          # Final cleaned datasets
  /scripts            # Extraction and processing scripts
  /documentation      # Data dictionaries and changelog
```

2. Establish version control procedures:
   - Commit raw data with checksums
   - Log all transformation steps
   - Tag stable processed versions

### 4.2 Agribalyse 3.1 Extraction

#### 4.2.1 Process

1. Visit https://agribalyse.ademe.fr and navigate to download section
2. Download bulk dataset in CSV/XLSX format
3. Check file integrity and encoding (UTF-8)
4. Verify version matches 3.1
5. Extract key columns:
   - Product identifiers and categories
   - Emission factors and units
   - Regional indicators
   - Production methods

#### 4.2.2 Transformation Steps

1. Convert to UTF-8 CSV if needed
2. Standardize column names to English
3. Apply initial cleaning rules
4. Store raw file as `agribalyse_3.1_raw.csv`

#### 4.2.3 Output

- Cleaned file: `processed/agribalyse_3.1_clean.csv`
- Required fields:
  - `product_id`: Unique identifier
  - `product_name`: Human-readable name
  - `product_category`: Classification
  - `ef_value`: Emission factor value
  - `ef_unit`: Unit of measurement (standardized to kg CO₂e/kg)
  - `region`: ISO country code or region identifier
  - `source`: "Agribalyse_3.1"
  - `timestamp`: Dataset version date

### 4.3 USEEIO v2.1 Extraction

#### 4.3.1 Process

1. Clone repository: `git clone https://github.com/USEPA/USEEIO`
2. Navigate to model directory for sector-specific EF data
3. Identify relevant CSV files containing emission factors
4. Extract matrices for direct emissions by sector

#### 4.3.2 Transformation Steps

1. Convert matrix format to tabular structure
2. Map NAICS/ISIC codes to standardized sector categories
3. Normalize units to match Agribalyse
4. Add region code "US" if not already specified

#### 4.3.3 Output

- Cleaned file: `processed/useeio_v2.1_clean.csv`
- Required fields:
  - `sector_id`: Industry classification code
  - `sector_name`: Human-readable sector name
  - `ef_value`: Emission factor value
  - `ef_unit`: Unit of measurement (standardized)
  - `region`: "US" or specific US state if available
  - `source`: "USEEIO_v2.1"
  - `timestamp`: Dataset version date

### 4.4 EXIOBASE 3.8 Extraction

#### 4.4.1 Process

1. Access EXIOBASE via https://www.exiobase.eu or Zenodo
2. Download relevant I/O tables and environmental extensions
3. Identify product and region codes relevant to project scope
4. Extract data using pandas or specialized I/O table parsers

#### 4.4.2 Transformation Steps

1. Restructure matrix data into tabular format
2. Map product codes to standardized categories
3. Convert units to kg CO₂e per functional unit
4. Extract regional multipliers for cross-border comparison

#### 4.4.3 Output

- Cleaned file: `processed/exiobase_3.8_clean.csv`
- Required fields:
  - `product_id`: EXIOBASE product code
  - `product_name`: Human-readable name
  - `ef_value`: Emission factor value
  - `ef_unit`: Unit of measurement (standardized)
  - `region`: ISO country code
  - `source`: "EXIOBASE_3.8"
  - `timestamp`: Dataset version date

### 4.5 Climate TRACE Integration

#### 4.5.1 Process

1. Register for Climate TRACE API access
2. Review API documentation for relevant endpoints
3. Develop Python client to interact with API
4. Set up authentication and request handling

#### 4.5.2 Implementation

1. Create scheduled job for weekly data retrieval
2. Focus on endpoints for agricultural and industrial emissions
3. Parse JSON responses and convert to standardized format
4. Implement error handling and retry logic

#### 4.5.3 Output

- API client script: `scripts/climate_trace_client.py`
- Regular updates to: `processed/climate_trace_latest.csv`
- Required fields:
  - `emission_source`: Activity or product
  - `ef_value`: Latest emission factor
  - `region`: Country or region code
  - `timestamp`: Update timestamp
  - `confidence`: Data quality indicator

### 4.6 IPCC AR6 Regional Multipliers

#### 4.6.1 Process

1. Access AR6 reports from https://www.ipcc.ch/report/sixth-assessment-report-cycle/
2. Identify sections containing regional factors and adjustments
3. Extract relevant multipliers for key regions and sectors
4. Document scientific basis for each multiplier

#### 4.6.2 Transformation

1. Structure multipliers in tabular format
2. Map to standardized region and sector codes
3. Add metadata including confidence levels and citations

#### 4.6.3 Output

- Multiplier table: `processed/ipcc_ar6_multipliers.csv`
- Required fields:
  - `region`: ISO country code or region
  - `sector`: Industry or product category
  - `multiplier_factor`: Numerical adjustment factor
  - `rationale`: Text explanation for adjustment
  - `source_page`: Reference to AR6 documentation

## 5. Data Cleaning Requirements

### 5.1 General Cleaning Rules

1. **Standardize Units**:

   - Convert all EF values to kg CO₂e per functional unit
   - Document conversion factors used
   - Flag any uncertain conversions for review
   - Actual results: 52.7% in kg CO2e, 46.1% in kg/USD, 1.2% in other units

2. **Normalize Names**:

   - Create master product/sector name mappings
   - Standardize capitalization and terminology
   - Remove special characters and normalize spaces

3. **Handle Missing Values**:

   - For critical fields (product_id, ef_value): reject or flag for review
   - For non-critical fields: use sensible defaults where appropriate
   - Document all imputation strategies

4. **Remove Duplicates**:

   - Create unique composite keys (product+region+source)
   - For genuine duplicates, keep most recent or highest quality
   - Log all duplicate resolution decisions
   - Actual results: Identified and removed duplicate records, bringing total from 34,220 to 23,520

5. **Outlier Detection**:
   - Implement robust outlier detection using both Z-score and IQR methods
   - Analyze outliers within entity type groups for better context
   - Flag statistical outliers while considering domain-specific patterns
   - Actual results: 474 records (2.0%) identified as outliers, all in the product category

### 5.2 Source-Specific Cleaning

#### 5.2.1 Agribalyse

- Translate French terminology to English using standardized mappings
- Convert functional units to per-kg basis where possible
- Map French regional indicators to ISO codes
- Standardize product categories to match global classification

#### 5.2.2 USEEIO

- Resolve many-to-many relationship between NAICS codes and products
- Normalize sector descriptions to match other datasets
- Extract state-specific data where available
- Convert economic input-output values to physical units

#### 5.2.3 EXIOBASE

- Handle multi-regional flows appropriately
- Resolve overlapping product categories
- Extract embedded regional factors
- Normalize monetary units to physical quantities

#### 5.2.4 Climate TRACE

- Handle API response variations
- Implement validation for real-time data
- Tag data with confidence scores
- Manage temporal aspects (historical vs. current)

### 5.3 Data Quality Metrics

1. **Completeness**: Percentage of fields populated

   - Target: >95% for critical fields
   - Actual results: All critical fields populated across harmonized dataset

2. **Consistency**: Cross-reference between datasets

   - Check same products across different sources
   - Flag discrepancies >20% for review

3. **Accuracy**: Validate against known benchmarks

   - Compare against published LCA studies
   - Document validation methodology
   - Actual confidence scores: 99.6% of records have high confidence (>0.7)

4. **Timeliness**: Track data freshness
   - Flag data older than 3 years
   - Prioritize updates for outdated high-impact sectors
   - Actual results: All records timestamped with year 2025

## 6. Data Harmonization

### 6.1 Schema Standardization

Create a unified schema with consistent field names and formats:

```python
standard_schema = {
    "entity_id": "str",         # Unique identifier
    "entity_name": "str",       # Human-readable name
    "entity_type": "str",       # "product", "sector", or "process"
    "ef_value": "float",        # Emission factor value
    "ef_unit": "str",           # Unit (kg CO₂e per functional unit)
    "region": "str",            # ISO country code
    "source_dataset": "str",    # Original source
    "confidence": "float",      # Confidence rating (0-1)
    "timestamp": "datetime",    # Last updated
    "tags": "list[str]"         # Additional classifiers
}
```

### 6.2 Cross-Dataset Mapping

#### 6.2.1 Product Mapping

1. Create master product taxonomy

   - Start with broadest dataset (likely Agribalyse)
   - Map other datasets to this taxonomy
   - Document mapping decisions

2. Develop crosswalk tables
   - Map product codes between datasets
   - Handle many-to-many relationships
   - Preserve original identifiers

#### 6.2.2 Regional Mapping

1. Standardize to ISO 3166-1 alpha-2 country codes
2. Create hierarchical structure
   - Country → Region → Continent → Global
3. Handle special cases
   - EU as both region and collection of countries
   - Special economic zones
   - Disputed territories
4. Actual regional distribution:
   - Global (GLB): 54.5% of records
   - France (FR): 12.0% of records
   - United States (USA): 9.3% of records
   - 60+ other countries: 24.2% of records

#### 6.2.3 Unit Conversion

1. Standardize all values to kg CO₂e
2. Document conversion factors by gas type
3. Handle functional unit variations
   - Per kg of product
   - Per functional unit (e.g., 1 serving)
   - Per economic value

### 6.3 Metadata Enrichment

1. Add confidence scores (0-1) for each harmonized record
   - Actual distribution: High (>0.7): 23,419 records (99.6%), Medium (0.6-0.7): 67 records (0.3%), Low (<0.6): 34 records (0.1%)
2. Tag with data provenance information
3. Include timestamp information
   - Original source date
   - Processing date
   - Expected validity period

### 6.4 Harmonization Workflow

1. Process each dataset through standardization pipeline
2. Apply crosswalk mappings
3. Merge datasets with conflict resolution rules
   - Prefer higher confidence sources
   - Prefer more recent data
   - Average values with similar confidence
4. Apply IPCC regional multipliers
   - Actual results: 1,233 records (5.2%) have multipliers applied
   - Distribution by entity type: Products (532), Sectors (482), Energy (93), Manufacturing (39), Agriculture (32), Buildings (27), Transportation (21), Other (7)
   - Distribution by source: USEEIO_v2.1 (482), Agribalyse_3.1 (381), Climate_TRACE (162), EXIOBASE_3.8 (151), IPCC_EFDB (57)
5. Generate final unified dataset
   - Entity types: Sectors (13,548), Products (3,821), Energy (1,490), Manufacturing (1,080), Agriculture (960), Processes (960), Transportation (720), Buildings (540), Fuel Pathways (234), other types (167)
   - Emission factor values: Very Low (<1): 78.0%, Low (1-10): 1.7%, Medium (10-100): 0.2%, High (100-1000): <0.1%, Very High (>1000): 20.2%

## 7. Quality Assurance and Implementation Results

### 7.1 Implementation Overview

The implementation has successfully delivered a harmonized dataset with the following characteristics:

- Total Records: 34,220 (from raw datasets before harmonization)
- Harmonized Records: 23,520 (after deduplication and merging)
- Regions Covered: 101 (including additional regions from IPCC AR6)
- Entity Types: 13 different types (product, process, sector, energy, manufacturing, agriculture, transportation, buildings, fuel_pathway, etc.)

### 7.2 Outlier Detection Implementation

The outlier detection strategy was implemented with a robust approach:

1. **Combined Methods**: Used both Z-score (for normal distributions) and IQR (for skewed distributions)
2. **Contextual Analysis**: Applied detection within entity type groups rather than globally
3. **Results**: 474 records (2.0%) identified as outliers
   - Distribution by source: Agribalyse_3.1 (336), EXIOBASE_3.8 (138)
   - All outliers are in the product category

### 7.3 Regional Multiplier Application

Regional multipliers were applied based on the following strategy:

1. **Source-based Strategy**: IPCC AR6 provided the scientific basis for multipliers
2. **Implementation**: Applied to 1,233 records (5.2% of dataset)
3. **Distribution**:
   - By entity type: Products (532), Sectors (482), Energy (93), Manufacturing (39), Agriculture (32), Buildings (27), Transportation (21), Other (7)
   - By source: USEEIO_v2.1 (482), Agribalyse_3.1 (381), Climate_TRACE (162), EXIOBASE_3.8 (151), IPCC_EFDB (57)

### 7.4 Data Quality Results

The implemented dataset demonstrates high quality characteristics:

1. **Confidence Scores**:

   - 99.6% of records have high confidence (>0.7)
   - 0.3% have medium confidence (0.6-0.7)
   - 0.1% have low confidence (<0.6)

2. **Emission Factor Values**:

   - Very Low (<1): 18,339 records (78.0%)
   - Low (1-10): 404 records (1.7%)
   - Medium (10-100): 36 records (0.2%)
   - High (100-1000): 1 record (<0.1%)
   - Very High (>1000): 4,740 records (20.2%)

3. **Standardized Units**:
   - kg CO2e: 12,403 records (52.7%)
   - kg/USD: 10,839 records (46.1%)
   - Other units: 278 records (1.2%) including ratio, kg/TJ, t/TJ, kg/kWh, kg/t, etc.
