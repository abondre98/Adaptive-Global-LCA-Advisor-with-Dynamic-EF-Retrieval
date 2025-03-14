# Adaptive Global LCA Advisor - Data Extraction and Harmonization

This project implements a data extraction, cleaning, and harmonization pipeline for creating a unified global emission factor dataset from multiple sources. The pipeline processes data from various emission factor databases and creates a standardized, harmonized dataset that can be used for life cycle assessment (LCA) calculations.

## Project Overview

The Adaptive Global LCA Advisor aims to develop an AI system that recommends region-specific emission factors (EFs) for accurate carbon accounting. This system addresses limitations in existing solutions like static datasets and single region focus. The project combines data from multiple sources to create a comprehensive global emission factor dataset.

## Fine-tuning Mistral-7B for Emission Factor Recommendations

To fine-tune the Mistral-7B model on our emission factor dataset, we provide a Google Colab notebook that handles the entire process. The notebook includes:

- Environment setup with GPU support
- Data preparation and loading
- Model configuration with LoRA
- Training loop with monitoring
- Evaluation and model export

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sbursu/Carbon-EF/blob/main/training/notebooks/mistral_finetuning.ipynb)

To use the notebook:

1. Click the "Open in Colab" badge above
2. Sign into your Google account
3. Set the runtime type to GPU (Runtime > Change runtime type)
4. Run the cells in sequence
5. Monitor the training progress in Weights & Biases

## Project Structure

```
data/
├── raw/                  # Raw data files downloaded from sources
├── interim/              # Intermediate processed data
├── processed/            # Final cleaned and harmonized datasets
├── scripts/              # Python scripts for data processing
│   ├── extractors/       # Dataset-specific extraction modules
│   ├── harmonization/    # Harmonization modules
│   ├── main.py           # Main script to run the pipeline
│   └── utils.py          # Utility functions
├── logs/                 # Log files
└── documentation/        # Documentation files
```

## Datasets

The pipeline processes the following datasets:

1. **Agribalyse 3.1** - French agricultural product emission factors (2,793 records)
2. **USEEIO v2.1** - US environmentally-extended input-output model (13,561 records)
3. **EXIOBASE 3.8** - Multi-regional input-output database (1,030 records)
4. **Climate TRACE** - Global emissions by sector and country (4,681 records)
5. **IPCC AR6** - Enhanced regional multipliers with time series and gas-specific data (10,769 records)
6. **OpenLCA** - Process-based LCA data for a wide range of products and services (961 records)
7. **IPCC EFDB** - IPCC Emission Factor Database containing emission factors for various sectors and gases (191 records)
8. **GREET Model** - Greenhouse gases, Regulated Emissions, and Energy use in Transportation Model from Argonne National Laboratory, focusing on transportation fuels and vehicle technologies (234 records)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/adaptive-global-lca-advisor.git
cd adaptive-global-lca-advisor
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Additional dependencies for enhanced PDF extraction:

```bash
pip install "camelot-py[cv]" PyMuPDF
brew install ghostscript  # For macOS users
```

## Usage

Run the main script to execute the entire pipeline:

```bash
python data/scripts/main.py
```

This will:

1. Download and extract data from all sources (or use simulated data if sources are unavailable)
2. Clean and standardize each dataset
3. Harmonize all datasets into a unified format
4. Generate a summary report and metadata

## Pipeline Steps

### 1. Data Extraction

Each dataset has a dedicated extractor module that:

- Downloads data from the source (or creates simulated data if unavailable)
- Extracts relevant files
- Validates the data structure

#### Dataset-Specific Extraction Processes:

- **Agribalyse 3.1**: Attempts to download from multiple official sources with robust validation; falls back to an enhanced simulated dataset with 2,793 food products across 28 detailed categories, representing the actual distribution of the official Agribalyse database. The extractor includes improved error handling, encoding detection, and comprehensive validation to ensure data quality.
- **USEEIO v2.1**: Clones the EPA GitHub repository and extracts emission factor data from CSV files
- **EXIOBASE 3.8**: Attempts to download from Zenodo; falls back to simulated data with 1,030 product-country combinations
- **Climate TRACE**: Simulates real-time emissions data for 50 sectors across multiple regions
- **IPCC AR6**: Enhanced extractor that uses advanced PDF parsing techniques to extract tables and figures from the IPCC AR6 WG3 report, particularly from Chapter 2 (Emissions Trends and Drivers). The extractor now includes:
  - **Expanded Regional Coverage**: 25 regions (up from 10), including specific countries and regional groupings
  - **Detailed Sectoral Breakdown**: 27 sectors (up from 8), representing the subsectors mentioned in the IPCC report
  - **Time Series Data**: Includes multipliers for four time periods (1990-2000, 2000-2010, 2010-2019, 2019-present)
  - **Gas-Specific Multipliers**: Separate multipliers for different greenhouse gases (CO2, CH4, N2O, F-gases)
  - **PDF Extraction**: Utilizes libraries like PyMuPDF and Camelot for accurate table and figure extraction
- **OpenLCA**: Attempts to access the OpenLCA Nexus; falls back to simulated data with 961 processes
- **IPCC EFDB**: Simulates emission factors for various sectors and gases across different regions
- **GREET Model**: Simulates transportation fuel lifecycle emissions for various fuel pathways and vehicle technologies

### 2. Data Cleaning

The cleaning process for each dataset:

- Standardizes column names
- Converts values to appropriate data types
- Removes duplicates and invalid entries
- Detects outliers (using statistical methods)
- Creates a standardized schema

### 3. Data Harmonization

The harmonization process:

- Creates a crosswalk between similar entities across datasets
- Standardizes units to kg CO2e
- Applies regional multipliers from IPCC AR6
- Merges all datasets into a single harmonized dataset
- Generates metadata and a summary report

### 4. Quality Assurance

- **Completeness**: Ensures >95% of critical fields are populated
- **Consistency**: Cross-references between datasets to flag discrepancies
- **Accuracy**: Validates against known benchmarks
- **Timeliness**: Flags data older than 3 years for review
- **Outlier Detection**: Identifies statistical outliers in emission factor values
- **Duplicate Prevention**: Ensures no duplicate records exist in the final dataset

## Output Files

The main outputs of the pipeline are:

- `data/processed/harmonized_global_ef_dataset.csv` - The harmonized dataset (23,533 records)
- `data/processed/harmonized_global_ef_dataset_metadata.json` - Metadata for the harmonized dataset
- `data/processed/harmonized_dataset_summary.txt` - Summary report with statistics

## Dataset Statistics

The final harmonized dataset contains:

- **Total Records**: 34,220 (from raw datasets before harmonization)
- **Harmonized Records**: 23,520 (after deduplication and merging)
- **Regions Covered**: 101 (including additional regions from IPCC AR6)
- **Entity Types**: 13 different types (product, process, sector, energy, manufacturing, agriculture, transportation, buildings, fuel_pathway, etc.)
- **Average Confidence Score**: 0.74
- **Data Quality Indicators**:
  - **Outliers**: 474 records (2.0%) - all in the product category
    - By source: Agribalyse_3.1 (336), EXIOBASE_3.8 (138)
  - **Regional Adjustments**: 1,233 records (5.2%) across various entity types:
    - Products: 532 records
    - Sectors: 482 records
    - Energy: 93 records
    - Manufacturing: 39 records
    - Agriculture: 32 records
    - Buildings: 27 records
    - Transportation: 21 records
    - Other categories: 7 records
    - By source: USEEIO_v2.1 (482), Agribalyse_3.1 (381), Climate_TRACE (162), EXIOBASE_3.8 (151), IPCC_EFDB (57)
  - **Confidence Scores**:
    - High (>0.7): 23,419 records (99.6%)
    - Medium (0.6-0.7): 67 records (0.3%)
    - Low (<0.6): 34 records (0.1%)
  - **Emission Factor Values**:
    - Very Low (<1): 18,339 records (78.0%)
    - Low (1-10): 404 records (1.7%)
    - Medium (10-100): 36 records (0.2%)
    - High (100-1000): 1 record (<0.1%)
    - Very High (>1000): 4,740 records (20.2%)
  - **Emission Factor Units**:
    - kg CO2e: 12,403 records (52.7%)
    - kg/USD: 10,839 records (46.1%)
    - Other units: 278 records (1.2%) including ratio, kg/TJ, t/TJ, kg/kWh, kg/t, etc.
  - **Temporal Coverage**:
    - All records are timestamped with the year 2025
  - **Source Datasets**:
    - USEEIO_v2.1: 13,548 records (57.6%)
    - Climate_TRACE: 4,680 records (19.9%)
    - Agribalyse_3.1: 2,792 records (11.9%)
    - EXIOBASE_3.8: 1,029 records (4.4%)
    - OpenLCA: 960 records (4.1%)
    - GREET_Model: 233 records (1.0%)
    - IPCC_EFDB: 130 records (0.6%)
    - IPCC_AR6: 88 records (0.4%)
    - Regional codes (NAM, SAM, AFR, etc.): 60 records (0.3%)
  - Geographic Coverage:
    - Global (GLB): 12,819 records (54.5%)
    - France (FR): 2,813 records (12.0%)
    - United States (USA): 2,191 records (9.3%)
    - Other countries: 5,697 records (24.2%) covering major economies including ZAF, TUR, SAU, RUS, NLD, MEX, KOR, JPN, ITA, IND, IDN, GBR, etc.

### Records by Entity Type

- Sectors: 13,548
- Products: 3,821
- Energy: 1,490
- Manufacturing: 1,080
- Agriculture: 960
- Processes: 960
- Transportation: 720
- Buildings: 540
- Fuel Pathways: 233
- Multipliers: 88
- Industrial Processes: 40
- Waste: 30
- Other: 10

### Records by Region

- Global: 12,826
- France: 2,813
- United States: 2,191
- Country-specific data: 5,690 (across 60+ countries)

## Dataset Schema

The harmonized dataset follows this schema:

| Column         | Description                                                              |
| -------------- | ------------------------------------------------------------------------ |
| id             | Unique identifier                                                        |
| entity_id      | Original identifier from source dataset                                  |
| entity_name    | Name of the entity (product, sector, etc.)                               |
| entity_type    | Type of entity (product, sector, process, multiplier)                    |
| ef_value       | Emission factor value                                                    |
| ef_unit        | Unit of the emission factor                                              |
| region         | Region or country code                                                   |
| source_dataset | Source dataset name                                                      |
| confidence     | Confidence score (0-1)                                                   |
| timestamp      | Timestamp of extraction                                                  |
| tags           | List of tags for categorization (includes gas type and time period data) |
| is_outlier     | Flag for outlier values                                                  |
| metadata       | JSON string with additional metadata (includes gas type and time period) |

## Enhanced IPCC AR6 Extractor

The IPCC AR6 extractor has been significantly enhanced to provide a more comprehensive and detailed dataset. Key improvements include:

### 1. Enhanced PDF Extraction

- Implemented advanced PDF parsing techniques using PyMuPDF and Camelot
- Created specialized functions to extract tables and figures from PDF documents
- Added a specific function to target Chapter 2 (Emissions Trends and Drivers) of the IPCC AR6 WG3 report

### 2. Expanded Regional Coverage

- Increased from 10 to 25 regions
- Added specific countries (USA, China, India, etc.)
- Included new regional groupings (EU, MENA, LDCs, etc.)
- Mapped all regions to standardized ISO codes

### 3. Detailed Sectoral Breakdown

- Expanded from 8 to 27 sectors
- Added detailed subsectors for energy, industry, transport, buildings, and AFOLU
- Aligned with the IPCC AR6 reporting structure

### 4. Time Series Data

- Added data for four distinct time periods:
  - 1990-2000 (Historical baseline)
  - 2000-2010 (Growth period)
  - 2010-2019 (Improvement period)
  - 2019-present (Recent improvements)
- Each multiplier includes time-specific adjustments

### 5. Gas-Specific Multipliers

- Created separate multipliers for:
  - CO2 (Carbon dioxide)
  - CH4 (Methane)
  - N2O (Nitrous oxide)
  - F-gases (Fluorinated gases)
- Each gas has specific adjustment factors based on typical emission patterns

These enhancements have expanded the IPCC AR6 dataset from 88 records to 10,769 records, providing much more detailed and specific multipliers for accurately adjusting emission factors across regions, sectors, time periods, and gas types.

## Access to Full Datasets

For access to the actual complete datasets (rather than the simulated versions):

- **Agribalyse 3.1**: Available through the ADEME data portal after registration or through platforms like SimaPro and openLCA. The actual dataset contains approximately 2,500 food products with detailed carbon footprint information. Contact ADEME directly for direct dataset access.
- **IPCC AR6**: The official IPCC AR6 reports are available at [https://www.ipcc.ch/report/ar6/wg3/](https://www.ipcc.ch/report/ar6/wg3/). For access to the underlying data, researchers can contact the IPCC Data Distribution Centre.

## License

[Add your license information here]

## Contributors

[Add contributor information here]
