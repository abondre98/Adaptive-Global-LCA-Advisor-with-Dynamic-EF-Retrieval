# IPCC AR6 Data Extractor

This module extracts and processes data from the IPCC's Sixth Assessment Report (AR6), focusing on the Working Group III report on climate change mitigation. The extractor uses advanced PDF parsing techniques to extract tables and figures from the AR6 WG3 report, particularly from Chapter 2 (Emissions Trends and Drivers), and converts them into regional multipliers that can be used to adjust emission factors across different regions, sectors, time periods, and gas types.

## Features

### 1. Enhanced PDF Extraction

- Utilizes PyMuPDF (imported as `fitz`) for PDF text extraction and figure extraction
- Uses Camelot for accurate table extraction from PDF documents
- Identifies and extracts tables and figures containing emissions data
- Targets Chapter 2 (Emissions Trends and Drivers) specifically for emissions trend data

### 2. Expanded Regional Coverage

- 25 regions (up from the original 10)
- **Original regions**: Global, North America, Europe, East Asia, South Asia, Latin America, Africa, Middle East, Oceania, Southeast Asia
- **New regions**: United States, Canada, Mexico, Brazil, European Union, United Kingdom, Russia, China, Japan, India, Australia, South Africa, Middle East and North Africa, Sub-Saharan Africa, Least Developed Countries
- All regions mapped to standardized ISO codes for consistency

### 3. Detailed Sectoral Breakdown

- 27 sectors (up from the original 8)
- **Main sectors**: Energy, Industry, Transport, Buildings, Agriculture, Forestry, Waste, Cross-sector
- **Energy subsectors**: Electricity and Heat, Petroleum Refining, Fossil Fuel Extraction, Other Energy Industries
- **Industry subsectors**: Iron and Steel, Non-ferrous Metals, Chemicals, Cement, Pulp and Paper, Food Processing
- **Transport subsectors**: Road, Aviation, Shipping, Rail
- **Buildings subsectors**: Residential, Commercial
- **AFOLU subsectors**: Cropland, Livestock, Forestry and Land Use

### 4. Time Series Data

- Includes multipliers for four distinct time periods:
  - 1990-2000 (Historical baseline)
  - 2000-2010 (Growth period)
  - 2010-2019 (Improvement period)
  - 2019-present (Recent improvements)
- Time-specific adjustments to reflect changing emission patterns over time

### 5. Gas-Specific Multipliers

- Separate multipliers for different greenhouse gases:
  - CO2 (Carbon dioxide)
  - CH4 (Methane)
  - N2O (Nitrous oxide)
  - F-gases (Fluorinated gases)
- Gas-specific adjustments to reflect varying regional patterns

## Data Structure

The IPCC AR6 dataset produces multipliers with the following attributes:

- **Region**: Geographic area (25 options)
- **Sector**: Economic or activity sector (27 options)
- **Gas Type**: Greenhouse gas type (4 options)
- **Time Period**: Historical period (4 options)
- **Multiplier Value**: Numeric value for adjusting emission factors
- **Confidence**: Confidence score for the multiplier (0-1)

The final cleaned dataset follows the standard schema with these specific enhancements:

- **entity_id**: Format `IPCC_AR6_{region}_{sector}_{gas_type}_{time_period}`
- **entity_name**: Format `{sector} {gas_type} in {region} ({time_period})`
- **tags**: Includes sector, region, gas_type, time_period, and region_code tags
- **metadata**: JSON with ipcc_version, region_name, region_code, gas_type, and time_period

## Usage

The extractor can be used as a standalone script or as a module within the harmonization pipeline:

### As a standalone script:

```bash
python data/scripts/extractors/ipcc_ar6_extractor.py
```

### As a module:

```python
from data.scripts.extractors.ipcc_ar6_extractor import extract_and_clean

# Run the extractor
output_path = extract_and_clean()
print(f"Cleaned data saved to: {output_path}")
```

### Dependencies

This extractor requires these additional libraries:

- PyMuPDF (imported as `fitz`)
- camelot-py

Install them with:

```bash
pip install "camelot-py[cv]" PyMuPDF
brew install ghostscript  # For macOS users
```

## Output

The extractor produces these files:

- `data/raw/ipcc_ar6_raw.json`: Raw extracted or simulated data
- `data/interim/ipcc_ar6_interim.csv`: Interim processed data
- `data/processed/ipcc_ar6_clean.csv`: Final cleaned dataset with 10,768 records

## Examples

Example of a CO2 multiplier for the energy sector in China for 2010-2019:

```
IPCC_AR6_China_Energy_CO2_2010_2019,Energy CO2 in China (2010-2019),multiplier,0.95,ratio,China,IPCC_AR6,0.76,2023-05-01,"['sector:Energy', 'region:China', 'gas_type:CO2', 'time_period:2010-2019', 'region_code:CHN']",False,"{""ipcc_version"": ""AR6"", ""region_name"": ""China"", ""region_code"": ""CHN"", ""gas_type"": ""CO2"", ""time_period"": ""2010-2019""}"
```

Example of a CH4 multiplier for the agricultural sector in India for 2000-2010:

```
IPCC_AR6_India_Agriculture_CH4_2000_2010,Agriculture CH4 in India (2000-2010),multiplier,1.28,ratio,India,IPCC_AR6,0.68,2023-05-01,"['sector:Agriculture', 'region:India', 'gas_type:CH4', 'time_period:2000-2010', 'region_code:IND']",False,"{""ipcc_version"": ""AR6"", ""region_name"": ""India"", ""region_code"": ""IND"", ""gas_type"": ""CH4"", ""time_period"": ""2000-2010""}"
```

## PDF Extraction Process

The PDF extraction process follows these steps:

1. Download PDF files from the IPCC AR6 website
2. Identify Chapter 2 (Emissions Trends and Drivers) within the PDF
3. Extract tables using Camelot's stream mode for complex table structures
4. Extract figures using PyMuPDF's image extraction capabilities
5. Process extracted tables to identify emissions data
6. Convert extracted data to the standardized multiplier format
7. Clean and validate the dataset

If PDF extraction fails or produces insufficient data, the extractor falls back to a simulated dataset based on the structure and patterns of the IPCC AR6 report.

## Future Enhancements

Potential future enhancements for this extractor:

- Extract uncertainty ranges for multipliers from the report
- Add support for extracting scenario-based multipliers (SSP1-5)
- Implement improved table detection for complex nested tables
- Add support for policy-specific multipliers based on Chapter 13-14
- Expand to include data from the AR6 Synthesis Report
