# Data Processing Pipeline

This directory contains the data processing pipeline for the Carbon Emission Factor project.

## Directory Structure

- `raw/`: Raw data files downloaded from sources
- `interim/`: Intermediate processed data
- `processed/`: Final cleaned and harmonized datasets
- `scripts/`: Python scripts for data processing
  - `extractors/`: Dataset-specific extraction modules
  - `harmonization/`: Harmonization modules
  - `main.py`: Main script to run the pipeline
  - `utils.py`: Utility functions
- `logs/`: Log files
- `documentation/`: Documentation files

## Usage

1. Place raw data files in the `raw/` directory
2. Run the processing pipeline:
   ```bash
   python data/scripts/main.py
   ```
3. Find processed data in the `processed/` directory
4. Check logs in the `logs/` directory
