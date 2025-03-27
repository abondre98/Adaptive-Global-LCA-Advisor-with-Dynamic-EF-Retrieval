#!/usr/bin/env python3
import os
import shutil


def create_directory_structure():
    # Define the base directories
    base_dirs = [
        "data/raw",
        "data/interim",
        "data/processed",
        "data/scripts/extractors",
        "data/scripts/harmonization",
        "data/logs",
        "data/documentation",
    ]

    # Create directories
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create __init__.py files
    init_files = [
        "data/__init__.py",
        "data/scripts/__init__.py",
        "data/scripts/extractors/__init__.py",
        "data/scripts/harmonization/__init__.py",
    ]

    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write('"""Data processing modules."""\n')
            print(f"Created file: {init_file}")

    # Create main.py
    main_py_content = '''#!/usr/bin/env python3
"""
Main script for data processing pipeline.
"""

import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/data_processing.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

def main():
    """Main function to run the data processing pipeline."""
    logger.info("Starting data processing pipeline...")
    # Add your pipeline steps here
    logger.info("Data processing pipeline completed.")

if __name__ == "__main__":
    main()
'''

    main_py_path = "data/scripts/main.py"
    if not os.path.exists(main_py_path):
        with open(main_py_path, "w") as f:
            f.write(main_py_content)
        print(f"Created file: {main_py_path}")

    # Create utils.py
    utils_py_content = '''#!/usr/bin/env python3
"""
Utility functions for data processing.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        "data/raw",
        "data/interim",
        "data/processed",
        "data/logs",
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def validate_data(data: Dict[str, Any]) -> bool:
    """
    Validate data structure and content.
    
    Args:
        data: Dictionary containing data to validate
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    required_fields = ["entity_id", "entity_name", "ef_value", "ef_unit"]
    
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return False
    
    return True

def clean_value(value: Any) -> Any:
    """
    Clean and standardize data values.
    
    Args:
        value: Value to clean
        
    Returns:
        Cleaned value
    """
    if isinstance(value, str):
        return value.strip().lower()
    return value
'''

    utils_py_path = "data/scripts/utils.py"
    if not os.path.exists(utils_py_path):
        with open(utils_py_path, "w") as f:
            f.write(utils_py_content)
        print(f"Created file: {utils_py_path}")

    # Create sample extractor
    extractor_content = '''#!/usr/bin/env python3
"""
Sample data extractor module.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def extract_data(source_path: str) -> List[Dict[str, Any]]:
    """
    Extract data from source file.
    
    Args:
        source_path: Path to source data file
        
    Returns:
        List of extracted data records
    """
    logger.info(f"Extracting data from: {source_path}")
    # Add your extraction logic here
    return []
'''

    extractor_path = "data/scripts/extractors/sample_extractor.py"
    if not os.path.exists(extractor_path):
        with open(extractor_path, "w") as f:
            f.write(extractor_content)
        print(f"Created file: {extractor_path}")

    # Create sample harmonization module
    harmonization_content = '''#!/usr/bin/env python3
"""
Sample data harmonization module.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def harmonize_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Harmonize data records.
    
    Args:
        data: List of data records to harmonize
        
    Returns:
        List of harmonized data records
    """
    logger.info(f"Harmonizing {len(data)} records")
    # Add your harmonization logic here
    return data
'''

    harmonization_path = "data/scripts/harmonization/sample_harmonizer.py"
    if not os.path.exists(harmonization_path):
        with open(harmonization_path, "w") as f:
            f.write(harmonization_content)
        print(f"Created file: {harmonization_path}")

    # Create README.md in data directory
    readme_content = """# Data Processing Pipeline

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
"""

    readme_path = "data/README.md"
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"Created file: {readme_path}")


if __name__ == "__main__":
    create_directory_structure()
    print("\nDirectory structure and files created successfully!")
