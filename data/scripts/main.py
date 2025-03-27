#!/usr/bin/env python3
"""
Main script for the Adaptive Global LCA Advisor data extraction and processing.
"""

import importlib
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

# Import utility functions
import utils

# Configure logging
LOG_DIR = "data/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(
    LOG_DIR, f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Constants
EXTRACTORS_DIR = "data/scripts/extractors"
HARMONIZATION_DIR = "data/scripts/harmonization"
PROCESSED_DIR = "data/processed"
FINAL_OUTPUT = "data/processed/harmonized_global_ef_dataset.csv"

# List of datasets to include in the processing pipeline
DATASETS_TO_INCLUDE = [
    "agribalyse",
    "useeio",
    "exiobase",
    "climate_trace",
    "ipcc",
    "ipcc_ar6",
    "openlca",
    "ipcc_efdb",
    "greet",
]


def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        "data/raw",
        "data/interim",
        "data/processed",
        "data/scripts",
        "data/documentation",
        "data/logs",
        "data/scripts/extractors",
        "data/scripts/harmonization",
    ]

    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory created or already exists: {directory}")


def get_extractor_modules():
    """
    Get all extractor modules from the extractors directory.

    Returns:
        List of extractor module names
    """
    extractor_files = [
        f
        for f in os.listdir(EXTRACTORS_DIR)
        if f.endswith("_extractor.py") and not f.startswith("__")
    ]

    extractor_modules = [os.path.splitext(f)[0] for f in extractor_files]
    logger.info(
        f"Found {len(extractor_modules)} extractor modules: {', '.join(extractor_modules)}"
    )

    return extractor_modules


def run_extractors():
    """
    Run all extractor modules to extract and clean datasets.

    Returns:
        Dictionary mapping dataset names to their cleaned file paths
    """
    logger.info("Starting extraction process for all datasets")

    extractor_modules = get_extractor_modules()
    cleaned_datasets = {}

    for module_name in extractor_modules:
        try:
            logger.info(f"Running extractor: {module_name}")
            start_time = time.time()

            # Import the module dynamically
            module_path = f"extractors.{module_name}"
            module = importlib.import_module(module_path)

            # Run the extract_and_clean function
            cleaned_file = module.extract_and_clean()

            # Store the cleaned file path
            dataset_name = module_name.replace("_extractor", "")
            cleaned_datasets[dataset_name] = cleaned_file

            elapsed_time = time.time() - start_time
            logger.info(f"Completed {module_name} in {elapsed_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error running extractor {module_name}: {e}", exc_info=True)

    return cleaned_datasets


def run_harmonization(cleaned_datasets):
    """
    Run the harmonization process on the cleaned datasets.

    Args:
        cleaned_datasets: Dictionary mapping dataset names to their cleaned file paths

    Returns:
        Path to the harmonized dataset
    """
    logger.info("Starting harmonization process")

    try:
        # Import the harmonization module
        harmonization_module = importlib.import_module("harmonization.harmonizer")

        # Run the harmonize function
        harmonized_file = harmonization_module.harmonize(cleaned_datasets, FINAL_OUTPUT)

        logger.info(f"Harmonization completed: {harmonized_file}")
        return harmonized_file

    except Exception as e:
        logger.error(f"Error in harmonization process: {e}", exc_info=True)
        raise


def generate_summary_report(harmonized_file):
    """
    Generate a summary report of the harmonized dataset.

    Args:
        harmonized_file: Path to the harmonized dataset

    Returns:
        Path to the summary report
    """
    logger.info("Generating summary report")

    try:
        # Read the harmonized dataset
        df = pd.read_csv(harmonized_file)

        # Generate summary statistics
        summary = {
            "total_records": len(df),
            "datasets_included": df["source_dataset"].nunique(),
            "regions_covered": df["region"].nunique(),
            "entity_types": df["entity_type"].value_counts().to_dict(),
            "average_confidence": df["confidence"].mean(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Count records by source dataset
        source_counts = df["source_dataset"].value_counts().to_dict()
        summary["records_by_source"] = source_counts

        # Count records by region
        region_counts = df["region"].value_counts().to_dict()
        summary["records_by_region"] = region_counts

        # Save summary to file
        summary_file = os.path.join(PROCESSED_DIR, "harmonized_dataset_summary.txt")

        with open(summary_file, "w") as f:
            f.write("# Harmonized Global Emission Factor Dataset Summary\n\n")
            f.write(f"Generated on: {summary['timestamp']}\n\n")

            f.write(f"Total records: {summary['total_records']}\n")
            f.write(f"Datasets included: {summary['datasets_included']}\n")
            f.write(f"Regions covered: {summary['regions_covered']}\n")
            f.write(
                f"Average confidence score: {summary['average_confidence']:.2f}\n\n"
            )

            f.write("## Records by entity type\n\n")
            for entity_type, count in summary["entity_types"].items():
                f.write(f"- {entity_type}: {count}\n")

            f.write("\n## Records by source dataset\n\n")
            for source, count in summary["records_by_source"].items():
                f.write(f"- {source}: {count}\n")

            f.write("\n## Records by region\n\n")
            for region, count in summary["records_by_region"].items():
                if count > 10:  # Only show regions with significant data
                    f.write(f"- {region}: {count}\n")

        logger.info(f"Summary report generated: {summary_file}")
        return summary_file

    except Exception as e:
        logger.error(f"Error generating summary report: {e}", exc_info=True)
        raise


def main():
    """Main function to run the entire extraction and processing pipeline."""
    logger.info("Starting Adaptive Global LCA Advisor data extraction and processing")

    try:
        # Create necessary directories
        create_directories()

        # Run all extractors
        cleaned_datasets = run_extractors()

        if not cleaned_datasets:
            logger.error("No datasets were successfully extracted and cleaned")
            return

        logger.info(
            f"Successfully extracted and cleaned {len(cleaned_datasets)} datasets"
        )

        # Run harmonization
        harmonized_file = run_harmonization(cleaned_datasets)

        # Generate summary report
        summary_file = generate_summary_report(harmonized_file)

        logger.info("Data extraction and processing completed successfully")
        logger.info(f"Harmonized dataset: {harmonized_file}")
        logger.info(f"Summary report: {summary_file}")

    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
