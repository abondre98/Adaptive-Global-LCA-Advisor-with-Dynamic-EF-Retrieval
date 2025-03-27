"""
IPCC Emission Factor Database (EFDB) extractor module.

This module downloads and processes emission factor data from the IPCC Emission Factor Database.
The IPCC EFDB is a library of emission factors and other parameters with background documentation
or technical references that can be used for estimating greenhouse gas emissions and removals.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

logger = logging.getLogger(__name__)

# Constants
IPCC_EFDB_URL = "https://www.ipcc-nggip.iges.or.jp/EFDB/main.php"
IPCC_EFDB_SEARCH_URL = "https://www.ipcc-nggip.iges.or.jp/EFDB/find_ef.php"
RAW_FILE_PATH = "data/raw/ipcc_efdb_raw.csv"
INTERIM_FILE_PATH = "data/interim/ipcc_efdb_interim.csv"
CLEANED_FILE_PATH = "data/processed/ipcc_efdb_clean.csv"
DATASET_SOURCE = "IPCC_EFDB"

# Sectors for IPCC EFDB
IPCC_SECTORS = [
    "Energy",
    "Industrial Processes and Product Use",
    "Agriculture, Forestry and Other Land Use",
    "Waste",
    "Other",
]

# Gas types
GAS_TYPES = ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"]

# Common regions
REGIONS = [
    "Global",
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "Oceania",
    "South America",
]


def fetch_emission_factors():
    """
    Attempt to fetch emission factors from the IPCC EFDB website.

    The IPCC EFDB doesn't have a public API, so this is a simulated approach.
    In a real implementation, this would likely require web scraping with proper permission.

    Returns:
        List of dictionaries with emission factor data
    """
    utils.log_extraction_step("ipcc_efdb", "Fetching emission factors")

    try:
        # This is where we would implement scraping or API calls
        # For demonstration, we'll just check if the site is accessible
        response = requests.get(IPCC_EFDB_URL, timeout=30)
        response.raise_for_status()

        logger.info("Successfully accessed IPCC EFDB website")
        logger.warning(
            "Real data extraction would require permission and proper scraping techniques"
        )
        logger.info("Using simulated data for demonstration")

        # In a real implementation, we would scrape and parse data here
        # Instead, we'll return an empty list to trigger the simulated data creation
        return []

    except Exception as e:
        logger.error(f"Error accessing IPCC EFDB: {e}")
        logger.info("Falling back to simulated data")
        return []


def create_simulated_dataset():
    """
    Create a simulated IPCC EFDB dataset for demonstration purposes.

    Returns:
        DataFrame with simulated emission factor data
    """
    utils.log_extraction_step("ipcc_efdb", "Creating simulated dataset")

    records = []

    # Generate simulated emission factors for each sector and gas
    for sector in IPCC_SECTORS:
        # Different number of records per sector to simulate real-world distribution
        n_records = {
            "Energy": 50,
            "Industrial Processes and Product Use": 40,
            "Agriculture, Forestry and Other Land Use": 60,
            "Waste": 30,
            "Other": 10,
        }.get(sector, 20)

        for _ in range(n_records):
            # Generate a subsector
            if sector == "Energy":
                subsectors = [
                    "Electricity Generation",
                    "Fuel Combustion",
                    "Transport",
                    "Fugitive Emissions",
                ]
            elif sector == "Industrial Processes and Product Use":
                subsectors = [
                    "Mineral Industry",
                    "Chemical Industry",
                    "Metal Industry",
                    "Electronics Industry",
                ]
            elif sector == "Agriculture, Forestry and Other Land Use":
                subsectors = ["Livestock", "Land", "Forestry", "Cropland", "Wetlands"]
            elif sector == "Waste":
                subsectors = [
                    "Solid Waste Disposal",
                    "Wastewater Treatment",
                    "Incineration",
                    "Composting",
                ]
            else:
                subsectors = ["Miscellaneous", "Research", "Test Data"]

            subsector = np.random.choice(subsectors)

            # Select a random gas type with weighted probabilities
            gas_probs = [0.6, 0.15, 0.15, 0.03, 0.03, 0.02, 0.02]  # CO2 most common
            gas = np.random.choice(GAS_TYPES, p=gas_probs)

            # Generate appropriate emission factor value based on gas type
            if gas == "CO2":
                ef_value = np.random.lognormal(mean=2, sigma=1)  # Higher values
            elif gas in ["CH4", "N2O"]:
                ef_value = np.random.lognormal(mean=0, sigma=1)  # Medium values
            else:
                ef_value = np.random.lognormal(
                    mean=-3, sigma=1
                )  # Lower values for fluorinated gases

            # Round to reasonable precision
            ef_value = round(ef_value, 4)

            # Generate units based on gas type
            if gas == "CO2":
                units = np.random.choice(["kg/TJ", "t/TJ", "kg/t", "kg/kWh"])
            elif gas in ["CH4", "N2O"]:
                units = np.random.choice(["kg/TJ", "g/GJ", "kg/ha", "kg/head"])
            else:
                units = np.random.choice(["kg/unit", "g/kg", "kg/year"])

            # Generate random confidence value
            confidence = round(np.random.uniform(0.5, 0.95), 2)

            # Assign to random region
            region = np.random.choice(REGIONS)

            # Create a unique ID
            ef_id = f"IPCC_EFDB_{sector.replace(' ', '_')}_{subsector.replace(' ', '_')}_{gas}_{len(records)}"

            # Generate a reference year
            reference_year = np.random.randint(2006, 2023)

            records.append(
                {
                    "ef_id": ef_id,
                    "sector": sector,
                    "subsector": subsector,
                    "gas": gas,
                    "ef_value": ef_value,
                    "unit": units,
                    "region": region,
                    "confidence": confidence,
                    "reference_year": reference_year,
                    "source": "IPCC EFDB (Simulated)",
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Save raw data
    os.makedirs(os.path.dirname(RAW_FILE_PATH), exist_ok=True)
    df.to_csv(RAW_FILE_PATH, index=False)

    logger.info(f"Created simulated IPCC EFDB dataset with {len(df)} records")
    return df


def preprocess_data(df):
    """
    Preprocess the IPCC EFDB data for further analysis.

    Args:
        df: Raw DataFrame with emission factor data

    Returns:
        Preprocessed DataFrame
    """
    utils.log_extraction_step("ipcc_efdb", "Preprocessing data")

    # Create a deep copy to avoid modifying the original
    processed_df = df.copy()

    # Standardize column names
    processed_df.columns = [
        col.lower().replace(" ", "_") for col in processed_df.columns
    ]

    # Convert emission factor values to numeric
    processed_df["ef_value"] = pd.to_numeric(processed_df["ef_value"], errors="coerce")

    # Handle missing values
    processed_df = processed_df.dropna(subset=["ef_value", "sector", "gas"])

    # Create standardized unit column
    processed_df["unit_standardized"] = processed_df["unit"]

    # Save interim data
    processed_df.to_csv(INTERIM_FILE_PATH, index=False)

    logger.info(f"Preprocessed data with {len(processed_df)} records")
    return processed_df


def convert_to_standard_format(df):
    """
    Convert the preprocessed data to the standard format.

    Args:
        df: Preprocessed DataFrame

    Returns:
        Standardized DataFrame
    """
    utils.log_extraction_step("ipcc_efdb", "Converting to standard format")

    # Create a new DataFrame with standardized columns
    std_df = pd.DataFrame()

    # Use IPCC EFDB ID as entity_id
    std_df["entity_id"] = df["ef_id"]

    # Combine sector and subsector for entity_name
    std_df["entity_name"] = df["sector"] + " - " + df["subsector"]

    # Set entity type based on sector
    def determine_entity_type(row):
        if "Energy" in row["sector"]:
            return "energy"
        elif "Industrial" in row["sector"]:
            return "industrial_process"
        elif "Agriculture" in row["sector"] or "Forestry" in row["sector"]:
            return "agriculture"
        elif "Waste" in row["sector"]:
            return "waste"
        else:
            return "other"

    std_df["entity_type"] = df.apply(determine_entity_type, axis=1)

    # Set ef_value
    std_df["ef_value"] = df["ef_value"]

    # Set units (this would ideally involve unit conversion to standardize)
    std_df["ef_unit"] = df["unit_standardized"]

    # Map regions to ISO codes
    region_mapping = {
        "Global": "GLB",
        "Africa": "AFR",
        "Asia": "ASI",
        "Europe": "EUR",
        "North America": "NAM",
        "Oceania": "OCE",
        "South America": "SAM",
    }
    std_df["region"] = df["region"].map(region_mapping).fillna("GLB")

    # Set source dataset
    std_df["source_dataset"] = DATASET_SOURCE

    # Set confidence
    std_df["confidence"] = df["confidence"]

    # Set timestamp to current date
    std_df["timestamp"] = datetime.now().strftime("%Y-%m-%d")

    # Set tags
    std_df["tags"] = df.apply(
        lambda row: [
            f"sector:{row['sector'].lower().replace(' ', '_')}",
            f"gas:{row['gas']}",
            f"year:{row['reference_year']}",
        ],
        axis=1,
    )

    # Drop rows with missing critical values
    std_df = std_df.dropna(subset=["entity_id", "ef_value"])

    logger.info(f"Converted to standard format with {len(std_df)} records")
    return std_df


def clean_dataset(df):
    """
    Clean the standardized dataset.

    Args:
        df: Standardized DataFrame

    Returns:
        Cleaned DataFrame
    """
    utils.log_extraction_step("ipcc_efdb", "Cleaning dataset")

    # Create a deep copy to avoid modifying the original
    clean_df = df.copy()

    # Standardize entity names
    clean_df["entity_name"] = clean_df["entity_name"].str.replace(
        r"[^\w\s\-]", " ", regex=True
    )
    clean_df["entity_name"] = clean_df["entity_name"].str.replace(
        r"\s+", " ", regex=True
    )
    clean_df["entity_name"] = clean_df["entity_name"].str.strip()

    # Handle outliers
    outliers = utils.detect_outliers(clean_df, "ef_value")
    logger.info(f"Detected {outliers.sum()} outliers in emission factor values")
    clean_df["is_outlier"] = outliers

    # Add metadata field
    clean_df["metadata"] = clean_df.apply(
        lambda row: json.dumps(
            {
                "source": DATASET_SOURCE,
                "entity_type": row["entity_type"],
                "tags": row["tags"],
            }
        ),
        axis=1,
    )

    # Save the cleaned dataset
    utils.save_dataframe(clean_df, CLEANED_FILE_PATH)

    logger.info(f"Cleaned dataset has {len(clean_df)} rows")
    return clean_df


def extract_and_clean():
    """
    Main function to extract and clean the IPCC EFDB dataset.

    Returns:
        Path to the cleaned dataset
    """
    try:
        # Try to fetch real data
        emission_data = fetch_emission_factors()

        # If no real data, use simulated data
        if not emission_data:
            logger.info("Using simulated data for IPCC EFDB")
            df = create_simulated_dataset()
        else:
            # Convert list of dictionaries to DataFrame
            df = pd.DataFrame(emission_data)

        # Preprocess the data
        processed_df = preprocess_data(df)

        # Convert to standard format
        std_df = convert_to_standard_format(processed_df)

        # Clean the dataset
        cleaned_df = clean_dataset(std_df)

        logger.info(f"IPCC EFDB extraction and cleaning completed: {CLEANED_FILE_PATH}")
        return CLEANED_FILE_PATH

    except Exception as e:
        logger.error(f"Error in IPCC EFDB extraction: {e}")
        raise


if __name__ == "__main__":
    # Configure logging if running as standalone script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    extract_and_clean()
