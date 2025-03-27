"""
GREET Model extractor module.

This module downloads and processes emission factor data from the GREET (Greenhouse gases, 
Regulated Emissions, and Energy use in Transportation) Model developed by 
Argonne National Laboratory. The GREET Model is a widely used lifecycle analysis tool
for transportation fuels and vehicle technologies.
"""

import json
import logging
import os
import sys
import zipfile
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

logger = logging.getLogger(__name__)

# Constants
GREET_URL = "https://greet.anl.gov/"
RAW_DIR_PATH = "data/raw/greet"
INTERIM_FILE_PATH = "data/interim/greet_interim.csv"
CLEANED_FILE_PATH = "data/processed/greet_clean.csv"
DATASET_SOURCE = "GREET_Model"

# Fuel pathways for simulation
FUEL_PATHWAYS = [
    "Conventional Gasoline",
    "Reformulated Gasoline",
    "Low-Sulfur Diesel",
    "Liquefied Petroleum Gas (LPG)",
    "Compressed Natural Gas (CNG)",
    "Liquefied Natural Gas (LNG)",
    "Ethanol from Corn",
    "Ethanol from Sugarcane",
    "Ethanol from Cellulosic Biomass",
    "Biodiesel from Soybean",
    "Biodiesel from Algae",
    "Renewable Diesel",
    "Hydrogen from Natural Gas",
    "Hydrogen from Biomass",
    "Hydrogen from Electrolysis",
    "Electricity from Coal",
    "Electricity from Natural Gas",
    "Electricity from Nuclear",
    "Electricity from Biomass",
    "Electricity from Wind",
    "Electricity from Solar",
    "Electricity from Hydroelectric",
    "Electricity Grid Mix (US Average)",
]

# Vehicle technologies
VEHICLE_TECHNOLOGIES = [
    "Conventional Internal Combustion Engine (ICE)",
    "Hybrid Electric Vehicle (HEV)",
    "Plug-in Hybrid Electric Vehicle (PHEV)",
    "Battery Electric Vehicle (BEV)",
    "Fuel Cell Electric Vehicle (FCEV)",
    "Compressed Natural Gas Vehicle (CNGV)",
    "Liquefied Petroleum Gas Vehicle (LPGV)",
    "Flexible Fuel Vehicle (FFV)",
]

# Emission types
EMISSION_TYPES = [
    "Well-to-Pump",  # Upstream emissions
    "Pump-to-Wheel",  # Operational emissions
    "Well-to-Wheel",  # Total lifecycle emissions
]


def attempt_download():
    """
    Attempt to download GREET model data.

    In reality, the GREET model is distributed as an Excel/software tool, not as raw data.
    This function would need special handling for extraction.

    Returns:
        Path to downloaded file or None if download failed
    """
    utils.log_extraction_step("greet", "Attempting to download GREET data")

    try:
        # Check if GREET website is accessible
        response = requests.get(GREET_URL, timeout=30)
        response.raise_for_status()

        logger.info("Successfully accessed GREET website")
        logger.warning(
            "GREET model data is distributed as Excel/software tool, not as raw data"
        )
        logger.warning(
            "Real implementation would require special handling to extract data from GREET models"
        )
        logger.info("Using simulated data for demonstration")

        # In a real implementation, we would download GREET Excel files or use the API
        # For now, return None to trigger simulated data creation
        return None

    except Exception as e:
        logger.error(f"Error accessing GREET website: {e}")
        logger.info("Falling back to simulated data")
        return None


def create_simulated_dataset():
    """
    Create a simulated GREET model dataset for demonstration purposes.

    Returns:
        DataFrame with simulated emission factor data
    """
    utils.log_extraction_step("greet", "Creating simulated dataset")

    records = []

    # Generate simulated emission factors for each fuel pathway and vehicle technology
    for fuel_pathway in FUEL_PATHWAYS:
        # Different baselines for different fuel types
        if "Gasoline" in fuel_pathway or "Diesel" in fuel_pathway:
            base_ghg = np.random.uniform(80, 100)  # g CO2e/MJ
        elif "Natural Gas" in fuel_pathway or "LPG" in fuel_pathway:
            base_ghg = np.random.uniform(60, 80)  # g CO2e/MJ
        elif "Ethanol" in fuel_pathway or "Biodiesel" in fuel_pathway:
            base_ghg = np.random.uniform(40, 70)  # g CO2e/MJ
        elif "Hydrogen" in fuel_pathway:
            base_ghg = np.random.uniform(
                30, 150
            )  # g CO2e/MJ (wide range depending on source)
        elif "Electricity" in fuel_pathway:
            if "Coal" in fuel_pathway:
                base_ghg = np.random.uniform(150, 200)  # g CO2e/MJ
            elif "Natural Gas" in fuel_pathway:
                base_ghg = np.random.uniform(80, 120)  # g CO2e/MJ
            elif (
                "Nuclear" in fuel_pathway
                or "Wind" in fuel_pathway
                or "Solar" in fuel_pathway
                or "Hydroelectric" in fuel_pathway
            ):
                base_ghg = np.random.uniform(2, 30)  # g CO2e/MJ
            else:
                base_ghg = np.random.uniform(50, 100)  # g CO2e/MJ
        else:
            base_ghg = np.random.uniform(50, 100)  # g CO2e/MJ

        # Generate well-to-pump (upstream) emissions
        wtp_ghg = base_ghg * np.random.uniform(0.2, 0.4)  # 20-40% of total

        # Generate pump-to-wheel (operational) emissions
        ptw_ghg = base_ghg * np.random.uniform(0.6, 0.8)  # 60-80% of total

        # Sanity check for total
        wtw_ghg = wtp_ghg + ptw_ghg

        # Add records for each emission type
        records.append(
            {
                "fuel_pathway": fuel_pathway,
                "emission_type": "Well-to-Pump",
                "vehicle_technology": "Not Applicable",  # Upstream independent of vehicle
                "ghg_emissions": round(wtp_ghg, 2),
                "unit": "g CO2e/MJ",
                "year": 2022,
                "region": "US",
                "confidence": round(np.random.uniform(0.7, 0.9), 2),
            }
        )

        # Add vehicle-specific operational emissions
        for vehicle_tech in VEHICLE_TECHNOLOGIES:
            # Skip irrelevant combinations
            if (
                "Electric" in vehicle_tech
                and "Electricity" not in fuel_pathway
                and "Hybrid" not in vehicle_tech
            ):
                continue
            if "Fuel Cell" in vehicle_tech and "Hydrogen" not in fuel_pathway:
                continue
            if "Natural Gas" in vehicle_tech and "Natural Gas" not in fuel_pathway:
                continue
            if "Liquefied Petroleum" in vehicle_tech and "LPG" not in fuel_pathway:
                continue

            # Adjust emissions based on vehicle technology
            if "Hybrid" in vehicle_tech:
                efficiency_factor = np.random.uniform(0.6, 0.8)  # 20-40% more efficient
            elif "Electric" in vehicle_tech and "Plug-in" not in vehicle_tech:
                efficiency_factor = 0.0  # Direct emissions are zero for BEV
            elif "Fuel Cell" in vehicle_tech:
                efficiency_factor = 0.0  # Direct emissions are zero for FCEV
            else:
                efficiency_factor = np.random.uniform(0.95, 1.05)  # ±5% variation

            adjusted_ptw = ptw_ghg * efficiency_factor
            adjusted_wtw = wtp_ghg + adjusted_ptw

            # Operational emissions
            records.append(
                {
                    "fuel_pathway": fuel_pathway,
                    "emission_type": "Pump-to-Wheel",
                    "vehicle_technology": vehicle_tech,
                    "ghg_emissions": round(adjusted_ptw, 2),
                    "unit": "g CO2e/MJ",
                    "year": 2022,
                    "region": "US",
                    "confidence": round(np.random.uniform(0.7, 0.9), 2),
                }
            )

            # Total lifecycle emissions
            records.append(
                {
                    "fuel_pathway": fuel_pathway,
                    "emission_type": "Well-to-Wheel",
                    "vehicle_technology": vehicle_tech,
                    "ghg_emissions": round(adjusted_wtw, 2),
                    "unit": "g CO2e/MJ",
                    "year": 2022,
                    "region": "US",
                    "confidence": round(np.random.uniform(0.7, 0.9), 2),
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Create directory if it doesn't exist
    os.makedirs(RAW_DIR_PATH, exist_ok=True)

    # Save raw data
    raw_file_path = os.path.join(RAW_DIR_PATH, "greet_simulated.csv")
    df.to_csv(raw_file_path, index=False)

    logger.info(f"Created simulated GREET dataset with {len(df)} records")
    return df


def preprocess_data(df):
    """
    Preprocess the GREET model data for further analysis.

    Args:
        df: Raw DataFrame with emission factor data

    Returns:
        Preprocessed DataFrame
    """
    utils.log_extraction_step("greet", "Preprocessing data")

    # Create a deep copy to avoid modifying the original
    processed_df = df.copy()

    # Standardize column names
    processed_df.columns = [
        col.lower().replace(" ", "_") for col in processed_df.columns
    ]

    # Convert GHG emissions to numeric
    processed_df["ghg_emissions"] = pd.to_numeric(
        processed_df["ghg_emissions"], errors="coerce"
    )

    # Handle missing values
    processed_df = processed_df.dropna(subset=["ghg_emissions", "fuel_pathway"])

    # Convert units if needed (in this case, we're using g CO2e/MJ consistently)
    # In a real implementation, this would handle various unit conversions

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
    utils.log_extraction_step("greet", "Converting to standard format")

    # Create a new DataFrame with standardized columns
    std_df = pd.DataFrame()

    # Create unique entity IDs
    std_df["entity_id"] = df.apply(
        lambda row: f"GREET_{row['fuel_pathway'].replace(' ', '_')}_{row['emission_type'].replace(' ', '_')}_{row['vehicle_technology'].replace(' ', '_')}",
        axis=1,
    )

    # Create entity name from fuel pathway, emission type, and vehicle technology
    std_df["entity_name"] = df.apply(
        lambda row: f"{row['fuel_pathway']} - {row['emission_type']} - {row['vehicle_technology']}",
        axis=1,
    )

    # Set entity type to 'fuel_pathway'
    std_df["entity_type"] = "fuel_pathway"

    # Set emission factor value
    std_df["ef_value"] = df["ghg_emissions"]

    # Set units
    std_df["ef_unit"] = df["unit"]

    # Set region to US (GREET is primarily US-focused)
    std_df["region"] = "US"

    # Set source dataset
    std_df["source_dataset"] = DATASET_SOURCE

    # Set confidence
    std_df["confidence"] = df["confidence"]

    # Set timestamp to current date
    std_df["timestamp"] = datetime.now().strftime("%Y-%m-%d")

    # Create tags
    std_df["tags"] = df.apply(
        lambda row: [
            f"fuel:{row['fuel_pathway'].lower().replace(' ', '_')}",
            f"emission_type:{row['emission_type'].lower().replace(' ', '_')}",
            f"vehicle:{row['vehicle_technology'].lower().replace(' ', '_')}",
            f"year:{row['year']}",
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
    utils.log_extraction_step("greet", "Cleaning dataset")

    # Create a deep copy to avoid modifying the original
    clean_df = df.copy()

    # Standardize entity names
    clean_df["entity_name"] = clean_df["entity_name"].str.replace(
        r"[^\w\s\-()]", " ", regex=True
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
    Main function to extract and clean the GREET model dataset.

    Returns:
        Path to the cleaned dataset
    """
    try:
        # Try to download real data
        download_path = attempt_download()

        # If download failed, use simulated data
        if not download_path:
            logger.info("Using simulated data for GREET model")
            df = create_simulated_dataset()
        else:
            # In a real implementation, this would extract data from GREET Excel files
            # For now, we'll just use simulated data
            logger.warning("Downloaded GREET files require specialized extraction")
            logger.info("Using simulated data for demonstration")
            df = create_simulated_dataset()

        # Preprocess the data
        processed_df = preprocess_data(df)

        # Convert to standard format
        std_df = convert_to_standard_format(processed_df)

        # Clean the dataset
        cleaned_df = clean_dataset(std_df)

        logger.info(
            f"GREET model extraction and cleaning completed: {CLEANED_FILE_PATH}"
        )
        return CLEANED_FILE_PATH

    except Exception as e:
        logger.error(f"Error in GREET model extraction: {e}")
        raise


if __name__ == "__main__":
    # Configure logging if running as standalone script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    extract_and_clean()
