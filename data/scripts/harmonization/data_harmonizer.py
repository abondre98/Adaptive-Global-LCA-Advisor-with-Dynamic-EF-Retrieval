"""
Data harmonization module for standardizing and merging datasets.
"""

import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

logger = logging.getLogger(__name__)

# Constants
HARMONIZED_FILE_PATH = "data/processed/harmonized_ef_dataset.csv"
CROSSWALK_FILE_PATH = "data/documentation/entity_crosswalk.csv"
REGION_MAPPING_FILE_PATH = "data/documentation/region_mapping.csv"
DATASET_METADATA_FILE_PATH = "data/documentation/dataset_metadata.json"


def load_cleaned_datasets(datasets):
    """
    Load cleaned datasets based on the list of datasets to process.

    Args:
        datasets: List of dataset names

    Returns:
        Dictionary of dataframes by dataset name
    """
    utils.log_extraction_step("harmonization", "Loading cleaned datasets")

    dataset_files = {
        "agribalyse": "data/processed/agribalyse_3.1_clean.csv",
        "useeio": "data/processed/useeio_v2.1_clean.csv",
        "exiobase": "data/processed/exiobase_3.8_clean.csv",
        "climate_trace": "data/processed/climate_trace_clean.csv",
        "ipcc": "data/processed/ipcc_ar6_multipliers.csv",
    }

    loaded_datasets = {}

    for dataset in datasets:
        if dataset in dataset_files:
            file_path = dataset_files[dataset]
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    loaded_datasets[dataset] = df
                    logger.info(f"Loaded {dataset} dataset with {len(df)} rows")
                except Exception as e:
                    logger.error(f"Error loading {dataset} dataset: {e}")
            else:
                logger.warning(f"Dataset file not found: {file_path}")

    if not loaded_datasets:
        raise ValueError("No datasets could be loaded")

    return loaded_datasets


def create_entity_crosswalk(datasets):
    """
    Create a crosswalk table for entity IDs across datasets.

    Args:
        datasets: Dictionary of dataframes by dataset name

    Returns:
        Dataframe with entity ID mappings
    """
    utils.log_extraction_step("harmonization", "Creating entity crosswalk")

    # Extract entity names and IDs from each dataset
    entity_data = []

    for dataset_name, df in datasets.items():
        if "entity_id" in df.columns and "entity_name" in df.columns:
            dataset_entities = df[["entity_id", "entity_name"]].copy()
            dataset_entities["source_dataset"] = dataset_name
            entity_data.append(dataset_entities)

    if not entity_data:
        logger.warning("No entity data found for crosswalk")
        return pd.DataFrame()

    # Combine all entity data
    all_entities = pd.concat(entity_data, ignore_index=True)

    # Generate a canonical ID for each unique entity name
    entity_names = all_entities[["entity_name"]].drop_duplicates()
    entity_names["canonical_id"] = "ENT_" + entity_names.index.astype(str).str.zfill(6)

    # Merge canonical IDs with entity data
    crosswalk = pd.merge(all_entities, entity_names, on="entity_name", how="left")

    # Save crosswalk
    os.makedirs(os.path.dirname(CROSSWALK_FILE_PATH), exist_ok=True)
    crosswalk.to_csv(CROSSWALK_FILE_PATH, index=False, encoding="utf-8")

    logger.info(f"Created entity crosswalk with {len(crosswalk)} mappings")
    return crosswalk


def create_region_mapping():
    """
    Create a standardized region mapping table.

    Returns:
        Dataframe with region mappings
    """
    utils.log_extraction_step("harmonization", "Creating region mapping")

    # Create a dictionary of region mappings
    region_data = {
        "region_code": [],
        "region_name": [],
        "continent": [],
        "economic_group": [],
    }

    # Add country mappings
    country_mappings = {
        "FR": ("France", "Europe", "EU"),
        "DE": ("Germany", "Europe", "EU"),
        "IT": ("Italy", "Europe", "EU"),
        "ES": ("Spain", "Europe", "EU"),
        "GB": ("United Kingdom", "Europe", "Non-EU"),
        "US": ("United States", "North America", "NAFTA"),
        "CA": ("Canada", "North America", "NAFTA"),
        "MX": ("Mexico", "North America", "NAFTA"),
        "CN": ("China", "Asia", "BRICS"),
        "IN": ("India", "Asia", "BRICS"),
        "BR": ("Brazil", "South America", "BRICS"),
        "RU": ("Russia", "Europe/Asia", "BRICS"),
        "ZA": ("South Africa", "Africa", "BRICS"),
        "JP": ("Japan", "Asia", "G7"),
        "AU": ("Australia", "Oceania", "G20"),
        "ID": ("Indonesia", "Asia", "G20"),
        "TR": ("Turkey", "Europe/Asia", "G20"),
        "WA": ("Rest of Asia and Pacific", "Asia/Oceania", "ROW"),
        "WF": ("Rest of Africa", "Africa", "ROW"),
        "WL": ("Rest of America", "Americas", "ROW"),
        "WM": ("Rest of Middle East", "Asia", "ROW"),
        "WE": ("Rest of Europe", "Europe", "ROW"),
    }

    for code, (name, continent, econ_group) in country_mappings.items():
        region_data["region_code"].append(code)
        region_data["region_name"].append(name)
        region_data["continent"].append(continent)
        region_data["economic_group"].append(econ_group)

    # Create dataframe
    region_df = pd.DataFrame(region_data)

    # Save region mapping
    os.makedirs(os.path.dirname(REGION_MAPPING_FILE_PATH), exist_ok=True)
    region_df.to_csv(REGION_MAPPING_FILE_PATH, index=False, encoding="utf-8")

    logger.info(f"Created region mapping with {len(region_df)} regions")
    return region_df


def standardize_units(datasets):
    """
    Standardize units across datasets.

    Args:
        datasets: Dictionary of dataframes by dataset name

    Returns:
        Dictionary of standardized dataframes
    """
    utils.log_extraction_step("harmonization", "Standardizing units")

    standardized = {}

    for dataset_name, df in datasets.items():
        if "ef_value" in df.columns and "ef_unit" in df.columns:
            # Create a copy to avoid modifying the original
            std_df = df.copy()

            # Standardize to kg CO2e/kg where possible
            for i, row in std_df.iterrows():
                if pd.notna(row["ef_unit"]) and row["ef_unit"] != "kg CO2e/kg":
                    if row["ef_unit"] == "g CO2e/kg":
                        std_df.loc[i, "ef_value"] = row["ef_value"] * 0.001
                        std_df.loc[i, "ef_unit"] = "kg CO2e/kg"
                    elif row["ef_unit"] == "ton CO2e/kg":
                        std_df.loc[i, "ef_value"] = row["ef_value"] * 1000
                        std_df.loc[i, "ef_unit"] = "kg CO2e/kg"
                    elif row["ef_unit"] == "kg CO2e/ton":
                        std_df.loc[i, "ef_value"] = row["ef_value"] * 0.001
                        std_df.loc[i, "ef_unit"] = "kg CO2e/kg"
                    # Add more unit conversions as needed

            standardized[dataset_name] = std_df
            logger.info(f"Standardized units for {dataset_name} dataset")
        else:
            standardized[dataset_name] = df
            logger.warning(f"Could not standardize units for {dataset_name} dataset")

    return standardized


def apply_multipliers(datasets, multipliers_df):
    """
    Apply IPCC regional multipliers to emission factors.

    Args:
        datasets: Dictionary of standardized dataframes
        multipliers_df: Dataframe with IPCC multipliers

    Returns:
        Dictionary of dataframes with multipliers applied
    """
    utils.log_extraction_step("harmonization", "Applying multipliers")

    # Skip if no multipliers dataset
    if multipliers_df is None or multipliers_df.empty:
        logger.warning(
            "No multipliers dataset available, skipping multiplier application"
        )
        return datasets

    adjusted = {}

    # Create a dictionary for quick multiplier lookup
    multiplier_dict = {}
    if "entity_id" in multipliers_df.columns and "ef_value" in multipliers_df.columns:
        for _, row in multipliers_df.iterrows():
            if pd.notna(row["entity_id"]) and pd.notna(row["ef_value"]):
                multiplier_dict[row["entity_id"]] = row["ef_value"]

    for dataset_name, df in datasets.items():
        # Skip the multipliers dataset itself
        if dataset_name == "ipcc":
            adjusted[dataset_name] = df
            continue

        # Create a copy to avoid modifying the original
        adj_df = df.copy()

        # Add column for adjusted values
        adj_df["ef_value_adjusted"] = adj_df["ef_value"]

        # Apply multipliers where applicable
        if "region" in adj_df.columns and "entity_type" in adj_df.columns:
            for i, row in adj_df.iterrows():
                region = row["region"]
                entity_type = row["entity_type"]

                # Look for matching multiplier
                # First try exact match
                multiplier_key = f"{region}_{entity_type}"
                if multiplier_key in multiplier_dict:
                    adj_df.loc[i, "ef_value_adjusted"] = (
                        row["ef_value"] * multiplier_dict[multiplier_key]
                    )
                else:
                    # Try wildcard match by sector
                    for key, multiplier in multiplier_dict.items():
                        if key.startswith(f"{region}_") and entity_type in key:
                            adj_df.loc[i, "ef_value_adjusted"] = (
                                row["ef_value"] * multiplier
                            )
                            break

        # Use adjusted values as the main ef_value
        adj_df["ef_value_original"] = adj_df["ef_value"]
        adj_df["ef_value"] = adj_df["ef_value_adjusted"]
        adj_df = adj_df.drop("ef_value_adjusted", axis=1)

        adjusted[dataset_name] = adj_df
        logger.info(f"Applied multipliers to {dataset_name} dataset")

    return adjusted


def merge_datasets(datasets):
    """
    Merge all datasets into a unified dataset.

    Args:
        datasets: Dictionary of dataframes with multipliers applied

    Returns:
        Merged dataframe
    """
    utils.log_extraction_step("harmonization", "Merging datasets")

    all_dfs = []

    # Ensure all datasets have the required columns
    required_columns = [
        "entity_id",
        "entity_name",
        "entity_type",
        "ef_value",
        "ef_unit",
        "region",
        "source_dataset",
        "confidence",
        "timestamp",
        "tags",
    ]

    for dataset_name, df in datasets.items():
        # Create a copy with only required columns
        std_df = pd.DataFrame()

        for col in required_columns:
            if col in df.columns:
                std_df[col] = df[col]
            else:
                if col == "tags":
                    std_df[col] = [[]]  # Empty list for missing tags
                else:
                    std_df[col] = None

        all_dfs.append(std_df)

    # Concatenate all dataframes
    if not all_dfs:
        raise ValueError("No datasets to merge")

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Generate a harmonized ID
    merged_df["harmonized_id"] = "EF_" + merged_df.index.astype(str).str.zfill(8)

    # Save harmonized dataset
    utils.save_dataframe(merged_df, HARMONIZED_FILE_PATH)

    logger.info(
        f"Merged {len(all_dfs)} datasets into harmonized dataset with {len(merged_df)} rows"
    )
    return merged_df


def create_dataset_metadata(datasets):
    """
    Create metadata about the datasets.

    Args:
        datasets: Dictionary of dataframes by dataset name

    Returns:
        Dictionary with dataset metadata
    """
    utils.log_extraction_step("harmonization", "Creating dataset metadata")

    metadata = {
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {},
    }

    for dataset_name, df in datasets.items():
        dataset_meta = {
            "row_count": len(df),
            "columns": list(df.columns),
            "regions": df["region"].nunique() if "region" in df.columns else 0,
            "entity_types": (
                df["entity_type"].value_counts().to_dict()
                if "entity_type" in df.columns
                else {}
            ),
            "source": dataset_name,
        }
        metadata["datasets"][dataset_name] = dataset_meta

    # Save metadata
    os.makedirs(os.path.dirname(DATASET_METADATA_FILE_PATH), exist_ok=True)
    with open(DATASET_METADATA_FILE_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created metadata for {len(datasets)} datasets")
    return metadata


def harmonize_datasets(datasets_to_process):
    """
    Main function to harmonize all datasets.

    Args:
        datasets_to_process: List of dataset names to process

    Returns:
        Path to the harmonized dataset
    """
    try:
        # Load cleaned datasets
        datasets = load_cleaned_datasets(datasets_to_process)

        # Create entity crosswalk
        crosswalk = create_entity_crosswalk(datasets)

        # Create region mapping
        region_mapping = create_region_mapping()

        # Standardize units
        standardized = standardize_units(datasets)

        # Extract multipliers dataset if available
        multipliers_df = standardized.get("ipcc", None)

        # Apply multipliers
        adjusted = apply_multipliers(standardized, multipliers_df)

        # Merge datasets
        harmonized = merge_datasets(adjusted)

        # Create dataset metadata
        metadata = create_dataset_metadata(datasets)

        logger.info(f"Data harmonization completed: {HARMONIZED_FILE_PATH}")
        return HARMONIZED_FILE_PATH
    except Exception as e:
        logger.error(f"Error in data harmonization: {e}")
        raise


if __name__ == "__main__":
    # Configure logging if running as standalone script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Process all datasets
    harmonize_datasets(["agribalyse", "useeio", "exiobase", "climate_trace", "ipcc"])
