"""
Harmonization module for combining and standardizing cleaned datasets.
"""

import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

logger = logging.getLogger(__name__)

# Constants
INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
CROSSWALK_FILE = os.path.join(INTERIM_DIR, "entity_crosswalk.csv")


def load_datasets(dataset_paths):
    """
    Load all cleaned datasets.

    Args:
        dataset_paths: Dictionary mapping dataset names to file paths

    Returns:
        Dictionary mapping dataset names to dataframes
    """
    utils.log_harmonization_step("Loading datasets")

    datasets = {}

    for dataset_name, file_path in dataset_paths.items():
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                datasets[dataset_name] = df
                logger.info(f"Loaded {dataset_name} dataset with {len(df)} rows")
            else:
                logger.warning(f"Dataset file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")

    return datasets


def create_entity_crosswalk(datasets):
    """
    Create a crosswalk between similar entities across datasets.

    Args:
        datasets: Dictionary mapping dataset names to dataframes

    Returns:
        Dataframe with entity crosswalk
    """
    utils.log_harmonization_step("Creating entity crosswalk")

    # Extract entities from all datasets
    all_entities = []

    for dataset_name, df in datasets.items():
        if "entity_id" in df.columns and "entity_name" in df.columns:
            entities = df[["entity_id", "entity_name"]].copy()
            entities["source_dataset"] = dataset_name
            all_entities.append(entities)

    if not all_entities:
        logger.warning("No entities found in datasets")
        return pd.DataFrame()

    # Combine all entities
    entity_df = pd.concat(all_entities, ignore_index=True)

    # Create a simplified name for matching
    entity_df["simplified_name"] = (
        entity_df["entity_name"]
        .str.lower()
        .str.replace(r"[^a-z0-9]", " ", regex=True)
        .str.strip()
    )

    # Group by simplified name to find matches
    grouped = entity_df.groupby("simplified_name")

    # Create crosswalk
    crosswalk_rows = []

    for name, group in grouped:
        if len(group) > 1:
            # Multiple datasets have this entity
            primary_id = group.iloc[0]["entity_id"]
            primary_name = group.iloc[0]["entity_name"]
            primary_source = group.iloc[0]["source_dataset"]

            for _, row in group.iloc[1:].iterrows():
                crosswalk_rows.append(
                    {
                        "primary_id": primary_id,
                        "primary_name": primary_name,
                        "primary_source": primary_source,
                        "secondary_id": row["entity_id"],
                        "secondary_name": row["entity_name"],
                        "secondary_source": row["source_dataset"],
                        "match_confidence": 0.9,  # High confidence for exact simplified name matches
                    }
                )

    # Create crosswalk dataframe
    if crosswalk_rows:
        crosswalk_df = pd.DataFrame(crosswalk_rows)

        # Save crosswalk
        os.makedirs(os.path.dirname(CROSSWALK_FILE), exist_ok=True)
        crosswalk_df.to_csv(CROSSWALK_FILE, index=False)

        logger.info(f"Created entity crosswalk with {len(crosswalk_df)} matches")
        return crosswalk_df
    else:
        logger.warning("No entity matches found across datasets")
        return pd.DataFrame()


def standardize_units(datasets):
    """
    Standardize units across datasets.

    Args:
        datasets: Dictionary mapping dataset names to dataframes

    Returns:
        Dictionary with standardized dataframes
    """
    utils.log_harmonization_step("Standardizing units")

    # Define unit conversions to kg CO2e
    unit_conversions = {
        "kg CO2e": 1.0,
        "g CO2e": 0.001,
        "t CO2e": 1000.0,
        "Mt CO2e": 1000000.0,
        "Gt CO2e": 1000000000.0,
    }

    standardized_datasets = {}

    for dataset_name, df in datasets.items():
        if "ef_unit" in df.columns and "ef_value" in df.columns:
            # Create a copy to avoid modifying the original
            std_df = df.copy()

            # Fix numeric values in ef_unit field (likely CSV parsing errors)
            numeric_units = std_df["ef_unit"].str.match(r"^[\d\.]+$", na=False)
            if numeric_units.any():
                logger.warning(
                    f"Found {numeric_units.sum()} numeric values in ef_unit column in {dataset_name}"
                )
                # Set to kg CO2e as a reasonable default
                std_df.loc[numeric_units, "ef_unit"] = "kg CO2e"

            # Apply unit conversions
            for unit, factor in unit_conversions.items():
                mask = std_df["ef_unit"].str.contains(unit, case=False, na=False)
                if mask.any():
                    logger.info(
                        f"Converting {mask.sum()} values from {unit} to kg CO2e in {dataset_name}"
                    )
                    std_df.loc[mask, "ef_value"] = std_df.loc[mask, "ef_value"] * factor
                    std_df.loc[mask, "ef_unit"] = "kg CO2e"

            # Handle special case for ratios/multipliers
            mask = std_df["ef_unit"].str.contains("ratio", case=False, na=False)
            if mask.any():
                logger.info(
                    f"Keeping {mask.sum()} ratio values as-is in {dataset_name}"
                )

            standardized_datasets[dataset_name] = std_df
        else:
            logger.warning(
                f"Dataset {dataset_name} missing required columns for unit standardization"
            )
            standardized_datasets[dataset_name] = df

    return standardized_datasets


def apply_regional_multipliers(datasets):
    """
    Apply regional multipliers to emission factors.

    Args:
        datasets: Dictionary mapping dataset names to dataframes

    Returns:
        Dictionary with adjusted dataframes
    """
    utils.log_harmonization_step("Applying regional multipliers")

    # Check if we have IPCC AR6 multipliers
    if "ipcc_ar6" in datasets:
        multipliers_df = datasets["ipcc_ar6"]

        # Extract multipliers as a dictionary for faster lookup
        multipliers = {}

        for _, row in multipliers_df.iterrows():
            if (
                "region" in row
                and "entity_type" in row
                and row["entity_type"] == "multiplier"
            ):
                region = row.get("region")

                # Extract tags properly, handle string representation of lists
                tags = row.get("tags", [])
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags.replace("'", '"'))
                    except:
                        tags = []

                sector_tags = [
                    tag.replace("sector:", "")
                    for tag in tags
                    if isinstance(tag, str) and tag.startswith("sector:")
                ]

                if region and sector_tags:
                    for sector in sector_tags:
                        key = (region, sector)
                        multipliers[key] = row.get("ef_value", 1.0)

                # Add global fallback multipliers
                if region != "Global" and sector_tags:
                    for sector in sector_tags:
                        global_key = ("Global", sector)
                        if global_key not in multipliers:
                            multipliers[global_key] = 1.0

        # Apply multipliers to other datasets
        adjusted_datasets = {}

        for dataset_name, df in datasets.items():
            if (
                dataset_name != "ipcc_ar6"
                and "region" in df.columns
                and "ef_value" in df.columns
            ):
                # Create a copy to avoid modifying the original
                adj_df = df.copy()

                # Add a column to track if multiplier was applied
                adj_df["multiplier_applied"] = False

                # Apply multipliers based on region and sector tags
                for i, row in adj_df.iterrows():
                    region = row.get("region")

                    # Standardize region codes for better matching
                    if region == "US":
                        region = "USA"

                    # Extract tags properly, handle string representation of lists
                    tags = row.get("tags", [])
                    if isinstance(tags, str):
                        try:
                            tags = json.loads(tags.replace("'", '"'))
                        except:
                            tags = []

                    # Get entity type as fallback if no sector tags
                    entity_type = row.get("entity_type", "")

                    # Try to find sector tags
                    sector_tags = [
                        tag.replace("sector:", "")
                        for tag in tags
                        if isinstance(tag, str) and tag.startswith("sector:")
                    ]

                    # If no sector tags, use entity_type as fallback
                    if not sector_tags and entity_type:
                        sector_tags = [entity_type]

                    # Try to find a matching multiplier
                    multiplier = 1.0
                    for sector in sector_tags:
                        key = (region, sector)
                        if key in multipliers:
                            multiplier = multipliers[key]
                            adj_df.at[i, "multiplier_applied"] = True
                            break

                    # If no specific multiplier found, try global multiplier
                    if not adj_df.at[i, "multiplier_applied"] and sector_tags:
                        for sector in sector_tags:
                            key = ("Global", sector)
                            if key in multipliers:
                                multiplier = multipliers[key]
                                adj_df.at[i, "multiplier_applied"] = True
                                break

                    # Apply the multiplier
                    if adj_df.at[i, "multiplier_applied"]:
                        adj_df.at[i, "ef_value"] = adj_df.at[i, "ef_value"] * multiplier

                logger.info(
                    f"Applied regional multipliers to {adj_df['multiplier_applied'].sum()} records in {dataset_name}"
                )
                adjusted_datasets[dataset_name] = adj_df
            else:
                adjusted_datasets[dataset_name] = df

        return adjusted_datasets
    else:
        logger.warning("No IPCC AR6 multipliers found, skipping regional adjustment")
        return datasets


def merge_datasets(datasets):
    """
    Merge all datasets into a single harmonized dataset.

    Args:
        datasets: Dictionary mapping dataset names to dataframes

    Returns:
        Merged dataframe
    """
    utils.log_harmonization_step("Merging datasets")

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
    ]

    valid_dfs = []

    for dataset_name, df in datasets.items():
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.warning(
                f"Dataset {dataset_name} missing columns: {', '.join(missing_columns)}"
            )
            continue

        # Add to valid dataframes
        valid_dfs.append(df)

    if not valid_dfs:
        raise ValueError("No valid datasets to merge")

    # Merge all valid dataframes
    merged_df = pd.concat(valid_dfs, ignore_index=True)

    # Add a unique identifier
    merged_df["id"] = [f"EF{i:08d}" for i in range(1, len(merged_df) + 1)]

    # Ensure all string columns are actually strings
    string_columns = [
        "entity_id",
        "entity_name",
        "entity_type",
        "ef_unit",
        "region",
        "source_dataset",
        "id",
    ]

    for col in string_columns:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype(str)

    # Ensure numeric columns are numeric
    numeric_columns = ["ef_value", "confidence"]

    for col in numeric_columns:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    # Handle tags column
    if "tags" in merged_df.columns:
        # Ensure tags are in list format
        def parse_tags(x):
            if isinstance(x, list):
                return x
            elif isinstance(x, str):
                try:
                    # Try to parse as JSON
                    parsed = json.loads(x)
                    if isinstance(parsed, list):
                        return parsed
                    else:
                        return [str(parsed)]
                except json.JSONDecodeError:
                    # If not valid JSON, treat as a single tag
                    return [x]
            else:
                return []

        merged_df["tags"] = merged_df["tags"].apply(parse_tags)

    logger.info(
        f"Merged {len(valid_dfs)} datasets into a single dataset with {len(merged_df)} records"
    )
    return merged_df


def generate_metadata(merged_df, dataset_paths):
    """
    Generate metadata for the harmonized dataset.

    Args:
        merged_df: Merged dataframe
        dataset_paths: Dictionary mapping dataset names to file paths

    Returns:
        Dictionary with metadata
    """
    utils.log_harmonization_step("Generating metadata")

    metadata = {
        "title": "Harmonized Global Emission Factor Dataset",
        "description": "A harmonized dataset of emission factors from multiple sources",
        "version": "1.0",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_datasets": list(dataset_paths.keys()),
        "record_count": len(merged_df),
        "columns": list(merged_df.columns),
        "regions": merged_df["region"].nunique(),
        "entity_types": merged_df["entity_type"].value_counts().to_dict(),
        "units": merged_df["ef_unit"].value_counts().to_dict(),
        "confidence_range": (
            [float(merged_df["confidence"].min()), float(merged_df["confidence"].max())]
            if not merged_df["confidence"].isna().all()
            else [0, 0]
        ),
        "processing_steps": [
            "Loading cleaned datasets",
            "Creating entity crosswalk",
            "Standardizing units",
            "Applying regional multipliers",
            "Merging datasets",
            "Generating metadata",
        ],
    }

    return metadata


def harmonize(dataset_paths, output_file):
    """
    Harmonize multiple datasets into a single dataset.

    Args:
        dataset_paths: Dictionary mapping dataset names to file paths
        output_file: Path to save the harmonized dataset

    Returns:
        Path to the harmonized dataset
    """
    try:
        # Load datasets
        datasets = load_datasets(dataset_paths)

        if not datasets:
            raise ValueError("No datasets loaded")

        # Create entity crosswalk
        crosswalk = create_entity_crosswalk(datasets)

        # Standardize units
        standardized_datasets = standardize_units(datasets)

        # Apply regional multipliers
        adjusted_datasets = apply_regional_multipliers(standardized_datasets)

        # Merge datasets
        merged_df = merge_datasets(adjusted_datasets)

        # Generate metadata
        metadata = generate_metadata(merged_df, dataset_paths)

        # Save metadata
        metadata_file = os.path.splitext(output_file)[0] + "_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save harmonized dataset
        utils.save_dataframe(merged_df, output_file)

        logger.info(f"Harmonized dataset saved to {output_file}")
        logger.info(f"Metadata saved to {metadata_file}")

        return output_file

    except Exception as e:
        logger.error(f"Error in harmonization process: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Configure logging if running as standalone script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example usage
    dataset_paths = {
        "agribalyse": "data/processed/agribalyse_3.1_clean.csv",
        "useeio": "data/processed/useeio_v2.1_clean.csv",
        "exiobase": "data/processed/exiobase_3.8_clean.csv",
        "climate_trace": "data/processed/climate_trace_clean.csv",
        "ipcc_ar6": "data/processed/ipcc_ar6_clean.csv",
        "openlca": "data/processed/openlca_clean.csv",
    }

    output_file = "data/processed/harmonized_global_ef_dataset.csv"

    harmonize(dataset_paths, output_file)
