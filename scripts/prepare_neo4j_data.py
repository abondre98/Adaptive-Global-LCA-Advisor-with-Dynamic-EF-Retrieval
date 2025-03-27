#!/usr/bin/env python3
"""
Neo4j Data Preparation Script
This script converts the harmonized emission factor dataset into CSV files optimized
for Neo4j import, creating separate files for nodes and relationships.
"""

import logging
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# File paths
INPUT_FILE = "data/processed/harmonized_global_ef_dataset.csv"
OUTPUT_DIR = "neo4j/import"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    """Main execution function."""
    logger.info(f"Starting Neo4j data preparation from {INPUT_FILE}")

    # Load the harmonized dataset
    try:
        df = pd.read_csv(INPUT_FILE)
        logger.info(f"Loaded {len(df)} records from harmonized dataset")
    except Exception as e:
        logger.error(f"Failed to load harmonized dataset: {e}")
        return

    # Extract unique regions
    regions_df = extract_regions(df)
    regions_df.to_csv(f"{OUTPUT_DIR}/regions.csv", index=False)
    logger.info(f"Created regions.csv with {len(regions_df)} records")

    # Extract unique entity types
    entity_types_df = extract_entity_types(df)
    entity_types_df.to_csv(f"{OUTPUT_DIR}/entity_types.csv", index=False)
    logger.info(f"Created entity_types.csv with {len(entity_types_df)} records")

    # Extract unique sources
    sources_df = extract_sources(df)
    sources_df.to_csv(f"{OUTPUT_DIR}/sources.csv", index=False)
    logger.info(f"Created sources.csv with {len(sources_df)} records")

    # Create emission factors nodes
    ef_df = create_emission_factors(df)
    ef_df.to_csv(f"{OUTPUT_DIR}/emission_factors.csv", index=False)
    logger.info(f"Created emission_factors.csv with {len(ef_df)} records")

    # Create relationships
    ef_to_region_df = create_ef_to_region_relationships(df, ef_df)
    ef_to_region_df.to_csv(f"{OUTPUT_DIR}/ef_to_region.csv", index=False)
    logger.info(f"Created ef_to_region.csv with {len(ef_to_region_df)} records")

    ef_to_entity_type_df = create_ef_to_entity_type_relationships(df, ef_df)
    ef_to_entity_type_df.to_csv(f"{OUTPUT_DIR}/ef_to_entity_type.csv", index=False)
    logger.info(
        f"Created ef_to_entity_type.csv with {len(ef_to_entity_type_df)} records"
    )

    ef_to_source_df = create_ef_to_source_relationships(df, ef_df)
    ef_to_source_df.to_csv(f"{OUTPUT_DIR}/ef_to_source.csv", index=False)
    logger.info(f"Created ef_to_source.csv with {len(ef_to_source_df)} records")

    # Create region hierarchy relationships
    region_hierarchy_df = create_region_hierarchy()
    region_hierarchy_df.to_csv(f"{OUTPUT_DIR}/region_hierarchy.csv", index=False)
    logger.info(f"Created region_hierarchy.csv with {len(region_hierarchy_df)} records")

    logger.info("Neo4j data preparation completed successfully")


def extract_regions(df):
    """Extract unique regions and create a DataFrame for Region nodes."""
    unique_regions = df["region"].unique()

    # Map regions to continents (simplified mapping)
    continent_map = {
        "USA": "North America",
        "FR": "Europe",
        "GLB": "Global",
        # Add more mappings as needed
    }

    regions_data = []
    for region_code in unique_regions:
        continent = continent_map.get(region_code, "Unknown")
        is_global = region_code == "GLB"

        # Get a descriptive name for the region
        name = get_region_name(region_code)

        regions_data.append(
            {
                "region_code": region_code,
                "name": name,
                "continent": continent,
                "is_global": is_global,
            }
        )

    return pd.DataFrame(regions_data)


def get_region_name(region_code):
    """Convert region code to full name."""
    region_names = {
        "USA": "United States",
        "FR": "France",
        "GLB": "Global",
        # Add more mappings as needed
    }
    return region_names.get(region_code, region_code)


def extract_entity_types(df):
    """Extract unique entity types and create a DataFrame for EntityType nodes."""
    unique_types = df["entity_type"].unique()

    entity_types_data = []
    for i, type_name in enumerate(unique_types):
        entity_types_data.append(
            {
                "type_id": f"type_{i+1}",
                "type_name": type_name,
                "description": f"Emission factors for {type_name}",
            }
        )

    return pd.DataFrame(entity_types_data)


def extract_sources(df):
    """Extract unique sources and create a DataFrame for Source nodes."""
    unique_sources = df["source_dataset"].unique()

    sources_data = []
    for i, source_name in enumerate(unique_sources):
        # Extract version from source name if available
        version = "1.0"  # Default version
        if "_" in source_name:
            parts = source_name.split("_")
            if len(parts) > 1 and parts[-1][0].isdigit():
                version = parts[-1]

        sources_data.append(
            {
                "source_id": f"source_{i+1}",
                "name": source_name,
                "version": version,
                "url": f"https://example.com/datasets/{source_name.lower().replace(' ', '_')}",
            }
        )

    return pd.DataFrame(sources_data)


def create_emission_factors(df):
    """Create DataFrame for EmissionFactor nodes."""
    emission_factors_data = []

    for _, row in df.iterrows():
        # Generate a unique ID for each emission factor
        ef_id = f"ef_{uuid.uuid4().hex[:8]}"

        emission_factors_data.append(
            {
                "ef_id": ef_id,
                "entity_id": row.get("entity_id", ""),
                "entity_name": row.get("entity_name", ""),
                "ef_value": row.get("ef_value", 0.0),
                "ef_unit": row.get("ef_unit", "kg CO2e"),
                "confidence": row.get("confidence", 0.5),
                "is_outlier": str(row.get("is_outlier", False)).lower() == "true",
                "multiplier_applied": str(row.get("multiplier_applied", False)).lower()
                == "true",
                "timestamp": row.get("timestamp", datetime.now().strftime("%Y-%m-%d")),
            }
        )

    return pd.DataFrame(emission_factors_data)


def create_ef_to_region_relationships(df, ef_df):
    """Create DataFrame for APPLIES_TO_REGION relationships."""
    relationships = []

    for i, (_, ef_row) in enumerate(ef_df.iterrows()):
        df_row = df.iloc[i]
        relationships.append(
            {
                "ef_id": ef_row["ef_id"],
                "region_code": df_row["region"],
                "confidence": ef_row["confidence"],
            }
        )

    return pd.DataFrame(relationships)


def create_ef_to_entity_type_relationships(df, ef_df):
    """Create DataFrame for HAS_ENTITY_TYPE relationships."""
    # Create mapping from entity_type to type_id
    entity_types_df = extract_entity_types(df)
    type_id_map = dict(zip(entity_types_df["type_name"], entity_types_df["type_id"]))

    relationships = []

    for i, (_, ef_row) in enumerate(ef_df.iterrows()):
        df_row = df.iloc[i]
        relationships.append(
            {
                "ef_id": ef_row["ef_id"],
                "type_id": type_id_map.get(df_row["entity_type"], ""),
                "confidence": ef_row["confidence"],
            }
        )

    return pd.DataFrame(relationships)


def create_ef_to_source_relationships(df, ef_df):
    """Create DataFrame for SOURCED_FROM relationships."""
    # Create mapping from source_dataset to source_id
    sources_df = extract_sources(df)
    source_id_map = dict(zip(sources_df["name"], sources_df["source_id"]))

    relationships = []

    for i, (_, ef_row) in enumerate(ef_df.iterrows()):
        df_row = df.iloc[i]
        relationships.append(
            {
                "ef_id": ef_row["ef_id"],
                "source_id": source_id_map.get(df_row["source_dataset"], ""),
                "timestamp": ef_row["timestamp"],
            }
        )

    return pd.DataFrame(relationships)


def create_region_hierarchy():
    """Create DataFrame for PART_OF relationships between regions."""
    # Define some example hierarchical relationships
    hierarchies = [
        # Country -> Continent
        {
            "child_region_code": "USA",
            "parent_region_code": "NAM",
            "relationship_type": "COUNTRY_IN_CONTINENT",
        },
        {
            "child_region_code": "FR",
            "parent_region_code": "EUR",
            "relationship_type": "COUNTRY_IN_CONTINENT",
        },
        # Continent -> Global
        {
            "child_region_code": "NAM",
            "parent_region_code": "GLB",
            "relationship_type": "CONTINENT_IN_GLOBAL",
        },
        {
            "child_region_code": "EUR",
            "parent_region_code": "GLB",
            "relationship_type": "CONTINENT_IN_GLOBAL",
        },
        {
            "child_region_code": "ASI",
            "parent_region_code": "GLB",
            "relationship_type": "CONTINENT_IN_GLOBAL",
        },
        {
            "child_region_code": "AFR",
            "parent_region_code": "GLB",
            "relationship_type": "CONTINENT_IN_GLOBAL",
        },
        {
            "child_region_code": "SAM",
            "parent_region_code": "GLB",
            "relationship_type": "CONTINENT_IN_GLOBAL",
        },
        {
            "child_region_code": "OCE",
            "parent_region_code": "GLB",
            "relationship_type": "CONTINENT_IN_GLOBAL",
        },
    ]

    return pd.DataFrame(hierarchies)


if __name__ == "__main__":
    main()
