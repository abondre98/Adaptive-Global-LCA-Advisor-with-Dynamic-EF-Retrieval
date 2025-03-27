"""
Utility functions for data extraction and cleaning.
"""

import hashlib
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def create_checksum(file_path: str) -> str:
    """
    Create an MD5 checksum for a file.

    Args:
        file_path: Path to the file

    Returns:
        MD5 checksum as a string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url: str, destination: str) -> str:
    """
    Download a file from a URL to a destination path.

    Args:
        url: URL to download from
        destination: Path to save the file to

    Returns:
        Path to the downloaded file
    """
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded {url} to {destination}")
        return destination
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        raise


def detect_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers in a dataframe column using z-score and IQR methods.

    This implements a more robust detection by combining:
    1. Z-score method for normal distributions
    2. IQR (Interquartile Range) method for skewed distributions

    Args:
        df: Dataframe to detect outliers in
        column: Column name to check for outliers
        threshold: Z-score threshold for outlier detection (default: 3.0)

    Returns:
        Boolean series with True for outliers
    """
    # Skip if column doesn't exist or has no values
    if column not in df.columns or df[column].isna().all():
        return pd.Series([False] * len(df))

    # Get values as numpy array, dropping NAs
    values = df[column].dropna().values

    if len(values) < 10:  # Not enough data for reliable outlier detection
        return pd.Series([False] * len(df))

    # Method 1: Z-score
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:  # Avoid division by zero
        z_score_outliers = pd.Series([False] * len(df))
    else:
        z_scores = np.abs((df[column] - mean) / std)
        z_score_outliers = z_scores > threshold

    # Method 2: IQR (more robust to skewed distributions)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    if iqr == 0:  # Avoid division by zero
        iqr_outliers = pd.Series([False] * len(df))
    else:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    # Combine the two methods (either method can flag an outlier)
    combined_outliers = z_score_outliers | iqr_outliers

    # Fill NA values with False
    combined_outliers = combined_outliers.fillna(False)

    return combined_outliers


def standardize_units(
    value: float, source_unit: str, target_unit: str = "kg CO2e/kg"
) -> float:
    """
    Convert emission factor values between different units.

    Args:
        value: Value to convert
        source_unit: Original unit
        target_unit: Target unit (default is kg CO2e/kg)

    Returns:
        Converted value
    """
    # Unit conversion factors
    conversion_factors = {
        "g CO2e/kg": 0.001,  # to kg CO2e/kg
        "ton CO2e/kg": 1000,  # to kg CO2e/kg
        "kg CO2e/ton": 0.001,  # to kg CO2e/kg
        # Add more conversion factors as needed
    }

    # If source and target are the same, no conversion needed
    if source_unit == target_unit:
        return value

    # Create conversion key
    conversion_key = f"{source_unit}_to_{target_unit}"

    # Check if we have a direct conversion factor
    if conversion_key in conversion_factors:
        return value * conversion_factors[conversion_key]

    # Handle more complex conversions or raise error
    if source_unit in conversion_factors:
        # Convert to kg CO2e/kg first
        intermediate = value * conversion_factors[source_unit]
        # Then convert to target if needed (not implemented here)
        logger.warning(
            f"Complex unit conversion from {source_unit} to {target_unit} not fully implemented"
        )
        return intermediate

    logger.error(f"Unit conversion from {source_unit} to {target_unit} not supported")
    return value  # Return original value if conversion not possible


def log_extraction_step(dataset: str, step: str) -> None:
    """
    Log a data extraction step.

    Args:
        dataset: Name of the dataset being processed
        step: Description of the extraction step
    """
    logger.info(f"[{dataset.upper()}] {step}")


def log_harmonization_step(step: str) -> None:
    """
    Log a harmonization step.

    Args:
        step: Description of the harmonization step
    """
    logger.info(f"[HARMONIZATION] {step}")


def save_dataframe(
    df: pd.DataFrame, file_path: str, include_index: bool = False
) -> None:
    """
    Save a dataframe to CSV with standardized formatting.

    Args:
        df: DataFrame to save
        file_path: Path to save the file to
        include_index: Whether to include the index in the output
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=include_index, encoding="utf-8")
    logger.info(f"Saved DataFrame with {len(df)} rows to {file_path}")

    # Create and log checksum
    checksum = create_checksum(file_path)
    checksum_file = f"{file_path}.md5"
    with open(checksum_file, "w") as f:
        f.write(checksum)
    logger.info(f"Created checksum {checksum} for {file_path}")


def create_standard_schema() -> Dict[str, str]:
    """
    Return the standard schema for harmonized datasets.

    Returns:
        Dictionary with field names and data types
    """
    return {
        "entity_id": "str",  # Unique identifier
        "entity_name": "str",  # Human-readable name
        "entity_type": "str",  # "product", "sector", or "process"
        "ef_value": "float",  # Emission factor value
        "ef_unit": "str",  # Unit (kg CO₂e per functional unit)
        "region": "str",  # ISO country code
        "source_dataset": "str",  # Original source
        "confidence": "float",  # Confidence rating (0-1)
        "timestamp": "datetime",  # Last updated
        "tags": "list[str]",  # Additional classifiers
    }
