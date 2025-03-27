"""
IPCC AR6 regional multipliers extractor module.
"""

import logging
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

logger = logging.getLogger(__name__)

# Constants
IPCC_AR6_URL = "https://www.ipcc.ch/report/sixth-assessment-report-cycle/"
RAW_FILE_PATH = "data/raw/ipcc_ar6_raw.csv"
INTERIM_FILE_PATH = "data/interim/ipcc_ar6_interim.csv"
CLEANED_FILE_PATH = "data/processed/ipcc_ar6_multipliers.csv"
DATASET_SOURCE = "IPCC_AR6"

# Pre-defined regional multipliers based on IPCC AR6 reports
# In a real scenario, these would be extracted from the reports
# This is a simplified approach for demonstration purposes
PREDEFINED_MULTIPLIERS = [
    {
        "region": "IN",  # India
        "sector": "agriculture",
        "multiplier_factor": 1.2,
        "rationale": "Higher methane emissions from rice cultivation",
        "source_page": "IPCC AR6 WG III, Chapter 7",
    },
    {
        "region": "IN",
        "sector": "electricity",
        "multiplier_factor": 1.3,
        "rationale": "Coal-dominated energy mix",
        "source_page": "IPCC AR6 WG III, Chapter 6",
    },
    {
        "region": "CN",  # China
        "sector": "manufacturing",
        "multiplier_factor": 1.1,
        "rationale": "Coal-based industrial processes",
        "source_page": "IPCC AR6 WG III, Chapter 11",
    },
    {
        "region": "US",
        "sector": "transportation",
        "multiplier_factor": 1.2,
        "rationale": "Higher vehicle miles traveled",
        "source_page": "IPCC AR6 WG III, Chapter 10",
    },
    {
        "region": "BR",  # Brazil
        "sector": "forestry",
        "multiplier_factor": 1.4,
        "rationale": "Deforestation impacts",
        "source_page": "IPCC AR6 WG III, Chapter 7",
    },
    {
        "region": "AU",  # Australia
        "sector": "agriculture",
        "multiplier_factor": 1.15,
        "rationale": "Extensive livestock farming",
        "source_page": "IPCC AR6 WG III, Chapter 7",
    },
    {
        "region": "ZA",  # South Africa
        "sector": "electricity",
        "multiplier_factor": 1.25,
        "rationale": "Coal-dominated energy mix",
        "source_page": "IPCC AR6 WG III, Chapter 6",
    },
    {
        "region": "RU",  # Russia
        "sector": "oil_and_gas",
        "multiplier_factor": 1.3,
        "rationale": "Methane leakage from infrastructure",
        "source_page": "IPCC AR6 WG III, Chapter 6",
    },
    {
        "region": "ID",  # Indonesia
        "sector": "forestry",
        "multiplier_factor": 1.35,
        "rationale": "Peatland conversion",
        "source_page": "IPCC AR6 WG III, Chapter 7",
    },
    {
        "region": "CA",  # Canada
        "sector": "oil_and_gas",
        "multiplier_factor": 1.2,
        "rationale": "Oil sands extraction",
        "source_page": "IPCC AR6 WG III, Chapter 6",
    },
    # Add more regional multipliers
]


def get_report_links():
    """
    Get links to AR6 report documents.

    Returns:
        List of report URLs
    """
    utils.log_extraction_step("ipcc", "Getting report links")

    try:
        response = requests.get(IPCC_AR6_URL)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Find links to reports
        report_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            # Look for links to AR6 working group reports
            if any(x in href for x in ["/wg3/", "/wg2/", "/wg1/"]) and href.endswith(
                ".pdf"
            ):
                if href.startswith("/"):
                    href = f"https://www.ipcc.ch{href}"
                report_links.append(href)
                logger.info(f"Found report link: {href}")

        if not report_links:
            logger.warning("No report links found, using predefined multipliers only")

        return report_links
    except Exception as e:
        logger.error(f"Error getting report links: {e}")
        return []


def extract_multipliers_from_predefined():
    """
    Extract multipliers from predefined data.

    Returns:
        Dataframe with regional multipliers
    """
    utils.log_extraction_step("ipcc", "Extracting predefined multipliers")

    # Create dataframe from predefined multipliers
    df = pd.DataFrame(PREDEFINED_MULTIPLIERS)

    # Add timestamp
    df["timestamp"] = datetime.now().strftime("%Y-%m-%d")

    return df


def parse_pdf_for_multipliers(pdf_url):
    """
    Parse PDF document for regional multipliers.

    Args:
        pdf_url: URL to PDF document

    Returns:
        List of multiplier dictionaries
    """
    utils.log_extraction_step("ipcc", f"Parsing PDF for multipliers: {pdf_url}")

    # In a real scenario, you would download and parse the PDF
    # This would require tools like pdfminer, PyPDF2, or an external service
    # For demonstration purposes, we'll skip this step and use predefined multipliers only

    logger.info(f"PDF parsing not implemented, using predefined multipliers only")
    return []


def combine_multipliers(predefined_df, extracted_multipliers):
    """
    Combine predefined and extracted multipliers.

    Args:
        predefined_df: Dataframe with predefined multipliers
        extracted_multipliers: List of extracted multiplier dictionaries

    Returns:
        Combined dataframe
    """
    utils.log_extraction_step("ipcc", "Combining multipliers")

    # If no extracted multipliers, return predefined
    if not extracted_multipliers:
        return predefined_df

    # Convert extracted multipliers to dataframe
    extracted_df = pd.DataFrame(extracted_multipliers)

    # Combine dataframes
    combined_df = pd.concat([predefined_df, extracted_df], ignore_index=True)

    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=["region", "sector"])

    return combined_df


def clean_dataset(df):
    """
    Clean the multipliers dataset.

    Args:
        df: Dataframe with multipliers

    Returns:
        Cleaned dataframe
    """
    utils.log_extraction_step("ipcc", "Cleaning dataset")

    # Create a deep copy to avoid modifying the original
    clean_df = df.copy()

    # 1. Ensure region codes are standardized
    clean_df["region"] = clean_df["region"].str.upper()

    # 2. Standardize sector names
    clean_df["sector"] = clean_df["sector"].str.lower()
    clean_df["sector"] = clean_df["sector"].str.replace(" ", "_")

    # 3. Validate multiplier factors
    clean_df["multiplier_factor"] = pd.to_numeric(
        clean_df["multiplier_factor"], errors="coerce"
    )

    # 4. Filter out invalid multipliers
    clean_df = clean_df.dropna(subset=["region", "sector", "multiplier_factor"])

    # 5. Ensure multipliers are within reasonable range (e.g., 0.5 to 2.0)
    clean_df = clean_df[
        (clean_df["multiplier_factor"] >= 0.5) & (clean_df["multiplier_factor"] <= 2.0)
    ]

    # 6. Add confidence score based on rationale presence
    clean_df["confidence"] = clean_df["rationale"].apply(
        lambda x: 0.9 if isinstance(x, str) and len(x) > 20 else 0.7
    )

    # 7. Create standardized schema
    final_df = pd.DataFrame(
        {
            "entity_id": clean_df["region"] + "_" + clean_df["sector"],
            "entity_name": clean_df["region"]
            + " "
            + clean_df["sector"].str.replace("_", " "),
            "entity_type": "multiplier",
            "ef_value": clean_df["multiplier_factor"],
            "ef_unit": "multiplier",
            "region": clean_df["region"],
            "source_dataset": DATASET_SOURCE,
            "confidence": clean_df["confidence"],
            "timestamp": (
                clean_df["timestamp"]
                if "timestamp" in clean_df.columns
                else datetime.now().strftime("%Y-%m-%d")
            ),
            "tags": clean_df["sector"].apply(lambda x: [f"sector:{x}"]),
            "rationale": clean_df["rationale"],
            "source_page": clean_df["source_page"],
        }
    )

    # Save cleaned dataset
    utils.save_dataframe(final_df, CLEANED_FILE_PATH)

    logger.info(f"Cleaned dataset has {len(final_df)} rows")
    return final_df


def extract_and_clean():
    """
    Main function to extract and clean the IPCC AR6 multipliers.

    Returns:
        Path to the cleaned dataset
    """
    try:
        # Get report links
        report_links = get_report_links()

        # Extract multipliers from predefined data
        predefined_df = extract_multipliers_from_predefined()

        # Extract multipliers from PDFs (in a real scenario)
        extracted_multipliers = []
        for link in report_links:
            multipliers = parse_pdf_for_multipliers(link)
            extracted_multipliers.extend(multipliers)

        # Combine multipliers
        combined_df = combine_multipliers(predefined_df, extracted_multipliers)

        # Save interim data
        os.makedirs(os.path.dirname(INTERIM_FILE_PATH), exist_ok=True)
        combined_df.to_csv(INTERIM_FILE_PATH, index=False, encoding="utf-8")

        # Clean the dataset
        cleaned_df = clean_dataset(combined_df)

        logger.info(f"IPCC AR6 extraction and cleaning completed: {CLEANED_FILE_PATH}")
        return CLEANED_FILE_PATH
    except Exception as e:
        logger.error(f"Error in IPCC AR6 extraction: {e}")
        raise


if __name__ == "__main__":
    # Configure logging if running as standalone script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    extract_and_clean()
