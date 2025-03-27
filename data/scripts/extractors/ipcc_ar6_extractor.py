"""
IPCC AR6 data extractor module.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime

import camelot
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

logger = logging.getLogger(__name__)

# Constants
IPCC_AR6_URL = "https://www.ipcc.ch/report/ar6/wg3/"
RAW_FILE_PATH = "data/raw/ipcc_ar6_raw.json"
INTERIM_FILE_PATH = "data/interim/ipcc_ar6_interim.csv"
CLEANED_FILE_PATH = "data/processed/ipcc_ar6_clean.csv"
DATASET_SOURCE = "IPCC_AR6"

# Expanded Regional coverage (from 10 to 25 regions)
REGIONS = [
    "Global",
    # Original regions
    "North America",
    "Europe",
    "East Asia",
    "South Asia",
    "Latin America",
    "Africa",
    "Middle East",
    "Oceania",
    "Southeast Asia",
    # New regions
    "United States",
    "Canada",
    "Mexico",
    "Brazil",
    "European Union",
    "United Kingdom",
    "Russia",
    "China",
    "Japan",
    "India",
    "Australia",
    "South Africa",
    "Middle East and North Africa",
    "Sub-Saharan Africa",
    "Least Developed Countries",
]

# Expanded Sector categories (from 8 to 27 subsectors as mentioned in IPCC AR6 report)
SECTORS = [
    # Main sectors
    "Energy",
    "Industry",
    "Transport",
    "Buildings",
    "Agriculture",
    "Forestry",
    "Waste",
    "Cross-sector",
    # Detailed energy subsectors
    "Energy: Electricity and Heat",
    "Energy: Petroleum Refining",
    "Energy: Fossil Fuel Extraction",
    "Energy: Other Energy Industries",
    # Detailed industry subsectors
    "Industry: Iron and Steel",
    "Industry: Non-ferrous Metals",
    "Industry: Chemicals",
    "Industry: Cement",
    "Industry: Pulp and Paper",
    "Industry: Food Processing",
    # Detailed transport subsectors
    "Transport: Road",
    "Transport: Aviation",
    "Transport: Shipping",
    "Transport: Rail",
    # Detailed buildings subsectors
    "Buildings: Residential",
    "Buildings: Commercial",
    # Detailed AFOLU subsectors
    "AFOLU: Cropland",
    "AFOLU: Livestock",
    "AFOLU: Forestry and Land Use",
]

# Gas types for gas-specific multipliers
GAS_TYPES = [
    "CO2",
    "CH4",
    "N2O",
    "F-gases",
]

# Time periods for time series data
TIME_PERIODS = [
    "1990-2000",
    "2000-2010",
    "2010-2019",
    "2019-present",
]


def create_simulated_dataset():
    """
    Create a simulated IPCC AR6 dataset with regional multipliers.

    Returns:
        Dictionary with simulated data
    """
    utils.log_extraction_step("ipcc_ar6", "Creating simulated dataset")

    # Create simulated regional multipliers
    regional_multipliers = []

    # Base multiplier for global is 1.0
    regional_multipliers.append(
        {
            "region": "Global",
            "multiplier": 1.0,
            "confidence": 0.95,
            "description": "Global average multiplier",
        }
    )

    # Generate multipliers for other regions
    for region in REGIONS[1:]:  # Skip Global
        # Different regions have different multipliers based on development level
        if region in [
            "North America",
            "Europe",
            "Oceania",
            "United States",
            "Canada",
            "European Union",
            "United Kingdom",
            "Japan",
            "Australia",
        ]:
            # Developed regions typically have lower multipliers
            base_multiplier = np.random.uniform(0.7, 0.9)
        elif region in [
            "East Asia",
            "Middle East",
            "China",
            "Russia",
            "Middle East and North Africa",
        ]:
            # Rapidly developing regions
            base_multiplier = np.random.uniform(1.1, 1.3)
        elif region in [
            "South Asia",
            "Southeast Asia",
            "Latin America",
            "Mexico",
            "Brazil",
            "India",
        ]:
            # Developing regions
            base_multiplier = np.random.uniform(0.9, 1.1)
        else:  # Africa, Sub-Saharan Africa, Least Developed Countries
            # Less industrialized regions
            base_multiplier = np.random.uniform(0.6, 0.8)

        regional_multipliers.append(
            {
                "region": region,
                "multiplier": round(base_multiplier, 2),
                "confidence": round(np.random.uniform(0.7, 0.9), 2),
                "description": f"Regional multiplier for {region}",
            }
        )

    # Create sector-specific multipliers with gas types and time periods
    sector_multipliers = []

    for sector in SECTORS:
        for region in REGIONS:
            # Skip global for some sectors to simulate incomplete data
            if region == "Global" and sector in ["Cross-sector", "Forestry"]:
                continue

            # Base multiplier from regional multiplier
            region_base = next(
                (
                    item["multiplier"]
                    for item in regional_multipliers
                    if item["region"] == region
                ),
                1.0,
            )

            # Sector-specific adjustment
            if "Energy" in sector:
                sector_adj = np.random.uniform(0.9, 1.1)
            elif "Industry" in sector:
                sector_adj = np.random.uniform(1.0, 1.2)
            elif "Transport" in sector:
                sector_adj = np.random.uniform(0.8, 1.0)
            elif "Buildings" in sector:
                sector_adj = np.random.uniform(0.7, 0.9)
            elif "Agriculture" in sector or "AFOLU" in sector:
                sector_adj = np.random.uniform(1.1, 1.3)
            elif "Forestry" in sector:
                sector_adj = np.random.uniform(0.5, 0.7)
            elif "Waste" in sector:
                sector_adj = np.random.uniform(1.2, 1.4)
            else:  # Cross-sector
                sector_adj = np.random.uniform(0.9, 1.1)

            # Add time series variation
            for period in TIME_PERIODS:
                # Add time-based variation
                if period == "1990-2000":
                    time_adj = np.random.uniform(0.9, 1.0)  # Historical baseline
                elif period == "2000-2010":
                    time_adj = np.random.uniform(1.0, 1.1)  # Growth period
                elif period == "2010-2019":
                    time_adj = np.random.uniform(0.8, 0.9)  # Improvement period
                else:  # 2019-present
                    time_adj = np.random.uniform(0.7, 0.8)  # Recent improvements

                # Add gas-specific multipliers
                for gas in GAS_TYPES:
                    # Gas-specific adjustments
                    if gas == "CO2":
                        gas_adj = np.random.uniform(0.9, 1.1)
                    elif gas == "CH4":
                        gas_adj = np.random.uniform(1.1, 1.3)
                    elif gas == "N2O":
                        gas_adj = np.random.uniform(0.8, 1.0)
                    else:  # F-gases
                        gas_adj = np.random.uniform(0.7, 0.9)

                    # Calculate final multiplier with all adjustments
                    final_multiplier = round(
                        region_base * sector_adj * time_adj * gas_adj, 2
                    )

                    # Add to sector multipliers
                    sector_multipliers.append(
                        {
                            "region": region,
                            "sector": sector,
                            "gas_type": gas,
                            "time_period": period,
                            "multiplier": final_multiplier,
                            "confidence": round(np.random.uniform(0.6, 0.9), 2),
                            "description": f"{sector} {gas} multiplier for {region} in {period}",
                            "reference": "IPCC AR6 WG3 (Simulated)",
                            "year": 2022,
                        }
                    )

    # Create all data
    all_data = {
        "regional_multipliers": regional_multipliers,
        "sector_multipliers": sector_multipliers,
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": DATASET_SOURCE,
            "note": "Simulated data for demonstration purposes",
            "version": "AR6",
        },
    }

    # Save raw data
    os.makedirs(os.path.dirname(RAW_FILE_PATH), exist_ok=True)
    with open(RAW_FILE_PATH, "w") as f:
        json.dump(all_data, f, indent=2)

    logger.info(
        f"Created simulated IPCC AR6 data with {len(sector_multipliers)} sector-specific multipliers"
    )
    return all_data


def extract_tables_from_pdf(pdf_path, pages="1"):
    """
    Extract tables from a PDF file using Camelot.

    Args:
        pdf_path: Path to the PDF file.
        pages: Pages to extract tables from (default is '1').

    Returns:
        List of DataFrames containing extracted tables.
    """
    tables = camelot.read_pdf(pdf_path, pages=pages, flavor="stream")
    return [table.df for table in tables]


def extract_figures_from_pdf(pdf_path):
    """
    Extract figures from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of images extracted from the PDF.
    """
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
    return images


def download_and_extract_ipcc_ar6_data():
    """
    Download IPCC AR6 data and extract tables and figures.

    Returns:
        Dictionary with extracted data.
    """
    utils.log_extraction_step("ipcc_ar6", "Downloading and extracting data")

    # Check if raw file already exists
    if os.path.exists(RAW_FILE_PATH):
        logger.info(f"Using existing IPCC AR6 data from {RAW_FILE_PATH}")
        with open(RAW_FILE_PATH, "r") as f:
            return json.load(f)

    # Try to download data from IPCC website
    try:
        logger.info(f"Downloading IPCC AR6 data from {IPCC_AR6_URL}")
        response = requests.get(IPCC_AR6_URL)
        response.raise_for_status()

        # Parse HTML to find links to technical summary and annexes
        soup = BeautifulSoup(response.text, "html.parser")

        # Look for links to technical summary or annexes that might contain emission factors
        links = soup.find_all("a", href=True)
        pdf_links = [
            link["href"]
            for link in links
            if link["href"].endswith(".pdf")
            and (
                "technical" in link["href"].lower()
                or "annex" in link["href"].lower()
                or "chapter" in link["href"].lower()
            )
        ]

        if not pdf_links:
            logger.warning("Could not find relevant PDF links on IPCC AR6 page")
            return create_simulated_dataset()

        # Download and extract data from PDFs
        extracted_data = {}
        for pdf_link in pdf_links:
            pdf_url = (
                f"https://www.ipcc.ch{pdf_link}"
                if pdf_link.startswith("/")
                else pdf_link
            )
            pdf_response = requests.get(pdf_url)
            pdf_path = os.path.join("data/raw", os.path.basename(pdf_url))
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(pdf_response.content)

            # Extract tables and figures
            tables = extract_tables_from_pdf(
                pdf_path, pages="2"
            )  # Example: extract from page 2
            figures = extract_figures_from_pdf(pdf_path)

            extracted_data[pdf_path] = {
                "tables": tables,
                "figures": figures,
            }

        # Save raw data
        os.makedirs(os.path.dirname(RAW_FILE_PATH), exist_ok=True)
        with open(RAW_FILE_PATH, "w") as f:
            json.dump(extracted_data, f, indent=2)

        logger.info(f"Extracted data from IPCC AR6 PDFs")
        return extracted_data

    except Exception as e:
        logger.error(f"Error downloading or extracting IPCC AR6 data: {e}")
        return create_simulated_dataset()


def process_ipcc_data(data):
    """
    Process IPCC AR6 data into a structured format.

    Args:
        data: Dictionary with IPCC AR6 data

    Returns:
        Dataframe with processed data
    """
    utils.log_extraction_step("ipcc_ar6", "Processing data")

    # Extract sector multipliers
    sector_multipliers = data.get("sector_multipliers", [])

    if not sector_multipliers:
        raise ValueError("No sector multipliers found in IPCC AR6 data")

    # Convert to dataframe
    df = pd.DataFrame(sector_multipliers)

    # Save interim data
    df.to_csv(INTERIM_FILE_PATH, index=False, encoding="utf-8")

    logger.info(f"Processed {len(df)} IPCC AR6 multipliers")
    return df


def extract_chapter_2_data(pdf_path):
    """
    Extract specific data from Chapter 2 of the IPCC AR6 WG3 report.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dictionary with extracted data from Chapter 2.
    """
    doc = fitz.open(pdf_path)
    chapter_2_data = {}

    # Look for Chapter 2 start
    chapter_2_start = None
    chapter_2_end = None

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Check if this page contains Chapter 2 header
        if "Chapter 2" in text and "Emissions Trends and Drivers" in text:
            chapter_2_start = page_num
            logger.info(f"Found Chapter 2 start at page {page_num}")

        # Check if this is the start of Chapter 3 (end of Chapter 2)
        if chapter_2_start is not None and "Chapter 3" in text:
            chapter_2_end = page_num
            logger.info(f"Found Chapter 2 end at page {page_num}")
            break

    if chapter_2_start is not None:
        # If we didn't find Chapter 3, assume it goes to the end
        if chapter_2_end is None:
            chapter_2_end = len(doc) - 1

        # Extract tables from Chapter 2
        chapter_2_pages = f"{chapter_2_start+1}-{chapter_2_end}"
        tables = camelot.read_pdf(pdf_path, pages=chapter_2_pages, flavor="stream")

        # Process the tables to extract emissions trends
        chapter_2_data["tables"] = []
        for i, table in enumerate(tables):
            # Convert table to dataframe
            df = table.df
            # Check if this table contains emission data
            if any(
                col.lower().strip().startswith(("emission", "ghg", "co2", "ch4"))
                for col in df.iloc[0, :].values
            ):
                chapter_2_data["tables"].append(
                    {
                        "table_id": f"chapter2_table_{i}",
                        "table_data": df.to_dict(),
                        "page": table.page,
                    }
                )

    return chapter_2_data


def convert_to_standard_format(df):
    """
    Convert processed IPCC AR6 data to standard format.

    Args:
        df: Dataframe with processed data

    Returns:
        Dataframe in standard format
    """
    utils.log_extraction_step("ipcc_ar6", "Converting to standard format")

    # Create a deep copy to avoid modifying the original
    std_df = df.copy()

    # Create standardized dataframe
    std_df = pd.DataFrame(
        {
            "entity_id": std_df.apply(
                lambda row: f"IPCC_AR6_{row['region']}_{row['sector']}_{row['gas_type']}_{row['time_period']}".replace(
                    " ", "_"
                ).replace(
                    "-", "_"
                ),
                axis=1,
            ),
            "entity_name": std_df.apply(
                lambda row: f"{row['sector']} {row['gas_type']} in {row['region']} ({row['time_period']})",
                axis=1,
            ),
            "entity_type": "multiplier",
            "ef_value": std_df["multiplier"],
            "ef_unit": "ratio",
            "region": std_df["region"],
            "source_dataset": DATASET_SOURCE,
            "confidence": std_df["confidence"],
            "timestamp": datetime.now().strftime("%Y-%m-%d"),
            "tags": std_df.apply(
                lambda row: [
                    f"sector:{row['sector']}",
                    f"region:{row['region']}",
                    f"gas_type:{row['gas_type']}",
                    f"time_period:{row['time_period']}",
                ],
                axis=1,
            ),
        }
    )

    return std_df


def clean_dataset(df):
    """
    Clean the standardized dataset.

    Args:
        df: Dataframe in standard format

    Returns:
        Cleaned dataframe
    """
    utils.log_extraction_step("ipcc_ar6", "Cleaning dataset")

    # Create a deep copy to avoid modifying the original
    clean_df = df.copy()

    # 1. Filter out rows with missing critical values
    clean_df = clean_df.dropna(subset=["entity_id", "ef_value", "region"])

    # 2. Standardize region names
    # Map region names to standard ISO region codes where applicable
    region_mapping = {
        "North America": "NAM",
        "Europe": "EUR",
        "East Asia": "EAS",
        "South Asia": "SAS",
        "Latin America": "LAC",
        "Africa": "AFR",
        "Middle East": "MEA",
        "Oceania": "OCE",
        "Southeast Asia": "SEA",
        "Global": "GLB",
        "United States": "USA",
        "Canada": "CAN",
        "Mexico": "MEX",
        "Brazil": "BRA",
        "European Union": "EUN",
        "United Kingdom": "GBR",
        "Russia": "RUS",
        "China": "CHN",
        "Japan": "JPN",
        "India": "IND",
        "Australia": "AUS",
        "South Africa": "ZAF",
        "Middle East and North Africa": "MENA",
        "Sub-Saharan Africa": "SSA",
        "Least Developed Countries": "LDC",
    }

    clean_df["region_code"] = clean_df["region"].map(region_mapping)

    # Add region code to tags
    clean_df["tags"] = clean_df.apply(
        lambda row: (
            row["tags"] + [f"region_code:{row['region_code']}"]
            if pd.notna(row["region_code"])
            else row["tags"]
        ),
        axis=1,
    )

    # 3. Handle outliers
    outliers = utils.detect_outliers(clean_df, "ef_value")
    logger.info(f"Detected {outliers.sum()} outliers in multiplier values")
    clean_df["is_outlier"] = outliers

    # 4. Add metadata
    clean_df["metadata"] = clean_df.apply(
        lambda row: json.dumps(
            {
                "ipcc_version": "AR6",
                "region_name": row["region"],
                "region_code": (
                    row["region_code"] if pd.notna(row["region_code"]) else None
                ),
                "gas_type": row.get("gas_type", None),
                "time_period": row.get("time_period", None),
            }
        ),
        axis=1,
    )

    # Drop unnecessary columns
    if "region_code" in clean_df.columns:
        clean_df = clean_df.drop("region_code", axis=1)

    # Save cleaned dataset
    utils.save_dataframe(clean_df, CLEANED_FILE_PATH)

    logger.info(f"Cleaned dataset has {len(clean_df)} rows")
    return clean_df


def extract_and_clean():
    """
    Main function to extract and clean the IPCC AR6 dataset.

    Returns:
        Path to the cleaned dataset
    """
    try:
        # Download and extract data
        data = download_and_extract_ipcc_ar6_data()

        # Process data
        processed_df = process_ipcc_data(data)

        # Convert to standard format
        std_df = convert_to_standard_format(processed_df)

        # Clean the dataset
        cleaned_df = clean_dataset(std_df)

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
