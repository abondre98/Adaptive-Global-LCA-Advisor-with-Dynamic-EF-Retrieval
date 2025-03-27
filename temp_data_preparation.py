#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation module for Mistral-7B fine-tuning.
This module handles loading, formatting, and preparing the training data
from both JSON files and Neo4j database.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


def load_and_prepare_data(
    train_path: str = "training/data/instructions_train.json",
    val_path: str = "training/data/instructions_val.json",
    test_path: str = "training/data/instructions_test.json",
    use_neo4j: bool = False,
    neo4j_credentials: Optional[Dict[str, str]] = None,
) -> Tuple[DatasetDict, DatasetDict]:
    """
    Load and prepare data for training, either from JSON files or Neo4j.

    Args:
        train_path: Path to training data JSON file
        val_path: Path to validation data JSON file
        test_path: Path to test data JSON file
        use_neo4j: Whether to use Neo4j database instead of files
        neo4j_credentials: Credentials for Neo4j database connection

    Returns:
        Tuple of (train_dataset, val_dataset) as DatasetDict objects
    """
    logger.info("Loading data for training and validation...")

    if use_neo4j and neo4j_credentials:
        logger.info("Loading data from Neo4j database...")
        train_data, val_data = _load_data_from_neo4j(neo4j_credentials)
    else:
        logger.info(f"Loading data from JSON files: {train_path}, {val_path}")
        train_data = _load_json_file(train_path)
        val_data = _load_json_file(val_path)

    # Check if data was loaded successfully
    if not train_data or not val_data:
        raise ValueError("Failed to load training or validation data")

    # Convert to Dataset objects
    train_dataset = DatasetDict({"train": Dataset.from_list(train_data)})
    val_dataset = DatasetDict({"train": Dataset.from_list(val_data)})

    logger.info(
        f"Loaded {len(train_data)} training examples and {len(val_data)} validation examples"
    )

    return train_dataset, val_dataset


def _load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of instruction examples
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        # Try to load line by line in case it's a JSONL file
        try:
            data = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            if data:
                logger.info(f"Successfully loaded as JSONL: {file_path}")
                return data
        except Exception as e:
            logger.error(f"Failed to load as JSONL: {e}")
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")

    return []


def _load_data_from_neo4j(
    credentials: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load data from Neo4j database.

    Args:
        credentials: Neo4j database credentials

    Returns:
        Tuple of (train_data, val_data)
    """
    try:
        # Import Neo4j only when needed
        try:
            from neo4j import GraphDatabase
        except ImportError:
            logger.error(
                "Neo4j module not installed. Please install it using 'pip install neo4j'"
            )
            logger.info("Returning empty datasets as fallback")
            return [], []

        uri = credentials.get("uri", "bolt://localhost:7687")
        username = credentials.get("username", "neo4j")
        password = credentials.get("password", "password")

        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))

        instructions = []

        with driver.session() as session:
            # Query emission factors and generate instructions
            query = """
            MATCH (ef:EmissionFactor)-[:APPLIES_TO_REGION]->(r:Region)
            MATCH (ef)-[:HAS_ENTITY_TYPE]->(et:EntityType)
            MATCH (ef)-[:SOURCED_FROM]->(s:Source)
            RETURN ef, r, et, s
            LIMIT 10000
            """

            result = session.run(query)

            for record in result:
                ef = record["ef"]
                region = record["r"]
                entity_type = record["et"]
                source = record["s"]

                # Create instruction example
                example = {
                    "instruction": f"What is the emission factor for {ef['entity_name']} in {region['name']}?",
                    "input": "",
                    "output": f"The emission factor for {ef['entity_name']} in {region['name']} is {ef['ef_value']} {ef['ef_unit']}. This data is sourced from {source['name']}.",
                    "metadata": {
                        "regions": [region["region_code"]],
                        "entity_types": [entity_type["type_name"]],
                        "difficulty": "basic",
                        "sources": [source["name"]],
                    },
                }

                instructions.append(example)

        # Shuffle and split data
        random.shuffle(instructions)
        split_idx = int(len(instructions) * 0.8)
        train_data = instructions[:split_idx]
        val_data = instructions[split_idx:]

        return train_data, val_data

    except Exception as e:
        logger.error(f"Error loading data from Neo4j: {e}")
        return [], []


def format_instruction(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Format example as instruction for Mistral-7B.

    Args:
        example: Dictionary containing instruction data

    Returns:
        Dictionary with formatted text
    """
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    # Format using Mistral chat template
    if input_text:
        formatted = f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output} </s>"
    else:
        formatted = f"<s>[INST] {instruction} [/INST] {output} </s>"

    return {"text": formatted}


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training/logs/data_preparation.log"),
            logging.StreamHandler(),
        ],
    )

    # Test data loading and preparation
    train_dataset, val_dataset = load_and_prepare_data()

    logger.info(f"Train dataset size: {len(train_dataset['train'])}")
    logger.info(f"Validation dataset size: {len(val_dataset['train'])}")

    # Format samples
    train_dataset = train_dataset.map(format_instruction)

    # Print a sample
    logger.info("Sample formatted instruction:")
    logger.info(train_dataset["train"][0]["text"][:500] + "...")
