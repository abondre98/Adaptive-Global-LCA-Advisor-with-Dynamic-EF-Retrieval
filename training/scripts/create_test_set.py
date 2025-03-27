#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import random
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training/logs/create_test_set.log"),
    ],
)

logger = logging.getLogger(__name__)


def create_test_set(test_percentage=0.1):
    """
    Create a test set by sampling a percentage of the full instruction dataset.

    Args:
        test_percentage: Percentage of data to use for test set (default: 0.1 or 10%)
    """
    # Ensure the data directory exists
    data_dir = Path("training/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load the full instruction set
    full_dataset_path = data_dir / "instructions.json"

    logger.info(f"Loading full instruction dataset from {full_dataset_path}")
    with open(full_dataset_path, "r") as f:
        full_dataset = json.load(f)

    # Calculate number of examples for test set
    total_examples = len(full_dataset)
    test_size = int(total_examples * test_percentage)

    logger.info(f"Total examples in dataset: {total_examples}")
    logger.info(
        f"Creating test set with {test_size} examples ({test_percentage*100:.1f}%)"
    )

    # Randomly sample test examples
    random.seed(42)  # For reproducibility
    test_indices = random.sample(range(total_examples), test_size)
    test_examples = [full_dataset[i] for i in test_indices]

    # Save test set
    test_dataset_path = data_dir / "instructions_test.json"
    with open(test_dataset_path, "w") as f:
        json.dump(test_examples, f, indent=2)

    logger.info(f"Saved {len(test_examples)} test instructions to {test_dataset_path}")

    # Optional: Check the distribution of instruction categories in the test set
    categories = {}
    for example in test_examples:
        difficulty = example["metadata"]["difficulty"]
        if difficulty in categories:
            categories[difficulty] += 1
        else:
            categories[difficulty] = 1

    logger.info("Test set category distribution:")
    for category, count in categories.items():
        logger.info(f"  {category}: {count} ({count/len(test_examples)*100:.1f}%)")

    return test_examples


if __name__ == "__main__":
    create_test_set(0.1)  # Create test set with 10% of data
