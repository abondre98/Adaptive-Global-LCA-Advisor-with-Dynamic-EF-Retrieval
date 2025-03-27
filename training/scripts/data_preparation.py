#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation script for Mistral-7B fine-tuning.
This script extracts data from the Neo4j knowledge graph and prepares
the instruction dataset as specified in the Milestone2_PRD.md.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training/logs/data_preparation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("training/logs", exist_ok=True)


class Neo4jDataExtractor:
    """Class to extract data from Neo4j knowledge graph."""

    def __init__(self, uri: str, username: str, password: str):
        """Initialize connection to Neo4j.

        Args:
            uri: Neo4j database URI
            username: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        logger.info(f"Connected to Neo4j database at {uri}")

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()
        logger.info("Closed Neo4j connection")

    def get_emission_factors(self) -> List[Dict[str, Any]]:
        """Extract emission factors from Neo4j.

        Returns:
            List of emission factor dictionaries
        """
        query = """
        MATCH (ef:EmissionFactor)-[:APPLIES_TO_REGION]->(r:Region),
              (ef)-[:HAS_ENTITY_TYPE]->(et:EntityType),
              (ef)-[:SOURCED_FROM]->(s:Source)
        RETURN ef.entity_id AS entity_id,
               ef.entity_name AS entity_name,
               ef.ef_value AS ef_value,
               ef.ef_unit AS ef_unit,
               ef.confidence AS confidence,
               ef.is_outlier AS is_outlier,
               ef.multiplier_applied AS multiplier_applied,
               r.name AS region,
               et.name AS entity_type,
               s.name AS source
        LIMIT 25000
        """

        with self.driver.session() as session:
            result = session.run(query)
            records = [record.data() for record in result]
            logger.info(f"Extracted {len(records)} emission factors from Neo4j")
            return records


class InstructionGenerator:
    """Class to generate instruction dataset for fine-tuning."""

    def __init__(self, emission_factors: List[Dict[str, Any]]):
        """Initialize with emission factors data.

        Args:
            emission_factors: List of emission factor dictionaries
        """
        self.emission_factors = emission_factors
        self.df = pd.DataFrame(emission_factors)
        logger.info(
            f"Initialized instruction generator with {len(emission_factors)} emission factors"
        )

    def generate_basic_lookup(self, count: int) -> List[Dict[str, Any]]:
        """Generate basic lookup instruction examples.

        Args:
            count: Number of examples to generate

        Returns:
            List of instruction dictionaries
        """
        instructions = []
        sample_rows = self.df.sample(n=min(count, len(self.df)))

        for _, row in sample_rows.iterrows():
            entity_name = row["entity_name"]
            region = row["region"]
            ef_value = row["ef_value"]
            ef_unit = row["ef_unit"]
            entity_type = row["entity_type"]
            source = row["source"]

            instruction = {
                "instruction": f"What is the emission factor for {entity_name} in {region}?",
                "input": "",
                "output": f"The emission factor for {entity_name} in {region} is {ef_value} {ef_unit}. This data is sourced from {source}.",
                "metadata": {
                    "regions": [region],
                    "entity_types": [entity_type],
                    "difficulty": "basic",
                    "sources": [source],
                },
            }
            instructions.append(instruction)

        logger.info(f"Generated {len(instructions)} basic lookup instructions")
        return instructions

    def generate_regional_comparison(self, count: int) -> List[Dict[str, Any]]:
        """Generate regional comparison instruction examples.

        Args:
            count: Number of examples to generate

        Returns:
            List of instruction dictionaries
        """
        instructions = []
        entities = self.df["entity_name"].unique()

        for entity in entities[: min(count, len(entities))]:
            entity_df = self.df[self.df["entity_name"] == entity]
            if len(entity_df) < 2:
                continue

            regions = entity_df["region"].unique()
            if len(regions) < 2:
                continue

            # Select two random regions for comparison
            region1, region2 = random.sample(list(regions), 2)

            ef1 = entity_df[entity_df["region"] == region1].iloc[0]
            ef2 = entity_df[entity_df["region"] == region2].iloc[0]

            instruction = {
                "instruction": f"Compare the emission factor for {entity} between {region1} and {region2}.",
                "input": "",
                "output": f"In {region1}, the emission factor for {entity} is {ef1['ef_value']} {ef1['ef_unit']}, while in {region2} it is {ef2['ef_value']} {ef2['ef_unit']}. "
                + f"This represents a {'higher' if ef1['ef_value'] > ef2['ef_value'] else 'lower'} carbon intensity in {region1} compared to {region2}. "
                + f"The data for {region1} is sourced from {ef1['source']}, while the data for {region2} is from {ef2['source']}.",
                "metadata": {
                    "regions": [region1, region2],
                    "entity_types": [ef1["entity_type"]],
                    "difficulty": "moderate",
                    "sources": [ef1["source"], ef2["source"]],
                },
            }
            instructions.append(instruction)

        logger.info(f"Generated {len(instructions)} regional comparison instructions")
        return instructions

    def generate_multi_entity_analysis(self, count: int) -> List[Dict[str, Any]]:
        """Generate multi-entity analysis instruction examples.

        Args:
            count: Number of examples to generate

        Returns:
            List of instruction dictionaries
        """
        instructions = []
        regions = self.df["region"].unique()

        for region in regions[: min(count, len(regions))]:
            region_df = self.df[self.df["region"] == region]
            entity_types = region_df["entity_type"].unique()

            for entity_type in entity_types:
                entity_type_df = region_df[region_df["entity_type"] == entity_type]
                if len(entity_type_df) < 3:
                    continue

                # Get top 3 entities by emission factor value
                top3 = entity_type_df.nlargest(3, "ef_value")

                entities_str = ", ".join(
                    [
                        f"{row['entity_name']} ({row['ef_value']} {row['ef_unit']})"
                        for _, row in top3.iterrows()
                    ]
                )
                sources = list(top3["source"].unique())

                instruction = {
                    "instruction": f"What are the top 3 {entity_type} entities with the highest emission factors in {region}?",
                    "input": "",
                    "output": f"The top 3 {entity_type} entities with the highest emission factors in {region} are: {entities_str}. "
                    + f"This data is sourced from {', '.join(sources)}.",
                    "metadata": {
                        "regions": [region],
                        "entity_types": [entity_type],
                        "difficulty": "complex",
                        "sources": sources,
                    },
                }
                instructions.append(instruction)

                if len(instructions) >= count:
                    break

            if len(instructions) >= count:
                break

        logger.info(f"Generated {len(instructions)} multi-entity analysis instructions")
        return instructions

    def generate_methodological_questions(self, count: int) -> List[Dict[str, Any]]:
        """Generate methodological instruction examples.

        Args:
            count: Number of examples to generate

        Returns:
            List of instruction dictionaries
        """
        # Filter for entities with multipliers applied
        multiplier_df = self.df[self.df["multiplier_applied"] == True]

        if len(multiplier_df) == 0:
            logger.warning(
                "No entities with multipliers found, using simulated data for methodological questions"
            )
            # Simulate some multiplier data for example purposes
            multiplier_df = self.df.sample(min(count * 2, len(self.df)))

        instructions = []
        for _, row in multiplier_df.sample(min(count, len(multiplier_df))).iterrows():
            entity_name = row["entity_name"]
            region = row["region"]
            global_region = "Global"

            # Here we're creating a hypothetical scenario for adjustment
            instruction = {
                "instruction": f"How should I adjust the global emission factor for {entity_name} when using it in {region}?",
                "input": "",
                "output": f"When using the global emission factor for {entity_name} in {region}, you should apply a regional adjustment factor. "
                + f"The global dataset may not account for specific conditions in {region} such as energy mix, production methods, or regulatory environment. "
                + f"Based on our analysis, a multiplier has been applied to account for these regional differences. "
                + f"Always refer to local data sources when available, as they provide more accurate emission factors for region-specific calculations.",
                "metadata": {
                    "regions": [global_region, region],
                    "entity_types": [row["entity_type"]],
                    "difficulty": "moderate",
                    "sources": [row["source"]],
                },
            }
            instructions.append(instruction)

        logger.info(f"Generated {len(instructions)} methodological instructions")
        return instructions

    def generate_edge_cases(self, count: int) -> List[Dict[str, Any]]:
        """Generate edge case instruction examples.

        Args:
            count: Number of examples to generate

        Returns:
            List of instruction dictionaries
        """
        # Use outliers for edge cases
        outlier_df = self.df[self.df["is_outlier"] == True]

        if len(outlier_df) == 0:
            logger.warning(
                "No outliers found, using randomly selected data for edge cases"
            )
            outlier_df = self.df.sample(min(count * 2, len(self.df)))

        instructions = []
        for _, row in outlier_df.sample(min(count, len(outlier_df))).iterrows():
            entity_name = row["entity_name"]
            entity_type = row["entity_type"]

            # Create an edge case scenario
            new_entity = f"new sustainable {entity_name.lower()}"

            instruction = {
                "instruction": f"What emission factor should I use for a {new_entity}?",
                "input": f"I'm working on a project involving a {new_entity} which doesn't have established emission factors yet.",
                "output": f"For a {new_entity} without established emission factors, I recommend the following approach: "
                + f"1. Start with the emission factor for conventional {entity_name.lower()} as a baseline ({row['ef_value']} {row['ef_unit']}). "
                + f"2. Apply an adjustment factor based on available lifecycle assessment studies for similar sustainable alternatives. "
                + f"3. Document your assumptions and data sources clearly. "
                + f"4. Consider using sensitivity analysis to understand how variations in the emission factor might impact your overall results. "
                + f"5. Update your calculations when more specific emission factors become available. "
                + f"This approach provides a reasonable estimate while acknowledging the uncertainty in emerging product categories.",
                "metadata": {
                    "regions": [row["region"]],
                    "entity_types": [entity_type],
                    "difficulty": "complex",
                    "sources": [row["source"]],
                },
            }
            instructions.append(instruction)

        logger.info(f"Generated {len(instructions)} edge case instructions")
        return instructions

    def generate_verification_questions(self, count: int) -> List[Dict[str, Any]]:
        """Generate verification instruction examples.

        Args:
            count: Number of examples to generate

        Returns:
            List of instruction dictionaries
        """
        instructions = []
        sample_rows = self.df.sample(min(count, len(self.df)))

        confidence_levels = {(0, 30): "low", (30, 70): "medium", (70, 100): "high"}

        for _, row in sample_rows.iterrows():
            entity_name = row["entity_name"]
            region = row["region"]
            confidence = row["confidence"]

            # Determine confidence level text
            confidence_text = "low"
            for (lower, upper), level in confidence_levels.items():
                if lower <= confidence < upper:
                    confidence_text = level

            is_outlier = row["is_outlier"]
            outlier_text = ""
            if is_outlier:
                outlier_text = " However, this value has been flagged as a potential outlier, so additional verification is recommended."

            instruction = {
                "instruction": f"How reliable is the emission factor data for {entity_name} in {region}?",
                "input": "",
                "output": f"The emission factor data for {entity_name} in {region} has a {confidence_text} confidence score of {confidence}%. "
                + f"This data is sourced from {row['source']}.{outlier_text} "
                + f"When using this emission factor, consider the confidence level and potentially conduct sensitivity analysis if it's critical to your calculations.",
                "metadata": {
                    "regions": [region],
                    "entity_types": [row["entity_type"]],
                    "difficulty": "moderate",
                    "sources": [row["source"]],
                },
            }
            instructions.append(instruction)

        logger.info(f"Generated {len(instructions)} verification instructions")
        return instructions

    def augment_instructions(
        self, instructions: List[Dict[str, Any]], variants_per_instruction: int = 3
    ) -> List[Dict[str, Any]]:
        """Augment instructions with paraphrased variants.

        Args:
            instructions: List of instruction dictionaries
            variants_per_instruction: Number of variants to create per instruction

        Returns:
            List of augmented instruction dictionaries
        """
        augmented_instructions = []

        # Keep original instructions
        augmented_instructions.extend(instructions)

        # Simple template-based paraphrasing for now
        # In a real implementation, you might use a more sophisticated paraphrasing approach
        paraphrase_templates = {
            "What is": [
                "Can you tell me",
                "I need to know",
                "Please provide",
                "Could you give me information about",
            ],
            "Compare": [
                "What is the difference between",
                "How do",
                "Can you contrast",
                "What's the comparison between",
            ],
            "How should I": [
                "What's the best way to",
                "What is the recommended approach for",
                "How do I",
                "What's the proper method for",
            ],
            "What are": [
                "Could you list",
                "Please identify",
                "Can you enumerate",
                "I need to know about",
            ],
        }

        for instruction in tqdm(instructions, desc="Augmenting instructions"):
            original_query = instruction["instruction"]

            # Generate variants
            variants = []
            for _ in range(variants_per_instruction):
                variant = original_query

                # Apply random paraphrasing
                for original, alternatives in paraphrase_templates.items():
                    if original in variant:
                        replacement = random.choice(alternatives)
                        variant = variant.replace(original, replacement, 1)

                if variant != original_query and variant not in variants:
                    variants.append(variant)

            # Create new instructions with variants
            for variant in variants:
                new_instruction = instruction.copy()
                new_instruction["instruction"] = variant
                augmented_instructions.append(new_instruction)

        logger.info(
            f"Augmented {len(instructions)} instructions to {len(augmented_instructions)} total instructions"
        )
        return augmented_instructions

    def generate_full_instruction_set(self) -> List[Dict[str, Any]]:
        """Generate the full instruction set according to PRD specifications.

        Returns:
            List of instruction dictionaries
        """
        # Calculate counts based on percentages in PRD
        total_count = 3000
        counts = {
            "basic_lookup": int(0.30 * total_count),
            "regional_comparison": int(0.25 * total_count),
            "multi_entity_analysis": int(0.15 * total_count),
            "methodological": int(0.15 * total_count),
            "edge_cases": int(0.10 * total_count),
            "verification": int(0.05 * total_count),
        }

        # Generate instructions for each category
        instructions = []
        instructions.extend(self.generate_basic_lookup(counts["basic_lookup"]))
        instructions.extend(
            self.generate_regional_comparison(counts["regional_comparison"])
        )
        instructions.extend(
            self.generate_multi_entity_analysis(counts["multi_entity_analysis"])
        )
        instructions.extend(
            self.generate_methodological_questions(counts["methodological"])
        )
        instructions.extend(self.generate_edge_cases(counts["edge_cases"]))
        instructions.extend(
            self.generate_verification_questions(counts["verification"])
        )

        # Augment instructions with variants
        augmented_instructions = self.augment_instructions(instructions)

        # Shuffle the instructions
        random.shuffle(augmented_instructions)

        logger.info(
            f"Generated full instruction set with {len(augmented_instructions)} instructions"
        )
        return augmented_instructions

    def save_instruction_set(
        self,
        instructions: List[Dict[str, Any]],
        output_file: str,
        split_ratio: float = 0.8,
    ) -> Tuple[str, str]:
        """Save the instruction set to a JSON file and split into train/val sets.

        Args:
            instructions: List of instruction dictionaries
            output_file: Path to output file
            split_ratio: Train/val split ratio

        Returns:
            Tuple of (train_file, val_file) paths
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save full set
        with open(output_file, "w") as f:
            json.dump(instructions, f, indent=2)

        # Create train/val split
        split_idx = int(len(instructions) * split_ratio)
        train_instructions = instructions[:split_idx]
        val_instructions = instructions[split_idx:]

        train_file = output_file.replace(".json", "_train.json")
        val_file = output_file.replace(".json", "_val.json")

        with open(train_file, "w") as f:
            json.dump(train_instructions, f, indent=2)

        with open(val_file, "w") as f:
            json.dump(val_instructions, f, indent=2)

        logger.info(f"Saved full instruction set to {output_file}")
        logger.info(
            f"Saved {len(train_instructions)} training instructions to {train_file}"
        )
        logger.info(
            f"Saved {len(val_instructions)} validation instructions to {val_file}"
        )

        return train_file, val_file


def create_synthetic_ef_data(count: int = 500) -> List[Dict[str, Any]]:
    """Create synthetic emission factor data when Neo4j is not available.

    Args:
        count: Number of synthetic records to create

    Returns:
        List of synthetic emission factor dictionaries
    """
    logger.warning(
        "Creating synthetic data - this should only be used for development!"
    )

    entity_types = [
        "Product",
        "Sector",
        "Energy",
        "Transport",
        "Agriculture",
        "Industrial Process",
        "Waste",
    ]
    regions = [
        "Global",
        "USA",
        "EU",
        "China",
        "India",
        "Brazil",
        "Japan",
        "UK",
        "France",
        "Germany",
        "Canada",
    ]
    sources = [
        "USEPA",
        "Agribalyse",
        "DEFRA",
        "IPCC",
        "EcoInvent",
        "USEEIO",
        "EXIOBASE",
        "GREET",
    ]

    # Create product names based on entity types
    product_templates = {
        "Product": [
            "Cotton T-shirt",
            "Polyester Fabric",
            "Aluminium Can",
            "Glass Bottle",
            "Paper Bag",
            "Plastic Container",
            "Smartphone",
            "Laptop",
            "Desktop Computer",
            "Television",
            "Refrigerator",
            "Washing Machine",
            "Sofa",
            "Wooden Table",
            "Steel Chair",
        ],
        "Sector": [
            "Retail",
            "Manufacturing",
            "Healthcare",
            "Education",
            "Financial Services",
            "Information Technology",
            "Construction",
            "Mining",
            "Food Production",
            "Hospitality",
        ],
        "Energy": [
            "Coal Electricity",
            "Natural Gas Electricity",
            "Solar Power",
            "Wind Power",
            "Hydroelectric Power",
            "Nuclear Power",
            "Biomass Energy",
            "Geothermal Energy",
        ],
        "Transport": [
            "Passenger Car",
            "Bus",
            "Train",
            "Airplane",
            "Cargo Ship",
            "Light Commercial Vehicle",
            "Heavy Duty Truck",
            "Motorcycle",
            "Ferry",
            "Electric Vehicle",
        ],
        "Agriculture": [
            "Wheat Production",
            "Rice Cultivation",
            "Corn Farming",
            "Cattle Farming",
            "Poultry Production",
            "Pig Farming",
            "Soybean Cultivation",
            "Cotton Farming",
        ],
        "Industrial Process": [
            "Cement Production",
            "Steel Manufacturing",
            "Aluminium Smelting",
            "Chemical Processing",
            "Petroleum Refining",
            "Pulp and Paper Manufacturing",
        ],
        "Waste": [
            "Landfill",
            "Incineration",
            "Composting",
            "Recycling",
            "Wastewater Treatment",
        ],
    }

    synthetic_data = []
    for i in range(count):
        entity_type = random.choice(entity_types)
        entity_name = random.choice(
            product_templates.get(entity_type, ["Generic Item"])
        )
        region = random.choice(regions)

        # Generate a reasonable emission factor based on entity type
        base_value = {
            "Product": random.uniform(0.1, 50),
            "Sector": random.uniform(10, 1000),
            "Energy": random.uniform(50, 1000),
            "Transport": random.uniform(50, 500),
            "Agriculture": random.uniform(1, 100),
            "Industrial Process": random.uniform(100, 2000),
            "Waste": random.uniform(10, 500),
        }.get(entity_type, random.uniform(1, 100))

        # Apply some regional variation
        region_multiplier = {
            "Global": 1.0,
            "USA": random.uniform(0.8, 1.2),
            "EU": random.uniform(0.7, 1.0),
            "China": random.uniform(1.0, 1.5),
            "India": random.uniform(1.0, 1.4),
            "Brazil": random.uniform(0.9, 1.2),
            "Japan": random.uniform(0.7, 1.0),
            "UK": random.uniform(0.7, 1.0),
            "France": random.uniform(0.6, 0.9),
            "Germany": random.uniform(0.7, 1.0),
            "Canada": random.uniform(0.8, 1.1),
        }.get(region, 1.0)

        ef_value = base_value * region_multiplier

        # Determine unit based on entity type
        ef_unit = random.choice(
            ["kg CO2e", "kg CO2e/kg", "kg CO2e/kWh", "kg CO2e/km", "kg CO2e/USD"]
        )

        is_outlier = random.random() < 0.05  # 5% chance of being an outlier
        multiplier_applied = (
            region != "Global" and random.random() < 0.2
        )  # 20% chance of having multiplier applied for non-global
        confidence = random.randint(70, 99)  # High confidence scores as per PRD

        synthetic_record = {
            "entity_id": f"EF{i:05d}",
            "entity_name": entity_name,
            "ef_value": round(ef_value, 2),
            "ef_unit": ef_unit,
            "confidence": confidence,
            "is_outlier": is_outlier,
            "multiplier_applied": multiplier_applied,
            "region": region,
            "entity_type": entity_type,
            "source": random.choice(sources),
        }
        synthetic_data.append(synthetic_record)

    return synthetic_data


def main():
    """Main function to prepare data for fine-tuning."""
    # Create output directories
    os.makedirs("training/data", exist_ok=True)

    try:
        # Try to connect to Neo4j and extract data
        # You'll need to replace these with actual Neo4j credentials
        uri = "bolt://localhost:7687"
        username = "neo4j"
        password = "password"

        extractor = Neo4jDataExtractor(uri, username, password)
        emission_factors = extractor.get_emission_factors()
        extractor.close()

    except Exception as e:
        logger.warning(f"Could not connect to Neo4j: {e}")
        logger.warning("Using synthetic data instead")
        emission_factors = create_synthetic_ef_data(2000)

    # Generate instruction dataset
    generator = InstructionGenerator(emission_factors)
    instructions = generator.generate_full_instruction_set()

    # Save instruction dataset
    train_file, val_file = generator.save_instruction_set(
        instructions, "training/data/instructions.json"
    )

    logger.info(
        f"Data preparation complete. Training file: {train_file}, Validation file: {val_file}"
    )


if __name__ == "__main__":
    main()
