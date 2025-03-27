#!/usr/bin/env python3
import sys

sys.path.append(".")

print("Trying to import data_preparation module...")
try:
    from training.scripts.data_preparation import (
        format_instruction,
        load_and_prepare_data,
    )

    print("✅ Successfully imported data_preparation module")

    print("\nTrying to load data with use_neo4j=False...")
    try:
        train_data, val_data = load_and_prepare_data(use_neo4j=False)
        print(f"✅ Successfully loaded data without Neo4j")
        print(f"   Train dataset size: {len(train_data['train'])}")
        print(f"   Validation dataset size: {len(val_data['train'])}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")

    print(
        "\nTrying to load data with use_neo4j=True (should gracefully handle missing neo4j)..."
    )
    try:
        train_data, val_data = load_and_prepare_data(
            use_neo4j=True, neo4j_credentials={"uri": "bolt://localhost:7687"}
        )
        print(
            f"✅ Successfully handled Neo4j request: {len(train_data)}, {len(val_data)}"
        )
    except Exception as e:
        print(f"❌ Error when testing with Neo4j: {e}")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this script from the project root directory")
