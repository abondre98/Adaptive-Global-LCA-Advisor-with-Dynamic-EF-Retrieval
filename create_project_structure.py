#!/usr/bin/env python3
import os
import shutil


def create_project_structure():
    # Define the base directories
    base_dirs = [
        "data/raw",
        "data/interim",
        "data/processed",
        "data/scripts/extractors",
        "data/scripts/harmonization",
        "data/logs",
        "data/documentation",
        "models",
        "models/checkpoints",
        "models/final",
        "notebooks",
        "tests",
        "configs",
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
    ]

    # Create directories
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create __init__.py files
    init_files = [
        "data/__init__.py",
        "data/scripts/__init__.py",
        "data/scripts/extractors/__init__.py",
        "data/scripts/harmonization/__init__.py",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py",
    ]

    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write('"""Project modules."""\n')
            print(f"Created file: {init_file}")

    # Create main training script
    training_script = '''#!/usr/bin/env python3
"""
Main training script for Mistral-7B fine-tuning.
"""

import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/training.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

def setup_model():
    """Initialize the model and LoRA configuration."""
    model_name = "mistralai/Mistral-7B-v0.1"
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    return model

def main():
    """Main training function."""
    logger.info("Starting model training...")
    model = setup_model()
    # Add training loop here
    logger.info("Model training completed.")

if __name__ == "__main__":
    main()
'''

    training_path = "src/training/train.py"
    if not os.path.exists(training_path):
        with open(training_path, "w") as f:
            f.write(training_script)
        print(f"Created file: {training_path}")

    # Create data preparation script
    data_prep_script = '''#!/usr/bin/env python3
"""
Data preparation script for fine-tuning.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/data_prep.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

def prepare_instruction_data() -> List[Dict[str, Any]]:
    """Prepare instruction data for fine-tuning."""
    instructions = []
    
    # Add your instruction preparation logic here
    # Example format:
    # {
    #     "instruction": "User query about emission factors",
    #     "input": "Additional context if needed",
    #     "output": "Expert response with emission factor data",
    #     "metadata": {
    #         "regions": ["USA", "EU"],
    #         "entity_types": ["product", "sector"],
    #         "difficulty": "basic",
    #         "sources": ["Agribalyse_3.1"]
    #     }
    # }
    
    return instructions

def main():
    """Main data preparation function."""
    logger.info("Starting data preparation...")
    instructions = prepare_instruction_data()
    
    # Save prepared data
    output_path = Path("data/processed/instructions.json")
    with open(output_path, "w") as f:
        json.dump(instructions, f, indent=2)
    
    logger.info(f"Data preparation completed. Saved {len(instructions)} instructions.")

if __name__ == "__main__":
    main()
'''

    data_prep_path = "src/data/prepare_data.py"
    if not os.path.exists(data_prep_path):
        with open(data_prep_path, "w") as f:
            f.write(data_prep_script)
        print(f"Created file: {data_prep_path}")

    # Create evaluation script
    eval_script = '''#!/usr/bin/env python3
"""
Evaluation script for the fine-tuned model.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data/logs/evaluation.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

def evaluate_model(model_path: str, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Evaluate model performance on test data."""
    metrics = {
        "precision_at_3": 0.0,
        "mape": 0.0,
        "hallucination_rate": 0.0,
        "source_attribution_accuracy": 0.0
    }
    
    # Add your evaluation logic here
    
    return metrics

def main():
    """Main evaluation function."""
    logger.info("Starting model evaluation...")
    
    # Load test data
    test_data_path = Path("data/processed/test_data.json")
    with open(test_data_path) as f:
        test_data = json.load(f)
    
    # Evaluate model
    model_path = "models/final"
    metrics = evaluate_model(model_path, test_data)
    
    # Log results
    logger.info("Evaluation results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
'''

    eval_path = "src/evaluation/evaluate.py"
    if not os.path.exists(eval_path):
        with open(eval_path, "w") as f:
            f.write(eval_script)
        print(f"Created file: {eval_path}")

    # Create configuration file
    config_content = """{
    "model": {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    },
    "training": {
        "learning_rate": 3e-4,
        "num_epochs": 3,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "max_sequence_length": 2048,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01
    },
    "data": {
        "train_path": "data/processed/train_data.json",
        "val_path": "data/processed/val_data.json",
        "test_path": "data/processed/test_data.json"
    }
}
"""

    config_path = "configs/training_config.json"
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(config_content)
        print(f"Created file: {config_path}")

    # Create requirements.txt
    requirements_content = """torch>=2.0.0
transformers>=4.34.0
peft>=0.5.0
accelerate>=0.21.0
datasets>=2.14.0
evaluate>=0.4.0
wandb>=0.15.0
tensorboard>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
"""

    requirements_path = "requirements.txt"
    if not os.path.exists(requirements_path):
        with open(requirements_path, "w") as f:
            f.write(requirements_content)
        print(f"Created file: {requirements_path}")

    # Create main README.md
    readme_content = """# Carbon Emission Factor Advisor

This project implements a fine-tuned Mistral-7B model for accurate emission factor recommendations.

## Project Structure

- `data/`: Data processing pipeline
  - `raw/`: Raw data files
  - `interim/`: Intermediate processed data
  - `processed/`: Final processed datasets
  - `scripts/`: Data processing scripts
  - `logs/`: Log files
  - `documentation/`: Documentation

- `models/`: Model artifacts
  - `checkpoints/`: Training checkpoints
  - `final/`: Final trained models

- `notebooks/`: Jupyter notebooks for analysis
- `src/`: Source code
  - `data/`: Data preparation modules
  - `models/`: Model architecture
  - `training/`: Training scripts
  - `evaluation/`: Evaluation scripts
  - `utils/`: Utility functions
- `tests/`: Test files
- `configs/`: Configuration files

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare data:
   ```bash
   python src/data/prepare_data.py
   ```

4. Train model:
   ```bash
   python src/training/train.py
   ```

5. Evaluate model:
   ```bash
   python src/evaluation/evaluate.py
   ```

## Performance Metrics

- Precision@3: >85%
- MAPE: <5%
- Hallucination Rate: <1%
- Response Latency: <3 seconds
- Source Attribution Accuracy: >95%

## Documentation

Detailed documentation can be found in the `data/documentation/` directory.
"""

    readme_path = "README.md"
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"Created file: {readme_path}")


if __name__ == "__main__":
    create_project_structure()
    print("\nProject structure and files created successfully!")
