#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for Mistral-7B fine-tuning.
This script orchestrates the entire fine-tuning process from data preparation
to evaluation as specified in the Milestone2_PRD.md.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training/logs/main.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("training/logs", exist_ok=True)


def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Mistral-7B fine-tuning pipeline")

    # Environment setup
    parser.add_argument(
        "--use_colab",
        action="store_true",
        help="Whether to use Google Colab integration",
    )
    parser.add_argument(
        "--mount_drive", action="store_true", help="Whether to mount Google Drive"
    )

    # Data preparation
    parser.add_argument(
        "--prepare_data", action="store_true", help="Whether to run data preparation"
    )
    parser.add_argument(
        "--neo4j_uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j database URI",
    )
    parser.add_argument(
        "--neo4j_username", type=str, default="neo4j", help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j_password", type=str, default="password", help="Neo4j password"
    )
    parser.add_argument(
        "--use_synthetic_data",
        action="store_true",
        help="Whether to use synthetic data instead of Neo4j",
    )
    parser.add_argument(
        "--synthetic_data_count",
        type=int,
        default=2000,
        help="Number of synthetic data points to generate",
    )

    # Model configuration
    parser.add_argument(
        "--configure_model",
        action="store_true",
        help="Whether to run model configuration",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Model name or path",
    )
    parser.add_argument(
        "--use_4bit", action="store_true", help="Whether to use 4-bit quantization"
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="float16",
        help="Compute dtype for 4-bit quantization",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=64, help="Rank of LoRA adapters"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="Alpha parameter for LoRA scaling"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout probability for LoRA layers",
    )

    # Training
    parser.add_argument(
        "--train_model", action="store_true", help="Whether to run model training"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="training/data/instructions_train.json",
        help="Path to training data",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default="training/data/instructions_val.json",
        help="Path to validation data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/models",
        help="Output directory for model",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_train_epochs", type=float, default=3.0, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Ratio of warmup steps to total training steps",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Number of steps between logging"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Number of steps between model checkpoints",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of steps between evaluations",
    )

    # Evaluation
    parser.add_argument(
        "--evaluate_model", action="store_true", help="Whether to run model evaluation"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="training/data/instructions_val.json",
        help="Path to test data",
    )
    parser.add_argument(
        "--evaluation_output_dir",
        type=str,
        default="training/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
    )

    # Process stages
    parser.add_argument(
        "--run_all", action="store_true", help="Whether to run all stages"
    )

    return parser.parse_args()


def setup_environment(args):
    """Set up the environment based on arguments.

    Args:
        args: Command line arguments

    Returns:
        Environment information
    """
    env_info = {}

    if args.use_colab:
        logger.info("Setting up Google Colab environment...")

        try:
            from colab_integration import is_colab_runtime, setup_colab_environment

            if is_colab_runtime():
                env_info = setup_colab_environment()
            else:
                logger.warning(
                    "Not running in Google Colab but --use_colab was specified"
                )
        except ImportError:
            logger.error("colab_integration module not found")
    else:
        logger.info("Setting up local environment...")

        # Create directory structure
        directories = {
            "base": "training",
            "data": "training/data",
            "models": "training/models",
            "logs": "training/logs",
            "scripts": "training/scripts",
            "evaluation": "training/evaluation",
        }

        for name, path in directories.items():
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")

        env_info["project_structure"] = directories

    return env_info


def prepare_data(args):
    """Prepare data for fine-tuning.

    Args:
        args: Command line arguments
    """
    logger.info("Preparing data...")

    # Use data_preparation.py script with arguments
    if args.use_synthetic_data:
        logger.info(
            f"Using synthetic data with {args.synthetic_data_count} data points"
        )

        # Create synthetic data
        sys.path.append("training/scripts")
        from data_preparation import InstructionGenerator, create_synthetic_ef_data

        # Generate synthetic data
        synthetic_data = create_synthetic_ef_data(args.synthetic_data_count)

        # Generate instruction dataset
        generator = InstructionGenerator(synthetic_data)
        instructions = generator.generate_full_instruction_set()

        # Save instruction dataset
        generator.save_instruction_set(instructions, "training/data/instructions.json")
    else:
        logger.info(f"Extracting data from Neo4j at {args.neo4j_uri}")

        # Import the data preparation module
        sys.path.append("training/scripts")
        try:
            from data_preparation import InstructionGenerator, Neo4jDataExtractor

            # Extract data from Neo4j
            extractor = Neo4jDataExtractor(
                args.neo4j_uri, args.neo4j_username, args.neo4j_password
            )
            emission_factors = extractor.get_emission_factors()
            extractor.close()

            # Generate instruction dataset
            generator = InstructionGenerator(emission_factors)
            instructions = generator.generate_full_instruction_set()

            # Save instruction dataset
            generator.save_instruction_set(
                instructions, "training/data/instructions.json"
            )
        except Exception as e:
            logger.error(f"Error extracting data from Neo4j: {e}")
            logger.info("Falling back to synthetic data...")

            # Fall back to synthetic data
            from data_preparation import InstructionGenerator, create_synthetic_ef_data

            # Generate synthetic data
            synthetic_data = create_synthetic_ef_data(args.synthetic_data_count)

            # Generate instruction dataset
            generator = InstructionGenerator(synthetic_data)
            instructions = generator.generate_full_instruction_set()

            # Save instruction dataset
            generator.save_instruction_set(
                instructions, "training/data/instructions.json"
            )

    logger.info("Data preparation complete")


def configure_model(args):
    """Configure the model for fine-tuning.

    Args:
        args: Command line arguments
    """
    logger.info("Configuring model...")

    # Use model_configuration.py script with arguments
    sys.path.append("training/scripts")
    from model_configuration import (
        LoraArguments,
        ModelArguments,
        ModelConfigurator,
        TrainingArguments,
    )

    # Create arguments
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        use_4bit=args.use_4bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
    )

    lora_args = LoraArguments(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    # Configure model
    configurator = ModelConfigurator(model_args, lora_args, training_args)
    model, tokenizer = configurator.configure()

    logger.info("Model configuration complete")


def train_model(args):
    """Train the model.

    Args:
        args: Command line arguments
    """
    logger.info("Training model...")

    # Use training_loop.py script with arguments
    sys.path.append("training/scripts")
    from training_loop import (
        DataArguments,
        ExtendedTrainingArguments,
        LoraArguments,
        ModelArguments,
        ModelTrainer,
    )

    # Create arguments
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        use_4bit=args.use_4bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
    )

    lora_args = LoraArguments(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    training_args = ExtendedTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        logging_dir="training/logs",
    )

    data_args = DataArguments(
        train_file=args.train_file, validation_file=args.validation_file
    )

    # Train model
    trainer = ModelTrainer(model_args, lora_args, training_args, data_args)
    train_result = trainer.train()

    logger.info("Model training complete")
    logger.info(f"Training metrics: {train_result.metrics}")


def evaluate_model(args):
    """Evaluate the model.

    Args:
        args: Command line arguments
    """
    logger.info("Evaluating model...")

    # Use evaluation.py script with arguments
    sys.path.append("training/scripts")
    from evaluation import EvaluationArguments, ModelEvaluator

    # Create arguments
    eval_args = EvaluationArguments(
        model_name_or_path=args.output_dir,
        test_file=args.test_file,
        output_dir=args.evaluation_output_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # Evaluate model
    evaluator = ModelEvaluator(eval_args)
    metrics = evaluator.evaluate()
    evaluator.generate_evaluation_report(metrics)

    logger.info("Model evaluation complete")
    logger.info(f"Evaluation metrics: {metrics}")


def main():
    """Main function to run the fine-tuning pipeline."""
    args = parse_args()

    # If run_all is specified, set all stage flags to True
    if args.run_all:
        args.prepare_data = True
        args.configure_model = True
        args.train_model = True
        args.evaluate_model = True

    # Set up environment
    env_info = setup_environment(args)

    # Run the pipeline stages based on arguments
    if args.prepare_data:
        prepare_data(args)

    if args.configure_model:
        configure_model(args)

    if args.train_model:
        train_model(args)

    if args.evaluate_model:
        evaluate_model(args)

    logger.info("Fine-tuning pipeline complete")


if __name__ == "__main__":
    main()
