#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training loop script for Mistral-7B fine-tuning.
This script implements the training loop for fine-tuning
the Mistral-7B model with LoRA adapters as specified in the Milestone2_PRD.md.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from model_configuration import LoraArguments, ModelArguments, ModelConfigurator
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training/logs/training_loop.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("training/logs", exist_ok=True)


@dataclass
class DataArguments:
    """Arguments for data configuration."""

    train_file: str = field(
        default="training/data/instructions_train.json",
        metadata={"help": "Path to the training data JSON file"},
    )
    validation_file: str = field(
        default="training/data/instructions_val.json",
        metadata={"help": "Path to the validation data JSON file"},
    )
    prompt_template: str = field(
        default="<s>[INST] {instruction} {input} [/INST] {output} </s>",
        metadata={"help": "Template for formatting instructions into prompts"},
    )


@dataclass
class ExtendedTrainingArguments(TrainingArguments):
    """Extended training arguments with additional parameters."""

    output_dir: str = field(
        default="training/models",
        metadata={"help": "Directory to store the trained model"},
    )
    learning_rate: float = field(
        default=3e-4, metadata={"help": "Initial learning rate"}
    )
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=4, metadata={"help": "Number of gradient accumulation steps"}
    )
    warmup_ratio: float = field(
        default=0.1, metadata={"help": "Ratio of warmup steps to total training steps"}
    )
    lr_scheduler_type: str = field(
        default="cosine", metadata={"help": "Learning rate scheduler type"}
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "Weight decay to apply"}
    )
    fp16: bool = field(default=True, metadata={"help": "Whether to use fp16 training"})
    logging_dir: str = field(
        default="training/logs", metadata={"help": "Directory for Tensorboard logs"}
    )
    logging_steps: int = field(
        default=50, metadata={"help": "Number of steps between logging"}
    )
    save_steps: int = field(
        default=500, metadata={"help": "Number of steps between model checkpoints"}
    )
    eval_steps: int = field(
        default=500, metadata={"help": "Number of steps between evaluations"}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Whether to load the best model at the end of training"},
    )
    metric_for_best_model: str = field(
        default="eval_loss", metadata={"help": "Metric to use for best model selection"}
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "Whether higher metric is better for model selection"},
    )
    save_total_limit: int = field(
        default=3, metadata={"help": "Maximum number of checkpoints to keep"}
    )
    remove_unused_columns: bool = field(
        default=True,
        metadata={"help": "Whether to remove unused columns from the dataset"},
    )


class InstructionDatasetProcessor:
    """Processor for instruction-based datasets."""

    def __init__(self, data_args: DataArguments, tokenizer):
        """Initialize dataset processor.

        Args:
            data_args: Data configuration arguments
            tokenizer: Tokenizer for the model
        """
        self.data_args = data_args
        self.tokenizer = tokenizer

        logger.info(
            f"Initializing dataset processor with template: {data_args.prompt_template}"
        )

    def load_datasets(self):
        """Load the training and validation datasets.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        data_files = {}

        # Check if train/validation files exist
        if os.path.exists(self.data_args.train_file):
            data_files["train"] = self.data_args.train_file
            logger.info(f"Training data file found: {self.data_args.train_file}")
        else:
            logger.warning(f"Training data file not found: {self.data_args.train_file}")

        if os.path.exists(self.data_args.validation_file):
            data_files["validation"] = self.data_args.validation_file
            logger.info(f"Validation data file found: {self.data_args.validation_file}")
        else:
            logger.warning(
                f"Validation data file not found: {self.data_args.validation_file}"
            )

        # Load the datasets
        raw_datasets = load_dataset("json", data_files=data_files)

        # Log dataset statistics
        if "train" in raw_datasets:
            logger.info(f"Loaded {len(raw_datasets['train'])} training examples")

        if "validation" in raw_datasets:
            logger.info(f"Loaded {len(raw_datasets['validation'])} validation examples")

        return raw_datasets

    def format_prompt(self, example):
        """Format an example into a prompt using the template.

        Args:
            example: Dictionary containing instruction, input, and output

        Returns:
            Formatted prompt
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        response = example.get("output", "")

        # If input is empty, adjust the format to avoid extra spaces
        if not input_text:
            formatted_prompt = self.data_args.prompt_template.format(
                instruction=instruction, input="", output=response  # Empty input
            )
        else:
            formatted_prompt = self.data_args.prompt_template.format(
                instruction=instruction,
                input=f"\n{input_text}",  # Add a newline before input if present
                output=response,
            )

        return formatted_prompt

    def preprocess_function(self, examples):
        """Preprocess a batch of examples.

        Args:
            examples: Batch of examples

        Returns:
            Preprocessed examples
        """
        # Format prompts
        prompts = [
            self.format_prompt(
                {"instruction": instruction, "input": input_text, "output": output}
            )
            for instruction, input_text, output in zip(
                examples["instruction"], examples["input"], examples["output"]
            )
        ]

        # Tokenize
        tokenized_inputs = self.tokenizer(
            prompts, padding=True, truncation=True, max_length=2048, return_tensors="pt"
        )

        # Set the labels equal to the inputs for causal language modeling
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()

        return tokenized_inputs

    def process_datasets(self):
        """Process the datasets for training.

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        raw_datasets = self.load_datasets()

        # Process the datasets
        processed_datasets = {}
        for split in raw_datasets:
            processed_datasets[split] = raw_datasets[split].map(
                self.preprocess_function,
                batched=True,
                remove_columns=raw_datasets[split].column_names,
                desc=f"Processing {split} dataset",
            )

        # Log processed dataset statistics
        for split in processed_datasets:
            logger.info(
                f"Processed {len(processed_datasets[split])} examples for {split}"
            )

        return processed_datasets.get("train"), processed_datasets.get("validation")


class ModelTrainer:
    """Trainer for Mistral-7B with LoRA adapters."""

    def __init__(
        self,
        model_args: ModelArguments,
        lora_args: LoraArguments,
        training_args: ExtendedTrainingArguments,
        data_args: DataArguments,
    ):
        """Initialize model trainer.

        Args:
            model_args: Model configuration arguments
            lora_args: LoRA configuration arguments
            training_args: Training configuration arguments
            data_args: Data configuration arguments
        """
        self.model_args = model_args
        self.lora_args = lora_args
        self.training_args = training_args
        self.data_args = data_args

        # Configure the model and tokenizer
        logger.info("Configuring model and tokenizer...")
        configurator = ModelConfigurator(model_args, lora_args, training_args)
        self.model, self.tokenizer = configurator.configure()

        # Process the datasets
        logger.info("Processing datasets...")
        processor = InstructionDatasetProcessor(data_args, self.tokenizer)
        self.train_dataset, self.eval_dataset = processor.process_datasets()

        # Create data collator
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, padding=True
        )

        logger.info("Trainer initialization complete")

    def setup_trainer(self):
        """Setup the Hugging Face Trainer.

        Returns:
            Configured Trainer
        """
        logger.info("Setting up trainer...")

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        return trainer

    def train(self):
        """Train the model.

        Returns:
            Training results
        """
        logger.info("Starting training...")

        trainer = self.setup_trainer()

        # Handle resumption from checkpoint
        checkpoint = None
        if os.path.isdir(self.training_args.output_dir):
            # Check if there's a checkpoint to resume from
            if any(
                f.startswith("checkpoint-")
                for f in os.listdir(self.training_args.output_dir)
            ):
                checkpoint_dirs = [
                    d
                    for d in os.listdir(self.training_args.output_dir)
                    if d.startswith("checkpoint-")
                ]
                latest_checkpoint = max(
                    checkpoint_dirs, key=lambda x: int(x.split("-")[1])
                )
                checkpoint = os.path.join(
                    self.training_args.output_dir, latest_checkpoint
                )
                logger.info(f"Resuming training from checkpoint: {checkpoint}")

        # Train the model
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # Save the final model
        logger.info("Training complete, saving final model...")
        trainer.save_model()

        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        return train_result


def parse_args():
    """Parse command line arguments.

    Returns:
        Tuple of (ModelArguments, LoraArguments, ExtendedTrainingArguments, DataArguments)
    """
    parser = HfArgumentParser(
        (ModelArguments, LoraArguments, ExtendedTrainingArguments, DataArguments)
    )
    model_args, lora_args, training_args, data_args = (
        parser.parse_args_into_dataclasses()
    )

    return model_args, lora_args, training_args, data_args


def main():
    """Main function to run the training loop."""
    model_args, lora_args, training_args, data_args = parse_args()

    # Create output directories
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(training_args.logging_dir, exist_ok=True)

    # Initialize and run trainer
    trainer = ModelTrainer(model_args, lora_args, training_args, data_args)
    train_result = trainer.train()

    logger.info(f"Training complete, final metrics: {train_result.metrics}")


if __name__ == "__main__":
    main()
