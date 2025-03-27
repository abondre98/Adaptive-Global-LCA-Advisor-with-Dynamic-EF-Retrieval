#!/usr/bin/env python3
"""
Mistral-7B Fine-tuning Script for Emission Factor Recommendations
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the training environment."""
    # Mount Google Drive (if in Colab)
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        logger.info("Google Drive mounted successfully")
    except ImportError:
        logger.info("Not running in Google Colab")

    # Install required packages
    os.system(
        "pip install -q torch>=2.0.0 transformers>=4.34.0 peft>=0.5.0 accelerate>=0.21.0 bitsandbytes>=0.40.0 trl>=0.7.1 tensorboard>=2.14.0 datasets>=2.14.0 evaluate>=0.4.0 tqdm>=4.66.1 pandas>=2.1.0 matplotlib>=3.7.2 seaborn>=0.12.2 sentencepiece>=0.1.99 scipy>=1.11.2 scikit-learn>=1.3.0 einops>=0.6.1 wandb>=0.15.10"
    )


def setup_model_and_tokenizer():
    """Set up the model and tokenizer with LoRA configuration."""
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    return model, tokenizer, lora_config, MODEL_NAME


def prepare_data():
    """Load and prepare the training data."""
    # Load training and validation data
    train_data = load_dataset(
        "json", data_files="training/data/instructions_train.json"
    )
    val_data = load_dataset("json", data_files="training/data/instructions_val.json")

    # Format instruction template
    def format_instruction(example):
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]

        if input_text:
            formatted = f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output} </s>"
        else:
            formatted = f"<s>[INST] {instruction} [/INST] {output} </s>"

        return {"text": formatted}

    # Apply formatting
    train_data = train_data.map(format_instruction)
    val_data = val_data.map(format_instruction)

    return train_data, val_data


def setup_training(model, tokenizer, train_data, val_data):
    """Set up the training configuration and trainer."""
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        warmup_steps=100,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data["train"],
        eval_dataset=val_data["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    return trainer, training_args


def train_model(trainer):
    """Train the model."""
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model("./final_model")
    logger.info("Training completed and model saved.")


def evaluate_model(trainer):
    """Evaluate the model on test data."""
    # Load test data
    test_data = load_dataset("json", data_files="training/data/instructions_test.json")
    test_data = test_data.map(format_instruction)

    # Evaluate model
    logger.info("Starting evaluation...")
    eval_results = trainer.evaluate(test_data["train"])

    # Print evaluation results
    print("Evaluation Results:")
    print(eval_results)

    return eval_results


def save_model(model, tokenizer, lora_config, training_args, eval_results, MODEL_NAME):
    """Save the model and configurations."""
    # Save model to Google Drive
    DRIVE_PATH = "/content/drive/MyDrive/carbon_ef_model"
    os.makedirs(DRIVE_PATH, exist_ok=True)

    # Save model and tokenizer
    model.save_pretrained(f"{DRIVE_PATH}/model")
    tokenizer.save_pretrained(f"{DRIVE_PATH}/tokenizer")

    # Save training configuration
    with open(f"{DRIVE_PATH}/training_config.json", "w") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "lora_config": lora_config.to_dict(),
                "training_args": training_args.to_dict(),
                "eval_results": eval_results,
            },
            f,
            indent=2,
        )

    logger.info(f"Model and configurations saved to {DRIVE_PATH}")


def main():
    """Main function to run the fine-tuning process."""
    # Set up environment
    setup_environment()

    # Set up model and tokenizer
    model, tokenizer, lora_config, MODEL_NAME = setup_model_and_tokenizer()

    # Prepare data
    train_data, val_data = prepare_data()

    # Set up training
    trainer, training_args = setup_training(model, tokenizer, train_data, val_data)

    # Train model
    train_model(trainer)

    # Evaluate model
    eval_results = evaluate_model(trainer)

    # Save model and configurations
    save_model(model, tokenizer, lora_config, training_args, eval_results, MODEL_NAME)


if __name__ == "__main__":
    main()
