#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model configuration script for Mistral-7B fine-tuning.
This script sets up the Mistral-7B model with LoRA adapters
as specified in the Milestone2_PRD.md.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training/logs/model_configuration.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("training/logs", exist_ok=True)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"
        },
    )
    use_4bit: bool = field(
        default=True, metadata={"help": "Whether to use 4-bit quantization"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16", metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type for 4-bit quantization (fp4 or nf4)"},
    )
    use_nested_quant: bool = field(
        default=False, metadata={"help": "Whether to use nested quantization"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""

    lora_rank: int = field(default=64, metadata={"help": "Rank of LoRA adapters"})
    lora_alpha: int = field(
        default=16, metadata={"help": "Alpha parameter for LoRA scaling"}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "Dropout probability for LoRA layers"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        metadata={"help": "List of module names to apply LoRA adapters to"},
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"},
    )


@dataclass
class TrainingArguments:
    """Arguments for training configuration."""

    output_dir: str = field(
        default="training/models",
        metadata={"help": "Directory to store the trained model"},
    )
    learning_rate: float = field(
        default=3e-4, metadata={"help": "Initial learning rate"}
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU for training"}
    )
    gradient_accumulation_steps: int = field(
        default=4, metadata={"help": "Number of gradient accumulation steps"}
    )
    warmup_ratio: float = field(
        default=0.1, metadata={"help": "Ratio of warmup steps to total training steps"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Maximum sequence length for training"}
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "Weight decay to apply"}
    )


class ModelConfigurator:
    """Class to configure the model for fine-tuning."""

    def __init__(
        self,
        model_args: ModelArguments,
        lora_args: LoraArguments,
        training_args: TrainingArguments,
    ):
        """Initialize model configurator.

        Args:
            model_args: Model configuration arguments
            lora_args: LoRA configuration arguments
            training_args: Training configuration arguments
        """
        self.model_args = model_args
        self.lora_args = lora_args
        self.training_args = training_args

        # Create output directory if it doesn't exist
        os.makedirs(training_args.output_dir, exist_ok=True)

        logger.info(
            f"Initializing model configurator with model: {model_args.model_name_or_path}"
        )

    def setup_tokenizer(self):
        """Setup the tokenizer.

        Returns:
            Configured tokenizer
        """
        tokenizer_name = (
            self.model_args.tokenizer_name_or_path or self.model_args.model_name_or_path
        )

        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True, padding_side="right"
        )

        # Enable padding with the PAD token
        tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
        return tokenizer

    def setup_quantization_config(self):
        """Setup the quantization configuration.

        Returns:
            BitsAndBytesConfig for quantization
        """
        if not self.model_args.use_4bit:
            logger.info("Quantization disabled, using full precision")
            return None

        logger.info(
            f"Setting up 4-bit quantization with {self.model_args.bnb_4bit_quant_type}"
        )

        compute_dtype = getattr(torch, self.model_args.bnb_4bit_compute_dtype)

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.model_args.use_nested_quant,
        )

    def setup_model(self):
        """Setup the model with quantization.

        Returns:
            Configured model
        """
        quantization_config = self.setup_quantization_config()

        logger.info(f"Loading model: {self.model_args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.model_args.use_4bit:
            logger.info("Preparing model for k-bit training")
            model = prepare_model_for_kbit_training(model)

        logger.info(f"Model loaded with {model.num_parameters()} parameters")
        return model

    def setup_lora_config(self):
        """Setup the LoRA configuration.

        Returns:
            LoraConfig for PEFT
        """
        logger.info(
            f"Setting up LoRA with rank={self.lora_args.lora_rank}, alpha={self.lora_args.lora_alpha}"
        )

        return LoraConfig(
            r=self.lora_args.lora_rank,
            lora_alpha=self.lora_args.lora_alpha,
            lora_dropout=self.lora_args.lora_dropout,
            bias=self.lora_args.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.lora_args.target_modules,
        )

    def apply_lora(self, model):
        """Apply LoRA adapters to the model.

        Args:
            model: Base model

        Returns:
            Model with LoRA adapters
        """
        lora_config = self.setup_lora_config()

        logger.info(
            f"Applying LoRA adapters to {len(self.lora_args.target_modules)} target modules"
        )
        peft_model = get_peft_model(model, lora_config)

        logger.info(
            f"LoRA adapters applied, trainable parameters: {peft_model.print_trainable_parameters()}"
        )
        return peft_model

    def configure(self):
        """Configure the model and tokenizer for fine-tuning.

        Returns:
            Tuple of (model, tokenizer)
        """
        tokenizer = self.setup_tokenizer()
        base_model = self.setup_model()
        peft_model = self.apply_lora(base_model)

        # Save model configuration for reference
        if hasattr(peft_model, "config") and hasattr(peft_model.config, "to_dict"):
            model_config = peft_model.config.to_dict()
            logger.info(f"Model configuration: {model_config}")

        return peft_model, tokenizer


def parse_args():
    """Parse command line arguments.

    Returns:
        Tuple of (ModelArguments, LoraArguments, TrainingArguments)
    """
    parser = HfArgumentParser((ModelArguments, LoraArguments, TrainingArguments))
    model_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, lora_args, training_args


def main():
    """Main function to configure the model."""
    model_args, lora_args, training_args = parse_args()

    configurator = ModelConfigurator(model_args, lora_args, training_args)
    model, tokenizer = configurator.configure()

    # Save tokenizer
    tokenizer_output_dir = os.path.join(training_args.output_dir, "tokenizer")
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_output_dir)

    logger.info(
        f"Model and tokenizer configuration complete. Tokenizer saved to {tokenizer_output_dir}"
    )
    logger.info(f"Model is ready for training with {model.num_parameters()} parameters")


if __name__ == "__main__":
    main()
