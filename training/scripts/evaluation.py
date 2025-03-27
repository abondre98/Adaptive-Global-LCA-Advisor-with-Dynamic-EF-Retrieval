#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for fine-tuned Mistral-7B model.
This script evaluates the performance of the fine-tuned model
on emission factor queries as specified in the Milestone2_PRD.md.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training/logs/evaluation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("training/logs", exist_ok=True)


@dataclass
class EvaluationArguments:
    """Arguments for model evaluation."""

    model_name_or_path: str = field(
        default="training/models",
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
    test_file: str = field(
        default="training/data/instructions_val.json",
        metadata={"help": "Path to the test data JSON file"},
    )
    output_dir: str = field(
        default="training/evaluation",
        metadata={"help": "Directory to store evaluation results"},
    )
    prompt_template: str = field(
        default="<s>[INST] {instruction} {input} [/INST]",
        metadata={"help": "Template for formatting instructions into prompts"},
    )
    max_new_tokens: int = field(
        default=512, metadata={"help": "Maximum number of new tokens to generate"}
    )
    batch_size: int = field(default=4, metadata={"help": "Batch size for evaluation"})


def format_prompt(example: Dict[str, Any], template: str) -> str:
    """Format an example into a prompt using the template.

    Args:
        example: Dictionary containing instruction and input
        template: Template string for formatting

    Returns:
        Formatted prompt
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")

    # If input is empty, adjust the format to avoid extra spaces
    if not input_text:
        formatted_prompt = template.format(
            instruction=instruction, input=""  # Empty input
        )
    else:
        formatted_prompt = template.format(
            instruction=instruction,
            input=f"\n{input_text}",  # Add a newline before input if present
        )

    return formatted_prompt


def extract_numeric_value(text: str) -> Optional[float]:
    """Extract a numeric value from a text response.

    Args:
        text: Response text

    Returns:
        Extracted numeric value or None if not found
    """
    import re

    # Look for patterns like "X kg CO2e" or "X kg CO2e/kg"
    patterns = [
        r"(\d+(?:\.\d+)?)\s*kg\s*CO2e",
        r"(\d+(?:\.\d+)?)\s*kg\s*CO2e/kg",
        r"(\d+(?:\.\d+)?)\s*kg\s*CO2e/kWh",
        r"(\d+(?:\.\d+)?)\s*kg\s*CO2e/km",
        r"(\d+(?:\.\d+)?)\s*kg\s*CO2e/USD",
        r"(\d+(?:\.\d+)?)\s*tons?\s*CO2e",
        r"(\d+(?:\.\d+)?)\s*g\s*CO2e",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))

    # If specific patterns fail, try to find any number
    numbers = re.findall(r"(\d+(?:\.\d+)?)", text)
    if numbers:
        return float(numbers[0])

    return None


def extract_source(text: str) -> Optional[str]:
    """Extract a source citation from a text response.

    Args:
        text: Response text

    Returns:
        Extracted source or None if not found
    """
    import re

    # Look for patterns indicating a source
    patterns = [
        r"from\s+(\w+(?:[-_.]\w+)*)",
        r"sourced\s+from\s+(\w+(?:[-_.]\w+)*)",
        r"according\s+to\s+(\w+(?:[-_.]\w+)*)",
        r"based\s+on\s+(\w+(?:[-_.]\w+)*)",
        r"source:\s+(\w+(?:[-_.]\w+)*)",
        r"data\s+from\s+(\w+(?:[-_.]\w+)*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def calculate_metrics(
    predictions: List[str], references: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Calculate evaluation metrics.

    Args:
        predictions: List of model predictions
        references: List of reference examples

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Extract numeric values from predictions and references
    pred_values = []
    ref_values = []
    value_errors = []

    # Extract sources from predictions and references
    pred_sources = []
    ref_sources = []
    source_errors = []

    # Check for hallucinations
    hallucinations = []

    for pred, ref in zip(predictions, references):
        # Process numeric values
        pred_value = extract_numeric_value(pred)
        ref_metadata = ref.get("metadata", {})
        ref_output = ref.get("output", "")
        ref_value = extract_numeric_value(ref_output)

        if pred_value is not None and ref_value is not None:
            pred_values.append(pred_value)
            ref_values.append(ref_value)

            # Calculate error
            if ref_value != 0:
                error = abs((pred_value - ref_value) / ref_value) * 100
                value_errors.append(error)

        # Process source attributions
        pred_source = extract_source(pred)
        ref_source = extract_source(ref_output)

        if pred_source is not None and ref_source is not None:
            pred_sources.append(pred_source.lower())
            ref_sources.append(ref_source.lower())

            # Check source accuracy
            source_match = pred_source.lower() == ref_source.lower()
            source_errors.append(0 if source_match else 1)

        # Check for hallucinations (mentioned sources that don't exist in reference)
        if pred_source is not None and ref_source is not None:
            ref_sources_list = ref_metadata.get("sources", [])
            if pred_source not in [s.lower() for s in ref_sources_list]:
                hallucinations.append(1)
            else:
                hallucinations.append(0)

    # Calculate value metrics
    if value_errors:
        metrics["mape"] = np.mean(value_errors)

    # Calculate source attribution metrics
    if source_errors:
        metrics["source_attribution_accuracy"] = (1 - np.mean(source_errors)) * 100

    # Calculate hallucination rate
    if hallucinations:
        metrics["hallucination_rate"] = np.mean(hallucinations) * 100

    # Return all metrics
    return metrics


class ModelEvaluator:
    """Evaluator for the fine-tuned Mistral-7B model."""

    def __init__(self, args: EvaluationArguments):
        """Initialize model evaluator.

        Args:
            args: Evaluation arguments
        """
        self.args = args

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Load tokenizer
        tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        # Load model
        logger.info(f"Loading model from {args.model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Check if model is a PeftModel and unwrap if needed
        if hasattr(self.model, "is_peft_model") and self.model.is_peft_model:
            logger.info("Model is a PeftModel, using as is")
        else:
            # Check if there's a peft_config.json in the model directory
            peft_config_path = os.path.join(
                args.model_name_or_path, "adapter_config.json"
            )
            if os.path.exists(peft_config_path):
                logger.info(f"Loading PEFT configuration from {peft_config_path}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    args.model_name_or_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )

        # Set up generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        logger.info("Evaluator initialization complete")

    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load the test dataset.

        Returns:
            List of test examples
        """
        logger.info(f"Loading test data from {self.args.test_file}")

        if not os.path.exists(self.args.test_file):
            logger.error(f"Test file not found: {self.args.test_file}")
            return []

        with open(self.args.test_file, "r") as f:
            test_data = json.load(f)

        logger.info(f"Loaded {len(test_data)} test examples")
        return test_data

    def generate_responses(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Generate model responses for a batch of examples.

        Args:
            examples: List of example dictionaries

        Returns:
            List of generated responses
        """
        responses = []

        # Process examples in batches
        for i in tqdm(
            range(0, len(examples), self.args.batch_size), desc="Generating responses"
        ):
            batch = examples[i : i + self.args.batch_size]

            # Format prompts
            prompts = [
                format_prompt(example, self.args.prompt_template) for example in batch
            ]

            # Tokenize
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                self.model.device
            )

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, generation_config=self.generation_config
                )

            # Decode outputs
            decoded_outputs = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # Extract the generated part (after the prompt)
            for prompt, output in zip(prompts, decoded_outputs):
                response = output[len(prompt) :].strip()
                responses.append(response)

        return responses

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model.

        Returns:
            Dictionary of evaluation metrics
        """
        # Load test data
        test_data = self.load_test_data()
        if not test_data:
            return {}

        # Generate responses
        logger.info("Generating responses...")
        responses = self.generate_responses(test_data)

        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = calculate_metrics(responses, test_data)

        # Save results
        results = {"metrics": metrics, "examples": []}

        for i, (example, response) in enumerate(zip(test_data, responses)):
            results["examples"].append(
                {
                    "id": i,
                    "instruction": example["instruction"],
                    "input": example["input"],
                    "expected_output": example["output"],
                    "model_output": response,
                    "metadata": example.get("metadata", {}),
                }
            )

        # Save to file
        output_file = os.path.join(self.args.output_dir, "evaluation_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to {output_file}")
        logger.info(f"Metrics: {metrics}")

        return metrics

    def generate_evaluation_report(self, metrics: Dict[str, float]) -> None:
        """Generate a human-readable evaluation report.

        Args:
            metrics: Evaluation metrics
        """
        report_file = os.path.join(self.args.output_dir, "evaluation_report.md")

        with open(report_file, "w") as f:
            f.write("# Mistral-7B Fine-Tuning Evaluation Report\n\n")

            # Metrics section
            f.write("## Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")

            for metric, value in metrics.items():
                f.write(f"| {metric} | {value:.2f} |\n")

            # Load detailed results
            results_file = os.path.join(self.args.output_dir, "evaluation_results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as rf:
                    results = json.load(rf)

                # Example outputs section
                f.write("\n## Example Outputs\n\n")

                # Select a few examples to display
                examples = results.get("examples", [])
                display_count = min(5, len(examples))

                for i in range(display_count):
                    example = examples[i]

                    f.write(f"### Example {i+1}\n\n")
                    f.write(f"**Instruction**: {example['instruction']}\n\n")

                    if example["input"]:
                        f.write(f"**Input**: {example['input']}\n\n")

                    f.write(f"**Expected Output**: {example['expected_output']}\n\n")
                    f.write(f"**Model Output**: {example['model_output']}\n\n")

                    # Add separator between examples
                    if i < display_count - 1:
                        f.write("---\n\n")

        logger.info(f"Evaluation report saved to {report_file}")


def parse_args():
    """Parse command line arguments.

    Returns:
        EvaluationArguments
    """
    parser = HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]

    return args


def main():
    """Main function to run the evaluation."""
    args = parse_args()

    evaluator = ModelEvaluator(args)
    metrics = evaluator.evaluate()
    evaluator.generate_evaluation_report(metrics)

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
