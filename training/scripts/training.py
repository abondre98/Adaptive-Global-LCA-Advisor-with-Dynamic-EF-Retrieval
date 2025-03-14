import torch
import wandb
from transformers import Trainer, TrainingArguments


def setup_trainer(model, tokenizer, train_data, val_data, config):
    """Set up the trainer with appropriate configuration."""
    # Initialize wandb
    wandb.init(project="carbon-ef-mistral", name="fine-tuning-run-1")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=3,
        per_device_train_batch_size=config["micro_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        warmup_steps=100,
        max_steps=config["train_steps"],
        report_to="wandb",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data["train"],
        eval_dataset=val_data["train"],
        tokenizer=tokenizer,
    )

    return trainer


def generate_recommendation(model, tokenizer, prompt):
    """Generate a recommendation using the fine-tuned model."""
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        num_return_sequences=1,
        do_sample=True,
        top_p=0.95,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_model(model, tokenizer):
    """Evaluate the model on sample queries."""
    test_queries = [
        "What is the emission factor for electricity consumption in California, USA?",
        "What is the carbon footprint of wheat production in France?",
        "What is the emission factor for steel manufacturing in China?",
        "What is the carbon intensity of natural gas consumption in Germany?",
    ]

    results = []
    for query in test_queries:
        response = generate_recommendation(model, tokenizer, query)
        results.append({"query": query, "response": response})

    return results


def save_model(model, tokenizer, output_dir):
    """Save the fine-tuned model and tokenizer."""
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
