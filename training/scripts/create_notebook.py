import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title and Prerequisites
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """# Mistral-7B Fine-tuning for Emission Factor Recommendations

This notebook implements the fine-tuning process for the Mistral-7B model to generate emission factor recommendations based on the PRD specifications.

## Prerequisites

1. GPU Runtime in Google Colab
2. Google Drive mounted for saving checkpoints
3. Hugging Face account with access to Mistral-7B-Instruct-v0.2
4. HF_TOKEN in Colab secrets
5. Weights & Biases account for experiment tracking"""
    )
)

# Environment Setup
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 1. Environment Setup

Install required packages and mount Google Drive."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required packages with strict versioning to ensure compatibility
!pip install -q numpy==1.26.4  # Pin to NumPy < 2.0
!pip install -q torch==2.0.1
!pip install -q huggingface_hub==0.17.3
!pip install -q accelerate==0.22.0
!pip install -q transformers==4.32.0
!pip install -q peft==0.5.0
!pip install -q datasets==2.14.6
!pip install -q scipy==1.11.4 wandb==0.15.12 trl==0.7.2
!pip install -q bitsandbytes==0.41.1

# Verify package versions
!pip list | grep -E 'numpy|torch|huggingface_hub|accelerate|transformers|peft'"""
    )
)

# Hugging Face Authentication
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 2. Hugging Face Authentication

Authenticate with Hugging Face and verify access to the model."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """import os
from huggingface_hub import login, HfApi

# Login to Hugging Face
login(token=os.environ.get('HF_TOKEN'))

# Verify access
api = HfApi()
try:
    api.model_info("mistralai/Mistral-7B-Instruct-v0.2")
    print("Successfully authenticated and have access to the model!")
except Exception as e:
    print(f"Error: {e}")"""
    )
)

# Add Weights & Biases setup
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 3. Weights & Biases Setup

Initialize Weights & Biases for experiment tracking."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """import wandb

# Initialize wandb - this will open a browser tab for authentication if not already logged in
# If you're already logged in through the browser, this will use your existing session
wandb.login()

# Define project and experiment details
WANDB_PROJECT = "mistral-carbon-ef"
EXPERIMENT_NAME = "mistral-7b-emission-factors"

# Initialize the wandb run
wandb.init(
    project=WANDB_PROJECT,
    name=EXPERIMENT_NAME,
    config={
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "epochs": 3,
        "learning_rate": 2e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lora_rank": 8,
        "lora_alpha": 16
    }
)

print("Weights & Biases initialized successfully!")"""
    )
)

# Model Configuration - UPDATED to use 16-bit precision and fix meta tensor error
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 4. Model and Tokenizer Configuration

Set up the Mistral-7B model with 16-bit precision and LoRA configuration."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Model configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Explicitly set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fixed model loading to avoid meta tensor error
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map=None,  # Don't use auto device mapping
    trust_remote_code=True,
)
model = model.to(device)  # Explicitly move to device

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Reduced rank for better memory usage
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
print("Model and tokenizer configured successfully!")"""
    )
)

# Data Preparation - UPDATED to create sample data
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 5. Data Preparation

Load and prepare the training data. If the data files don't exist, create sample data for testing."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Clone the repository
!git clone https://github.com/Sbursu/Carbon-EF.git
%cd Carbon-EF

# Create sample data if it doesn't exist
import os
import json
from datasets import load_dataset

# Check for the data directory and create it if it doesn't exist
!mkdir -p training/data

# Define sample data for emission factors
sample_data = [
    {
        "instruction": "Recommend an emission factor for rice production in Asia.",
        "output": "For rice production in Asia, I recommend using an emission factor of 1.46 kg CO2e per kg of rice, based on the Agribalyse database."
    },
    {
        "instruction": "What is the carbon footprint of beef production?",
        "output": "The carbon footprint of beef production is approximately 27 kg CO2e per kg of beef, making it one of the highest emission foods."
    },
    {
        "instruction": "Provide an emission factor for electricity generation from coal.",
        "output": "For electricity generation from coal, the emission factor is approximately 1000 g CO2e per kWh, which is significantly higher than renewable sources."
    }
]

# Create train, validation, and test files if they don't exist
data_files = {
    'training/data/instructions_train.json': sample_data[:2],
    'training/data/instructions_val.json': [sample_data[2]],
    'training/data/instructions_test.json': [sample_data[2]]
}

for file_path, data in data_files.items():
    if not os.path.exists(file_path):
        print(f"Creating sample data file: {file_path}")
        with open(file_path, 'w') as f:
            json.dump(data, f)

# Now load the datasets
try:
    train_data = load_dataset('json', data_files='training/data/instructions_train.json')
    val_data = load_dataset('json', data_files='training/data/instructions_val.json')
    test_data = load_dataset('json', data_files='training/data/instructions_test.json')
    
    # Format instruction template
    def format_instruction(example):
        instruction = example['instruction']
        input_text = example.get('input', '')
        output = example['output']
        
        if input_text:
            formatted = f"<s>[INST] {instruction}\\n\\n{input_text} [/INST] {output} </s>"
        else:
            formatted = f"<s>[INST] {instruction} [/INST] {output} </s>"
        
        return {'text': formatted}

    # Apply formatting
    train_data = train_data.map(format_instruction)
    val_data = val_data.map(format_instruction)
    test_data = test_data.map(format_instruction)

    print(f"Training samples: {len(train_data['train'])}")
    print(f"Validation samples: {len(val_data['train'])}")
    print(f"Test samples: {len(test_data['train'])}")
    
except Exception as e:
    print(f"Error loading datasets: {e}")
    print("Please check that the data files exist and are properly formatted.")"""
    )
)

# Training Configuration - UPDATED to fix deprecation warnings
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 6. Training Configuration

Set up training arguments and initialize the trainer."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Training arguments
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/mistral-ef-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduced batch size for better memory usage
    gradient_accumulation_steps=8,  # Increased gradient accumulation steps
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",  # Fixed deprecated warning
    load_best_model_at_end=True,
    report_to="wandb",  # Log metrics to Weights & Biases
    run_name=EXPERIMENT_NAME,  # Use the same experiment name for wandb
    warmup_ratio=0.1,
    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
    optim="adamw_torch",
    max_grad_norm=0.3
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Initialize trainer with fixed parameters (avoid deprecated warning)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data['train'],
    eval_dataset=val_data['train'],
    data_collator=data_collator
)

print("Training configuration completed!")"""
    )
)

# Training Process
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 7. Training Process

Start the fine-tuning process."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Start training
trainer.train()

# Save the final model
trainer.save_model("/content/drive/MyDrive/mistral-ef-final")"""
    )
)

# Model Evaluation
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 8. Model Evaluation

Evaluate the fine-tuned model on the test set."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Evaluate on test set
test_results = trainer.evaluate(test_data['train'])
print(f"Test results: {test_results}")

# Log test results to wandb
wandb.log({"test_loss": test_results["loss"]})"""
    )
)

# Save and Export
nb.cells.append(
    nbf.v4.new_markdown_cell(
        """## 9. Save and Export

Save the model and tokenizer to Google Drive."""
    )
)

nb.cells.append(
    nbf.v4.new_code_cell(
        """# Save model and tokenizer
model.save_pretrained("/content/drive/MyDrive/mistral-ef-final")
tokenizer.save_pretrained("/content/drive/MyDrive/mistral-ef-final")

# Save training configuration
import json
with open("/content/drive/MyDrive/mistral-ef-final/training_config.json", "w") as f:
    json.dump({
        "model_name": MODEL_NAME,
        "lora_config": lora_config.to_dict(),
        "training_args": training_args.to_dict(),
        "test_results": test_results
    }, f, indent=2)

# Finish wandb run
wandb.finish()

print("Model and configuration saved successfully!")"""
    )
)

# Write the notebook to a file
with open("training/notebooks/mistral_finetuning.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook created successfully!")
