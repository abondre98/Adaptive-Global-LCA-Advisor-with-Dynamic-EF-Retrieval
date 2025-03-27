# Product Requirements Document: Milestone 2 - Mistral-7B Fine-Tuning

## 1. Introduction and Project Context

### 1.1 Purpose

This document outlines the comprehensive requirements for Milestone 2 of the Adaptive Global LCA Advisor project, focusing on fine-tuning the Mistral-7B language model to provide accurate, region-specific emission factor recommendations. This milestone establishes the AI foundation that will power the recommendation system by transferring domain knowledge from our Neo4j graph database into the language model.

### 1.2 Project Overview

The Adaptive Global LCA Advisor aims to develop an AI system that recommends region-specific emission factors (EFs) for accurate carbon accounting. Milestone 1 successfully created a Neo4j knowledge graph with 23,520 emission factor nodes and their relationships. Milestone 2 will leverage this knowledge graph to fine-tune the Mistral-7B model for specialized environmental data retrieval and reasoning.

### 1.3 Scope

This PRD covers:

- Data preparation for fine-tuning
- Instruction set creation methodology
- Google Colab integration for GPU support
- LoRA (Low-Rank Adaptation) configuration and setup
- Fine-tuning workflow and optimization
- Model validation and performance metrics
- Model packaging and distribution

## 2. Business Requirements

### 2.1 Problem Statement

- Traditional retrieval methods lack contextual understanding of emission factors
- Region-specific emission factor selection requires specialized domain knowledge
- Raw text embeddings alone cannot capture the complex relationships in LCA data
- Current LLMs lack specific understanding of emission factor data relationships
- Proper attribution and confidence indication is missing from existing solutions

### 2.2 Success Criteria

- Precision@3 of >85% for emission factor queries (correct answer in top 3 responses)
- Mean Absolute Percentage Error (MAPE) of <5% compared to ground truth values
- Reduction in hallucination rate to <1% for domain-specific queries
- Response latency of <3 seconds on consumer hardware
- Proper source attribution in >95% of responses
- Accurate handling of at least 90% of regional variation queries

## 3. Data Preparation and Instruction Set

### 3.1 Training Data Composition

| Data Category          | Source                 | Volume          | Description                                                     |
| ---------------------- | ---------------------- | --------------- | --------------------------------------------------------------- |
| Knowledge Graph Export | Neo4j database         | ~25,000 entries | Structured data from emission factor knowledge graph            |
| Text Descriptions      | Environmental datasets | ~5,000 entries  | Detailed descriptions of products, processes, and methodologies |
| Expert Instructions    | Subject matter experts | ~1,000 entries  | Guidance on proper emission factor selection and application    |
| User Query Patterns    | Synthetic + real-world | ~2,000 entries  | Representative user queries with varying complexity levels      |

### 3.2 Instruction Format Design

Each instruction will follow this structure:

```
{
    "instruction": "User query about emission factors",
    "input": "Additional context if needed (optional)",
    "output": "Expert-guided response with accurate emission factor data and source attribution",
    "metadata": {
        "regions": ["USA", "EU", "GLB", ...],
        "entity_types": ["product", "sector", "energy", ...],
        "difficulty": "basic|moderate|complex",
        "sources": ["Agribalyse_3.1", "USEEIO_v2.1", ...]
    }
}
```

### 3.3 Instruction Categories and Distribution

| Category              | Description                                        | Percentage | Examples                                                                                        |
| --------------------- | -------------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------- |
| Basic Lookup          | Simple EF retrieval for single entity/region       | 30%        | "What is the emission factor for cement production in the USA?"                                 |
| Regional Comparison   | Compare EFs across regions                         | 25%        | "Compare the emission factor for wheat production between France and the USA."                  |
| Multi-Entity Analysis | Complex queries involving multiple entities        | 15%        | "What are the emission factors for the top 3 contributors to agricultural emissions in Europe?" |
| Methodological        | Questions about adjustment factors and application | 15%        | "How should I adjust the global emission factor for rice production when using it in Thailand?" |
| Edge Cases            | Unusual or challenging scenarios                   | 10%        | "What emission factor should I use for a new plant-based leather alternative?"                  |
| Verification          | Data quality and confidence assessment             | 5%         | "How reliable is the emission factor data for aluminum production in China?"                    |

### 3.4 Data Augmentation and Enhancement

- Implement paraphrasing for query variation (3-5 variants per base question)
- Include multinational scenario queries
- Add temporal variation (recent vs historical data requests)
- Create complex multi-step reasoning examples
- Generate uncertainty handling instructions

## 4. Google Colab Integration

### 4.1 Hardware Requirements

- **GPU Type**: NVIDIA T4 or NVIDIA A100 (preferred)
- **VRAM Required**: Minimum 16GB for Mistral-7B with LoRA
- **Storage**: At least 50GB for model, datasets, and checkpoints
- **Runtime**: High-RAM runtime configuration

### 4.2 Environment Setup

- Python 3.10+ environment
- Key libraries: PyTorch 2.0+, Transformers 4.34+, PEFT 0.5+, Accelerate 0.21+
- Integration with Google Drive for persistent storage
- Checkpointing strategy to accommodate Colab timeouts

### 4.3 Fallback Provisions

- Code for automated checkpoint saving every 500 steps
- State recovery capabilities from saved checkpoints
- Modular training loop that can resume from interruptions
- Backup to alternative compute platforms if needed

## 5. LoRA Configuration and Setup

### 5.1 Base Model Selection

- **Model**: Mistral-7B (latest available version)
- **Variant**: Instruct-tuned variant preferred as base
- **Source**: Hugging Face Hub
- **Tokenizer**: Default Mistral tokenizer with optional domain-specific tokens

### 5.2 LoRA Parameters

| Parameter      | Value                                                                         | Justification                                           |
| -------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------- |
| Rank           | 64 (32 if memory constrained)                                                 | Balance between parameter efficiency and model capacity |
| Alpha          | 16                                                                            | Standard scaling factor for LoRA                        |
| Dropout        | 0.05                                                                          | Prevents overfitting                                    |
| Target Modules | ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] | Targets attention and MLP layers                        |
| Bias           | "none"                                                                        | Avoid adding trainable bias terms                       |

### 5.3 Training Hyperparameters

| Parameter                   | Value                                            | Justification                                        |
| --------------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| Learning Rate               | 3e-4                                             | Standard for LoRA fine-tuning                        |
| Training Epochs             | 3-5                                              | Sufficient for domain adaptation without overfitting |
| Batch Size                  | 8 (dynamically adjusted based on available VRAM) | Maximize GPU utilization                             |
| Gradient Accumulation Steps | 4                                                | Simulate larger batch sizes                          |
| Learning Rate Schedule      | Cosine with warmup (10% of steps)                | Smooth convergence                                   |
| Weight Decay                | 0.01                                             | Reduce overfitting risk                              |
| Max Sequence Length         | 2048 tokens                                      | Balance between context length and memory usage      |

## 6. Fine-Tuning Workflow

### 6.1 Training Process

1. **Stage 1: Initialization**

   - Load base Mistral-7B model with 4-bit quantization
   - Configure LoRA adapters with specified parameters
   - Initialize training dataset and validation split (80/20)

2. **Stage 2: Initial Training**

   - Train for 1 epoch with learning rate warmup
   - Evaluate on validation set
   - Adjust hyperparameters if necessary

3. **Stage 3: Main Training**

   - Continue training for remaining epochs
   - Monitor validation metrics after each epoch
   - Save checkpoint every 500 steps

4. **Stage 4: Optimization**
   - Fine-tune learning rate based on validation performance
   - Apply early stopping if no improvement for 1000 steps
   - Save best-performing model checkpoint

### 6.2 Training Code Structure

The training code will be structured with the following components:

```python
# Main modules
- data_preparation.py      # Dataset loading and preprocessing
- model_configuration.py   # LoRA setup and model initialization
- training_loop.py         # Training execution
- evaluation.py            # Metrics calculation
- checkpointing.py         # Save/load functionality
- colab_integration.py     # Google Colab specific utilities
- main.py                  # Orchestration
```

### 6.3 Monitoring and Logging

- TensorBoard integration for real-time metrics visualization
- Logging of training and validation losses
- Tracking of GPU memory usage
- Periodic sample generation for qualitative assessment
- Automated alerts for training anomalies

## 7. Model Validation and Performance Metrics

### 7.1 Quantitative Metrics

| Metric                      | Target     | Measurement Method                                           |
| --------------------------- | ---------- | ------------------------------------------------------------ |
| Precision@3                 | >85%       | Percentage of queries where correct EF is in top 3 responses |
| MAPE                        | <5%        | Mean absolute percentage error compared to ground truth      |
| Hallucination Rate          | <1%        | Rate of generated non-existent EF values or sources          |
| Response Latency            | <3 seconds | Average generation time for typical queries                  |
| Source Attribution Accuracy | >95%       | Percentage of responses with correct source citation         |

### 7.2 Test Set Composition

- 200 held-out queries not seen during training
- Balanced distribution across regions and entity types
- Inclusion of edge cases and challenging scenarios
- Representative of real-world usage patterns

### 7.3 Evaluation Protocol

1. Generate responses for all test queries
2. Compare numeric values against ground truth
3. Check source attribution against knowledge graph
4. Analyze failure cases and error patterns
5. Generate comprehensive evaluation report

## 8. Model Packaging and Distribution

### 8.1 Deployment Formats

- Full model + LoRA adapter weights (for high-performance deployment)
- Merged model (base + adapter) in safetensors format
- GGUF quantized version for CPU deployment
- Model card with usage examples and performance metrics

### 8.2 Distribution Method

- Primary: Hugging Face Hub repository
- Secondary: Direct download links for enterprise deployment
- Versioning scheme: semantic versioning (MAJOR.MINOR.PATCH)

### 8.3 Usage Documentation

- Inference code examples in Python
- Prompt engineering guidelines
- Fine-tuning extension instructions
- Performance optimization recommendations
- Integration examples with Neo4j knowledge graph

## 9. Risks and Mitigation Strategies

| Risk                             | Likelihood | Impact | Mitigation Strategy                                                         |
| -------------------------------- | ---------- | ------ | --------------------------------------------------------------------------- |
| Google Colab disconnections      | High       | Medium | Implement robust checkpointing, automate restart                            |
| Insufficient GPU memory          | Medium     | High   | Implement gradient checkpointing, reduce batch size, use 4-bit quantization |
| Overfitting to training data     | Medium     | High   | Use dropout, early stopping, and regularization                             |
| Poor performance on edge cases   | Medium     | Medium | Include diverse examples, augment instruction set                           |
| Resource constraints (GPU quota) | Medium     | High   | Schedule training during off-peak hours, optimize training efficiency       |

## 10. Future Extensions

- Implementation of domain-specific continued pre-training
- Exploration of smaller model variants for edge deployment
- Integration of retrieval augmentation for handling the long tail of emission factors
- Development of confidence scoring mechanisms
- Extension to multilingual capabilities

## 11. Deliverables

1. **Fine-Tuned Model**:

   - Mistral-7B base model with LoRA adapters
   - Merged model in safetensors format
   - Quantized versions for different deployment scenarios

2. **Documentation**:

   - Model card with performance metrics
   - Usage examples and integration guides
   - Training methodology details
   - Limitations and recommended use cases

3. **Evaluation Assets**:

   - Test set with ground truth answers
   - Evaluation report with detailed metrics
   - Analysis of error patterns and improvement areas

4. **Training Artifacts**:
   - Google Colab notebooks for reproducibility
   - Training logs and TensorBoard files
   - Dataset preparation scripts
   - Custom evaluation code
