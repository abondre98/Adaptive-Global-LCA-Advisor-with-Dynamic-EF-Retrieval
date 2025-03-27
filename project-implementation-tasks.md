# Adaptive Global LCA Advisor
# Project Implementation Tasks by Milestone

## Overview

This document provides a comprehensive task breakdown for implementing the Adaptive Global LCA Advisor project - an AI system that recommends region-specific emission factors (EFs) for accurate carbon accounting. The project combines LLM fine-tuning with dynamic RAG for real-time EF updates and consists of four primary milestones.

## Milestone 1: Regional EF Knowledge Graph

### 1. Data Collection and Preparation

#### 1.1 Gather Existing Datasets
- [ ] Acquire Agribalyse 3.1 from INRAE Portal (https://agribalyse.ademe.fr/)
- [ ] Download USEEIO v2.1 from EPA GitHub (https://github.com/USEPA/USEEIO)
- [ ] Obtain OpenLCA data from Nexus platform (https://nexus.openlca.org/)
- [ ] Access EXIOBASE 3.8 for multi-regional data (https://www.exiobase.eu/)
- [ ] Set up Climate TRACE API access for real-time emission updates

#### 1.2 Structure and Standardize Data
- [ ] Create organized folder structure for all datasets
- [ ] Ensure consistent file formats (convert as needed)
- [ ] Standardize column headers across datasets
- [ ] Remove duplicate records
- [ ] Create data dictionary documenting all fields and sources

#### 1.3 Identify and Apply IPCC AR6 Regional Multipliers
- [ ] Access IPCC AR6 reports for regional factors
- [ ] Create mapping table linking countries to IPCC regions
- [ ] Standardize country codes using ISO 3166-1
- [ ] Develop formula for EF adjustment using multipliers
- [ ] Apply multipliers to base EFs and validate results

### 2. Data Processing with Regional Adjustments

#### 2.1 Apply IPCC AR6 Multipliers to Base EFs
- [ ] Compile reference table mapping each region to its multiplier
- [ ] Implement adjustment formula: Adjusted_EF = Base_EF × IPCC_multiplier(region)
- [ ] Process all datasets to create adjusted EF values
- [ ] Add metadata columns for multiplier tracking

#### 2.2 Validate Adjusted Emission Factors
- [ ] Select representative sample (1-5%) across regions and products
- [ ] Manually verify adjustment calculations
- [ ] Check for outliers and extreme values
- [ ] Document validation methodology and results

#### 2.3 Clean and Finalize Adjusted Dataset
- [ ] Identify and handle incomplete or missing data
- [ ] Remove or flag outliers
- [ ] Consolidate into final format with standardized fields
- [ ] Export to format suitable for Neo4j import

### 3. Neo4j Knowledge Graph Design

#### 3.1 Define Schema / Data Model
- [ ] Create node definitions:
  - [ ] EF nodes (attributes: ef_id, value, unit, data_source, etc.)
  - [ ] Region nodes (attributes: region_id, country_code, continent, ipcc_region)
  - [ ] Industry nodes (attributes: industry_id, industry_name, etc.)
- [ ] Define relationships:
  - [ ] PRODUCED_IN (linking EF to Region)
  - [ ] HAS_IMPACT (linking Industry to EF)

#### 3.2 Set Up Neo4j Environment
- [ ] Deploy Neo4j instance (Aura Free Tier or Docker-based)
- [ ] Configure authentication and access controls
- [ ] Create indexes for frequently queried properties
- [ ] Set up constraints for data integrity

#### 3.3 Import Data into Neo4j
- [ ] Prepare CSV/JSON files for import
- [ ] Use appropriate import method (neo4j-admin or LOAD CSV)
- [ ] Verify node and relationship creation
- [ ] Confirm total node count meets target (50K+ EF nodes)

#### 3.4 Preliminary Testing
- [ ] Run basic data integrity queries
- [ ] Test region and industry lookups
- [ ] Verify relationship correctness
- [ ] Measure query performance

### 4. Documentation and Validation

#### 4.1 Documentation
- [ ] Document data sources, versions, and licenses
- [ ] Create detailed schema diagram
- [ ] Document IPCC multiplier integration methodology
- [ ] Prepare user guide for querying the knowledge graph

#### 4.2 Validation and Delivery
- [ ] Verify 50K+ EF nodes coverage
- [ ] Test sample queries for region-industry pairs
- [ ] Document query performance metrics
- [ ] Prepare final deliverables:
  - [ ] Unified EF dataset (post-adjustment)
  - [ ] Neo4j graph database
  - [ ] Documentation package
  - [ ] Validation query set and results

## Milestone 2: Mistral-7B Fine-Tuning

### 1. Data Preparation and Instruction Set Creation

#### 1.1 Aggregate Training Data
- [ ] Extract text fields from environmental datasets
- [ ] Export relevant descriptions from Neo4j knowledge graph
- [ ] Identify typical user questions and patterns
- [ ] Compile domain-specific scenarios and templates

#### 1.2 Generate Instruction-Prompt Pairs
- [ ] Create diverse question variations for each product-region combination
- [ ] Formulate accurate answers with source attribution
- [ ] Ensure coverage of multiple regions and industries
- [ ] Balance query complexity (simple lookups vs. comparisons)
- [ ] Add metadata tags (region, industry, difficulty_level, source)

#### 1.3 Clean and Finalize Dataset
- [ ] Remove duplicate or near-duplicate entries
- [ ] Filter out irrelevant or trivial prompts
- [ ] Split into training (80%) and validation (20%) sets
- [ ] Ensure balanced representation across regions/industries
- [ ] Store in standardized format (JSON/CSV)

### 2. LoRA Setup and Configuration

#### 2.1 Environment Preparation
- [ ] Create dedicated Python environment
- [ ] Install PyTorch, Transformers, and PEFT libraries
- [ ] Verify GPU availability and CUDA compatibility
- [ ] Estimate and allocate required disk space

#### 2.2 Set Hyperparameters
- [ ] Configure target modules (q_proj, v_proj)
- [ ] Set LoRA rank (default: 64, fallback: 32)
- [ ] Define learning rate (3e-4)
- [ ] Determine batch size for GPU utilization
- [ ] Set epoch count (3-5) and early stopping criteria

#### 2.3 Model Initialization
- [ ] Download base Mistral-7B model
- [ ] Initialize LoRA adapters with specified configuration
- [ ] Verify model loading and parameter freezing
- [ ] Test model with sample inputs before training

### 3. Fine-Tuning Workflow

#### 3.1 Training Iterations
- [ ] Implement training loop with loss calculation
- [ ] Set up monitoring for training and validation metrics
- [ ] Configure mixed-precision (FP16) training
- [ ] Run initial training phase with base parameters

#### 3.2 Checkpoints and Logging
- [ ] Create checkpoint saving routine (every 500-1000 steps)
- [ ] Set up logging system (TensorBoard or W&B)
- [ ] Track training and validation losses
- [ ] Monitor Precision@3 on validation set

#### 3.3 Optimization Tuning
- [ ] Implement learning rate scheduler or adjustment strategy
- [ ] Monitor for plateaus in validation metrics
- [ ] Adjust batch size or LoRA rank if needed
- [ ] Implement early stopping if no improvement

### 4. Model Validation and Performance Metrics

#### 4.1 Precision@3 Evaluation
- [ ] Create test set of region-specific queries
- [ ] Generate top 3 responses for each query
- [ ] Check if correct EF appears in top 3 results
- [ ] Calculate overall Precision@3 score (target: >85%)

#### 4.2 MAPE Calculation
- [ ] Collect ground truth EF values from reference sources
- [ ] Generate model predictions for test set
- [ ] Calculate percentage error for each prediction
- [ ] Compute overall MAPE (target: <5%)

#### 4.3 Edge Case Testing
- [ ] Test model with underrepresented regions
- [ ] Evaluate performance on unusual industries
- [ ] Document any biases or systematic errors
- [ ] Create report on error patterns and categories

### 5. Finalizing and Packaging the Model

#### 5.1 Export Model Artifacts
- [ ] Save LoRA adapter weights
- [ ] Document base model version reference
- [ ] Store best-performing checkpoint
- [ ] Create model card with performance metrics

#### 5.2 Model Distribution
- [ ] Decide on distribution method (HuggingFace Hub or internal)
- [ ] Create repository with appropriate access controls
- [ ] Upload model weights and configuration
- [ ] Provide loading examples and instructions

#### 5.3 Documentation
- [ ] Document final hyperparameters and training details
- [ ] Record resource usage and training duration
- [ ] Provide sample queries and model responses
- [ ] Note any limitations or known issues

## Milestone 3: Hybrid RAG Pipeline

### 1. Vector Index and Embedding Pipeline

#### 1.1 Embedding Selection
- [ ] Evaluate embedding models for domain fit
- [ ] Select final embedding model (e.g., gte-large)
- [ ] Test embedding quality on sample EF data
- [ ] Document embedding dimensions and properties

#### 1.2 Data Vectorization
- [ ] Extract relevant text from Neo4j knowledge graph
- [ ] Create vectorization pipeline for all entities
- [ ] Generate and store embeddings for all EF data
- [ ] Implement embedding refresh strategy for updates

#### 1.3 Quality Verification
- [ ] Compare similarity between known related items
- [ ] Test retrieval precision on sample queries
- [ ] Evaluate embedding clustering quality
- [ ] Optimize vector parameters if needed

### 2. RedisVL Setup and Integration

#### 2.1 Redis Configuration
- [ ] Deploy Redis instance (cloud or self-hosted)
- [ ] Configure vector search capabilities
- [ ] Set up appropriate memory allocation
- [ ] Implement backup strategy

#### 2.2 Data Loading
- [ ] Develop batch loading script for vectors
- [ ] Import all embeddings into Redis
- [ ] Configure index parameters (similarity metrics, algorithms)
- [ ] Verify successful indexing and retrieval

#### 2.3 Query Flow Implementation
- [ ] Create end-to-end query pipeline
- [ ] Implement vector similarity search
- [ ] Integrate Neo4j for additional context
- [ ] Test retrieval accuracy and speed

### 3. Input Guardrails Implementation

#### 3.1 Query Validation
- [ ] Create syntactic validation rules
- [ ] Implement query normalization processes
- [ ] Build logging system for input monitoring
- [ ] Test with varied input formats

#### 3.2 Intent Recognition
- [ ] Develop query classification system
- [ ] Define handling logic for each intent type
- [ ] Implement intent-based routing
- [ ] Test intent recognition accuracy

#### 3.3 Entity Extraction
- [ ] Build product/material entity extractor
- [ ] Create region/country recognizer
- [ ] Implement industry sector identification
- [ ] Develop fallback mechanisms for ambiguity

#### 3.4 Input Sanitization
- [ ] Implement security filtering
- [ ] Add rate limiting functionality
- [ ] Create handling for malformed queries
- [ ] Test with potentially problematic inputs

#### 3.5 Query Rewriting
- [ ] Develop enhancement for underconstrained queries
- [ ] Create logic for splitting compound queries
- [ ] Build abbreviation expansion system
- [ ] Test query transformation accuracy

### 4. Semantic Cache Implementation

#### 4.1 Frequent Query Pre-Caching
- [ ] Identify top queries for pre-caching
- [ ] Implement caching structure in Redis
- [ ] Develop cache warming strategy
- [ ] Monitor cache size and memory usage

#### 4.2 Cache Management
- [ ] Configure Time-to-Live policies
- [ ] Implement cache invalidation triggers
- [ ] Set up different TTLs based on data volatility
- [ ] Create cache analytics dashboard

#### 4.3 API Integration
- [ ] Add cache lookup to query pipeline
- [ ] Implement fuzzy matching for similar queries
- [ ] Create cache update mechanism
- [ ] Optimize for minimal latency

#### 4.4 Performance Testing
- [ ] Measure retrieval times with and without cache
- [ ] Calculate cache hit rate metrics
- [ ] Optimize cache configuration for target latency (<200ms)
- [ ] Document cache performance findings

### 5. Real-Time Updates Integration

#### 5.1 Climate TRACE Connection
- [ ] Configure Climate TRACE API client
- [ ] Set up scheduled data retrieval
- [ ] Implement data validation and parsing
- [ ] Create error handling and retry logic

#### 5.2 Data Update Propagation
- [ ] Build Neo4j update mechanism
- [ ] Create Redis embedding refresh process
- [ ] Implement cache invalidation for affected entries
- [ ] Test update propagation end-to-end

#### 5.3 Version Control
- [ ] Create data versioning system
- [ ] Implement backup before updates
- [ ] Build logging for all data changes
- [ ] Develop rollback capability

#### 5.4 Update Verification
- [ ] Create test cases for update scenarios
- [ ] Verify updates appear in search results
- [ ] Confirm cache invalidation works correctly
- [ ] Measure update propagation time

### 6. Output Guardrails Implementation

#### 6.1 Response Validation
- [ ] Implement physical plausibility checks
- [ ] Add range validation against historical data
- [ ] Create consistency verification against benchmarks
- [ ] Develop confidence scoring system

#### 6.2 Uncertainty Quantification
- [ ] Build uncertainty calculation methods
- [ ] Implement confidence interval generation
- [ ] Create flagging for high-uncertainty results
- [ ] Test with varied data quality scenarios

#### 6.3 Source Reconciliation
- [ ] Develop weighted averaging for conflicting data
- [ ] Add source attribution to all responses
- [ ] Create discrepancy detection system
- [ ] Document reconciliation methodology

#### 6.4 Response Explanation
- [ ] Build explanation generation system
- [ ] Include source and adjustment information
- [ ] Add confidence statements to responses
- [ ] Test explanation clarity and completeness

#### 6.5 Output Formatting
- [ ] Define standardized response structure
- [ ] Implement both machine and human-readable formats
- [ ] Add appropriate metadata to responses
- [ ] Create fallback response templates

#### 6.6 Response Filtering
- [ ] Add sensitivity controls for misleading outputs
- [ ] Implement appropriate disclaimers
- [ ] Create flagging for synthetic data usage
- [ ] Test with edge cases and unusual queries

### 7. Performance Optimization

#### 7.1 Load Testing
- [ ] Develop concurrent request simulation
- [ ] Identify performance bottlenecks
- [ ] Test system stability under load
- [ ] Document performance under various conditions

#### 7.2 Latency Optimization
- [ ] Tune Redis parameters for speed
- [ ] Implement asynchronous processing
- [ ] Optimize guardrail processing overhead
- [ ] Measure and document latency improvements

#### 7.3 Edge Case Handling
- [ ] Test with ambiguous or incomplete queries
- [ ] Verify handling of out-of-database requests
- [ ] Ensure graceful failure modes
- [ ] Document edge case behavior

### 8. API Development and Documentation

#### 8.1 API Implementation
- [ ] Create FastAPI endpoints for EF queries
- [ ] Implement error handling and status codes
- [ ] Add request/response logging
- [ ] Test API functionality and performance

#### 8.2 Documentation Creation
- [ ] Write comprehensive API documentation
- [ ] Create usage examples and tutorials
- [ ] Document error codes and troubleshooting
- [ ] Explain guardrail behaviors

#### 8.3 Final System Verification
- [ ] Conduct end-to-end testing
- [ ] Verify all guardrails function correctly
- [ ] Confirm overall system latency meets target
- [ ] Prepare final deliverables and report

## Milestone 4: Edge Deployment

### 1. Model Distillation

#### 1.1 Teacher-Student Setup
- [ ] Configure fine-tuned Mistral-7B as teacher
- [ ] Select appropriate student model (TinyLlama-1.1B)
- [ ] Set up distillation framework
- [ ] Prepare training environment

#### 1.2 Distillation Data Preparation
- [ ] Create or extend dataset for knowledge distillation
- [ ] Ensure coverage of region-specific examples
- [ ] Prepare data loading pipeline
- [ ] Split into training and evaluation sets

#### 1.3 Training Process
- [ ] Implement distillation training loop
- [ ] Configure loss functions for knowledge transfer
- [ ] Set up logging and checkpointing
- [ ] Monitor training progress and convergence

#### 1.4 Performance Validation
- [ ] Compare student model to teacher on test set
- [ ] Calculate Precision@3 and MAPE metrics
- [ ] Identify any knowledge gaps
- [ ] Fine-tune if performance is below acceptable threshold

### 2. Model Quantization

#### 2.1 Quantization Strategy
- [ ] Research optimal quantization approach
- [ ] Select quantization level (4-bit/NF4)
- [ ] Prepare quantization tools and scripts
- [ ] Estimate final model size

#### 2.2 Implementation
- [ ] Export model to appropriate format
- [ ] Apply quantization process
- [ ] Verify model integrity post-quantization
- [ ] Optimize for target hardware

#### 2.3 Performance Testing
- [ ] Measure inference speed on target platforms
- [ ] Calculate memory footprint
- [ ] Compare accuracy to non-quantized version
- [ ] Document performance tradeoffs

### 3. Web Interface Development

#### 3.1 Framework Selection and Setup
- [ ] Set up Streamlit or alternative framework
- [ ] Configure development environment
- [ ] Design interface mockups
- [ ] Create project structure

#### 3.2 Backend Integration
- [ ] Load quantized model with appropriate runtime
- [ ] Connect to Redis and Neo4j services
- [ ] Implement inference pipeline
- [ ] Add guardrails from Milestone 3

#### 3.3 Frontend Implementation
- [ ] Create input forms for EF queries
- [ ] Design results display components
- [ ] Add visualizations for EF comparisons
- [ ] Implement user feedback mechanisms

#### 3.4 Performance Optimization
- [ ] Optimize load time and responsiveness
- [ ] Implement caching at application level
- [ ] Test end-to-end latency
- [ ] Profile and improve bottlenecks

### 4. Documentation and Final Validation

#### 4.1 User Documentation
- [ ] Create setup instructions
- [ ] Write user manual for interface
- [ ] Document example workflows
- [ ] Add troubleshooting section

#### 4.2 Technical Documentation
- [ ] Document distillation and quantization process
- [ ] Create system architecture diagram
- [ ] Record deployment requirements
- [ ] Note known limitations

#### 4.3 Final QA and Delivery
- [ ] Conduct comprehensive testing
- [ ] Verify accuracy against benchmarks
- [ ] Confirm memory and performance targets
- [ ] Package final deliverables:
  - [ ] Distilled and quantized model
  - [ ] Web-based demo application
  - [ ] Technical documentation
  - [ ] Validation results

## Project Timeline

| Milestone | Duration | Key Dependencies |
|-----------|----------|------------------|
| 1: Regional EF Knowledge Graph | 8 weeks | Dataset availability |
| 2: Mistral-7B Fine-Tuning | 6 weeks | Completion of Milestone 1, GPU availability |
| 3: Hybrid RAG Pipeline | 6 weeks | Completion of Milestones 1-2 |
| 4: Edge Deployment | 4 weeks | Completion of Milestones 1-3 |

## Resource Allocation

| Role | Primary Responsibility | Key Milestones |
|------|------------------------|----------------|
| Data Engineer | Dataset preparation, Neo4j implementation | 1, 3 |
| ML Engineer | Fine-tuning, model distillation | 2, 4 |
| Full-Stack Developer | API development, web interface | 3, 4 |

## Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| Data gaps in specific regions | Use Claude 3 for synthetic EF generation |
| GPU availability constraints | Reduce LoRA rank to 32, use gradient checkpointing |
| Latency exceeds target | Implement progressive loading with cached results first |
| Model drift from real-time updates | Daily validation against Climate TRACE |
