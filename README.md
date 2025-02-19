# Adaptive Global LCA Advisor with Dynamic EF Retrieval  
## Carbon Footprint Recommendation System  

## Overview  
This project aims to develop an AI-powered system that provides region-specific **emission factor (EF)** recommendations for accurate carbon accounting. The approach integrates **fine-tuned LLMs** with **dynamic Retrieval-Augmented Generation (RAG)** to enable real-time EF updates.  

## Business Use Case  

### Problem Statement  
- A significant percentage of food products lack environmental impact labels, leading to **inconsistent carbon footprint calculations**.  
- Manual selection of emission factors introduces **15-30% error rates** due to outdated or regionally incorrect data.  
- Regulatory requirements, such as the **EU Carbon Border Adjustment Mechanism (CBAM)**, necessitate **precise regional EF mapping** for compliance.  

### Solution  
- **Automated EF recommendations** tailored to specific regions (e.g., wheat production in different countries).  
- **Real-time EF retrieval** leveraging AI and structured databases.  
- **Web and mobile interface** for supply chain managers and auditors.  

## Technical Components  

| Component               | Technology Stack                 |  
|------------------------|--------------------------------|  
| **Base LLM**          | Mistral-7B (fine-tuned with LoRA) |  
| **Vector Database**    | RedisVL, FAISS                 |  
| **Knowledge Graph**    | Neo4j                          |  
| **Quantized Deployment** | TinyLlama (4-bit GGUF)        |  
| **Real-Time EF Updates** | Climate TRACE API             |  

## Dataset Requirements  
This system integrates structured datasets containing **region-specific emission factors**. The datasets include:  

1. **Primary Emission Factor Databases:**  
   - **EXIOBASE** – Multi-regional input-output database.  
   - **Ecoinvent** – LCA database with global EFs for various sectors.  
   - **US EPA Factors** – Environmental impact datasets for industrial processes.  

2. **Real-Time Data Feeds:**  
   - **Climate TRACE API** – Live updates for industry-specific emissions.  
   - **National GHG Inventories** – Country-wise emission datasets for regulatory compliance.  

3. **Internal Fine-Tuning Dataset:**  
   - Historical EF selections from **manual audit logs**.  
   - Region-wise LCA study data from open-access research. 
