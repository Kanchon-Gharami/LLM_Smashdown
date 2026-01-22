# LLM-Smashdown: Black-Box LLM Replication via Logit Leakage and Distillation

This repository provides a prototype implementation of the methods described in our paper:

Clone What You Can’t Steal: Black-Box LLM Replication via Logit Leakage and Distillation

The codebase demonstrates the overall pipeline, structure, and experimental workflow used in the study, including projection-matrix recovery, knowledge distillation, and evaluation. This is not the exact original research code; instead, it is a cleaned, representative implementation intended for transparency and reproducibility.

---

## Overview

The project studies black-box model replication of large language models (LLMs) under constrained access, assuming only top-k logit exposure via an API.

The pipeline consists of two stages:

1. Stealing stage – Recovering the output projection subspace using SVD over leaked logits.
2. Cloning stage – Distilling the remaining transformer behavior into compact student models.

---

## Repository Structure

LLM_Smashdown/
├── LLM_smashdown.py  
│   Main training script for local execution. Implements projection recovery, knowledge distillation, and student model training.  
│
├── LLM_smashdown_HPC.py  
│   Cluster/HPC version of the training script. Used for large-scale or multi-GPU training.  
│
├── config.json  
│   Central configuration file containing model depth, temperature, loss weights, dataset paths, and hyperparameters.  
│
├── requirements.txt  
│   Python dependencies required to run the code.  
│
├── EDA.ipynb  
│   Exploratory Data Analysis notebook for dataset inspection and sanity checks.  
│
├── Evaluation.ipynb  
│   Primary evaluation notebook for perplexity, NLL, KL divergence, cosine similarity, and token-level alignment.  
│
├── Evaluation_2.ipynb  
│   Extended evaluation notebook covering generalization, AIC/AICc analysis, and efficiency trade-offs.  
│
├── Evaluation Results/  
│   Saved plots and intermediate evaluation artifacts.  
│
├── results/  
│   Final experiment outputs, logs, and summarized metrics.  
│
└── Directory_Structure.txt  
    Reference description of the expected folder layout.

---

## Running the Code

Install dependencies:
pip install -r requirements.txt

Local training:
python LLM_smashdown.py

HPC / cluster training:
python LLM_smashdown_HPC.py

---

## Evaluation

All evaluation is performed using Jupyter notebooks:
- EDA.ipynb for dataset understanding
- Evaluation.ipynb for core metrics
- Evaluation_2.ipynb for advanced analysis

Results are saved under Evaluation Results/ and results/.

---

## Notes

- No proprietary models, weights, or APIs are included.
- All experiments rely on public datasets and simulated black-box access.
- This repository is intended for research, red-team simulation, and defensive analysis.

---

## Citation

If you use this repository, please cite the corresponding paper.

---

## Author

Kanchon Gharami  
Department of Electrical Engineering and Computer Science  
Embry-Riddle Aeronautical University
