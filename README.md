# SERS-AI-Analysis
Cysteine Quantification in Pea Cultivars from SERS Spectra Using AI
Workflow
Overall workflow for SERS data acquisition and AI-based cysteine prediction (figures/Figure1_workflow.png)

This repository contains:

Data preparation scripts to assemble the SERS dataset
Optional preprocessing pipeline (Savitzky–Golay smoothing, ModPoly baseline correction, Min–Max normalization)
Regression baselines (LR, PLSR, SVR, RFR)
Deep learning model (1D-CNN)
Leave-One-Cultivar-Out (LOCO) generalization experiments
SHAP-based interpretability and noise robustness study

Repository structure
Data Preparation/
Scripts to build the dataset used by all models.

Preprocessing_sers/
Preprocessing module for SERS spectra (SG + ModPoly + Min–Max).

Machine Learning models:

Linear Regression/
Partial Least Square Regression/
Support Vector Machine Regression/
Random Forest Regression/

Deep learning:
1D-CNN/

Applications:
SHAP Analysis/
Noise Study/

Setup
Create an environment:

Option A: pip python -m venv .venv macOS/Linux: source .venv/bin/activate Windows: .venv\Scripts\activate

pip install -r requirements.txt

Option B: conda conda create -n sers python=3.10 -y conda activate sers pip install -r requirements.txt

