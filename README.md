# Gastric Cancer Mutational Signatures & Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive, end-to-end bioinformatics and machine learning pipeline for extracting mutational signatures, performing feature engineering, and evaluating predictive models for Gastric Cancer. This pipeline processes raw mutation data (MAF), extracts *de novo* mutational signatures via NMF, maps them to COSMIC v3.4, and utilizes advanced machine learning architectures to predict clinical subtypes and survival outcomes.

---

## 🧬 Pipeline Architecture

The pipeline consists of modular, distinct steps which can be run sequentially to reproduce the entire study:

### 1. Mutational Matrix Generation
- **Script:** `build_sbs96_matrix.py`
- **Description:** Parses raw somatic mutation data from MAF files to construct a 96-channel Single Base Substitution (SBS96) matrix, capturing the specific trinucleotide contexts of mutations.

### 2. *De Novo* Signature Extraction
- **Script:** `extract_signatures.py`
- **Description:** Employs Non-Negative Matrix Factorization (NMF) to extract optimal *de novo* mutational signatures (e.g., 9 signatures) from the generated SBS96 matrix.

### 3. COSMIC Signature Assignment
- **Script:** `cosmic_assignment.py`
- **Description:** Utilizes Non-Negative Least Squares (NNLS) fitting to accurately map the extracted *de novo* signatures to established COSMIC v3.4 reference signatures.

### 4. Clinical Data Integration
- **Script:** `get_clinical_data.py`
- **Description:** Harvests, cleans, and integrates patient clinical data and established molecular subtypes from the Genomic Data Commons (GDC) and cBioPortal.

### 5. Advanced Feature Engineering
- **Script:** `feature_engineering.py`
- **Description:** Synthesizes complex biological data to generate a robust set of 179 machine learning features, incorporating signature exposures, clinical variables, and genomic metadata.

### 6. Machine Learning Classification
- **Script:** `ml_classification.py`
- **Description:** Trains, tunes, and evaluates an ensemble of 5 advanced machine learning models to predict gastric cancer subtypes and clinical outcomes with high precision.

### 7. High-Dimensional Visualization & Interpretability
- **Script:** `visualize.py`
- **Description:** Generates publication-ready figures including t-SNE, UMAP, Kaplan-Meier survival plots, and SHAP value summaries for model interpretability.

### 8. Automated Reporting
- **Script:** `generate_report.py`
- **Description:** Automatically compiles all metrics, findings, and visualizations into an interactive HTML summary report.



---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/Gastric-Cancer-Genomics-ML.git
cd Gastric-Cancer-Genomics-ML
pip install -r requirements.txt
```

### 2. Prepare Data
Place your raw MAF (Mutation Annotation Format) files and any clinical data inside the `data/` directory. (See `data/INSTRUCTIONS.md` for details).

### 3. Run Pipeline
To run the entire pipeline seamlessly from end to end, execute the master shell script:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Alternatively, you can run each Python script sequentially based on the architecture outlined above.

## 📁 Repository Structure
```
├── data/                    # Place raw MAF and clinical data here
├── output/                  # Generated plots, reports, and models
├── build_sbs96_matrix.py
├── cosmic_assignment.py
├── extract_signatures.py
├── feature_engineering.py
├── generate_report.py
├── get_clinical_data.py
├── ml_classification.py
├── run_pipeline.sh
├── visualize.py
├── requirements.txt
└── README.md
```

## 🛠 Requirements

All dependencies are listed in `requirements.txt`. Major libraries include:
- Python 3.8+
- Scikit-learn, Pandas, Numpy
- Matplotlib, Seaborn
- SHAP, UMAP-learn, SciPy

---
*Note: This repository is intended for academic research in precision oncology and computational biology.*
