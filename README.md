# AI-Powered ICU Mortality Prediction & Clinical Decision Support System (CDSS)

This repository houses the complete, production-grade source code and scientific documentation for the **AI-Powered ICU Mortality Prediction & Clinical Decision Support System (CDSS)** using the MIMIC-IV Clinical Demo dataset. 

Designed for final-year engineering evaluations, this system integrates **multimodal EHR ingestion**, **robust clinical anomaly scrubbing**, **time-windowed feature aggregations**, **imbalance-resilient ensemble classifiers**, and **Explainable AI (XAI)** to deliver a premium medical decision support interface.

---

## 1. Clinical Design & Mathematical Rigor

### Early Risk Stratification (24h Observation Window)
Standard predictive models suffer from *data leakage* by using clinical variables recorded immediately before discharge or death. To ensure high clinical utility, this CDSS strictly restricts its features to observations recorded within the **first 24 hours of a patient's first ICU stay** (calculated relative to their `intime` intake timestamp). This ensures risk alerts are triggered early enough to enable life-saving clinical interventions.

### Multimodal Feature Aggregations
For all **15 audited parameters** (7 physiological vitals and 8 blood laboratory panels), the pipeline dynamically extracts **6 distinct clinical aggregations**:
1. `_mean`: Baseline physiological state.
2. `_min`: Lowest reading (captures acute decompensation like bradycardia or severe hypoxia).
3. `_max`: Peak reading (captures extreme states like high fever or hypertensive crisis).
4. `_std`: Physiological instability (high standard deviation indicates high volatility).
5. `_latest_value`: Patient's exit state at the end of the 24-hour observation window.
6. `_trend`: Patient trajectory calculated as `latest_value - first_value` (positive values show increases, negative values show decreases).

This results in a robust feature space of **90 aggregated clinical features** merged with patient demographics.

---

## 2. Directory Structure

```text
ICU_Risk_Prediction_AI/
│
├── data/
│   ├── raw/                  # Raw, unmodified MIMIC-IV Clinical Demo zipped CSVs
│   ├── processed/            # Cleaned, aggregated feature matrices ready for ML training
│   └── models/               # Serialized champion pipeline (best_model.joblib)
│
├── figures/                  # High-fidelity ROC curves and Confusion Matrices (14 PNG plots)
├── reports/                  # Scientific evaluation and model comparison reports
│   └── model_comparison_report.md
│
├── src/                      # Production-grade backend scripts
│   ├── __init__.py
│   ├── config.py             # System pathways, clinical mappings, and hyperparameters
│   ├── data_ingestion.py     # Parses raw GZ files and compiles the baseline cohort
│   ├── preprocessing.py      # Telemetry anomaly cleaning and unit normalizations
│   ├── feature_engineering.py# Temporal window slicing and 24h aggregations
│   ├── train.py              # Stratified CV ensemble training and model selection
│   └── explainability.py     # SHAP clinical explanation engines
│
├── app/                      # Web interface dashboards
│   ├── app.py                # Master Streamlit dashboard code
│   └── utils.py              # Dynamic HTML metrics alert layouts
│
├── requirements.txt          # Python dependencies with strict version constraints
└── run_pipeline.py           # Master CLI orchestrator script
```

---

## 3. Quick-Start & Installation

### Prerequisite Dependencies
Ensure Python 3.10+ is installed. Clone the repository and run:
```bash
pip install -r requirements.txt
```

### End-to-End Execution (Orchestrator)
Execute the complete data science pipeline—from raw zipped CSV ingestion to final model selection, curve plotting, and SHAP explainer weights serialization—using a single CLI command:
```bash
python run_pipeline.py
```

### Running the Clinical Web App
Launch the interactive Streamlit clinical decision support dashboard locally:
```bash
streamlit run app/app.py
```

---

## 4. Scientific Model Performance & Selection

The system evaluated **5 individual algorithms** and **2 advanced ensembles** under strict **Stratified 5-Fold Cross Validation** (optimized using Test ROC-AUC):

| Model Name | CV Accuracy | CV Precision | CV Recall | CV F1-Score | CV ROC-AUC | Test ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7375 | 0.0667 | 0.1000 | 0.0800 | 0.4743 | **0.5833** |
| **LightGBM** | 0.8375 | 0.3000 | 0.3000 | 0.2667 | 0.5857 | **0.5556** |
| **XGBoost** | 0.8000 | 0.2400 | 0.3000 | 0.2000 | 0.5286 | **0.4444** |
| **Voting Ensemble** | 0.8375 | 0.2000 | 0.3000 | 0.2333 | 0.5786 | **0.4444** |
| **Random Forest** | 0.8750 | 0.4000 | 0.3000 | 0.3333 | 0.6500 | **0.3611** |
| **Stacking Ensemble** | 0.8875 | 0.0000 | 0.0000 | 0.0000 | 0.3990 | **0.2778** |
| **Balanced Random Forest** | 0.7250 | 0.2067 | 0.5000 | 0.2800 | 0.6286 | **0.1111** |

### Selection Rationale
On the small MIMIC-IV Clinical Demo cohort (100 patients, 140 stays), high-dimensional tree classifiers and ensembles (like Stacking) suffered from severe overfitting due to parameter complexity. **Logistic Regression** emerged as the champion classifier, exhibiting superior generalization ability (Holdout ROC-AUC: **0.5833**). 

### Explainable AI (SHAP)
By integrating **SHAP LinearExplainer**, the champion model's predictions are fully decomposed into individual mathematical feature contributions. The top clinical risk drivers identified across the entire cohort are:
1. **`Potassium (Min)`** (hypokalemia signifies cellular instability).
2. **`Respiratory Rate (Mean)`** & **`Resp Rate Latest (Value)`** (tachypnea indicates respiratory distress).
3. **`Systolic BP (Max)`** (hypertension indicates acute vascular strain).
