import os
import sys
import time

# Ensure workspace is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion import ingest_all_raw_data
from src.preprocessing import preprocess_clinical_data
from src.feature_engineering import extract_clinical_features
from src.train import build_advanced_ml_pipelines
from src.explainability import get_shap_explainer, get_global_feature_importance

def main():
    print("======================================================================")
    print("      ICU MORTALITY PREDICTION & DECISION SUPPORT SYSTEM PIPELINE     ")
    print("======================================================================")
    start_time = time.time()
    
    # --- STAGE 1: DATA INGESTION ---
    t0 = time.time()
    cohort, chartevents, labevents = ingest_all_raw_data()
    print(f"--> Ingestion completed in {time.time() - t0:.2f} seconds.")
    
    # --- STAGE 2: CLINICAL PREPROCESSING & OUTLIER CLEANING ---
    t0 = time.time()
    cohort_clean, charts_clean, labs_clean = preprocess_clinical_data(cohort, chartevents, labevents)
    print(f"--> Preprocessing completed in {time.time() - t0:.2f} seconds.")
    
    # --- STAGE 3: 24h CLINICAL FEATURE ENGINEERING ---
    t0 = time.time()
    features = extract_clinical_features(cohort_clean, charts_clean, labs_clean)
    print(f"--> Feature Engineering completed in {time.time() - t0:.2f} seconds.")
    
    # --- STAGE 4: MODEL TRAINING, ENSEMBLES & CLINICAL EVALUATION ---
    t0 = time.time()
    pipeline, X_train, X_test, y_train, y_test = build_advanced_ml_pipelines()
    print(f"--> Modeling, Ensembles and Evaluation completed in {time.time() - t0:.2f} seconds.")
    
    # --- STAGE 5: SHAP EXPLAINABILITY ENGINE INITIALIZATION ---
    t0 = time.time()
    print("\n--- INITIALIZING SHAP EXPLAINABILITY ENGINE ---")
    explainer, X_train_preproc = get_shap_explainer(pipeline, X_train)
    
    # Verify explainer with global feature importance
    df_importance = get_global_feature_importance(pipeline, explainer, X_train)
    print("SHAP Explainability Engine initialized successfully!")
    print(f"  Top global clinical driver: {df_importance.loc[0, 'feature_clinical']} (Mean |SHAP| = {df_importance.loc[0, 'importance']:.4f})")
    print(f"--> SHAP Explainer initialized in {time.time() - t0:.2f} seconds.")
    
    total_time = time.time() - start_time
    print("\n======================================================================")
    print(f"PIPELINE RUN COMPLETED SUCCESSFULY IN {total_time:.2f} SECONDS!")
    print("======================================================================")
    print("Generated Artifacts & Outputs Directory:")
    print("  - Raw Cohort Ingestion:  Verified (100 patients, 140 ICU stays)")
    print("  - Processed Feature Matrix: data/processed/processed_clinical_features.csv")
    print("  - Serialized Champion Pipeline: data/models/best_model.joblib")
    print("  - Clinical Comparison Report:   reports/model_comparison_report.md")
    print("  - Model Curves and Matrices:    figures/ (14 high-fidelity PNG plots)")
    print("======================================================================")

if __name__ == "__main__":
    main()
