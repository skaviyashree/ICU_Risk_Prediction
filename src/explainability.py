import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import shap

# Append parent directory to sys.path to support importing src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from src import config

def clean_feature_name(name):
    """
    Cleans raw machine learning column names into publication-ready clinical labels.
    Examples:
      - num__vital_heart_rate_mean -> Heart Rate (Mean)
      - cat__race_WHITE -> Race: WHITE
      - num__lab_creatinine_latest_value -> Creatinine (Latest)
      - num__vital_spo2_trend -> SpO2 (Trend)
      - num__is_missing_glucose -> Glucose (Missing Flag)
    """
    name = name.replace("num__", "").replace("cat__", "")
    
    # Clean missingness flags
    if name.startswith("is_missing_"):
        lab = name.replace("is_missing_", "").upper()
        return f"{lab} (Missing Flag)"
        
    # Clean vital signs
    if name.startswith("vital_"):
        parts = name.split("_")
        agg = parts[-1].title()
        var_name = " ".join(parts[1:-1]).title()
        # Abbreviate SpO2 and Blood Pressure properly
        var_name = var_name.replace("Spo2", "SpO2").replace("Bp", "BP").replace("Map", "MAP")
        return f"{var_name} ({agg})"
        
    # Clean laboratory values
    if name.startswith("lab_"):
        parts = name.split("_")
        agg = parts[-1].title()
        if agg == "Value" and parts[-2] == "latest":
            var_name = " ".join(parts[1:-2]).title()
            agg = "Latest"
        else:
            var_name = " ".join(parts[1:-1]).title()
        var_name = var_name.replace("Wbc", "WBC").replace("Bun", "BUN")
        return f"{var_name} ({agg})"
        
    # Clean demographic fields (one-hot encoded)
    for prefix in ['race_', 'admission_type_', 'insurance_', 'marital_status_']:
        if name.startswith(prefix):
            val = name.replace(prefix, "").upper()
            field = prefix[:-1].replace("_", " ").title()
            return f"{field}: {val}"
            
    # Default fallback
    return name.replace("_", " ").title()

def get_shap_explainer(pipeline, X_train):
    """
    Transforms the training set through the pipeline preprocessor
    and initializes a SHAP Explainer on the classifier.
    """
    preprocessor = pipeline.named_steps['preproc']
    classifier = pipeline.named_steps['classifier']
    
    # Fit-transform training set to obtain preprocessed background
    X_train_preproc = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    
    # Wrap transformed training data into a DataFrame with exact column names
    X_train_df = pd.DataFrame(X_train_preproc, columns=feature_names)
    
    # Initialize SHAP Explainer (LinearExplainer for LR, TreeExplainer for GBDTs)
    explainer = shap.Explainer(classifier, X_train_df)
    return explainer, X_train_df

def explain_single_patient(pipeline, explainer, patient_df, save_path=None):
    """
    Computes SHAP feature contributions for a single patient record,
    maps columns to beautiful clinical labels, and generates a waterfall plot.
    """
    preprocessor = pipeline.named_steps['preproc']
    
    # Preprocess patient features
    patient_preproc = preprocessor.transform(patient_df)
    feature_names = preprocessor.get_feature_names_out()
    
    # Wrap preprocessed patient record
    patient_preproc_df = pd.DataFrame(patient_preproc, columns=feature_names)
    
    # Compute SHAP values
    shap_explanation = explainer(patient_preproc_df)
    
    # Extract values and base value
    values = shap_explanation.values[0]
    base_value = shap_explanation.base_values[0]
    data = shap_explanation.data[0]
    
    # Handle multi-class TreeExplainer outputs (survived vs deceased)
    if len(values.shape) == 2 and values.shape[1] == 2:
        values = values[:, 1]
    if isinstance(base_value, (np.ndarray, list)) and len(base_value) == 2:
        base_value = base_value[1]
    
    # Map raw feature names to clinical labels
    clinical_feature_names = [clean_feature_name(col) for col in feature_names]
    
    # Create customized SHAP Explanation object for standard waterfall plotting
    explanation_custom = shap.Explanation(
        values=values,
        base_values=base_value,
        data=data,
        feature_names=clinical_feature_names
    )
    
    # Generate and save waterfall plot
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation_custom, max_display=10, show=False)
    plt.title("Patient Mortality Risk Explanation (SHAP Contribution)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"SHAP explanation waterfall plot saved successfully to: {save_path}")
    plt.close()
    
    return explanation_custom

def get_global_feature_importance(pipeline, explainer, X_train):
    """
    Computes global clinical feature importances by taking the mean
    absolute SHAP value of each feature across the training cohort.
    """
    preprocessor = pipeline.named_steps['preproc']
    X_train_preproc = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_preproc, columns=feature_names)
    
    shap_values = explainer.shap_values(X_train_df)
    
    # Handle SHAP multi-class outputs for tree models (list of arrays or 3D arrays)
    if isinstance(shap_values, list):
        # Index 1 is for class 1 (deceased)
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
        
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    clinical_names = [clean_feature_name(col) for col in feature_names]
    
    df_importance = pd.DataFrame({
        "feature_raw": feature_names,
        "feature_clinical": clinical_names,
        "importance": mean_abs_shap
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    return df_importance

def extract_patient_nlp_explanation(pipeline, explainer, patient_df):
    """
    Identifies top 3 clinical features increasing mortality risk (highest positive SHAP)
    and top 3 clinical features reducing mortality risk (highest negative SHAP),
    and generates a clinician-friendly natural language explanation paragraph.
    """
    preprocessor = pipeline.named_steps['preproc']
    patient_preproc = preprocessor.transform(patient_df)
    feature_names = preprocessor.get_feature_names_out()
    patient_preproc_df = pd.DataFrame(patient_preproc, columns=feature_names)
    
    # Compute SHAP values
    shap_explanation = explainer(patient_preproc_df)
    values = shap_explanation.values[0]
    
    # Handle multi-class case where values is 2D instead of 1D (due to classes axis)
    if len(values.shape) == 2 and values.shape[1] == 2:
        values = values[:, 1]
    
    clinical_names = [clean_feature_name(col) for col in feature_names]
    features_zipped = list(zip(clinical_names, values))
    
    # Filter and sort
    drivers = sorted([f for f in features_zipped if f[1] > 0.001], key=lambda x: x[1], reverse=True)[:3]
    mitigators = sorted([f for f in features_zipped if f[1] < -0.001], key=lambda x: x[1])[:3]
    
    # Construct explanation text
    if not drivers and not mitigators:
        return "Patient observations indicate stable physiology, near the baseline median. No primary high-risk triggers detected."
        
    explanation = ""
    
    if drivers:
        driver_phrases = [f"{d[0]} (+{d[1]*100:.1f}% risk impact)" for d in drivers]
        explanation += "High-risk factors increasing mortality risk include " + ", ".join(driver_phrases[:-1]) + (f", and {driver_phrases[-1]}" if len(driver_phrases) > 1 else driver_phrases[0]) + ". "
        
    if mitigators:
        mit_phrases = [f"{m[0]} ({m[1]*100:.1f}% risk impact)" for m in mitigators]
        explanation += "Stable physiological factors mitigating risk include " + ", ".join(mit_phrases[:-1]) + (f", and {mit_phrases[-1]}" if len(mit_phrases) > 1 else mit_phrases[0]) + "."
        
    return explanation

if __name__ == "__main__":
    try:
        print("\n--- STARTING CLINICAL EXPLAINABILITY ENGINE Standalone Test ---")
        
        # 1. Load pipeline and data
        df = pd.read_csv(config.PATH_PROCESSED_FEATURES)
        model_obj = joblib.load(config.PATH_BEST_MODEL)
        pipeline = model_obj['pipeline'] if isinstance(model_obj, dict) else model_obj
        
        X = df.drop(columns=['stay_id', 'subject_id', 'hospital_expire_flag'])
        y = df['hospital_expire_flag']
        
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=config.TEST_SPLIT_SIZE, stratify=y, random_state=config.RANDOM_STATE
        )
        
        # 2. Get explainer
        explainer, X_train_preproc = get_shap_explainer(pipeline, X_train)
        print("SHAP Explainer initialized successfully!")
        
        # 3. Test global feature importance
        df_importance = get_global_feature_importance(pipeline, explainer, X_train)
        print("\nTop 5 Global Clinical Feature Predictors:")
        print(df_importance.head(5).to_string())
        
        # 4. Test local patient explanation (First patient in test set)
        patient_sample = X_test.iloc[[0]]
        fig_path = os.path.join(config.BASE_DIR, "figures", "sample_patient_shap_waterfall.png")
        explain_single_patient(pipeline, explainer, patient_sample, fig_path)
        
        print("\nExplainability Stage Test Succeeded!")
    except Exception as e:
        print(f"\nExplainability Stage Test Failed: {e}", file=sys.stderr)
