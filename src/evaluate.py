import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Append parent directory to sys.path to support importing src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

from src import config

def evaluate_best_model():
    print("\n--- STARTING PERFORMANCE EVALUATION STAGE ---")
    
    # 1. Load clinical feature matrix and best saved model
    if not os.path.exists(config.PATH_PROCESSED_FEATURES):
        raise FileNotFoundError(f"Feature matrix not found at: {config.PATH_PROCESSED_FEATURES}")
    if not os.path.exists(config.PATH_BEST_MODEL):
        raise FileNotFoundError(f"Best model not found at: {config.PATH_BEST_MODEL}. Please run model training first.")
        
    df = pd.read_csv(config.PATH_PROCESSED_FEATURES)
    model_data = joblib.load(config.PATH_BEST_MODEL)
    
    # Extract pipeline and optimal threshold
    if isinstance(model_data, dict):
        pipeline = model_data['pipeline']
        optimal_threshold = model_data.get('optimal_threshold', 0.5)
    else:
        pipeline = model_data
        optimal_threshold = 0.5
        
    # Separate features and target
    X = df.drop(columns=['stay_id', 'subject_id', 'hospital_expire_flag'])
    y = df['hospital_expire_flag']
    
    # 2. Re-create the Stratified Holdout Train-Test Split (strictly matching train.py)
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SPLIT_SIZE, 
        stratify=y, 
        random_state=config.RANDOM_STATE
    )
    
    # 3. Model Predictions on Holdout Set using Youden's J optimized threshold
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # 4. Compute Standard Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Specificity calculation (True Negative Rate)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (len(y_test) - y_test.sum(), 0, y_test.sum(), 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Curves Areas
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    # Extract model name dynamically for plotting labels
    model_name = type(pipeline.named_steps['classifier']).__name__
    
    print("\n==========================================")
    print("CLINICAL DIAGNOSTIC PERFORMANCE REPORT:")
    print("==========================================")
    print(f"  Holdout Test Set Size: {len(y_test)} (Mortality: {y_test.sum()})")
    print(f"  Champion Classifier:  {model_name}")
    print(f"  Averaged Threshold:   {optimal_threshold:.4f}")
    print(f"  Accuracy:             {accuracy:.4f}")
    print(f"  Precision (PPV):      {precision:.4f}")
    print(f"  Recall (Sensitivity):  {recall:.4f}")
    print(f"  Specificity (TNR):    {specificity:.4f}")
    print(f"  F1-Score:             {f1:.4f}")
    print(f"  ROC-AUC:              {roc_auc:.4f}")
    print(f"  PR-AUC (AUPRC):       {pr_auc:.4f}")
    print("==========================================")
    print(f"Confusion Matrix:\n  Survived (Predicted/Actual):\n    True Negatives (TNs): {tn} | False Positives (FPs): {fp}")
    print(f"  Deceased (Predicted/Actual):\n    False Negatives (FNs): {fn} | True Positives (TPs):  {tp}")
    print("==========================================")
    
    # 5. Plotting Calibration and Curves
    # Configure plotting style
    sns.set_theme(style="whitegrid")
    
    # Curve A: ROC Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#0f62fe', lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#8d8d8d', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(config.PROCESSED_DATA_DIR, "roc_curve.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    
    # Curve B: Precision-Recall Curve
    plt.figure(figsize=(6, 5))
    plt.plot(rec_vals, prec_vals, color='#0d5c3a', lw=2, label=f'{model_name} (AUPRC = {pr_auc:.3f})')
    plt.axhline(y=y_test.mean(), color='#8d8d8d', lw=1, linestyle='--', label=f'Baseline (Rate = {y_test.mean():.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=11)
    plt.ylabel('Precision (PPV)', fontsize=11)
    plt.title('Precision-Recall (PR) Curve', fontsize=12, fontweight='bold')
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_plot_path = os.path.join(config.PROCESSED_DATA_DIR, "pr_curve.png")
    plt.savefig(pr_plot_path, dpi=300)
    plt.close()
    
    # Curve C: Calibration Curve
    # Measures how well model predicted probabilities align with actual risk frequencies
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=5, strategy='uniform')
    
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#f59e0b', label=model_name)
    plt.plot([0, 1], [0, 1], color='#8d8d8d', lw=1, linestyle='--', label='Perfect Calibration')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability', fontsize=11)
    plt.ylabel('Fraction of Positives', fontsize=11)
    plt.title('Clinical Calibration Curve', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    cal_plot_path = os.path.join(config.PROCESSED_DATA_DIR, "calibration_curve.png")
    plt.savefig(cal_plot_path, dpi=300)
    plt.close()
    
    print("Clinical curves plots generated and saved successfully:")
    print(f"  - ROC Curve:         {roc_plot_path}")
    print(f"  - PR Curve:          {pr_plot_path}")
    print(f"  - Calibration Curve: {cal_plot_path}")
    print("Performance Evaluation stage completed successfully!")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }

if __name__ == "__main__":
    try:
        metrics = evaluate_best_model()
        print("\nEvaluation Stage Test Succeeded!")
        print("Verification completed successfully!")
    except Exception as e:
        print(f"\nEvaluation Stage Test Failed: {e}", file=sys.stderr)
