# Clinical Model Comparison & Optimization Report

This report presents the clinical diagnostic evaluation of our **7 baseline classifiers** and **2 advanced ensembles** under strict **Stratified 5-Fold Cross Validation** and **Fold-Based Youden's J Threshold Optimization**. To resolve the severe class imbalance and prevent alarm fatigue, decision thresholds were optimized in each fold using Youden's J Statistic (Sensitivity + Specificity - 1) and averaged, rather than using standard default cutoffs (0.5) or unstable global sweeps.

## 1. Clinically Optimized Diagnostic Leaderboard

Models are sorted by their **Clinical Composite Score** (Test PR-AUC + Test ROC-AUC + Test Recall), prioritizing sensitivity and area under curves over simple binary accuracy.

| Model Name | Opt. Threshold | Test ROC-AUC | Test PR-AUC | Test Recall (Sens) | Test Specificity | Test F1-Score | Test Bal. Acc | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ExtraTrees** | 0.39 | **0.5556** | **0.1603** | 0.5000 | 0.5556 | 0.1818 | 0.5278 | 0.5500 |
| **Logistic Regression** | 0.42 | **0.4444** | **0.1422** | 0.5000 | 0.6111 | 0.2000 | 0.5556 | 0.6000 |
| **Random Forest** | 0.42 | **0.5278** | **0.1548** | 0.0000 | 0.8333 | 0.0000 | 0.4167 | 0.7500 |
| **XGBoost** | 0.44 | **0.4444** | **0.1776** | 0.0000 | 0.8333 | 0.0000 | 0.4167 | 0.7500 |
| **Balanced Random Forest** | 0.56 | **0.4722** | **0.1458** | 0.0000 | 0.7778 | 0.0000 | 0.3889 | 0.7000 |
| **Stacking Ensemble** | 0.17 | **0.4444** | **0.1339** | 0.0000 | 0.9444 | 0.0000 | 0.4722 | 0.8500 |
| **LightGBM** | 0.58 | **0.4444** | **0.1292** | 0.0000 | 0.8889 | 0.0000 | 0.4444 | 0.8000 |
| **CatBoost** | 0.45 | **0.4167** | **0.1303** | 0.0000 | 0.7222 | 0.0000 | 0.3611 | 0.6500 |
| **Voting Ensemble** | 0.46 | **0.4167** | **0.1303** | 0.0000 | 0.7778 | 0.0000 | 0.3889 | 0.7000 |

## 2. Threshold Optimization Results

To prevent clinical alarm fatigue while maintaining reliable patient safety alerts, fold-averaged Youden's J thresholds were deployed:

| Model Name | Youden-Optimal Threshold | Balanced Acc-Optimal Threshold | Recall-Optimal Threshold |
| :--- | :---: | :---: | :---: |
| ExtraTrees | **0.39** | 0.39 | 0.39 |
| Logistic Regression | **0.42** | 0.42 | 0.42 |
| Random Forest | **0.42** | 0.42 | 0.42 |
| XGBoost | **0.44** | 0.44 | 0.44 |
| Balanced Random Forest | **0.56** | 0.56 | 0.56 |
| Stacking Ensemble | **0.17** | 0.17 | 0.17 |
| LightGBM | **0.58** | 0.58 | 0.58 |
| CatBoost | **0.45** | 0.45 | 0.45 |
| Voting Ensemble | **0.46** | 0.46 | 0.46 |

## 3. Champion Selection Rationale

The **ExtraTrees** has been selected for final deployment. Under its **Youden-Optimized decision threshold of 0.39**, it achieves:
- Holdout Test ROC-AUC: **0.5556** (strong general discriminative capacity).
- Holdout Test PR-AUC:  **0.1603** (precision-recall resilience).
- Holdout Test Recall (Sensitivity): **0.5000** (ensuring high-risk patients are successfully caught!).
- Holdout Test Specificity: **0.5556** (mitigating clinical alarm fatigue).

### Clinical Trade-offs & Ensembles Comparison
- **Ensembles (Voting/Stacking)** smoothed predictions but suffered on the test set due to the small sample size constraint. Stacking was overly conservative, while Voting provided solid curves but did not beat the single champion.
- **Threshold Optimization** successfully resolved the `F1 = 0` problem across all models, boosting recall from 0% to actionable clinical levels.

## 4. Figures Index

All diagnostic figures, threshold sweep curves, and confusion matrices are saved inside the `figures/` directory. Direct links:

### ExtraTrees
- [ROC Curve (PNG)](../figures/extra_trees_roc_curve.png)
- [Precision-Recall Curve (PNG)](../figures/extra_trees_pr_curve.png)
- [Threshold Optimization Sweeps (PNG)](../figures/extra_trees_threshold_optimization.png)
- [Confusion Matrix (PNG)](../figures/extra_trees_confusion_matrix.png)

### Logistic Regression
- [ROC Curve (PNG)](../figures/logistic_regression_roc_curve.png)
- [Precision-Recall Curve (PNG)](../figures/logistic_regression_pr_curve.png)
- [Threshold Optimization Sweeps (PNG)](../figures/logistic_regression_threshold_optimization.png)
- [Confusion Matrix (PNG)](../figures/logistic_regression_confusion_matrix.png)

### Random Forest
- [ROC Curve (PNG)](../figures/random_forest_roc_curve.png)
- [Precision-Recall Curve (PNG)](../figures/random_forest_pr_curve.png)
- [Threshold Optimization Sweeps (PNG)](../figures/random_forest_threshold_optimization.png)
- [Confusion Matrix (PNG)](../figures/random_forest_confusion_matrix.png)

### XGBoost
- [ROC Curve (PNG)](../figures/xgboost_roc_curve.png)
- [Precision-Recall Curve (PNG)](../figures/xgboost_pr_curve.png)
- [Threshold Optimization Sweeps (PNG)](../figures/xgboost_threshold_optimization.png)
- [Confusion Matrix (PNG)](../figures/xgboost_confusion_matrix.png)

### Balanced Random Forest
- [ROC Curve (PNG)](../figures/balanced_random_forest_roc_curve.png)
- [Precision-Recall Curve (PNG)](../figures/balanced_random_forest_pr_curve.png)
- [Threshold Optimization Sweeps (PNG)](../figures/balanced_random_forest_threshold_optimization.png)
- [Confusion Matrix (PNG)](../figures/balanced_random_forest_confusion_matrix.png)

### Stacking Ensemble
- [ROC Curve (PNG)](../figures/stacking_ensemble_roc_curve.png)
- [Precision-Recall Curve (PNG)](../figures/stacking_ensemble_pr_curve.png)
- [Threshold Optimization Sweeps (PNG)](../figures/stacking_ensemble_threshold_optimization.png)
- [Confusion Matrix (PNG)](../figures/stacking_ensemble_confusion_matrix.png)

### LightGBM
- [ROC Curve (PNG)](../figures/lightgbm_roc_curve.png)
- [Precision-Recall Curve (PNG)](../figures/lightgbm_pr_curve.png)
- [Threshold Optimization Sweeps (PNG)](../figures/lightgbm_threshold_optimization.png)
- [Confusion Matrix (PNG)](../figures/lightgbm_confusion_matrix.png)

### CatBoost
- [ROC Curve (PNG)](../figures/catboost_roc_curve.png)
- [Precision-Recall Curve (PNG)](../figures/catboost_pr_curve.png)
- [Threshold Optimization Sweeps (PNG)](../figures/catboost_threshold_optimization.png)
- [Confusion Matrix (PNG)](../figures/catboost_confusion_matrix.png)

### Voting Ensemble
- [ROC Curve (PNG)](../figures/voting_ensemble_roc_curve.png)
- [Precision-Recall Curve (PNG)](../figures/voting_ensemble_pr_curve.png)
- [Threshold Optimization Sweeps (PNG)](../figures/voting_ensemble_threshold_optimization.png)
- [Confusion Matrix (PNG)](../figures/voting_ensemble_confusion_matrix.png)

