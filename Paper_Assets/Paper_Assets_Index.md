# Publication Assets Index: ICU Mortality Prediction CDSS

This index describes all the assets compiled, audited, and copied into the `Paper_Assets/` folder for your IEEE Transactions / Conference paper. All figures are renamed consistently to match standard publication formatting.

---

## 📂 Assets Inventory

### Figures

#### **Fig1_System_Architecture.png**
* **Source**: `figures/post_retraining/system_architecture_diagram.png`
* **Format**: PNG (300 DPI, clean vector layout)
* **Description**: Detailed horizontal block diagram representing the full clinical machine learning pipeline and bedside decision support interface. It shows three primary tiers: (1) Data Ingestion and Multimodal Preprocessing; (2) Predictive Machine Learning Engine (ExtraTrees champion); and (3) Clinical Actionability Tier (Streamlit/PDF export).
* **IEEE Insertion Target**: **Section III. SYSTEM ARCHITECTURE** (Insert as a full-width figure at the top of Page 3).
* **IEEE Caption**: 
  > **Fig. 1.** End-to-end system architecture of the Explainable AI (XAI) clinical decision support system (CDSS) for bedside ICU mortality prediction. The horizontal pipeline ingests raw patient clinical data, applies a leak-free multimodal preprocessing layer, performs feature selection (107 to 9 features), and executes predictions using the champion ExtraTrees Classifier under a fold-averaged Youden's J optimized threshold of 0.39. Model explanations are derived via Tree SHAP, with downstream integration into a bedside Streamlit dashboard and vector PDF chart generator.

#### **Fig2_Dataset_Statistics.png**
* **Source**: `figures/post_retraining/dataset_statistics_summary.png`
* **Format**: PNG (300 DPI)
* **Description**: Demographic breakdown of the cohort stays, mapping the patient counts by race, admission type (e.g. EW Emergency, Observation, Urgent), and gender.
* **IEEE Insertion Target**: **Section IV-A. Cohort Characteristics** (Insert in a single-column layout on Page 4).
* **IEEE Caption**: 
  > **Fig. 2.** Distribution of cohort stays by gender, ethnicity/race, and hospital admission source type across the N=140 admissions cohort extracted from the MIMIC-IV Clinical Demo database.

#### **Fig3_Variable_Correlation.png**
* **Source**: `figures/post_retraining/variable_correlation_heatmap.png`
* **Format**: PNG (300 DPI)
* **Description**: Multicollinearity heatmap analyzing the correlation coefficients between the 9 selected clinical biomarkers. It shows high correlation between kidney markers (BUN and Creatinine = 0.82), explaining why L2 regularization or random feature-subspacing (ExtraTrees) is necessary to prevent variance instability.
* **IEEE Insertion Target**: **Section IV-B. Feature Selection and Collinearity** (Insert in a single-column layout on Page 4).
* **IEEE Caption**: 
  > **Fig. 3.** Correlation heatmap of the 9 publication-grade clinical biomarkers selected for model training. The high correlation between BUN and Creatinine (r = 0.82) illustrates the multicollinearity threat, justifying the deployment of the randomized feature subspacing in the ExtraTrees classifier to stabilize estimator variance.

#### **Fig4_Model_Comparison_Ranking.png**
* **Source**: `figures/post_retraining/model_performance_ranking.png`
* **Format**: PNG (300 DPI)
* **Description**: Bar chart ranking all 9 candidate classifiers (including Stacking and Voting ensembles) based on their Clinical Composite Score ($\text{PR-AUC} + \text{ROC-AUC} + \text{Sensitivity}$).
* **IEEE Insertion Target**: **Section V-A. Quantitative Evaluation** (Insert in a single-column layout on Page 5).
* **IEEE Caption**: 
  > **Fig. 4.** Model performance rankings sorted by Clinical Composite Score (Sensitivity + ROC-AUC + PR-AUC). Ensembles (Stacking/Voting) suffered from overfitting due to cohort size constraints, while regularized ExtraTrees emerged as the champion.

#### **Fig5_Combined_ROC_Curves.png**
* **Source**: `figures/post_retraining/combined_roc_comparison.png`
* **Format**: PNG (300 DPI)
* **Description**: Receiver Operating Characteristic (ROC) curve overlay chart mapping the curves for all 9 evaluated classifiers on the holdout test set.
* **IEEE Insertion Target**: **Section V-B. Classification Performance** (Insert as a single-column figure on Page 5).
* **IEEE Caption**: 
  > **Fig. 5.** Overlay comparison of Receiver Operating Characteristic (ROC) curves on the holdout test set ($N=20$) for the seven base classifiers and two ensembles.

#### **Fig6_ExtraTrees_Threshold_Sweeps.png**
* **Source**: `figures/extra_trees_threshold_optimization.png`
* **Format**: PNG (300 DPI)
* **Description**: Sweep curves showing F1-Score, Recall (Sensitivity), and Balanced Accuracy against decision thresholds (0.01 to 0.99) for the ExtraTrees classifier, indicating the optimal Youden's J threshold at 0.39.
* **IEEE Insertion Target**: **Section V-C. Decision Threshold Optimization** (Insert as a single-column figure on Page 6).
* **IEEE Caption**: 
  > **Fig. 6.** Threshold optimization sweep curves for the ExtraTrees classifier. Sweeping the classification threshold across 99 steps on out-of-fold validation sets identifies the optimal Youden's J threshold at 0.39, balancing sensitivity and specificity.

#### **Fig7_ExtraTrees_Confusion_Matrix.png**
* **Source**: `figures/extra_trees_confusion_matrix.png`
* **Format**: PNG (300 DPI)
* **Description**: Confusion matrix of the champion ExtraTrees model on the holdout test set at the Youden-optimal threshold of 0.39, showcasing a balanced distribution (10 TNs, 1 TP, 8 FPs, 1 FN).
* **IEEE Insertion Target**: **Section V-D. Error Analysis** (Insert as a small single-column figure on Page 6).
* **IEEE Caption**: 
  > **Fig. 7.** Confusion matrix of the champion ExtraTrees classifier evaluated on the holdout test set under the optimized decision threshold of 0.39.

#### **Fig8_ExtraTrees_Radar_Profile.png**
* **Source**: `figures/post_retraining/performance_radar_chart.png`
* **Format**: PNG (300 DPI)
* **Description**: Asymmetric radar (spider) chart mapping the multi-metric performance profile of the Youden-optimized champion ExtraTrees model.
* **IEEE Insertion Target**: **Section V-D. Error Analysis** (Insert as a single-column figure on Page 6).
* **IEEE Caption**: 
  > **Fig. 8.** Radar profile showing the multi-metric trade-off of the Youden-optimized ExtraTrees champion, displaying balanced accuracy, sensitivity, specificity, and PR-AUC.

#### **Fig9_SHAP_Global_Swarm.png**
* **Source**: `figures/post_retraining/shap_summary_plot.png`
* **Format**: PNG (300 DPI)
* **Description**: Global Tree SHAP swarm/density plot deconstructing how high (red) or low (blue) physiological values drive risk across the entire cohort.
* **IEEE Insertion Target**: **Section VI-A. Explainable AI (XAI) Analysis** (Insert as a full-width figure at the top of Page 7).
* **IEEE Caption**: 
  > **Fig. 9.** Global SHAP swarm plot for the champion ExtraTrees classifier across the cohort stays. Features are ordered by their average absolute impact. Red points indicate high physiological measurements; blue points represent low measurements. Points to the right of the center line indicate a positive contribution to predicted mortality risk.

#### **Fig10_SHAP_Global_Importance.png**
* **Source**: `figures/post_retraining/shap_bar_plot.png`
* **Format**: PNG (300 DPI)
* **Description**: Global bar chart plotting the mean absolute SHAP value of each clinical biomarker, ranking their absolute predictive strength.
* **IEEE Insertion Target**: **Section VI-B. Biophysical Risk Drivers** (Insert as a single-column figure on Page 7).
* **IEEE Caption**: 
  > **Fig. 10.** Global clinical biomarker importance ranked by mean absolute SHAP value, identifying Potassium (Min) and BUN (Mean) as the primary risk predictors.

#### **Fig11_Local_Patient_SHAP_Waterfall.png**
* **Source**: `figures/sample_patient_shap_waterfall.png`
* **Format**: PNG (300 DPI)
* **Description**: Local bedside SHAP waterfall plot deconstructing the exact physiological factors driving risk up (red) or down (blue) relative to the baseline for a specific ICU patient.
* **IEEE Insertion Target**: **Section VI-C. Local Explanations and Case Study** (Insert as a single-column figure on Page 8).
* **IEEE Caption**: 
  > **Fig. 11.** Local SHAP waterfall plot explaining a specific patient's bedside mortality risk prediction. The base value represents the cohort median average prediction; arrows show the additive biophysical drivers pushing risk up or down.

#### **Fig12_Clinical_Dashboard_UI.png**
* **Source**: `Screenshots/clinical Dashboard_2.png`
* **Format**: PNG (300 DPI)
* **Description**: Screenshot of the Streamlit dashboard clinical mode, displaying patient demographic summary cards,calculated risk percentages, warning alerts, local SHAP narratives, and bedside trend charts.
* **IEEE Insertion Target**: **Section VII. CLINICAL SYSTEM DEPLOYMENT** (Insert as a single-column figure on Page 9).
* **IEEE Caption**: 
  > **Fig. 12.** Deployed bedside clinician interface (Streamlit) showing the Clinical Mode lookup registry, risk severity alert banners, local SHAP narratives, bedside line trajectories, and rule-based guideline panels.

#### **Fig13_Research_Dashboard_UI.png**
* **Source**: `Screenshots/Research Dashboard_.png`
* **Format**: PNG (300 DPI)
* **Description**: Screenshot of the Streamlit dashboard research mode, displaying the performance leaderboards, model galleries, and global SHAP interpretability graphs.
* **IEEE Insertion Target**: **Section VII. CLINICAL SYSTEM DEPLOYMENT** (Insert as a single-column figure on Page 9).
* **IEEE Caption**: 
  > **Fig. 13.** Deployed dashboard Research Mode, allowing clinical researchers to audit validation leaderboards, ROC/PR curves, global SHAP swarm charts, and correlation heatmaps.

#### **Fig14_ExtraTrees_Calibration_Curve.png**
* **Source**: `data/processed/calibration_curve.png`
* **Format**: PNG (300 DPI)
* **Description**: Reliability/calibration curve of the champion ExtraTrees classifier, indicating predicted risk probability against actual fraction of positives.
* **IEEE Insertion Target**: **Section V-E. Calibration Analysis** (Insert as a single-column figure on Page 6).
* **IEEE Caption**: 
  > **Fig. 14.** Probability calibration curve showing predicted risk probability against actual observed mortality rate for the champion ExtraTrees classifier.

#### **Fig15_ExtraTrees_ROC_Curve.png**
* **Source**: `data/processed/roc_curve.png`
* **Format**: PNG (300 DPI)
* **Description**: Receiver Operating Characteristic (ROC) curve of the single champion ExtraTrees model on the holdout test set.
* **IEEE Insertion Target**: **Section V-B. Classification Performance** (Insert as a small single-column figure on Page 5).
* **IEEE Caption**: 
  > **Fig. 15.** Receiver Operating Characteristic (ROC) curve of the champion ExtraTrees classifier on the holdout test set (Test AUC = 0.556).

#### **Fig16_ExtraTrees_PR_Curve.png**
* **Source**: `data/processed/pr_curve.png`
* **Format**: PNG (300 DPI)
* **Description**: Precision-Recall (PR) curve of the champion ExtraTrees model on the holdout test set (Test AUPRC = 0.160).
* **IEEE Insertion Target**: **Section V-B. Classification Performance** (Insert as a small single-column figure on Page 5).
* **IEEE Caption**: 
  > **Fig. 16.** Precision-Recall (PR) curve of the champion ExtraTrees classifier on the holdout test set (Test PR-AUC = 0.160), with reference baseline.

---

### Tables

#### **Table1_Dataset_Statistics.csv**
* **Source**: `audit_results.csv`
* **Format**: CSV / Tabular
* **Description**: Ingested telemetry and laboratory observations coverage statistics, mapping total observation counts, unique patient counts, overall stay coverage %, and 24-hour stay coverage %.
* **IEEE Insertion Target**: **Section IV-A. Cohort Characteristics** (Insert as a full-width Table at the top of Page 4).
* **IEEE Caption**: 
  > **TABLE I.** Data Acquisition and Stay Coverage Statistics for the Ingested MIMIC-IV Clinical Demo Cohort.

#### **Table2_Model_Performance.csv**
* **Source**: Extracted from `reports/model_comparison_report.md`
* **Format**: CSV / Tabular
* **Description**: Diagnostic validation leaderboard sorting all 9 candidate models by their Clinical Composite Score. Columns include Youden-optimal threshold, Test ROC-AUC, Test PR-AUC, Test Recall (Sensitivity), Specificity, F1-Score, Balanced Accuracy, and Accuracy.
* **IEEE Insertion Target**: **Section V-A. Quantitative Evaluation** (Insert as a full-width Table at the top of Page 5).
* **IEEE Caption**: 
  > **TABLE II.** Clinically Optimized Diagnostic Leaderboard of Candidate Classifiers and Ensembles.

---

### Documents

#### **Doc1_Clinical_PDF_Report.pdf**
* **Source**: `reports/test_patient_report.pdf`
* **Format**: PDF (Vector document)
* **Description**: Hospital-grade, vector-rendered Bedside Patient Risk Registry A4 chart. It features the patient demographics table, calculated risk percentages, SHAP explanation summaries, rule-based clinical guides, and physician signature lines.
* **IEEE Insertion Target**: **Section VII. CLINICAL SYSTEM DEPLOYMENT** (Embed as an Appendix page or reference).
* **IEEE Description**: Vector bedside clinical chart generated automatically by the decision support engine, ready for physician signature and physical clinical registry filing.

---

## 🚫 Missing Assets / Exclusions Report

The following expected assets do not exist or were excluded:
* **Learning Curves**: Not applicable. Because we train shallow non-temporal models (regularized Logistic Regression and ExtraTrees with max_depth=3) on a static clinical feature matrix, standard iterative learning curves (loss vs. epochs) are not applicable.
* **Clinical Recommendation Screenshots** / **PDF Report Screenshots**: Screenshots of these subcomponents are already embedded directly within `Fig12_Clinical_Dashboard_UI.png` and `Doc1_Clinical_PDF_Report.pdf` (the actual PDF), so generating separate screenshot pngs was excluded to avoid asset redundancy.
