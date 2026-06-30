# Project Audit & Technical Evaluation Report

**Document ID**: CDSS-AUDIT-2026-001  
**Project**: AI-Powered ICU Mortality Clinical Decision Support System (CDSS)  
**Lead Auditor**: Senior Clinical AI Research Scientist & Thesis Supervisor  
**Status**: COMPLETE (Phase I Audit)  

---

## Executive Summary

A comprehensive technical and clinical audit was performed on the completed **ICU Mortality Prediction CDSS** repository. The core data pipelines, telemetry cleaning algorithms, feature engineering modules, and serialized modeling pipelines were inspected for clinical validity, MLOps robustness, dashboard usability, and academic completeness. 

While the system is fully functional and successfully resolves the class imbalance problem ($F1 = 0$) using synthetic oversampling and threshold optimization, several critical gaps must be addressed to upgrade this pilot application into a **publication-quality, thesis-defense-ready, hospital-grade platform**. This report details the clinical, mathematical, and architectural findings of the audit.

---

## 1. Clinical Inconsistencies & Risk Mappings

### Finding 1.1: Decision Threshold vs. Clinical Risk Bands
* **The Issue**: The champion model (**Logistic Regression**) utilizes an out-of-fold F1-optimized decision threshold of **0.01** (or 1.00%). In the current dashboard, any prediction exceeding this threshold (e.g., a patient with a **19.83% mortality probability**) is immediately flagged as **CRITICAL RISK / HIGH RISK**. 
* **The Technical Rationale**: Mathematically, because the cohort's baseline mortality rate is only 11.00%, a 1.00% cutoff is required to maximize recall (sensitivity) and ensure that 100% of deceased holdout patients are captured.
* **The Clinical Flaw**: Clinicians are trained to interpret probabilities in standardized biological bands. Flagging a 19.83% risk as "CRITICAL" leads to severe clinical skepticism and instant alarm fatigue. A 19.83% risk represents an elevated, moderate threat that requires closer bedside evaluation—not rapid escalation.
* **The Solution**: Maintain the mathematical threshold of 0.01 for binary classification internally, but replace the user-facing risk display with four distinct **Clinical Probability Bands**:
  * `0% - 10%`: **LOW RISK** (Stable baseline, routine care)
  - `10% - 30%`: **MODERATE RISK** (Advisory alert, review labs)
  - `30% - 60%`: **HIGH RISK** (Immediate physician review, prepare support)
  - `60% - 100%`: **CRITICAL RISK** (Trigger Rapid Response Team, prioritize airway/perfusion)

---

## 2. Dashboard Weaknesses & Clinician Usability

### Finding 2.1: Lack of Bedside Physiological Trends
* **The Issue**: While the 24-hour observation window aggregates min, max, mean, and trend statistics, the dashboard only displays raw numerical means in a static table. 
* **The Clinical Flaw**: Bedside clinicians do not evaluate patients in static time slices. They look for physiological trajectories (e.g., whether blood pressure is progressively dropping over the 24-hour stay or stabilizing).
* **The Solution**: Construct a dynamic **24-Hour Physiological Trend Visualizer** that plots hourly heart rate, respiratory rate, SpO2, systolic BP, and temperature trajectories.

### Finding 2.2: Absence of Rule-Based Clinical Recommendations
* **The Issue**: The system outputs a high-level alert and a SHAP waterfall, but does not provide specific physiological treatment guidance.
* **The Clinical Flaw**: A predictive model is only useful if it suggests actionable clinical pathways. 
* **The Solution**: Implement a **Rule-Based Clinical Recommendation Engine** that maps abnormal physiological features to standard-of-care recommendations (e.g., if MAP < 65 mmHg, recommend hemodynamic support; if SpO2 < 95%, suggest oxygen therapy review).

### Finding 2.3: "Black-Box" SHAP Presentation
* **The Issue**: The SHAP waterfall is rendered as a raw scientific plot without natural language description.
* **The Clinical Flaw**: bedside clinicians are not machine learning practitioners; they cannot easily interpret abstract Shapley additive values.
* **The Solution**: Generate an automated **Natural Language Risk Summary** detailing exactly which top 3 features increased the patient's risk, and which top 3 features helped reduce it.

---

## 3. Missing Academic & Thesis Components

### Finding 3.1: Model Performance Analytics Center
* **The Issue**: The project lacks a dedicated user-facing workspace to compare all trained models.
* **The Solution**: Implement an interactive **Model Performance Analytics Center** within the Streamlit dashboard that:
  - Loads metrics for all 7 models and 2 ensembles.
  - Highlights the **Champion Model** with automated mathematical justification (discussing high-bias regularized linear boundaries under small, sparse datasets).
  - Embeds the dynamic ROC comparison plot.

### Finding 3.2: Missing Thesis-Grade Visualizations
To meet the rigorous standards of a final-year thesis defense, the following **10 high-fidelity visualizations** are missing and must be pre-generated:
1. **ROC Comparison Plot** (All 9 models plotted together).
2. **Model Ranking Plot** (Sorted bar chart of clinical composite scores).
3. **Performance Radar Chart** (Visualizing precision, recall, F1, AUC, and balanced accuracy).
4. **Feature Importance Plot** (Global SHAP clinical driver rankings).
5. **Mortality Distribution Plot** (Probability density plots for deceased vs. survived).
6. **Clinical Variable Correlation Heatmap** (Identifying multicollinearity).
7. **SHAP Summary Figure** (Representing global feature impacts).
8. **SHAP Waterfall Figure** (A high-quality benchmark patient explanation).
9. **Dataset Statistics Figure** (Flow chart or summary of cohort demographics).
10. **Research Workflow Diagram** (System architecture horizontal flow chart).

---

## 4. Code Quality & MLOps Gaps

* **Type Hinting**: Major functions lack complete Python type annotations, which compromises readability and code quality.
* **Logging**: Diagnostic prints are used instead of Python's standard `logging` module, making troubleshooting in production clinical systems difficult.
* **Modular Integration**: The Streamlit interface (`app/app.py`) generates a temporary SHAP plot in the feature data directory, which creates clutter. Temporary visual assets should be managed dynamically.

---

## Conclusion & Action Plan

The existing project has a solid data and algorithmic foundation. However, implementing the **Multi-Page Upgrade**, **Physiological Trend Charts**, **Automated Matplotlib PDF Bedside Report Engine**, **Clinical Recommendation Cards**, and **Thesis Performance Galleries** will transform this functional pilot into a premier, publication-ready clinical platform.

We will execute these upgrades systematically according to the approved **Implementation Plan**.
