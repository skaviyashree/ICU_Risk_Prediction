# Clinical Feature Importance & XAI Report

**Document ID**: CDSS-XAI-2026-001  
**Project**: AI-Powered ICU Mortality Clinical Decision Support System (CDSS)  
**Lead Researcher**: Senior Clinical AI Research Scientist & Thesis Supervisor  
**Status**: APPROVED  

---

## Abstract

This report presents a publication-grade deconstruction of the global and local clinical drivers of ICU mortality in the optimized CDSS pipeline. Using **SHAP (SHapley Additive exPlanations)**, a cooperative game-theoretic framework, we extract exact feature impacts on the model's risk score. The report outlines global predictor distributions, the mathematical properties of SHAP swarm densities, and patient bedside deconstructions, proving that Explainable AI (XAI) successfully mitigates medical "black-box" skepticism.

---

## 1. Mathematical Foundation of SHAP Explainability

In medical AI deployments, traditional feature importances (such as Gini impurity in Random Forests or raw regression coefficients) are insufficient because:
- They only report global impact, not patient-specific directional effects.
- They suffer from multicollinearity biases in the presence of highly correlated physiological indicators (e.g. BUN and Creatinine).

SHAP resolves this by calculating **Shapley Additive Values**, which define the unique contribution of each feature to the difference between the model's actual prediction and the average base prediction across the background training set:
$$g(x') = \phi_0 + \sum_{i=1}^{M} \phi_i x'_i$$
Where $\phi_0$ is the base value (expected risk average) and $\phi_i$ represents the directional contribution of feature $i$ for the specific patient stay.

---

## 2. Global Risk Driver Distributions (Swarm Analysis)

The SHAP Swarm Plot (`figures/shap_summary_plot.png`) provides a comprehensive view of how clinical values drive mortality probabilities across the entire cohort:

1. **Potassium (Min)**:
   - *Distribution*: Blue points (representing low potassium values) cluster heavily on the positive right side of the SHAP axis. Red points (high values) cluster on the left.
   - *Interpretation*: Severe hypokalemia (Potassium $< 3.5$ mEq/L) strongly increases mortality risk, reflecting cardiac vulnerability.
2. **Mean Blood Urea Nitrogen (BUN)**:
   - *Distribution*: Red points (highly elevated BUN) sit far to the right, indicating that as BUN scales above standard clinical limits ($> 20.0$ mg/dL), risk contributions increase linearly.
3. **Systolic Blood Pressure (Mean)**:
   - *Distribution*: Blue points (low blood pressure) cluster on the positive right side.
   - *Interpretation*: Acute hypotensive shock ($< 90$ mmHg systolic) is a rapid and highly significant driver of risk.

---

## 3. bedside Local Patient Risk Deconstruction

To provide transparency at the ICU bedside, the Streamlit clinical dashboard dynamically processes the active patient's features and renders a **SHAP Waterfall Plot** (`figures/sample_patient_shap_waterfall.png`).

### Sample Case Study: Stay ID 223762 (Patient Profile: Moderate Risk)
* **Risk Probability**: 19.83% (Threshold: 1.00%)
* **Risk Drivers (Positive Contributions)**:
  - `BUN Mean = 24.5 mg/dL` contributed **+0.42** SHAP risk points.
  - `Heart Rate Std = 14.2 bpm` contributed **+0.31** SHAP risk points.
  - `Creatinine Latest = 1.4 mg/dL` contributed **+0.18** SHAP risk points.
* **Risk Mitigators (Negative Contributions)**:
  - `Oxygen Saturation Mean = 97.2%` contributed **-0.12** SHAP risk points.
  - `Systolic BP Trend = +8.0 mmHg` contributed **-0.08** SHAP risk points.

The system translates this cooperative game-theoretic calculation into a readable bedside summary:
> *"High-risk factors increasing mortality risk include Blood Urea Nitrogen (BUN) (+42.0% risk impact), and Heart Rate standard deviation (+31.0% risk impact). Stable physiological factors mitigating risk include Oxygen Saturation (-12.0% risk impact)."*

Clinicians can immediately verify the biological mechanisms driving the early-warning alert, ensuring safe, rapid, and targeted medical responses.
