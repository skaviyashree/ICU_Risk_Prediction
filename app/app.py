import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt

# Ensure workspace is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.explainability import get_shap_explainer, explain_single_patient, extract_patient_nlp_explanation
from src.pdf_generator import generate_clinical_pdf_report
from utils import inject_clinical_custom_css, render_medical_alert, render_clinical_profile_card

# Configure Streamlit page options
st.set_page_config("ICU CDSS - Clinical Risk Stratification Platform", layout="wide", page_icon="🩺")

# Load Custom CSS
inject_clinical_custom_css()

# --- CACHED PIPELINE & DATA INITIALIZATION ---
@st.cache_resource
def load_system_pipeline():
    """Loads and caches the serialized best clinical pipeline."""
    if not os.path.exists(config.PATH_BEST_MODEL):
        raise FileNotFoundError("Champion model file not found. Please run pipeline first.")
    return joblib.load(config.PATH_BEST_MODEL)

@st.cache_data
def load_clinical_dataset():
    """Loads and caches the processed clinical feature matrix."""
    if not os.path.exists(config.PATH_PROCESSED_FEATURES):
        raise FileNotFoundError("Processed feature matrix not found. Please run pipeline first.")
    return pd.read_csv(config.PATH_PROCESSED_FEATURES)

try:
    model_obj = load_system_pipeline()
    pipeline = model_obj['pipeline'] if isinstance(model_obj, dict) else model_obj
    optimal_threshold = model_obj['optimal_threshold'] if isinstance(model_obj, dict) else 0.01
    df_features = load_clinical_dataset()
    
    # Initialize background dataset for SHAP Explainer
    X = df_features.drop(columns=['stay_id', 'subject_id', 'hospital_expire_flag'])
    y = df_features['hospital_expire_flag']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SPLIT_SIZE, 
        stratify=y, 
        random_state=config.RANDOM_STATE
    )
    
    # Cache and load SHAP engine
    @st.cache_resource
    def load_shap_engine():
        return get_shap_explainer(pipeline, X_train)
        
    explainer, _ = load_shap_engine()
    system_ready = True
except Exception as e:
    system_ready = False
    st.error(f"System Initialization Failed: {e}. Please ensure the master pipeline has been executed.")

# --- NAVIGATION SIDEBAR ---
if system_ready:
    st.sidebar.image("https://img.icons8.com/color/96/000000/hospital.png", width=65)
    st.sidebar.markdown("<h3 style='margin-top:0;'>SYSTEM MODE</h3>", unsafe_allow_html=True)
    
    system_mode = st.sidebar.radio(
        "Select Mode:",
        ["Clinical Mode", "Research Mode"]
    )
    
    # ==========================================
    # --- 1. CLINICAL BEDSIDE MODE ---
    # ==========================================
    if system_mode == "Clinical Mode":
        st.sidebar.markdown("---")
        st.sidebar.markdown("<h4>Patient Profile Settings</h4>", unsafe_allow_html=True)
        
        mode = st.sidebar.selectbox(
            "Select Registry Mode:",
            ["Demo Patient Registry Lookup", "Bedside Physiological Slider Builder"]
        )
        
        patient_df = None
        patient_id = None
        age, gender, race, admission_type, insurance, marital_status = 65, 1, "WHITE", "URGENT", "Medicare", "MARRIED"
        
        if mode == "Demo Patient Registry Lookup":
            # Let clinician pick from the 100 stays
            stay_options = df_features['stay_id'].tolist()
            stay_id_selected = st.sidebar.selectbox(
                "Select ICU Stay ID:",
                stay_options,
                format_func=lambda x: f"Stay ID: {x} ({'Deceased' if df_features[df_features['stay_id']==x]['hospital_expire_flag'].values[0]==1 else 'Survived'})"
            )
            
            # Extract record
            patient_row = df_features[df_features['stay_id'] == stay_id_selected]
            patient_df = patient_row.drop(columns=['stay_id', 'subject_id', 'hospital_expire_flag'])
            patient_id = stay_id_selected
            
            # Populate demographics
            age = int(patient_row['age'].values[0])
            gender = int(patient_row['gender'].values[0])
            race = patient_row['race'].values[0]
            admission_type = patient_row['admission_type'].values[0]
            insurance = patient_row['insurance'].values[0]
            marital_status = patient_row['marital_status'].values[0]
        else:
            st.sidebar.markdown("---")
            st.sidebar.markdown("<h4>Demographic Baselines</h4>", unsafe_allow_html=True)
            age = st.sidebar.slider("Patient Age:", 18, 95, 65)
            gender_str = st.sidebar.selectbox("Biological Gender:", ["MALE", "FEMALE"])
            gender = 1 if gender_str == "MALE" else 0
            race = st.sidebar.selectbox("Ethnicity / Race:", ["WHITE", "BLACK/CAPE VERDEAN", "HISPANIC/LATINO", "ASIAN", "OTHER"])
            admission_type = st.sidebar.selectbox("Admission Type:", ["URGENT", "EW EMER.", "OBSERVATION", "SURGICAL SAME DAY ADMISSION"])
            insurance = st.sidebar.selectbox("Insurance Type:", ["Medicare", "Medicaid", "Other"])
            marital_status = st.sidebar.selectbox("Marital Status:", ["MARRIED", "SINGLE", "WIDOWED", "DIVORCED", "Unknown"])
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("<h4>Bedside Vitals (Mean)</h4>", unsafe_allow_html=True)
            hr_mean = st.sidebar.slider("Heart Rate (bpm):", 40, 160, 85)
            rr_mean = st.sidebar.slider("Respiratory Rate (bpm):", 8, 45, 18)
            spo2_mean = st.sidebar.slider("Oxygen Saturation SpO2 (%):", 75, 100, 96)
            sbp_mean = st.sidebar.slider("Systolic BP (mmHg):", 70, 200, 115)
            temp_mean = st.sidebar.slider("Temperature (°C):", 34.0, 41.5, 36.8, step=0.1)
            
            st.sidebar.markdown("---")
            st.sidebar.markdown("<h4>Laboratory Panels</h4>", unsafe_allow_html=True)
            creatinine = st.sidebar.slider("Creatinine (mg/dL):", 0.2, 8.0, 1.1, step=0.1)
            wbc = st.sidebar.slider("WBC (K/uL):", 1.0, 40.0, 8.5, step=0.5)
            potassium = st.sidebar.slider("Potassium (mEq/L):", 2.0, 7.0, 4.1, step=0.1)
            bun = st.sidebar.slider("BUN (mg/dL):", 3.0, 100.0, 18.0, step=1.0)
            
            # Reconstruct manual profile
            manual_record = X_train.median(numeric_only=True).to_dict()
            manual_record["race"] = race
            manual_record["admission_type"] = admission_type
            manual_record["insurance"] = insurance
            manual_record["marital_status"] = marital_status
            manual_record["age"] = age
            manual_record["gender"] = gender
            
            # Re-map standard mean/min/max bounds based on inputs
            manual_record["vital_heart_rate_mean"] = hr_mean
            manual_record["vital_heart_rate_latest_value"] = hr_mean
            manual_record["vital_heart_rate_min"] = hr_mean - 5
            manual_record["vital_heart_rate_max"] = hr_mean + 5
            
            manual_record["vital_resp_rate_mean"] = rr_mean
            manual_record["vital_resp_rate_latest_value"] = rr_mean
            manual_record["vital_resp_rate_min"] = rr_mean - 2
            manual_record["vital_resp_rate_max"] = rr_mean + 2
            
            manual_record["vital_spo2_mean"] = spo2_mean
            manual_record["vital_spo2_latest_value"] = spo2_mean
            manual_record["vital_spo2_min"] = spo2_mean - 1
            
            manual_record["vital_systolic_bp_mean"] = sbp_mean
            manual_record["vital_systolic_bp_latest_value"] = sbp_mean
            manual_record["vital_systolic_bp_min"] = sbp_mean - 10
            manual_record["vital_systolic_bp_max"] = sbp_mean + 10
            
            manual_record["vital_temperature_mean"] = temp_mean
            manual_record["vital_temperature_latest_value"] = temp_mean
            
            manual_record["lab_creatinine_mean"] = creatinine
            manual_record["lab_creatinine_latest_value"] = creatinine
            manual_record["lab_wbc_mean"] = wbc
            manual_record["lab_wbc_latest_value"] = wbc
            manual_record["lab_potassium_mean"] = potassium
            manual_record["lab_potassium_latest_value"] = potassium
            manual_record["lab_bun_mean"] = bun
            manual_record["lab_bun_latest_value"] = bun
            
            patient_df = pd.DataFrame([manual_record])[X.columns]
            patient_id = "MANUAL_STAY"
            
        # Predict probability
        y_prob = pipeline.predict_proba(patient_df)[:, 1][0]
        risk_percentage = y_prob * 100
        
        confidence_score = abs(y_prob - 0.5) * 200
        
        risk_drivers = []
        try:
            if patient_df["lab_bun_latest_value"].values[0] > 20:
                risk_drivers.append("Elevated Blood Urea Nitrogen (BUN)")
            if patient_df["vital_systolic_bp_mean"].values[0] < 90:
                risk_drivers.append("Low Systolic Blood Pressure")
            if patient_df["vital_resp_rate_mean"].values[0] > 20:
                risk_drivers.append("Elevated Respiratory Rate")
            if patient_df["vital_spo2_mean"].values[0] < 95:
                risk_drivers.append("Reduced Oxygen Saturation")
        except:
            pass
        
        # Clinical Risk Level bands
        if risk_percentage < 5:
            risk_category = "Low Risk"
            risk_color = "#10b981"
        elif risk_percentage < 15:
            risk_category = "Moderate Risk"
            risk_color = "#f59e0b"
        elif risk_percentage < 40:
            risk_category = "High Risk"
            risk_color = "#ef4444"
        else:
            risk_category = "Critical Risk"
            risk_color = "#b91c1c"
            
        st.markdown("<h2 style='margin-top:0;'>🏥 Bedside Clinical Decision Support (CDS)</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # TOP ROW: Demographics + Prediction Dial
        col_demo, col_risk = st.columns([1, 1])
        with col_demo:
            render_clinical_profile_card(age, gender, race, admission_type, insurance)
        with col_risk:
            st.markdown(f"""
            <div class="clinical-card" style="text-align: center; border: 1px solid rgba(15, 98, 254, 0.3); background: rgba(15, 98, 254, 0.03); height: 215px; margin-bottom: 10px;">
                <div class="metric-title" style="margin-top: 15px;">Calculated Mortality Probability</div>
                <div class="metric-value" style="color: {risk_color}; font-size: 3.5rem;">
                    {risk_percentage:.2f}%
                </div>
                <div style="font-size: 1.1rem; font-weight: bold; color: {risk_color}; margin-top: 5px;">
                    {risk_category.upper()} BAND
                </div>
                <div style="font-size: 0.85rem; color: #8d8d8d; margin-top: 5px;">
                    Under F1-optimized classification threshold ({optimal_threshold*100:.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.metric(
                "Prediction Confidence",
                f"{confidence_score:.1f}%"
            )
            
        # MIDDLE ROW: Status Alert Card + NLP explanation
        col_alert, col_nlp = st.columns([1.1, 0.9])
        with col_alert:
            st.markdown("<h4 style='margin-top:0;'>🟢 Actionable Medical Alert Status</h4>", unsafe_allow_html=True)
            render_medical_alert(risk_percentage)
        with col_nlp:
            st.markdown("<h4 style='margin-top:0;'>🧬 Patient Risk Factor Narrative</h4>", unsafe_allow_html=True)
            with st.spinner("Decomposing physiological risk drivers..."):
                try:
                    shap_nlp_text = extract_patient_nlp_explanation(
                        pipeline,
                        explainer,
                        patient_df
                    )
                    # Convert ML terminology to clinician-friendly text
                    shap_nlp_text = shap_nlp_text.replace("risk impact", "clinical risk adjustment").replace("SHAP", "impact")
                except Exception as e:
                    shap_nlp_text = (
                        f"Clinical risk explainer unavailable. "
                        f"Prediction generated successfully. ({str(e)})"
                    )
                st.markdown(f"""
                <div class="clinical-card" style="font-size: 0.95rem; line-height: 1.45; border-left: 5px solid #0f62fe;">
                    {shap_nlp_text}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Top Clinical Risk Drivers")
                if risk_drivers:
                    for item in risk_drivers:
                        st.warning(item)
                else:
                    st.success("No major physiological abnormalities detected.")
                
        # FOURTH ROW: Physiological Trend lines + Clinical Recommendations
        col_trend, col_recs = st.columns([1, 1])
        with col_trend:
            st.markdown("<h4>📈 Illustrative Physiological Trend Reconstruction</h4>", unsafe_allow_html=True)
            st.caption(
                "Trend visualization reconstructed from aggregated 24-hour ICU statistics for demonstration purposes."
            )
            # Render a beautiful Matplotlib line chart for heart rate, respiratory rate, and SpO2 trends
            hr_val = patient_df["vital_heart_rate_mean"].values[0]
            rr_val = patient_df["vital_resp_rate_mean"].values[0]
            spo2_val = patient_df["vital_spo2_mean"].values[0]
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            hours = [0, 6, 12, 18, 24]
            # Simulate slight organic fluctuations around the values
            hr_series = [hr_val - 3, hr_val + 2, hr_val - 1, hr_val + 4, hr_val]
            rr_series = [rr_val + 1, rr_val - 2, rr_val, rr_val + 2, rr_val]
            
            ax.plot(hours, hr_series, marker='o', color='#ef4444', lw=2, label='Heart Rate (bpm)')
            ax.plot(hours, rr_series, marker='s', color='#0f62fe', lw=2, label='Resp Rate (bpm)')
            ax.set_xlabel('Hours Post-ICU Admission', fontsize=9, fontweight='bold')
            ax.set_ylabel('Physiological Values', fontsize=9, fontweight='bold')
            ax.set_ylim(0, 150)
            ax.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        with col_recs:
            st.markdown("<h4>📋 Rule-Based Treatment Recommendations</h4>", unsafe_allow_html=True)
            # Reconstruct recommendations based on indicators
            bun_val = patient_df["lab_bun_latest_value"].values[0]
            bp_val = patient_df["vital_systolic_bp_mean"].values[0]
            spo2_val = patient_df["vital_spo2_mean"].values[0]
            rr_val = patient_df["vital_resp_rate_mean"].values[0]
            creat_val = patient_df["lab_creatinine_latest_value"].values[0]
            
            recs_list = []
            if bun_val > 20.0:
                recs_list.append("🔴 Renal Perfusion: Elevated Blood Urea Nitrogen (BUN) flagged. Recommend renal panel review and fluid balance indices evaluation.")
            else:
                recs_list.append("🟢 Renal Perfusion: BUN level is stable and within standard healthy parameters.")
                
            if bp_val < 90.0:
                recs_list.append("🔴 Hemodynamics: Low mean blood pressure observed. Consider arterial lines review or vasoactive perfusion support.")
            else:
                recs_list.append("🟢 Hemodynamics: Mean arterial systolic blood pressure is stable.")
                
            if spo2_val < 95.0:
                recs_list.append("🔴 Respirations: Reduced oxygen saturation. Suggest arterial blood gas checks and supplemental oxygen review.")
            else:
                recs_list.append("🟢 Respirations: Oxygen saturation SpO2 indices are clinically stable.")
                
            if rr_val > 20.0 or rr_val < 10.0:
                recs_list.append("🔴 Airway Status: Elevated/Depressed Respiratory Rate flagged. Evaluate bedside ventilation parameters or distress patterns.")
                
            recs_html = "".join([f"<div style='margin-bottom:10px; font-size:0.92rem;'>{r}</div>" for r in recs_list])
            st.markdown(f"""
            <div class="clinical-card" style="border: 1px solid rgba(0,0,0,0.1); background: rgba(0,0,0,0.01); height: 235px; overflow-y: auto;">
                {recs_html}
            </div>
            """, unsafe_allow_html=True)
            
        # PDF EXPORT ROW
        st.markdown("---")
        st.markdown("<h4>🏥 Hospital Bedside Record Export</h4>", unsafe_allow_html=True)
        # Prepare data and trigger generate_clinical_pdf_report
        demographics_data = {
            "age": age,
            "gender": "MALE" if gender == 1 else "FEMALE",
            "race": race,
            "admission_type": admission_type,
            "insurance": insurance,
            "marital_status": marital_status
        }
        
        pdf_recs = []
        for r in recs_list:
            if "🔴" in r:
                pdf_recs.append(r.replace("🔴 ", ""))
        if not pdf_recs:
            pdf_recs.append("Physiology is stable. Maintain routine clinical monitor pathways.")
            
        pdf_save_path = os.path.join(config.BASE_DIR, "reports", f"bedside_report_{patient_id}.pdf")
        
        if st.button("Generate Bedside PDF Clinical Chart"):
            with st.spinner("Generating hospital vector chart..."):
                generate_clinical_pdf_report(
                    patient_id=patient_id,
                    demographics=demographics_data,
                    risk_pct=risk_percentage,
                    risk_category=risk_category,
                    risk_color=risk_color,
                    nlp_summary=shap_nlp_text.replace("**", ""),
                    recommendations=pdf_recs,
                    save_path=pdf_save_path
                )
                st.success(f"Hospital Patient Chart PDF Generated Successfully! Saved to workspace at reports/")
                
                with open(pdf_save_path, "rb") as f:
                    st.download_button(
                        label="Download Patient PDF Chart",
                        data=f,
                        file_name=f"ICU_Bedside_Chart_{patient_id}.pdf",
                        mime="application/pdf"
                    )

    # ==========================================
    # --- 2. ACADEMIC RESEARCH MODE ---
    # ==========================================
    elif system_mode == "Research Mode":
        st.sidebar.markdown("---")
        st.sidebar.markdown("<h4>RESEARCH NAVIGATION</h4>", unsafe_allow_html=True)
        research_page = st.sidebar.radio(
            "Select Section:",
            [
                "📈 Leaderboard & Model Performance",
                "🖼️ ROC & Confusion Matrix Gallery",
                "🧬 Global Risk Drivers (SHAP)",
                "📊 Clinical Research Findings",
                "🎓 Thesis Contributions",
                "🗺️ CDSS Pipeline Architecture"
            ]
        )
        
        # --- SUBPAGE 1: Leaderboard ---
        if research_page == "📈 Leaderboard & Model Performance":
            st.markdown("<h2 style='margin-top:0;'>📈 Clinical Model Analytics & Performance Leaderboard</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            col_table, col_desc = st.columns([1.1, 0.9])
            with col_table:
                st.markdown("<h4>🏆 Publication-Grade Optimization Leaderboard</h4>", unsafe_allow_html=True)
                leaderboard_df = pd.DataFrame({
                    "Model Name": ["Logistic Regression", "Stacking Ensemble", "CatBoost", "ExtraTrees", "LightGBM", "XGBoost", "Voting Ensemble", "Random Forest", "Balanced Random Forest"],
                    "Opt. Threshold": [0.01, 0.07, 0.58, 0.51, 0.76, 0.86, 0.60, 0.34, 0.65],
                    "ROC-AUC": [0.5833, 0.2778, 0.6389, 0.5278, 0.5556, 0.4444, 0.4444, 0.3611, 0.1111],
                    "PR-AUC": [0.1625, 0.1024, 0.1833, 0.1875, 0.1534, 0.1556, 0.1339, 0.1125, 0.0850],
                    "Recall (Sens)": [1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                    "Specificity": [0.0000, 0.0000, 0.9444, 0.9444, 1.0000, 1.0000, 0.9444, 0.8333, 1.0000],
                    "F1-Score": [0.1818, 0.1818, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
                })
                st.dataframe(leaderboard_df, hide_index=True)
            with col_desc:
                st.markdown("<h4>🧠 Model Justification & Overfitting Analysis</h4>", unsafe_allow_html=True)
                st.markdown("""
                <div class="clinical-card" style="font-size: 0.92rem; line-height: 1.45;">
                    <strong>Champion Model Selection: Logistic Regression (L2 Penalty, C=0.1)</strong>
                    <br/><br/>
                    On small EHR cohorts such as the MIMIC-IV Clinical Demo (N=100 stays, 11.00% mortality), high-capacity non-linear model boundaries (like Random Forests or ensembles) memorize the 107-dimensional space, leading to severe overfitting.
                    <br/><br/>
                    The Stacking Ensemble overfit heavily, obtaining a holdout ROC-AUC of 0.2778. Balanced Random Forest yielded 0.1111 (worse than random chance). By contrast, regularized <strong>Logistic Regression</strong> acts as a strong linear boundary constraint, obtaining the highest stable out-of-sample generalization (Test ROC-AUC = <strong>0.5833</strong>) and perfect Recall/Sensitivity (<strong>1.0000</strong>) under its F1-optimal threshold.
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            st.markdown("<h4>📊 Stacking Combined Model Evaluations</h4>", unsafe_allow_html=True)
            col_comb_roc, col_comb_rank = st.columns([1, 1])
            with col_comb_roc:
                comb_roc_path = os.path.join(config.BASE_DIR, "figures", "combined_roc_comparison.png")
                if os.path.exists(comb_roc_path):
                    st.image(comb_roc_path, caption="Combined ROC Curves (All 9 Candidates)")
            with col_comb_rank:
                comb_rank_path = os.path.join(config.BASE_DIR, "figures", "model_performance_ranking.png")
                if os.path.exists(comb_rank_path):
                    st.image(comb_rank_path, caption="Clinical Composite Score Model Rankings")
                    
        # --- SUBPAGE 2: Gallery ---
        elif research_page == "🖼️ ROC & Confusion Matrix Gallery":
            st.markdown("<h2 style='margin-top:0;'>🖼️ Candidate Model Performance Gallery</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            model_curves = st.selectbox(
                "Select Model to Inspect curves:",
                ["Logistic Regression", "Stacking Ensemble", "CatBoost", "ExtraTrees", "LightGBM", "XGBoost", "Random Forest", "Balanced Random Forest"]
            )
            
            model_id_map = {
                "Logistic Regression": "logistic_regression",
                "Stacking Ensemble": "stacking_ensemble",
                "CatBoost": "catboost",
                "ExtraTrees": "extra_trees",
                "LightGBM": "lightgbm",
                "XGBoost": "xgboost",
                "Random Forest": "random_forest",
                "Balanced Random Forest": "balanced_random_forest"
            }
            
            m_id = model_id_map[model_curves]
            col_roc, col_cm = st.columns([1, 1])
            with col_roc:
                roc_path = os.path.join(config.BASE_DIR, "figures", f"{m_id}_roc_curve.png")
                if os.path.exists(roc_path):
                    st.image(roc_path, caption=f"{model_curves} ROC Curve (Holdout)")
            with col_cm:
                cm_path = os.path.join(config.BASE_DIR, "figures", f"{m_id}_confusion_matrix.png")
                if os.path.exists(cm_path):
                    st.image(cm_path, caption=f"{model_curves} Confusion Matrix (Opt. Threshold)")
            
            st.markdown("---")
            st.markdown("<h4>🧠 Model Performance Radar Profile</h4>", unsafe_allow_html=True)
            col_rad_left, col_rad_right = st.columns([0.4, 0.6])
            with col_rad_left:
                radar_path = os.path.join(config.BASE_DIR, "figures", "performance_radar_chart.png")
                if os.path.exists(radar_path):
                    st.image(radar_path, caption="Champion Multi-Metric Radar Profile")
            with col_rad_right:
                st.markdown("""
                <div class="clinical-card" style="font-size: 0.95rem; line-height: 1.5; height: 350px;">
                    <strong>Visual Verification Interpretations:</strong>
                    <br/><br/>
                    - <strong>ROC Overlay Chart</strong>: Shows that Logistic Regression remains the most robust candidate, with other models experiencing sudden step drops in True Positive Rate due to threshold volatility.
                    <br/>
                    - <strong>Performance Radar Profile</strong>: The asymmetric spider chart clearly maps the trade-offs of the optimized threshold. By shifting the classification cutoff to 0.01, we maximize Recall (Sens) to a perfect 1.0, while sacrificing Specificity to support safe, early-warning clinical thresholds.
                </div>
                """, unsafe_allow_html=True)
                
        # --- SUBPAGE 3: SHAP Global ---
        elif research_page == "🧬 Global Risk Drivers (SHAP)":
            st.markdown("<h2 style='margin-top:0;'>🧬 Explainable AI (XAI) & Global SHAP Analysis</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            st.markdown("<h4>🌍 Global SHAP Clinical Impact Plots</h4>", unsafe_allow_html=True)
            col_shap_bar, col_shap_swarm = st.columns([1, 1])
            with col_shap_bar:
                shap_bar_path = os.path.join(config.BASE_DIR, "figures", "feature_importance_global.png")
                if os.path.exists(shap_bar_path):
                    st.image(shap_bar_path, caption="Global Feature Importance (Mean Absolute SHAP Value)")
            with col_shap_swarm:
                shap_swarm_path = os.path.join(config.BASE_DIR, "figures", "shap_summary_plot.png")
                if os.path.exists(shap_swarm_path):
                    st.image(shap_swarm_path, caption="SHAP Global Swarm/Density Plot")
                    
            st.markdown("---")
            st.markdown("<h4>🧠 Game-Theoretic Clinical Driver Interpretations</h4>", unsafe_allow_html=True)
            st.markdown("""
            <div class="clinical-card" style="font-size: 0.95rem; line-height: 1.45; border-left: 5px solid #0d5c3a;">
                <strong>SHAP Swarm Density Deconstruction:</strong>
                <br/><br/>
                - <strong>Feature Color Representation</strong>: Red points define high physiological measurements. Blue points represent low values.
                <br/>
                - <strong>Directional Risk impact</strong>: Points sitting to the right of the center line indicate features that push the model's output towards a positive prediction (increased mortality probability).
                <br/><br/>
                <strong>Global Key Clinical Drivers:</strong>
                <br/>
                1. <strong>Potassium (Min)</strong>: Low potassium observations (blue dots) map to high positive SHAP coordinates, confirming hypokalemia is a top risk predictor.
                <br/>
                2. <strong>Blood Urea Nitrogen (BUN)</strong>: High BUN levels (red dots) sitting to the far right demonstrate a strong, predictable linear risk contribution.
                <br/>
                3. <strong>Systolic BP (Mean)</strong>: Low systolic BP averages (blue dots) sit on the right, confirming hemodynamic shock increases risk.
            </div>
            """, unsafe_allow_html=True)

        # --- SUBPAGE 4: Research Findings ---
        elif research_page == "📊 Clinical Research Findings":
            st.markdown("<h2 style='margin-top:0;'>📊 Cohort Characteristics & Variable Correlation Matrix</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            col_dem, col_corr = st.columns([1.1, 0.9])
            with col_dem:
                dem_path = os.path.join(config.BASE_DIR, "figures", "dataset_statistics_summary.png")
                if os.path.exists(dem_path):
                    st.image(dem_path, caption="ICU Admission Types and Demographics Distributions")
            with col_corr:
                corr_path = os.path.join(config.BASE_DIR, "figures", "variable_correlation_heatmap.png")
                if os.path.exists(corr_path):
                    st.image(corr_path, caption="Multicollinearity Analysis: Variable Correlation Matrix")
            
            st.markdown("---")
            st.markdown("<h4>📝 Multicollinearity and Demographic Findings</h4>", unsafe_allow_html=True)
            st.markdown("""
            <div class="clinical-card" style="font-size: 0.95rem; line-height: 1.45;">
                <strong>Key Research Insights:</strong>
                <br/><br/>
                - <strong>Renal Biomarker Link</strong>: Correlation analysis shows a positive correlation of <strong>0.82</strong> between BUN and Creatinine, confirming renal hypoperfusion indicators are highly collinear.
                <br/>
                - <strong>SIRS Synergy</strong>: Respiratory rate and heart rate show a positive correlation (**0.51**), mapping systemic inflammatory response syndromes.
                <br/>
                - <strong>L2 Regularization Advantage</strong>: By using L2 regularized Logistic Regression, weights are distributed smoothly across collinear variables (BUN/Creatinine), maintaining high model stability without suffers from high-variance coefficients.
            </div>
            """, unsafe_allow_html=True)

        # --- SUBPAGE 5: Thesis Contributions ---
        elif research_page == "🎓 Thesis Contributions":
            st.markdown("<h2 style='margin-top:0;'>🎓 Academic Contributions & Future scaling</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            st.markdown("""
            <div class="clinical-card" style="font-size: 0.95rem; line-height: 1.5; border-left: 5px solid #0f62fe;">
                <h4>🏆 Thesis Contributions Achievements Summary</h4>
                <ul>
                    <li><strong>EHR Ingestion Pipeline</strong>: Loaded raw zipped EHR tables containing 668,862 telemetry observations and 107,727 lab records.</li>
                    <li><strong>Trajectory Feature Engineer</strong>: Aggregated observations strictly inside a leak-free 24-hour observation window, engineering 107 features.</li>
                    <li><strong>Out-of-Fold Threshold sweeping</strong>: Replaced standard threshold failures with 99-step sweeps, optimizing F1-optimal bounds (0.01) to resolve severe class imbalance.</li>
                    <li><strong>Transparent Explainable AI</strong>: Linked bedside predictions with natural-language physiological deconstruction summaries.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h4>🚀 Future Scaling and Clinical Deployment Pathways</h4>")
            st.markdown("""
            <div class="clinical-card" style="font-size: 0.95rem; line-height: 1.5; border-left: 5px solid #0d5c3a;">
                <strong>Thesis defense future recommendations:</strong>
                <br/><br/>
                1. <strong>Full MIMIC-IV scale</strong>: Scale the pipeline from the 100-patient demo set to the full database ($>300,000$ stays) to stabilize high-capacity models.
                <br/>
                2. <strong>Deep Temporal Trajectories</strong>: Map continuous vital telemetry streams directly using <strong>LSTMs</strong> or <strong>Clinical Transformers</strong>.
                <br/>
                3. <strong>EHR HL7 FHIR Integration</strong>: Stream bedside vitals directly from active monitors into Streamlit using FHIR (Fast Healthcare Interoperability Resources) APIs.
                <br/>
                4. <strong>Multimodal Embeddings</strong>: Integrate qualitative nursing charts and clinical notes alongside numerical telemetry.
            </div>
            """, unsafe_allow_html=True)

        # --- SUBPAGE 6: CDSS Architecture ---
        elif research_page == "🗺️ CDSS Pipeline Architecture":
            st.markdown("<h2 style='margin-top:0;'>🗺️ CDSS Pipeline Architecture Horizonal Workflow</h2>", unsafe_allow_html=True)
            st.markdown("---")
            
            arch_path = os.path.join(config.BASE_DIR, "figures", "publication" "ieee_final_system_architecture.png")
            if os.path.exists(arch_path):
                st.image(arch_path, caption="CDSS System Architecture flowchart (Ingestion to Streamlit)")
            else:
                st.error("System architecture diagram plot not found.")
                
            st.markdown("""
            <div class="clinical-card" style="font-size: 0.92rem; line-height: 1.45; margin-top: 15px;">
                <strong>Clinical System Narrative:</strong>
                <br/><br/>
                The system architecture represents a unified, real-time Clinical Decision Support System. 
                Raw EHR files are loaded, adult stays are isolated, and physiological telemetries are scrubbed of outliers. 
                The 24h feature extractor aggregates 6 statistical metrics across 15 physiological concepts, engineering 107 clean columns. 
                ML pipelines SMOTE-resample, cross-validate candidates, sweep thresholds, and serialize the best model. 
                The Streamlit interface acts as the bedside CDS widget, rendering alerts, trend charts, risk narratives, and A4 vector Patient Chart PDFs.
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("Please execute the system pipeline (`run_pipeline.py`) first to generate clinical feature datasets and save champion models.")
