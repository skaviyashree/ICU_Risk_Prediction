import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure workspace is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

def generate_clinical_pdf_report(
    patient_id, 
    demographics, 
    risk_pct, 
    risk_category, 
    risk_color,
    nlp_summary, 
    recommendations,
    save_path=None
):
    """
    Generates a publication-grade, hospital-style Patient Clinical Risk Assessment PDF 
    using Matplotlib's highly stable vector rendering backend.
    """
    if save_path is None:
        save_path = os.path.join(config.BASE_DIR, "reports", "patient_report.pdf")
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create matplotlib figure (A4 size ratio approx: 8.27 x 11.69 inches)
    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Define vertical layout markers
    y_top = 0.96
    
    # 1. Hospital Header Band
    ax.fill_between([0.0, 1.0], [y_top-0.08, y_top-0.08], [y_top, y_top], color='#0f62fe', alpha=0.9, transform=ax.transAxes)
    ax.text(0.05, y_top-0.03, "SACRED HEART METROPOLITAN HOSPITAL", ha="left", va="center", color="white", fontsize=14, fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, y_top-0.06, "ICU CRITICAL CARE DIVISION — PATIENT RISK REGISTRY CHART", ha="left", va="center", color="#d1fae5", fontsize=10, fontweight="bold", transform=ax.transAxes)
    
    # Date & Timestamp
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.95, y_top-0.04, f"Report Generated:\n{now_str}", ha="right", va="center", color="white", fontsize=8, transform=ax.transAxes)
    
    # 2. Section: Patient Demographic Profile
    y_demo = y_top - 0.12
    ax.text(0.05, y_demo, "I. PATIENT DEMOGRAPHIC PROFILE", ha="left", va="center", color="#12161a", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.plot([0.05, 0.95], [y_demo-0.01, y_demo-0.01], color="#0f62fe", lw=1.5, transform=ax.transAxes)
    
    # Render Demographic Box
    ax.fill_between([0.05, 0.95], [y_demo-0.12, y_demo-0.12], [y_demo-0.02, y_demo-0.02], color="#f4f4f4", ec="none", transform=ax.transAxes)
    
    col1 = f"""PATIENT STAY ID:   {patient_id}
AGE / GENDER:      {demographics.get('age', 'N/A')} yrs / {demographics.get('gender', 'N/A')}
ADMISSION TYPE:    {demographics.get('admission_type', 'N/A')}"""
    
    col2 = f"""ETHNICITY / RACE:   {demographics.get('race', 'N/A')}
INSURANCE PLAN:    {demographics.get('insurance', 'N/A')}
MARITAL STATUS:    {demographics.get('marital_status', 'N/A')}"""
    
    ax.text(0.08, y_demo-0.07, col1, ha="left", va="center", color="#12161a", fontsize=9, fontfamily="monospace", transform=ax.transAxes)
    ax.text(0.52, y_demo-0.07, col2, ha="left", va="center", color="#12161a", fontsize=9, fontfamily="monospace", transform=ax.transAxes)
    
    # 3. Section: Early-Warning Clinical Assessment
    y_assess = y_demo - 0.16
    ax.text(0.05, y_assess, "II. MORTALITY RISK CLINICAL ASSESSMENT (24-HOUR INTERCEPT)", ha="left", va="center", color="#12161a", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.plot([0.05, 0.95], [y_assess-0.01, y_assess-0.01], color="#0f62fe", lw=1.5, transform=ax.transAxes)
    
    # Render Large Risk Probability Dial Box
    ax.fill_between([0.05, 0.95], [y_assess-0.12, y_assess-0.12], [y_assess-0.02, y_assess-0.02], color="#fafafa", ec="#e0e0e0", lw=1, transform=ax.transAxes)
    
    ax.text(0.25, y_assess-0.07, f"{risk_pct:.2f}%", ha="center", va="center", color=risk_color, fontsize=32, fontweight="bold", transform=ax.transAxes)
    ax.text(0.25, y_assess-0.10, "In-Hospital Mortality Risk", ha="center", va="center", color="#8d8d8d", fontsize=8, fontweight="bold", transform=ax.transAxes)
    
    ax.text(0.70, y_assess-0.05, "SEVERITY LEVEL BANDS:", ha="center", va="center", color="#12161a", fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(0.70, y_assess-0.08, risk_category.upper(), ha="center", va="center", color=risk_color, fontsize=16, fontweight="bold", transform=ax.transAxes)
    ax.text(0.70, y_assess-0.10, "Logistic Regression early-warning pipeline", ha="center", va="center", color="#8d8d8d", fontsize=7, transform=ax.transAxes)
    
    # 4. Section: Dynamic Explainable AI Risk Drivers (SHAP)
    y_xai = y_assess - 0.16
    ax.text(0.05, y_xai, "III. DYNAMIC EXPLAINABLE AI (XAI) BEDSIDE DRIVERS", ha="left", va="center", color="#12161a", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.plot([0.05, 0.95], [y_xai-0.01, y_xai-0.01], color="#0f62fe", lw=1.5, transform=ax.transAxes)
    
    # Wrap text nicely for NLP explanation
    import textwrap
    wrapped_nlp = textwrap.fill(nlp_summary, width=95)
    
    ax.text(0.05, y_xai-0.07, wrapped_nlp, ha="left", va="center", color="#12161a", fontsize=9.5, transform=ax.transAxes)
    
    # 5. Section: Actionable Clinical Bedside Recommendations
    y_recs = y_xai - 0.14
    ax.text(0.05, y_recs, "IV. ACTIONABLE CLINICAL TREATMENT ADVISORY GUIDELINES", ha="left", va="center", color="#12161a", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.plot([0.05, 0.95], [y_recs-0.01, y_recs-0.01], color="#0f62fe", lw=1.5, transform=ax.transAxes)
    
    y_bullet = y_recs - 0.04
    for r in recommendations:
        ax.text(0.05, y_bullet, "•", ha="left", va="center", color="#ef4444", fontsize=14, fontweight="bold", transform=ax.transAxes)
        ax.text(0.08, y_bullet, r, ha="left", va="center", color="#12161a", fontsize=9.5, transform=ax.transAxes)
        y_bullet -= 0.03
        
    # 6. Clinical Validation Signature Footer
    y_footer = 0.08
    ax.plot([0.05, 0.95], [y_footer+0.05, y_footer+0.05], color="#e0e0e0", lw=1, transform=ax.transAxes)
    ax.text(0.05, y_footer+0.02, "PREDICTIVE SCORE VALIDITY NOTICE:\nThis early CDSS tool uses first-stay telemetry observations and laboratory conceptual measurements from raw ICU electronic medical charts. Predicted risks must be evaluated alongside professional clinical workflows.", ha="left", va="top", color="#8d8d8d", fontsize=7, transform=ax.transAxes)
    
    ax.text(0.70, y_footer, "Physician In-Charge Signature:", ha="left", va="bottom", color="#8d8d8d", fontsize=8, transform=ax.transAxes)
    ax.plot([0.70, 0.95], [y_footer-0.02, y_footer-0.02], color="#8d8d8d", lw=1, transform=ax.transAxes)
    
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hospital-grade PDF Bedside Report generated successfully at: {save_path}")
    return save_path

if __name__ == "__main__":
    # Test generation
    test_demo = {"age": 67, "gender": "MALE", "race": "WHITE", "admission_type": "URGENT", "insurance": "Medicare", "marital_status": "MARRIED"}
    test_recs = ["High Blood Urea Nitrogen (BUN) flagged: Review renal function perfusion, verify hydration indices.", "Oxygen Saturation (SpO2) depressed (<95%): Evaluate supplementary respiratory support channels."]
    test_nlp = "The patient exhibits several high-risk telemetry and laboratory biomarkers. Most notably, an elevated Blood Urea Nitrogen (BUN) level (+0.42 SHAP) and high Heart Rate standard deviation (+0.31 SHAP) significantly push their risk score above baseline. Conversely, healthy Potassium levels (-0.12 SHAP) partially offset mortality probability."
    
    generate_clinical_pdf_report("STAY_TEST_999", test_demo, 19.83, "Moderate Risk", "#f59e0b", test_nlp, test_recs, "reports/test_patient_report.pdf")
