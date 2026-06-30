import streamlit as st

def inject_clinical_custom_css():
    """
    Injects high-fidelity glassmorphism, responsive styles, and modern typography
    custom CSS into the Streamlit rendering engine.
    """
    custom_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Glassmorphic Cards */
    .clinical-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }
    
    .light-theme .clinical-card {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(0, 0, 0, 0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.05);
    }
    
    /* Medical Status Cards */
    .status-card-low {
        background: rgba(16, 185, 129, 0.15);
        border-left: 5px solid rgb(16, 185, 129);
        border-radius: 8px;
        padding: 15px;
        color: #10b981;
    }
    
    .status-card-moderate {
        background: rgba(245, 158, 11, 0.15);
        border-left: 5px solid rgb(245, 158, 11);
        border-radius: 8px;
        padding: 15px;
        color: #f59e0b;
    }
    
    .status-card-high {
        background: rgba(239, 68, 68, 0.15);
        border-left: 5px solid rgb(239, 68, 68);
        border-radius: 8px;
        padding: 15px;
        color: #ef4444;
        font-weight: bold;
    }
    
    .status-card-critical {
        background: rgba(185, 28, 28, 0.15);
        border-left: 5px solid rgb(185, 28, 28);
        border-radius: 8px;
        padding: 15px;
        color: #b91c1c;
        font-weight: bold;
    }
    
    /* Dynamic Metric Dials */
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 5px 0;
    }
    
    .metric-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        color: #8d8d8d;
        letter-spacing: 1px;
    }
    
    /* Streamlit overrides */
    div[data-testid="stSidebar"] {
        background-color: #12161a;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_medical_alert(risk_pct, threshold_pct=15.0):
    """
    Renders a premium clinical status advisory card based on the risk percentage
    mapped to clinical risk probability bands:
      0-10%      LOW RISK
      10-30%     MODERATE RISK
      30-60%     HIGH RISK
      60-100%    CRITICAL RISK
    """
    if risk_pct <= 10.0:
        html = f"""
        <div class="status-card-low">
            <h4 style="margin: 0 0 5px 0; color: #10b981;">🟢 STABLE CLINICAL PROFILE (LOW RISK)</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #d1fae5;">
                Mortality risk probability is <strong>{risk_pct:.2f}%</strong>. The patient exhibits stable physiological baselines.
                <br/><strong>Actionable Recommendation:</strong> Maintain standard bedside telemetry monitoring and routine care.
            </p>
        </div>
        """
    elif risk_pct <= 30.0:
        html = f"""
        <div class="status-card-moderate">
            <h4 style="margin: 0 0 5px 0; color: #f59e0b;">🟡 ADVISORY ALERT (MODERATE RISK)</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #fef3c7;">
                Mortality risk probability is <strong>{risk_pct:.2f}%</strong>. Subtle vital fluctuations or electrolyte deviations.
                <br/><strong>Actionable Recommendation:</strong> Increase bedside charting frequency, review laboratory panels, and schedule physician check-in within the next 4 hours.
            </p>
        </div>
        """
    elif risk_pct <= 60.0:
        html = f"""
        <div class="status-card-high">
            <h4 style="margin: 0 0 5px 0; color: #ef4444;">🚨 INTERMEDIATE DECOMPENSATION ALERT (HIGH RISK)</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #fee2e2;">
                Mortality risk probability is <strong>{risk_pct:.2f}%</strong>. Notable respiratory or cardiovascular vital signs distress.
                <br/><strong>Actionable Recommendation:</strong> Immediate bedside physician review required. Prepare non-invasive supportive channels and verify organ perfusion markers.
            </p>
        </div>
        """
    else:
        html = f"""
        <div class="status-card-critical">
            <h4 style="margin: 0 0 5px 0; color: #b91c1c;">🔥 PHYSIOLOGICAL CRITICAL ESCALATION (CRITICAL RISK)</h4>
            <p style="margin: 0; font-size: 0.95rem; color: #fecaca;">
                Mortality risk probability is <strong>{risk_pct:.2f}%</strong>. Profound physiological failure detected.
                <br/><strong>Actionable Recommendation:</strong> Trigger immediate Rapid Response Team (RRT) bedside evaluation. Initiate aggressive cardiovascular, hemodynamic, or mechanical airway support.
            </p>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

def render_clinical_profile_card(age, gender, race, admission_type, insurance):
    """
    Generates a premium patient demographic summary panel using modern CSS.
    """
    gender_str = "MALE" if gender == 1 else "FEMALE"
    html = f"""
    <div class="clinical-card" style="margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #0f62fe;">🩺 PATIENT CLINICAL REGISTRY PROFILE</h4>
        <table style="width: 100%; font-size: 0.95rem; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.08); height: 35px;">
                <td style="color: #8d8d8d;">AGE / GENDER</td>
                <td style="text-align: right; font-weight: bold;">{age} yrs / {gender_str}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.08); height: 35px;">
                <td style="color: #8d8d8d;">ADMISSION SOURCE TYPE</td>
                <td style="text-align: right; font-weight: bold; color: #f59e0b;">{admission_type}</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.08); height: 35px;">
                <td style="color: #8d8d8d;">ETHNICITY / RACE</td>
                <td style="text-align: right; font-weight: bold;">{race}</td>
            </tr>
            <tr style="height: 35px;">
                <td style="color: #8d8d8d;">INSURANCE TYPE</td>
                <td style="text-align: right; font-weight: bold;">{insurance}</td>
            </tr>
        </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
