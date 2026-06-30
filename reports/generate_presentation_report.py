import os
import sys
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Set up paths relative to current script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

pdf_save_path = os.path.join(REPORTS_DIR, "Project_Presentation_Guide.pdf")

def add_header_footer(fig, slide_title, slide_num, total_slides=9):
    """Draws the dark blue header banner and subtle footer band on a slide."""
    # Header Banner
    ax_header = fig.add_axes([0, 0.88, 1, 0.12])
    ax_header.axis('off')
    ax_header.fill_between([0, 1], [0, 0], [1, 1], color='#0f62fe') # IBM Carbon Blue
    ax_header.text(0.04, 0.5, slide_title.upper(), color='white', fontsize=18, fontweight='bold', va='center')
    ax_header.text(0.96, 0.5, "ICU MORTALITY PREDICTION CDSS", color='#d1fae5', fontsize=10, fontweight='bold', ha='right', va='center')
    
    # Footer Band
    ax_footer = fig.add_axes([0, 0, 1, 0.06])
    ax_footer.axis('off')
    ax_footer.plot([0.04, 0.96], [0.8, 0.8], color='#e0e0e0', lw=1)
    ax_footer.text(0.04, 0.3, "Presenter Study Guide & Oral Examination Reference Sheet", color='#8d8d8d', fontsize=8, va='center')
    ax_footer.text(0.96, 0.3, f"Page {slide_num} of {total_slides}", color='#8d8d8d', fontsize=8, ha='right', va='center')

def draw_bullet_points(ax, points, x_start=0.0, y_start=0.95, font_size=9.5, spacing=0.12):
    """Draws nicely spaced multiline bullet points with colored bullet marks."""
    y = y_start
    for p in points:
        # Check if it's a section header or bullet point
        if p.startswith("##"):
            # Subtitle
            text_clean = p.replace("##", "").strip()
            ax.text(x_start, y, text_clean, color='#0f62fe', fontsize=font_size+1.5, fontweight='bold', va='top')
            y -= spacing * 0.9
        else:
            # Bullet
            ax.text(x_start, y, "•", color='#ef4444', fontsize=font_size+3, fontweight='bold', va='top')
            # Handle text wrap
            text_clean = p.strip()
            # Wrap text manually if too long
            words = text_clean.split(" ")
            lines = []
            current_line = []
            for w in words:
                if len(" ".join(current_line + [w])) * (font_size * 0.04) > 4.2: # width boundary
                    lines.append(" ".join(current_line))
                    current_line = [w]
                else:
                    current_line.append(w)
            lines.append(" ".join(current_line))
            
            # Draw wrapped lines
            for i, line in enumerate(lines):
                ax.text(x_start + 0.03, y - (i * font_size * 0.0022), line, color='#12161a', fontsize=font_size, va='top', linespacing=1.3)
            y -= spacing * (1 + 0.25 * (len(lines) - 1))

def embed_image(fig, img_filename, rect):
    """Safely reads a PNG and draws it in the specified axes rect coordinates."""
    img_path = os.path.join(FIGURES_DIR, img_filename)
    if os.path.exists(img_path):
        ax_img = fig.add_axes(rect)
        img = plt.imread(img_path)
        ax_img.imshow(img)
        ax_img.axis('off')
        # Draw soft thin border
        rect_patch = patches.Rectangle((0, 0), 1, 1, transform=ax_img.transAxes, color='#e0e0e0', fill=False, lw=1)
        ax_img.add_patch(rect_patch)
        return True
    return False

def build_pdf():
    print(f"Compiling project presentation guide PDF to {pdf_save_path}...")
    
    with PdfPages(pdf_save_path) as pp:
        total_pages = 9
        
        # ---------------------------------------------
        # PAGE 1: TITLE SLIDE
        # ---------------------------------------------
        fig = plt.figure(figsize=(13.33, 7.5), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        # Background graphics
        ax.fill_between([0, 1], [0, 0], [1, 1], color='#f4f4f4')
        ax.fill_between([0, 0.45], [0, 0], [1, 1], color='#0f62fe', alpha=0.9)
        
        # Left Panel (Title branding)
        ax.text(0.05, 0.75, "EXPLAINABLE AI (XAI)\nCLINICAL DECISION\nSUPPORT SYSTEM", color='white', fontsize=26, fontweight='bold', va='top', linespacing=1.2)
        ax.text(0.05, 0.38, "ICU Mortality Prediction Pipeline\nUsing the MIMIC-IV Clinical Database", color='#d1fae5', fontsize=12, fontweight='bold', va='top')
        
        # Details Box
        details = """COURSE: Final Year Engineering Project
EVALUATION BOARD: Oral Presentation & Defense
TOPIC: Deep Clinical Machine Learning & Interpretability
CHAMPION MODEL: Youden-Optimized ExtraTrees
UI PLATFORM: Streamlit Clinical Bedside App"""
        ax.text(0.05, 0.25, details, color='white', fontsize=8.5, fontfamily='monospace', va='top', linespacing=1.4)
        
        # Right Panel (Overview and Outline)
        ax.text(0.5, 0.82, "PROJECT PRESENTATION & ORAL DEFENSE MANUAL", color='#0f62fe', fontsize=16, fontweight='bold', va='top')
        ax.text(0.5, 0.78, "Designed for rapid student preparation and examiner walkthroughs.", color='#8d8d8d', fontsize=9.5, va='top')
        
        bullets = [
            "## 💡 How to Use This Document:",
            "Use the left text columns as your oral preparation cheat-sheet. It contains the exact technical justifications you need to defend your project choices.",
            "Show the right column diagrams directly to your evaluator (Mam) to explain data flow, modeling results, and bedside recommendations.",
            "Review the Part 3 questions at the end of the markdown guide to practice handling critical clinical and mathematical challenges.",
            "Ensure the Streamlit web dashboard is running locally during your evaluation so you can showcase the live predictions."
        ]
        
        ax_bullets = fig.add_axes([0.5, 0.15, 0.45, 0.58])
        ax_bullets.axis('off')
        draw_bullet_points(ax_bullets, bullets, x_start=0.0, y_start=0.95, font_size=10, spacing=0.15)
        
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # ---------------------------------------------
        # PAGE 2: SYSTEM ARCHITECTURE
        # ---------------------------------------------
        fig = plt.figure(figsize=(13.33, 7.5), dpi=300)
        add_header_footer(fig, "1. System Architecture & Flowchart", 2, total_pages)
        
        ax_left = fig.add_axes([0.04, 0.08, 0.42, 0.76])
        ax_left.axis('off')
        
        bullets = [
            "## 🛠️ System Design & Layers:",
            "Clinical Data Acquisition Layer: Ingests patient demographics, vital telemetry charts, and clinical blood laboratory databases from the MIMIC-IV registry.",
            "Clinical Intelligence Layer: Cleans telemetry sensor outliers, maps observations to the first 24 hours of ICU stay, aggregates 6 statistical features, and trains 9 ML models.",
            "Clinical Decision Support Layer: Integrates risk classifiers, computes game-theoretic local/global SHAP explanations, renders bedside advisories, and exports PDF charts.",
            "## 📁 Project Code Directories:",
            "- Ingestion & Quality: data_ingestion.py & preprocessing.py",
            "- Features & Training: feature_engineering.py & train.py",
            "- Interpretability: explainability.py",
            "- Bedside UI & PDFs: app.py, utils.py & pdf_generator.py"
        ]
        draw_bullet_points(ax_left, bullets, spacing=0.10)
        
        # Embed Ingestion/Architecture flowchart
        embed_image(fig, "system_architecture_diagram.png", [0.49, 0.12, 0.47, 0.72])
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---------------------------------------------
        # PAGE 3: COHORT SELECTION & PREPROCESSING
        # ---------------------------------------------
        fig = plt.figure(figsize=(13.33, 7.5), dpi=300)
        add_header_footer(fig, "2. Clinical Ingestion & Quality scrubbing", 3, total_pages)
        
        ax_left = fig.add_axes([0.04, 0.08, 0.42, 0.76])
        ax_left.axis('off')
        
        bullets = [
            "## 🏥 Baseline Cohort Selection:",
            "Filter to First Stays: Selects only the first ICU admission per patient. This is standard in clinical ML to prevent data leakage and correlation bias from repeat stays.",
            "Adult Filter: Excludes pediatric cases (age < 18). Exact age is calculated as anchor_age + (intime_year - anchor_year).",
            "Target Flag: Extracts hospital_expire_flag (mortality flag) from admissions table as the ground truth label.",
            "## 🧹 Physiological Outlier Cleaning:",
            "Unit Unification: Fahrenheit temperature readings (item ID 223761) are normalized to Celsius: C = (F - 32) * 5/9, and merged with Celsius measurements (item ID 223762).",
            "Outlier Bound Filters: Removes telemetry noise and sensor artifacts using medical limits (e.g. Heart Rate: 30-220 bpm, SpO2: 50-100%, Temperature: 32-42°C). Invalid values are set to NaN."
        ]
        draw_bullet_points(ax_left, bullets, spacing=0.11)
        
        # Embed Ingestion/Demographics chart
        embed_image(fig, "dataset_statistics_summary.png", [0.49, 0.12, 0.47, 0.72])
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---------------------------------------------
        # PAGE 4: FEATURE ENGINEERING & IMPUTATION
        # ---------------------------------------------
        fig = plt.figure(figsize=(13.33, 7.5), dpi=300)
        add_header_footer(fig, "3. Time-Windowed Feature Engineering", 4, total_pages)
        
        ax_left = fig.add_axes([0.04, 0.08, 0.42, 0.76])
        ax_left.axis('off')
        
        bullets = [
            "## ⏱️ Early 24h Observation Window:",
            "To support early clinical warning, features are restricted strictly to telemetry and laboratory events recorded within the first 24 hours of ICU admission.",
            "## 📊 6 Multimodal Aggregations:",
            "For all vitals and labs, we extract: Mean (baseline), Min (acute dip), Max (peak crisis), Std (instability/volatility), Latest (exit status), and Trend (trajectory: Latest - First).",
            "## 🧬 Clinical Imputation Strategy:",
            "Vital signs: Filled with global cohort medians.",
            "Laboratory Panels: Filled with healthy reference standards (e.g. WBC = 7.5 K/uL, Potassium = 4.2 mEq/L). Using healthy baselines represents clinical normalcy unless observed otherwise, avoiding bias.",
            "Informative Missingness: Binary flags (is_missing_<lab_name>) are kept, since ordering a lab test is itself a strong indicator of patient status."
        ]
        draw_bullet_points(ax_left, bullets, spacing=0.09)
        
        # Embed correlation heatmap
        embed_image(fig, "variable_correlation_heatmap.png", [0.49, 0.12, 0.47, 0.72])
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---------------------------------------------
        # PAGE 5: CANDIDATE CLASSIFIERS & CV
        # ---------------------------------------------
        fig = plt.figure(figsize=(13.33, 7.5), dpi=300)
        add_header_footer(fig, "4. Machine Learning & Ensemble Pipelines", 5, total_pages)
        
        ax_left = fig.add_axes([0.04, 0.08, 0.42, 0.76])
        ax_left.axis('off')
        
        bullets = [
            "## 🛡️ Preventing Overfitting (p > N):",
            "With only 100-140 stays, training all 107 raw features causes high-variance overfitting. We constrain the pipeline to 9 biologically validated indicators: Age, Heart Rate Std, SpO2 Mean, Systolic BP Mean, MAP Min, Temperature Mean, BUN Mean, Creatinine Latest, and Potassium Min.",
            "## 🔄 Stratified 5-Fold Cross-Validation:",
            "Splitting is stratified by the target label to maintain identical deceased-to-survived proportions in each fold, preventing evaluation bias on imbalanced data.",
            "## 🎭 Models Evaluated:",
            "Base Models: Logistic Regression, Random Forest, Balanced Random Forest, XGBoost, LightGBM, CatBoost, ExtraTrees.",
            "Ensembles: Soft-Voting (weighted average) and Stacking Ensemble (combining candidates with a Logistic Regression meta-learner)."
        ]
        draw_bullet_points(ax_left, bullets, spacing=0.11)
        
        # Embed combined model comparison / ranking chart
        embed_image(fig, "model_performance_ranking.png", [0.49, 0.12, 0.47, 0.72])
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---------------------------------------------
        # PAGE 6: THRESHOLD OPTIMIZATION (YOUDEN'S J)
        # ---------------------------------------------
        fig = plt.figure(figsize=(13.33, 7.5), dpi=300)
        add_header_footer(fig, "5. Youden's J Threshold Optimization", 6, total_pages)
        
        ax_left = fig.add_axes([0.04, 0.08, 0.42, 0.76])
        ax_left.axis('off')
        
        bullets = [
            "## ⚖️ The Class Imbalance Challenge:",
            "Only ~11% of the cohort deceased. Standard classifiers trained on this dataset default to predicting 0 (survival) for all records to maximize simple accuracy.",
            "This leads to a Recall (Sensitivity) of 0%, which is clinically unacceptable (the model fails to identify any patient at risk).",
            "## 📈 Youden's J Threshold Sweep:",
            "We sweep thresholds from 0.1 to 0.9 on Out-of-Fold (OOF) validation sets, finding the threshold maximizing Youden's J Index: Sensitivity + Specificity - 1 (equivalent to Balanced Accuracy).",
            "We average these fold-specific thresholds to obtain a robust global threshold, preventing overfitting to train/test splits.",
            "For the ExtraTrees model, the threshold is optimized to 0.39, boosting recall to 50.00% on unseen test data."
        ]
        draw_bullet_points(ax_left, bullets, spacing=0.10)
        
        # Embed threshold sweep plot for ExtraTrees
        embed_image(fig, "extra_trees_threshold_optimization.png", [0.49, 0.12, 0.47, 0.72])
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---------------------------------------------
        # PAGE 7: MODEL EVALUATION RESULTS
        # ---------------------------------------------
        fig = plt.figure(figsize=(13.33, 7.5), dpi=300)
        add_header_footer(fig, "6. Champion Model Evaluation Results", 7, total_pages)
        
        ax_left = fig.add_axes([0.04, 0.08, 0.42, 0.76])
        ax_left.axis('off')
        
        bullets = [
            "## 🏆 Champion Predictor Selection:",
            "Ranked by Clinical Score (PR-AUC + ROC-AUC + Recall) on the unseen holdout test split. ExtraTrees with optimized threshold 0.39 was selected as the champion model.",
            "Holdout Test ROC-AUC: 0.5556",
            "Holdout Test Recall (Sensitivity): 0.5000",
            "Holdout Test Specificity: 0.5556",
            "## 🩺 Clinical Trade-offs:",
            "High sensitivity is crucial to ensure high-risk patients are flagged early. Standard models focus too much on accuracy, ignoring deceased cases.",
            "Our threshold optimization successfully solves this issue. The Stacking Ensemble suffered from high complexity and overfit on the small test set, validating the choice of regularized trees."
        ]
        draw_bullet_points(ax_left, bullets, spacing=0.11)
        
        # Embed combined ROC curves
        embed_image(fig, "combined_roc_comparison.png", [0.49, 0.12, 0.47, 0.72])
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---------------------------------------------
        # PAGE 8: EXPLAINABLE AI (SHAP GLOBAL)
        # ---------------------------------------------
        fig = plt.figure(figsize=(13.33, 7.5), dpi=300)
        add_header_footer(fig, "7. Explainable AI & Global Risk Drivers", 8, total_pages)
        
        ax_left = fig.add_axes([0.04, 0.08, 0.42, 0.76])
        ax_left.axis('off')
        
        bullets = [
            "## 🧠 The SHAP Interpretability Framework:",
            "Clinicians distrust 'black-box' models. We use SHAP (SHapley Additive exPlanations) from cooperative game theory to make predictions transparent.",
            "SHAP calculates the exact contribution (payout) of each vital sign/lab feature to a patient's mortality risk prediction.",
            "## 🌍 Global Risk Drivers Across the Cohort:",
            "1. Potassium (Min): Low potassium levels (hypokalemia) map to high positive SHAP coordinates, indicating cardiac instability.",
            "2. Blood Urea Nitrogen (BUN): Elevated BUN values show a direct, linear risk contribution, indicating renal strain.",
            "3. Systolic BP (Mean): Low blood pressure averages contribute heavily to mortality risk, signaling hemodynamic shock."
        ]
        draw_bullet_points(ax_left, bullets, spacing=0.10)
        
        # Embed SHAP summary swarm plot
        embed_image(fig, "shap_summary_plot.png", [0.49, 0.12, 0.47, 0.72])
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ---------------------------------------------
        # PAGE 9: BEDSIDE DASHBOARD & PDF CHART
        # ---------------------------------------------
        fig = plt.figure(figsize=(13.33, 7.5), dpi=300)
        add_header_footer(fig, "8. Interactive CDS Dashboard & PDF Chart", 9, total_pages)
        
        ax_left = fig.add_axes([0.04, 0.08, 0.42, 0.76])
        ax_left.axis('off')
        
        bullets = [
            "## 🖥️ Streamlit Clinician Dashboard:",
            "Demo Patient Registry: Selects a real patient stay from the dataset. Loads their 24h demographics, vitals, and laboratory panels.",
            "Physiological Slider Builder: Let's a physician manually adjust bedside sliders (age, heart rate, BP, SpO2, creatinine, BUN) to test virtual scenarios.",
            "Actionable Alert Bands: Low (<5%), Moderate (<15%), High (<40%), and Critical (>=40%) risk levels trigger specific hospital alert protocols.",
            "Local SHAP Waterfall: Renders patient-specific waterfall charts showing the exact biomarkers driving their risk up or down.",
            "## 📄 Hospital PDF Export:",
            "Compiles patient metrics, calculated risks, SHAP narratives, rule-based clinical recommendations, and a signature line into a vector A4 PDF."
        ]
        draw_bullet_points(ax_left, bullets, spacing=0.09)
        
        # Embed Sample Patient waterfall chart
        embed_image(fig, "sample_patient_shap_waterfall.png", [0.49, 0.12, 0.47, 0.72])
        pp.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    print(f"Presentation Guide PDF generated successfully! Saved to: {pdf_save_path}")

if __name__ == "__main__":
    try:
        build_pdf()
        sys.exit(0)
    except Exception as e:
        print(f"Error generating PDF: {e}", file=sys.stderr)
        sys.exit(1)
