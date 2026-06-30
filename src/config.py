import os

# --- PATH CONFIGURATIONS ---
# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "mimic-iv-clinical-database-demo-2.2")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")

# Raw file paths (zipped CSVs)
PATH_PATIENTS = os.path.join(RAW_DATA_DIR, "hosp", "patients.csv.gz")
PATH_ADMISSIONS = os.path.join(RAW_DATA_DIR, "hosp", "admissions.csv.gz")
PATH_ICUSTAYS = os.path.join(RAW_DATA_DIR, "icu", "icustays.csv.gz")
PATH_CHARTEVENTS = os.path.join(RAW_DATA_DIR, "icu", "chartevents.csv.gz")
PATH_LABEVENTS = os.path.join(RAW_DATA_DIR, "hosp", "labevents.csv.gz")
PATH_D_ITEMS = os.path.join(RAW_DATA_DIR, "icu", "d_items.csv.gz")
PATH_D_LABITEMS = os.path.join(RAW_DATA_DIR, "hosp", "d_labitems.csv.gz")

# Processed feature and model paths
PATH_PROCESSED_FEATURES = os.path.join(PROCESSED_DATA_DIR, "processed_clinical_features.csv")
PATH_BEST_MODEL = os.path.join(MODELS_DIR, "best_model.joblib")

# --- CLINICAL ITEM ID MAPPINGS ---
# Vitals from chartevents.csv.gz
VITALS_MAP = {
    "heart_rate": [220045],
    "resp_rate": [220210],
    "spo2": [220277],
    "systolic_bp": [220179, 220050],
    "diastolic_bp": [220180, 220051],
    "map": [220052, 220181, 224326],
    "temperature": [223762, 223761]
}

# Laboratories from labevents.csv.gz
LABS_MAP = {
    "wbc": [51301, 51300],
    "hemoglobin": [51222],
    "platelets": [51265],
    "creatinine": [50912],
    "glucose": [50931, 50809],
    "sodium": [50983, 50824],
    "potassium": [50971, 50822],
    "bun": [51006, 52647]
}

# --- OUTLIER CLEANING BOUNDS ---
# Normal and extreme clinical limits to remove telemetry noise and artifacts
OUTLIER_BOUNDS = {
    "heart_rate": {"min": 30, "max": 220},
    "resp_rate": {"min": 5, "max": 60},
    "spo2": {"min": 50, "max": 100},
    "systolic_bp": {"min": 40, "max": 260},
    "diastolic_bp": {"min": 20, "max": 150},
    "map": {"min": 30, "max": 180},
    "temperature": {"min": 32, "max": 42}  # Celsius
}

# --- MACHINE LEARNING CONFIGURATIONS ---
OBSERVATION_WINDOW_HOURS = 24
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.2

# SMOTE and Class Weights Configurations
SMOTE_SAMPLING_STRATEGY = 0.5  # Bring minority to 50% of majority class size
CLASS_WEIGHT_RATIO = 13.0      # Reflects 93% to 7% severe imbalance

# Cross-Validation
CV_FOLDS = 5
