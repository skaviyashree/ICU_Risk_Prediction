import pandas as pd
import numpy as np
import os
import sys

# Append parent directory to sys.path to support importing src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

# Standard healthy clinical reference values for lab imputation
CLINICAL_REF_LABS = {
    "wbc": 7.5,        # K/uL
    "hemoglobin": 14.5, # g/dL
    "platelets": 250.0, # K/uL
    "creatinine": 0.9,  # mg/dL
    "glucose": 90.0,    # mg/dL
    "sodium": 140.0,    # mEq/L
    "potassium": 4.2,   # mEq/L
    "bun": 14.0         # mg/dL
}

def extract_clinical_features(cohort, chartevents, labevents):
    """
    Slices raw clinical recordings to the first 24 hours of ICU stay,
    computes 6 clinical aggregations for each of the 15 vitals/labs,
    tracks laboratory missingness indicator flags, and joins demographics.
    """
    print("\n--- STARTING CLINICAL FEATURE ENGINEERING STAGE ---")
    
    # Ensure correct datetimes
    cohort = cohort.copy()
    cohort['intime'] = pd.to_datetime(cohort['intime'])
    
    chartevents = chartevents.copy()
    chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
    # Unified temperature itemid in preprocessing to 223762
    
    labevents = labevents.copy()
    labevents['charttime'] = pd.to_datetime(labevents['charttime'])
    
    # Map labevents to stays to easily slice the 24h window
    lab_stays = pd.merge(
        labevents,
        cohort[['subject_id', 'hadm_id', 'stay_id', 'intime']],
        on=['subject_id', 'hadm_id'],
        how='inner'
    )
    
    feature_records = []
    
    # Loop over each patient ICU stay in the baseline cohort
    for idx, row in cohort.iterrows():
        stay_id = row['stay_id']
        subject_id = row['subject_id']
        intime = row['intime']
        t_end = intime + pd.Timedelta(hours=config.OBSERVATION_WINDOW_HOURS)
        
        # Initialize dictionary for patient stay features
        patient_feats = {
            "stay_id": stay_id,
            "subject_id": subject_id,
            "age": row['age'],
            "gender": 1 if row['gender'] == 'M' else 0, # Male = 1, Female = 0
            "race": row['race'],
            "admission_type": row['admission_type'],
            "insurance": row['insurance'],
            "marital_status": row['marital_status'] if pd.notnull(row['marital_status']) else 'Unknown',
            "hospital_expire_flag": row['hospital_expire_flag']
        }
        
        # 1. Filter Bedside Vitals (from chartevents) for this stay and first 24h window
        stay_charts = chartevents[
            (chartevents['stay_id'] == stay_id) & 
            (chartevents['charttime'] >= intime) & 
            (chartevents['charttime'] <= t_end)
        ]
        
        # Extract 6 aggregations for vitals
        for vital_name, itemids in config.VITALS_MAP.items():
            # Treat temp as 223762
            target_ids = [223762] if vital_name == "temperature" else itemids
            sub_vital = stay_charts[stay_charts['itemid'].isin(target_ids)].sort_values(by='charttime')
            
            prefix = f"vital_{vital_name}"
            
            if len(sub_vital) > 0:
                values = sub_vital['valuenum'].dropna()
                if len(values) > 0:
                    patient_feats[f"{prefix}_mean"] = values.mean()
                    patient_feats[f"{prefix}_min"] = values.min()
                    patient_feats[f"{prefix}_max"] = values.max()
                    patient_feats[f"{prefix}_std"] = values.std() if len(values) > 1 else 0.0
                    patient_feats[f"{prefix}_latest_value"] = values.iloc[-1]
                    patient_feats[f"{prefix}_trend"] = values.iloc[-1] - values.iloc[0]
                else:
                    patient_feats[f"{prefix}_mean"] = np.nan
                    patient_feats[f"{prefix}_min"] = np.nan
                    patient_feats[f"{prefix}_max"] = np.nan
                    patient_feats[f"{prefix}_std"] = np.nan
                    patient_feats[f"{prefix}_latest_value"] = np.nan
                    patient_feats[f"{prefix}_trend"] = np.nan
            else:
                patient_feats[f"{prefix}_mean"] = np.nan
                patient_feats[f"{prefix}_min"] = np.nan
                patient_feats[f"{prefix}_max"] = np.nan
                patient_feats[f"{prefix}_std"] = np.nan
                patient_feats[f"{prefix}_latest_value"] = np.nan
                patient_feats[f"{prefix}_trend"] = np.nan
                
        # 2. Filter Laboratories (from lab_stays) for this stay and first 24h window
        stay_labs = lab_stays[
            (lab_stays['stay_id'] == stay_id) & 
            (lab_stays['charttime'] >= intime) & 
            (lab_stays['charttime'] <= t_end)
        ]
        
        # Extract 6 aggregations + missingness flag for labs
        for lab_name, itemids in config.LABS_MAP.items():
            sub_lab = stay_labs[stay_labs['itemid'].isin(itemids)].sort_values(by='charttime')
            
            prefix = f"lab_{lab_name}"
            
            if len(sub_lab) > 0:
                values = sub_lab['valuenum'].dropna()
                if len(values) > 0:
                    # Present
                    patient_feats[f"is_missing_{lab_name}"] = 0
                    patient_feats[f"{prefix}_mean"] = values.mean()
                    patient_feats[f"{prefix}_min"] = values.min()
                    patient_feats[f"{prefix}_max"] = values.max()
                    patient_feats[f"{prefix}_std"] = values.std() if len(values) > 1 else 0.0
                    patient_feats[f"{prefix}_latest_value"] = values.iloc[-1]
                    patient_feats[f"{prefix}_trend"] = values.iloc[-1] - values.iloc[0]
                else:
                    # Record is present but value was NaN
                    patient_feats[f"is_missing_{lab_name}"] = 1
                    patient_feats[f"{prefix}_mean"] = np.nan
                    patient_feats[f"{prefix}_min"] = np.nan
                    patient_feats[f"{prefix}_max"] = np.nan
                    patient_feats[f"{prefix}_std"] = np.nan
                    patient_feats[f"{prefix}_latest_value"] = np.nan
                    patient_feats[f"{prefix}_trend"] = np.nan
            else:
                # Totally missing
                patient_feats[f"is_missing_{lab_name}"] = 1
                patient_feats[f"{prefix}_mean"] = np.nan
                patient_feats[f"{prefix}_min"] = np.nan
                patient_feats[f"{prefix}_max"] = np.nan
                patient_feats[f"{prefix}_std"] = np.nan
                patient_feats[f"{prefix}_latest_value"] = np.nan
                patient_feats[f"{prefix}_trend"] = np.nan
                
        feature_records.append(patient_feats)
        
    df_features = pd.DataFrame(feature_records)
    print(f"Aggregated features compiled successfully. Base shape: {df_features.shape}")
    
    # 3. Impute remaining missing values
    # For vitals: Impute missing aggregations with the global cohort median
    # For labs: Impute missing aggregations with healthy baseline clinical reference values
    print("Applying clinical baseline imputations...")
    for vital_name in config.VITALS_MAP.keys():
        prefix = f"vital_{vital_name}"
        cols = [f"{prefix}_mean", f"{prefix}_min", f"{prefix}_max", f"{prefix}_std", f"{prefix}_latest_value", f"{prefix}_trend"]
        for col in cols:
            median_val = df_features[col].median()
            # If standard deviation is missing, default to 0.0 stability
            default_val = 0.0 if "_std" in col or "_trend" in col else median_val
            # Fallback if median itself is NaN (should not happen due to audit, but safe)
            if pd.isnull(default_val):
                default_val = 0.0
            df_features[col] = df_features[col].fillna(default_val)
            
    for lab_name in config.LABS_MAP.keys():
        prefix = f"lab_{lab_name}"
        ref_val = CLINICAL_REF_LABS[lab_name]
        cols = [f"{prefix}_mean", f"{prefix}_min", f"{prefix}_max", f"{prefix}_std", f"{prefix}_latest_value", f"{prefix}_trend"]
        for col in cols:
            # Impute standard deviation and trend to 0.0 stability
            default_val = 0.0 if "_std" in col or "_trend" in col else ref_val
            df_features[col] = df_features[col].fillna(default_val)
            
    # Double check if any missing values remain in numerical features
    null_counts = df_features.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print(f"  WARNING: Unhandled missing columns: \n{null_cols}")
    else:
        print("  All clinical features imputed successfully. Zero missing values remain in final matrix!")
        
    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(config.PATH_PROCESSED_FEATURES), exist_ok=True)
    df_features.to_csv(config.PATH_PROCESSED_FEATURES, index=False)
    print(f"Saved processed multimodal clinical feature matrix to: {config.PATH_PROCESSED_FEATURES}")
    return df_features

if __name__ == "__main__":
    from src.data_ingestion import ingest_all_raw_data
    from src.preprocessing import preprocess_clinical_data
    
    try:
        cohort, charts, labs = ingest_all_raw_data()
        cohort_clean, charts_clean, labs_clean = preprocess_clinical_data(cohort, charts, labs)
        features = extract_clinical_features(cohort_clean, charts_clean, labs_clean)
        print("\nFeature Engineering Test Succeeded!")
        print(f"Final feature matrix shape: {features.shape}")
        
        # Verify columns count: 
        # 1 stay_id, 1 subject_id, 1 age, 1 gender, 4 demographics (race, type, insurance, marital), 1 target = 9 base cols
        # 7 vitals * 6 aggregations = 42 cols
        # 8 labs * (6 aggregations + 1 flag) = 56 cols
        # Total columns expected: 9 + 42 + 56 = 107 columns!
        print(f"  Total columns: {len(features.columns)}")
        print(f"  Clinical Vitals features: {len([c for c in features.columns if 'vital_' in c])}")
        print(f"  Clinical Labs features: {len([c for c in features.columns if 'lab_' in c])}")
        print(f"  Missingness indicator flags: {len([c for c in features.columns if 'is_missing_' in c])}")
        
        # Quick check of a few records
        print("\nSample records (First 2 patients):")
        subset = features[['stay_id', 'age', 'gender', 'vital_heart_rate_mean', 'vital_heart_rate_trend', 'lab_creatinine_latest_value', 'is_missing_creatinine', 'hospital_expire_flag']]
        print(subset.head(2).to_string())
    except Exception as e:
        print(f"\nFeature Engineering Test Failed: {e}", file=sys.stderr)
