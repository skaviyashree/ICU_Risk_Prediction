import pandas as pd
import numpy as np
import os
import sys

# Append parent directory to sys.path to support importing src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

def load_table(path, nrows=None):
    """Loads a gzipped CSV table from the dataset directory."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required table file not found at: {path}")
    print(f"Loading table: {os.path.basename(path)}...")
    return pd.read_csv(path, nrows=nrows)

def build_base_cohort(patients, admissions, icustays):
    """
    Builds the baseline adult patient cohort.
    Filters:
      - Selects only the first ICU stay per patient to prevent data leakage.
      - Calculates exact age at ICU admission: anchor_age + (year of intime - anchor_year).
      - Selects adult patients (age >= 18).
    """
    print("Building base patient-ICU cohort...")
    
    # 1. Parse timestamps
    icustays = icustays.copy()
    icustays['intime'] = pd.to_datetime(icustays['intime'])
    icustays['outtime'] = pd.to_datetime(icustays['outtime'])
    
    admissions = admissions.copy()
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    
    # 2. Select the first ICU stay for each patient
    # In clinical research, using the first stay is standard to prevent bias
    icustays_sorted = icustays.sort_values(by=['subject_id', 'intime'])
    first_stays = icustays_sorted.groupby('subject_id').first().reset_index()
    print(f"Filtered to first ICU stays: {len(first_stays)} stays (from {len(icustays)} total stays)")
    
    # 3. Merge with patients demographics
    cohort = pd.merge(first_stays, patients, on='subject_id', how='inner')
    
    # Calculate exact age at ICU admission
    cohort['age'] = cohort['anchor_age'] + (cohort['intime'].dt.year - cohort['anchor_year'])
    
    # 4. Filter for adult patients (age >= 18)
    cohort = cohort[cohort['age'] >= 18]
    print(f"Filtered out pediatric stays: {len(cohort)} adult stays remaining")
    
    # 5. Merge with admissions for outcomes and hospital demographics
    cohort = pd.merge(
        cohort, 
        admissions[['hadm_id', 'admission_type', 'insurance', 'marital_status', 'race', 'hospital_expire_flag']], 
        on='hadm_id', 
        how='inner'
    )
    print(f"Merged with hospital admission outcomes: {len(cohort)} records in baseline cohort")
    
    # Reorder and keep essential columns
    keep_cols = [
        'subject_id', 'hadm_id', 'stay_id', 'gender', 'age', 
        'first_careunit', 'last_careunit', 'intime', 'outtime', 'los',
        'admission_type', 'insurance', 'marital_status', 'race', 'hospital_expire_flag'
    ]
    return cohort[keep_cols]

def ingest_all_raw_data(nrows=None):
    """
    Orchestrates the complete data ingestion stage.
    Loads raw tables, builds cohort, and loads chartevents and labevents.
    """
    print("\n--- STARTING DATA INGESTION STAGE ---")
    
    # Load hospital-wide tables
    patients = load_table(config.PATH_PATIENTS)
    admissions = load_table(config.PATH_ADMISSIONS)
    icustays = load_table(config.PATH_ICUSTAYS)
    
    # Build baseline cohort
    cohort = build_base_cohort(patients, admissions, icustays)
    
    # Load clinical events (vitals and labs)
    chartevents = load_table(config.PATH_CHARTEVENTS, nrows=nrows)
    labevents = load_table(config.PATH_LABEVENTS, nrows=nrows)
    
    print("Data Ingestion stage completed successfully!")
    print(f"Cohort size: {cohort.shape}")
    print(f"Chartevents records: {chartevents.shape}")
    print(f"Labevents records: {labevents.shape}")
    
    return cohort, chartevents, labevents

if __name__ == "__main__":
    # Test execution
    try:
        cohort, charts, labs = ingest_all_raw_data()
        print("\nIngestion Test Succeeded!")
        print("Cohort Demographics:")
        print(f"  Gender counts:\n{cohort['gender'].value_counts()}")
        print(f"  Mean Age: {cohort['age'].mean():.2f}")
        print(f"  Mortality cases (hospital_expire_flag): {cohort['hospital_expire_flag'].sum()} ({cohort['hospital_expire_flag'].mean()*100:.2f}%)")
    except Exception as e:
        print(f"\nIngestion Test Failed: {e}", file=sys.stderr)
