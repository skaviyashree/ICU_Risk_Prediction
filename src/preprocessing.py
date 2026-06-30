import pandas as pd
import numpy as np
import os
import sys

# Append parent directory to sys.path to support importing src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

def clean_vitals_outliers(chartevents):
    """
    Cleans physiological outlier values in chartevents based on clinical boundaries.
    Also normalizes temperature units (Fahrenheit to Celsius).
    """
    print("Cleaning vital signs outliers in chartevents...")
    df = chartevents.copy()
    
    # Ensure correct data type
    df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
    
    # 1. Temperature Normalization (Fahrenheit to Celsius)
    # itemid 223761 is Fahrenheit, 223762 is Celsius
    f_mask = (df['itemid'] == 223761)
    # Convert Fahrenheit to Celsius: (F - 32) * 5/9
    df.loc[f_mask, 'valuenum'] = (df.loc[f_mask, 'valuenum'] - 32) * 5 / 9
    # Map itemid 223761 to 223762 so all temperature is unified under one itemid
    df.loc[f_mask, 'itemid'] = 223762
    
    # 2. Apply outlier boundaries for each vital sign
    cleaned_rows = 0
    total_rows_before = len(df)
    
    for vital_name, itemids in config.VITALS_MAP.items():
        # Treat temperature as 223762 after normalization
        target_ids = [223762] if vital_name == "temperature" else itemids
        
        mask = df['itemid'].isin(target_ids)
        bounds = config.OUTLIER_BOUNDS.get(vital_name)
        
        if bounds:
            # Identify values outside clinical limits
            outlier_mask = mask & ((df['valuenum'] < bounds['min']) | (df['valuenum'] > bounds['max']))
            cleaned_rows += outlier_mask.sum()
            # Set outliers to NaN so they can be imputed during feature engineering
            df.loc[outlier_mask, 'valuenum'] = np.nan
            
    print(f"Outlier Cleaning Summary:")
    print(f"  Total vitals observations: {total_rows_before}")
    print(f"  Artifacts removed (set to NaN): {cleaned_rows} ({cleaned_rows/total_rows_before*100:.3f}%)")
    
    return df

def clean_labs_outliers(labevents):
    """
    Cleans lab events by ensuring valuenum is numeric and filtering out negative values.
    Lab events are usually pre-cleaned by hospital systems, but we ensure basic quality.
    """
    print("Cleaning labevents data quality...")
    df = labevents.copy()
    
    df['valuenum'] = pd.to_numeric(df['valuenum'], errors='coerce')
    
    # Remove implausible negative values
    neg_mask = (df['itemid'].isin([item for sublist in config.LABS_MAP.values() for item in sublist])) & (df['valuenum'] < 0)
    cleaned_neg = neg_mask.sum()
    df.loc[neg_mask, 'valuenum'] = np.nan
    
    print(f"  Removed negative lab events: {cleaned_neg}")
    return df

def preprocess_clinical_data(cohort, chartevents, labevents):
    """
    Enforces quality checks and performs outlier removal for vitals and labs.
    """
    print("\n--- STARTING CLINICAL PREPROCESSING STAGE ---")
    
    # 1. Enforce Datetime Data Types
    cohort_clean = cohort.copy()
    cohort_clean['intime'] = pd.to_datetime(cohort_clean['intime'])
    cohort_clean['outtime'] = pd.to_datetime(cohort_clean['outtime'])
    
    chartevents_clean = chartevents.copy()
    chartevents_clean['charttime'] = pd.to_datetime(chartevents_clean['charttime'])
    
    labevents_clean = labevents.copy()
    labevents_clean['charttime'] = pd.to_datetime(labevents_clean['charttime'])
    
    # 2. Clean Vitals (bedsides)
    chartevents_clean = clean_vitals_outliers(chartevents_clean)
    
    # 3. Clean Labs
    labevents_clean = clean_labs_outliers(labevents_clean)
    
    # 4. Remove logical discrepancies in Cohort (if any)
    # Check if intime is after outtime
    invalid_stays = (cohort_clean['intime'] > cohort_clean['outtime'])
    if invalid_stays.sum() > 0:
        print(f"  WARNING: Found {invalid_stays.sum()} stays where intime > outtime. Dropping stays.")
        cohort_clean = cohort_clean[~invalid_stays]
        
    print("Clinical Preprocessing stage completed successfully!")
    return cohort_clean, chartevents_clean, labevents_clean

if __name__ == "__main__":
    # Test execution
    from src.data_ingestion import ingest_all_raw_data
    try:
        cohort, charts, labs = ingest_all_raw_data()
        cohort_clean, charts_clean, labs_clean = preprocess_clinical_data(cohort, charts, labs)
        print("\nPreprocessing Test Succeeded!")
        
        # Verify temperature unification
        temp_f_count = charts[charts['itemid'] == 223761].shape[0]
        temp_c_count = charts[charts['itemid'] == 223762].shape[0]
        temp_unified_count = charts_clean[charts_clean['itemid'] == 223762].shape[0]
        
        print(f"Temperature Unification Check:")
        print(f"  Raw Temperature Fahrenheit (223761) observations: {temp_f_count}")
        print(f"  Raw Temperature Celsius (223762) observations: {temp_c_count}")
        print(f"  Cleaned Temperature Celsius (223762) observations: {temp_unified_count}")
        assert temp_unified_count == temp_f_count + temp_c_count, "Temperature count mismatch!"
        print("  Temperature unification assertion passed successfully!")
    except Exception as e:
        print(f"\nPreprocessing Test Failed: {e}", file=sys.stderr)
