"""
ComplianceNet Demo
==================
Simple demo showing how to use the ComplianceNet system.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cvee import extract_compliance_vector
from src.pipeline import full_pipeline_predict, classify_phenotype

def create_sample_data():
    """Create sample actigraphy data for demo."""
    np.random.seed(42)
    n_days = 7
    n_epochs = n_days * 17280  # 7 days at 5-second epochs
    
    # Simulate "Escapist" phenotype - nocturnal non-wear
    timestamps = pd.date_range('2024-01-01', periods=n_epochs, freq='5s')
    hours = timestamps.hour
    
    non_wear = np.zeros(n_epochs)
    night_mask = (hours >= 22) | (hours <= 6)
    non_wear[night_mask] = np.random.binomial(1, 0.4, night_mask.sum())
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'enmo': np.random.exponential(30, n_epochs),
        'anglez': np.random.uniform(-90, 90, n_epochs),
        'light': np.where(non_wear == 1, np.random.uniform(0, 5, n_epochs),
                         np.random.exponential(100, n_epochs)),
        'battery_voltage': np.tile(np.linspace(4200, 3700, 17280), n_days),
        'non_wear_flag': non_wear.astype(int)
    })

def main():
    print("=" * 50)
    print("ComplianceNet Demo")
    print("=" * 50)
    
    # Create sample data
    print("\n1. Creating sample actigraphy data (7 days)...")
    data = create_sample_data()
    print(f"   - {len(data)} epochs ({len(data)//17280} days)")
    print(f"   - Non-wear rate: {data['non_wear_flag'].mean()*100:.1f}%")
    
    # Extract compliance vector
    print("\n2. Extracting Compliance Vector (20 features)...")
    cv = extract_compliance_vector(data)
    print(f"   - Battery Neglect Slope: {cv['beta_batt']:.2f}")
    print(f"   - Nocturnal Disconnect Index: {cv['ndi']:.3f}")
    print(f"   - Micro-Removal Frequency: {cv['mu_freq']:.2f}")
    print(f"   - Sensory Rejection Vector: {cv['v_sens']:.2f}")
    print(f"   - Weekend Dropout Differential: {cv['delta_wknd']:.3f}")
    print(f"   - Total Wear Percentage: {cv['total_wear_percentage']*100:.1f}%")
    
    # Classify phenotype
    print("\n3. Classifying Compliance Phenotype...")
    phenotype = classify_phenotype(data)
    print(f"   - Primary Phenotype: {phenotype['primary'].upper()}")
    print(f"   - Confidence: {phenotype['confidence']:.2f}")
    print(f"   - Battery Health: {phenotype['battery_health']}")
    print(f"   - ASD Risk Flag: {phenotype['asd_risk_flag']}")
    
    # Predict SII
    print("\n4. Predicting Severity Impairment Index (SII)...")
    sii = full_pipeline_predict(data)
    sii_labels = ['None', 'Mild', 'Moderate', 'Severe']
    print(f"   - Predicted SII: {sii} ({sii_labels[sii]})")
    
    print("\n" + "=" * 50)
    print("Demo Complete!")
    print("=" * 50)

if __name__ == '__main__':
    main()
