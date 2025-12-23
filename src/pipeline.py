"""
ComplianceNet Pipeline
======================
End-to-end pipeline for PIU prediction.
"""

import pandas as pd
from typing import Dict, List

from .cvee import extract_compliance_vector
from .dtcm import generate_day_profiles, perform_dtw_clustering, calculate_phenotype_distribution
from .tme import preprocess_for_transformer, create_tme_model, decode_ordinal_prediction


def full_pipeline_predict(data: pd.DataFrame, model=None) -> int:
    """
    Run full ComplianceNet pipeline and return SII prediction.
    
    Args:
        data: Actigraphy DataFrame with required columns
        model: Pre-trained TME model (creates new if None)
        
    Returns:
        SII prediction (0-3)
    """
    # Stage 1: Extract compliance vector
    cv = extract_compliance_vector(data)
    
    # Stage 2: Generate profiles and cluster
    profiles = generate_day_profiles(data)
    if len(profiles) > 0:
        clusters, _ = perform_dtw_clustering(profiles, n_clusters=4)
        phenotype_dist = calculate_phenotype_distribution(clusters)
    else:
        phenotype_dist = [0.25, 0.25, 0.25, 0.25]
    
    # Stage 3: Transformer prediction
    sequence = preprocess_for_transformer(data)
    
    if model is None:
        model = create_tme_model()
    
    sii = model.predict(sequence, cv, phenotype_dist, {})
    
    return sii


def process_batch(participants: List[pd.DataFrame], model=None) -> List[Dict]:
    """
    Process batch of participants through the pipeline.
    
    Args:
        participants: List of participant DataFrames
        model: Pre-trained TME model
        
    Returns:
        List of result dictionaries
    """
    if model is None:
        model = create_tme_model()
    
    results = []
    
    for data in participants:
        try:
            # Extract all features
            cv = extract_compliance_vector(data)
            
            profiles = generate_day_profiles(data)
            if len(profiles) > 0:
                clusters, _ = perform_dtw_clustering(profiles, n_clusters=4)
                phenotype_dist = calculate_phenotype_distribution(clusters)
            else:
                phenotype_dist = [0.25, 0.25, 0.25, 0.25]
            
            sequence = preprocess_for_transformer(data)
            sii = model.predict(sequence, cv, phenotype_dist, {})
            
            results.append({
                'sii_prediction': sii,
                'compliance_vector': cv,
                'phenotype_distribution': phenotype_dist,
                'status': 'success'
            })
            
        except Exception as e:
            results.append({
                'sii_prediction': None,
                'compliance_vector': None,
                'phenotype_distribution': None,
                'status': f'error: {str(e)}'
            })
    
    return results


def classify_phenotype(data: pd.DataFrame) -> Dict:
    """
    Classify participant's dominant compliance phenotype.
    
    Args:
        data: Actigraphy DataFrame
        
    Returns:
        Dict with phenotype classification and confidence
    """
    cv = extract_compliance_vector(data)
    
    profiles = generate_day_profiles(data)
    if len(profiles) == 0:
        return {
            'primary': 'compliant',
            'secondary': None,
            'confidence': 0.5,
            'battery_health': 'unknown',
            'asd_risk_flag': False
        }
    
    clusters, centroids = perform_dtw_clustering(profiles, n_clusters=4)
    distribution = calculate_phenotype_distribution(clusters)
    
    # Map clusters to phenotypes based on characteristics
    phenotype_names = ['compliant', 'escapist', 'disorganized', 'sensory_avoider']
    
    # Determine primary phenotype
    primary_idx = max(range(4), key=lambda i: distribution[i])
    secondary_idx = sorted(range(4), key=lambda i: distribution[i], reverse=True)[1]
    
    confidence = distribution[primary_idx]
    
    # Heuristics for phenotype detection
    if cv['halt_count'] >= 2 or cv['beta_batt'] < 50:
        primary = 'disorganized'
        battery_health = 'poor'
    elif cv['ndi'] > 0.3:
        primary = 'escapist'
        battery_health = 'good'
    elif cv['v_sens'] > 80 or cv['high_agitation_ratio'] > 0.5:
        primary = 'sensory_avoider'
        battery_health = 'good'
    else:
        primary = 'compliant'
        battery_health = 'good'
    
    # ASD risk flag
    asd_risk = cv['v_sens'] > 60 or cv['high_agitation_ratio'] > 0.4
    
    return {
        'primary': primary,
        'secondary': phenotype_names[secondary_idx] if confidence < 0.7 else None,
        'confidence': float(confidence),
        'battery_health': battery_health,
        'asd_risk_flag': asd_risk
    }
