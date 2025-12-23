"""
Utility Functions for ComplianceNet
===================================
Cross-validation, metrics, and helper functions.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import StratifiedKFold, GroupKFold


def calculate_qwk(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate Quadratic Weighted Kappa.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        QWK score (-1 to 1)
    """
    from sklearn.metrics import cohen_kappa_score
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle edge case: if all predictions are same or all true are same
    if len(set(y_true)) == 1 and len(set(y_pred)) == 1:
        # Perfect agreement on single class
        return 1.0 if y_true[0] == y_pred[0] else -1.0
    
    # Use sklearn's implementation which handles edge cases
    return float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))


def create_stratified_folds(y: List[int], n_folds: int = 5) -> List[Tuple[List[int], List[int]]]:
    """
    Create stratified cross-validation folds.
    
    Args:
        y: Target labels
        n_folds: Number of folds
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    y = np.array(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
        folds.append((train_idx.tolist(), val_idx.tolist()))
    
    return folds


def create_participant_folds(participant_ids: List[str], n_folds: int = 5) -> List[Tuple[List[int], List[int]]]:
    """
    Create participant-level cross-validation folds (no data leakage).
    
    Args:
        participant_ids: List of participant identifiers
        n_folds: Number of folds
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Map participant IDs to group numbers
    unique_ids = list(set(participant_ids))
    id_to_group = {pid: i for i, pid in enumerate(unique_ids)}
    groups = np.array([id_to_group[pid] for pid in participant_ids])
    
    gkf = GroupKFold(n_splits=min(n_folds, len(unique_ids)))
    
    folds = []
    for train_idx, val_idx in gkf.split(np.zeros(len(participant_ids)), groups=groups):
        folds.append((train_idx.tolist(), val_idx.tolist()))
    
    return folds


def get_intervention_recommendations() -> Dict[str, str]:
    """
    Get phenotype-to-intervention mapping.
    
    Returns:
        Dict mapping phenotype names to intervention recommendations
    """
    return {
        'escapist': 'Parental mediation, sleep hygiene education, and screen time boundaries',
        'disorganized': 'Executive function training, ADHD evaluation, and organizational support',
        'sensory_avoider': 'Sensory integration assessment, ASD screening, and sensory-friendly alternatives',
        'compliant': 'Preventive education and continued monitoring'
    }
