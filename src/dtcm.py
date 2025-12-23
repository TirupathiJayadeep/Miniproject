"""
DTW-Enhanced Temporal Clustering Module (DTCM)
==============================================
Stage 2 of ComplianceNet: Unsupervised phenotype discovery via DTW clustering.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from collections import Counter


def generate_day_profiles(data: pd.DataFrame) -> List[np.ndarray]:
    """
    Generate 96-point day profiles (15-min epochs) from actigraphy data.
    
    Each day is represented as a 96-element binary vector indicating
    non-wear in each 15-minute epoch.
    
    Args:
        data: DataFrame with 'timestamp' and 'non_wear_flag' columns
        
    Returns:
        List of numpy arrays, each of length 96
    """
    if 'non_wear_flag' not in data.columns:
        raise KeyError("non_wear_flag column required")
    if 'timestamp' not in data.columns:
        raise KeyError("timestamp column required")
    
    data = data.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['date'] = data['timestamp'].dt.date
    data['hour'] = data['timestamp'].dt.hour
    data['minute'] = data['timestamp'].dt.minute
    
    # 15-minute epoch index (0-95)
    data['epoch_15min'] = data['hour'] * 4 + data['minute'] // 15
    
    profiles = []
    
    for date, day_data in data.groupby('date'):
        # Create 96-point profile
        profile = np.zeros(96)
        
        for epoch_idx, epoch_data in day_data.groupby('epoch_15min'):
            if 0 <= epoch_idx < 96:
                # Use majority rule: >50% non-wear in epoch -> mark as 1
                if epoch_data['non_wear_flag'].mean() > 0.5:
                    profile[int(epoch_idx)] = 1
        
        # Exclude days with >80% non-wear (invalid days)
        if profile.mean() <= 0.8:
            profiles.append(profile)
    
    return profiles


def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Simple Euclidean distance."""
    return np.sqrt(np.sum((x - y) ** 2))


def _dtw_distance(x: np.ndarray, y: np.ndarray, window: int = None) -> float:
    """
    Compute Dynamic Time Warping distance between two sequences.
    
    Args:
        x, y: Input sequences
        window: Sakoe-Chiba band width (None for no constraint)
        
    Returns:
        DTW distance
    """
    n, m = len(x), len(y)
    
    if window is None:
        window = max(n, m)
    
    # Initialize cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m + 1, i + window + 1)):
            cost = (x[i-1] - y[j-1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return np.sqrt(dtw_matrix[n, m])


def _kmeans_plusplus_init(profiles: List[np.ndarray], n_clusters: int) -> List[np.ndarray]:
    """K-means++ initialization for centroids."""
    np.random.seed(42)
    n = len(profiles)
    
    # First centroid: random
    centroids = [profiles[np.random.randint(n)].copy()]
    
    for _ in range(1, n_clusters):
        # Compute distances to nearest centroid
        distances = []
        for profile in profiles:
            min_dist = min(_euclidean_distance(profile, c) for c in centroids)
            distances.append(min_dist ** 2)
        
        # Choose next centroid with probability proportional to distance^2
        total_dist = sum(distances)
        if total_dist == 0 or np.isnan(total_dist):
            # All same distance, pick random
            next_idx = np.random.randint(n)
        else:
            probs = np.array(distances) / total_dist
            probs = np.nan_to_num(probs, nan=1.0/n)
            probs = probs / probs.sum()  # Normalize
            next_idx = np.random.choice(n, p=probs)
        centroids.append(profiles[next_idx].copy())
    
    return centroids


def perform_dtw_clustering(
    profiles: List[np.ndarray], 
    n_clusters: int = 4,
    sakoe_chiba_radius: int = 12,
    max_iter: int = 50
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Perform DTW-based k-means clustering on day profiles.
    
    Uses simplified DTW distance with Sakoe-Chiba band constraint.
    
    Args:
        profiles: List of 96-point day profiles
        n_clusters: Number of clusters (default 4 for phenotypes)
        sakoe_chiba_radius: DTW window constraint
        max_iter: Maximum iterations
        
    Returns:
        Tuple of (cluster assignments, cluster centroids)
    """
    if len(profiles) == 0:
        raise ValueError("No valid profiles provided for clustering")
    
    profiles_arr = [np.array(p) for p in profiles]
    n = len(profiles_arr)
    
    # Handle edge case: fewer profiles than clusters
    if n < n_clusters:
        # Assign each profile to its own cluster, pad with empty
        clusters = list(range(n))
        centroids = profiles_arr + [np.zeros(96) for _ in range(n_clusters - n)]
        return clusters, centroids
    
    # Initialize centroids using k-means++
    centroids = _kmeans_plusplus_init(profiles_arr, n_clusters)
    
    clusters = [0] * n
    
    for iteration in range(max_iter):
        # Assignment step
        old_clusters = clusters.copy()
        
        for i, profile in enumerate(profiles_arr):
            min_dist = float('inf')
            best_cluster = 0
            
            for k, centroid in enumerate(centroids):
                # Use DTW distance with Sakoe-Chiba constraint
                dist = _dtw_distance(profile, centroid, window=sakoe_chiba_radius)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = k
            
            clusters[i] = best_cluster
        
        # Check convergence
        if clusters == old_clusters:
            break
        
        # Update step: compute new centroids
        for k in range(n_clusters):
            cluster_profiles = [profiles_arr[i] for i in range(n) if clusters[i] == k]
            if len(cluster_profiles) > 0:
                # Simple mean for centroid (DBA approximation)
                centroids[k] = np.mean(cluster_profiles, axis=0)
    
    return clusters, centroids


def calculate_phenotype_distribution(clusters: List[int], n_clusters: int = 4) -> List[float]:
    """
    Calculate phenotype distribution from cluster assignments.
    
    Args:
        clusters: List of cluster assignments (0 to n_clusters-1)
        n_clusters: Total number of clusters
        
    Returns:
        List of proportions for each cluster, summing to 1.0
    """
    if len(clusters) == 0:
        return [1.0 / n_clusters] * n_clusters
    
    counts = Counter(clusters)
    total = len(clusters)
    
    distribution = []
    for k in range(n_clusters):
        proportion = counts.get(k, 0) / total
        distribution.append(float(proportion))
    
    return distribution
