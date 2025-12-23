"""
Compliance Vector Extraction Engine (CVEE)
==========================================
Stage 1 of ComplianceNet: Extract 20-dimensional compliance features from actigraphy data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import warnings


def calculate_battery_neglect_slope(data: pd.DataFrame) -> Dict:
    """
    Calculate Battery Neglect Slope (β_batt) features.
    
    Quantifies executive dysfunction via charging behavior entropy.
    
    Returns:
        Dict with keys: beta_batt, charge_variance, halt_count, mean_trough_voltage
    """
    if data.empty:
        raise ValueError("Empty dataset provided")
    
    if 'battery_voltage' not in data.columns:
        raise KeyError("battery_voltage column required")
    
    voltage = data['battery_voltage'].values
    
    # Smooth voltage with rolling median
    voltage_series = pd.Series(voltage)
    voltage_smooth = voltage_series.rolling(window=120, min_periods=1, center=True).median().values
    
    # Identify HALT events (voltage <= 3200mV followed by gap/zero)
    halt_threshold = 3200
    halt_events = 0
    in_halt = False
    
    for i in range(1, len(voltage_smooth)):
        if voltage_smooth[i] <= halt_threshold and voltage_smooth[i-1] > halt_threshold:
            halt_events += 1
            in_halt = True
        elif voltage_smooth[i] > halt_threshold + 200:  # Recovered (charged)
            in_halt = False
    
    # Identify charging events (voltage increase > 50mV in short period)
    charge_times = []
    for i in range(1, len(voltage_smooth)):
        if voltage_smooth[i] - voltage_smooth[i-1] > 50:
            charge_times.append(i)
    
    # Calculate charge interval variance
    if len(charge_times) >= 2:
        intervals = np.diff(charge_times)
        charge_variance = float(np.var(intervals))
    else:
        charge_variance = 0.0
    
    # Calculate total runtime (non-zero voltage periods)
    runtime = np.sum(voltage > 0) * 5 / 3600  # Convert epochs to hours
    
    # Beta_batt = Runtime / (HALT_Count + 1)
    beta_batt = runtime / (halt_events + 1)
    
    # Mean trough voltage (daily minimum)
    if 'timestamp' in data.columns:
        data_copy = data.copy()
        data_copy['date'] = pd.to_datetime(data_copy['timestamp']).dt.date
        daily_min = data_copy.groupby('date')['battery_voltage'].min()
        mean_trough = float(daily_min.mean()) if len(daily_min) > 0 else 4000.0
    else:
        mean_trough = float(np.min(voltage[voltage > 0])) if np.any(voltage > 0) else 0.0
    
    return {
        'beta_batt': float(beta_batt),
        'charge_variance': float(charge_variance),
        'halt_count': int(halt_events),
        'mean_trough_voltage': float(mean_trough)
    }


def calculate_nocturnal_disconnect_index(data: pd.DataFrame) -> Dict:
    """
    Calculate Nocturnal Disconnect Index (NDI) features.
    
    Detects strategic hiding of late-night digital activity.
    
    Returns:
        Dict with keys: ndi, ndi_ratio, night_removal_count, mean_night_removal_duration
    """
    if 'non_wear_flag' not in data.columns:
        raise KeyError("non_wear_flag column required")
    
    if 'timestamp' not in data.columns:
        raise KeyError("timestamp column required")
    
    data = data.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    
    # Define night interval: 22:00 to 06:00
    night_mask = (data['hour'] >= 22) | (data['hour'] <= 6)
    
    # Get non-wear during night
    night_nonwear = data[night_mask & (data['non_wear_flag'] == 1)]
    day_nonwear = data[~night_mask & (data['non_wear_flag'] == 1)]
    
    total_night_epochs = night_mask.sum()
    
    if total_night_epochs == 0:
        return {
            'ndi': 0.0,
            'ndi_ratio': 0.0,
            'night_removal_count': 0,
            'mean_night_removal_duration': 0.0
        }
    
    # Calculate strategic score based on light levels
    if 'light' in data.columns and len(night_nonwear) > 0:
        mean_light = night_nonwear['light'].mean()
        strategic_weight = 1.0 if mean_light < 5 else 0.3
    else:
        strategic_weight = 1.0
    
    strategic_score = len(night_nonwear) * strategic_weight
    casual_score = len(day_nonwear) * 0.5
    
    # NDI = strategic non-wear / total night minutes
    ndi = strategic_score / total_night_epochs
    
    # NDI ratio
    ndi_ratio = strategic_score / (casual_score + 1e-6)
    
    # Count night removal episodes
    night_data = data[night_mask].copy()
    if len(night_data) > 0:
        night_data['episode'] = (night_data['non_wear_flag'].diff() != 0).cumsum()
        episodes = night_data[night_data['non_wear_flag'] == 1].groupby('episode').size()
        night_removal_count = len(episodes)
        mean_duration = float(episodes.mean() * 5 / 60) if len(episodes) > 0 else 0.0  # minutes
    else:
        night_removal_count = 0
        mean_duration = 0.0
    
    return {
        'ndi': float(min(ndi, 1.0)),
        'ndi_ratio': float(ndi_ratio),
        'night_removal_count': int(night_removal_count),
        'mean_night_removal_duration': float(mean_duration)
    }


def calculate_micro_removal_frequency(data: pd.DataFrame) -> Dict:
    """
    Calculate Micro-Removal Frequency (μ_freq) features.
    
    Captures sporadic, short-duration non-wear (15-90 min).
    
    Returns:
        Dict with keys: mu_freq, entropy_mu, mean_micro_duration, max_daily_micro_count
    """
    if 'non_wear_flag' not in data.columns:
        raise KeyError("non_wear_flag column required")
    
    data = data.copy()
    
    # Identify continuous non-wear blocks
    data['episode'] = (data['non_wear_flag'].diff() != 0).cumsum()
    nonwear_episodes = data[data['non_wear_flag'] == 1].groupby('episode')
    
    # Filter micro-removals: 15-90 minutes (180-1080 epochs at 5s)
    min_epochs = 180  # 15 min
    max_epochs = 1080  # 90 min
    
    micro_removals = []
    micro_starts = []
    
    for episode_id, episode_data in nonwear_episodes:
        duration_epochs = len(episode_data)
        if min_epochs <= duration_epochs <= max_epochs:
            duration_minutes = duration_epochs * 5 / 60
            micro_removals.append(duration_minutes)
            if 'timestamp' in data.columns:
                start_hour = pd.to_datetime(episode_data['timestamp'].iloc[0]).hour
                micro_starts.append(start_hour)
    
    # Calculate metrics
    if 'timestamp' in data.columns:
        data['date'] = pd.to_datetime(data['timestamp']).dt.date
        n_days = data['date'].nunique()
    else:
        n_days = max(1, len(data) // 17280)  # Estimate days
    
    mu_freq = len(micro_removals) / max(n_days, 1)
    
    # Shannon entropy of removal timing
    if len(micro_starts) > 0:
        hour_counts = np.histogram(micro_starts, bins=24, range=(0, 24))[0]
        hour_probs = hour_counts / (hour_counts.sum() + 1e-6)
        hour_probs = hour_probs[hour_probs > 0]
        entropy_mu = float(-np.sum(hour_probs * np.log(hour_probs + 1e-10)))
    else:
        entropy_mu = 0.0
    
    mean_micro_duration = float(np.mean(micro_removals)) if micro_removals else 0.0
    
    # Max daily micro count
    if 'timestamp' in data.columns and len(micro_removals) > 0:
        # Simplified: just use total / days
        max_daily = int(np.ceil(len(micro_removals) / max(n_days, 1)))
    else:
        max_daily = len(micro_removals)
    
    return {
        'mu_freq': float(mu_freq),
        'entropy_mu': float(entropy_mu),
        'mean_micro_duration': float(mean_micro_duration),
        'max_daily_micro_count': int(max_daily)
    }


def calculate_sensory_rejection_vector(data: pd.DataFrame) -> Dict:
    """
    Calculate Sensory Rejection Vector (V_sens) features.
    
    Biomarker for neurodivergent-driven PIU via agitation-triggered removal.
    
    Returns:
        Dict with keys: v_sens, high_agitation_ratio, max_pre_removal_enmo
    """
    if 'non_wear_flag' not in data.columns:
        raise KeyError("non_wear_flag column required")
    if 'enmo' not in data.columns:
        raise KeyError("enmo column required")
    
    data = data.copy()
    
    # Find non-wear onset points
    data['nonwear_onset'] = (data['non_wear_flag'].diff() == 1)
    onset_indices = data[data['nonwear_onset']].index.tolist()
    
    if len(onset_indices) == 0:
        return {
            'v_sens': 0.0,
            'high_agitation_ratio': 0.0,
            'max_pre_removal_enmo': 0.0
        }
    
    pre_removal_enmos = []
    pre_removal_anglez_vars = []
    
    # 5-minute window = 60 epochs at 5s
    window_size = 60
    high_agitation_threshold = 80  # mg
    
    for onset_idx in onset_indices:
        # Get position in dataframe
        pos = data.index.get_loc(onset_idx)
        start_pos = max(0, pos - window_size)
        
        window = data.iloc[start_pos:pos]
        
        if len(window) > 0:
            mean_enmo = window['enmo'].mean()
            pre_removal_enmos.append(mean_enmo)
            
            if 'anglez' in data.columns:
                anglez_var = window['anglez'].std()
                pre_removal_anglez_vars.append(anglez_var)
    
    if len(pre_removal_enmos) == 0:
        return {
            'v_sens': 0.0,
            'high_agitation_ratio': 0.0,
            'max_pre_removal_enmo': 0.0
        }
    
    # V_sens = mean pre-removal ENMO + α * anglez variance
    alpha = 0.5
    mean_enmo = np.mean(pre_removal_enmos)
    mean_anglez_var = np.mean(pre_removal_anglez_vars) if pre_removal_anglez_vars else 0
    
    v_sens = mean_enmo + alpha * mean_anglez_var
    
    # High agitation ratio
    high_agitation_count = sum(1 for e in pre_removal_enmos if e > high_agitation_threshold)
    high_agitation_ratio = high_agitation_count / len(pre_removal_enmos)
    
    max_pre_removal_enmo = float(max(pre_removal_enmos))
    
    return {
        'v_sens': float(v_sens),
        'high_agitation_ratio': float(high_agitation_ratio),
        'max_pre_removal_enmo': float(max_pre_removal_enmo)
    }


def calculate_weekend_dropout_differential(data: pd.DataFrame) -> Dict:
    """
    Calculate Weekend Dropout Differential (Δ_wknd) features.
    
    Detects compliance collapse during unstructured time.
    
    Returns:
        Dict with keys: delta_wknd
    """
    if 'non_wear_flag' not in data.columns:
        raise KeyError("non_wear_flag column required")
    if 'timestamp' not in data.columns:
        raise KeyError("timestamp column required")
    
    data = data.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['dayofweek'] = data['timestamp'].dt.dayofweek
    
    # Weekend = Saturday (5) and Sunday (6)
    weekend_mask = data['dayofweek'] >= 5
    weekday_mask = data['dayofweek'] < 5
    
    weekend_data = data[weekend_mask]
    weekday_data = data[weekday_mask]
    
    if len(weekend_data) == 0 or len(weekday_data) == 0:
        return {'delta_wknd': 1.0}
    
    # Wear time = 1 - non_wear_flag
    weekend_wear = 1 - weekend_data['non_wear_flag'].mean()
    weekday_wear = 1 - weekday_data['non_wear_flag'].mean()
    
    # Δ_wknd = weekend wear / weekday wear
    delta_wknd = weekend_wear / (weekday_wear + 1e-6)
    
    return {'delta_wknd': float(min(delta_wknd, 2.0))}


def extract_compliance_vector(data: pd.DataFrame) -> Dict:
    """
    Extract complete 20-dimensional Compliance Vector from actigraphy data.
    
    Combines all CVEE sub-features into unified feature vector.
    """
    # Validate input
    if data.empty or len(data) < 100:
        raise ValueError("Insufficient data for compliance vector extraction")
    
    required_cols = ['timestamp', 'enmo', 'non_wear_flag']
    for col in required_cols:
        if col not in data.columns:
            raise KeyError(f"Required column missing: {col}")
    
    # Data validation
    if 'enmo' in data.columns and (data['enmo'] < 0).any():
        raise ValueError("Invalid ENMO values: negative values detected")
    
    if 'light' in data.columns and (data['light'] < 0).any():
        raise ValueError("Invalid light values: negative values detected")
    
    # Check for numeric types
    if not pd.api.types.is_numeric_dtype(data['enmo']):
        raise TypeError("Expected numeric type for enmo column")
    
    # Sort by timestamp
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    # Check for duplicates
    if data['timestamp'].duplicated().any():
        warnings.warn("Duplicate timestamps detected", UserWarning)
        data = data.drop_duplicates('timestamp')
    
    # Check for unusual battery voltage
    if 'battery_voltage' in data.columns:
        valid_voltage = data['battery_voltage'][(data['battery_voltage'] > 0)]
        if len(valid_voltage) > 0 and valid_voltage.mean() < 3000:
            warnings.warn("Unusual battery voltage values detected", UserWarning)
    
    # Check for invalid participant (100% non-wear)
    if data['non_wear_flag'].mean() > 0.95:
        warnings.warn("Invalid participant: >95% non-wear", UserWarning)
    
    # Extract all sub-features
    battery_features = calculate_battery_neglect_slope(data) if 'battery_voltage' in data.columns else {
        'beta_batt': 0.0, 'charge_variance': 0.0, 'halt_count': 0, 'mean_trough_voltage': 4000.0
    }
    
    ndi_features = calculate_nocturnal_disconnect_index(data)
    micro_features = calculate_micro_removal_frequency(data)
    sensory_features = calculate_sensory_rejection_vector(data)
    weekend_features = calculate_weekend_dropout_differential(data)
    
    # Additional aggregate features
    total_wear_percentage = 1 - data['non_wear_flag'].mean()
    
    # Max consecutive non-wear (in hours)
    nonwear_runs = (data['non_wear_flag'] != data['non_wear_flag'].shift()).cumsum()
    nonwear_lengths = data[data['non_wear_flag'] == 1].groupby(nonwear_runs[data['non_wear_flag'] == 1]).size()
    max_consecutive_nonwear = float(nonwear_lengths.max() * 5 / 3600) if len(nonwear_lengths) > 0 else 0.0
    
    # Day/night compliance ratio
    data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
    day_mask = (data['hour'] >= 8) & (data['hour'] <= 20)
    night_mask = ~day_mask
    
    day_wear = 1 - data[day_mask]['non_wear_flag'].mean() if day_mask.any() else 1.0
    night_wear = 1 - data[night_mask]['non_wear_flag'].mean() if night_mask.any() else 1.0
    day_night_ratio = day_wear / (night_wear + 1e-6)
    
    # Compliance trend slope
    data['date'] = pd.to_datetime(data['timestamp']).dt.date
    daily_wear = data.groupby('date')['non_wear_flag'].apply(lambda x: 1 - x.mean())
    if len(daily_wear) >= 2:
        x = np.arange(len(daily_wear))
        slope, _, _, _, _ = stats.linregress(x, daily_wear.values)
        compliance_trend_slope = float(slope)
    else:
        compliance_trend_slope = 0.0
    
    # Combine all features
    cv = {
        # Battery features (4)
        'beta_batt': battery_features['beta_batt'],
        'charge_variance': battery_features['charge_variance'],
        'halt_count': battery_features['halt_count'],
        'mean_trough_voltage': battery_features['mean_trough_voltage'],
        
        # NDI features (4)
        'ndi': ndi_features['ndi'],
        'ndi_ratio': ndi_features['ndi_ratio'],
        'night_removal_count': ndi_features['night_removal_count'],
        'mean_night_removal_duration': ndi_features['mean_night_removal_duration'],
        
        # Micro-removal features (4)
        'mu_freq': micro_features['mu_freq'],
        'entropy_mu': micro_features['entropy_mu'],
        'mean_micro_duration': micro_features['mean_micro_duration'],
        'max_daily_micro_count': micro_features['max_daily_micro_count'],
        
        # Sensory features (3)
        'v_sens': sensory_features['v_sens'],
        'high_agitation_ratio': sensory_features['high_agitation_ratio'],
        'max_pre_removal_enmo': sensory_features['max_pre_removal_enmo'],
        
        # Weekend and aggregate features (5)
        'delta_wknd': weekend_features['delta_wknd'],
        'total_wear_percentage': float(total_wear_percentage),
        'max_consecutive_nonwear': float(max_consecutive_nonwear),
        'day_night_compliance_ratio': float(min(day_night_ratio, 10.0)),
        'compliance_trend_slope': float(compliance_trend_slope)
    }
    
    return cv
