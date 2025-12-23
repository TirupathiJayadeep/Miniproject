"""
Data Validation Test Suite
==========================
Tests for validating input data quality and structure.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ============================================================================
# SCHEMA VALIDATION TESTS
# ============================================================================

class TestDataSchema:
    """Tests for data schema validation."""
    
    def test_required_columns_present(self, sample_data):
        """Test all required columns are present."""
        required_cols = [
            'timestamp', 'enmo', 'anglez', 'light', 
            'battery_voltage', 'non_wear_flag'
        ]
        
        assert all(col in sample_data.columns for col in required_cols)
    
    def test_timestamp_is_datetime(self, sample_data):
        """Test timestamp column is datetime type."""
        assert pd.api.types.is_datetime64_any_dtype(sample_data['timestamp'])
    
    def test_numeric_columns_are_numeric(self, sample_data):
        """Test numeric columns have correct dtype."""
        numeric_cols = ['enmo', 'anglez', 'light', 'battery_voltage']
        
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_data[col])
    
    def test_non_wear_flag_is_binary(self, sample_data):
        """Test non_wear_flag contains only 0 and 1."""
        assert set(sample_data['non_wear_flag'].unique()).issubset({0, 1})


class TestDataQuality:
    """Tests for data quality checks."""
    
    def test_no_duplicate_timestamps(self, sample_data):
        """Test no duplicate timestamps exist."""
        assert not sample_data['timestamp'].duplicated().any()
    
    def test_timestamps_sorted(self, sample_data):
        """Test timestamps are sorted chronologically."""
        assert sample_data['timestamp'].is_monotonic_increasing
    
    def test_consistent_sampling_rate(self, sample_data):
        """Test sampling rate is consistent (5 seconds)."""
        diffs = sample_data['timestamp'].diff()[1:]
        expected_diff = timedelta(seconds=5)
        
        # Allow some tolerance for missing epochs
        valid_diffs = (diffs == expected_diff) | (diffs > expected_diff)
        assert valid_diffs.all()
    
    def test_enmo_non_negative(self, sample_data):
        """Test ENMO values are non-negative."""
        assert (sample_data['enmo'] >= 0).all()
    
    def test_light_non_negative(self, sample_data):
        """Test light values are non-negative."""
        assert (sample_data['light'] >= 0).all()
    
    def test_battery_voltage_range(self, sample_data):
        """Test battery voltage is in valid range."""
        valid_range = (sample_data['battery_voltage'] >= 3000) & \
                      (sample_data['battery_voltage'] <= 4500)
        # Most should be valid
        assert valid_range.mean() > 0.95
    
    def test_anglez_range(self, sample_data):
        """Test anglez is in valid range (-90 to 90)."""
        assert (sample_data['anglez'] >= -90).all()
        assert (sample_data['anglez'] <= 90).all()


class TestDataCompleteness:
    """Tests for data completeness."""
    
    def test_minimum_duration(self, sample_data):
        """Test minimum recording duration (1 day)."""
        duration = sample_data['timestamp'].max() - sample_data['timestamp'].min()
        assert duration >= timedelta(days=1)
    
    def test_minimum_wear_percentage(self, sample_data):
        """Test minimum wear percentage (20%)."""
        wear_pct = 1 - sample_data['non_wear_flag'].mean()
        assert wear_pct >= 0.20
    
    def test_no_full_nan_columns(self, sample_data):
        """Test no columns are entirely NaN."""
        for col in sample_data.columns:
            assert not sample_data[col].isna().all()
    
    def test_nan_percentage_threshold(self, sample_data):
        """Test NaN percentage is below threshold (5%)."""
        numeric_cols = ['enmo', 'anglez', 'light', 'battery_voltage']
        
        for col in numeric_cols:
            nan_pct = sample_data[col].isna().mean()
            assert nan_pct < 0.05, f"{col} has {nan_pct:.1%} NaN values"


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Generate valid sample data."""
    np.random.seed(42)
    n_epochs = 17281  # Just over 1 day (17280 = 24h exactly at 5s intervals)
    
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_epochs, freq='5s'),
        'enmo': np.random.exponential(30, n_epochs),
        'anglez': np.random.uniform(-90, 90, n_epochs),
        'light': np.random.exponential(100, n_epochs),
        'battery_voltage': np.linspace(4200, 3500, n_epochs),
        'non_wear_flag': np.random.binomial(1, 0.1, n_epochs)
    })


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
