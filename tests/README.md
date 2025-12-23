# ComplianceNet Test Suite Documentation

## Overview

This test suite provides comprehensive validation for the ComplianceNet system. The tests cover all three stages of the architecture plus data validation and performance metrics.

---

## Test Files

| File | Purpose | Test Count |
|------|---------|------------|
| `test_compliance_net.py` | Main component tests (CVEE, DTCM, TME) | ~80 tests |
| `test_data_validation.py` | Input data schema and quality | ~20 tests |
| `test_phenotype_classification.py` | Phenotype detection accuracy | ~25 tests |
| `test_model_performance.py` | Performance metrics and benchmarks | ~25 tests |

---

## Test Categories

### 1. Stage 1: CVEE Tests (Compliance Vector Extraction Engine)

| Test Class | What It Validates |
|------------|-------------------|
| `TestBatteryNeglectSlope` | β_batt calculation, HALT detection, charge variance |
| `TestNocturnalDisconnectIndex` | NDI calculation, night interval boundaries, light context |
| `TestMicroRemovalFrequency` | Micro-removal counting, duration filtering, entropy |
| `TestSensoryRejectionVector` | V_sens calculation, agitation detection, ENMO spikes |
| `TestWeekendDropoutDifferential` | Δ_wknd calculation, weekend detection |
| `TestComplianceVectorIntegration` | All 20 features extracted correctly |

### 2. Stage 2: DTCM Tests (DTW Clustering)

| Test Class | What It Validates |
|------------|-------------------|
| `TestDayProfileGeneration` | 96-point profiles, binary values, day exclusion |
| `TestSoftDTWClustering` | 4 clusters, correct assignments, Sakoe-Chiba band |
| `TestPhenotypeDistribution` | Distribution sums to 1, dominant phenotype detection |

### 3. Stage 3: TME Tests (Transformer Encoder)

| Test Class | What It Validates |
|------------|-------------------|
| `TestSequencePreprocessing` | 720-length sequences, padding, binary values |
| `TestPositionalEncoding` | Sinusoidal encoding, time-of-day, day-of-week |
| `TestTransformerArchitecture` | Output shape, attention extraction, gradient flow |
| `TestCORNOrdinalLoss` | Loss calculation, rank consistency |
| `TestPredictionDecoding` | SII decoding from ordinal probabilities |

### 4. Phenotype Tests

| Test Class | What It Validates |
|------------|-------------------|
| `TestEscapistPhenotype` | Nocturnal removal, low light, PCIAT correlation |
| `TestDisorganizedPhenotype` | Battery neglect, HALT events, ADHD correlation |
| `TestSensoryAvoiderPhenotype` | Agitation triggers, short removals, ASD correlation |
| `TestPhenotypeDiscrimination` | Clear separation between phenotypes |

### 5. Performance Tests

| Test Class | What It Validates |
|------------|-------------------|
| `TestQWKCalculation` | QWK metric correctness |
| `TestBaselineBenchmarks` | Beats random, majority, meets target QWK |
| `TestPerClassPerformance` | Severe/Moderate recall, balanced predictions |
| `TestAblationPerformance` | Each component contributes positively |
| `TestCrossValidation` | Fold variance, confidence intervals |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_compliance_net.py -v

# Run specific test class
pytest tests/test_compliance_net.py::TestBatteryNeglectSlope -v

# Run with markers
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ -m "unit"       # Only unit tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Test Data Fixtures

The test suite includes fixtures that generate synthetic data for each phenotype:

| Fixture | Description |
|---------|-------------|
| `sample_actigraphy_data` | 1 day of generic actigraphy data |
| `multi_day_data` | 7 days of actigraphy data |
| `escapist_phenotype_data` | Nocturnal removal pattern with low light |
| `disorganized_phenotype_data` | Battery drain and HALT events |
| `sensory_avoider_phenotype_data` | High ENMO before short removals |
| `compliant_data` | 100% wear, no removals |

---

## Expected Results Summary

### Target Performance Metrics

| Metric | Baseline | Target | Test Threshold |
|--------|----------|--------|----------------|
| QWK | 0.42 | 0.50 | ≥ 0.45 |
| Severe Recall | 0.25 | 0.48 | ≥ 0.40 |
| Moderate Recall | 0.38 | 0.52 | ≥ 0.45 |
| CV Fold Variance | - | < 0.01 | < 0.01 |

### Phenotype Detection Accuracy

| Phenotype | Expected Accuracy | Test Threshold |
|-----------|------------------|----------------|
| Escapist | 85% | > 70% |
| Disorganized | 80% | > 70% |
| Sensory Avoider | 75% | > 65% |

---

## Implementation Notes

### Stub Functions

The test files contain stub functions marked with `raise NotImplementedError`. These must be implemented in the main source code:

```python
# Core functions to implement:
- calculate_battery_neglect_slope(data)
- calculate_nocturnal_disconnect_index(data)
- calculate_micro_removal_frequency(data)
- calculate_sensory_rejection_vector(data)
- calculate_weekend_dropout_differential(data)
- extract_compliance_vector(data)
- generate_day_profiles(data)
- perform_dtw_clustering(profiles, n_clusters)
- calculate_phenotype_distribution(clusters)
- preprocess_for_transformer(data)
- create_tme_model()
- classify_phenotype(data)
- predict_sii(data)
```

### Test Dependencies

```
pytest>=7.0
numpy>=1.21
pandas>=2.0
scikit-learn>=1.0
```

---

## Continuous Integration

Add to `.github/workflows/tests.yml`:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short
```
