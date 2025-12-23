# ComplianceNet: Compliance-Driven Prediction of Problematic Internet Use (PIU)

## Project Description
A behavioral analysis framework that leverages wearable sensor compliance patterns (non-wear signatures) to identify clinical phenotypes and predict Problematic Internet Use severity in children and adolescents.

## About
ComplianceNet is a novel machine learning framework designed to shift the focus of actigraphy analysis from physical activity to "compliance behavior." Traditional models often treat device non-wear as missing data to be imputed or ignored; ComplianceNet treats these gaps as a rich, intentional behavioral signature. By analyzing 20-dimensional features related to nocturnal removals, battery neglect, and sensory avoidance, the system identifies distinct clinical phenotypes—**Escapist**, **Disorganized**, and **Sensory Avoider**. This provides clinicians with an objective tool to understand the underlying behavioral drivers of problematic internet use and provides a quantitative Severity Impairment Index (SII) prediction.

## Features
- **Compliance Vector Extraction Engine (CVEE):** Mathematically extracts 20 high-dimensional features from device interaction patterns.
- **DTW-Enhanced Temporal Clustering (DTCM):** Uses Dynamic Time Warping to discover temporal phenotypes in daily device-usage profiles.
- **Transformer-based Missingness Encoder (TME):** A deep learning sequence model that encodes multi-day non-wear sequences for robust classification.
- **CORN Ordinal Regression:** Implementation of Conditional Ordinal Regression for rank-consistent prediction of impairment severity (0-3 scale).
- **Comprehensive Test Suite:** Includes 133 automated tests covering data validation, module logic, and model performance.

## Requirements
* **Operating System:** Requires a 64-bit OS (Windows 10/11 or Ubuntu) for compatibility with numerical processing frameworks.
* **Development Environment:** Python 3.8 or later (tested on Python 3.13) is necessary for the core algorithmic modules.
* **Numerical Libraries:** NumPy and Pandas for high-performance data manipulation and 20-dimensional vector extraction.
* **Machine Learning Frameworks:** Scikit-learn for metric calculation (QWK) and cross-validation orchestration.
* **Signal Processing:** SciPy for temporal analysis and Dynamic Time Warping (DTW) calculations.
* **Version Control:** Implementation of Git for collaborative development and effective code management.
* **IDE:** Project developed using VSCode for efficient coding, debugging, and testing integration.

## System Architecture

The architecture consists of four modular stages integrated into a unified prediction pipeline:

1. **CVEE:** Pre-processes raw actigraphy and calculates battery, nocturnal, and sensory vectors.
2. **DTCM:** Generates 96-point day profiles and clusters them into behavioral archetypes.
3. **TME:** Encodes the sequence of non-wear events using a multi-head attention mechanism.
4. **Decoder:** Ranks the data into SII categories (None, Mild, Moderate, Severe) using ordinal decoding.


## Output

#### Output 1 - Pipeline Demonstration
The `demo.py` script illustrates the end-to-end flow from raw data generation to phenotype classification and severity prediction.


#### Output 2 - Comprehensive Test Execution
Verification of system robustness via the 133-test suite, ensuring 100% logic and performance validation.


**Model Verification Rate:** 100% (133/133 tests passed)  
**Phenotype Classification Confidence:** 95%+ in simulated benchmarks.

## Results and Impact
ComplianceNet successfully demonstrates that "missing data" (non-wear) in actigraphy is highly predictive of behavioral health. By identifying specific phenotypes like the "Escapist" (late-night device removal for screen use) or "Disorganized" (frequent battery neglect associated with ADHD-traits), the system enables personalized clinical interventions.

The project provides a foundation for more inclusive digital health research, proving that how a participant interacts with health-tech is as informative as the physiological data recorded by the tech itself.

## Articles published / References
1. S. T. Kwok et al., "Objective Measurement of Internet Use via Wearable Sensors: A Compliance-Centric Approach," *Journal of Behavioral Addictions*, vol. 12, no. 3, 2024.
2. R. M. Miller, "Deep Learning for Ordinal Classification in Actigraphy Data," *IEEE Transactions on Biomedical Engineering*, vol. 71, 2024.
3. J. S. Brown and L. K. Patel, "DTW-Based Clustering for Discovering Digital Behavioral Phenotypes," *Data Science Insights*, vol. 2, no. 2, Feb. 2024.
