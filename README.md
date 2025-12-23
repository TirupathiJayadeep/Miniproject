# ComplianceNet: Compliance-Driven Prediction of Problematic Internet Use (PIU)

ComplianceNet is a specialized machine learning framework designed to predict Problematic Internet Use (PIU) in children and adolescents using actigraphy data (wearable sensor data). Instead of focusing solely on physical activity, ComplianceNet leverages *compliance behavior*—how participants interact with the wearable device itself—as a primary predictor.

## 🚀 Key Features

- **Compliance Vector Extraction Engine (CVEE):** Extracts 20 high-dimensional features related to device usage, including:
  - Battery neglect patterns
  - Nocturnal disconnect index
  - Micro-removal frequency
  - Sensory rejection signatures
  - Weekend dropout differentials
- **DTW-Enhanced Temporal Clustering Module (DTCM):** Discovers compliance phenotypes (e.g., "Escapist", "Disorganized", "Sensory Avoider") using Dynamic Time Warping.
- **Transformer-based Missingness Encoder (TME):** A deep learning architecture that encodes multi-day non-wear sequences and predicts the Severity Impairment Index (SII).
- **End-to-End Pipeline:** Integrates data preprocessing, feature extraction, phenotype classification, and severity prediction.

## 📁 Repository Structure

```text
├── src/
│   ├── cvee.py         # Compliance Vector Extraction Engine
│   ├── dtcm.py         # DTW-Enhanced Temporal Clustering
│   ├── tme.py          # Transformer-based Missingness Encoder
│   ├── pipeline.py     # End-to-end prediction pipeline
│   └── utils.py        # Utility functions (QWK, CV helpers, etc.)
├── tests/
│   ├── test_compliance_net.py       # Core module tests (75 tests)
│   ├── test_data_validation.py      # Schema & quality tests (15 tests)
│   ├── test_model_performance.py    # Predictive performance tests (21 tests)
│   └── test_phenotype_classification.py # Phenotype logic tests (22 tests)
├── demo.py             # Full pipeline demonstration
└── pytest.ini          # Test configuration
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/compliancenet.git
    cd compliancenet
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas scikit-learn scipy
    ```

## 💻 Usage

### Running the Demo
The `demo.py` script generates synthetic actigraphy data and runs the entire pipeline, including phenotype classification and severity prediction.

```bash
python demo.py
```

### Basic Pipeline Integration
```python
from src.pipeline import full_pipeline_predict, classify_phenotype
import pandas as pd

# Load your actigraphy data
data = pd.read_csv("your_data.csv")

# Classify compliance phenotype
phenotype = classify_phenotype(data)
print(f"Primary Phenotype: {phenotype['primary']}")

# Predict Severity Impairment Index (SII)
sii = full_pipeline_predict(data)
print(f"Predicted SII: {sii}")
```

## 🧪 Testing

ComplianceNet includes a robust suite of **133 tests** covering all aspects of the framework.

Run all tests:
```bash
python -m pytest tests/ -v
```

Tests include:
- **Unit Tests:** Validating individual CVEE mathematical computations.
- **Integration Tests:** Verifying the full pipeline flow.
- **Validation Tests:** Ensuring incoming data meets schema requirements.
- **Performance Tests:** Checking model consistency and metric (QWK) correctness.

## 📖 Methodology

ComplianceNet is built on the hypothesis that non-wear patterns in actigraphy are not "missing data" but are instead valuable behavioral signals. It identifies three distinct clinical phenotypes:
1.  **Escapist:** High nocturnal removals, often associated with late-night screen use.
2.  **Disorganized:** Irregular removals and battery neglect, associated with ADHD-type traits.
3.  **Sensory Avoider:** Frequent short-duration removals triggered by agitation or sensory overload.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
