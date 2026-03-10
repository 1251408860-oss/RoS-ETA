core_experiments/README.md
# Core Experiments Heterogeneous Benchmark Evaluation

## Module Overview and Data Provenance

This module is explicitly engineered to serve as the primary evaluation sandbox for the RoS-ETA framework. The central objective is to rigorously benchmark our physics-informed robust regression architecture against state-of-the-art statistical defense baselines under a severe zero-trust assumption. To ensure the framework possesses universal applicability across varying network entropy environments the testbed integrates three highly heterogeneous publicly available datasets

- **CIC-IDS2017** representing enterprise IT networks with complex background noise is available at https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
- **UNSW-NB15** representing mixed academic environments containing intricate exploitation patterns is available at https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
- **CIC-IoT2023** representing massive IoT botnets governed by extremely rigid hardware constraints is available at https://www.kaggle.com/datasets/himadri07/ciciot2023

By systematically simulating high-intensity availability poisoning this setup evaluates the capability of our intent-aware spectral sanitization and dual-view fusion mechanisms to bypass the statistical density trap.

## Environment and Dependency Preparation

Executing this evaluation suite requires a robust Python data science ecosystem. The architecture is entirely CPU-based and deliberately avoids the prohibitive computational overhead associated with deep learning models making it highly suitable for standard desktop workstations.

A Python environment of version 3.8 or higher is mandatory. Please install the following fundamental system packages via your package manager

- **pandas** utilized for efficient chunked loading and structural manipulation of the massive raw datasets
- **numpy** deployed for high-speed matrix operations within the logarithmic physical space
- **scikit-learn** utilized to construct the Huber loss robust regressors and to instantiate the comparison baselines
- **scipy** required to execute the highly optimized eigenvalue decomposition during spectral sanitization
- **matplotlib** and **seaborn** deployed to render publication-ready distribution plots and adversarial trade-off visualizations

Bash

```
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

## Step 1 Heterogeneous Data Preprocessing

Before executing the core algorithms the raw network telemetry must be mathematically harmonized into a unified physical feature space. We authored three distinct preprocessing scripts utilizing dynamic chunking mechanisms to safely handle gigabyte-scale files without triggering memory overflows. Please ensure the downloaded raw CSV datasets are placed in their respective target directories before running the following scripts

Bash

```
python data_handle_CIC-IDS2017.py
python data_handle_UNSW_NB15.py
python data_handle_CICIOT23.py
```

Running the respective scripts accomplishes critical dataset-specific normalizations The CIC-IDS2017 pipeline extracts foundational metrics and applies logarithmic transformations to stabilize heavy-tailed distributions. The UNSW-NB15 script autonomously aggregates bidirectional byte streams and strictly synchronizes temporal units from seconds to microseconds bridging the architectural gap between distinct capture environments. The CIC-IoT2023 pipeline seamlessly maps highly specific IoT attributes to our standardized physical invariants ensuring strict dimensional alignment across all benchmarks.

## Step 2 Threat Modeling and Physical Manifold Construction

Once the datasets are preprocessed the main evaluation script takes over to simulate the adversarial environment and construct the physical defense.

Bash

```
python two_model.py
```

Upon execution the system autonomously injects a **10%** availability poisoning ratio explicitly prioritizing volumetric DoS samples to mimic a severe collaborative attack. The framework then extracts macro-rheological metrics and intelligently searches for micro-entropy proxies mapping the highly distorted raw telemetry into a stable log-linear manifold. This phase guarantees that traditional density-based detectors will face a heavily manipulated feature space where malicious traffic clusters perfectly mimic the statistical core of legitimate data distributions.

## Step 3 Spectral Sanitization and Robust Regression

This step constitutes the algorithmic core of the RoS-ETA defense. The integrated spectral sanitizer automatically computes the covariance matrix of the augmented physical features and extracts the principal right singular vector to identify low-rank attack intents. An adaptive cutoff is then applied to the spectral weights to precisely excise high-energy anomalies.

Subsequently the system parallelizes the training of two independent regressors utilizing the Huber loss function. These dual-view models focus respectively on the rheology and capacity physical dimensions autonomously tuning their penalty parameters to achieve optimal convergence on the benign main sequence band outperforming classic statistical baselines including Isolation Forest and RANSAC.

## Scientific Discoveries and Result Evaluation

Running the automated pipeline yields a comprehensive suite of analytical visualizations directly within the results directory validating the theoretical claims of the framework.

The output artifacts systematically verify our core defensive advantages The generated scatter plots visually confirm the **Physical Linearization Proof** demonstrating that highly heteroscedastic raw traffic mathematically converges into strict linear boundaries after the physical embedding operator is applied. The ROC comparison charts demonstrate absolute **Algorithmic Superiority** where RoS-ETA establishes decisive performance barriers against complex threats while statistical baselines catastrophically collapse into the density trap. Most importantly the adaptive attack simulation explicitly quantifies **The Rate Paradox Enforcement** through evasion cost scatter plots. This empirical data proves that any attacker attempting to mathematically fit the benign manifold is forced to endure massive duration inflation triggering an inescapable collapse in their effective tactical throughput.
