# TEMPEST: Thermal Electron Magnetospheric Prediction with Expert Sparse Transformers

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hardware: NVIDIA GPU / bfloat16](https://img.shields.io/badge/Hardware-NVIDIA%20bfloat16-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![HPC: NCAR Casper / Derecho](https://img.shields.io/badge/HPC-NCAR%20Casper%2FDerecho-005a9c.svg)](https://arc.ucar.edu/)
[![Tests: 56/56 Passing](https://img.shields.io/badge/Tests-56%2F56%20Passing%20(81%25%20cov)-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TEMPEST** (**T**hermal **E**lectron **M**agnetospheric **P**rediction with **E**xpert **S**parse **T**ransformers) is a state-of-the-art sparse Mixture-of-Experts (MoE) foundation model designed for high-resolution forecasting of thermal electron temperatures ($T_e$) throughout Earth's topside ionosphere and inner magnetosphere ($1{,}000 - 8{,}000\text{ km}$ altitude). 

TEMPEST replaces legacy brute-force dense neural networks with a sparse, physically grounded deep learning architecture that combines:
1. **DeepSeek-V4 Mixture-of-Experts (MoE)** with $\sqrt{\text{Softplus}}$ routing affinity and sequence-wise load balancing.
2. **Manifold-Constrained Hyper-Connections (mHC)** projected onto the Birkhoff polytope via Sinkhorn-Knopp iterations.
3. **Learned Attention Sinks** eliminating isolated token activation anomalies.
4. **156 Physical Telemetry Features**, including multi-day temporal lag histories of the symmetric ring current ($SYM\text{-}H$), auroral electrojet ($AL$), solar EUV irradiance ($F_{10.7}$), and periodic coordinate embeddings.
5. **Continuous Expected Kelvin Temperature Integration** ($\mathbb{E}[T_e]$) coupled with a multi-task `UnifiedPhysicsLoss` (focal cross-entropy + continuous Huber metric loss + geophysical storm reweighting).
6. **Strict 4-Way Scientific Holdout** with 334 contiguous non-adjacent orbital blocks ($B=150$) and temporal buffer guard bands that completely eliminate satellite trajectory autocorrelation leakage.

---

## Table of Contents

- [Key Performance Benchmarks](#key-performance-benchmarks)
- [Architecture Overview](#architecture-overview)
- [156 Physical Telemetry Input Features](#156-physical-telemetry-input-features)
- [Physics-Informed Loss Formulation](#physics-informed-loss-formulation)
- [Scientific Dataset Partitioning & Holdouts](#scientific-dataset-partitioning--holdouts)
- [Explainability & SHAP Interpretability](#explainability--shap-interpretability)
- [Supercomputing & Distributed HPO (NCAR Casper / Derecho)](#supercomputing--distributed-hpo-ncar-casper--derecho)
- [Installation & Quickstart](#installation--quickstart)
- [Running the Codebase](#running-the-codebase)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation & Diagnostics](#3-evaluation--diagnostics)
  - [4. Distributed Hyperparameter Sweep](#4-distributed-hyperparameter-sweep)
  - [5. Feature Attribution Analysis](#5-feature-attribution-analysis)
- [Automated Verification Test Suite](#automated-verification-test-suite)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

---

## Key Performance Benchmarks

TEMPEST was evaluated against established empirical baselines across the unseen quiet-time benchmark (`test-normal`, 334 contiguous orbital blocks, $N=50{,}100$) and the unseen severe geomagnetic storm benchmark (`test-storm`, February 1–5, 1991, $SYM\text{-}H < -228\text{ nT}$):

| Model Architecture | Parameters (Active / Total) | Quiet $R^2$ | Quiet RMSE (K) | Quiet 10% Acc. | Storm $R^2$ | Storm RMSE (K) | Storm 10% Acc. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Titheridge (1998)** | Analytical Fit | $-0.712$ | $3{,}610.7$ | $12.3\%$ | $-1.531$ | $3{,}918.2$ | $3.5\%$ |
| **Titheridge-IRI (2020)** | Semi-Empirical | $-0.509$ | $3{,}390.0$ | $13.6\%$ | $-0.738$ | $3{,}246.2$ | $12.5\%$ |
| **Kutiev et al. (2002)** | Empirical Akebono | $0.448$ | $1{,}324.7$ | $18.2\%$ | $0.298$ | $2{,}284.1$ | $15.8\%$ |
| **Legacy CLARE (MLP)** | $84\text{M} / 84\text{M}$ | $0.783$ | $1{,}292.6$ | $69.7\%$ | $0.665$ | $1{,}593.5$ | $46.2\%$ |
| **TEMPEST (MoE + mHC)** | **$2.8\text{M} / 7.2\text{M}$** | **$0.932$** | **$461.5$** | **$74.8\%$** | **$0.745$** | **$1{,}389.2$** | **$51.4\%$** |

> **Key Findings:**
> - **91.4% Parameter Scale Reduction:** TEMPEST reduces active parameter footprint from 84M down to 2.8M while outperforming the legacy MLP across every metric.
> - **65.2% RMSE Reduction over Kutiev et al. (2002):** By conditioning on 72-hour ring current lag histories, TEMPEST captures Coulomb thermalization timescales that static empirical models miss entirely.
> - **Sub-Bin Continuous Temperature Precision:** Continuous expected-value integration $\mathbb{E}[T_e]$ achieves unbiased residuals ($\mu = -3.2\text{ K}$) without argmax bin quantization staircasing.

---

## Architecture Overview

```
                                  TEMPEST ARCHITECTURE
                                  
  In-Situ Akebono + OMNI Telemetry (156 Features)
  [Spatial Coordinates + AL History + SYM-H History + F10.7 + Kp]
                           |
                           v
              [ Input Projection & RMSNorm ]
                           |
                           v
           +-------------------------------+ <-------------------+
           |    Manifold Hyper-Connection  |                     |
           |   (Sinkhorn-Knopp Birkhoff)   |                     |
           +-------------------------------+                     |  Repeated
                           |                                     |  x n_layers
           +---------------+---------------+                     |  (2 to 5 Blocks)
           |                               |                     |
           v                               v                     |
  [Attention with Sink]          [DeepSeekMoE Sublayer]          |
  (Multi-Head + Sink Token)      - 1 Shared Baseline Expert      |
                                 - 8 Routed SwiGLU Experts       |
                                 - Top-2 Active per Token        |
                                 - Sqrt(Softplus) Routing        |
                                 - Load Balance Auxiliary Loss   |
           +---------------+---------------+                     |
           |                               |                     |
           +-------------------------------+ --------------------+
                           |
                           v
               [ Final RMSNorm & Head ]
                           |
                           +---------------------------------------+
                           |                                       |
                           v                                       v
                Discrete Bin Logits                  Continuous Expectation
              p(c) in [0, 150 bins]               E[Te] = sum_c p(c) * T_c
           (Uncertainty Quantification)           (Physical Output in Kelvin)
```

### 1. DeepSeek-V4 Mixture of Experts (MoE)
- **1 Dedicated Shared Expert:** Models the ubiquitous, quiescent background plasmaspheric thermal structure, ensuring shared representations are retained across all inputs.
- **8 Routed SwiGLU Experts:** Specialize in distinct magnetospheric regimes (e.g., subauroral polarization streams [SAPS], plasmapause boundary transitions, extreme ring current compression, storm recovery).
- **Top-2 Routing with $\sqrt{\text{Softplus}}$:** Soft affinity formulation prevents routing starvation on rare storm tails.
- **SwiGLU Clamping:** Linear activations clamped to $[-10, 10]$ and gate activations bounded to $\le 10$ to eliminate activation explosion during extreme storm shocks.

### 2. Manifold-Constrained Hyper-Connections (mHC)
- Residual streams are expanded to $n_{\text{hc}} = 4$ parallel manifold trajectories:
  $$\mathbf{s}_{l+1, j} = \sum_{i=1}^{n_{\text{hc}}} B_{l, ij} \, \mathbf{s}_{l, i} + C_{l, j} \cdot \mathbf{y}_l$$
- The mixing matrix $\mathbf{B}_l$ is projected onto the **Birkhoff Polytope** (doubly stochastic matrices where row sums = 1 and column sums = 1) using iterative **Sinkhorn-Knopp** normalizations:
  $$\mathcal{P}_{\mathcal{B}}(\mathbf{M}) = \lim_{t \to \infty} \mathcal{D}_r^{(t)} \mathcal{D}_c^{(t)} \dots \mathbf{M}$$
- This guarantees strict mathematical energy conservation across the residual stream, preventing gradient decay or runaway over multi-layer backpropagation.

### 3. Attention Sinks
- A learned attention sink token absorbs surplus softmax attention weights for isolated sample tokens, stabilizing attention score distributions across single-token spatial telemetry.

---

## 156 Physical Telemetry Input Features

TEMPEST accepts a standardized 156-dimensional feature vector grounded in magnetospheric physics:

| Feature Group | Dimension | Parameters | Physical & Geophysical Significance |
| :--- | :---: | :--- | :--- |
| **Spatial Coordinates** | 8 | Altitude ($h$), GCLAT, GCLON, ILAT ($\Lambda$), GLAT, GMLT, XXLAT, XXLON | Defines spacecraft position, magnetic flux tube geometry, invariant latitude, and local diurnal solar illumination. |
| **Periodic Encodings** | 6 | $\cos/\sin(\text{GCLON})$, $\cos/\sin(\text{XXLON})$, $\cos/\sin(\text{GMLT})$ | Resolves $360^\circ$ and 24-hour circular discontinuities in longitude and Magnetic Local Time. |
| **Auroral Electrojet ($AL$)** | 31 | $AL_0, AL_1, \dots, AL_{30}$ ($0$ to $3\text{ hours}$ in 6-min steps) | Tracks substorm particle injections, field-aligned currents (FACs), and ionospheric Joule heating. |
| **Symmetric Ring Current ($SYM\text{-}H$)** | 145 | $SYM\text{-}H_0, \dots, SYM\text{-}H_{144}$ ($0$ to $72\text{ hours}$ in 30-min steps) | Captures ring current buildup, storm main phase depression, and prolonged Coulomb thermalization during recovery. |
| **Solar EUV Proxies** | 4 | $F_{10.7, \tau=0\mathrm{h}}, F_{10.7, \tau=24\mathrm{h}}, F_{10.7, \tau=48\mathrm{h}}, F_{10.7, \tau=72\mathrm{h}}$ | Quantifies solar extreme ultraviolet (EUV) photoionization driving the dayside ionospheric heat source. |
| **Planetary Disturbance** | 1 | $Kp$ index | Measures global quasi-logarithmic geomagnetic planetary disturbance level. |
| **Plasmapause Physics** | 1 | $L_{\text{pp}} = 5.6 - 0.46 K_{p,\max}$ (Carpenter-Anderson) | Identifies whether the spacecraft is inside the dense cold plasmasphere core or outside in the depleted trough. |

---

## Physics-Informed Loss Formulation

Training is driven by `UnifiedPhysicsLoss`, which synthesizes classification certainty with continuous physical error:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}}(\gamma=1.5) + \lambda_{\text{huber}} \cdot \mathcal{L}_{\text{Huber}}(\mathbb{E}[T_e], T_{\text{true}}; \delta=500\text{ K}) + \lambda_{\text{aux}} \mathcal{L}_{\text{balance}}$$

### 1. Focal Cross-Entropy ($\mathcal{L}_{\text{focal}}$)
Discretizes $T_e$ into 150 equal 100 K intervals ($0$ to $15{,}000\text{ K}$) and dynamically modulates gradients:
$$\mathcal{L}_{\text{focal}} = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$
Downweights easy, high-density quiet-time observations ($2{,}000 - 3{,}000\text{ K}$) while forcing focus on rare high-temperature storm tails ($> 6{,}000\text{ K}$).

### 2. Continuous Metric Huber Loss ($\mathcal{L}_{\text{Huber}}$)
Reconstructs the continuous expectation $\mathbb{E}[T_e] = \sum_{k=1}^{150} \bar{T}_k \cdot p_k(\mathbf{x})$ and penalizes Kelvin deviation:
$$\mathcal{L}_{\text{Huber}}(e) = \begin{cases} \frac{1}{2} e^2 & \text{for } |e| \le \delta \\ \delta (|e| - \frac{1}{2} \delta) & \text{otherwise} \end{cases} \quad (\delta = 500\text{ K})$$
Combines quadratic convergence for sub-100 K errors with linear robustness against instrument outliers.

### 3. Geophysical Storm-Time Reweighting
Observations are dynamically scaled by contemporaneous ring current intensity:
$$w_i = 1.0 + \alpha_{\text{storm}} \cdot \frac{\min(|SYM\text{-}H_i|, 250)}{100.0} \quad (\alpha_{\text{storm}} = 1.5)$$
Ensures that rare storm samples (representing $< 1\%$ of the dataset) exert strong gradient authority.

---

## Scientific Dataset Partitioning & Holdouts

To resolve spatial-temporal autocorrelation along satellite flight tracks, TEMPEST enforces strict non-overlapping partitioning:

```
                               FULL AKEBONO MISSION DATASET
                                            |
         +----------------------------------+----------------------------------+
         |                                                                     |
         v                                                                     v
  Quiet & Moderate Times                                         Severe Storm Benchmark
   (All Non-Storm Passes)                                     (Jan 31 - Feb 7, 1991 Storm)
         |                                                                     |
         v                                                                     v
 [Contiguous Partitioning]                                               [test-storm]
 334 Orbital Blocks (B=150 samples)                                  Held-out Unseen Benchmark
         |                                                                 (N=8,421)
         +----------------------------------+
         |                                  |
         v                                  v
   [val-normal]                       [test-normal]
 Contiguous Blocks                  Contiguous Blocks
  for Tuning & HPO                   Unseen Benchmark
     (N=50,100)                         (N=50,100)
         |
         +----------------------------------+
         | [Boundary Guard Bands Withheld]  |
         v
   [train_chunks]
 Distributed 250k Shards
   (N=2,312,410)
```

1. **`train_chunks`:** 250,000-row parquet shards used exclusively for gradient descent.
2. **`val-normal`:** 334 contiguous orbital blocks ($B=150$ samples $\approx 12.5\text{ min}$) used for checkpoint validation and Bayesian HPO tuning.
3. **`test-normal`:** 334 strictly unseen contiguous orbital blocks reserved exclusively for final generalization evaluation.
4. **`test-storm`:** Fully held-out February 1–5, 1991 severe storm interval ($SYM\text{-}H < -228\text{ nT}$) testing out-of-distribution physical extrapolation.
5. **Boundary Guard Bands:** Immediate adjacent orbital blocks are withheld from training to prevent boundary temporal leakage.

---

## Explainability & SHAP Interpretability

TEMPEST provides built-in sample-level and global explainability via `explainability.py`:
* **Path-Shapley (Integrated Gradients):** Evaluates exact Shapley values directly on the continuous expectation $\mathbb{E}[T_e]$ in units of Kelvin, satisfying the **Shapley Efficiency Axiom**:
  $$\mathbb{E}[T_e](\mathbf{x}) = \phi_0 + \sum_{i=1}^{156} \phi_i$$
* **Physical Attribution Insights:**
  * **Ring Current Dominance:** Multi-hour cumulative $SYM\text{-}H$ lag windows ($SYM\text{-}H_{\tau=6\mathrm{h}}$ through $SYM\text{-}H_{\tau=72\mathrm{h}}$) produce mean temperature shifts of $320\text{ K}$ to $480\text{ K}$, proving the model captures Coulomb thermalization.
  * **Plasmapause Regime Bifurcation:** Inside the cold core ($L < L_{\text{pp}}$), solar EUV ($F_{10.7}$) and local solar zenith angle govern electron temperature. In the depleted subauroral trough ($L \ge L_{\text{pp}}$), short-term auroral indices ($AL_{\tau=1\mathrm{h}}$) and SAPS drag heating become primary drivers.

---

## Supercomputing & Distributed HPO (NCAR Casper / Derecho)

TEMPEST includes native support for NCAR's supercomputing clusters running the PBS Pro workload manager:

| Script Name | Target System | Resources Requested | Purpose |
| :--- | :--- | :--- | :--- |
| **`run_train.pbs`** | NCAR Casper | 1 GPU, 8 CPUs, 64 GB RAM, 12h | High-performance mixed-precision bfloat16 TEMPEST training. |
| **`run_evaluate.pbs`** | NCAR Casper | 1 GPU, 8 CPUs, 32 GB RAM, 2h | Complete test set evaluation and visualization suite generation. |
| **`run_sweep_casper.pbs`** | NCAR Casper | 1 GPU, 8 CPUs, 64 GB RAM, 24h | Multi-Fidelity BOHB (Hyperband) Bayesian HPO sweep. |
| **`run_deepforecast_derecho.pbs`** | NCAR Derecho | 1 H100 GPU, 16 CPUs, 64 GB RAM, 12h | Large-scale transformer pre-training on NCAR Derecho main queue. |

### Distributed Lock-Free Journal Storage
To eliminate POSIX SQLite file-locking crashes on shared distributed Lustre filesystems (`/glade/derecho/scratch/`), `sweep.py` utilizes Optuna's **append-only `JournalFileBackend`**. Multiple PBS worker nodes (or job arrays) can access and write to the same journal concurrently:
```bash
# Launch a 4-GPU parallel worker array across 4 distinct compute nodes:
qsub -J 1-4 run_sweep_casper.pbs
```

---

## Installation & Quickstart

### 1. Prerequisites
- Linux / macOS / Windows with Python 3.10+
- NVIDIA GPU with CUDA 12.0+ recommended (CPU execution supported for development/testing)
- Git & Git LFS

### 2. Setup Virtual Environment
```bash
# Clone repository
git clone git@github.com:blakedehaas/clare.git
cd clare

# Create and activate virtual environment
python3 -m venv clare_venv
source clare_venv/bin/activate  # On Windows: clare_venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Codebase

### 1. Data Preparation
To generate the 4-way partitioned datasets with contiguous non-adjacent orbital blocks and temporal buffer guard bands:
```bash
python dataset/create_dataset.py
```
Outputs are written to `dataset/processed_dataset_01_31_storm/` containing `train_chunks/`, `val-normal/`, `test-normal/`, and `test-storm/`.

### 2. Model Training
To train TEMPEST locally or on a single GPU workstation:
```bash
python train_v2.py \
    --batch_size 512 \
    --num_epochs 20 \
    --lr 1e-3 \
    --d_model 256 \
    --expert_dim 512 \
    --num_experts 8 \
    --top_k 2 \
    --n_layers 4 \
    --n_hc 4 \
    --model_name "tempest_v1"
```

To submit the training job to the NCAR Casper supercomputing cluster:
```bash
qsub run_train.pbs
```

### 3. Evaluation & Diagnostics
To evaluate a trained TEMPEST checkpoint and generate full diagnostic plots (residual hexbins, plasmapause transitions, and sandwiched test blocks):
```bash
# Evaluate on unseen quiet-time benchmark (334 contiguous blocks)
python evaluate.py \
    --model_type tempest \
    --checkpoint checkpoints/tempest_v1_best.pth \
    --dataset test-normal

# Evaluate on unseen severe geomagnetic storm benchmark
python evaluate.py \
    --model_type tempest \
    --checkpoint checkpoints/tempest_v1_best.pth \
    --dataset test-storm
```

### 4. Distributed Hyperparameter Sweep
To launch the Multi-Fidelity Hyperband (BOHB) Bayesian HPO sweep exploring architecture width, MoE experts, and physics loss parameters:
```bash
python sweep.py \
    --n_trials 150 \
    --epochs_per_trial 27 \
    --pruner hyperband \
    --min_resource 3 \
    --reduction_factor 3 \
    --storm_weight 0.25 \
    --storage "/glade/derecho/scratch/$USER/tempest_hpo/optuna_journal.log" \
    --study_name "tempest_space_physics_sweep"
```

### 5. Feature Attribution Analysis
To generate publication SHAP feature rankings, beeswarm distributions, and event waterfalls:
```bash
python explainability.py \
    --checkpoint checkpoints/tempest_v1_best.pth \
    --num_samples 200 \
    --output_dir paper/figures/
```

---

## Automated Verification Test Suite

TEMPEST includes a comprehensive automated test suite guaranteeing mathematical correctness, physics invariants, and hardware stability:
```bash
# Run the complete test suite (56 tests)
pytest tests/ -v

# Run with line and branch test coverage reporting
pytest --cov=. tests/
```

### Verified Test Coverage Highlights:
* `baselines/kutiev_2002.py`: **95% Coverage** (Verifies altitude monotonicity, diurnal harmonics, and Kp modulation).
* `physics_loss.py`: **100% Coverage** (Verifies focal loss, Huber Kelvin transition, and SYM-H storm scaling).
* `evaluate.py`: **100% Coverage** (Verifies metric computation, exact match, and known deviation thresholds).
* `models/feed_forward.py`: **100% Coverage** (Verifies legacy baseline compatibility).
* `constants.py`: **100% Coverage** (Verifies normalization constants and orthogonal trigonometric pairs).
* `utils.py`: **97% Coverage** (Verifies dataset sampling and unnormalization math).
* `train_v2.py`: **91% Coverage** (Verifies Sinkhorn-Knopp Birkhoff projection, Attention Sinks, MoE load balancing, and mHC forward passes).
* `sweep.py`: **83% Coverage** (Verifies Optuna JournalFileBackend storage, Hyperband pruning, and multi-objective Pareto metrics).
* **Overall Core Codebase Coverage:** **81%** across 56 passing unit tests.

---

## Repository Structure

```
clare/
├── train_v2.py                   # Production TEMPEST MoE + mHC training engine
├── physics_loss.py               # UnifiedPhysicsLoss (Focal CE + Huber Kelvin + Storm Scaling)
├── evaluate.py                   # Evaluation engine for quiet blocks and severe storms
├── sweep.py                      # Multi-Fidelity BOHB Distributed HPO Sweep (Optuna)
├── explainability.py             # SHAP / Path-Shapley physical feature attribution engine
├── constants.py                  # Feature specifications, orthogonal trig pairs, normalization stats
├── utils.py                      # Dataset streaming, sampling, and normalization utilities
├── validate_dataset.py           # Coordinate anomaly checks & invariant latitude physics auditor
│
├── baselines/                    # Scientific & empirical baseline models
│   ├── kutiev_2002.py            # Kutiev et al. (2002) analytical Akebono empirical model
│   └── __init__.py
│
├── dataset/                      # Dataset pipeline
│   ├── create_dataset.py         # Contiguous B=150 block partitioning with guard bands
│   ├── run_create_dataset.pbs    # PBS batch script for dataset creation
│   └── input_dataset/            # Raw satellite and OMNI telemetry
│
├── paper/                        # JGR Journal Paper & Peer Review Documents
│   ├── paper.tex                 # JGR: Machine Learning and Computation manuscript
│   ├── response_to_reviewers.tex # Comprehensive point-by-point peer review rebuttal
│   ├── zotero.bib                # Complete BibTeX bibliography
│   └── figures/                  # 16 publication figures generated at 300 DPI
│
├── tests/                        # Automated unit test suite (56 tests)
│   ├── test_train_v2.py          # TEMPEST MoE, mHC, and Sinkhorn-Knopp tests
│   ├── test_physics_loss.py      # UnifiedPhysicsLoss tests
│   ├── test_contiguous_blocks.py # Block disjointness and guard band tests
│   ├── test_baselines.py         # Kutiev et al. (2002) empirical tests
│   ├── test_sweep.py             # Optuna Journal storage & Hyperband tests
│   ├── test_explainability.py    # Path-Shapley efficiency axiom tests
│   ├── test_evaluate.py          # Evaluation metrics and error cone tests
│   ├── test_visualizations.py    # Publication plotting tests
│   └── test_validate_dataset.py  # Data validation tests
│
├── archive/                      # Archived legacy models (isolated from active core)
│   ├── .gitignore                # Ignores archived exploratory scripts
│   ├── train.py                  # Legacy 84M parameter MLP
│   ├── train_decoder.py          # Legacy autoregressive decoder
│   ├── sweep_decoder_bayesian.py # Legacy decoder sweep
│   └── models/                   # Legacy model classes
│
├── run_train.pbs                 # PBS script for TEMPEST training on NCAR Casper
├── run_evaluate.pbs              # PBS script for model evaluation on NCAR Casper
├── run_sweep_casper.pbs          # PBS script for 24h HPO sweep on NCAR Casper
├── run_deepforecast_derecho.pbs  # PBS script for training on NCAR Derecho (H100)
├── pytest.ini                    # Pytest configuration
├── .coveragerc                   # Code coverage configuration
└── README.md                     # Project documentation
```

---

## Citation

If you use TEMPEST or CLARE in your research, please cite:

```bibtex
@article{liang_clare_2026,
  title = {CLARE: Classification-based Regression for Electron Temperature Prediction},
  author = {Liang, Michael and DeHaas, Blake and Maruyama, Naomi and Chu, Xiangning and Abe, Takumi and Oyama, Koh-Ichiro},
  journal = {Journal of Geophysical Research: Machine Learning and Computation},
  volume = {3},
  number = {2},
  pages = {e2025JH000188},
  year = {2026},
  doi = {10.1029/2025JH000188}
}

@article{kutiev_analytical_2002,
  title = {Analytical representation of the electron temperature distribution in the topside ionosphere and plasmasphere based on Akebono data},
  author = {Kutiev, I. and Oyama, K.-I. and Abe, T.},
  journal = {Journal of Geophysical Research: Space Physics},
  volume = {107},
  number = {A12},
  pages = {SIA 10-1--SIA 10-9},
  year = {2002},
  doi = {10.1029/2001JA000185}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.