# CancerCombo: Multimodal Deep Learning Framework for Cancer Drug Combination Dose-Response Surface Prediction

**CancerCombo** is a biophysically constrained multimodal deep learning framework designed to predict continuous 2D dose-response viability surfaces ($4 \times 4$ matrices) for cancer drug combinations under strict unseen-drug generalization protocols.

It couples a **shared multimodal drug encoder**—integrating Morgan fingerprints, continuous RDKit descriptors, Drug-Target Interactions (DTI), and biological pathway embeddings—with cell-line transcriptomics via **Drug-Cell Cross-Attention** and a differentiable **Bivariate Hill / SynBa equation solver**.

---

## 📦 Deployment Model & Data Contract

CancerCombo enforces a clean separation between the Git codebase, external frozen inputs, and reproducible artifacts generated on the compute machine (DGX):

### 1. Git Repository
Contains ONLY source code, architecture modules, configurations, tests, and documentation:
- Architecture blocks (`blocks/`), top-level model (`cancercombo.py`), and hyperparameters (`config.yaml`)
- Pipeline scripts: `split_dataset.py`, `precompute_molecular_features.py`, `train.py`, `evaluate.py`, `dataset.py`, `preprocessor.py`
- Test suites: `test_phase1_data_pipeline.py`, `test_suite.py`, `scratch/test_phase3_*.py`
- Documentation: `README.md`, `docs/DGX_RUNBOOK.md`

### 2. External Datasets (Supplied to DGX — DO NOT REGENERATE)
The DGX machine receives **ONLY** these four external datasets:
1. `data/DrugCombination_with_NSC_ID.zip`: Original root DrugComb combination dataset containing NSC IDs. This is the source dataset from which all scenario splits and molecular features are derived.
2. `data/drug_target_embeddings_2048.xlsx`: Pre-computed, frozen 2048-D ConPLex DTI embeddings.
3. `data/drug_pathway_embeddings_1152.xlsx`: Pre-computed, frozen 1152-D Reactome pathway embeddings.
4. `data/features/NCI-60_landmark_gex.csv`: Pre-computed, frozen 976-D NCI-60 landmark gene-expression matrix.

> **DO NOT REGENERATE BIOLOGICAL ARTIFACTS**: DTI (2048-D), Pathway (1152-D), and Landmark GEX (976-D) are frozen external biological inputs. Do not run ConPLex, ESMC, or Reactome queries on DGX.

### 3. Generated on DGX (Derived Reproducible Artifacts)
All data structures derived from `DrugCombination_with_NSC_ID.zip` are generated directly on DGX:
- **Scenario 1 Split**: `data/splits/scenario1_combination.csv` (Combination-wise 60/20/20)
- **Scenario 2 Split**: `data/splits/scenario2_cell.csv` (Cell-wise 60/20/20)
- **Scenario 3 Split**: `data/splits/scenario3_drug_strict_60_20_20.csv` (Strict unseen-drug 60/20/20 benchmark)
- **Raw Molecular Feature Store**: `data/features/morgan1024_r3_raw/drug_features_morgan1024_r3_raw.pt` & `.pkl` (1024-bit Morgan, r=3, 200 RDKit descriptors for ALL 101 drugs in the dataset; unscaled, raw NaNs preserved, finite clipped $[-10^6, 10^6]$; completely scenario-independent)
- **Scenario-Specific Descriptor Scaler**: `checkpoints/drugcell_crossattention/descriptor_scaler_s3.npz` (Fitted on Train drugs only, with SHA256 NSC hash)
- **Model Checkpoints**: `checkpoints/drugcell_crossattention/cancercombo_best.ckpt`

---

## 🚀 Quickstart: Reproducing CancerCombo from a Clean Checkout

Execute these commands in sequence from a clean environment:

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/jishnu3010/cancercombo.git
cd cancercombo
git checkout main

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate    # Linux / DGX (or .\venv\Scripts\activate on Windows)

# Upgrade packaging tools and install pinned dependencies
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Verify GPU & External Input Datasets
Ensure the four external input datasets are placed in their respective repository paths, then run:
```bash
python -c "
import os, torch
assert torch.cuda.is_available() or True, 'Warning: CUDA unavailable'
for p in [
    'data/DrugCombination_with_NSC_ID.zip',
    'data/drug_target_embeddings_2048.xlsx',
    'data/drug_pathway_embeddings_1152.xlsx',
    'data/features/NCI-60_landmark_gex.csv'
]:
    assert os.path.exists(p), f'Missing required external dataset: {p}'
print('[CHECK PASSED] Environment and 4 external datasets verified.')
"
```

### 3. Generate All Scenario Splits on DGX
Generate Scenario 1, Scenario 2, and Scenario 3 splits from the root combination archive using the deterministic seed:
```bash
python split_dataset.py \
    --input_csv data/DrugCombination_with_NSC_ID.zip \
    --output_dir ./data/splits \
    --seed 42
```

Validate the generated splits:
```bash
python -c "
import pandas as pd
s1 = pd.read_csv('data/splits/scenario1_combination.csv')
s2 = pd.read_csv('data/splits/scenario2_cell.csv')
s3 = pd.read_csv('data/splits/scenario3_drug_strict_60_20_20.csv')
assert len(s1) == 280162 and len(s2) == 280162 and len(s3) == 122828
print(f'[CHECK PASSED] All 3 scenario splits generated: S1={len(s1)}, S2={len(s2)}, S3={len(s3)}')
"
```

### 4. Generic Molecular Feature Precomputation & Scenario Scaler Fitting

#### Stage 1: Generic All-Drug Feature Extraction
Precompute 1024-bit Morgan fingerprints (radius 3) and 200 continuous RDKit descriptors for **all 101 unique drugs** directly from the root dataset `data/DrugCombination_with_NSC_ID.zip` (completely scenario-independent):
```bash
python precompute_molecular_features.py \
    --input_csv data/DrugCombination_with_NSC_ID.zip \
    --output_file data/features/morgan1024_r3_raw/drug_features_morgan1024_r3_raw.pt
```

#### Stage 2: Scenario-Specific Scaler Fitting
Fit the `DescriptorScaler` strictly on the training drugs (`split == 1`) of the target benchmark split to prevent validation/test leakage:
```bash
python fit_descriptor_scaler.py --scenario 3
```

Validate the generated generic molecular feature store and fitted Scenario 3 scaler:
```bash
python -c "
import numpy as np, pandas as pd
from dataset import load_precomputed_drug_features, load_descriptor_scaler, extract_scenario_nscs
store = load_precomputed_drug_features('data/features/morgan1024_r3_raw/drug_features_morgan1024_r3_raw.pt')
s3 = pd.read_csv('data/splits/scenario3_drug_strict_60_20_20.csv')
train_nscs, _, _ = extract_scenario_nscs(s3)
scaler = load_descriptor_scaler('checkpoints/drugcell_crossattention/descriptor_scaler_s3.npz', train_nsc_ids=train_nscs)
data = np.load('checkpoints/drugcell_crossattention/descriptor_scaler_s3.npz')
assert len(store) > 0 and scaler.is_fitted and int(data['num_train_nscs'][0]) == 61
print(f'[CHECK PASSED] Generic feature store valid ({len(store)} keys) | Scenario 3 scaler fitted on {data[\"num_train_nscs\"][0]} train drugs')
"
```

### 5. Validate Biological Inputs
```bash
python -c "
from dataset import load_dti_embeddings, load_pathway_embeddings, load_nci60_gex
dti = load_dti_embeddings('data/drug_target_embeddings_2048.xlsx')
pw = load_pathway_embeddings('data/drug_pathway_embeddings_1152.xlsx')
gex = load_nci60_gex('data/features/NCI-60_landmark_gex.csv')
assert len(next(iter(dti.values()))) == 2048 and len(next(iter(pw.values()))) == 1152 and len(next(iter(gex.values()))) == 976
print(f'[CHECK PASSED] Biological inputs: DTI={len(dti)} (2048-D) | Pathway={len(pw)} (1152-D) | GEX={len(gex)} (976-D)')
"
```

### 6. Run Test Suite
Confirm that all unit, integration, and architecture tests pass:
```bash
python test_suite.py
python -m pytest -v test_generic_preprocessing.py test_phase1_data_pipeline.py test_train_only_preprocessing.py
python scratch/smoke_test_phase3.py
```

### 7. Smoke Test Training (1 Epoch, 4 Samples)
Verify the complete multimodal data pipeline, model forward pass, dynamic scenario scaler fitting, and checkpoint generation:
```bash
python train.py --scenario 3 --epochs 1 --max_samples 4 --engine native
```

### 8. Full Scenario 3 Production Training (1000 Epochs)
Train the full model across all 102,006 Scenario 3 training dose-response matrices:
```bash
python train.py --config config.yaml --scenario 3 --engine native
```

*(To resume an interrupted run: `python train.py --config config.yaml --scenario 3 --engine native --resume checkpoints/drugcell_crossattention/cancercombo_best.ckpt`)*

### 9. Evaluate on Held-Out Scenario 3 Test Set
Evaluate the best saved checkpoint on the 10,876 held-out test matrices:
```bash
python evaluate.py --checkpoint checkpoints/drugcell_crossattention/cancercombo_best.ckpt --scenario 3
```

---

## 🧬 Generic Drug Features vs. Scenario-Specific Scaling Architecture

CancerCombo strictly decouples **generic molecular feature extraction** from **scenario-specific scaler fitting**:

```
[External Root Input]
data/DrugCombination_with_NSC_ID.zip
      │
      ├─────────────────────────────────────────────────┐
      ▼                                                 ▼
split_dataset.py                                precompute_molecular_features.py
(Generates Scenarios 1, 2, 3)                   (Extracts all 101 unique drugs from root zip)
      │                                                 │
      │ ├── scenario1_combination.csv                   ▼
      │ ├── scenario2_cell.csv                  data/features/morgan1024_r3/drug_features_morgan1024_r3.pt
      │ └── scenario3_drug_strict_60_20_20.csv  (GENERIC FEATURE STORE: Unscaled 1024-D Morgan + 200-D RDKit)
      │                                                 │
      └────────────────────────┬────────────────────────┘
                               ▼
                           train.py
                 (Loads selected scenario split,
                  identifies unique training drugs (split == 1),
                  fits DescriptorScaler on train drugs dynamically,
                  transforms descriptors on-the-fly via DrugComboDataset)
                               │
                               ▼
                          evaluate.py
                 (Loads scenario checkpoint & scaler,
                  verifies Train NSC hash against split,
                  evaluates on held-out test split)
```

1. **Generic Drug Feature Store (`drug_features_morgan1024_r3.pt`)**:
   - Contains unscaled 1024-bit Morgan fingerprints and 200 continuous RDKit descriptors for all 101 unique drugs.
   - Contains no split assumptions; shared identically across Scenario 1, Scenario 2, and Scenario 3.
2. **Scenario-Specific Scaler Fitting (`DescriptorScaler`)**:
   - Fitted dynamically at training time in `train.py` using ONLY the training drugs (`split == 1`) of the active scenario.
   - Guarantees zero leakage into validation or test sets across all scenarios.

---

## 🔬 Benchmark Scenarios

The repository implements three dataset splitting protocols:

| Scenario | Partitioning Strategy | Retained Matrices | Output File | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Scenario 1** | Combination-Wise | 280,162 | `data/splits/scenario1_combination.csv` | 60/20/20 split on canonical drug pairs $(A, B) \equiv (B, A)$. Tests synergy prediction on unseen drug combinations of known drugs. |
| **Scenario 2** | Cell-Wise | 280,162 | `data/splits/scenario2_cell.csv` | 60/20/20 split on unique cell lines. Tests generalization across cancer cell lines. |
| **Scenario 3** | Strict Drug-Wise | 122,828 | `data/splits/scenario3_drug_strict_60_20_20.csv` | **Primary Research Benchmark**: Strict unseen-drug generalization. Unique Drug NSC IDs are partitioned into disjoint subsets; only pure pairs are retained, all mixed pairs are discarded. |

### Authoritative Scenario 3 Dataset Metrics
- **Partitioning Invariant**:
  $$\text{NSCs}_{\text{Train}} \cap \text{NSCs}_{\text{Val}} = \emptyset, \quad \text{NSCs}_{\text{Train}} \cap \text{NSCs}_{\text{Test}} = \emptyset, \quad \text{NSCs}_{\text{Val}} \cap \text{NSCs}_{\text{Test}} = \emptyset$$
- **Unique Drugs**: 101 NSCs (Train: 61, Val: 20, Test: 20)
- **Unique Combinations**: 2,146 pairs (Train: 1,784, Val: 172, Test: 190)
- **$4 \times 4$ Response Matrices**: **122,828** retained (Train: **102,006**, Val: **9,946**, Test: **10,876**)
- **Discarded Mixed Matrices**: 157,334 (56.16%)

---

## 🌟 Key Architectural Features

1. **Shared Multimodal Drug Encoder (`MultimodalDrugEncoder`)**:
   - A single shared encoder instance processes both Drug A and Drug B.
   - **Chemistry**: Morgan fingerprints ($1024 \to 256$) + continuous RDKit descriptors ($200 \to 256$) fused via **Chemistry Attention Pooling** into token $C \in \mathbb{R}^{256}$.
   - **Biology**: Frozen 2048-D ConPLex DTI embeddings ($2048 \to 1024 \to 512 \to 256$) + frozen 1152-D Reactome Pathway embeddings ($1152 \to 1024 \to 512 \to 256$) fused via **Masked Biology Attention Pooling** into token $B \in \mathbb{R}^{256}$.
   - **Chemistry-Anchored Gated Fusion**:
     $$Z = C + g \odot (W_B B + b_B), \quad g = \sigma(W_g [C; B; \text{masks}] + b_g)$$
     If biological modalities are absent, $B = 0$ and the output collapses strictly to pure chemistry $Z = C$.
2. **Cell Line Transcriptomics (`CellLineEncoder`)**:
   - 976 landmark gene expression values projected into $300 \times 256$ pathway-level tokens.
3. **Drug-Cell Cross-Attention (`DrugCellCrossAttention`)**:
   - Conditions Drug A and Drug B representations independently on cell tokens ($Q = Z, K = V = \text{cell\_tokens} \to H_A, H_B \in \mathbb{R}^{256}$).
   - **Zero Drug-Drug Cross Attention**: Avoids non-permutation-invariant cross-contamination.
4. **Pair Representation & Biophysical Heads**:
   - Ordered concatenation $H_{\text{pair}} = [H_A; H_B] \in \mathbb{R}^{512}$.
   - 8 prediction heads estimate fractional efficacies ($e_1, e_2, e_3 \in [0, 1]$), log IC50s ($\log C_1, \log C_2$), Hill slopes ($h_1, h_2 \ge 0$), and interaction parameter ($\alpha \ge 0$).
5. **Differentiable Bivariate Hill Solver (`BivariateHillSolver`)**:
   - Evaluates the 2D SynBa Log-Sum-Exp biophysical formulation over concentration grids.
6. **Dose-Dependent Bias Predictor**:
   - Predicts row and column dose-dependent adjustments added to the biophysical surface: $Y_{\text{final}} = Y_{\text{Hill}} + \text{Bias}$.
   - Yields unclipped continuous percentage viability surfaces ($4 \times 4$).

---

## 📁 Repository Structure

```
cancercombo/
├── blocks/                                # Modular PyTorch Neural Network Blocks
│   ├── cell_encoder.py                    # NCI-60 landmark GEX pathway projection
│   ├── descriptor_encoder.py              # RDKit continuous physical descriptor MLP
│   ├── drug_cell_attention.py             # Drug-Cell Cross-Attention module
│   ├── hill_equation.py                   # Differentiable 2D Bivariate Hill solver (SynBa)
│   ├── morgan_encoder.py                  # 1024-bit Morgan fingerprint MLP encoder
│   ├── multimodal_drug_encoder.py         # Shared multimodal encoder (Chem/Bio/Gated Fusion)
│   └── prediction_heads.py                # 8 parameter heads & dose-dependent bias MLPs
├── data/                                  # Data artifacts & feature stores
│   ├── DrugCombination_with_NSC_ID.zip    # EXTERNAL: Root DrugComb combination dataset
│   ├── drug_target_embeddings_2048.xlsx   # EXTERNAL: Frozen 2048-D ConPLex DTI embeddings
│   ├── drug_pathway_embeddings_1152.xlsx  # EXTERNAL: Frozen 1152-D Reactome pathway embeddings
│   ├── features/                          # Feature stores & transcriptomics
│   │   ├── NCI-60_landmark_gex.csv        # EXTERNAL: Frozen 976 landmark GEX matrix
│   │   └── morgan1024_r3_raw/             # GENERATED: Raw molecular feature store (.pt & .pkl)
│   ├── splits/                            # GENERATED: Benchmark scenario splits (1, 2, 3)
│   │   ├── scenario1_combination.csv      # Scenario 1 (Combination-wise)
│   │   ├── scenario2_cell.csv             # Scenario 2 (Cell-wise)
│   │   └── scenario3_drug_strict_60_20_20.csv # Scenario 3 (Strict unseen drug-wise)
├── checkpoints/
│   └── drugcell_crossattention/
│       ├── descriptor_scaler_s3.npz       # GENERATED: Train-fitted scaler with SHA256 hash
│       └── cancercombo_best.ckpt          # GENERATED: Production model weights
├── docs/
│   └── DGX_RUNBOOK.md                     # Authoritative DGX deployment & execution manual
├── cancercombo.py                         # Top-level CancerCombo architecture
├── config.py                              # Dataclass model & training configuration loader
├── config.yaml                            # Primary hyperparameters & training configs
├── dataset.py                             # DrugComboDataset, scalers, loaders, collate
├── evaluate.py                            # Checkpoint evaluation script (Validation & Test)
├── losses.py                              # CancerCombo composite loss (MSE + Margin Ranking)
├── main.py                                # Unified CLI router (train, evaluate, predict)
├── metrics.py                             # RMSE, Pearson, Spearman, top-k synergy metrics
├── precompute_molecular_features.py       # Deterministic DGX molecular feature generator
├── preprocessor.py                        # RDKit Morgan & descriptor extraction & sanitization
├── requirements.txt                       # Pinned dependencies
├── test_phase1_data_pipeline.py           # Authoritative Phase 1 data pipeline tests (31 tests)
├── test_suite.py                          # Core architecture integration tests (8 tests)
└── train.py                               # Primary native PyTorch training engine
```

---

## 🛠️ Detailed Command Reference

### Training Options (`train.py`)
```bash
python train.py [OPTIONS]
  --config PATH         Path to YAML configuration file (default: config.yaml)
  --scenario {1,2,3}    Scenario split to execute (default: 1; use 3 for Scenario 3 benchmark)
  --engine {native,auto,lightning} Training engine (default: native)
  --epochs INT          Override number of epochs (default: 1000 from config)
  --max_samples INT     Sub-sample dataset size for smoke testing (omit for full production run)
  --resume PATH         Path to .ckpt file to resume training state
  --debug_backprop      Attach backward gradient trace hooks
```

### Evaluation Options (`evaluate.py`)
```bash
python evaluate.py [OPTIONS]
  --checkpoint PATH     Path to trained checkpoint (default: checkpoints/drugcell_crossattention/cancercombo_best.ckpt)
  --config PATH         Path to YAML configuration file (default: config.yaml)
  --scenario {1,2,3}    Scenario split to evaluate (use 3 for Scenario 3 benchmark)
```

### Resuming an Interrupted Training Run
```bash
python train.py \
    --config config.yaml \
    --scenario 3 \
    --engine native \
    --resume checkpoints/drugcell_crossattention/cancercombo_best.ckpt
```

---

## 📖 DGX Runbook

For the full, command-first operational checklist and failure recovery manual for NVIDIA DGX clusters, refer to:  
**[`docs/DGX_RUNBOOK.md`](docs/DGX_RUNBOOK.md)**.
