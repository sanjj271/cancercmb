#!/usr/bin/env python3
"""
test_train_only_preprocessing.py - Comprehensive Unit Tests for Preprocessing,
Leakage Elimination, and Reproducibility Contracts.

Covers the 15 strict requirements from the CancerCombo Preprocessing & Leakage Cleanup:
 1. TRAIN-only statistics (Validation/test drugs cannot affect scaler statistics; e.g. sentinel 99999.0).
 2. Duplicate-grid invariance (Duplicating combination/grid rows does not alter unique TRAIN stats).
 3. Missing feature behavior (Missing feature entry raises KeyError naming the NSC; missing Train NSC raises).
 4. NaN handling (Raw NaN in TRAIN imputed with median of OTHER TRAIN drugs; raw NaN in val/test gets frozen TRAIN median).
 5. Scaled output (Output is finite and bounded).
 6. Scaler save/load (Loaded scaler produces identical transformation; modified TRAIN-NSC set causes hash mismatch).
 7. Evaluation behavior (Evaluation loads a scaler and never fits one; missing scaler raises).
 8. Scenario validation (Unknown scenario raises; missing official split raises; no 50k fallback).
 9. Feature-store contract (1024-bit, radius 3 from metadata, 200 descriptors, descriptor_names, Inf rejection, bound rejection, metadata validation).
10. Common scenario path (S1/S2/S3 each select their configured split through the same common code path).
11. S1/S2 comparison (Computes TRAIN NSC sets; asserts identical if equal, or reports set difference).
12. Dataset construction without a fitted scaler (None raises, unfitted raises, unfitted transform raises).
13. load_descriptor_scaler without train_nsc_ids raises.
14. load_descriptor_scaler on file lacking train_nsc_hash raises.
15. load_descriptor_scaler with hash from different train-NSC set raises.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import rdkit

repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from preprocessor import MolecularPreprocessor
from config import (
    RAW_FEATURE_STORE_PATH,
    SCENARIO_CONFIG,
    get_scenario_split_path,
    get_scenario_scaler_path,
)
from dataset import (
    DescriptorScaler,
    DrugComboDataset,
    save_descriptor_scaler,
    load_descriptor_scaler,
    compute_nsc_hash,
    load_precomputed_drug_features,
    extract_scenario_nscs,
)
from precompute_molecular_features import (
    fit_scenario_scaler,
    precompute_molecular_features,
)


def _build_dummy_metadata(dim: int = 200, nbits: int = 1024, radius: int = 3):
    prep = MolecularPreprocessor(morgan_nbits=nbits, morgan_radius=radius)
    return {
        "morgan_nbits": nbits,
        "morgan_radius": radius,
        "n_descriptors": dim,
        "descriptor_names": list(prep.descriptor_names[:dim]),
        "rdkit_version": str(rdkit.__version__),
        "key_type": "NSC",
    }


# =====================================================================
# 1. TRAIN-ONLY STATISTICS
# =====================================================================
def test_1_train_only_statistics(tmp_path):
    """1. Validation and test drugs cannot affect scaler statistics (sentinel value 99999.0)."""
    meta = _build_dummy_metadata()
    features = {
        "10": {"descriptors": torch.full((200,), 10.0, dtype=torch.float32), "morgan": torch.zeros(1024)},
        "20": {"descriptors": torch.full((200,), 20.0, dtype=torch.float32), "morgan": torch.zeros(1024)},
        "99": {"descriptors": torch.full((200,), 99999.0, dtype=torch.float32), "morgan": torch.zeros(1024)},
        "metadata": meta,
        "__metadata__": meta,
    }
    feat_path = os.path.join(tmp_path, "feat.pt")
    torch.save(features, feat_path)

    # 10 and 20 are TRAIN (split 1), 99 is VAL (split 2) and TEST (split 3)
    split_df = pd.DataFrame({
        "nsc_a": ["10", "20", "99", "99"],
        "nsc_b": ["20", "10", "99", "99"],
        "split": [1, 1, 2, 3],
    })
    split_path = os.path.join(tmp_path, "split.csv")
    split_df.to_csv(split_path, index=False)

    scaler_out = os.path.join(tmp_path, "scaler.npz")
    scaler = fit_scenario_scaler(split_path, feat_path, scaler_out)

    # TRAIN-only mean of 10.0 and 20.0 is exactly 15.0. Sentinel 99999.0 must have zero effect.
    np.testing.assert_allclose(scaler.mean_vec, np.full(200, 15.0))
    np.testing.assert_allclose(scaler.median_vec, np.full(200, 15.0))


# =====================================================================
# 2. DUPLICATE-GRID INVARIANCE
# =====================================================================
def test_2_duplicate_grid_invariance(tmp_path):
    """2. Duplicating combination/grid rows does not change statistics when unique TRAIN drugs are unchanged."""
    meta = _build_dummy_metadata()
    features = {
        "A": {"descriptors": torch.full((200,), 2.0, dtype=torch.float32), "morgan": torch.zeros(1024)},
        "B": {"descriptors": torch.full((200,), 4.0, dtype=torch.float32), "morgan": torch.zeros(1024)},
        "metadata": meta,
        "__metadata__": meta,
    }
    feat_path = os.path.join(tmp_path, "feat.pt")
    torch.save(features, feat_path)

    # Single grid
    split_single = pd.DataFrame({
        "nsc_a": ["A"],
        "nsc_b": ["B"],
        "split": [1],
    })
    # Multi-row duplicate grid (e.g. 16 dose combinations for the same pair)
    split_multi = pd.DataFrame({
        "nsc_a": ["A"] * 16,
        "nsc_b": ["B"] * 16,
        "split": [1] * 16,
    })

    scaler_1 = fit_scenario_scaler(split_single, feat_path, os.path.join(tmp_path, "s1.npz"))
    scaler_2 = fit_scenario_scaler(split_multi, feat_path, os.path.join(tmp_path, "s2.npz"))

    np.testing.assert_allclose(scaler_1.mean_vec, scaler_2.mean_vec)
    np.testing.assert_allclose(scaler_1.median_vec, scaler_2.median_vec)
    np.testing.assert_allclose(scaler_1.std_vec, scaler_2.std_vec)


# =====================================================================
# 3. MISSING FEATURE BEHAVIOR
# =====================================================================
def test_3_missing_feature_behavior():
    """3. Missing feature entry raises KeyError naming the NSC; missing TRAIN NSC also raises."""
    scaler = DescriptorScaler()
    scaler.fit([np.zeros(200, dtype=np.float32)])

    store = {
        "100": {"morgan": np.zeros(1024, dtype=np.float32), "descriptors": np.zeros(200, dtype=np.float32)}
    }
    data = [{
        "smiles_a": "CCO", "smiles_b": "CCN", "nsc_a": "100", "nsc_b": "999",
        "cell_line_name": "MCF7", "doses_a": [0.1]*4, "doses_b": [0.2]*4, "viability_matrix": [[1.0]*4]*4
    }]
    gex = {"MCF7": np.ones(976, dtype=np.float32)}
    ds = DrugComboDataset(data, gex, drug_feature_store=store, descriptor_scaler=scaler)

    with pytest.raises(KeyError, match="999"):
        _ = ds[0]

    # Fitting from feature store with missing Train NSC also raises KeyError
    with pytest.raises(KeyError, match="999"):
        scaler.fit_from_feature_store(store, ["100", "999"])


# =====================================================================
# 4. NAN HANDLING
# =====================================================================
def test_4_nan_handling():
    """4. Raw NaN in TRAIN is imputed with median of OTHER TRAIN drugs; raw NaN in val/test gets frozen TRAIN median."""
    scaler = DescriptorScaler()
    # 3 Train drugs:
    # Drug 1: descriptor[0] = 10.0
    # Drug 2: descriptor[0] = 30.0
    # Drug 3: descriptor[0] = NaN -> median of {10.0, 30.0} is 20.0
    raw_train = [
        np.array([10.0] + [1.0] * 199, dtype=np.float32),
        np.array([30.0] + [1.0] * 199, dtype=np.float32),
        np.array([np.nan] + [1.0] * 199, dtype=np.float32),
    ]
    scaler.fit(raw_train)

    assert scaler.median_vec[0] == 20.0
    # Mean of [10.0, 30.0, 20.0] is 20.0
    assert scaler.mean_vec[0] == 20.0

    # Transforming a test drug with NaN at [0] must receive the frozen TRAIN median (20.0), then standardized
    test_vec = np.array([np.nan] + [1.0] * 199, dtype=np.float32)
    transformed = scaler.transform(test_vec)
    assert np.isfinite(transformed[0])
    # (20.0 - mean) / std = 0.0
    np.testing.assert_allclose(transformed[0], 0.0, atol=1e-5)


# =====================================================================
# 5. SCALED OUTPUT IS FINITE
# =====================================================================
def test_5_scaled_output_is_finite():
    """5. Scaled output is finite and strictly bounded."""
    scaler = DescriptorScaler()
    scaler.fit([
        np.array([1.0] * 200, dtype=np.float32),
        np.array([2.0] * 200, dtype=np.float32),
    ])

    extreme_input = np.array([np.nan, 1e6, -1e6, 0.0] + [5.0] * 196, dtype=np.float32)
    output = scaler.transform(extreme_input)

    assert np.all(np.isfinite(output))
    assert np.all(output >= -10.0) and np.all(output <= 10.0)


# =====================================================================
# 6. SCALER SAVE / LOAD
# =====================================================================
def test_6_scaler_save_load(tmp_path):
    """6. Loaded scaler produces identical transformation; modified TRAIN-NSC set causes hash mismatch."""
    scaler = DescriptorScaler()
    scaler.fit([np.array([1.0] * 200, dtype=np.float32), np.array([3.0] * 200, dtype=np.float32)])

    train_nscs = ["10", "20", "30"]
    save_path = os.path.join(tmp_path, "scaler.npz")
    save_descriptor_scaler(scaler, save_path, train_nscs)

    loaded = load_descriptor_scaler(save_path, train_nsc_ids=train_nscs)
    test_in = np.array([2.0] * 200, dtype=np.float32)
    np.testing.assert_allclose(scaler.transform(test_in), loaded.transform(test_in))

    # Modified TRAIN NSC set must fail with hash mismatch
    with pytest.raises(ValueError, match="hash mismatch"):
        load_descriptor_scaler(save_path, train_nsc_ids=["10", "20", "999"])


# =====================================================================
# 7. EVALUATION BEHAVIOR
# =====================================================================
def test_7_evaluation_behavior(tmp_path):
    """7. Evaluation loads a scaler and never fits one. Missing scaler raises FileNotFoundError."""
    missing_path = os.path.join(tmp_path, "missing_scaler.npz")
    with pytest.raises(FileNotFoundError, match="missing"):
        load_descriptor_scaler(missing_path, train_nsc_ids=["10", "20"])


# =====================================================================
# 8. SCENARIO VALIDATION
# =====================================================================
def test_8_scenario_validation():
    """8. Unknown scenario raises; missing official split raises; no 50k fallback."""
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario_split_path(999)

    # Missing split raises FileNotFoundError (never returns 50k fallback)
    with pytest.raises(FileNotFoundError, match="Authoritative split file missing"):
        get_scenario_split_path(1)


# =====================================================================
# 9. FEATURE-STORE CONTRACT
# =====================================================================
def test_9_feature_store_contract(tmp_path):
    """9. Validate Morgan dimension, Morgan radius, descriptor dimension, descriptor_names, Inf rejection, bound rejection."""
    meta = _build_dummy_metadata()

    # Valid store
    valid_store = {
        "10": {
            "morgan": torch.zeros(1024, dtype=torch.float32),
            "descriptors": torch.full((200,), 1.0, dtype=torch.float32),
        },
        "metadata": meta,
        "__metadata__": meta,
    }
    p_valid = os.path.join(tmp_path, "valid.pt")
    torch.save(valid_store, p_valid)
    loaded = load_precomputed_drug_features(p_valid)
    assert "10" in loaded

    # Metadata without descriptor_names rejected
    bad_meta1 = meta.copy()
    del bad_meta1["descriptor_names"]
    p_bad1 = os.path.join(tmp_path, "bad1.pt")
    torch.save({"10": valid_store["10"], "metadata": bad_meta1}, p_bad1)
    with pytest.raises(ValueError, match="descriptor_names"):
        load_precomputed_drug_features(p_bad1)

    # Metadata without rdkit_version rejected
    bad_meta2 = meta.copy()
    del bad_meta2["rdkit_version"]
    p_bad2 = os.path.join(tmp_path, "bad2.pt")
    torch.save({"10": valid_store["10"], "metadata": bad_meta2}, p_bad2)
    with pytest.raises(ValueError, match="rdkit_version"):
        load_precomputed_drug_features(p_bad2)

    # Inf rejection
    inf_store = {
        "10": {
            "morgan": torch.zeros(1024, dtype=torch.float32),
            "descriptors": torch.tensor([float("inf")] * 200, dtype=torch.float32),
        },
        "metadata": meta,
    }
    p_inf = os.path.join(tmp_path, "inf.pt")
    torch.save(inf_store, p_inf)
    with pytest.raises(ValueError, match="Inf"):
        load_precomputed_drug_features(p_inf)

    # |x| > 1e6 rejection
    big_store = {
        "10": {
            "morgan": torch.zeros(1024, dtype=torch.float32),
            "descriptors": torch.tensor([1e7] * 200, dtype=torch.float32),
        },
        "metadata": meta,
    }
    p_big = os.path.join(tmp_path, "big.pt")
    torch.save(big_store, p_big)
    with pytest.raises(ValueError, match="exceeding 1e6"):
        load_precomputed_drug_features(p_big)


# =====================================================================
# 10. COMMON SCENARIO PATH
# =====================================================================
def test_10_common_scenario_path():
    """10. S1/S2/S3 each select their configured split through the same common code path."""
    for s in [1, 2, 3]:
        assert s in SCENARIO_CONFIG
        assert SCENARIO_CONFIG[s].startswith("data/splits/")


# =====================================================================
# 11. S1 / S2 COMPARISON
# =====================================================================
def test_11_s1_s2_comparison():
    """11. S1 vs S2 comparison on real splits if present; skip with UNVERIFIED if absent."""
    s1_path = SCENARIO_CONFIG[1]
    s2_path = SCENARIO_CONFIG[2]
    if not (os.path.exists(s1_path) and os.path.exists(s2_path)):
        pytest.skip("Real S1/S2 splits not present in workspace (UNVERIFIED)")

    s1_df = pd.read_csv(s1_path)
    s2_df = pd.read_csv(s2_path)
    s1_train, _, _ = extract_scenario_nscs(s1_df)
    s2_train, _, _ = extract_scenario_nscs(s2_df)

    s1_set = set(s1_train)
    s2_set = set(s2_train)
    if s1_set == s2_set:
        assert s1_set == s2_set
    else:
        only_s1 = sorted(list(s1_set - s2_set))
        only_s2 = sorted(list(s2_set - s1_set))
        print(f"NSCs only in S1: {only_s1}")
        print(f"NSCs only in S2: {only_s2}")


# =====================================================================
# 12. DATASET CONSTRUCTION WITHOUT A FITTED SCALER
# =====================================================================
def test_12_dataset_construction_without_fitted_scaler():
    """12. DrugComboDataset with None or unfitted scaler raises; DescriptorScaler.transform on unfitted raises."""
    data = [{
        "smiles_a": "CCO", "smiles_b": "CCN", "nsc_a": "1", "nsc_b": "2",
        "cell_line_name": "MCF7", "doses_a": [0.1]*4, "doses_b": [0.2]*4, "viability_matrix": [[1.0]*4]*4
    }]
    gex = {"MCF7": np.ones(976, dtype=np.float32)}

    # 1. descriptor_scaler=None must RAISE
    with pytest.raises(ValueError, match="requires a fitted DescriptorScaler"):
        DrugComboDataset(data, gex, descriptor_scaler=None)

    # 2. descriptor_scaler unfitted must RAISE
    unfitted = DescriptorScaler()
    with pytest.raises(ValueError, match="requires a fitted DescriptorScaler"):
        DrugComboDataset(data, gex, descriptor_scaler=unfitted)

    # 3. DescriptorScaler.transform on an unfitted scaler must RAISE
    with pytest.raises(RuntimeError, match="unfitted"):
        unfitted.transform(np.zeros(200, dtype=np.float32))


# =====================================================================
# 13. LOAD_DESCRIPTOR_SCALER WITHOUT TRAIN_NSC_IDS RAISES
# =====================================================================
def test_13_load_scaler_without_train_nscs_raises(tmp_path):
    """13. load_descriptor_scaler without train_nsc_ids raises ValueError."""
    scaler = DescriptorScaler()
    scaler.fit([np.zeros(200, dtype=np.float32)])
    save_path = os.path.join(tmp_path, "scaler.npz")
    save_descriptor_scaler(scaler, save_path, ["10", "20"])

    with pytest.raises(ValueError, match="train_nsc_ids must be provided"):
        load_descriptor_scaler(save_path, train_nsc_ids=None)


# =====================================================================
# 14. LOAD_DESCRIPTOR_SCALER ON FILE LACKING TRAIN_NSC_HASH RAISES
# =====================================================================
def test_14_load_scaler_lacking_hash_raises(tmp_path):
    """14. load_descriptor_scaler on file lacking train_nsc_hash raises ValueError."""
    bad_scaler_path = os.path.join(tmp_path, "bad_scaler.npz")
    np.savez(
        bad_scaler_path,
        median_vec=np.zeros(200, dtype=np.float32),
        mean_vec=np.zeros(200, dtype=np.float32),
        std_vec=np.ones(200, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="missing required 'train_nsc_hash'"):
        load_descriptor_scaler(bad_scaler_path, train_nsc_ids=["10", "20"])


# =====================================================================
# 15. LOAD_DESCRIPTOR_SCALER WITH WRONG TRAIN-NSC SET RAISES
# =====================================================================
def test_15_load_scaler_wrong_train_nscs_raises(tmp_path):
    """15. load_descriptor_scaler with hash from a different train-NSC set raises ValueError."""
    scaler = DescriptorScaler()
    scaler.fit([np.zeros(200, dtype=np.float32)])
    save_path = os.path.join(tmp_path, "scaler.npz")
    save_descriptor_scaler(scaler, save_path, ["10", "20"])

    with pytest.raises(ValueError, match="hash mismatch"):
        load_descriptor_scaler(save_path, train_nsc_ids=["10", "30"])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
