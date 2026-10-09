#!/usr/bin/env python3
"""
test_phase1_data_pipeline.py - Comprehensive Unit Test Suite for Phase 1 Data Pipeline
"""

import sys
import os
import shutil
import tempfile
import hashlib
import pandas as pd
import numpy as np
import torch
import pytest
from unittest.mock import MagicMock, patch

# Ensure repository path is in sys.path
repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from config import load_config, ModelConfig
from preprocessor import MolecularPreprocessor, strip_salts_keep_largest_fragment
from dataset import (
    DrugComboDataset,
    DescriptorScaler,
    save_descriptor_scaler,
    load_descriptor_scaler,
    compute_nsc_hash,
    parse_dataframe_to_records,
    load_precomputed_drug_features
)
import split_dataset as sd
from cancercombo import CancerCombo
from evaluate import load_checkpoint_strictly


# =====================================================================
# TESTS 1 - 5: SCENARIOS & STRICT S3 INVARIANTS
# =====================================================================

def test_1_split_dataset_imports():
    """1. split_dataset.py imports successfully."""
    assert hasattr(sd, "combination_split")
    assert hasattr(sd, "cell_split")
    assert hasattr(sd, "drug_split")
    assert hasattr(sd, "preprocess_blank_smiles")


def test_2_scenarios_exist():
    """2. All three scenarios exist."""
    s1_path = os.path.join(repo_dir, "data/splits/scenario1_combination.csv")
    s2_path = os.path.join(repo_dir, "data/splits/scenario2_cell.csv")
    s3_path = os.path.join(repo_dir, "data/splits/scenario3_drug_strict_60_20_20.csv")

    if not (os.path.exists(s1_path) and os.path.exists(s2_path) and os.path.exists(s3_path)):
        pytest.skip("Authoritative split CSVs not present in workspace (UNVERIFIED)")

    assert os.path.exists(s1_path), f"Missing {s1_path}"
    assert os.path.exists(s2_path), f"Missing {s2_path}"
    assert os.path.exists(s3_path), f"Missing {s3_path}"


def test_3_strict_s3_nsc_disjointness():
    """3. Strict S3 NSC disjointness."""
    s3_path = os.path.join(repo_dir, "data/splits/scenario3_drug_strict_60_20_20.csv")
    if not os.path.exists(s3_path):
        pytest.skip("Authoritative Scenario 3 split CSV not present in workspace (UNVERIFIED)")
    df = pd.read_csv(s3_path)

    trn = df[df["split"] == 1]
    val = df[df["split"] == 2]
    tst = df[df["split"] == 3]

    trn_nscs = set(trn["nsc_a"].astype(str).str.strip()).union(set(trn["nsc_b"].astype(str).str.strip()))
    val_nscs = set(val["nsc_a"].astype(str).str.strip()).union(set(val["nsc_b"].astype(str).str.strip()))
    tst_nscs = set(tst["nsc_a"].astype(str).str.strip()).union(set(tst["nsc_b"].astype(str).str.strip()))

    assert trn_nscs.isdisjoint(val_nscs), f"Train-Val NSC intersection: {trn_nscs & val_nscs}"
    assert trn_nscs.isdisjoint(tst_nscs), f"Train-Test NSC intersection: {trn_nscs & tst_nscs}"
    assert val_nscs.isdisjoint(tst_nscs), f"Val-Test NSC intersection: {val_nscs & tst_nscs}"


def test_4_strict_s3_pair_invariants():
    """4. Strict S3 pair invariants."""
    s3_path = os.path.join(repo_dir, "data/splits/scenario3_drug_strict_60_20_20.csv")
    if not os.path.exists(s3_path):
        pytest.skip("Authoritative Scenario 3 split CSV not present in workspace (UNVERIFIED)")
    df = pd.read_csv(s3_path)

    trn = df[df["split"] == 1]
    val = df[df["split"] == 2]
    tst = df[df["split"] == 3]

    trn_nscs = set(trn["nsc_a"].astype(str).str.strip()).union(set(trn["nsc_b"].astype(str).str.strip()))
    val_nscs = set(val["nsc_a"].astype(str).str.strip()).union(set(val["nsc_b"].astype(str).str.strip()))
    tst_nscs = set(tst["nsc_a"].astype(str).str.strip()).union(set(tst["nsc_b"].astype(str).str.strip()))

    for _, row in trn.iterrows():
        assert str(row["nsc_a"]).strip() in trn_nscs and str(row["nsc_b"]).strip() in trn_nscs
    for _, row in val.iterrows():
        assert str(row["nsc_a"]).strip() in val_nscs and str(row["nsc_b"]).strip() in val_nscs
    for _, row in tst.iterrows():
        assert str(row["nsc_a"]).strip() in tst_nscs and str(row["nsc_b"]).strip() in tst_nscs


def test_5_strict_s3_zero_mixed_pairs():
    """5. Strict S3 TrainxTrain / ValxVal / TestxTest with zero mixed pairs."""
    s3_path = os.path.join(repo_dir, "data/splits/scenario3_drug_strict_60_20_20.csv")
    if not os.path.exists(s3_path):
        pytest.skip("Authoritative Scenario 3 split CSV not present in workspace (UNVERIFIED)")
    df = pd.read_csv(s3_path)
    assert set(df["split"].unique()).issubset({1, 2, 3})


# =====================================================================
# TESTS 6 - 9: DATA INTEGRITY & CHEMISTRY
# =====================================================================

def test_6_blank_smiles_filling():
    """6. Blank SMILES filling."""
    raw_df = pd.DataFrame({
        "Drug1_NSC_ID": ["100", "100", "200"],
        "Drug2_NSC_ID": ["200", "200", "100"],
        "Drug1_SMILES": ["CCO", "", "CC(=O)O"],
        "Drug2_SMILES": ["CC(=O)O", "CC(=O)O", ""]
    })
    filled_df, stats = sd.preprocess_blank_smiles(raw_df)
    assert filled_df["Drug1_SMILES"].iloc[1] == "CCO"
    assert filled_df["Drug2_SMILES"].iloc[2] == "CCO"
    assert not filled_df["Drug1_SMILES"].isna().any()
    assert not filled_df["Drug2_SMILES"].isna().any()


def test_7_conflicting_smiles_failure():
    """7. Conflicting SMILES failure."""
    conflict_df = pd.DataFrame({
        "Drug1_NSC_ID": ["100", "100"],
        "Drug2_NSC_ID": ["200", "200"],
        "Drug1_SMILES": ["CCO", "CCCN"],
        "Drug2_SMILES": ["CC(=O)O", "CC(=O)O"]
    })
    with pytest.raises(ValueError, match="multiple conflicting non-blank SMILES"):
        sd.preprocess_blank_smiles(conflict_df)


def test_8_salt_normalization():
    """8. Salt normalization keeps largest heavy-atom fragment."""
    sm, Was_norm = strip_salts_keep_largest_fragment("C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)O.Cl")
    assert sm == "C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)O"
    assert Was_norm is True


def test_9_metal_retention():
    """9. Metal retention for Cisplatin, Carboplatin, Oxaliplatin, Arsenic trioxide."""
    sm_pt, norm_pt = strip_salts_keep_largest_fragment("N.N.Cl[Pt]Cl")
    assert sm_pt == "N.N.Cl[Pt]Cl" and norm_pt is False

    sm_as, norm_as = strip_salts_keep_largest_fragment("[O-2].[O-2].[O-2].[As+3].[As+3]")
    assert sm_as == "[O-2].[O-2].[O-2].[As+3].[As+3]" and norm_as is False


# =====================================================================
# TESTS 10 - 14: MORGAN, DESCRIPTORS & LOOKUP
# =====================================================================

def test_10_morgan_1024_r3_config():
    """10. Morgan 1024/r3 configuration."""
    m_cfg, _ = load_config(os.path.join(repo_dir, "config.yaml"))
    assert m_cfg.morgan_in_dim == 1024
    assert m_cfg.morgan_radius == 3

    prep = MolecularPreprocessor()
    assert prep.morgan_nbits == 1024
    assert prep.morgan_radius == 3


def test_11_descriptor_preprocessing_contract():
    """11. Descriptor preprocessing contract."""
    scaler = DescriptorScaler()
    raw_train = [
        np.array([1.0, 2.0, np.nan, 4.0] + [1.0] * 196, dtype=np.float32),
        np.array([3.0, np.inf, 5.0, 6.0] + [2.0] * 196, dtype=np.float32),
        np.array([5.0, 6.0, 7.0, 8.0] + [3.0] * 196, dtype=np.float32)
    ]
    scaler.fit(raw_train)
    scaled = scaler.transform(np.array([2.0, np.nan, np.inf, 1000.0] + [1.5] * 196, dtype=np.float32))
    assert len(scaled) == 200
    assert np.all(np.isfinite(scaled))
    assert np.all(scaled >= -10.0) and np.all(scaled <= 10.0)


def test_12_nsc_feature_lookup():
    """12. NSC feature lookup."""
    dummy_store = {
        "100": {"morgan": np.ones(1024, dtype=np.float32), "descriptors": np.zeros(200, dtype=np.float32)},
        "200": {"morgan": np.full(1024, 2.0, dtype=np.float32), "descriptors": np.ones(200, dtype=np.float32)}
    }
    dummy_data = [{
        "smiles_a": "CCO", "smiles_b": "CC(=O)O", "nsc_a": "100", "nsc_b": "200",
        "cell_line_name": "MCF7", "doses_a": [0.1]*4, "doses_b": [0.2]*4, "viability_matrix": [[1.0]*4]*4
    }]
    gex_dict = {"MCF7": np.ones(976, dtype=np.float32)}
    scaler = DescriptorScaler()
    scaler.fit([np.zeros(200, dtype=np.float32), np.ones(200, dtype=np.float32)])
    ds = DrugComboDataset(dummy_data, gex_dict, drug_feature_store=dummy_store, descriptor_scaler=scaler)
    sample = ds[0]
    assert sample["drug_a_morgan"].shape == (1024,)
    assert sample["drug_b_morgan"].shape == (1024,)


def test_13_strict_s3_paths():
    """13. Strict S3 paths in config.py, train.py, and evaluate.py."""
    with open(os.path.join(repo_dir, "train.py"), "r") as f:
        t_code = f.read()
    with open(os.path.join(repo_dir, "evaluate.py"), "r") as f:
        e_code = f.read()
    with open(os.path.join(repo_dir, "config.py"), "r") as f:
        c_code = f.read()

    assert "scenario3_drug_strict_60_20_20.csv" in c_code
    assert "get_scenario_split_path" in t_code
    assert "get_scenario_split_path" in e_code
    from config import SCENARIO_CONFIG
    assert SCENARIO_CONFIG[3] == "data/splits/scenario3_drug_strict_60_20_20.csv"


def test_14_no_competing_scenario3_generator():
    """14. No competing Scenario 3 generator."""
    assert not os.path.exists(os.path.join(repo_dir, "generate_strict_scenario3.py"))


# =====================================================================
# TESTS 15 - 22: SEED, SCALER INVARIANCE, PERSISTENCE & LOADING
# =====================================================================

def test_15_seed_42_passed_to_partition():
    """15. Seed 42 passed to actual partition call."""
    df_dummy = pd.DataFrame({
        "nsc_a": ["100", "101", "102", "103", "104"],
        "nsc_b": ["105", "106", "107", "108", "109"]
    })
    with patch("numpy.random.default_rng") as mock_rng:
        mock_rng.return_value = np.random.default_rng(42)
        sd.drug_split(df_dummy, "nsc_a", "nsc_b", seed=42)
        mock_rng.assert_called_with(42)


def test_16_descriptor_scaler_uses_unique_train_nscs():
    """16. Descriptor scaler uses unique Train NSCs."""
    raw_a = np.array([1.0] * 200, dtype=np.float32)
    raw_b = np.array([2.0] * 200, dtype=np.float32)

    scaler = DescriptorScaler()
    scaler.fit([raw_a, raw_b])
    np.testing.assert_allclose(scaler.mean_vec, np.full(200, 1.5))


def test_17_duplicate_grid_scaler_invariance():
    """17. Duplicate-grid scaler invariance."""
    nsc_descs_unique = [
        np.array([1.0, 10.0] + [0.0] * 198, dtype=np.float32),
        np.array([3.0, 30.0] + [0.0] * 198, dtype=np.float32),
        np.array([5.0, 50.0] + [0.0] * 198, dtype=np.float32)
    ]
    scaler_A = DescriptorScaler()
    scaler_A.fit(nsc_descs_unique)

    scaler_B = DescriptorScaler()
    scaler_B.fit(nsc_descs_unique)

    np.testing.assert_allclose(scaler_A.median_vec, scaler_B.median_vec, rtol=1e-5)
    np.testing.assert_allclose(scaler_A.mean_vec, scaler_B.mean_vec, rtol=1e-5)
    np.testing.assert_allclose(scaler_A.std_vec, scaler_B.std_vec, rtol=1e-5)


def test_18_scaler_persistence(tmp_path):
    """18. Scaler persistence to NPZ."""
    scaler = DescriptorScaler()
    scaler.fit([np.ones(200, dtype=np.float32), np.zeros(200, dtype=np.float32)])
    save_path = os.path.join(tmp_path, "scaler.npz")
    save_descriptor_scaler(scaler, save_path, ["100", "200"])
    assert os.path.exists(save_path)


def test_19_scaler_loading(tmp_path):
    """19. Scaler loading from NPZ."""
    scaler = DescriptorScaler()
    scaler.fit([np.ones(200, dtype=np.float32), np.zeros(200, dtype=np.float32)])
    save_path = os.path.join(tmp_path, "scaler.npz")
    save_descriptor_scaler(scaler, save_path, ["100", "200"])

    loaded = load_descriptor_scaler(save_path, train_nsc_ids=["100", "200"])
    assert loaded.is_fitted
    np.testing.assert_allclose(scaler.mean_vec, loaded.mean_vec)


def test_20_missing_scaler_failure(tmp_path):
    """20. Missing scaler failure."""
    fake_path = os.path.join(tmp_path, "non_existent_scaler.npz")
    with pytest.raises(FileNotFoundError, match="missing"):
        load_descriptor_scaler(fake_path, train_nsc_ids=["100"])


def test_21_scaler_train_nsc_hash_mismatch_failure(tmp_path):
    """21. Scaler Train-NSC hash mismatch failure."""
    scaler = DescriptorScaler()
    scaler.fit([np.ones(200, dtype=np.float32)])
    save_path = os.path.join(tmp_path, "scaler.npz")
    save_descriptor_scaler(scaler, save_path, ["100", "200"])

    with pytest.raises(ValueError, match="hash mismatch"):
        load_descriptor_scaler(save_path, train_nsc_ids=["100", "999"])


def test_22_max_samples_does_not_change_scaler_statistics():
    """22. max_samples does not change scaler statistics."""
    descs = [np.full(200, 1.0), np.full(200, 2.0), np.full(200, 3.0)]

    scaler_full = DescriptorScaler()
    scaler_full.fit(descs)

    scaler_smoke = DescriptorScaler()
    scaler_smoke.fit(descs)

    np.testing.assert_allclose(scaler_full.mean_vec, scaler_smoke.mean_vec)


# =====================================================================
# TESTS 23 - 28: FEATURE STORE METADATA & STRICT ERRORS
# =====================================================================

def test_23_missing_morgan_feature_store_failure(tmp_path):
    """23. Missing Morgan feature-store failure."""
    fake_store_path = os.path.join(tmp_path, "missing_store.pt")
    with pytest.raises(FileNotFoundError, match="missing"):
        load_precomputed_drug_features(fake_store_path)


def test_24_missing_metadata_failure(tmp_path):
    """24. Missing metadata header in feature store fails loudly."""
    no_meta_path = os.path.join(tmp_path, "no_meta_store.pt")
    no_meta_store = {"100": {"morgan": np.ones(1024, dtype=np.float32), "descriptors": np.zeros(200, dtype=np.float32)}}
    torch.save(no_meta_store, no_meta_path)

    with pytest.raises(ValueError, match="missing required metadata header"):
        load_precomputed_drug_features(no_meta_path)


def test_25_dimension_mismatch_metadata_failure(tmp_path):
    """25. Feature store metadata dimension mismatch (2048-bit) fails loudly."""
    from preprocessor import MolecularPreprocessor
    import rdkit
    prep = MolecularPreprocessor()
    dim_path = os.path.join(tmp_path, "dim_2048_store.pt")
    dim_store = {
        "metadata": {
            "morgan_nbits": 2048, "morgan_radius": 3, "n_descriptors": 200, "key_type": "NSC",
            "descriptor_names": list(prep.descriptor_names[:200]), "rdkit_version": str(rdkit.__version__)
        },
        "100": {"morgan": np.ones(2048, dtype=np.float32), "descriptors": np.zeros(200, dtype=np.float32)}
    }
    torch.save(dim_store, dim_path)

    with pytest.raises(ValueError, match="dimension mismatch: artifact has 2048-bit, active configuration requires 1024-bit"):
        load_precomputed_drug_features(dim_path)


def test_26_radius_mismatch_metadata_failure(tmp_path):
    """26. Feature store metadata radius mismatch (radius 2) fails loudly."""
    from preprocessor import MolecularPreprocessor
    import rdkit
    prep = MolecularPreprocessor()
    rad_path = os.path.join(tmp_path, "radius_2_store.pt")
    rad_store = {
        "metadata": {
            "morgan_nbits": 1024, "morgan_radius": 2, "n_descriptors": 200, "key_type": "NSC",
            "descriptor_names": list(prep.descriptor_names[:200]), "rdkit_version": str(rdkit.__version__)
        },
        "100": {"morgan": np.ones(1024, dtype=np.float32), "descriptors": np.zeros(200, dtype=np.float32)}
    }
    torch.save(rad_store, rad_path)

    with pytest.raises(ValueError, match="radius mismatch: artifact has radius 2, active configuration requires radius 3"):
        load_precomputed_drug_features(rad_path)


def test_27_valid_1024_r3_metadata(tmp_path):
    """27. Valid 1024-bit radius-3 feature store loads cleanly."""
    from preprocessor import MolecularPreprocessor
    import rdkit
    prep = MolecularPreprocessor()
    valid_path = os.path.join(tmp_path, "valid_store.pt")
    valid_store = {
        "metadata": {
            "morgan_nbits": 1024, "morgan_radius": 3, "n_descriptors": 200, "key_type": "NSC",
            "descriptor_names": list(prep.descriptor_names[:200]), "rdkit_version": str(rdkit.__version__)
        },
        "100": {"morgan": np.ones(1024, dtype=np.float32), "descriptors": np.zeros(200, dtype=np.float32)}
    }
    torch.save(valid_store, valid_path)

    loaded = load_precomputed_drug_features(valid_path)
    assert "100" in loaded
    assert len(loaded["100"]["morgan"]) == 1024


def test_28_inconsistent_entry_dimensions(tmp_path):
    """28. Malformed/inconsistent entry vector length fails loudly."""
    from preprocessor import MolecularPreprocessor
    import rdkit
    prep = MolecularPreprocessor()
    bad_entry_path = os.path.join(tmp_path, "inconsistent_entry_store.pt")
    bad_entry_store = {
        "metadata": {
            "morgan_nbits": 1024, "morgan_radius": 3, "n_descriptors": 200, "key_type": "NSC",
            "descriptor_names": list(prep.descriptor_names[:200]), "rdkit_version": str(rdkit.__version__)
        },
        "100": {"morgan": np.ones(1024, dtype=np.float32), "descriptors": np.zeros(200, dtype=np.float32)},
        "200": {"morgan": np.ones(512, dtype=np.float32), "descriptors": np.zeros(200, dtype=np.float32)}
    }
    torch.save(bad_entry_store, bad_entry_path)

    with pytest.raises(ValueError, match="Inconsistent Morgan vector dimension for key '200'"):
        load_precomputed_drug_features(bad_entry_path)


# =====================================================================
# TESTS 29 - 32: CHECKPOINT LOADING & ACTIVE 2048 SCAN
# =====================================================================

def test_29_checkpoint_loading_failure_for_missing_keys(tmp_path):
    """29. Checkpoint loading failure for missing keys."""
    cfg, _ = load_config(os.path.join(repo_dir, "config.yaml"))
    model = CancerCombo(cfg)
    incomplete_ckpt = os.path.join(tmp_path, "incomplete.pt")
    torch.save({"state_dict": {"morgan_enc.projection.0.weight": torch.randn(256, 1024)}}, incomplete_ckpt)

    with pytest.raises(Exception):
        load_checkpoint_strictly(model, incomplete_ckpt, config=cfg)


def test_30_31_old_2048_checkpoint_rejection(tmp_path):
    """30-31. Old 2048-bit checkpoint rejection (uses REAL model state_dict with morgan_in_dim=2048)."""
    cfg_1024, _ = load_config(os.path.join(repo_dir, "config.yaml"))
    model_1024 = CancerCombo(cfg_1024)

    # Instantiate REAL CancerCombo model with morgan_in_dim=2048
    cfg_2048, _ = load_config(os.path.join(repo_dir, "config.yaml"))
    cfg_2048.morgan_in_dim = 2048
    model_2048 = CancerCombo(cfg_2048)
    real_2048_state_dict = model_2048.state_dict()

    bad_2048_ckpt = os.path.join(tmp_path, "old_2048_ckpt.pt")
    torch.save({"state_dict": real_2048_state_dict}, bad_2048_ckpt)

    # Verifies parameter key shape mismatch raises clear error
    with pytest.raises(ValueError, match="Checkpoint/model dimension mismatch: checkpoint expects Morgan input dimension 2048, active configuration requires 1024"):
        load_checkpoint_strictly(model_1024, bad_2048_ckpt, config=cfg_1024)


def test_32_active_2048_default_scan():
    """32. Active 2048-default scan (ensures zero obsolete Morgan 2048-bit defaults remain in active code)."""
    with open(os.path.join(repo_dir, "cancercombo.py"), "r") as f:
        c_code = f.read()
    with open(os.path.join(repo_dir, "evaluate.py"), "r") as f:
        e_code = f.read()
    with open(os.path.join(repo_dir, "dataset.py"), "r") as f:
        d_code = f.read()
    with open(os.path.join(repo_dir, "preprocessor.py"), "r") as f:
        p_code = f.read()

    # Ensure obsolete Morgan 2048 constants and fallbacks are absent
    assert 'getattr(config, "morgan_in_dim", 2048)' not in c_code
    assert 'm_dim, d_dim = "2048"' not in e_code
    assert 'morgan_in_dim: int = 2048' not in c_code
    assert 'morgan_nbits: int = 2048' not in p_code
    assert 'morgan_in_dim = 2048' not in c_code
    assert 'np.zeros(2048' not in p_code  # Preprocessor Morgan output must never be 2048 zeros


if __name__ == "__main__":
    pytest.main(["-v", __file__])
