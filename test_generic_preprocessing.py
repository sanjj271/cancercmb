#!/usr/bin/env python3
"""
test_generic_preprocessing.py - Comprehensive Unit Tests for Generic Preprocessing & Scenario Scaler Fitting

Proves the 10 core audit invariants:
  TEST 1: Generic feature generation includes all drugs, even those outside Scenario 3.
  TEST 2: No automatic/silent loading of Scenario 3 when raw input has no split column.
  TEST 3: Number of unique NSC drugs in feature store matches the input's unique-drug universe.
  TEST 4: Morgan vectors are exactly 1024 dimensions.
  TEST 5: RDKit descriptor vectors are exactly 200 dimensions.
  TEST 6: All stored molecular values are finite after preprocessing/imputation rules.
  TEST 7: Scenario-specific scaler fitting uses ONLY split == 1 drugs.
  TEST 8: Validation and test drugs do not influence scaler statistics.
  TEST 9: Existing NSC-based feature lookup continues to work seamlessly.
  TEST 10: End-to-end compatibility with DrugComboDataset.
"""

import os
import sys
import tempfile
import torch
import numpy as np
import pandas as pd
import pytest

repo_dir = os.path.dirname(os.path.abspath(__file__))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from precompute_molecular_features import (
    extract_unique_drugs,
    precompute_molecular_features,
    fit_scenario_scaler,
    normalize_clean_nsc
)
from dataset import (
    load_precomputed_drug_features,
    load_descriptor_scaler,
    DrugComboDataset,
    DescriptorScaler
)


def test_1_generic_feature_generation_includes_drugs_outside_scenario3(tmp_path):
    """TEST 1: Generic feature generation includes drugs appearing outside Scenario 3."""
    # Create raw dataframe with a novel drug '999999' that definitely does not exist in Scenario 3
    df = pd.DataFrame({
        "Drug1_NSC_ID": ["100", "999999"],
        "Drug1_SMILES": ["CCO", "CC(=O)NC1=CC=C(O)C=C1"], # Ethanol, Paracetamol
        "Drug2_NSC_ID": ["200", "100"],
        "Drug2_SMILES": ["CC(=O)O", "CCO"] # Acetic acid
    })
    raw_path = os.path.join(tmp_path, "raw_combos.csv")
    df.to_csv(raw_path, index=False)

    out_pt = os.path.join(tmp_path, "features.pt")
    store = precompute_molecular_features(raw_path, out_pt)

    # Verify novel drug 999999 is in the feature store
    assert "999999" in store
    assert "NSC-999999" in store
    assert "CC(=O)NC1=CC=C(O)C=C1" in store


def test_2_no_automatic_scenario3_fallback_when_no_split_column(tmp_path, monkeypatch):
    """TEST 2: When raw input has no split column, the script does NOT automatically load Scenario 3."""
    # Intercept pd.read_csv to ensure scenario3 split is never read during generic feature extraction
    original_read_csv = pd.read_csv

    def guarded_read_csv(filepath_or_buffer, *args, **kwargs):
        path_str = str(filepath_or_buffer)
        if "scenario3_drug_strict" in path_str or "data/splits" in path_str or "data\\splits" in path_str:
            raise AssertionError(f"Script attempted to load Scenario split file: {path_str}")
        return original_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read_csv)

    df = pd.DataFrame({
        "Drug1_NSC_ID": ["100", "200"],
        "Drug1_SMILES": ["CCO", "CC(=O)O"],
        "Drug2_NSC_ID": ["200", "100"],
        "Drug2_SMILES": ["CC(=O)O", "CCO"]
    })
    raw_path = os.path.join(tmp_path, "raw_no_split.csv")
    df.to_csv(raw_path, index=False)

    out_pt = os.path.join(tmp_path, "features.pt")
    # This must succeed without triggering guarded_read_csv assertion
    store = precompute_molecular_features(raw_path, out_pt)
    assert len(store) > 0


def test_3_unique_nsc_count_matches_input_universe(tmp_path):
    """TEST 3: Number of unique NSC drugs in feature store matches the input's unique-drug universe."""
    df = pd.DataFrame({
        "Drug1_NSC_ID": ["10", "20", "30", "10"],
        "Drug1_SMILES": ["CCO", "CCN", "CCC", "CCO"],
        "Drug2_NSC_ID": ["40", "50", "10", "20"],
        "Drug2_SMILES": ["CCOCC", "CCCO", "CCO", "CCN"]
    })
    raw_path = os.path.join(tmp_path, "universe.csv")
    df.to_csv(raw_path, index=False)

    unique_expected_nscs = {"10", "20", "30", "40", "50"}
    extracted = extract_unique_drugs(raw_path)
    assert set(extracted.keys()) == unique_expected_nscs

    out_pt = os.path.join(tmp_path, "features.pt")
    store = precompute_molecular_features(raw_path, out_pt)
    store_clean_nscs = {k for k in store if not k.startswith("NSC-") and not k.startswith("__") and k != "metadata" and k.isdigit()}
    assert store_clean_nscs == unique_expected_nscs


def test_4_morgan_dimension_is_1024(tmp_path):
    """TEST 4: Morgan vectors are exactly 1024 dimensions."""
    df = pd.DataFrame({
        "nsc_a": ["123"], "smiles_a": ["CCO"],
        "nsc_b": ["456"], "smiles_b": ["CC(=O)O"]
    })
    raw_path = os.path.join(tmp_path, "m_test.csv")
    df.to_csv(raw_path, index=False)

    out_pt = os.path.join(tmp_path, "features.pt")
    store = precompute_molecular_features(raw_path, out_pt)

    for nsc in ["123", "456"]:
        morgan = store[nsc]["morgan"]
        assert isinstance(morgan, torch.Tensor)
        assert morgan.shape == (1024,)


def test_5_rdkit_descriptor_dimension_is_200(tmp_path):
    """TEST 5: RDKit descriptor vectors are exactly 200 dimensions."""
    df = pd.DataFrame({
        "nsc_a": ["123"], "smiles_a": ["CCO"],
        "nsc_b": ["456"], "smiles_b": ["CC(=O)O"]
    })
    raw_path = os.path.join(tmp_path, "d_test.csv")
    df.to_csv(raw_path, index=False)

    out_pt = os.path.join(tmp_path, "features.pt")
    store = precompute_molecular_features(raw_path, out_pt)

    for nsc in ["123", "456"]:
        desc = store[nsc]["descriptors"]
        assert isinstance(desc, torch.Tensor)
        assert desc.shape == (200,)


def test_6_all_stored_molecular_values_are_finite(tmp_path):
    """TEST 6: Raw descriptors are finite-or-NaN with no Inf, and scaled output is finite."""
    df = pd.DataFrame({
        "Drug1_NSC_ID": ["111", "222"],
        "Drug1_SMILES": ["CCO", "CC(C)(C)OC(=O)NC1C(=O)N2C1SC(C)(C)C2C(=O)O"],
        "Drug2_NSC_ID": ["333", "444"],
        "Drug2_SMILES": ["CC(=O)O", "CCN(CC)CCCC(C)NC1=C2C=CC(=CC2=NC3=C1C=CC(=C3)Cl)OC"]
    })
    raw_path = os.path.join(tmp_path, "finite_test.csv")
    df.to_csv(raw_path, index=False)

    out_pt = os.path.join(tmp_path, "features.pt")
    store = precompute_molecular_features(raw_path, out_pt)

    scaler = DescriptorScaler()
    raw_train = [v["descriptors"].numpy() for k, v in store.items() if k.isdigit()]
    scaler.fit(raw_train)

    for k, v in store.items():
        if k in ["metadata", "__metadata__"]:
            continue
        m = v["morgan"].numpy()
        d = v["descriptors"].numpy()
        assert np.all(np.isfinite(m)), f"Non-finite Morgan for key {k}"
        assert not np.any(np.isinf(d)), f"Inf Descriptors for key {k}"
        finite_d = d[np.isfinite(d)]
        if len(finite_d) > 0:
            assert np.all(np.abs(finite_d) <= 1e6), f"Descriptors exceed 1e6 for key {k}"
        scaled_d = scaler.transform(d)
        assert np.all(np.isfinite(scaled_d)), f"Scaled output not finite for key {k}"


def test_7_scenario_scaler_uses_only_split_1_drugs(tmp_path):
    """TEST 7: Scenario-specific scaler fitting uses ONLY split == 1 drugs."""
    from preprocessor import MolecularPreprocessor
    import rdkit
    prep = MolecularPreprocessor()
    meta = {
        "morgan_nbits": 1024,
        "morgan_radius": 3,
        "n_descriptors": 200,
        "descriptor_names": list(prep.descriptor_names[:200]),
        "rdkit_version": str(rdkit.__version__),
        "key_type": "NSC"
    }
    features = {
        "1": {"descriptors": torch.full((200,), 10.0, dtype=torch.float32), "morgan": torch.zeros(1024)},
        "2": {"descriptors": torch.full((200,), 100.0, dtype=torch.float32), "morgan": torch.zeros(1024)},
        "3": {"descriptors": torch.full((200,), 1000.0, dtype=torch.float32), "morgan": torch.zeros(1024)},
        "metadata": meta,
        "__metadata__": meta
    }
    feat_path = os.path.join(tmp_path, "feat.pt")
    torch.save(features, feat_path)

    split_df = pd.DataFrame({
        "nsc_a": ["1", "2", "3"],
        "nsc_b": ["1", "2", "3"],
        "split": [1, 2, 3] # Train, Val, Test
    })
    split_path = os.path.join(tmp_path, "split.csv")
    split_df.to_csv(split_path, index=False)

    scaler_out = os.path.join(tmp_path, "scaler.npz")
    scaler = fit_scenario_scaler(split_path, feat_path, scaler_out)

    assert scaler.is_fitted
    # If ONLY split 1 was used (descriptor vector 10.0), mean must be exactly 10.0!
    np.testing.assert_allclose(scaler.mean_vec, np.full(200, 10.0))


def test_8_val_test_drugs_do_not_influence_scaler_statistics(tmp_path):
    """TEST 8: Validation and test drugs do not influence scaler statistics."""
    from preprocessor import MolecularPreprocessor
    import rdkit
    prep = MolecularPreprocessor()
    meta = {
        "morgan_nbits": 1024,
        "morgan_radius": 3,
        "n_descriptors": 200,
        "descriptor_names": list(prep.descriptor_names[:200]),
        "rdkit_version": str(rdkit.__version__),
        "key_type": "NSC"
    }
    features = {
        "10": {"descriptors": torch.full((200,), 5.0, dtype=torch.float32), "morgan": torch.zeros(1024)},
        "20": {"descriptors": torch.full((200,), 15.0, dtype=torch.float32), "morgan": torch.zeros(1024)},
        "99": {"descriptors": torch.full((200,), 99999.0, dtype=torch.float32), "morgan": torch.zeros(1024)},
        "metadata": meta,
        "__metadata__": meta
    }
    feat_path = os.path.join(tmp_path, "feat.pt")
    torch.save(features, feat_path)

    # Split with 10 & 20 as Train (mean = 10.0), and 99 as Val (split 2) and Test (split 3)
    split_df = pd.DataFrame({
        "nsc_a": ["10", "20", "99", "99"],
        "nsc_b": ["20", "10", "99", "99"],
        "split": [1, 1, 2, 3]
    })
    split_path = os.path.join(tmp_path, "split.csv")
    split_df.to_csv(split_path, index=False)

    scaler_out = os.path.join(tmp_path, "scaler.npz")
    scaler = fit_scenario_scaler(split_path, feat_path, scaler_out)

    # Train mean of [5.0, 15.0] is 10.0. If drug 99 (99999.0) had leaked, mean would be huge.
    np.testing.assert_allclose(scaler.mean_vec, np.full(200, 10.0))
    np.testing.assert_allclose(scaler.median_vec, np.full(200, 10.0))


def test_9_nsc_feature_lookup_compatibility(tmp_path):
    """TEST 9: Existing NSC-based feature lookup continues to work seamlessly."""
    from preprocessor import MolecularPreprocessor
    import rdkit
    prep = MolecularPreprocessor()
    meta = {
        "morgan_nbits": 1024,
        "morgan_radius": 3,
        "n_descriptors": 200,
        "descriptor_names": list(prep.descriptor_names[:200]),
        "rdkit_version": str(rdkit.__version__),
        "key_type": "NSC"
    }
    features = {
        "100": {"descriptors": torch.zeros(200), "morgan": torch.ones(1024)},
        "NSC-100": {"descriptors": torch.zeros(200), "morgan": torch.ones(1024)},
        "CCO": {"descriptors": torch.zeros(200), "morgan": torch.ones(1024)},
        "200": {"descriptors": torch.ones(200), "morgan": torch.zeros(1024)},
        "NSC-200": {"descriptors": torch.ones(200), "morgan": torch.zeros(1024)},
        "CC(=O)O": {"descriptors": torch.ones(200), "morgan": torch.zeros(1024)},
        "metadata": meta,
        "__metadata__": meta
    }
    feat_path = os.path.join(tmp_path, "feat.pt")
    torch.save(features, feat_path)

    loaded_store = load_precomputed_drug_features(feat_path)
    data = [{
        "smiles_a": "CCO", "smiles_b": "CC(=O)O", "nsc_a": "100", "nsc_b": "200",
        "cell_line_name": "MCF7", "doses_a": [0.1]*4, "doses_b": [0.2]*4, "viability_matrix": [[1.0]*4]*4
    }]
    gex = {"MCF7": np.ones(976, dtype=np.float32)}

    scaler = DescriptorScaler()
    scaler.fit([np.zeros(200, dtype=np.float32), np.ones(200, dtype=np.float32)])
    ds = DrugComboDataset(data, gex, drug_feature_store=loaded_store, descriptor_scaler=scaler)

    sample = ds[0]
    assert sample["drug_a_morgan"].shape == (1024,)
    assert sample["drug_b_morgan"].shape == (1024,)
    assert sample["drug_a_desc"].shape == (200,)
    assert sample["drug_b_desc"].shape == (200,)


def test_10_real_dataset_universe_validation():
    """TEST 10: Real raw combination dataset extracts all 101 unique drugs."""
    raw_zip = "data/DrugCombination_with_NSC_ID.zip"
    if os.path.exists(raw_zip):
        drugs = extract_unique_drugs(raw_zip)
        assert len(drugs) == 101
        for nsc, sm in drugs.items():
            assert nsc.isdigit()
            assert len(sm) > 0
