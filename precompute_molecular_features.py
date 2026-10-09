#!/usr/bin/env python3
"""
precompute_molecular_features.py - DGX-Reproducible Generic Molecular Preprocessing & Scaler Fitting Pipeline

Decoupled Architecture:
  Stage 1 (Generic Feature Extraction):
    Extracts all unique drugs from the root combination dataset (e.g. data/DrugCombination_with_NSC_ID.zip).
    Computes 1024-bit Morgan fingerprints (radius 3) and 200 continuous RDKit descriptors.
    Saves a reusable, unscaled molecular feature store (.pt and .pkl).
    Completely scenario-independent (no reliance on or automatic loading of any scenario split).

  Stage 2 (Scenario-Specific Scaler Fitting):
    Loads a specific scenario split (e.g. Scenario 1, 2, or 3).
    Extracts ONLY unique training drugs (split == 1).
    Fits DescriptorScaler strictly on training descriptors to prevent leakage.
    Saves fitted scaler with SHA256 training NSC hash to .npz.

Usage:
  # Stage 1: Generic all-drug feature store extraction
  python precompute_molecular_features.py \
      --input_csv data/DrugCombination_with_NSC_ID.zip \
      --output_file data/features/morgan1024_r3_raw/drug_features_morgan1024_r3_raw.pt

  # Stage 2: Scenario-specific scaler fitting (e.g. for Scenario 3 benchmark)
  python fit_descriptor_scaler.py --scenario 3
  # or via precompute_molecular_features.py:
  python precompute_molecular_features.py \
      --mode fit_scaler \
      --split_csv data/splits/scenario3_drug_strict_60_20_20.csv \
      --feature_file data/features/morgan1024_r3_raw/drug_features_morgan1024_r3_raw.pt \
      --scaler_output checkpoints/drugcell_crossattention/descriptor_scaler_s3.npz
"""

import os
for _k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "1"

from helpers import enforce_single_thread
enforce_single_thread()

import sys
import re
import argparse
import logging
import zipfile
import pickle
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Union, Optional

import rdkit
from config import RAW_FEATURE_STORE_PATH
from preprocessor import MolecularPreprocessor
from dataset import DescriptorScaler, save_descriptor_scaler, compute_nsc_hash, load_precomputed_drug_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FeaturePrecomputer")


def normalize_clean_nsc(nsc: Any) -> str:
    """Normalize raw NSC string to clean integer digits string."""
    raw = str(nsc).strip()
    return re.sub(r'^(NSC[-_]?)', '', raw, flags=re.IGNORECASE).strip()


def extract_unique_drugs(input_path: str) -> Dict[str, str]:
    """Extract mapping of NSC ID -> SMILES string for all unique drugs in the dataset.
    
    Operates strictly on the provided input dataset (CSV or ZIP).
    Does NOT depend on, search for, or load any scenario split file.

    Args:
        input_path: Path to CSV or ZIP archive.

    Returns:
        Dict[str, str]: Dictionary mapping clean NSC ID to SMILES string.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: '{input_path}'")
        
    if input_path.endswith(".zip"):
        with zipfile.ZipFile(input_path, "r") as z:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_files:
                raise ValueError("No CSV found in ZIP archive.")
            dfs = []
            for csv_file in csv_files:
                with z.open(csv_file) as f:
                    dfs.append(pd.read_csv(f))
            df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(input_path)

    nsc_map: Dict[str, str] = {}

    # 1. Authoritative schema: nsc_a, smiles_a, nsc_b, smiles_b
    if "nsc_a" in df.columns and "smiles_a" in df.columns:
        sub_a = df[["nsc_a", "smiles_a"]].dropna()
        for nsc, sm in zip(sub_a["nsc_a"], sub_a["smiles_a"]):
            clean_k = normalize_clean_nsc(nsc)
            v = str(sm).strip()
            if clean_k and v:
                nsc_map[clean_k] = v

    if "nsc_b" in df.columns and "smiles_b" in df.columns:
        sub_b = df[["nsc_b", "smiles_b"]].dropna()
        for nsc, sm in zip(sub_b["nsc_b"], sub_b["smiles_b"]):
            clean_k = normalize_clean_nsc(nsc)
            v = str(sm).strip()
            if clean_k and v:
                nsc_map[clean_k] = v

    # 2. Raw dataset schema: Drug1_NSC_ID, Drug1_SMILES, Drug2_NSC_ID, Drug2_SMILES
    if "Drug1_NSC_ID" in df.columns and "Drug1_SMILES" in df.columns:
        sub_1 = df[["Drug1_NSC_ID", "Drug1_SMILES"]].dropna()
        for nsc, sm in zip(sub_1["Drug1_NSC_ID"], sub_1["Drug1_SMILES"]):
            clean_k = normalize_clean_nsc(nsc)
            v = str(sm).strip()
            if clean_k and v:
                nsc_map[clean_k] = v

    if "Drug2_NSC_ID" in df.columns and "Drug2_SMILES" in df.columns:
        sub_2 = df[["Drug2_NSC_ID", "Drug2_SMILES"]].dropna()
        for nsc, sm in zip(sub_2["Drug2_NSC_ID"], sub_2["Drug2_SMILES"]):
            clean_k = normalize_clean_nsc(nsc)
            v = str(sm).strip()
            if clean_k and v:
                nsc_map[clean_k] = v

    # 3. Generic pair schema: drug1, smiles1, drug2, smiles2
    if "drug1" in df.columns and "smiles1" in df.columns:
        sub_g1 = df[["drug1", "smiles1"]].dropna()
        for nsc, sm in zip(sub_g1["drug1"], sub_g1["smiles1"]):
            clean_k = normalize_clean_nsc(nsc)
            v = str(sm).strip()
            if clean_k and v:
                nsc_map[clean_k] = v

    if "drug2" in df.columns and "smiles2" in df.columns:
        sub_g2 = df[["drug2", "smiles2"]].dropna()
        for nsc, sm in zip(sub_g2["drug2"], sub_g2["smiles2"]):
            clean_k = normalize_clean_nsc(nsc)
            v = str(sm).strip()
            if clean_k and v:
                nsc_map[clean_k] = v

    logger.info(f"Extracted {len(nsc_map)} unique drugs from '{input_path}'.")
    return nsc_map


def extract_drugs_and_train_nscs(input_path: str, split_csv: Optional[str] = None) -> Tuple[Dict[str, str], List[str]]:
    """Legacy compatibility helper extracting drugs and optional training NSCs if an explicit split_csv is provided.
    
    NOTE: NEVER silently falls back to any scenario split file if split_csv is not provided.
    """
    nsc_map = extract_unique_drugs(input_path)
    train_nsc_set = set()

    split_df = None
    if split_csv and os.path.exists(split_csv):
        split_df = pd.read_csv(split_csv)
    else:
        # Check if the input file itself contains a split column
        try:
            if input_path.endswith(".zip"):
                with zipfile.ZipFile(input_path, "r") as z:
                    dfs = [pd.read_csv(z.open(f)) for f in z.namelist() if f.endswith(".csv")]
                    if dfs and "split" in dfs[0].columns:
                        split_df = pd.concat(dfs, ignore_index=True)
            else:
                raw_head = pd.read_csv(input_path, nrows=5)
                if "split" in raw_head.columns:
                    split_df = pd.read_csv(input_path)
        except Exception:
            pass

    if split_df is not None and "split" in split_df.columns:
        train_rows = split_df[split_df["split"] == 1]
        for col in ["nsc_a", "nsc_b", "Drug1_NSC_ID", "Drug2_NSC_ID", "drug1", "drug2"]:
            if col in train_rows.columns:
                for val in train_rows[col].dropna():
                    clean_k = normalize_clean_nsc(val)
                    if clean_k and clean_k in nsc_map:
                        train_nsc_set.add(clean_k)

    train_nsc_list = sorted(list(train_nsc_set))
    return nsc_map, train_nsc_list


def precompute_molecular_features(
    input_csv: str,
    output_file: str,
    morgan_bits: int = 1024,
    morgan_radius: int = 3,
    n_descriptors: int = 200,
) -> Dict[str, Dict[str, Any]]:
    """Stage 1: Precompute scenario-independent raw molecular feature store for all unique drugs.
    
    Extracts all unique drugs from input_csv, generates 1024-bit Morgan fingerprints (r=3) and
    200 continuous RDKit descriptors with sanitization (+/-Inf -> NaN, finite values clipped to [-1e6, 1e6]).
    
    Contains NO scenario-dependent imputation (no median over drugs, no 0.0 filling of undefined descriptors).
    Undefined descriptor values remain NaN, to be imputed strictly with TRAIN-only statistics by DescriptorScaler.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    nsc_map = extract_unique_drugs(input_csv)
    if not nsc_map:
        raise ValueError(f"No valid drug mappings found in '{input_csv}'.")

    preprocessor = MolecularPreprocessor(morgan_nbits=morgan_bits, morgan_radius=morgan_radius)
    logger.info(f"Precomputing raw Morgan ({morgan_bits}-bit, r={morgan_radius}) and RDKit descriptors for {len(nsc_map)} drugs...")

    # Build finalized raw feature store
    feature_store: Dict[str, Dict[str, torch.Tensor]] = {}
    for idx, (nsc_id, smiles) in enumerate(nsc_map.items()):
        morgan, raw_desc, ok = preprocessor.process_smiles(smiles)
        clean_desc = np.array(raw_desc, dtype=np.float32).copy()

        # 1. Convert +Inf / -Inf to NaN
        clean_desc[np.isinf(clean_desc)] = np.nan

        # 2. Clip finite values to [-1e6, +1e6] (NaNs preserved)
        finite_mask = np.isfinite(clean_desc)
        clean_desc[finite_mask] = np.clip(clean_desc[finite_mask], -1e6, 1e6)

        # 3. Strict validation assertions
        assert not np.any(np.isinf(clean_desc)), f"Inf descriptor value found for NSC {nsc_id}!"
        finite_vals = clean_desc[finite_mask]
        if len(finite_vals) > 0:
            assert np.all(np.abs(finite_vals) <= 1e6), f"Raw descriptor exceeds 1e6 for NSC {nsc_id}!"
        assert len(morgan) == morgan_bits, f"Morgan vector length mismatch for NSC {nsc_id}!"
        assert np.all(np.isfinite(morgan)), f"Non-finite Morgan vector for NSC {nsc_id}!"
        assert np.all((morgan == 0.0) | (morgan == 1.0)), f"Non-binary Morgan vector for NSC {nsc_id}!"
        assert len(clean_desc) == n_descriptors, f"Descriptor length mismatch for NSC {nsc_id}!"

        morgan_t = torch.from_numpy(morgan.astype(np.float32))
        desc_t = torch.from_numpy(clean_desc.astype(np.float32))

        feat_entry = {
            "morgan": morgan_t,
            "descriptors": desc_t
        }

        # Key by clean NSC ID, prefixed NSC-ID, and SMILES
        feature_store[nsc_id] = feat_entry
        feature_store[f"NSC-{nsc_id}"] = feat_entry
        feature_store[smiles] = feat_entry

    metadata = {
        "morgan_nbits": morgan_bits,
        "morgan_radius": morgan_radius,
        "n_descriptors": n_descriptors,
        "descriptor_names": list(preprocessor.descriptor_names[:n_descriptors]),
        "rdkit_version": str(rdkit.__version__),
        "key_type": "NSC"
    }
    feature_store["metadata"] = metadata
    feature_store["__metadata__"] = metadata

    # Save to PyTorch .pt
    torch.save(feature_store, output_file)
    logger.info(f"Successfully saved generic feature store to '{output_file}' ({len(feature_store)} unique keys).")

    # Save to Pickle .pkl
    pkl_file = output_file.replace(".pt", ".pkl")
    with open(pkl_file, "wb") as f:
        pickle.dump(feature_store, f)
    logger.info(f"Successfully saved Pickle feature store to '{pkl_file}'.")

    return feature_store


def fit_scenario_scaler(
    split_csv: Union[str, pd.DataFrame],
    feature_store_or_path: Union[str, Dict[str, Dict[str, Any]]],
    scaler_output: str
) -> DescriptorScaler:
    """Stage 2: Fit DescriptorScaler strictly on training drugs (split == 1) of the specified scenario.
    
    Prevents leakage from validation or test drugs and persists scaler parameters with SHA256 NSC hash.
    Can accept either a split CSV file path or an in-memory split DataFrame.
    """
    os.makedirs(os.path.dirname(os.path.abspath(scaler_output)), exist_ok=True)

    # 1. Load feature store if path provided
    if isinstance(feature_store_or_path, str):
        feature_store = load_precomputed_drug_features(feature_store_or_path)
    else:
        feature_store = feature_store_or_path

    # 2. Load split CSV and identify unique Train NSCs (split == 1) ONLY
    if isinstance(split_csv, pd.DataFrame):
        split_df = split_csv
    elif not os.path.exists(split_csv):
        raise FileNotFoundError(f"Scenario split CSV missing at: '{split_csv}'")
    else:
        split_df = pd.read_csv(split_csv)

    if "split" not in split_df.columns:
        raise ValueError("Split table does not contain required 'split' column!")

    from dataset import extract_scenario_nscs
    train_nsc_list, val_nsc_list, test_nsc_list = extract_scenario_nscs(split_df)
    if len(train_nsc_list) == 0:
        raise ValueError("No training rows (split == 1) found in split table!")

    logger.info(f"Identified {len(train_nsc_list)} unique TRAIN drugs (split == 1) (VAL: {len(val_nsc_list)}, TEST: {len(test_nsc_list)}).")

    # 3. Fit DescriptorScaler strictly on training drugs using explicit NSC mapping
    scaler = DescriptorScaler()
    scaler.fit_from_feature_store(feature_store, train_nsc_list)

    # 4. Persist with SHA256 training NSC hash
    save_descriptor_scaler(scaler, scaler_output, train_nsc_ids=train_nsc_list)
    logger.info(f"Successfully saved scenario-specific DescriptorScaler to '{scaler_output}' (Hash: {compute_nsc_hash(train_nsc_list)[:12]}...).")
    return scaler


def precompute_and_save_pipeline(
    input_csv: str,
    output_file: str,
    scaler_output: Optional[str] = None,
    split_csv: Optional[str] = None,
    morgan_bits: int = 1024,
    morgan_radius: int = 3,
    n_descriptors: int = 200
):
    """Complete pipeline: generates generic feature store, and optionally fits scenario scaler if split_csv provided."""
    feature_store = precompute_molecular_features(
        input_csv=input_csv,
        output_file=output_file,
        morgan_bits=morgan_bits,
        morgan_radius=morgan_radius,
        n_descriptors=n_descriptors
    )

    if scaler_output is not None and split_csv is not None:
        fit_scenario_scaler(
            split_csv=split_csv,
            feature_store_or_path=feature_store,
            scaler_output=scaler_output
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generic Molecular Feature Extraction & Scenario Scaler Fitting for CancerCombo"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="features",
        choices=["features", "fit_scaler", "all"],
        help="Execution mode: 'features' (generic all-drug feature extraction), 'fit_scaler' (scenario-specific scaler fitting), or 'all'."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/DrugCombination_with_NSC_ID.zip",
        help="Path to input raw dataset CSV or ZIP."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=RAW_FEATURE_STORE_PATH,
        help="Path to output .pt feature store."
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        default=None,
        help="Path to scenario split CSV (required for 'fit_scaler' mode)."
    )
    parser.add_argument(
        "--feature_file",
        type=str,
        default=None,
        help="Path to existing feature store for 'fit_scaler' mode (defaults to --output_file)."
    )
    parser.add_argument(
        "--scaler_output",
        type=str,
        default="checkpoints/drugcell_crossattention/descriptor_scaler_s3.npz",
        help="Path to output .npz scaler file."
    )
    parser.add_argument("--morgan_bits", type=int, default=1024, help="Morgan fingerprint bits.")
    parser.add_argument("--morgan_radius", type=int, default=3, help="Morgan radius.")
    parser.add_argument("--n_descriptors", type=int, default=200, help="Number of continuous RDKit descriptors.")
    args = parser.parse_args()

    if args.mode == "features":
        precompute_molecular_features(
            input_csv=args.input_csv,
            output_file=args.output_file,
            morgan_bits=args.morgan_bits,
            morgan_radius=args.morgan_radius,
            n_descriptors=args.n_descriptors,
        )
        if args.split_csv and args.scaler_output:
            fit_scenario_scaler(
                split_csv=args.split_csv,
                feature_store_or_path=args.output_file,
                scaler_output=args.scaler_output
            )

    elif args.mode == "fit_scaler":
        if not args.split_csv:
            raise ValueError("--split_csv is required when running in 'fit_scaler' mode.")
        feat_path = args.feature_file if args.feature_file else args.output_file
        fit_scenario_scaler(
            split_csv=args.split_csv,
            feature_store_or_path=feat_path,
            scaler_output=args.scaler_output
        )

    elif args.mode == "all":
        feat_store = precompute_molecular_features(
            input_csv=args.input_csv,
            output_file=args.output_file,
            morgan_bits=args.morgan_bits,
            morgan_radius=args.morgan_radius,
            n_descriptors=args.n_descriptors,
        )
        if args.split_csv and args.scaler_output:
            fit_scenario_scaler(
                split_csv=args.split_csv,
                feature_store_or_path=feat_store,
                scaler_output=args.scaler_output
            )
        else:
            logger.info("Mode 'all' completed feature extraction. Scaler fitting skipped (no --split_csv provided).")


if __name__ == "__main__":
    main()
