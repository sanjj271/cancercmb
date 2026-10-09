#!/usr/bin/env python3
"""
split_dataset.py - Production-Ready Dataset Splitting Pipeline for CancerCombo

Implements benchmark evaluation protocols across three independent partitioning scenarios:
  1. Scenario 1 — Combination-Wise Split (60/20/20 on canonicalized drug pairs)
  2. Scenario 2 — Cell-Wise Split (60/20/20 on unique cell lines)
  3. Scenario 3 — Drug-Wise Split (60/20/20 on unique drugs with zero drug leakage)

Usage:
    python split_dataset.py --input_csv path/to/dataset.csv --output_dir ./data/splits --seed 42
"""

import os
for _k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "1"

from helpers import enforce_single_thread
enforce_single_thread()

import sys
import json
import logging
import argparse
import zipfile
import numpy as np
import pandas as pd
from typing import Tuple, List, Set, Dict, Any, Optional
from dataset import match_cell_line, load_nci60_gex

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DatasetSplitter")


def detect_columns(
    df: pd.DataFrame,
    drug1_col: Optional[str] = None,
    drug2_col: Optional[str] = None,
    cell_col: Optional[str] = None
) -> Tuple[str, str, str]:
    """Detect or validate required drug1, drug2, and cell column names in the DataFrame.

    Args:
        df: Input pandas DataFrame.
        drug1_col: Explicit drug 1 column name or None.
        drug2_col: Explicit drug 2 column name or None.
        cell_col: Explicit cell line column name or None.

    Returns:
        Tuple[str, str, str]: (resolved_drug1_col, resolved_drug2_col, resolved_cell_col).

    Raises:
        ValueError: If required columns cannot be identified in the DataFrame.
    """
    cols = list(df.columns)
    
    # Resolve Drug 1 Column
    if drug1_col and drug1_col in cols:
        d1 = drug1_col
    else:
        d1_candidates = ["drug1", "Drug1", "smiles_a", "smiles1", "SMILES_A", "Drug1_SMILES", "drug_a", "DrugA"]
        d1 = next((c for c in d1_candidates if c in cols), None)
        
    # Resolve Drug 2 Column
    if drug2_col and drug2_col in cols:
        d2 = drug2_col
    else:
        d2_candidates = ["drug2", "Drug2", "smiles_b", "smiles2", "SMILES_B", "Drug2_SMILES", "drug_b", "DrugB"]
        d2 = next((c for c in d2_candidates if c in cols), None)
        
    # Resolve Cell Line Column
    if cell_col and cell_col in cols:
        c_col = cell_col
    else:
        c_candidates = ["cell", "cell_line_name", "CELL_NAME", "cell_line", "Cell", "cell_name"]
        c_col = next((c for c in c_candidates if c in cols), None)

    if not d1 or not d2 or not c_col:
        raise ValueError(
            f"Failed to auto-detect required columns in DataFrame.\n"
            f"Found columns: {cols}\n"
            f"Resolved: drug1='{d1}', drug2='{d2}', cell='{c_col}'.\n"
            f"Please specify --drug1_col, --drug2_col, and --cell_col explicitly."
        )
        
    logger.info(f"Resolved dataset columns -> Drug1: '{d1}', Drug2: '{d2}', Cell: '{c_col}'")
    return d1, d2, c_col


def get_canonical_combos(df: pd.DataFrame, drug1_col: str, drug2_col: str) -> List[Tuple[str, str]]:
    """Fast vectorized canonicalization of drug pairs to guarantee (DrugA, DrugB) == (DrugB, DrugA).

    Args:
        df: Input DataFrame.
        drug1_col: Drug 1 column name.
        drug2_col: Drug 2 column name.

    Returns:
        List[Tuple[str, str]]: Canonical tuple list for every row.
    """
    d1_vals = df[drug1_col].astype(str).str.strip().values
    d2_vals = df[drug2_col].astype(str).str.strip().values
    mask = d1_vals <= d2_vals
    first = np.where(mask, d1_vals, d2_vals)
    second = np.where(mask, d2_vals, d1_vals)
    return list(zip(first, second))


# =====================================================================
# SCENARIO 1 — COMBINATION-WISE SPLIT
# =====================================================================

def combination_split(
    df: pd.DataFrame,
    drug1_col: str,
    drug2_col: str,
    seed: int = 42
) -> pd.DataFrame:
    """Perform Scenario 1 Combination-Wise Split (60/20/20 on canonicalized drug pairs).

    Guarantees that every sample belonging to a unique drug combination remains
    in the same split.

    Args:
        df: Input DataFrame.
        drug1_col: Column name for drug 1.
        drug2_col: Column name for drug 2.
        seed: Random seed for deterministic shuffling.

    Returns:
        pd.DataFrame: Copy of input DataFrame with added 'split' column (1=Train, 2=Val, 3=Test).
    """
    logger.info("Starting Scenario 1: Combination-Wise Split...")
    out_df = df.copy()
    
    # 1. Vectorized canonical combination keys
    combo_keys = get_canonical_combos(out_df, drug1_col, drug2_col)
    out_df["_combo_key"] = combo_keys
    
    # 2. Extract unique combinations and shuffle deterministically
    unique_combos = sorted(list(set(combo_keys)))
    rng = np.random.default_rng(seed)
    shuffled_combos = rng.permutation(unique_combos)
    
    # Convert back to tuples after numpy permutation
    shuffled_tuples = [tuple(x) for x in shuffled_combos]
    
    # 3. Partition combinations 60% / 20% / 20%
    n_total = len(shuffled_tuples)
    n_train = int(round(n_total * 0.60))
    n_val = int(round(n_total * 0.20))
    
    train_combos = set(shuffled_tuples[:n_train])
    val_combos = set(shuffled_tuples[n_train:n_train + n_val])
    test_combos = set(shuffled_tuples[n_train + n_val:])
    
    # 4. Fast mapping of combination key to split
    split_map = {}
    for c in train_combos: split_map[c] = 1
    for c in val_combos: split_map[c] = 2
    for c in test_combos: split_map[c] = 3
    
    out_df["split"] = out_df["_combo_key"].map(split_map).astype(int)
    out_df.drop(columns=["_combo_key"], inplace=True)
    
    logger.info(
        f"Scenario 1 Split Complete -> Unique Combos: Total={n_total}, "
        f"Train={len(train_combos)}, Val={len(val_combos)}, Test={len(test_combos)}"
    )
    return out_df


def validate_combination_split(df: pd.DataFrame, drug1_col: str, drug2_col: str) -> bool:
    """Verify zero combination leakage across Train, Validation, and Test splits.

    Args:
        df: DataFrame with 'split' column.
        drug1_col: Column name for drug 1.
        drug2_col: Column name for drug 2.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If combination leakage is detected.
    """
    logger.info("Validating Scenario 1: Combination-Wise Leakage...")
    
    train_df = df[df["split"] == 1]
    val_df = df[df["split"] == 2]
    test_df = df[df["split"] == 3]
    
    train_combos = set(get_canonical_combos(train_df, drug1_col, drug2_col))
    val_combos = set(get_canonical_combos(val_df, drug1_col, drug2_col))
    test_combos = set(get_canonical_combos(test_df, drug1_col, drug2_col))
    
    leakage_train_val = train_combos.intersection(val_combos)
    leakage_train_test = train_combos.intersection(test_combos)
    leakage_val_test = val_combos.intersection(test_combos)
    
    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise ValueError(
            f"CRITICAL LEAKAGE DETECTED in Scenario 1!\n"
            f"Train-Val Leakage Count: {len(leakage_train_val)}\n"
            f"Train-Test Leakage Count: {len(leakage_train_test)}\n"
            f"Val-Test Leakage Count: {len(leakage_val_test)}"
        )
        
    logger.info("[PASSED] Combination leakage = 0")
    return True


# =====================================================================
# SCENARIO 2 — CELL-WISE SPLIT
# =====================================================================

def cell_split(
    df: pd.DataFrame,
    cell_col: str,
    seed: int = 42
) -> pd.DataFrame:
    """Perform Scenario 2 Cell-Wise Split (60/20/20 on unique cell lines).

    Guarantees that every sample belonging to a cell line remains in the same split.

    Args:
        df: Input DataFrame.
        cell_col: Column name for cell lines.
        seed: Random seed for deterministic shuffling.

    Returns:
        pd.DataFrame: Copy of input DataFrame with added 'split' column (1=Train, 2=Val, 3=Test).
    """
    logger.info("Starting Scenario 2: Cell-Wise Split...")
    out_df = df.copy()
    
    # 1. Extract unique cell lines and shuffle deterministically
    unique_cells = sorted(list(set(out_df[cell_col].astype(str).str.strip())))
    rng = np.random.default_rng(seed)
    shuffled_cells = rng.permutation(unique_cells)
    
    # 2. Partition cell lines 60% / 20% / 20%
    n_total = len(shuffled_cells)
    n_train = int(round(n_total * 0.60))
    n_val = int(round(n_total * 0.20))
    
    train_cells = set(shuffled_cells[:n_train])
    val_cells = set(shuffled_cells[n_train:n_train + n_val])
    test_cells = set(shuffled_cells[n_train + n_val:])
    
    # 3. Map each cell line to its split value
    split_map = {}
    for c in train_cells: split_map[c] = 1
    for c in val_cells: split_map[c] = 2
    for c in test_cells: split_map[c] = 3
    
    out_df["split"] = out_df[cell_col].astype(str).str.strip().map(split_map).astype(int)
    
    logger.info(
        f"Scenario 2 Split Complete -> Unique Cell Lines: Total={n_total}, "
        f"Train={len(train_cells)}, Val={len(val_cells)}, Test={len(test_cells)}"
    )
    return out_df


def validate_cell_split(df: pd.DataFrame, cell_col: str) -> bool:
    """Verify zero cell line leakage across Train, Validation, and Test splits.

    Args:
        df: DataFrame with 'split' column.
        cell_col: Column name for cell lines.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If cell line leakage is detected.
    """
    logger.info("Validating Scenario 2: Cell-Wise Leakage...")
    
    train_cells = set(df[df["split"] == 1][cell_col].astype(str).str.strip())
    val_cells = set(df[df["split"] == 2][cell_col].astype(str).str.strip())
    test_cells = set(df[df["split"] == 3][cell_col].astype(str).str.strip())
    
    leakage_train_val = train_cells.intersection(val_cells)
    leakage_train_test = train_cells.intersection(test_cells)
    leakage_val_test = val_cells.intersection(test_cells)
    
    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise ValueError(
            f"CRITICAL LEAKAGE DETECTED in Scenario 2!\n"
            f"Train-Val Cell Leakage: {len(leakage_train_val)}\n"
            f"Train-Test Cell Leakage: {len(leakage_train_test)}\n"
            f"Val-Test Cell Leakage: {len(leakage_val_test)}"
        )
        
    logger.info("[PASSED] Cell line leakage = 0")
    return True


def preprocess_blank_smiles(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build NSC -> non-blank SMILES lookup map and fill all blank SMILES in-memory using fast vectorized pandas ops."""
    logger.info("Checking and filling blank SMILES strings via NSC lookup map...")
    out_df = df.copy()
    
    p1 = out_df[["Drug1_NSC_ID", "Drug1_SMILES"]].dropna().rename(columns={"Drug1_NSC_ID": "nsc", "Drug1_SMILES": "smiles"})
    p2 = out_df[["Drug2_NSC_ID", "Drug2_SMILES"]].dropna().rename(columns={"Drug2_NSC_ID": "nsc", "Drug2_SMILES": "smiles"})
    
    pairs = pd.concat([p1, p2], ignore_index=True)
    pairs["nsc"] = pairs["nsc"].astype(str).str.strip()
    pairs["smiles"] = pairs["smiles"].astype(str).str.strip()
    pairs = pairs[(pairs["nsc"] != "") & (pairs["smiles"] != "")].drop_duplicates()
    
    counts = pairs.groupby("nsc")["smiles"].nunique()
    multi_nscs = counts[counts > 1]
    if not multi_nscs.empty:
        raise ValueError(f"CRITICAL ERROR: NSCs map to multiple conflicting non-blank SMILES: {multi_nscs.to_dict()}")
        
    nsc_smiles_map = dict(zip(pairs["nsc"], pairs["smiles"]))
    
    s1_blank = out_df["Drug1_SMILES"].isna() | (out_df["Drug1_SMILES"].astype(str).str.strip() == "")
    s2_blank = out_df["Drug2_SMILES"].isna() | (out_df["Drug2_SMILES"].astype(str).str.strip() == "")
    
    d1_nscs_affected = set(out_df.loc[s1_blank, "Drug1_NSC_ID"].astype(str).str.strip().unique()) if "Drug1_NSC_ID" in out_df.columns else set()
    d2_nscs_affected = set(out_df.loc[s2_blank, "Drug2_NSC_ID"].astype(str).str.strip().unique()) if "Drug2_NSC_ID" in out_df.columns else set()
    all_affected_nscs = d1_nscs_affected.union(d2_nscs_affected)
    
    rows_filled = int((s1_blank | s2_blank).sum())
    grids_filled = rows_filled // 16 if rows_filled % 16 == 0 else rows_filled / 16.0
    
    nsc1_str = out_df["Drug1_NSC_ID"].astype(str).str.strip()
    nsc2_str = out_df["Drug2_NSC_ID"].astype(str).str.strip()
    
    fill1 = nsc1_str.map(nsc_smiles_map)
    fill2 = nsc2_str.map(nsc_smiles_map)
    
    out_df["Drug1_SMILES"] = out_df["Drug1_SMILES"].fillna(fill1)
    out_df.loc[out_df["Drug1_SMILES"].astype(str).str.strip() == "", "Drug1_SMILES"] = fill1
    
    out_df["Drug2_SMILES"] = out_df["Drug2_SMILES"].fillna(fill2)
    out_df.loc[out_df["Drug2_SMILES"].astype(str).str.strip() == "", "Drug2_SMILES"] = fill2

    rem_s1 = out_df["Drug1_SMILES"].isna() | (out_df["Drug1_SMILES"].astype(str).str.strip() == "")
    rem_s2 = out_df["Drug2_SMILES"].isna() | (out_df["Drug2_SMILES"].astype(str).str.strip() == "")
    if rem_s1.any() or rem_s2.any():
        unfilled_nscs = set(out_df.loc[rem_s1, "Drug1_NSC_ID"]).union(set(out_df.loc[rem_s2, "Drug2_NSC_ID"]))
        raise RuntimeError(f"Failed to fill blank SMILES for NSCs: {unfilled_nscs}")

    logger.info(
        f"[BLANK SMILES PREPROCESSING COMPLETE]\n"
        f"  Affected NSCs    : {sorted(list(all_affected_nscs))}\n"
        f"  Rows Filled      : {rows_filled}\n"
        f"  Grids Affected   : {grids_filled:.0f}\n"
        f"  Remaining Blanks : 0"
    )
    
    stats = {
        "affected_nscs": sorted(list(all_affected_nscs)),
        "rows_filled": rows_filled,
        "grids_filled": grids_filled,
        "remaining_blanks": 0
    }
    return out_df, stats


# =====================================================================
# SCENARIO 3 — STRICT UNSEEN-DRUG SPLIT (NSC-BASED)
# =====================================================================

def drug_split(
    df: pd.DataFrame,
    drug1_col: str,
    drug2_col: str,
    seed: int = 42
) -> Tuple[pd.DataFrame, Set[str], Set[str], Set[str], int]:
    """Perform Strict Scenario 3 Drug-Wise Split (60/20/20 on unique NSC IDs).

    STRICT Requirements:
      - Canonical drug identity: Drug1_NSC_ID and Drug2_NSC_ID.
      - Unique NSCs partitioned 60/20/20 into TrainNSCs, ValNSCs, TestNSCs.
      - Train pairs (split=1): BOTH Drug1 ∈ TrainNSCs AND Drug2 ∈ TrainNSCs.
      - Val pairs (split=2): BOTH Drug1 ∈ ValNSCs AND Drug2 ∈ ValNSCs.
      - Test pairs (split=3): BOTH Drug1 ∈ TestNSCs AND Drug2 ∈ TestNSCs.
      - Discard ALL mixed pairs (e.g. Train x Val, Train x Test, Val x Test, etc.).

    Args:
        df: Input DataFrame containing Drug1_NSC_ID and Drug2_NSC_ID columns.
        drug1_col: Column name for drug 1 NSC ID.
        drug2_col: Column name for drug 2 NSC ID.
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple[pd.DataFrame, Set[str], Set[str], Set[str], int]:
          (filtered_df_with_split, train_nscs, val_nscs, test_nscs, discarded_count).
    """
    logger.info("Starting Strict Scenario 3: NSC-Wise Unseen-Drug Split...")
    
    d1_vals = df[drug1_col].astype(str).str.strip().values
    d2_vals = df[drug2_col].astype(str).str.strip().values
    
    # 1. Extract all unique NSCs across both drug columns
    all_nscs = sorted(list(set(d1_vals).union(set(d2_vals))))
    
    # 2. Shuffle unique NSCs deterministically
    rng = np.random.default_rng(seed)
    shuffled_nscs = rng.permutation(all_nscs)
    
    # 3. Partition NSCs 60% / 20% / 20%
    n_total = len(shuffled_nscs)
    n_train = int(round(n_total * 0.60))
    n_val = int(round(n_total * 0.20))
    
    train_nscs = set(shuffled_nscs[:n_train])
    val_nscs = set(shuffled_nscs[n_train:n_train + n_val])
    test_nscs = set(shuffled_nscs[n_train + n_val:])
    
    logger.info(
        f"NSC Set Partitioning -> Total Unique NSCs={n_total}, "
        f"Train={len(train_nscs)}, Val={len(val_nscs)}, Test={len(test_nscs)}"
    )
    
    # 4. Strict categorization: Train x Train, Val x Val, Test x Test
    n_samples = len(df)
    
    train_nscs_arr = np.array(list(train_nscs))
    val_nscs_arr = np.array(list(val_nscs))
    test_nscs_arr = np.array(list(test_nscs))
    
    d1_is_train = np.isin(d1_vals, train_nscs_arr)
    d2_is_train = np.isin(d2_vals, train_nscs_arr)
    
    d1_is_val = np.isin(d1_vals, val_nscs_arr)
    d2_is_val = np.isin(d2_vals, val_nscs_arr)
    
    d1_is_test = np.isin(d1_vals, test_nscs_arr)
    d2_is_test = np.isin(d2_vals, test_nscs_arr)
    
    train_mask = d1_is_train & d2_is_train
    val_mask = d1_is_val & d2_is_val
    test_mask = d1_is_test & d2_is_test
    
    assigned_splits = np.full(n_samples, -1, dtype=int)
    assigned_splits[train_mask] = 1
    assigned_splits[val_mask] = 2
    assigned_splits[test_mask] = 3
            
    out_df = df.copy()
    out_df["split"] = assigned_splits
    
    # Filter out discarded mixed-pair rows
    filtered_df = out_df[out_df["split"] != -1].copy()
    discarded_count = n_samples - len(filtered_df)
    
    logger.info(
        f"Strict Scenario 3 Categorization Complete -> Total Samples={n_samples}, "
        f"Kept={len(filtered_df)}, Discarded Mixed Pairs={discarded_count} "
        f"({(discarded_count / n_samples) * 100:.2f}%)"
    )
    
    return filtered_df, train_nscs, val_nscs, test_nscs, discarded_count


def validate_drug_split(
    df: pd.DataFrame,
    drug1_col: str,
    drug2_col: str,
    train_nscs: Set[str],
    val_nscs: Set[str],
    test_nscs: Set[str]
) -> bool:
    """Verify zero NSC leakage across Train, Validation, and Test splits in Strict Scenario 3.

    Args:
        df: DataFrame with 'split' column.
        drug1_col: Column name for drug 1 NSC ID.
        drug2_col: Column name for drug 2 NSC ID.
        train_nscs: Partitioned train NSC set.
        val_nscs: Partitioned validation NSC set.
        test_nscs: Partitioned test NSC set.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If NSC set leakage or row assignment corruption is detected.
    """
    logger.info("Validating Strict Scenario 3: NSC Leakage & Pair Rules...")
    
    # 1. Verify NSC sets are strictly disjoint
    leak_train_val = train_nscs.intersection(val_nscs)
    leak_train_test = train_nscs.intersection(test_nscs)
    leak_val_test = val_nscs.intersection(test_nscs)
    
    if leak_train_val or leak_train_test or leak_val_test:
        raise ValueError(
            f"CRITICAL NSC SET LEAKAGE DETECTED in Strict Scenario 3!\n"
            f"Train-Val NSC Intersection: {len(leak_train_val)}\n"
            f"Train-Test NSC Intersection: {len(leak_train_test)}\n"
            f"Val-Test NSC Intersection: {len(leak_val_test)}"
        )
        
    # 2. Verify row-level drug assignment invariants (Train x Train, Val x Val, Test x Test)
    train_df = df[df["split"] == 1]
    val_df = df[df["split"] == 2]
    test_df = df[df["split"] == 3]
    
    train_d1 = train_df[drug1_col].astype(str).str.strip()
    train_d2 = train_df[drug2_col].astype(str).str.strip()
    if not (train_d1.isin(train_nscs) & train_d2.isin(train_nscs)).all():
        raise ValueError("Strict Scenario 3 Train row violation: pair contains non-train NSC!")
            
    val_d1 = val_df[drug1_col].astype(str).str.strip()
    val_d2 = val_df[drug2_col].astype(str).str.strip()
    if not (val_d1.isin(val_nscs) & val_d2.isin(val_nscs)).all():
        raise ValueError("Strict Scenario 3 Validation row violation: pair is not Val x Val!")

    test_d1 = test_df[drug1_col].astype(str).str.strip()
    test_d2 = test_df[drug2_col].astype(str).str.strip()
    if not (test_d1.isin(test_nscs) & test_d2.isin(test_nscs)).all():
        raise ValueError("Strict Scenario 3 Test row violation: pair is not Test x Test!")
            
    logger.info("[PASSED] NSC leakage = 0 | Strict Train x Train, Val x Val, Test x Test verified!")
    return True


# =====================================================================
# SUMMARY REPORT PRINTING
# =====================================================================

def print_scenario_summary(
    scenario_name: str,
    df: pd.DataFrame,
    drug1_col: str,
    drug2_col: str,
    cell_col: str,
    discarded_count: int = 0
):
    """Print comprehensive dataset statistics and leakage verification status for a scenario.

    Args:
        scenario_name: Name of the scenario.
        df: Processed DataFrame with 'split' column.
        drug1_col: Column name for drug 1.
        drug2_col: Column name for drug 2.
        cell_col: Column name for cell line.
        discarded_count: Number of discarded ambiguous samples.
    """
    train_df = df[df["split"] == 1]
    val_df = df[df["split"] == 2]
    test_df = df[df["split"] == 3]
    
    def get_stats(sub_df: pd.DataFrame) -> Dict[str, int]:
        if len(sub_df) == 0:
            return {"samples": 0, "drugs": 0, "combos": 0, "cells": 0}
        d1 = sub_df[drug1_col].astype(str).str.strip().values
        d2 = sub_df[drug2_col].astype(str).str.strip().values
        drugs = set(d1).union(set(d2))
        combos = set(get_canonical_combos(sub_df, drug1_col, drug2_col))
        cells = set(sub_df[cell_col].astype(str).str.strip().values)
        return {
            "samples": len(sub_df),
            "drugs": len(drugs),
            "combos": len(combos),
            "cells": len(cells)
        }
        
    tot = get_stats(df)
    trn = get_stats(train_df)
    val = get_stats(val_df)
    tst = get_stats(test_df)
    
    print("\n" + "=" * 75)
    print(f" {scenario_name.upper()} SUMMARY REPORT")
    print("=" * 75)
    print(f"Total Samples Kept         : {tot['samples']}")
    if discarded_count > 0:
        print(f"Discarded Samples          : {discarded_count}")
    print(f"Number of Training Samples   : {trn['samples']} ({trn['samples']/tot['samples']*100:.1f}%)")
    print(f"Number of Validation Samples : {val['samples']} ({val['samples']/tot['samples']*100:.1f}%)")
    print(f"Number of Testing Samples    : {tst['samples']} ({tst['samples']/tot['samples']*100:.1f}%)")
    print("-" * 75)
    print(f"Unique Drugs              : Total={tot['drugs']} | Train={trn['drugs']} | Val={val['drugs']} | Test={tst['drugs']}")
    print(f"Unique Drug Combinations  : Total={tot['combos']} | Train={trn['combos']} | Val={val['combos']} | Test={tst['combos']}")
    print(f"Unique Cell Lines         : Total={tot['cells']} | Train={trn['cells']} | Val={val['cells']} | Test={tst['cells']}")
    print("-" * 75)
    
    if "Combination" in scenario_name:
        print("[PASSED] Combination leakage = 0")
    elif "Cell" in scenario_name:
        print("[PASSED] Cell line leakage = 0")
    elif "Drug" in scenario_name:
        print("[PASSED] Drug leakage = 0")
    print("=" * 75 + "\n")


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production-Ready Dataset Splitting Pipeline for CancerCombo"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/DrugCombination_with_NSC_ID.zip",
        help="Path to input CSV or ZIP dataset archive."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/splits",
        help="Directory to save generated scenario CSV files."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default=42)."
    )
    parser.add_argument("--drug1_col", type=str, default=None, help="Column name for Drug 1 / SMILES 1.")
    parser.add_argument("--drug2_col", type=str, default=None, help="Column name for Drug 2 / SMILES 2.")
    parser.add_argument("--cell_col", type=str, default=None, help="Column name for Cell Line.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Initialized DatasetSplitter with Random Seed={args.seed}")
    print(f"Scenario split seed: {args.seed}")
    
    # 1. Load Input Dataset & Gene Expression Data
    gex_dict = load_nci60_gex()
    if not gex_dict:
        raise ValueError("Failed to load NCI-60 GEX landmark data from data/features/NCI-60_landmark_gex.csv")
        
    if not os.path.exists(args.input_csv):
        logger.warning(f"Input file not found at '{args.input_csv}'. Creating mock dataset for pipeline demonstration...")
        raw_df = pd.DataFrame({
            "Sample": [f"Drug_{i%10}_Drug_{(i+3)%10}_MCF7" for i in range(100)],
            "Drug1_SMILES": ["CCO" for _ in range(100)],
            "Drug2_SMILES": ["CC(=O)O" for _ in range(100)],
            "Drug1_NSC_ID": [f"{100+i%10}" for i in range(100)],
            "Drug2_NSC_ID": [f"{100+(i+3)%10}" for i in range(100)],
            "Drug1_Dose": [0.1 for _ in range(100)],
            "Drug2_Dose": [0.2 for _ in range(100)],
            "Response": [100.0 for _ in range(100)]
        })
    elif args.input_csv.endswith(".zip"):
        logger.info(f"Extracting dataset from ZIP archive: '{args.input_csv}'...")
        with zipfile.ZipFile(args.input_csv, "r") as z:
            csv_names = [f for f in z.namelist() if f.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV file found inside ZIP archive '{args.input_csv}'.")
            dfs = []
            for csv_name in csv_names:
                with z.open(csv_name) as f:
                    dfs.append(pd.read_csv(f))
            raw_df = pd.concat(dfs, ignore_index=True)
    else:
        logger.info(f"Loading CSV dataset: '{args.input_csv}'...")
        raw_df = pd.read_csv(args.input_csv)
        
    logger.info(f"Loaded raw input dataset with {len(raw_df)} rows.")

    # 1b. Preprocess Blank SMILES via NSC Lookup Map
    raw_df, blank_stats = preprocess_blank_smiles(raw_df)
    
    # 2. Parse and Group into 2D Dose-Response Matrix Surface Samples
    logger.info("Parsing and assembling 2D dose-response matrix surface samples...")
    from dataset import parse_dataframe_to_records
    records = parse_dataframe_to_records(raw_df, known_gex_dict=gex_dict)
    
    if not records:
        raise ValueError("No valid matched surface samples extracted from input dataset!")
        
    # Convert records to DataFrame for splitting
    records_df_list = []
    for r in records:
        records_df_list.append({
            "smiles_a": r["smiles_a"],
            "smiles_b": r["smiles_b"],
            "nsc_a": r.get("nsc_a", ""),
            "nsc_b": r.get("nsc_b", ""),
            "cell_line_name": r["cell_line_name"],
            "doses_a": json.dumps(r["doses_a"]),
            "doses_b": json.dumps(r["doses_b"]),
            "viability_matrix": json.dumps(r["viability_matrix"])
        })
    df = pd.DataFrame(records_df_list)
    logger.info(f"Assembled {len(df)} 2D dose-response surface samples for scenario splitting.")
    
    # 3. Resolve Column Names
    drug1_col, drug2_col, cell_col = detect_columns(df, "smiles_a", "smiles_b", "cell_line_name")
    
    # -----------------------------------------------------------------
    # SCENARIO 1 — COMBINATION-WISE SPLIT
    # -----------------------------------------------------------------
    s1_df = combination_split(df, drug1_col, drug2_col, seed=args.seed)
    validate_combination_split(s1_df, drug1_col, drug2_col)
    s1_path = os.path.join(args.output_dir, "scenario1_combination.csv")
    s1_df.to_csv(s1_path, index=False)
    logger.info(f"Saved Scenario 1 output to: {s1_path}")
    print_scenario_summary("Scenario 1 — Combination-Wise Split", s1_df, drug1_col, drug2_col, cell_col)
    
    # -----------------------------------------------------------------
    # SCENARIO 2 — CELL-WISE SPLIT
    # -----------------------------------------------------------------
    s2_df = cell_split(df, cell_col, seed=args.seed)
    validate_cell_split(s2_df, cell_col)
    s2_path = os.path.join(args.output_dir, "scenario2_cell.csv")
    s2_df.to_csv(s2_path, index=False)
    logger.info(f"Saved Scenario 2 output to: {s2_path}")
    print_scenario_summary("Scenario 2 — Cell-Wise Split", s2_df, drug1_col, drug2_col, cell_col)
    
    # -----------------------------------------------------------------
    # SCENARIO 3 — STRICT UNSEEN-DRUG SPLIT (NSC-BASED)
    # -----------------------------------------------------------------
    nsc1_col = "nsc_a" if "nsc_a" in df.columns and df["nsc_a"].str.len().gt(0).any() else drug1_col
    nsc2_col = "nsc_b" if "nsc_b" in df.columns and df["nsc_b"].str.len().gt(0).any() else drug2_col
    
    s3_df, trn_nsc, val_nsc, tst_nsc, discarded = drug_split(df, nsc1_col, nsc2_col, seed=args.seed)
    validate_drug_split(s3_df, nsc1_col, nsc2_col, trn_nsc, val_nsc, tst_nsc)
    s3_path = os.path.join(args.output_dir, "scenario3_drug_strict_60_20_20.csv")
    s3_df.to_csv(s3_path, index=False)
    logger.info(f"Saved Strict Scenario 3 output to: {s3_path}")
    print(f"Scenario 3 split file: {s3_path}")
    print_scenario_summary("Scenario 3 — Strict Unseen-Drug Split", s3_df, nsc1_col, nsc2_col, cell_col, discarded_count=discarded)
    
    logger.info("All 3 Dataset Splitting Scenarios executed, validated, and saved successfully!")


if __name__ == "__main__":
    main()
