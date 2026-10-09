#!/usr/bin/env python3
"""
fit_descriptor_scaler.py - Scenario-Specific Descriptor Scaler Fitting Pipeline

Fits DescriptorScaler using ONLY the training drugs (split == 1) of a specified scenario split.
Prevents data leakage by strictly ignoring validation (split == 2) and test (split == 3) drugs.
Persists the fitted parameters alongside a deterministic SHA256 training NSC hash.

Usage:
    python fit_descriptor_scaler.py --scenario 3
    # or explicitly:
    python fit_descriptor_scaler.py \
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
import argparse
import logging
from config import RAW_FEATURE_STORE_PATH, get_scenario_split_path, get_scenario_scaler_path
from precompute_molecular_features import fit_scenario_scaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ScalerFitter")


def main():
    parser = argparse.ArgumentParser(
        description="Fit Scenario-Specific Descriptor Scaler on Training Drugs (split == 1) Only"
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Scenario number (1, 2, or 3) to automatically route official split CSV and output scaler path."
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        default=None,
        help="Path to scenario split CSV (e.g. data/splits/scenario3_drug_strict_60_20_20.csv)"
    )
    parser.add_argument(
        "--feature_file",
        type=str,
        default=RAW_FEATURE_STORE_PATH,
        help="Path to generic molecular feature store (.pt)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/drugcell_crossattention",
        help="Checkpoint directory where scaler should be saved."
    )
    parser.add_argument(
        "--scaler_output",
        type=str,
        default=None,
        help="Exact path to save fitted scaler (.npz). Defaults to <checkpoint_dir>/descriptor_scaler_s{scenario}.npz."
    )
    args = parser.parse_args()

    split_csv = args.split_csv
    scaler_output = args.scaler_output

    if args.scenario is not None:
        if not split_csv:
            split_csv = get_scenario_split_path(args.scenario)
        if not scaler_output:
            scaler_output = get_scenario_scaler_path(args.checkpoint_dir, args.scenario)
    elif not split_csv:
        raise ValueError("Either --scenario (1, 2, 3) or --split_csv must be provided.")

    if not scaler_output:
        scaler_output = os.path.join(args.checkpoint_dir, "descriptor_scaler_s3.npz")

    logger.info(f"Fitting DescriptorScaler for split: '{split_csv}' -> Output: '{scaler_output}'")
    scaler = fit_scenario_scaler(
        split_csv=split_csv,
        feature_store_or_path=args.feature_file,
        scaler_output=scaler_output
    )
    logger.info(f"Done. Scaler fitted and persisted to '{scaler_output}'.")


if __name__ == "__main__":
    main()
