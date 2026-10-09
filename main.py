import os
for _k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "1"

from helpers import enforce_single_thread
enforce_single_thread()

import argparse
import torch
import sys
from train import run_training
from evaluate import run_evaluation
from predictor import SynergyPredictor
from config import load_config
from helpers import generate_mock_data
from logger import setup_logger
import numpy as np

def run_prediction_cli(checkpoint_path: str):
    logger = setup_logger("CancerCombo Predict CLI")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
        
    m_config, _ = load_config("config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = SynergyPredictor(checkpoint_path, m_config, device=device)
    
    # Run CLI test case
    smiles_a = "CC(=O)NC1=CC=C(C=C1)O"
    smiles_b = "CC1=CC(=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
    
    cell_line_gene_expr = np.random.randn(976).astype(np.float32)
    doses_a = np.array([0.0, 0.1, 1.0, 10.0], dtype=np.float32)
    doses_b = np.array([0.0, 0.2, 2.0, 20.0], dtype=np.float32)
    
    logger.info("Executing CLI sample prediction...")
    res = predictor.predict(
        smiles_a=smiles_a,
        smiles_b=smiles_b,
        cell_line_gene_expr=cell_line_gene_expr,
        doses_a=doses_a,
        doses_b=doses_b
    )
    
    logger.info("Prediction successful.")
    logger.info("Estimated parameters:")
    for p, val in res["pharmacological_parameters"].items():
        logger.info(f"  {p}: {val:.6f}")
        
    logger.info("Response surface grid viability:")
    logger.info("\n" + str(res["predicted_viability_matrix"]))

def main():
    parser = argparse.ArgumentParser(description="CancerCombo CLI Management Tool")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluate", "predict"],
        help="CancerCombo execution pipeline mode (default: train)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/drugcell_crossattention/cancercombo_best.ckpt",
        help="Path to model checkpoint (.ckpt or .pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Scenario split number: 1 (combination-wise), 2 (cell-wise), or 3 (drug-wise) (default: 1)"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="native",
        choices=["auto", "lightning", "native"],
        help="Execution engine: native, lightning, or auto (default: native — Lightning's TQDMProgressBar callback has caused hangs on this hardware; see commit b83196e)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Sub-sample dataset for fast debug or testing"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        run_training(
            config_path=args.config,
            scenario=args.scenario,
            engine=args.engine,
            epochs=args.epochs,
            max_samples=args.max_samples
        )
    elif args.mode == "evaluate":
        run_evaluation(checkpoint_path=args.checkpoint, config_path=args.config, scenario=args.scenario)
    elif args.mode == "predict":
        run_prediction_cli(args.checkpoint)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
