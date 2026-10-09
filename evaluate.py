import os
for _k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "1"

from helpers import enforce_single_thread
enforce_single_thread()

import torch
if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

import pandas as pd
from torch.utils.data import DataLoader
from config import load_config, RAW_FEATURE_STORE_PATH, get_scenario_split_path, get_scenario_scaler_path
from dataset import DrugComboDataset, load_nci60_gex
from cancercombo import CancerCombo
from logger import setup_logger
from metrics import calculate_metrics
from typing import Any, Optional
import numpy as np


def load_checkpoint_strictly(model: torch.nn.Module, checkpoint_path: str, config: Any = None, device: str = "cpu") -> dict:
    """Load model weights with strict state_dict matching and shape validation."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Required checkpoint file missing at '{checkpoint_path}'. Fail loudly.")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    expected_morgan_in_dim = getattr(config, "morgan_in_dim", 1024) if config else getattr(getattr(model, "config", None), "morgan_in_dim", 1024)

    # Check for legacy Phase 1 (pre-multimodal) checkpoint
    old_weight_key = "morgan_enc.projection.0.weight"
    new_weight_key = "drug_encoder.morgan_enc.projection.0.weight"

    if old_weight_key in state_dict and new_weight_key not in state_dict:
        raise ValueError(
            "Incompatible Phase 1 chemistry-only checkpoint detected (missing 'drug_encoder' keys). "
            "Phase 3 multimodal architecture requires a new checkpoint/training run."
        )

    if new_weight_key in state_dict:
        weight_shape = state_dict[new_weight_key].shape
        in_dim = weight_shape[1]
        if in_dim != expected_morgan_in_dim:
            raise ValueError(
                f"Checkpoint/model dimension mismatch: checkpoint expects Morgan input dimension {in_dim}, "
                f"active configuration requires {expected_morgan_in_dim}."
            )

    model.load_state_dict(state_dict, strict=True)
    return checkpoint


class ModelEvaluator:
    """Evaluates the CancerCombo model metrics on validation or test sets."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader):
        """Runs validation loop and calculates standard synergy matrix prediction metrics."""
        model.eval()
        model.to(self.device)

        preds_list = []
        trues_list = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                b_gpu = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                viability = b_gpu.get("viability", b_gpu.get("viability_matrix"))

                da_m = b_gpu.get("drug_a_morgan")
                db_m = b_gpu.get("drug_b_morgan")
                da_d = b_gpu.get("drug_a_desc")
                db_d = b_gpu.get("drug_b_desc")
                da_dti = b_gpu.get("drug_a_dti")
                db_dti = b_gpu.get("drug_b_dti")
                da_pw = b_gpu.get("drug_a_pathway")
                db_pw = b_gpu.get("drug_b_pathway")
                da_m_dti = b_gpu.get("drug_a_m_dti")
                db_m_dti = b_gpu.get("drug_b_m_dti")
                da_m_pw = b_gpu.get("drug_a_m_pathway")
                db_m_pw = b_gpu.get("drug_b_m_pathway")

                cell = b_gpu.get("cell_line")
                d_a = b_gpu.get("doses_a")
                d_b = b_gpu.get("doses_b")

                y_pred, _ = model(
                    drug_a_morgan=da_m,
                    drug_a_desc=da_d,
                    drug_a_dti=da_dti,
                    drug_a_pathway=da_pw,
                    drug_a_m_dti=da_m_dti,
                    drug_a_m_pathway=da_m_pw,
                    drug_b_morgan=db_m,
                    drug_b_desc=db_d,
                    drug_b_dti=db_dti,
                    drug_b_pathway=db_pw,
                    drug_b_m_dti=db_m_dti,
                    drug_b_m_pathway=db_m_pw,
                    cell_line=cell,
                    doses_a=d_a,
                    doses_b=d_b
                )

                if batch_idx == 0:
                    logger = setup_logger("CancerCombo Eval")
                    logger.info("=" * 60)
                    logger.info("[EVAL FIRST BATCH TENSOR DIAGNOSTICS]")
                    logger.info(f"  drug_a_morgan shape : {tuple(da_m.shape) if da_m is not None else None}")
                    logger.info(f"  drug_a_desc shape   : {tuple(da_d.shape) if da_d is not None else None}")
                    logger.info(f"  drug_a_dti shape    : {tuple(da_dti.shape) if da_dti is not None else None}")
                    logger.info(f"  drug_a_pw shape     : {tuple(da_pw.shape) if da_pw is not None else None}")
                    logger.info(f"  drug_b_morgan shape : {tuple(db_m.shape) if db_m is not None else None}")
                    logger.info(f"  drug_b_desc shape   : {tuple(db_d.shape) if db_d is not None else None}")
                    logger.info(f"  drug_b_dti shape    : {tuple(db_dti.shape) if db_dti is not None else None}")
                    logger.info(f"  drug_b_pw shape     : {tuple(db_pw.shape) if db_pw is not None else None}")
                    logger.info(f"  cell_line shape     : {tuple(cell.shape) if cell is not None else None}")
                    logger.info(f"  doses_a shape       : {tuple(d_a.shape) if d_a is not None else None}")
                    logger.info(f"  doses_b shape       : {tuple(d_b.shape) if d_b is not None else None}")
                    logger.info(f"  target shape        : {tuple(viability.shape) if viability is not None else None}")
                    logger.info(f"  prediction shape    : {tuple(y_pred.shape)}")
                    logger.info("=" * 60)

                preds_list.append(y_pred.cpu().numpy())
                trues_list.append(viability.cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)
        trues = np.concatenate(trues_list, axis=0)

        logger = setup_logger("CancerCombo Eval")
        logger.info("=" * 60)
        logger.info("[EVALUATION DATASET & PREDICTION AUDIT]")
        logger.info(f"  y_true shape : {trues.shape}")
        logger.info(f"  y_pred shape : {preds.shape}")
        logger.info("-" * 60)
        logger.info(f"  y_true stats -> min: {trues.min():.4f}, max: {trues.max():.4f}, mean: {trues.mean():.4f}, std: {trues.std():.4f}")
        logger.info(f"  y_pred stats -> min: {preds.min():.4f}, max: {preds.max():.4f}, mean: {preds.mean():.4f}, std: {preds.std():.4f}")
        logger.info("=" * 60)

        return calculate_metrics(preds, trues)

def run_evaluation(checkpoint_path: str = "checkpoints/drugcell_crossattention/cancercombo_best.ckpt", config_path: str = "config.yaml", scenario: int = 1):
    """Load model checkpoint and evaluate performance.

    Args:
        checkpoint_path: Path to checkpoint.
        config_path: Path to configuration YAML.
        scenario: Split scenario (1, 2, or 3).
    """
    logger = setup_logger("CancerCombo Eval")
    logger.info("Setting up configs and real held-out evaluation dataset...")
    
    m_config, t_config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure PyTorch CUDA backends to avoid hangs/deadlocks on GPU container setups
    if device == "cuda":
        logger.info("Configuring PyTorch CUDA settings...")
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            logger.info("  [SUCCESS] Disabled FlashAttention and MemEfficient Attention SDP backends.")
        except Exception as e:
            logger.warning(f"  [WARNING] Failed to configure SDPA kernels: {e}")
    
    split_path = get_scenario_split_path(scenario)

    logger.info(f"Loading held-out test split (Scenario {scenario}) from {split_path}...")
    if scenario == 3:
        print(f"Scenario 3 split file: {split_path}")
        logger.info(f"Scenario 3 split file: {split_path}")

    split_df = pd.read_csv(split_path)
    if "split" not in split_df.columns:
        logger.error(f"Split file does not contain a 'split' column: {split_path}")
        return

    val_df = split_df[split_df["split"] == 2].copy()
    test_df = split_df[split_df["split"] == 3].copy()
    if test_df.empty:
        logger.error(f"No held-out test rows found in {split_path} (split == 3).")
        return

    cell_features = load_nci60_gex("data/features/NCI-60_landmark_gex.csv", target_dim=m_config.cell_in_dim)
    if not cell_features:
        logger.error("Cell feature file not found or unreadable: data/features/NCI-60_landmark_gex.csv")
        return

    from dataset import parse_dataframe_to_records, load_precomputed_drug_features, load_descriptor_scaler, load_dti_embeddings, load_pathway_embeddings
    test_records = parse_dataframe_to_records(test_df, known_gex_dict=cell_features)
    val_records = parse_dataframe_to_records(val_df, known_gex_dict=cell_features) if not val_df.empty else []

    dti_store = load_dti_embeddings("data/drug_target_embeddings_2048.xlsx")
    pathway_store = load_pathway_embeddings("data/drug_pathway_embeddings_1152.xlsx")
    logger.info(f"Loaded DTI embeddings for {len(dti_store)} keys from data/drug_target_embeddings_2048.xlsx.")
    logger.info(f"Loaded Pathway embeddings for {len(pathway_store)} keys from data/drug_pathway_embeddings_1152.xlsx.")

    import re
    def log_split_bio_availability(split_name: str, records: list, dti_s: dict, pw_s: dict):
        dti_missing_cnt = 0
        pw_missing_cnt = 0
        missing_dti_nscs = set()
        missing_pw_nscs = set()
        for rec in records:
            for key_nsc in ['nsc_a', 'nsc_b']:
                nsc_val = str(rec.get(key_nsc, '')).strip()
                clean_nsc = re.sub(r'^(NSC[-_]?)', '', nsc_val, flags=re.IGNORECASE).strip()
                keys_to_try = [nsc_val, clean_nsc, f"NSC-{clean_nsc}"]
                if not any(k in dti_s for k in keys_to_try):
                    dti_missing_cnt += 1
                    missing_dti_nscs.add(clean_nsc)
                if not any(k in pw_s for k in keys_to_try):
                    pw_missing_cnt += 1
                    missing_pw_nscs.add(clean_nsc)
        logger.info(
            f"Split '{split_name}' Bio Availability Audit ({len(records)} records) | "
            f"DTI-missing drug instances: {dti_missing_cnt} (NSCs: {sorted(list(missing_dti_nscs))}) | "
            f"Pathway-missing drug instances: {pw_missing_cnt} (NSCs: {sorted(list(missing_pw_nscs))})"
        )

    if val_records:
        log_split_bio_availability("val", val_records, dti_store, pathway_store)
    log_split_bio_availability("test", test_records, dti_store, pathway_store)

    feature_store_path = RAW_FEATURE_STORE_PATH
    drug_features = load_precomputed_drug_features(feature_path=feature_store_path)
    if not drug_features:
        raise FileNotFoundError(f"Required Morgan 1024 r=3 feature store missing at '{feature_store_path}'. Fail loudly.")
    logger.info(f"Loaded Morgan (1024-dim) + RDKit (200-dim) drug features for {len(drug_features)} keys from {feature_store_path}.")

    # Load fitted DescriptorScaler associated strictly with Train NSCs of the scenario
    from dataset import extract_scenario_nscs, load_descriptor_scaler
    unique_train_nscs, unique_val_nscs, unique_test_nscs = extract_scenario_nscs(split_df)

    checkpoint_dir = getattr(t_config, "checkpoint_dir", "checkpoints/drugcell_crossattention")
    scaler_path = get_scenario_scaler_path(checkpoint_dir, scenario)

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Required DescriptorScaler artifact missing for Scenario {scenario} at '{scaler_path}'. "
            f"Evaluation never fits a scaler on the fly. Train the model or run fit_descriptor_scaler.py first."
        )

    desc_scaler = load_descriptor_scaler(scaler_path, train_nsc_ids=unique_train_nscs)
    logger.info(f"Loaded DescriptorScaler from {scaler_path} and verified Train NSC hash!")

    unique_drugs_test = set(test_df.get('smiles_a', test_df.get('Drug1_SMILES', []))).union(
        set(test_df.get('smiles_b', test_df.get('Drug2_SMILES', [])))
    )
    unique_cells_test = set(test_df.get('cell_line_name', test_df.get('Sample', [])))

    logger.info("=" * 60)
    logger.info(f"EVALUATION DATASET AUDIT (Scenario {scenario}):")
    logger.info(f"  Validation Records Count (Split 2) : {len(val_records)}")
    logger.info(f"  Test Records Count       (Split 3) : {len(test_records)}")
    logger.info(f"  Train Unique Drugs Count (Split 1) : {len(unique_train_nscs)}")
    logger.info(f"  Val Unique Drugs Count   (Split 2) : {len(unique_val_nscs)}")
    logger.info(f"  Test Unique Drugs Count  (Split 3) : {len(unique_drugs_test)}")
    logger.info(f"  Test Unique Cells Count            : {len(unique_cells_test)}")
    logger.info("=" * 60)

    eval_batch_size = getattr(t_config, "batch_size", 128)

    test_dataset = DrugComboDataset(
        test_records, cell_features, drug_feature_store=drug_features, descriptor_scaler=desc_scaler,
        feature_store_path=feature_store_path,
        dti_store=dti_store, pathway_store=pathway_store,
        use_pretrained_molformer=m_config.use_pretrained_molformer,
        molformer_model_name=m_config.molformer_model_name
    )
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    if val_records:
        val_dataset = DrugComboDataset(
            val_records, cell_features, drug_feature_store=drug_features, descriptor_scaler=desc_scaler,
            feature_store_path=feature_store_path,
            dti_store=dti_store, pathway_store=pathway_store,
            use_pretrained_molformer=m_config.use_pretrained_molformer,
            molformer_model_name=m_config.molformer_model_name
        )
        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    else:
        val_loader = None

    model = CancerCombo(m_config)
    checkpoint = load_checkpoint_strictly(model, checkpoint_path, config=m_config, device=device)
    ckpt_epoch = checkpoint.get("epoch", "N/A")
    ckpt_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", "N/A"))

    logger.info("=" * 60)
    logger.info(f"CHECKPOINT LOADED SUCCESSFULLY (STRICT MATCH):")
    logger.info(f"  Path              : {checkpoint_path}")
    logger.info(f"  Checkpoint Epoch  : {ckpt_epoch}")
    logger.info(f"  Training Val Loss : {ckpt_val_loss}")
    logger.info("=" * 60)
    evaluator = ModelEvaluator(device=device)

    if val_loader:
        logger.info("\n=== EVALUATING ON VALIDATION SET (Split 2) ===")
        val_results = evaluator.evaluate(model, val_loader)
        logger.info("Validation Set (Split 2) Results:")
        for metric, val in val_results.items():
            logger.info(f"  VAL_{metric.upper()}: {val:.4f}")

    logger.info("\n=== EVALUATING ON HELD-OUT TEST SET (Split 3) ===")
    results = evaluator.evaluate(model, test_loader)
    
    logger.info("Held-Out Test Set (Split 3) Results:")
    for metric, val in results.items():
        logger.info(f"  TEST_{metric.upper()}: {val:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate CancerCombo")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/drugcell_crossattention/cancercombo_best.ckpt")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--scenario", type=int, default=1, help="Split scenario (1, 2, or 3)")
    args = parser.parse_args()
    
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        alt_path = os.path.join("checkpoints", "cancercombo_best.ckpt")
        if os.path.exists(alt_path):
            ckpt_path = alt_path
        else:
            print(f"Error: Specified checkpoint '{ckpt_path}' does not exist.")
            print("Please train a model first or specify a valid checkpoint path via '--checkpoint'.")
            import sys
            sys.exit(1)
    run_evaluation(ckpt_path, config_path=args.config, scenario=args.scenario)

