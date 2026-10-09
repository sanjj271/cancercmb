import os
import sys
import argparse

for _k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "1"

from helpers import enforce_single_thread
enforce_single_thread()

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from config import load_config
from dataset import (
    load_nci60_gex,
    parse_dataframe_to_records,
    load_precomputed_drug_features,
    DrugComboDataset,
)
from cancercombo import CancerCombo


def load_state_into(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith("molformer_enc.pretrained_")]
    if missing:
        print(f"[A3] WARNING: {len(missing)} trainable keys missing from checkpoint, e.g. {missing[:5]}")
    print(f"[A3] Loaded checkpoint: {checkpoint_path} (missing non-trainable: {len(incompatible.missing_keys) - len(missing)})")


def main():
    ap = argparse.ArgumentParser(description="Phase A3 - bias dominance diagnostic")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Experiment 0 checkpoint path (required unless --random-init).")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--split", type=str, default="data/splits/scenario3_drug_strict_60_20_20.csv")
    ap.add_argument("--split-value", type=int, default=2, help="split column value to evaluate (2=val, 3=test)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--random-init", action="store_true", help="Use untrained model (sanity check only).")
    args = ap.parse_args()

    if not args.random_init and not args.checkpoint:
        print("[A3] ERROR: provide --checkpoint or --random-init")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m_config, _t_config = load_config(args.config)

    cell_features = load_nci60_gex("data/features/NCI-60_landmark_gex.csv", target_dim=m_config.cell_in_dim)
    if not cell_features:
        print("[A3] ERROR: gene expression features not found.")
        return 1

    if not os.path.exists(args.split):
        print(f"[A3] ERROR: split file not found: {args.split}")
        return 1
    split_df = pd.read_csv(args.split)
    val_df = split_df[split_df["split"] == args.split_value].copy()
    print(f"[A3] Evaluating split={args.split_value} on {args.split}: {len(val_df)} rows")

    val_records = parse_dataframe_to_records(val_df, known_gex_dict=cell_features)
    if not val_records:
        print("[A3] ERROR: no parseable validation records.")
        return 1
    if args.max_batches:
        val_records = val_records[: args.max_batches * args.batch_size]

    drug_features = load_precomputed_drug_features("data/features/drug_features.pt")
    if not drug_features:
        drug_features = load_precomputed_drug_features("data/features/morgan_rdkit_only/drug_features_morgan_rdkit.pt")
    if not drug_features:
        drug_features = load_precomputed_drug_features("data/features/morgan_rdkit_only/drug_features_morgan_rdkit.pkl")
    if drug_features:
        print(f"[A3] Using precomputed drug features for {len(drug_features)} SMILES.")
    else:
        print("[A3] WARNING: no precomputed features; on-the-fly RDKit preprocessing (slow).")

    dataset = DrugComboDataset(
        val_records, cell_features, drug_feature_store=drug_features,
        use_pretrained_molformer=m_config.use_pretrained_molformer,
        molformer_model_name=m_config.molformer_model_name,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = CancerCombo(m_config).to(device)
    if args.checkpoint:
        load_state_into(model, args.checkpoint, device)
    else:
        print("[A3] WARNING: running with RANDOM INIT weights (sanity check only, NOT Experiment 0).")

    model.eval()

    hill_abs = []
    bias_abs = []
    pred_abs = []
    targ_abs = []
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            da_m, db_m = b.get("drug_a_morgan"), b.get("drug_b_morgan")
            da_d, db_d = b.get("drug_a_desc"), b.get("drug_b_desc")
            cell = b.get("cell_line")
            doses_a, doses_b = b.get("doses_a"), b.get("doses_b")
            target = b.get("viability")

            y_pred, params = model(
                drug_a_morgan=da_m, drug_a_desc=da_d,
                drug_b_morgan=db_m, drug_b_desc=db_d,
                cell_line=cell, doses_a=doses_a, doses_b=doses_b,
            )
            e1, e2, e3, log_c1, log_c2, h1, h2, alpha = params
            y_hill = model.hill_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
            bias = y_pred - y_hill

            hill_abs.append(y_hill.abs().cpu().numpy())
            bias_abs.append(bias.abs().cpu().numpy())
            pred_abs.append(y_pred.abs().cpu().numpy())
            targ_abs.append(target.abs().cpu().numpy())
            n_batches += 1

    hill_abs = np.concatenate(hill_abs)
    bias_abs = np.concatenate(bias_abs)
    pred_abs = np.concatenate(pred_abs)
    targ_abs = np.concatenate(targ_abs)

    def pct(x):
        q = np.percentile(x, [5, 25, 50, 75, 95])
        return (f"mean={x.mean():8.3f}  p5={q[0]:8.3f} p25={q[1]:8.3f} "
                f"p50={q[2]:8.3f} p75={q[3]:8.3f} p95={q[4]:8.3f}")

    print("\n" + "=" * 78)
    print("PHASE A3 REPORT - BIAS vs HILL COMPONENT (validation data)")
    print("=" * 78)
    print(f"components  : {len(hill_abs)}  (batches={n_batches}, device={device})")
    print(f"  mean_abs_y_hill   : {pct(hill_abs)}")
    print(f"  mean_abs_bias     : {pct(bias_abs)}")
    print(f"  mean_abs_y_pred   : {pct(pred_abs)}")
    print(f"  mean_abs_target   : {pct(targ_abs)}")
    print()
    ratio_bh = np.nan_to_num(bias_abs / np.where(hill_abs > 1e-12, hill_abs, np.nan), nan=0.0, posinf=0.0)
    ratio_bp = np.nan_to_num(bias_abs / np.where(pred_abs > 1e-12, pred_abs, np.nan), nan=0.0, posinf=0.0)
    print(f"  bias/hill ratio        : mean={ratio_bh.mean():.4f}  median={np.median(ratio_bh):.4f}  "
          f"p95={np.percentile(ratio_bh, 95):.4f}  frac(bias>hill)={(bias_abs > hill_abs).mean():.4f}")
    print(f"  bias/prediction ratio  : mean={ratio_bp.mean():.4f}  median={np.median(ratio_bp):.4f}  "
          f"p95={np.percentile(ratio_bp, 95):.4f}  frac(bias>0.5*pred)={(bias_abs > 0.5 * pred_abs).mean():.4f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())