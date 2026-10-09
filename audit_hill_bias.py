"""
Read-only audit: can Hill + dose-dependent bias represent responses of 0-208 %?

Run from the drugcomb (same place you run evaluate.py):

    python audit_hill_bias.py --checkpoint checkpoints/<your_run>/cancercombo_best.ckpt \
                              --split data/splits/scenario3_drug.csv --split-id 3

It does NOT change targets, the model or any file. It only prints:
  A) worked examples near 150 % and the largest target (Hill / Bias / Final / Target)
  B) a summary by target range (<=100, 100-110, 110-150, >150)
  C) saturation checks (is the bias branch dead, huge, or pinned?)
"""
import argparse
import os

for _k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import load_config
from dataset import (DrugComboDataset, load_nci60_gex,
                     parse_dataframe_to_records, load_precomputed_drug_features,
                     extract_scenario_nscs, load_descriptor_scaler)
from cancercombo import CancerCombo


def build(args):
    if not getattr(args, "scaler", None):
        raise ValueError("--scaler argument is required for audit_hill_bias.py")

    m_cfg, t_cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_df = pd.read_csv(args.split)
    train_nscs, _, _ = extract_scenario_nscs(full_df)
    scaler = load_descriptor_scaler(args.scaler, train_nsc_ids=train_nscs)

    df = full_df.copy()
    if "split" in df.columns and args.split_id is not None:
        df = df[df["split"] == args.split_id].copy()
    if args.max_rows:
        df = df.head(args.max_rows)

    cells = load_nci60_gex("data/features/NCI-60_landmark_gex.csv", target_dim=m_cfg.cell_in_dim)
    records = parse_dataframe_to_records(df, known_gex_dict=cells)

    feats = (load_precomputed_drug_features("data/features/drug_features.pt")
             or load_precomputed_drug_features("data/features/morgan_rdkit_only/drug_features_morgan_rdkit.pt")
             or load_precomputed_drug_features("data/features/morgan_rdkit_only/drug_features_morgan_rdkit.pkl"))

    ds = DrugComboDataset(records, cells, drug_feature_store=feats,
                          descriptor_scaler=scaler,
                          use_pretrained_molformer=m_cfg.use_pretrained_molformer,
                          molformer_model_name=m_cfg.molformer_model_name)
    loader = DataLoader(ds, batch_size=getattr(t_cfg, "batch_size", 128), shuffle=False)

    model = CancerCombo(m_cfg)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("model.", ""): v for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    missing = [k for k in inc.missing_keys if not k.startswith("molformer_enc.pretrained_")]
    if missing:
        raise RuntimeError(f"Checkpoint does not match this architecture; missing keys: {missing[:5]} ...")
    return model.to(device).eval(), loader, device


@torch.no_grad()
def collect(model, loader, device):
    """Re-runs the forward pass in pieces so Hill and bias are visible separately."""
    rows = []
    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        target = b.get("viability", b.get("viability_matrix"))

        # same steps as CancerCombo.forward
        da = model.morgan_enc(b["drug_a_morgan"]) + model.descriptor_enc(b["drug_a_desc"])
        db = model.morgan_enc(b["drug_b_morgan"]) + model.descriptor_enc(b["drug_b_desc"])
        cell = model.cell_enc(b["cell_line"])
        pair = torch.cat([model.drug_cell_attn(da, cell), model.drug_cell_attn(db, cell)], dim=1)

        e1, e2, e3, lc1, lc2, h1, h2, alpha = model.heads(pair)
        y_hill = model.hill_solver(b["doses_a"], b["doses_b"], e1, e2, e3, lc1, lc2, h1, h2, alpha)
        bias = model.heads.predict_bias(pair, b["doses_a"], b["doses_b"])
        # raw slopes the bias branch produced (B,4) each, before multiplying by dose
        s_a = model.heads.bias_predictor1(pair)
        s_b = model.heads.bias_predictor2(pair)
        final = y_hill + bias

        for i in range(target.shape[0]):
            rows.append(dict(
                target=target[i].cpu().numpy(), hill=y_hill[i].cpu().numpy(),
                bias=bias[i].cpu().numpy(), final=final[i].cpu().numpy(),
                dose_a=b["doses_a"][i].cpu().numpy(), dose_b=b["doses_b"][i].cpu().numpy(),
                slope_a=s_a[i].cpu().numpy(), slope_b=s_b[i].cpu().numpy(),
            ))
    return rows


def show_example(r, i, j, tag):
    print(f"  [{tag}] dose grid cell ({i},{j}): dose_a={r['dose_a'][i]:.4g} uM, dose_b={r['dose_b'][j]:.4g} uM")
    print(f"      Target={r['target'][i, j]:8.2f}   Hill={r['hill'][i, j]:8.2f}   "
          f"Bias={r['bias'][i, j]:8.2f}   Final={r['final'][i, j]:8.2f}   "
          f"(slope_a[{i}]={r['slope_a'][i]:.3g}, slope_b[{j}]={r['slope_b'][j]:.3g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True, help="split CSV (e.g. data/splits/scenario3_drug.csv)")
    ap.add_argument("--scaler", required=True, help="Path to fitted scenario DescriptorScaler (.npz)")
    ap.add_argument("--split-id", type=int, default=3, help="value of the 'split' column to audit (2=val, 3=test)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-rows", type=int, default=0, help="optional cap on CSV rows for a quick run")
    args = ap.parse_args()

    model, loader, device = build(args)
    rows = collect(model, loader, device)
    T = np.stack([r["target"] for r in rows]); H = np.stack([r["hill"] for r in rows])
    B = np.stack([r["bias"] for r in rows]);   F = np.stack([r["final"] for r in rows])
    SA = np.stack([r["slope_a"] for r in rows]); SB = np.stack([r["slope_b"] for r in rows])
    print(f"\nAudited {len(rows)} dose-response matrices ({T.size} grid points)\n")

    # ---------- A) worked examples ----------
    print("A) WORKED EXAMPLES (closest grid point to each target value)")
    for goal in (150.0, T.max()):
        flat = np.abs(T - goal).reshape(-1).argmin()
        n, i, j = np.unravel_index(flat, T.shape)
        show_example(rows[n], i, j, f"target ~ {goal:.0f}")
    print("  Reading guide: Hill ~ 90-100 with Bias ~ Target-Hill and Final ~ Target  => representable.\n"
          "                 Huge slope_*, Bias pinned far from Target, or Final << Target => architectural problem.\n")

    # ---------- B) summary by target range ----------
    print("B) SUMMARY BY TARGET RANGE")
    print(f"  {'range':>10} {'n':>9} {'mean Tgt':>9} {'mean Hill':>9} {'mean Bias':>9} {'mean Final':>10} "
          f"{'bias/(T-100)':>12} {'RMSE':>7} {'mean err':>9}")
    edges = [(-1e9, 100), (100, 110), (110, 150), (150, 1e9)]
    for lo, hi in edges:
        m = (T > lo) & (T <= hi)
        if not m.any():
            continue
        err = F[m] - T[m]
        need = np.clip(T[m] - 100, 1e-9, None)
        ratio = (B[m] / need).mean() if lo >= 100 else float("nan")
        label = f"<=100" if lo < 0 else (f">{lo:.0f}" if hi > 1e8 else f"{lo:.0f}-{hi:.0f}")
        print(f"  {label:>10} {m.sum():9d} {T[m].mean():9.2f} {H[m].mean():9.2f} {B[m].mean():9.2f} "
              f"{F[m].mean():10.2f} {ratio:12.2f} {np.sqrt((err**2).mean()):7.2f} {err.mean():9.2f}")
    print("  mean err = Final - Target; a clearly negative value in the >100 rows means the model under-reaches.\n")

    # ---------- C) saturation checks ----------
    print("C) SATURATION CHECKS")
    print(f"  Hill output max over all points        : {H.max():.2f}  (design cap is 100 = e0 with sigmoid efficacies)")
    print(f"  Hill share within 1 of the 100 cap     : {(H > 99).mean() * 100:.1f}% of points")
    zero_a = (SA == 0).mean() * 100; zero_b = (SB == 0).mean() * 100
    print(f"  Bias slopes exactly 0 (ReLU dead)      : A {zero_a:.1f}%   B {zero_b:.1f}%")
    print(f"  Bias slope magnitude (A/B)             : median {np.median(np.r_[SA.ravel(), SB.ravel()]):.3g}, "
          f"99th pct {np.percentile(np.r_[SA.ravel(), SB.ravel()], 99):.3g}, max {np.r_[SA.ravel(), SB.ravel()].max():.3g}")
    hi_pts = T > 110
    if hi_pts.any():
        print(f"  Points with target>110 where Final<100 : {(F[hi_pts] < 100).mean() * 100:.1f}%")
        print(f"  Points with target>110 where Bias==0   : {(B[hi_pts] == 0).mean() * 100:.1f}%")
    print(f"  Final < 0 anywhere                     : {(F < 0).mean() * 100:.2f}% of points")
    print("\nNothing was clipped or modified. Share the printed output and we decide what, if anything, to change.")


if __name__ == "__main__":
    main()
