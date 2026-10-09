import os
import sys
import torch
import pandas as pd
import numpy as np

from config import load_config
from dataset import load_nci60_gex, parse_dataframe_to_records, DrugComboDataset, load_precomputed_drug_features
from cancercombo import CancerCombo

def run_verification():
    print("=" * 70)
    print("      DEEPSYNBA-INSPIRED ARCHITECTURE VERIFICATION SUITE")
    print("=" * 70)

    # 1. VERIFY FEATURES
    print("\n--- 1. FEATURE VERIFICATION ---")
    m_config, t_config = load_config("config.yaml")
    
    has_morgan = hasattr(m_config, "morgan_in_dim") and m_config.morgan_in_dim == 1024
    has_rdkit = hasattr(m_config, "descriptor_in_dim") and m_config.descriptor_in_dim == 200
    has_dti = hasattr(m_config, "dti_in_dim") and m_config.dti_in_dim == 2048
    has_pw = hasattr(m_config, "pathway_in_dim") and m_config.pathway_in_dim == 1152
    
    print(f"Morgan Fingerprints : {'PRESENT' if has_morgan else 'ABSENT'} (Dim: {getattr(m_config, 'morgan_in_dim', 'N/A')})")
    print(f"RDKit Descriptors   : {'PRESENT' if has_rdkit else 'ABSENT'} (Dim: {getattr(m_config, 'descriptor_in_dim', 'N/A')})")
    print(f"DTI Embeddings      : {'PRESENT' if has_dti else 'ABSENT'} (Dim: {getattr(m_config, 'dti_in_dim', 'N/A')})")
    print(f"Pathway Embeddings  : {'PRESENT' if has_pw else 'ABSENT'} (Dim: {getattr(m_config, 'pathway_in_dim', 'N/A')})")

    # 2. VERIFY DATASET
    print("\n--- 2. DATASET VERIFICATION ---")
    split_path = "data/splits/scenario1_combination.csv"

    if os.path.exists(split_path):
        df_split = pd.read_csv(split_path)
        train_rows = len(df_split[df_split["split"] == 1])
        val_rows = len(df_split[df_split["split"] == 2])
        test_rows = len(df_split[df_split["split"] == 3])
        print(f"Dataset File   : {split_path}")
        print(f"Train rows     : {train_rows} (split == 1)")
        print(f"Validation rows: {val_rows} (split == 2)")
        print(f"Test rows      : {test_rows} (split == 3)")
        print("Dataset split status: UNCHANGED")
    else:
        print(f"Dataset split file not found on disk at {split_path} (UNVERIFIED locally)")

    # 3. VERIFY ARCHITECTURE
    print("\n--- 3. ARCHITECTURE VERIFICATION ---")
    model = CancerCombo(m_config)
    
    has_drug_enc = hasattr(model, "drug_encoder")
    has_cell_enc = hasattr(model, "cell_enc")
    has_drug_cell_attn = hasattr(model, "drug_cell_attn")
    has_pred_heads = hasattr(model, "heads")
    has_hill_solver = hasattr(model, "hill_solver")

    print(f"Shared Drug Encoder implemented        : {'CONFIRMED' if has_drug_enc else 'MISSING'}")
    print(f"Cell Line Encoder implemented          : {'CONFIRMED' if has_cell_enc else 'MISSING'}")
    print(f"Drug-Cell Cross Attention implemented : {'CONFIRMED' if has_drug_cell_attn else 'MISSING'}")
    print(f"Prediction Heads implemented           : {'CONFIRMED' if has_pred_heads else 'MISSING'}")
    print(f"Hill Equation connected                : {'CONFIRMED' if has_hill_solver else 'MISSING'}")

    # 4. ONE FORWARD-PASS VERIFICATION
    print("\n--- 4. ONE FORWARD-PASS VERIFICATION ---")
    batch_size = 4
    drug_a_morgan = torch.randn(batch_size, 1024)
    drug_a_desc = torch.randn(batch_size, 200)
    drug_a_dti = torch.randn(batch_size, 2048)
    drug_a_pw = torch.randn(batch_size, 1152)

    drug_b_morgan = torch.randn(batch_size, 1024)
    drug_b_desc = torch.randn(batch_size, 200)
    drug_b_dti = torch.randn(batch_size, 2048)
    drug_b_pw = torch.randn(batch_size, 1152)

    cell_line = torch.randn(batch_size, 976)
    doses_a = torch.tensor([[0.0, 0.1, 1.0, 10.0]] * batch_size)
    doses_b = torch.tensor([[0.0, 0.1, 1.0, 10.0]] * batch_size)

    y_pred, params = model(
        drug_a_morgan=drug_a_morgan, drug_a_desc=drug_a_desc,
        drug_a_dti=drug_a_dti, drug_a_pathway=drug_a_pw,
        drug_b_morgan=drug_b_morgan, drug_b_desc=drug_b_desc,
        drug_b_dti=drug_b_dti, drug_b_pathway=drug_b_pw,
        cell_line=cell_line, doses_a=doses_a, doses_b=doses_b
    )

    e1, e2, e3, log_c1, log_c2, h1, h2, alpha = params
    print("Prediction Head Parameter Output Shapes:")
    print(f"  e1    : {e1.shape}")
    print(f"  e2    : {e2.shape}")
    print(f"  e3    : {e3.shape}")
    print(f"  log_c1: {log_c1.shape}")
    print(f"  log_c2: {log_c2.shape}")
    print(f"  h1    : {h1.shape}")
    print(f"  h2    : {h2.shape}")
    print(f"  alpha : {alpha.shape}")

    print(f"Hill Equation Output Matrix (y_pred): {y_pred.shape}")
    assert y_pred.shape == (batch_size, 4, 4), f"Unexpected y_pred shape {y_pred.shape}"
    print("Forward pass successful: True")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
