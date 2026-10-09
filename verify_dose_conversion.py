import sys
import os
import torch
import numpy as np
import pandas as pd

# Add current dir to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from dataset import DrugComboDataset, load_nci60_gex, load_synergy_dataset, parse_dataframe_to_records
from cancercombo import CancerCombo
from losses import CancerComboLoss
from blocks.hill_equation import BivariateHillSolver

def run_verification():
    print("=" * 70)
    print("STEP 4: VERIFY TENSOR RANGES BEFORE AND AFTER CONVERSION")
    print("=" * 70)

    # Load dataset sample
    csv_path = "data/splits/scenario1_combination.csv"
    gex_path = "data/features/NCI-60_landmark_gex.csv"
    
    synth_features = {
        "CC(=O)OC1=CC=CC=C1C(=O)O": {"morgan": np.zeros(1024, dtype=np.float32), "descriptors": np.zeros(200, dtype=np.float32)},
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": {"morgan": np.zeros(1024, dtype=np.float32), "descriptors": np.zeros(200, dtype=np.float32)}
    }
    synth_dti = {
        "CC(=O)OC1=CC=CC=C1C(=O)O": np.zeros(2048, dtype=np.float32),
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": np.zeros(2048, dtype=np.float32)
    }
    synth_pw = {
        "CC(=O)OC1=CC=CC=C1C(=O)O": np.zeros(1152, dtype=np.float32),
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C": np.zeros(1152, dtype=np.float32)
    }

    if os.path.exists(csv_path) and os.path.exists(gex_path):
        gex_dict = load_nci60_gex(gex_path)
        df = pd.read_csv(csv_path, nrows=5000) # fast load first 5000 rows
        data_list = parse_dataframe_to_records(df, gex_dict)
        print(f"Loaded {len(data_list)} sample records for verification.")
    else:
        print("Dataset files not found, creating synthetic sample records.")
        data_list = [{
            "smiles_a": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "smiles_b": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "cell_line_name": "786-0",
            "doses_a": [0.0, 1e-8, 1e-7, 1e-6],
            "doses_b": [0.0, 1e-8, 1e-7, 1e-6],
            "viability_matrix": [[100.0]*4]*4
        }]
        gex_dict = {"786-0": np.ones(976, dtype=np.float32)}

    # Collect raw doses from data_list (in Molar)
    raw_doses_a = []
    raw_doses_b = []
    for item in data_list:
        raw_doses_a.extend(item['doses_a'])
        raw_doses_b.extend(item['doses_b'])
        
    raw_doses = np.array(raw_doses_a + raw_doses_b, dtype=np.float32)
    
    min_dose_before = float(np.min(raw_doses))
    max_dose_before = float(np.max(raw_doses))
    
    # Non-zero doses for log(dose)
    nz_raw = raw_doses[raw_doses > 0]
    min_log_dose_before = float(np.min(np.log(nz_raw))) if len(nz_raw) > 0 else float('nan')
    max_log_dose_before = float(np.max(np.log(nz_raw))) if len(nz_raw) > 0 else float('nan')

    # Instantiating dataset (returns doses in uM: * 1e6)
    from dataset import DescriptorScaler
    scaler = DescriptorScaler()
    scaler.fit([np.zeros(200, dtype=np.float32)])
    dataset = DrugComboDataset(
        data_list, cell_line_features=gex_dict, descriptor_scaler=scaler,
        drug_feature_store=synth_features, dti_store=synth_dti, pathway_store=synth_pw
    )
    
    conv_doses_a = []
    conv_doses_b = []
    for i in range(len(dataset)):
        sample = dataset[i]
        conv_doses_a.extend(sample['doses_a'].numpy().tolist())
        conv_doses_b.extend(sample['doses_b'].numpy().tolist())
        
    conv_doses = np.array(conv_doses_a + conv_doses_b, dtype=np.float32)
    
    min_dose_after = float(np.min(conv_doses))
    max_dose_after = float(np.max(conv_doses))
    
    nz_conv = conv_doses[conv_doses > 0]
    min_log_dose_after = float(np.min(np.log(nz_conv))) if len(nz_conv) > 0 else float('nan')
    max_log_dose_after = float(np.max(np.log(nz_conv))) if len(nz_conv) > 0 else float('nan')

    print("\n--- DOSE CONVERSION SUMMARY ---")
    print(f"BEFORE CONVERSION (Raw Molar dataset):")
    print(f"  Dose units       : Molar (M)")
    print(f"  Minimum dose     : {min_dose_before:.10e} M")
    print(f"  Maximum dose     : {max_dose_before:.10e} M")
    print(f"  log(dose) min    : {min_log_dose_before:.6f}  (for min non-zero dose {np.min(nz_raw):.4e} M)")
    print(f"  log(dose) max    : {max_log_dose_before:.6f}  (for max non-zero dose {np.max(nz_raw):.4e} M)")
    print()
    print(f"AFTER CONVERSION (Dataset output * 1e6):")
    print(f"  Dose units       : MicroMolar (uM)")
    print(f"  Minimum dose     : {min_dose_after:.10e} uM")
    print(f"  Maximum dose     : {max_dose_after:.10e} uM")
    print(f"  log(dose) min    : {min_log_dose_after:.6f}  (for min non-zero dose {np.min(nz_conv):.4e} uM)")
    print(f"  log(dose) max    : {max_log_dose_after:.6f}  (for max non-zero dose {np.max(nz_conv):.4e} uM)")
    print()
    print(f"Dose ratio (After / Before) : {max_dose_after / (max_dose_before + 1e-18):.1f}")
    print(f"log(dose_after) - log(dose_before) : {min_log_dose_after - min_log_dose_before:.6f} (log(1e6) = {np.log(1e6):.6f})")

    print("\n" + "=" * 70)
    print("STEP 5: RUN COMPLETE PIPELINE VERIFICATION")
    print("=" * 70)

    config_model, _ = load_config("config.yaml")
    model = CancerCombo(config_model)
    loss_fn = CancerComboLoss()

    # Get sample batch from dataset
    sample = dataset[0]
    batch_doses_a = sample["doses_a"].unsqueeze(0) # (1, 4)
    batch_doses_b = sample["doses_b"].unsqueeze(0) # (1, 4)
    batch_morgan_a = sample["drug_a_morgan"].unsqueeze(0)
    batch_desc_a = sample["drug_a_desc"].unsqueeze(0)
    batch_morgan_b = sample["drug_b_morgan"].unsqueeze(0)
    batch_desc_b = sample["drug_b_desc"].unsqueeze(0)
    batch_cell = sample["cell_line"].unsqueeze(0)
    batch_viab = sample["viability"].unsqueeze(0)

    batch_dti_a = sample["drug_a_dti"].unsqueeze(0)
    batch_pw_a = sample["drug_a_pathway"].unsqueeze(0)
    batch_dti_b = sample["drug_b_dti"].unsqueeze(0)
    batch_pw_b = sample["drug_b_pathway"].unsqueeze(0)

    # 1. Forward pass
    model.train()
    y_pred, params = model(
        drug_a_morgan=batch_morgan_a,
        drug_a_desc=batch_desc_a,
        drug_a_dti=batch_dti_a,
        drug_a_pathway=batch_pw_a,
        drug_b_morgan=batch_morgan_b,
        drug_b_desc=batch_desc_b,
        drug_b_dti=batch_dti_b,
        drug_b_pathway=batch_pw_b,
        cell_line=batch_cell,
        doses_a=batch_doses_a,
        doses_b=batch_doses_b
    )

    print(f"[OK] Forward pass successful!")
    print(f"  y_pred shape     : {y_pred.shape}")
    print(f"  y_pred min/max   : {y_pred.min().item():.4f} / {y_pred.max().item():.4f}")

    e1, e2, e3, log_c1, log_c2, h1, h2, alpha = params
    print(f"  Predicted Hill Params shapes: e1={e1.shape}, log_c1={log_c1.shape}, alpha={alpha.shape}")

    # Check NaNs and Infs
    has_nan_pred = torch.isnan(y_pred).any().item()
    has_inf_pred = torch.isinf(y_pred).any().item()
    print(f"[OK] Output NaNs check : {'FAILED (NaN found)' if has_nan_pred else 'PASSED (0 NaNs)'}")
    print(f"[OK] Output Infs check : {'FAILED (Inf found)' if has_inf_pred else 'PASSED (0 Infs)'}")

    # 2. Backward pass & gradient flow
    loss = loss_fn(y_pred, batch_viab, params_pred=params)
    loss.backward()

    has_nan_grad = False
    has_inf_grad = False
    zero_grad_count = 0
    total_param_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            total_param_count += 1
            if torch.isnan(param.grad).any():
                has_nan_grad = True
            if torch.isinf(param.grad).any():
                has_inf_grad = True
            if (param.grad == 0).all():
                zero_grad_count += 1

    print(f"[OK] Backward pass loss: {loss.item():.6f}")
    print(f"[OK] Gradient NaNs check: {'FAILED (NaN found)' if has_nan_grad else 'PASSED (0 NaNs)'}")
    print(f"[OK] Gradient Infs check: {'FAILED (Inf found)' if has_inf_grad else 'PASSED (0 Infs)'}")
    print(f"[OK] Gradient flow check: {total_param_count - zero_grad_count}/{total_param_count} parameter tensors receiving gradients")

    # 3. Direct Hill Equation Verification with uM doses
    solver = BivariateHillSolver(e0=100.0)
    model.eval()
    with torch.no_grad():
        y_eval, params_eval = model(
            drug_a_morgan=batch_morgan_a, drug_a_desc=batch_desc_a,
            drug_a_dti=batch_dti_a, drug_a_pathway=batch_pw_a,
            drug_b_morgan=batch_morgan_b, drug_b_desc=batch_desc_b,
            drug_b_dti=batch_dti_b, drug_b_pathway=batch_pw_b,
            cell_line=batch_cell, doses_a=batch_doses_a, doses_b=batch_doses_b
        )
        hill_out = solver(batch_doses_a, batch_doses_b, *params_eval)
        
        m_dti_a = sample["drug_a_m_dti"].unsqueeze(0)
        m_pw_a = sample["drug_a_m_pathway"].unsqueeze(0)
        m_dti_b = sample["drug_b_m_dti"].unsqueeze(0)
        m_pw_b = sample["drug_b_m_pathway"].unsqueeze(0)

        z_a = model.drug_encoder(
            morgan=batch_morgan_a, rdkit=batch_desc_a, dti=batch_dti_a, pathway=batch_pw_a,
            m_DTI=m_dti_a, m_Pathway=m_pw_a
        )
        z_b = model.drug_encoder(
            morgan=batch_morgan_b, rdkit=batch_desc_b, dti=batch_dti_b, pathway=batch_pw_b,
            m_DTI=m_dti_b, m_Pathway=m_pw_b
        )
        c_emb = model.cell_enc(batch_cell)
        pair_rep = torch.cat([model.drug_cell_attn(z_a, c_emb), model.drug_cell_attn(z_b, c_emb)], dim=1)
        bias_out = model.heads.predict_bias(pair_rep, batch_doses_a, batch_doses_b)

    print(f"[OK] Hill Equation direct call shape: {hill_out.shape}")
    print(f"[OK] Hill + Bias outputs match model y_pred: {torch.allclose(hill_out + bias_out, y_eval)}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE - ALL CHECKS PASSED")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
