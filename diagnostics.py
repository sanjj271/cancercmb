import os
import sys
import time
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import multiprocessing

# Set thread limits
for _k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "1"

from helpers import enforce_single_thread
enforce_single_thread()

from config import load_config
from dataset import DrugComboDataset, load_nci60_gex, parse_dataframe_to_records, load_precomputed_drug_features
from cancercombo import CancerCombo
from losses import CancerComboLoss

# Disable SDPA optimized kernels globally
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

def run_test_case(test_id, device):
    """Subprocess target that runs a specific forward-backward pass configurations."""
    try:
        # Load configuration
        m_config, t_config = load_config("config.yaml")
        
        # Construct synthetic tensor batch for fast autograd diagnostic testing
        batch_size = 4
        M, N = 4, 4
        b_gpu = {
            "drug_a_ids": torch.randint(1, 10, (batch_size, 128), device=device),
            "drug_a_mask": torch.ones(batch_size, 128, device=device),
            "drug_a_morgan": torch.randn(batch_size, m_config.morgan_in_dim, device=device),
            "drug_a_desc": torch.randn(batch_size, m_config.descriptor_in_dim, device=device),
            "drug_b_ids": torch.randint(1, 10, (batch_size, 128), device=device),
            "drug_b_mask": torch.ones(batch_size, 128, device=device),
            "drug_b_morgan": torch.randn(batch_size, m_config.morgan_in_dim, device=device),
            "drug_b_desc": torch.randn(batch_size, m_config.descriptor_in_dim, device=device),
            "cell_line": torch.randn(batch_size, m_config.cell_in_dim, device=device),
            "doses_a": torch.randn(batch_size, M, device=device).abs(),
            "doses_b": torch.randn(batch_size, N, device=device).abs(),
            "viability": torch.randn(batch_size, M, N, device=device).abs()
        }

        
        # Initialize model
        net = CancerCombo(m_config).to(device)
        loss_fn = CancerComboLoss()
        
        # Apply ablation based on test_id
        if test_id == "Bypass_Hill_Solver":
            # Replace hill solver step with a simple linear projection to predict viability
            original_hill = net.hill_solver
            class MockHillSolver(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.proj = nn.Linear(m_config.d_model, 16)
                def forward(self, *args, **kwargs):
                    # args[2] is z_combo (or we can just project a combination embedding)
                    # Let's project z_combo to (B, 4, 4)
                    return self.proj(args[0]).view(-1, 4, 4) # fallback mapping
            # Replace head output and solver
            net.forward = lambda drug_a_ids, drug_a_mask, drug_a_morgan, drug_a_desc, \
                                 drug_b_ids, drug_b_mask, drug_b_morgan, drug_b_desc, \
                                 cell_line, doses_a, doses_b: \
                (MockHillSolver().to(device)(net.symmetric_fusion(
                    net.drug_cell_attn(net.fusion(net.molformer_enc(drug_a_ids, drug_a_mask)[1], net.morgan_enc(drug_a_morgan), net.descriptor_enc(drug_a_desc)), net.cell_enc(cell_line)),
                    net.drug_cell_attn(net.fusion(net.molformer_enc(drug_b_ids, drug_b_mask)[1], net.morgan_enc(drug_b_morgan), net.descriptor_enc(drug_b_desc)), net.cell_enc(cell_line))
                )), None)
                
        elif test_id == "Bypass_Transformer_Encoder":
            # Bypass LocalTransformerEncoder entirely inside MolFormerEncoder
            # We mock the forward pass of molformer_enc to return direct embedding
            original_forward = net.molformer_enc.forward
            def mock_molformer_forward(input_ids, attention_mask):
                seq_feats = net.molformer_enc.embedding(input_ids)
                pooled = seq_feats.mean(dim=1)
                return seq_feats, pooled
            net.molformer_enc.forward = mock_molformer_forward
            
        elif test_id == "Bypass_Cross_Attentions":
            # Bypass both Drug-Cell Cross Attention and Drug-Drug Attention
            net.drug_cell_attn.forward = lambda x, y: x
            if hasattr(net, "drug_drug_attn"):
                net.drug_drug_attn.forward = lambda x, y: (x, y)
            
        elif test_id == "Disable_Ranking_Loss":
            # Set ranking loss weight lambda to 0
            loss_fn.rank_lambda = 0.0
            
        elif test_id == "Pure_Linear_Baseline":
            # Pure linear baseline from raw inputs (no encoders, no solver)
            class PureLinearModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.proj = nn.Linear(m_config.morgan_in_dim * 2, 16)
                def forward(self, *args):
                    # concatenate morgan fingerprints
                    combined = torch.cat([args[2], args[6]], dim=-1)
                    return self.proj(combined).view(-1, 4, 4), None
            net = PureLinearModel().to(device)
            
        # Run forward pass
        y_pred, params = net(
            b_gpu["drug_a_ids"], b_gpu["drug_a_mask"], b_gpu["drug_a_morgan"], b_gpu["drug_a_desc"],
            b_gpu["drug_b_ids"], b_gpu["drug_b_mask"], b_gpu["drug_b_morgan"], b_gpu["drug_b_desc"],
            b_gpu["cell_line"], b_gpu["doses_a"], b_gpu["doses_b"]
        )
        
        # Run backward pass
        loss = loss_fn(y_pred, b_gpu.get("viability", b_gpu.get("viability_matrix")), params)
        loss.backward()
        
        # Finished successfully
        sys.exit(0)
    except Exception as e:
        print(f"Exception in process: {e}")
        sys.exit(1)

def run_with_timeout(test_id, device, timeout=10.0):
    """Runs a test case in a separate process with a timeout."""
    p = multiprocessing.Process(target=run_test_case, args=(test_id, device))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return "HUNG/TIMED_OUT"
    else:
        return "SUCCESS" if p.exitcode == 0 else "FAILED"

def main():
    print("="*80)
    print(" CANCERCOMBO GPU AUTOGRAD DIAGNOSTIC ENGINE")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device.upper()}")
    
    if device == "cpu":
        print("Warning: Diagnostics should be run on GPU (where the hang occurs). Running on CPU instead.")
        
    tests = [
        ("Pure_Linear_Baseline", "Tests standard PyTorch backward logic on GPU (sanity check)"),
        ("Disable_Ranking_Loss", "Tests the model backprop *without* the ranking loss computation"),
        ("Bypass_Hill_Solver", "Tests the model backprop *without* the Bivariate Hill Solver module"),
        ("Bypass_Transformer_Encoder", "Tests the model backprop *without* the LocalTransformerEncoder module"),
        ("Bypass_Cross_Attentions", "Tests the model backprop *without* the Drug-Cell/Drug-Drug Attention modules"),
        ("Full_Model", "Tests the full unmodified model backward pass")
    ]
    
    results = {}
    for test_id, desc in tests:
        print(f"\nRunning test: {test_id}...")
        print(f"  Description: {desc}")
        sys.stdout.flush()
        
        status = run_with_timeout(test_id, device, timeout=12.0)
        results[test_id] = status
        print(f"  Result: {status}")
        sys.stdout.flush()
        
    print("\n" + "="*80)
    print(" DIAGNOSTIC SUMMARY")
    print("="*80)
    for test_id, status in results.items():
        print(f"{test_id:30s} : {status}")
    print("="*80)

if __name__ == "__main__":
    # Multiprocessing setup for spawn
    multiprocessing.set_start_method('spawn', force=True)
    main()
