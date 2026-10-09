import sys
import os
import torch
import numpy as np

# Add current dir to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from cancercombo import CancerCombo
from blocks.prediction_heads import DoseResponsePredictor
from losses import CancerComboLoss

def verify_bias_predictor_implementation():
    print("=" * 70)
    print("STEP 5 & 6: TENSOR SHAPES, GRADIENTS, AND BIAS PREDICTOR VERIFICATION")
    print("=" * 70)

    config, _ = load_config("config.yaml")
    model = CancerCombo(config)

    # Step 8 Check: Confirm bias predictor structure
    assert hasattr(model.heads, "bias_predictor1"), "bias_predictor1 missing from model.heads!"
    assert hasattr(model.heads, "bias_predictor2"), "bias_predictor2 missing from model.heads!"
    assert hasattr(model.heads, "predict_bias"), "predict_bias method missing from model.heads!"
    assert isinstance(model.heads.bias_predictor1, DoseResponsePredictor), "bias_predictor1 is not DoseResponsePredictor!"
    assert isinstance(model.heads.bias_predictor2, DoseResponsePredictor), "bias_predictor2 is not DoseResponsePredictor!"
    print("[OK] Step 8 check: Exactly two Bias Predictor modules exist in model.heads!")

    # Create dummy batch (B=4, 4x4 dose grid)
    B = 4
    drug_a_morgan = torch.randn(B, config.morgan_in_dim)
    drug_a_desc = torch.randn(B, config.descriptor_in_dim)
    drug_b_morgan = torch.randn(B, config.morgan_in_dim)
    drug_b_desc = torch.randn(B, config.descriptor_in_dim)
    cell_line = torch.randn(B, config.cell_in_dim)
    
    # MicroMolar doses
    doses_a = torch.tensor([[0.0, 0.1, 1.0, 10.0]] * B, dtype=torch.float32)
    doses_b = torch.tensor([[0.0, 0.2, 2.0, 20.0]] * B, dtype=torch.float32)
    viability_target = torch.full((B, 4, 4), 50.0, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        # Test pair_rep -> bias predictor shapes step-by-step
        morgan_a = model.morgan_enc(drug_a_morgan)
        desc_a = model.descriptor_enc(drug_a_desc)
        drug_emb_a = morgan_a + desc_a

        morgan_b = model.morgan_enc(drug_b_morgan)
        desc_b = model.descriptor_enc(drug_b_desc)
        drug_emb_b = morgan_b + desc_b

        cell_emb = model.cell_enc(cell_line)
        cond_a = model.drug_cell_attn(drug_emb_a, cell_emb)
        cond_b = model.drug_cell_attn(drug_emb_b, cell_emb)
        pair_rep = torch.cat([cond_a, cond_b], dim=1) # (B, 512)

        print(f"  pair_rep shape         : {tuple(pair_rep.shape)} (Expected: ({B}, 512))")
        assert pair_rep.shape == (B, 512)

        out1 = model.heads.bias_predictor1(pair_rep)
        out2 = model.heads.bias_predictor2(pair_rep)
        print(f"  out1 shape (predictor1): {tuple(out1.shape)} (Expected: ({B}, 4))")
        print(f"  out2 shape (predictor2): {tuple(out2.shape)} (Expected: ({B}, 4))")
        assert out1.shape == (B, 4) and out2.shape == (B, 4)

        bias = model.heads.predict_bias(pair_rep, doses_a, doses_b)
        print(f"  bias matrix shape      : {tuple(bias.shape)} (Expected: ({B}, 4, 4))")
        assert bias.shape == (B, 4, 4)

        # Full forward pass in eval mode
        y_pred, params = model(
            drug_a_morgan=drug_a_morgan,
            drug_a_desc=drug_a_desc,
            drug_b_morgan=drug_b_morgan,
            drug_b_desc=drug_b_desc,
            cell_line=cell_line,
            doses_a=doses_a,
            doses_b=doses_b
        )

        y_hill = model.hill_solver(doses_a, doses_b, *params)
        print(f"  y_hill shape           : {tuple(y_hill.shape)}")
        print(f"  y_pred shape           : {tuple(y_pred.shape)}")
        assert y_pred.shape == (B, 4, 4)
        assert torch.allclose(y_pred, y_hill + bias), "y_pred does not equal y_hill + bias in eval mode!"
        print("[OK] Step 4 check: y_pred == y_hill + bias verified mathematically in deterministic mode!")

        # Step 7 check: NaNs and Infs
        assert not torch.isnan(y_pred).any(), "NaN detected in y_pred!"
        assert not torch.isinf(y_pred).any(), "Inf detected in y_pred!"
        assert not torch.isnan(bias).any(), "NaN detected in bias!"
        assert not torch.isinf(bias).any(), "Inf detected in bias!"
        print("[OK] Step 7 check: 0 NaNs, 0 Infs detected in outputs!")

    # Step 6 check: Backward pass and gradient propagation in train mode
    model.train()
    y_pred_tr, params_tr = model(
        drug_a_morgan=drug_a_morgan,
        drug_a_desc=drug_a_desc,
        drug_b_morgan=drug_b_morgan,
        drug_b_desc=drug_b_desc,
        cell_line=cell_line,
        doses_a=doses_a,
        doses_b=doses_b
    )

    loss_fn = CancerComboLoss()
    loss = loss_fn(y_pred_tr, viability_target, params_pred=params_tr)
    loss.backward()

    # Check gradients for all critical components
    bias1_grads = [p.grad for name, p in model.heads.bias_predictor1.named_parameters() if p.requires_grad]
    bias2_grads = [p.grad for name, p in model.heads.bias_predictor2.named_parameters() if p.requires_grad]
    hill_head_grads = [p.grad for name, p in model.heads.head_e1.named_parameters() if p.requires_grad]
    drug_enc_grads = [p.grad for name, p in model.morgan_enc.named_parameters() if p.requires_grad]
    cell_enc_grads = [p.grad for name, p in model.cell_enc.named_parameters() if p.requires_grad]
    attn_grads = [p.grad for name, p in model.drug_cell_attn.named_parameters() if p.requires_grad]

    assert all(g is not None and not torch.isnan(g).any() for g in bias1_grads), "bias_predictor1 gradients invalid!"
    assert all(g is not None and not torch.isnan(g).any() for g in bias2_grads), "bias_predictor2 gradients invalid!"
    assert all(g is not None and not torch.isnan(g).any() for g in hill_head_grads), "hill head gradients invalid!"
    assert all(g is not None and not torch.isnan(g).any() for g in drug_enc_grads), "morgan_enc gradients invalid!"
    assert all(g is not None and not torch.isnan(g).any() for g in cell_enc_grads), "cell_enc gradients invalid!"
    assert all(g is not None and not torch.isnan(g).any() for g in attn_grads), "drug_cell_attn gradients invalid!"

    print("[OK] Step 6 check: Gradient flow verified across bias_predictor1, bias_predictor2, Hill heads, drug encoders, cell encoder, and cross-attention!")

    print("\n" + "=" * 70)
    print("BIAS PREDICTOR IMPLEMENTATION VERIFIED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    verify_bias_predictor_implementation()
