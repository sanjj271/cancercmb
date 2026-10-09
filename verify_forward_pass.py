import torch
from config import load_config
from cancercombo import CancerCombo

def test_forward():
    print("=" * 60)
    print("RUNNING FORWARD PASS VERIFICATION FOR ABLATION 2 + DRUG-DRUG ATTENTION")
    print("=" * 60)
    
    # Load config
    model_config, _ = load_config("config.yaml")
    model = CancerCombo(model_config)
    model.eval()
    
    # Dummy Batch Inputs (Batch Size = 4)
    B = 4
    drug_a_morgan = torch.randn(B, 2048)
    drug_a_desc = torch.randn(B, 200)
    drug_b_morgan = torch.randn(B, 2048)
    drug_b_desc = torch.randn(B, 200)
    cell_line = torch.randn(B, 976)
    doses_a = torch.tensor([[0.0, 0.1, 1.0, 10.0]] * B)
    doses_b = torch.tensor([[0.0, 0.2, 2.0, 20.0]] * B)
    
    # Step-by-step trace matching CancerCombo.forward()
    with torch.no_grad():
        morgan_a = model.morgan_enc(drug_a_morgan)
        desc_a = model.descriptor_enc(drug_a_desc)
        fused_a = model.fusion(morgan_a, desc_a)

        morgan_b = model.morgan_enc(drug_b_morgan)
        desc_b = model.descriptor_enc(drug_b_desc)
        fused_b = model.fusion(morgan_b, desc_b)
        
        cell_features = model.cell_enc(cell_line)
        cond_a = model.drug_cell_attn(fused_a, cell_features)
        cond_b = model.drug_cell_attn(fused_b, cell_features)
        
        # Drug-Drug Cross Attention
        aware_a, aware_b = model.drug_drug_attn(cond_a, cond_b)
        enhanced_pair_rep = torch.cat([aware_a, aware_b], dim=1)
        
        # Prediction Heads
        e1, e2, e3, log_c1, log_c2, h1, h2, alpha = model.heads(enhanced_pair_rep)
        
        # Bivariate Hill Solver
        y_pred = model.hill_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)

    print(f"Drug A Conditioned Representation Shape : {list(cond_a.shape)}")
    print(f"Drug B Conditioned Representation Shape : {list(cond_b.shape)}")
    print(f"Drug–Drug Attention Input Shape        : {list(cond_a.shape)} & {list(cond_b.shape)}")
    print(f"Drug–Drug Attention Output Shape       : {list(aware_a.shape)} & {list(aware_b.shape)}")
    print(f"Concatenated Representation Shape       : {list(enhanced_pair_rep.shape)}")
    print(f"Prediction Head Input Shape             : {list(enhanced_pair_rep.shape)}")
    print(f"Prediction Output Shape                 : e1: {list(e1.shape)}, Matrix y_pred: {list(y_pred.shape)}")
    print("Forward Pass Successful: TRUE")
    print("=" * 60)

if __name__ == "__main__":
    test_forward()
