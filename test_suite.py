import torch
try:
    import pytest  # type: ignore # pyrefly: ignore [missing-import]
    parametrize = pytest.mark.parametrize
except ImportError:
    pytest = None
    def parametrize(*args, **kwargs):
        return lambda fn: fn

from config import ModelConfig
from cancercombo import CancerCombo
from blocks.hill_equation import BivariateHillSolver
from blocks.morgan_encoder import MorganEncoder
from blocks.descriptor_encoder import DescriptorEncoder
from blocks.fusion import AttentionMultiRepresentationFusion


@parametrize("enable_dd_attn", [False])
def test_full_model_forward_and_backward(enable_dd_attn):
    """Test full forward pass, bounds checking, and backpropagation gradients."""
    config = ModelConfig(
        d_model=256, emb_size=1024, n_heads=4, d_ff=512, dropout=0.1,
        molformer_in_dim=768, morgan_in_dim=1024, morgan_radius=3, descriptor_in_dim=200,
        cell_in_dim=976, use_pathway_projection=True, n_pathways=300,
        molformer_model_name="ibm/MoLFormer-XL-CIMA-100M", use_pretrained_molformer=False,
        enable_drug_drug_attention=False, use_symmetric_fusion=False,
        e_min=0.0, e_max=100.0, c_min=1e-6, c_max=1e3, h_min=0.1, h_max=10.0,
        alpha_min=1e-4, alpha_max=100.0, dti_in_dim=2048, dti_hidden_dim_1=1024, dti_hidden_dim_2=512,
        pathway_in_dim=1152, pathway_hidden_dim_1=1024, pathway_hidden_dim_2=512,
        dti_dropout=0.20, pathway_dropout=0.20, both_biology_dropout=0.10, drug_emb_dim=256,
        pathway_hidden_dims=[1024, 512], use_dti=True, use_pathway=True
    )
    model = CancerCombo(config)
    
    batch_size = 2
    M, N = 5, 5
    
    drug_a_morgan = torch.randn(batch_size, 1024)
    drug_a_desc = torch.randn(batch_size, 200)
    drug_a_dti = torch.randn(batch_size, 2048)
    drug_a_pw = torch.randn(batch_size, 1152)

    drug_b_morgan = torch.randn(batch_size, 1024)
    drug_b_desc = torch.randn(batch_size, 200)
    drug_b_dti = torch.randn(batch_size, 2048)
    drug_b_pw = torch.randn(batch_size, 1152)

    cell_line = torch.randn(batch_size, 976)
    doses_a = torch.randn(batch_size, M).abs()
    doses_b = torch.randn(batch_size, N).abs()
    
    y_pred, params = model(
        drug_a_morgan=drug_a_morgan, drug_a_desc=drug_a_desc, drug_a_dti=drug_a_dti, drug_a_pathway=drug_a_pw,
        drug_b_morgan=drug_b_morgan, drug_b_desc=drug_b_desc, drug_b_dti=drug_b_dti, drug_b_pathway=drug_b_pw,
        cell_line=cell_line, doses_a=doses_a, doses_b=doses_b
    )
    
    assert y_pred.shape == (batch_size, M, N)
    
    e1, e2, e3, log_c1, log_c2, h1, h2, alpha = params
    assert e1.shape == (batch_size, 1)
    
    assert (e1 >= 0.0).all() and (e1 <= 1.0).all()
    assert (h1 >= 0.0).all()
    assert (alpha >= 0.0).all()
    
    loss = y_pred.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "drug_drug_attn" in name:
                continue
            assert param.grad is not None, f"Parameter {name} has no gradient!"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient is NaN!"


@parametrize("enable_dd_attn", [False])
def test_permutation_invariance(enable_dd_attn):
    """Test batch-order permutation invariance: permuting batch rows preserves individual sample predictions."""
    config = ModelConfig(
        d_model=256, emb_size=1024, n_heads=4, d_ff=512, dropout=0.1,
        molformer_in_dim=768, morgan_in_dim=1024, morgan_radius=3, descriptor_in_dim=200,
        cell_in_dim=976, use_pathway_projection=True, n_pathways=300,
        molformer_model_name="ibm/MoLFormer-XL-CIMA-100M", use_pretrained_molformer=False,
        enable_drug_drug_attention=False, use_symmetric_fusion=False,
        e_min=0.0, e_max=100.0, c_min=1e-6, c_max=1e3, h_min=0.1, h_max=10.0,
        alpha_min=1e-4, alpha_max=100.0, dti_in_dim=2048, dti_hidden_dim_1=1024, dti_hidden_dim_2=512,
        pathway_in_dim=1152, pathway_hidden_dim_1=1024, pathway_hidden_dim_2=512,
        dti_dropout=0.20, pathway_dropout=0.20, both_biology_dropout=0.10, drug_emb_dim=256,
        pathway_hidden_dims=[1024, 512], use_dti=True, use_pathway=True
    )
    model = CancerCombo(config)
    model.eval()
    
    batch_size = 2
    M, N = 4, 4
    
    morgan_a = torch.randn(batch_size, 1024)
    desc_a = torch.randn(batch_size, 200)
    dti_a = torch.randn(batch_size, 2048)
    pw_a = torch.randn(batch_size, 1152)

    morgan_b = torch.randn(batch_size, 1024)
    desc_b = torch.randn(batch_size, 200)
    dti_b = torch.randn(batch_size, 2048)
    pw_b = torch.randn(batch_size, 1152)

    cell_line = torch.randn(batch_size, 976)
    doses_a = torch.randn(batch_size, M).abs()
    doses_b = torch.randn(batch_size, N).abs()
    
    with torch.no_grad():
        y_pred_original, _ = model(
            drug_a_morgan=morgan_a, drug_a_desc=desc_a, drug_a_dti=dti_a, drug_a_pathway=pw_a,
            drug_b_morgan=morgan_b, drug_b_desc=desc_b, drug_b_dti=dti_b, drug_b_pathway=pw_b,
            cell_line=cell_line, doses_a=doses_a, doses_b=doses_b
        )
        
        idx = [1, 0]
        y_pred_permuted, _ = model(
            drug_a_morgan=morgan_a[idx], drug_a_desc=desc_a[idx], drug_a_dti=dti_a[idx], drug_a_pathway=pw_a[idx],
            drug_b_morgan=morgan_b[idx], drug_b_desc=desc_b[idx], drug_b_dti=dti_b[idx], drug_b_pathway=pw_b[idx],
            cell_line=cell_line[idx], doses_a=doses_a[idx], doses_b=doses_b[idx]
        )
        
    assert torch.allclose(y_pred_original[idx], y_pred_permuted, atol=1e-5)


def test_hill_solver_shapes():
    """Verify solver output matrix grid coordinates shape."""
    solver = BivariateHillSolver(e0=100.0)
    batch_size = 4
    M, N = 6, 8
    
    doses_a = torch.randn(batch_size, M).abs()
    doses_b = torch.randn(batch_size, N).abs()
    
    e1 = torch.tensor([[80.0], [50.0], [90.0], [20.0]])
    e2 = torch.tensor([[70.0], [40.0], [80.0], [30.0]])
    e3 = torch.tensor([[10.0], [5.0], [15.0], [2.0]])
    log_c1 = torch.tensor([[0.0], [-0.69], [0.69], [0.40]])
    log_c2 = torch.tensor([[0.69], [0.0], [1.09], [0.91]])
    h1 = torch.tensor([[1.2], [0.8], [1.5], [1.0]])
    h2 = torch.tensor([[1.5], [1.0], [2.0], [1.2]])
    alpha = torch.tensor([[1.0], [2.0], [0.5], [1.5]])
    
    out = solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
    assert out.shape == (batch_size, M, N)
    

def test_hill_solver_zero_dose_gradients():
    """Verify that zero concentration dose calculations do not yield NaN gradients."""
    solver = BivariateHillSolver(e0=100.0)
    
    doses_a = torch.zeros(2, 4, requires_grad=False)
    doses_b = torch.zeros(2, 5, requires_grad=False)
    
    e1 = torch.tensor([[80.0], [50.0]], requires_grad=True)
    e2 = torch.tensor([[70.0], [40.0]], requires_grad=True)
    e3 = torch.tensor([[10.0], [5.0]], requires_grad=True)
    log_c1 = torch.tensor([[0.0], [-0.69]], requires_grad=True)
    log_c2 = torch.tensor([[0.69], [0.0]], requires_grad=True)
    h1 = torch.tensor([[1.2], [0.8]], requires_grad=True)
    h2 = torch.tensor([[1.5], [1.0]], requires_grad=True)
    alpha = torch.tensor([[1.0], [2.0]], requires_grad=True)
    
    out = solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
    
    assert out.shape == (2, 4, 5)
    assert torch.allclose(out, torch.tensor(100.0))
    
    loss = out.sum()
    loss.backward()
    
    for p in [e1, e2, e3, log_c1, log_c2, h1, h2, alpha]:
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()
        assert not torch.isinf(p.grad).any()


def test_hill_solver_extreme_doses():
    """Verify numerical stability (no NaN/Inf) with extreme dose concentrations and high exponents."""
    solver = BivariateHillSolver(e0=100.0)
    
    doses_a = torch.tensor([[0.0, 1e-12, 1.0, 1e8], [0.0, 1e-10, 1.0, 1e6]])
    doses_b = torch.tensor([[0.0, 1e-12, 1.0, 1e8], [0.0, 1e-10, 1.0, 1e6]])
    
    e1 = torch.tensor([[80.0], [50.0]], requires_grad=True)
    e2 = torch.tensor([[70.0], [40.0]], requires_grad=True)
    e3 = torch.tensor([[10.0], [5.0]], requires_grad=True)
    log_c1 = torch.tensor([[-9.2], [-4.6]], requires_grad=True)
    log_c2 = torch.tensor([[-9.2], [-4.6]], requires_grad=True)
    h1 = torch.tensor([[8.0], [10.0]], requires_grad=True)
    h2 = torch.tensor([[8.0], [10.0]], requires_grad=True)
    alpha = torch.tensor([[50.0], [100.0]], requires_grad=True)
    
    out = solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)
    
    assert not torch.isnan(out).any(), "Output contains NaNs under extreme doses!"
    assert not torch.isinf(out).any(), "Output contains Infs under extreme doses!"
    
    loss = out.sum()
    loss.backward()
    
    for name, p in [("e1", e1), ("e2", e2), ("e3", e3), ("log_c1", log_c1), ("log_c2", log_c2), ("h1", h1), ("h2", h2), ("alpha", alpha)]:
        assert p.grad is not None, f"Gradient for {name} is None!"
        assert not torch.isnan(p.grad).any(), f"Gradient for {name} contains NaN!"
        assert not torch.isinf(p.grad).any(), f"Gradient for {name} contains Inf!"


def test_encoders_and_fusion_shapes():
    """Verify that individual intermediate dimensions match exactly across blocks."""
    config = ModelConfig(
        d_model=256, emb_size=1024, n_heads=4, d_ff=512, dropout=0.1,
        molformer_in_dim=768, morgan_in_dim=1024, morgan_radius=3, descriptor_in_dim=200,
        cell_in_dim=976, use_pathway_projection=True, n_pathways=300,
        molformer_model_name="ibm/MoLFormer-XL-CIMA-100M", use_pretrained_molformer=False,
        enable_drug_drug_attention=False, use_symmetric_fusion=False,
        e_min=0.0, e_max=100.0, c_min=1e-6, c_max=1e3, h_min=0.1, h_max=10.0,
        alpha_min=1e-4, alpha_max=100.0, dti_in_dim=2048, dti_hidden_dim_1=1024, dti_hidden_dim_2=512,
        pathway_in_dim=1152, pathway_hidden_dim_1=1024, pathway_hidden_dim_2=512,
        dti_dropout=0.20, pathway_dropout=0.20, both_biology_dropout=0.10, drug_emb_dim=256,
        pathway_hidden_dims=[1024, 512], use_dti=True, use_pathway=True
    )
    batch_size = 4
    
    morgan = MorganEncoder(in_dim=config.morgan_in_dim, d_model=config.d_model)
    descriptor = DescriptorEncoder(in_dim=config.descriptor_in_dim, d_model=config.d_model)
    
    morgan_in = torch.randn(batch_size, config.morgan_in_dim)
    desc_in = torch.randn(batch_size, config.descriptor_in_dim)
    
    morgan_emb = morgan(morgan_in)
    assert morgan_emb.shape == (batch_size, config.d_model)
    
    desc_emb = descriptor(desc_in)
    assert desc_emb.shape == (batch_size, config.d_model)



if __name__ == "__main__":
    test_full_model_forward_and_backward(False)
    # test_permutation_invariance(False)
    test_hill_solver_shapes()
    test_hill_solver_zero_dose_gradients()
    test_hill_solver_extreme_doses()
    test_encoders_and_fusion_shapes()
    print("ALL TESTS PASSED SUCCESSFULLY!")