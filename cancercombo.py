import torch
import torch.nn as nn
from typing import Tuple, Optional

from blocks.multimodal_drug_encoder import MultimodalDrugEncoder
from blocks.cell_encoder import CellLineEncoder
from blocks.drug_cell_attention import DrugCellCrossAttention
from blocks.prediction_heads import CancerComboPredictionHeads
from blocks.hill_equation import BivariateHillSolver

class CancerCombo(nn.Module):
    """CancerCombo Architecture (Phase 3 Multimodal Shared Drug Encoder).
    
    Replaces chemistry-only drug representation with MultimodalDrugEncoder, using ONE shared
    encoder instance for both Drug A and Drug B. Only final Z_A and Z_B enter Drug-Cell Cross Attention
    to yield H_A and H_B (256-D), concatenated into H_pair (512-D).
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = getattr(config, "d_model", 256)
        n_heads = getattr(config, "n_heads", 4)
        dropout = getattr(config, "dropout", 0.1)
        
        cell_dim = getattr(config, "cell_in_dim", 976)
        n_pathways = getattr(config, "n_pathways", 300)
        use_pathway_proj = getattr(config, "use_pathway_projection", True)
        
        # 1. ONE Shared Multimodal Drug Encoder
        self.drug_encoder = MultimodalDrugEncoder(config)

        # 2. Cell Line Encoder
        self.cell_enc = CellLineEncoder(
            in_dim=cell_dim, d_model=d_model, n_pathways=n_pathways,
            use_pathway_projection=use_pathway_proj, dropout=dropout
        )

        # 3. Drug-Cell Cross Attention Module
        self.drug_cell_attn = DrugCellCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

        # 4. Prediction Heads for Bivariate Hill parameters & Dose Bias
        self.heads = CancerComboPredictionHeads(config)
        
        # 5. Bivariate Hill Solver
        self.hill_solver = BivariateHillSolver(e0=100.0)

    @property
    def morgan_enc(self):
        return self.drug_encoder.morgan_enc

    @property
    def descriptor_enc(self):
        return self.drug_encoder.descriptor_enc

    def forward(
        self,
        drug_a_ids=None, drug_a_mask=None, drug_a_morgan=None, drug_a_desc=None,
        drug_b_ids=None, drug_b_mask=None, drug_b_morgan=None, drug_b_desc=None,
        cell_line=None, doses_a=None, doses_b=None,
        drug_a_emb=None, drug_b_emb=None,
        drug_a_dti=None, drug_a_pathway=None, drug_a_m_dti=None, drug_a_m_pathway=None,
        drug_b_dti=None, drug_b_pathway=None, drug_b_m_dti=None, drug_b_m_pathway=None
    ):
        if doses_a is not None and doses_a.ndim != 2:
            raise ValueError(f"Expected doses_a tensor shape [batch_size, num_doses] (ndim=2), got shape {tuple(doses_a.shape)}")
        if doses_b is not None and doses_b.ndim != 2:
            raise ValueError(f"Expected doses_b tensor shape [batch_size, num_doses] (ndim=2), got shape {tuple(doses_b.shape)}")

        # Infer batch size and device
        if drug_a_morgan is not None:
            batch_size = drug_a_morgan.size(0)
            device = drug_a_morgan.device
        elif cell_line is not None:
            batch_size = cell_line.size(0)
            device = cell_line.device
        else:
            raise ValueError("Could not infer batch size and device from forward inputs.")

        # Check biological modality switches and enforce no silent zero-filling
        use_dti = getattr(self.config, "use_dti", True)
        use_pathway = getattr(self.config, "use_pathway", True)

        dti_dim = getattr(self.config, "dti_in_dim", 2048)
        pw_dim = getattr(self.config, "pathway_in_dim", 1152)

        if use_dti:
            if drug_a_dti is None or drug_b_dti is None:
                raise ValueError("Biological modality 'use_dti' is enabled in config, but drug_a_dti or drug_b_dti input tensor is None. Fail loudly.")
            if drug_a_m_dti is None:
                drug_a_m_dti = torch.ones(batch_size, device=device)
            if drug_b_m_dti is None:
                drug_b_m_dti = torch.ones(batch_size, device=device)
        else:
            drug_a_dti = torch.zeros(batch_size, dti_dim, device=device)
            drug_b_dti = torch.zeros(batch_size, dti_dim, device=device)
            drug_a_m_dti = torch.zeros(batch_size, device=device)
            drug_b_m_dti = torch.zeros(batch_size, device=device)

        if use_pathway:
            if drug_a_pathway is None or drug_b_pathway is None:
                raise ValueError("Biological modality 'use_pathway' is enabled in config, but drug_a_pathway or drug_b_pathway input tensor is None. Fail loudly.")
            if drug_a_m_pathway is None:
                drug_a_m_pathway = torch.ones(batch_size, device=device)
            if drug_b_m_pathway is None:
                drug_b_m_pathway = torch.ones(batch_size, device=device)
        else:
            drug_a_pathway = torch.zeros(batch_size, pw_dim, device=device)
            drug_b_pathway = torch.zeros(batch_size, pw_dim, device=device)
            drug_a_m_pathway = torch.zeros(batch_size, device=device)
            drug_b_m_pathway = torch.zeros(batch_size, device=device)

        # 1. Compute Drug A representation Z_A using shared drug encoder
        Z_A = self.drug_encoder(
            morgan=drug_a_morgan,
            rdkit=drug_a_desc,
            dti=drug_a_dti,
            pathway=drug_a_pathway,
            m_DTI=drug_a_m_dti,
            m_Pathway=drug_a_m_pathway
        ) # (B, 256)

        # 2. Compute Drug B representation Z_B using shared drug encoder
        Z_B = self.drug_encoder(
            morgan=drug_b_morgan,
            rdkit=drug_b_desc,
            dti=drug_b_dti,
            pathway=drug_b_pathway,
            m_DTI=drug_b_m_dti,
            m_Pathway=drug_b_m_pathway
        ) # (B, 256)

        # 3. Encode Cell Line transcriptomics
        cell_emb = self.cell_enc(cell_line) # (B, n_pathways, 256)

        # 4. Condition Drug A and Drug B on Cell Line via Drug-Cell Cross Attention
        cond_a = self.drug_cell_attn(Z_A, cell_emb) # (B, 256) (H_A)
        cond_b = self.drug_cell_attn(Z_B, cell_emb) # (B, 256) (H_B)

        # 5. Direct Concatenation of Conditioned Drug Representations
        pair_rep = torch.cat([cond_a, cond_b], dim=1) # (B, 512) (H_pair)

        # 6. Predict Biophysical Parameters with Prediction Heads
        e1, e2, e3, log_c1, log_c2, h1, h2, alpha = self.heads(pair_rep)

        # 7. Solve 2D Dose-Response Matrix with Bivariate Hill Equation Solver
        y_hill = self.hill_solver(doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha)

        # 8. Predict and Add Dose-Dependent Linear Bias Matrix
        bias = self.heads.predict_bias(pair_rep, doses_a, doses_b)
        y_pred = y_hill + bias

        return y_pred, (e1, e2, e3, log_c1, log_c2, h1, h2, alpha)

