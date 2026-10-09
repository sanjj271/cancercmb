import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union, List

from blocks.morgan_encoder import MorganEncoder
from blocks.descriptor_encoder import DescriptorEncoder


class DTIEncoder(nn.Module):
    """Layer-by-layer projection for frozen 2048-D pre-pooled DTI embeddings.
    
    Architecture:
      2048 -> 1024 (Linear -> GELU -> LayerNorm)
      -> 512 (Linear -> GELU -> LayerNorm)
      -> 256 (Linear -> LayerNorm)
    """
    def __init__(self, in_dim: int = 2048, hidden1: int = 1024, hidden2: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.LayerNorm(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.LayerNorm(hidden2),
            nn.Linear(hidden2, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PathwayEncoder(nn.Module):
    """Config-driven projection for frozen 1152-D precomputed pathway embeddings.
    
    Architecture:
      If hidden_dims == []:
        1152 -> out_dim (Linear -> LayerNorm)
      If hidden_dims == [1024, 512]:
        1152 -> 1024 (Linear -> GELU -> LayerNorm)
        -> 512 (Linear -> GELU -> LayerNorm)
        -> 256 (Linear -> LayerNorm)
    """
    def __init__(self, in_dim: int = 1152, hidden_dims: Optional[List[int]] = None, out_dim: int = 256):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512]

        layers = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim)
            ])
            curr_dim = h_dim

        layers.extend([
            nn.Linear(curr_dim, out_dim),
            nn.LayerNorm(out_dim)
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ChemistryAttentionPooling(nn.Module):
    """Attention pooling over Morgan and RDKit chemistry tokens [M, R].
    
    Scores:
      s_i = v_C^T tanh(W_C x_i)
      alpha = softmax(s)
      C = alpha_M * M + alpha_R * R  (C in R^256)
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.attn_proj = nn.Linear(d_model, d_model)
        self.v_attn = nn.Linear(d_model, 1, bias=False)

    def forward(self, M: torch.Tensor, R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack([M, R], dim=1) # (B, 2, d_model)
        proj = torch.tanh(self.attn_proj(tokens)) # (B, 2, d_model)
        scores = self.v_attn(proj).squeeze(-1) # (B, 2)
        weights = F.softmax(scores, dim=-1) # (B, 2)
        
        C = weights[:, 0:1] * M + weights[:, 1:2] * R
        return C, weights


class BiologyAttentionPooling(nn.Module):
    """Masked attention pooling over DTI and Pathway biology tokens [T, P].
    
    Scores:
      s_i = v_B^T tanh(W_B x_i)
      alpha_i = m_i * exp(s_i - max_valid_s) / (sum_j m_j * exp(s_j - max_valid_s) + eps)
      B = alpha_DTI * T + alpha_Pathway * P  (B in R^256)
    
    If both modalities missing (m_DTI=0, m_Pathway=0): B = 0.
    """
    def __init__(self, d_model: int = 256, eps: float = 1e-8):
        super().__init__()
        self.attn_proj = nn.Linear(d_model, d_model)
        self.v_attn = nn.Linear(d_model, 1, bias=False)
        self.eps = eps

    def forward(
        self,
        T: torch.Tensor,
        P: torch.Tensor,
        m_DTI: torch.Tensor,
        m_Pathway: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m_dti_b = m_DTI.view(-1, 1).float()
        m_path_b = m_Pathway.view(-1, 1).float()
        masks = torch.cat([m_dti_b, m_path_b], dim=1) # (B, 2)

        tokens = torch.stack([T, P], dim=1) # (B, 2, d_model)
        proj = torch.tanh(self.attn_proj(tokens)) # (B, 2, d_model)
        scores = self.v_attn(proj).squeeze(-1) # (B, 2)

        # Subtract max over VALID scores only for numerical stability
        masked_scores = scores.masked_fill(masks == 0, -1e9)
        max_scores = torch.max(masked_scores, dim=-1, keepdim=True)[0]
        max_scores = torch.where(max_scores <= -1e8, torch.zeros_like(max_scores), max_scores)

        exp_scores = torch.exp(scores - max_scores)
        masked_exp = exp_scores * masks # (B, 2)
        sum_exp = masked_exp.sum(dim=-1, keepdim=True) + self.eps # (B, 1)
        weights = masked_exp / sum_exp # (B, 2)

        both_missing = (masks.sum(dim=-1, keepdim=True) == 0) # (B, 1)
        B = weights[:, 0:1] * T + weights[:, 1:2] * P
        B = torch.where(both_missing, torch.zeros_like(B), B)

        return B, weights


class ChemistryAnchoredGatedFusion(nn.Module):
    """Fuses chemistry representation C with biology representation B using chemistry anchor.
    
    Formula:
      B_tilde = W_B * B + b_B
      g = sigmoid(W_g * [C; B; m_DTI; m_Pathway] + b_g)
      Z = C + g * B_tilde
      
    If both biological modalities missing (m_DTI=0, m_Pathway=0):
      B = 0, B_tilde = 0, Z = C (pure chemistry fallback).
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.b_transform = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2 + 2, d_model)

    def forward(
        self,
        C: torch.Tensor,
        B: torch.Tensor,
        m_DTI: torch.Tensor,
        m_Pathway: torch.Tensor
    ) -> torch.Tensor:
        m_dti_b = m_DTI.view(-1, 1).float()
        m_path_b = m_Pathway.view(-1, 1).float()
        both_missing = ((m_dti_b + m_path_b) == 0) # (B, 1)

        B_tilde = self.b_transform(B)
        B_tilde = torch.where(both_missing, torch.zeros_like(B_tilde), B_tilde)

        gate_input = torch.cat([C, B, m_dti_b, m_path_b], dim=-1) # (B, 514)
        g = torch.sigmoid(self.gate_proj(gate_input)) # (B, d_model)

        Z = C + g * B_tilde
        Z = torch.where(both_missing, C, Z)
        return Z


class MultimodalDrugEncoder(nn.Module):
    """ONE Shared Multimodal Drug Encoder processing both Drug A and Drug B.
    
    Integrates Morgan, RDKit, DTI, and Pathway representations into a unified 256-D vector Z.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = getattr(config, "d_model", 256)
        dropout = getattr(config, "dropout", 0.1)

        morgan_dim = getattr(config, "morgan_in_dim", 1024)
        desc_dim = getattr(config, "descriptor_in_dim", 200)
        dti_dim = getattr(config, "dti_in_dim", 2048)
        pathway_dim = getattr(config, "pathway_in_dim", 1152)

        pw_hidden_dims = getattr(config, "pathway_hidden_dims", [1024, 512])

        self.morgan_enc = MorganEncoder(in_dim=morgan_dim, d_model=d_model, dropout=dropout)
        self.descriptor_enc = DescriptorEncoder(in_dim=desc_dim, d_model=d_model, dropout=dropout)
        self.dti_enc = DTIEncoder(in_dim=dti_dim, hidden1=getattr(config, "dti_hidden_dim_1", 1024), hidden2=getattr(config, "dti_hidden_dim_2", 512), out_dim=d_model)
        self.pathway_enc = PathwayEncoder(in_dim=pathway_dim, hidden_dims=pw_hidden_dims, out_dim=d_model)

        self.chem_attn = ChemistryAttentionPooling(d_model=d_model)
        self.bio_attn = BiologyAttentionPooling(d_model=d_model)
        self.gated_fusion = ChemistryAnchoredGatedFusion(d_model=d_model)

        self.p_dti_drop = getattr(config, "dti_dropout", 0.20)
        self.p_pathway_drop = getattr(config, "pathway_dropout", 0.20)
        self.p_both_bio_drop = getattr(config, "both_biology_dropout", 0.10)

    def _apply_modality_dropout(
        self,
        m_DTI: torch.Tensor,
        m_Pathway: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m_dti_b = m_DTI.view(-1, 1).float()
        m_path_b = m_Pathway.view(-1, 1).float()

        # Respect use_dti and use_pathway switches
        if not getattr(self.config, "use_dti", True):
            m_dti_b = torch.zeros_like(m_dti_b)
        if not getattr(self.config, "use_pathway", True):
            m_path_b = torch.zeros_like(m_path_b)

        if not self.training:
            return m_dti_b, m_path_b

        batch_size = m_DTI.size(0)
        device = m_DTI.device

        # Draw u_both ~ U(0, 1)
        u_both = torch.rand(batch_size, 1, device=device)
        both_drop = (u_both < self.p_both_bio_drop)

        # Draw separate u_dti and u_path ~ U(0, 1)
        u_dti = torch.rand(batch_size, 1, device=device)
        u_path = torch.rand(batch_size, 1, device=device)

        dti_drop = both_drop | (~both_drop & (u_dti < self.p_dti_drop))
        path_drop = both_drop | (~both_drop & (u_path < self.p_pathway_drop))

        eff_m_dti = m_dti_b * (~dti_drop).float()
        eff_m_path = m_path_b * (~path_drop).float()

        return eff_m_dti, eff_m_path

    def forward(
        self,
        morgan: torch.Tensor,
        rdkit: torch.Tensor,
        dti: torch.Tensor,
        pathway: torch.Tensor,
        m_DTI: torch.Tensor,
        m_Pathway: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        M = self.morgan_enc(morgan)
        R = self.descriptor_enc(rdkit)
        C, chem_weights = self.chem_attn(M, R)

        T = self.dti_enc(dti)
        P = self.pathway_enc(pathway)

        eff_m_dti, eff_m_pathway = self._apply_modality_dropout(m_DTI, m_Pathway)
        B, bio_weights = self.bio_attn(T, P, eff_m_dti, eff_m_pathway)
        Z = self.gated_fusion(C, B, eff_m_dti, eff_m_pathway)

        if return_components:
            components = {
                "M": M, "R": R, "C": C,
                "T": T, "P": P, "B": B,
                "chem_weights": chem_weights,
                "bio_weights": bio_weights,
                "eff_m_dti": eff_m_dti,
                "eff_m_pathway": eff_m_pathway,
                "Z": Z
            }
            return Z, components

        return Z
