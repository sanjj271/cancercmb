import torch
import torch.nn as nn
import math

class AttentionMultiRepresentationFusion(nn.Module):
    """Dynamic multi-head self-attention fusion block for multi-modal drug representations (Morgan + RDKit descriptors)."""
    
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        
        # Learnable linear pooling layer (fuses 2 representations: Morgan Fingerprints and RDKit Descriptors)
        self.pooling = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, morgan_emb: torch.Tensor, descriptor_emb: torch.Tensor) -> torch.Tensor:
        """Fuses Morgan fingerprint and RDKit descriptor representations dynamically using manual self-attention.

        Args:
            morgan_emb: Projected Morgan fingerprint tensor of shape (B, d_model).
            descriptor_emb: Projected RDKit descriptor tensor of shape (B, d_model).

        Returns:
            torch.Tensor: Enhanced fused drug embedding of shape (B, d_model).
        """
        B = morgan_emb.size(0)
        stacked = torch.stack([morgan_emb, descriptor_emb], dim=1) # (B, seq_len=2, d_model)
        seq_len = 2
        
        Q = self.q_proj(stacked)
        K = self.k_proj(stacked)
        V = self.v_proj(stacked)
        
        Q = Q.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)
        
        # 1. Flatten and apply learnable linear pooling (matching CancerCombo1 pattern)
        flat_attn = attn_out.reshape(B, -1) # (B, 2 * d_model)
        fusion_output = self.pooling(flat_attn) # (B, d_model)

        # 2. Symmetric Residual Connection (Equal contribution from Morgan and RDKit Descriptors)
        residual = (morgan_emb + descriptor_emb) / 2.0 # (B, d_model)

        # 3. Layer Normalization
        fused = self.norm(fusion_output + residual) # (B, d_model)
        return fused
