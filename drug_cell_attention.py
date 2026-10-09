import torch
import torch.nn as nn
import math

class DrugCellCrossAttention(nn.Module):
    """Conditions drug representations on the cell line transcriptomics using cross-attention."""
    
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
        
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, drug_emb: torch.Tensor, cell_emb: torch.Tensor) -> torch.Tensor:
        B = drug_emb.size(0)
        q = drug_emb.unsqueeze(1)
        Q = self.q_proj(q)
        K = self.k_proj(cell_emb)
        V = self.v_proj(cell_emb)
        
        Q = Q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        attn_out = self.out_proj(attn_out)
        
        x = self.norm(q + attn_out)
        out = self.ffn(x) + x
        return out.squeeze(1)
