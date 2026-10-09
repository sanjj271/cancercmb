############################################################
# NEW CODE - DRUG–DRUG ATTENTION
############################################################
import torch
import torch.nn as nn
from typing import Tuple

class DrugDrugCrossAttention(nn.Module):
    """Mutually cross-attends conditioned Drug A and Drug B representations."""
    
    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, cond_a: torch.Tensor, cond_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cond_a: Drug A representation (B, d_model)
            cond_b: Drug B representation (B, d_model)
            
        Returns:
            aware_a: Cross-attended Drug A representation (B, d_model)
            aware_b: Cross-attended Drug B representation (B, d_model)
        """
        q_a = cond_a.unsqueeze(1) # (B, 1, d_model)
        q_b = cond_b.unsqueeze(1) # (B, 1, d_model)
        
        # Drug A queries Drug B
        attn_a, _ = self.cross_attn(query=q_a, key=q_b, value=q_b)
        aware_a = self.norm(q_a + attn_a).squeeze(1) # (B, d_model)
        
        # Drug B queries Drug A
        attn_b, _ = self.cross_attn(query=q_b, key=q_a, value=q_a)
        aware_b = self.norm(q_b + attn_b).squeeze(1) # (B, d_model)
        
        return aware_a, aware_b
