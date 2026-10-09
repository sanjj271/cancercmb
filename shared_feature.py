import torch
import torch.nn as nn

class SymmetricComboFusion(nn.Module):
    """Enforces permutation invariance for drug combination representations."""
    
    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.shared_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Projects and sums both drug vectors, ensuring that swapped inputs yield the same output.

        Args:
            z_a: Drug A features of shape (B, d_model).
            z_b: Drug B features of shape (B, d_model).

        Returns:
            torch.Tensor: Combined combination vector of shape (B, d_model).
        """
        proj_a = self.shared_proj(z_a)
        proj_b = self.shared_proj(z_b)
        return proj_a + proj_b
