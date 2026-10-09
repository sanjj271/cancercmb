import torch
import torch.nn as nn

class DescriptorEncoder(nn.Module):
    """Molecular descriptor feature extractor block."""
    
    def __init__(self, in_dim: int = 200, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(in_dim), # Input LayerNorm standardizes raw descriptor scales (ranging from 1e-3 to 1e3)
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects continuous descriptors to latent space.

        Args:
            x: Descriptors tensor of shape (B, in_dim).

        Returns:
            torch.Tensor: Projected features of shape (B, d_model).
        """
        return self.projection(x)
