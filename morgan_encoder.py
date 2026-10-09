import torch
import torch.nn as nn

class MorganEncoder(nn.Module):
    """Morgan fingerprint feature extractor block."""
    
    def __init__(self, in_dim: int = 1024, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects binary fingerprint vectors to latent space.

        Args:
            x: Raw fingerprint tensor of shape (B, in_dim).

        Returns:
            torch.Tensor: Projected features of shape (B, d_model).
        """
        return self.projection(x)
