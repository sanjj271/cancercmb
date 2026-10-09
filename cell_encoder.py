import torch
import torch.nn as nn

class CellLineEncoder(nn.Module):
    """Encodes cell line gene expression vectors using pathway-based token projection."""
    
    def __init__(
        self,
        in_dim: int = 976,
        d_model: int = 256,
        n_pathways: int = 300,
        use_pathway_projection: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_pathway_projection = use_pathway_projection
        self.d_model = d_model
        self.n_pathways = n_pathways
        
        if use_pathway_projection:
            self.pathway_map = nn.Linear(in_dim, n_pathways)
            self.pathway_projection = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(n_pathways, n_pathways),
                nn.LayerNorm(n_pathways)
            )
            self.pathway_embeddings = nn.Parameter(torch.randn(n_pathways, d_model))
        else:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects transcriptomic vector to a sequence of pathway-level tokens.

        Args:
            x: Raw expression vector of shape (B, in_dim).

        Returns:
            torch.Tensor: Sequence of tokens of shape (B, n_pathways, d_model) or (B, 1, d_model).
        """
        if self.use_pathway_projection:
            pathway_scores = self.pathway_projection(self.pathway_map(x))
            seq_out = pathway_scores.unsqueeze(-1) * self.pathway_embeddings.unsqueeze(0)
            return seq_out
        else:
            vec_out = self.projection(x)
            return vec_out.unsqueeze(1)
