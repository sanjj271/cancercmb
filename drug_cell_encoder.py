import torch
import torch.nn as nn

class DeepSynBaDrugCellEncoder(nn.Module):
    """DeepSynBa-style Drug-Cell Encoder MLP block.
    
    Processes concatenated drug features (Morgan + RDKit Descriptors) and cell line gene expression.
    """
    
    def __init__(self, in_dim: int = 1232, d_model: int = 256, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Initialize weights with Kaiming Normal (matching DeepSynBa)
        for name, param in self.encoder.named_parameters():
            if 'weight' in name and len(param.data.shape) > 1:
                nn.init.kaiming_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, drug_cell_concat: torch.Tensor) -> torch.Tensor:
        """Forward pass for Drug-Cell encoder.
        
        Args:
            drug_cell_concat: Concatenated drug and cell line tensor (B, in_dim).
            
        Returns:
            torch.Tensor: Encoded drug representation of shape (B, d_model).
        """
        return self.encoder(drug_cell_concat)
