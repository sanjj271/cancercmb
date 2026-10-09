import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig
from typing import Tuple

class DeepSynBaBlock(nn.Module):
    """DeepSynBa MLP Block with Linear, LayerNorm, ReLU, and Dropout."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(int(in_channels), int(out_channels)),
            nn.LayerNorm((int(out_channels),), eps=1e-05, elementwise_affine=True),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepSynBaPredictionHead(nn.Module):
    """DeepSynBa 3-layer MLP Prediction Head for biophysical parameters."""
    def __init__(self, in_channels: int = 512, emb_size: int = 1024, dropout: float = 0.2):
        super().__init__()
        layers = [
            DeepSynBaBlock(in_channels, emb_size, dropout=dropout),
            DeepSynBaBlock(emb_size, emb_size / 2, dropout=dropout),
            DeepSynBaBlock(emb_size / 2, emb_size / 4, dropout=dropout),
            nn.Linear(int(emb_size / 4), 1)
        ]
        self.prediction_head = nn.Sequential(*layers)

        for name, param in self.prediction_head.named_parameters():
            if 'weight' in name and len(param.data.shape) > 1:
                nn.init.kaiming_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(input)


class DoseResponsePredictor(nn.Module):
    """DeepSynBa 4-layer MLP Dose-Response Predictor for bias vectors."""
    def __init__(self, in_channels: int = 512, emb_size: int = 1024, dropout: float = 0.2):
        super().__init__()
        layers = [
            DeepSynBaBlock(in_channels, emb_size, dropout=dropout),
            DeepSynBaBlock(emb_size, emb_size / 2, dropout=dropout),
            DeepSynBaBlock(emb_size / 2, emb_size / 4, dropout=dropout),
            nn.Linear(int(emb_size / 4), 4),
            nn.ReLU()
        ]
        self.prediction_head = nn.Sequential(*layers)

        for name, param in self.prediction_head.named_parameters():
            if 'weight' in name and len(param.data.shape) > 1:
                nn.init.kaiming_normal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.prediction_head(input)


#######################################################
# OLD CODE - CANCERCOMBO ATTENTION
#######################################################
# class CancerComboPredictionHeads(nn.Module):
#     """Symmetric parameter prediction heads enforcing exact biophysical permutation invariance."""
#     
#     def __init__(self, config: ModelConfig):
#         super().__init__()
#         self.config = config
#         d_model = config.d_model
#         d_ff = config.d_ff
#         
#         # Calculate log-space boundaries for C1 and C2
#         c_min_safe = max(config.c_min, 1e-12)
#         c_max_safe = max(config.c_max, 1e-12)
#         self.log_c_min = math.log(c_min_safe)
#         self.log_c_max = math.log(c_max_safe)
#         
#         # Shared single-drug parameter heads
#         self.head_e_single = self._build_head(d_model, d_ff)
#         self.head_log_c_single = self._build_head(d_model, d_ff)
#         self.head_h_single = self._build_head(d_model, d_ff)
#         
#         # Combination & interaction heads
#         self.head_e3 = self._build_head(d_model, d_ff)
#         self.head_alpha = self._build_head(d_model, d_ff)
#         self.output_bias = nn.Parameter(torch.zeros(8))
#         
#     def _build_head(self, d_model: int, d_ff: int) -> nn.Sequential:
#         head = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.ReLU(),
#             nn.Dropout(self.config.dropout),
#             nn.Linear(d_ff, 1)
#         )
#         nn.init.normal_(head[-1].weight, std=0.01)
#         nn.init.zeros_(head[-1].bias)
#         return head
# 
#     def forward(self, aware_a, aware_b, z_combo):
#         ...
#######################################################


class CancerComboPredictionHeads(nn.Module):
    """DeepSynBa-style parameter prediction heads for Bivariate Hill parameters."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        in_channels = getattr(config, "d_model", 256) * 2  # Unified representation dim (256 * 2 = 512)
        emb_size = getattr(config, "emb_size", 1024)
        dropout = getattr(config, "dropout", 0.2)
        
        # Calculate log-space boundaries for C1 and C2
        c_min_safe = max(config.c_min, 1e-12)
        c_max_safe = max(config.c_max, 1e-12)
        self.log_c_min = math.log(c_min_safe)
        self.log_c_max = math.log(c_max_safe)

        # 8 DeepSynBa-style Prediction Heads for Bivariate Hill parameters
        self.head_e1 = DeepSynBaPredictionHead(in_channels, emb_size, dropout)
        self.head_e2 = DeepSynBaPredictionHead(in_channels, emb_size, dropout)
        self.head_e3 = DeepSynBaPredictionHead(in_channels, emb_size, dropout)
        self.head_log_c1 = DeepSynBaPredictionHead(in_channels, emb_size, dropout)
        self.head_log_c2 = DeepSynBaPredictionHead(in_channels, emb_size, dropout)
        self.head_h1 = DeepSynBaPredictionHead(in_channels, emb_size, dropout)
        self.head_h2 = DeepSynBaPredictionHead(in_channels, emb_size, dropout)
        self.head_alpha = DeepSynBaPredictionHead(in_channels, emb_size, dropout)

        # 2 DeepSynBa-style Bias Predictors
        self.bias_predictor1 = DoseResponsePredictor(in_channels, emb_size, dropout)
        self.bias_predictor2 = DoseResponsePredictor(in_channels, emb_size, dropout)

    def forward(
        self,
        unified_rep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts e1, e2, e3, log_c1, log_c2, h1, h2, and alpha parameters over unified representation.

        Args:
            unified_rep: Unified drug combination feature tensor of shape (B, 512).

        Returns:
            Tuple[torch.Tensor, ...]: Constrained scalars of shape (B, 1) each.
        """
        raw_e1 = self.head_e1(unified_rep)
        raw_e2 = self.head_e2(unified_rep)
        raw_e3 = self.head_e3(unified_rep)
        raw_log_c1 = self.head_log_c1(unified_rep)
        raw_log_c2 = self.head_log_c2(unified_rep)
        raw_h1 = self.head_h1(unified_rep)
        raw_h2 = self.head_h2(unified_rep)
        raw_alpha = self.head_alpha(unified_rep)

        # Official DeepSynBa parameter transformations
        e1 = torch.sigmoid(raw_e1)
        e2 = torch.sigmoid(raw_e2)
        e3 = torch.sigmoid(raw_e3)

        log_c1 = raw_log_c1
        log_c2 = raw_log_c2

        h1 = F.relu(raw_h1)
        h2 = F.relu(raw_h2)

        alpha = F.relu(raw_alpha)

        return e1, e2, e3, log_c1, log_c2, h1, h2, alpha

    def predict_bias(
        self,
        unified_rep: torch.Tensor,
        doses_a: torch.Tensor,
        doses_b: torch.Tensor
    ) -> torch.Tensor:
        """Predicts and broadcasts 2D dose-dependent bias matrix matching official DeepSynBa.

        Args:
            unified_rep: Unified drug combination feature tensor of shape (B, 512).
            doses_a: Drug A doses tensor of shape (B, M).
            doses_b: Drug B doses tensor of shape (B, N).

        Returns:
            torch.Tensor: 2D bias matrix of shape (B, M, N).
        """
        out1 = self.bias_predictor1(unified_rep)  # (B, 4)
        out2 = self.bias_predictor2(unified_rep)  # (B, 4)

        if doses_a.ndim != 2:
            raise ValueError(f"Expected doses_a tensor shape [batch, num_doses] (ndim=2), got shape {tuple(doses_a.shape)}")
        if doses_b.ndim != 2:
            raise ValueError(f"Expected doses_b tensor shape [batch, num_doses] (ndim=2), got shape {tuple(doses_b.shape)}")

        b_size = unified_rep.shape[0]
        M = doses_a.shape[1]
        N = doses_b.shape[1]

        if M == 4 and N == 4:
            out1_grid = out1.reshape(b_size, 4, 1).repeat(1, 1, 4)
            out2_grid = out2.reshape(b_size, 1, 4).repeat(1, 4, 1)
        else:
            out1_grid = out1[:, :M].unsqueeze(2).repeat(1, 1, N) if out1.shape[1] >= M else torch.nn.functional.interpolate(out1.unsqueeze(1), size=M, mode='linear', align_corners=False).squeeze(1).unsqueeze(2).repeat(1, 1, N)
            out2_grid = out2[:, :N].unsqueeze(1).repeat(1, M, 1) if out2.shape[1] >= N else torch.nn.functional.interpolate(out2.unsqueeze(1), size=N, mode='linear', align_corners=False).squeeze(1).unsqueeze(1).repeat(1, M, 1)

        d1_grid = doses_a.reshape(b_size, M, 1).repeat(1, 1, N)
        d2_grid = doses_b.reshape(b_size, 1, N).repeat(1, M, 1)

        bias = torch.mul(out1_grid, d1_grid) + torch.mul(out2_grid, d2_grid)
        return bias
