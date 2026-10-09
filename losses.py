import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

class CancerComboLoss(nn.Module):
    """Composite loss module combining Mean Squared Error over dose viability matrices,
    pairwise Margin Ranking Loss (for Spearman rank correlation optimization),
    and optional Auxiliary Parameter Supervision loss.
    """
    
    def __init__(self, rank_lambda: float = 1.0, aux_lambda: float = 0.05, num_ranking_pairs: int = 256):

        super().__init__()
        self.mse = nn.MSELoss()
        self.margin_ranking = nn.MarginRankingLoss(margin=0.0)
        self.smooth_l1 = nn.SmoothL1Loss()
        self.rank_lambda = rank_lambda
        self.aux_lambda = aux_lambda
        self.num_ranking_pairs = num_ranking_pairs
        
    def _compute_ranking_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes pairwise margin ranking loss across flattened dose grid points."""
        B = y_pred.size(0)
        pred_flat = y_pred.view(B, -1)
        true_flat = y_true.view(B, -1)
        N = pred_flat.size(1)
        
        if N < 2:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            
        # Sample pairs of indices
        idx1 = torch.randint(0, N, (self.num_ranking_pairs,), device=y_pred.device)
        idx2 = torch.randint(0, N, (self.num_ranking_pairs,), device=y_pred.device)
        
        # Avoid self-pairs
        mask = idx1 != idx2
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        
        if len(idx1) == 0:
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            
        pred1 = pred_flat[:, idx1] # (B, P)
        pred2 = pred_flat[:, idx2] # (B, P)
        true1 = true_flat[:, idx1] # (B, P)
        true2 = true_flat[:, idx2] # (B, P)
        
        target_sign = torch.sign(true1 - true2) # +1 if true1 > true2, -1 if true1 < true2
        valid_mask = (target_sign != 0).float()
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
            
        # Mathematically equivalent pairwise margin ranking loss using a dense multiplier mask.
        # This keeps shapes static/known and avoids the dynamic CUDA masked_select reduction deadlock.
        loss_pairs = torch.clamp(-target_sign * (pred1 - pred2), min=0.0)
        num_valid = valid_mask.sum().clamp(min=1.0)
        return (loss_pairs * valid_mask).sum() / num_valid

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        params_pred: Optional[Tuple[torch.Tensor, ...]] = None,
        params_true: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Computes the composite loss: Surface MSE + λ_rank * Ranking Loss + λ_aux * Param Loss.

        Args:
            y_pred: Predicted viability of shape (B, M, N).
            y_true: Ground truth viability of shape (B, M, N).
            params_pred: Optional tuple of predicted parameter tensors (e1, e2, e3, log_c1, log_c2, h1, h2, alpha).
            params_true: Optional dict of ground-truth parameter tensors if available.

        Returns:
            torch.Tensor: Scalar combined loss value.
        """
        loss_mse = self.mse(y_pred, y_true)
        loss_rank = self._compute_ranking_loss(y_pred, y_true)
        total_loss = loss_mse + self.rank_lambda * loss_rank
        
        # Auxiliary parameter supervision if ground truth Hill parameters are provided in batch
        if params_pred is not None and params_true is not None:
            param_names = ["e1", "e2", "e3", "log_c1", "log_c2", "h1", "h2", "alpha"]
            loss_aux = 0.0
            count = 0
            for pred, name in zip(params_pred, param_names):
                if name in params_true:
                    target = params_true[name]
                    if target.dim() == 1:
                        target = target.unsqueeze(-1)
                    loss_aux += self.smooth_l1(pred, target)
                    count += 1
            if count > 0:
                total_loss += self.aux_lambda * (loss_aux / count)
                
        return total_loss

