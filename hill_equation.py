import torch
import torch.nn as nn
import numpy as np

class BivariateHillSolver(nn.Module):
    """Numerically stable 2D Bivariate Hill Solver using SynBa Log-Sum-Exp formulation."""
    
    def __init__(self, e0: float = 100.0):
        super().__init__()
        self.e0 = e0

    def synba_likelihood_2d(
        self, x_1, x_2, e_1, e_2, e_3, logC_1, logC_2, h_1, h_2, alpha, sigma=None, add_noise=False
    ):
        """SynBa 2D dose-response likelihood evaluation with Log-Sum-Exp numerical stability.

        Args:
            x_1: Dose grid concentrations for Drug A of shape (B, M) or (B, M, 1).
            x_2: Dose grid concentrations for Drug B of shape (B, N) or (B, 1, N).
            e_1: Drug A fractional efficacy parameter of shape (B, 1) or scalar.
            e_2: Drug B fractional efficacy parameter of shape (B, 1) or scalar.
            e_3: Combination fractional efficacy parameter of shape (B, 1) or scalar.
            logC_1: Log IC50 for Drug A of shape (B, 1) or scalar.
            logC_2: Log IC50 for Drug B of shape (B, 1) or scalar.
            h_1: Hill slope coefficient for Drug A of shape (B, 1) or scalar.
            h_2: Hill slope coefficient for Drug B of shape (B, 1) or scalar.
            alpha: Interaction parameter of shape (B, 1) or scalar.
            sigma: Optional noise magnitude parameter.
            add_noise: Whether to add Gaussian noise to the predicted viability matrix.

        Returns:
            torch.Tensor: Predicted viability grid of shape (B, M, N).
        """
        e_0 = self.e0

        # Reshape parameters to 3D tensors (B, 1, 1) for matrix broadcasting
        def _to_3d(p):
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(p, dtype=torch.float32, device=x_1.device if isinstance(x_1, torch.Tensor) else None)
            while p.dim() < 3:
                p = p.unsqueeze(-1)
            return p

        e_1_u = _to_3d(e_1)
        e_2_u = _to_3d(e_2)
        e_3_u = _to_3d(e_3)

        # Scale efficacy parameters to fractional [0, 1] range if passed as percentage [0, 100]
        if (e_1_u > 1.0).any(): e_1_u = e_1_u / 100.0
        if (e_2_u > 1.0).any(): e_2_u = e_2_u / 100.0
        if (e_3_u > 1.0).any(): e_3_u = e_3_u / 100.0

        logC_1_u = _to_3d(logC_1)
        logC_2_u = _to_3d(logC_2)
        h_1_u = _to_3d(h_1)
        h_2_u = _to_3d(h_2)
        alpha_u = _to_3d(alpha)

        # Reshape dose concentration inputs
        if x_1.dim() == 2:
            x_1 = x_1.unsqueeze(2)  # (B, M, 1)
        if x_2.dim() == 2:
            x_2 = x_2.unsqueeze(1)  # (B, 1, N)

        # Zero-dose presence masks
        mask_a = (x_1 > 0.0)
        mask_b = (x_2 > 0.0)
        mask_ab = mask_a & mask_b

        # Numerical safety: shift zero concentrations to 1e-6 before log() to prevent log(0) NaNs
        x_1_safe = torch.where(mask_a, x_1, torch.ones_like(x_1) * 1e-6)
        x_2_safe = torch.where(mask_b, x_2, torch.ones_like(x_2) * 1e-6)

        log_x1 = torch.log(x_1_safe)
        log_x2 = torch.log(x_2_safe)

        # 4 Log-exponent terms
        term_A = logC_1_u * h_1_u + logC_2_u * h_2_u
        term_B = log_x1 * h_1_u + logC_2_u * h_2_u
        term_C = logC_1_u * h_1_u + log_x2 * h_2_u
        term_D = log_x1 * h_1_u + log_x2 * h_2_u

        # Log-sum-exp trick for numerical stability
        max_exp = torch.maximum(
            term_A,
            torch.maximum(
                term_B,
                torch.maximum(term_C, term_D)
            )
        )

        w_A = torch.exp(term_A - max_exp)
        w_B = torch.where(mask_a, torch.exp(term_B - max_exp), torch.zeros_like(term_B))
        w_C = torch.where(mask_b, torch.exp(term_C - max_exp), torch.zeros_like(term_C))
        w_D = torch.where(mask_ab, torch.exp(term_D - max_exp), torch.zeros_like(term_D))

        exp_A = w_A * e_0
        exp_B = w_B * e_1_u * e_0
        exp_C = w_C * e_2_u * e_0
        exp_D = w_D * e_3_u * e_0 * alpha_u

        numerator = exp_A + exp_B + exp_C + exp_D
        denominator = w_A + w_B + w_C + w_D * alpha_u

        if add_noise and sigma is not None:
            if isinstance(sigma, torch.Tensor):
                while sigma.dim() < 3:
                    sigma = sigma.unsqueeze(-1)
                z = torch.randn_like(numerator) * sigma
            else:
                z = torch.randn_like(numerator) * float(sigma)
        else:
            z = 0.0

        y = numerator / denominator + z
        return y

    def forward(self, doses_a, doses_b, e1, e2, e3, log_c1, log_c2, h1, h2, alpha, sigma=None, add_noise=False):
        return self.synba_likelihood_2d(
            x_1=doses_a,
            x_2=doses_b,
            e_1=e1,
            e_2=e2,
            e_3=e3,
            logC_1=log_c1,
            logC_2=log_c2,
            h_1=h1,
            h_2=h2,
            alpha=alpha,
            sigma=sigma,
            add_noise=add_noise
        )