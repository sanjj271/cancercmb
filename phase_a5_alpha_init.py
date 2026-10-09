import sys
import math
import torch

from config import load_config
from blocks.prediction_heads import CancerComboPredictionHeads

def main():
    mcfg, _tcfg = load_config("config.yaml")

    print("=" * 70)
    print("PHASE A5 - ALPHA INITIALIZATION ACROSS SEEDS")
    print(f"alpha_min={mcfg.alpha_min}  alpha_max={mcfg.alpha_max}")
    print(f"transformation: alpha = relu(raw_alpha) + alpha_min")
    print(f"neutral target alpha(raw=0) = {mcfg.alpha_min}  (1.0 would be pharmacologically neutral)")
    print("=" * 70)

    B = 512
    in_dim = 2 * mcfg.d_model  # 512
    seeds = [0, 1, 2, 3, 4]

    for seed in seeds:
        torch.manual_seed(seed)
        heads = CancerComboPredictionHeads(mcfg)
        heads.eval()
        unified_rep = torch.randn(B, in_dim) * 0.1

        with torch.no_grad():
            raw_alpha = heads.head_alpha(unified_rep)  # (B,1)
            alpha = torch.relu(raw_alpha) + mcfg.alpha_min

        frac_relu_dead = (raw_alpha <= 0.0).float().mean().item()
        frac_at_min = ((alpha - mcfg.alpha_min).abs() < 1e-6).float().mean().item()

        print(f"\nseed {seed}:")
        print(f"  raw_alpha  mean={raw_alpha.mean().item():+.6f}  std={raw_alpha.std().item():.6f}  "
              f"min={raw_alpha.min().item():+.6f}  max={raw_alpha.max().item():+.6f}")
        print(f"  alpha      mean={alpha.mean().item():.6f}  std={alpha.std().item():.6f}  "
              f"min={alpha.min().item():.6f}  max={alpha.max().item():.6f}")
        print(f"  raw<=0 (ReLU-dead fraction)    : {frac_relu_dead:.4f}")
        print(f"  alpha == alpha_min fraction    : {frac_at_min:.4f}")

    print("\n" + "=" * 70)
    print("A5 INTERPRETATION")
    print("At init, raw_alpha centers near 0; ReLU drives a large fraction to exactly")
    print(f"alpha = alpha_min = {mcfg.alpha_min} instead of the neutral value 1.0.")
    print("This matches the established degraded initialization (alpha pinned at bottom).")
    print("=" * 70)

if __name__ == "__main__":
    sys.exit(main())