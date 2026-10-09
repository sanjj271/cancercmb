import numpy as np
try:
    from scipy.stats import pearsonr
except ImportError:
    def pearsonr(x, y):
        corr = np.corrcoef(x, y)[0, 1]
        return corr, 0.0
    def spearmanr(x, y):
        def _rankdata(a):
            sorter = np.argsort(a)
            inv = np.empty_like(sorter, dtype=np.float64)
            inv[sorter] = np.arange(len(a), dtype=np.float64)
            return inv
        rx = _rankdata(x)
        ry = _rankdata(y)
        corr = np.corrcoef(rx, ry)[0, 1]
        return corr, 0.0
from typing import Dict

def calculate_metrics(y_pred: np.ndarray, y_true: np.ndarray, top_k_ratio: float = 0.1) -> Dict[str, float]:
    """Calculate statistical validation and synergy ranking metrics over viability matrices.

    Args:
        y_pred: Flattened or multi-dimensional predicted viability array.
        y_true: Flattened or multi-dimensional experimental viability array.
        top_k_ratio: Top percentile fraction for synergy hit evaluation (default: top 10%).

    Returns:
        Dict[str, float]: Dictionary containing MSE, RMSE, MAE, R2, Pearson, Spearman,
                          Top-K Recall, Top-K Precision, and Top-K Hit Rate scores.
    """
    pred_flat = y_pred.flatten().astype(np.float64)
    true_flat = y_true.flatten().astype(np.float64)
    
    # Regression metrics
    mse_val = float(np.mean((pred_flat - true_flat) ** 2))
    rmse_val = float(np.sqrt(mse_val))
    mae_val = float(np.mean(np.abs(pred_flat - true_flat)))
    
    # R-squared (Coefficient of Determination)
    ss_tot = float(np.sum((true_flat - np.mean(true_flat)) ** 2))
    ss_res = float(np.sum((true_flat - pred_flat) ** 2))
    r2_val = 1.0 - (ss_res / (ss_tot + 1e-12)) if ss_tot > 1e-12 else 0.0
    
    # Correlation metrics
    if np.std(pred_flat) < 1e-9 or np.std(true_flat) < 1e-9:
        pearson_val = 0.0
        spearman_val = 0.0
    else:
        try:
            pearson_val, _ = pearsonr(pred_flat, true_flat)
            if np.isnan(pearson_val):
                pearson_val = 0.0
        except Exception:
            pearson_val = 0.0
            
        try:
            rx = np.argsort(np.argsort(pred_flat)).astype(np.float64)
            ry = np.argsort(np.argsort(true_flat)).astype(np.float64)
            spearman_val = float(np.corrcoef(rx, ry)[0, 1])
            if np.isnan(spearman_val):
                spearman_val = 0.0
        except Exception:
            spearman_val = 0.0
        
    # Top-K Synergy / Maximum Inhibition Ranking Metrics
    # In drug viability, lowest viability = highest inhibition / maximum synergy
    k = max(1, int(len(true_flat) * top_k_ratio))
    
    # Top K indices for true and predicted maximum inhibition (lowest viability values)
    true_top_k_indices = set(np.argsort(true_flat)[:k])
    pred_top_k_indices = set(np.argsort(pred_flat)[:k])
    
    hits = len(true_top_k_indices.intersection(pred_top_k_indices))
    top_k_precision = hits / k
    top_k_recall = hits / len(true_top_k_indices)
    top_k_hit_rate = 1.0 if hits > 0 else 0.0
    
    return {
        "mse": mse_val,
        "rmse": rmse_val,
        "mae": mae_val,
        "r2": r2_val,
        "pearson": float(pearson_val),
        "spearman": float(spearman_val),
        "top_k_precision": float(top_k_precision),
        "top_k_recall": float(top_k_recall),
        "top_k_hit_rate": float(top_k_hit_rate)
    }

