import torch
try:
    import pytorch_lightning as pl  # type: ignore # pyrefly: ignore [missing-import]
except ImportError:
    import torch.nn as _nn
    class _DummyPLModule(_nn.Module):
        def save_hyperparameters(self): pass
        def log(self, *args, **kwargs): pass
    pl = type("pl", (), {"LightningModule": _DummyPLModule})

from config import ModelConfig, TrainingConfig
from cancercombo import CancerCombo
from losses import CancerComboLoss
from metrics import calculate_metrics
import numpy as np
from typing import Dict, Any, Tuple, List

class CancerComboLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper module for training CancerCombo."""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.training_config = training_config
        
        self.model = CancerCombo(model_config)
        self.loss_fn = CancerComboLoss()
        
        self.validation_step_outputs: List[Dict[str, np.ndarray]] = []
        self.test_step_outputs: List[Dict[str, np.ndarray]] = []
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.model(
            drug_a_morgan=batch.get("drug_a_morgan"),
            drug_a_desc=batch.get("drug_a_desc"),
            drug_a_dti=batch.get("drug_a_dti"),
            drug_a_pathway=batch.get("drug_a_pathway"),
            drug_a_m_dti=batch.get("drug_a_m_dti"),
            drug_a_m_pathway=batch.get("drug_a_m_pathway"),
            drug_b_morgan=batch.get("drug_b_morgan"),
            drug_b_desc=batch.get("drug_b_desc"),
            drug_b_dti=batch.get("drug_b_dti"),
            drug_b_pathway=batch.get("drug_b_pathway"),
            drug_b_m_dti=batch.get("drug_b_m_dti"),
            drug_b_m_pathway=batch.get("drug_b_m_pathway"),
            cell_line=batch.get("cell_line", batch.get("cell")),
            doses_a=batch.get("doses_a"),
            doses_b=batch.get("doses_b")
        )
        
    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self):
        max_epochs = getattr(self.trainer, "max_epochs", "?") if hasattr(self, "trainer") and self.trainer else "?"
        metrics = getattr(self.trainer, "logged_metrics", {})
        tr_loss = metrics.get("train_loss_epoch", metrics.get("train_loss_step", 0.0))
        val_l = metrics.get("val_loss", 0.0)
        val_r = metrics.get("val_rmse", 0.0)
        val_p = metrics.get("val_pearson", 0.0)
        val_s = metrics.get("val_spearman", 0.0)
        
        tr_val = f"{tr_loss:.4f}" if isinstance(tr_loss, (float, int)) or hasattr(tr_loss, "item") else str(tr_loss)
        val_l_val = f"{val_l:.4f}" if isinstance(val_l, (float, int)) or hasattr(val_l, "item") else str(val_l)
        val_r_val = f"{val_r:.4f}" if isinstance(val_r, (float, int)) or hasattr(val_r, "item") else str(val_r)
        val_p_val = f"{val_p:.4f}" if isinstance(val_p, (float, int)) or hasattr(val_p, "item") else str(val_p)
        val_s_val = f"{val_s:.4f}" if isinstance(val_s, (float, int)) or hasattr(val_s, "item") else str(val_s)
        
        print(f"\n--- Epoch [{self.current_epoch + 1}/{max_epochs}] Summary | Train Loss: {tr_val} | Val Loss: {val_l_val} | Val RMSE: {val_r_val} | Val Pearson: {val_p_val} | Val Spearman: {val_s_val} ---\n")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        y_pred, params_pred = self(batch)
        y_true = batch["viability"]
        
        # Extract ground-truth parameter dict if available for auxiliary supervision
        params_true = {p: batch[p] for p in ["e1", "e2", "e3", "log_c1", "log_c2", "h1", "h2", "alpha"] if p in batch}
        
        loss = self.loss_fn(y_pred, y_true, params_pred=params_pred, params_true=params_true if params_true else None)
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=False)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=False)
        return loss
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        y_pred, params_pred = self(batch)
        y_true = batch["viability"]
        
        params_true = {p: batch[p] for p in ["e1", "e2", "e3", "log_c1", "log_c2", "h1", "h2", "alpha"] if p in batch}
        loss = self.loss_fn(y_pred, y_true, params_pred=params_pred, params_true=params_true if params_true else None)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        
        self.validation_step_outputs.append({
            "pred": y_pred.detach().cpu().numpy(),
            "true": y_true.detach().cpu().numpy()
        })
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        preds = np.concatenate([x["pred"] for x in self.validation_step_outputs], axis=0)
        trues = np.concatenate([x["true"] for x in self.validation_step_outputs], axis=0)
        
        metrics = calculate_metrics(preds, trues)
        self.log("val_rmse", metrics["rmse"], prog_bar=True, sync_dist=False)
        self.log("val_mae", metrics["mae"], prog_bar=False, sync_dist=False)
        self.log("val_r2", metrics["r2"], prog_bar=False, sync_dist=False)
        self.log("val_pearson", metrics["pearson"], prog_bar=True, sync_dist=False)
        self.log("val_spearman", metrics["spearman"], prog_bar=True, sync_dist=False)
        self.log("val_top_k_precision", metrics["top_k_precision"], prog_bar=False, sync_dist=False)
        self.log("val_top_k_recall", metrics["top_k_recall"], prog_bar=False, sync_dist=False)
        self.log("val_top_k_hit_rate", metrics["top_k_hit_rate"], prog_bar=False, sync_dist=False)
        
        self.validation_step_outputs.clear()
        
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        y_pred, _ = self(batch)
        y_true = batch["viability"]
        
        loss = self.loss_fn(y_pred, y_true)
        self.log("test_loss", loss, sync_dist=False)
        
        self.test_step_outputs.append({
            "pred": y_pred.detach().cpu().numpy(),
            "true": y_true.detach().cpu().numpy()
        })
        return loss

    def on_test_epoch_start(self):
        self.test_step_outputs.clear()

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
            
        preds = np.concatenate([x["pred"] for x in self.test_step_outputs], axis=0)
        trues = np.concatenate([x["true"] for x in self.test_step_outputs], axis=0)
        
        metrics = calculate_metrics(preds, trues)
        self.log("test_rmse", metrics["rmse"], sync_dist=False)
        self.log("test_pearson", metrics["pearson"], sync_dist=False)
        self.log("test_spearman", metrics["spearman"], sync_dist=False)
        
        self.test_step_outputs.clear()
        
    def configure_optimizers(self) -> Dict[str, Any]:
        opt_name = getattr(self.training_config, "optimizer_name", "AdamW").lower()
        if opt_name == "adam":
            opt_cls = torch.optim.Adam
        else:
            opt_cls = torch.optim.AdamW

        optimizer = opt_cls(
            self.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay
        )
        factor = getattr(self.training_config, "scheduler_factor", 0.5)
        patience = getattr(self.training_config, "scheduler_patience", 10)
        min_lr = getattr(self.training_config, "min_lr", 1.0e-6)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }
