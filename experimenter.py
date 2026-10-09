import os
for _k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "1"

from helpers import enforce_single_thread
enforce_single_thread()

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

try:
    import pytorch_lightning as pl  # type: ignore # pyrefly: ignore [missing-import]
except ImportError:
    pl = None

from config import load_config, ModelConfig, TrainingConfig
from dataset import DrugComboDataset
from trainer import CancerComboLightningModule
from cancercombo import CancerCombo
from losses import CancerComboLoss
from helpers import generate_mock_data, set_seed
from logger import setup_logger


class Experimenter:
    """Manages structural and hyperparameter experiments for comparison reports.
    
    Supports both PyTorch Lightning and native PyTorch execution engines.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.logger = setup_logger("CancerCombo Experimenter")
        
    def _run_single_ablation(
        self,
        enable_dd_attn: bool,
        m_config: ModelConfig,
        t_config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 2
    ) -> Dict[str, float]:
        """Executes a single ablation experiment case with dual PyTorch Lightning / Native fallback engine.

        Args:
            enable_dd_attn: Whether to enable drug-drug cross-attention.
            m_config: Model configuration dataclass.
            t_config: Training configuration dataclass.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Number of ablation training epochs.

        Returns:
            Dict[str, float]: Dictionary containing validation metrics.
        """
        m_config.enable_drug_drug_attention = enable_dd_attn

        if pl is not None:
            model = CancerComboLightningModule(m_config, t_config)
            trainer = pl.Trainer(
                max_epochs=epochs,
                accelerator="cpu",
                devices=1,
                enable_checkpointing=False,
                logger=False
            )
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            val_res = trainer.validate(model, dataloaders=val_loader, verbose=False)[0]
            return val_res
        else:
            self.logger.warning("PyTorch Lightning unavailable. Running native PyTorch ablation loop...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                self.logger.info("Configuring PyTorch CUDA settings...")
                try:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                    self.logger.info("  [SUCCESS] Disabled FlashAttention and MemEfficient Attention SDP backends.")
                except Exception as e:
                    self.logger.warning(f"  [WARNING] Failed to configure SDPA kernels: {e}")
            net = CancerCombo(m_config).to(device)
            loss_fn = CancerComboLoss()
            optimizer = torch.optim.AdamW(net.parameters(), lr=t_config.lr, weight_decay=t_config.weight_decay)

            for epoch in range(1, epochs + 1):
                self.logger.info(f"--- Ablation Epoch [{epoch}/{epochs}] Started ---")
                net.train()
                train_loss_sum = 0.0
                for batch in train_loader:
                    optimizer.zero_grad()
                    b_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    y_pred, params = net(
                        b_gpu["drug_a_ids"], b_gpu["drug_a_mask"], b_gpu["drug_a_morgan"], b_gpu["drug_a_desc"],
                        b_gpu["drug_b_ids"], b_gpu["drug_b_mask"], b_gpu["drug_b_morgan"], b_gpu["drug_b_desc"],
                        b_gpu["cell_line"], b_gpu["doses_a"], b_gpu["doses_b"]
                    )
                    loss = loss_fn(y_pred, b_gpu["viability"], params)
                    loss.backward()
                    optimizer.step()
                    train_loss_sum += loss.item()
                avg_train_loss = train_loss_sum / max(len(train_loader), 1)
                self.logger.info(f"--- Ablation Epoch [{epoch}/{epochs}] Finished | Avg Train Loss: {avg_train_loss:.4f} ---")

            net.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    b_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    y_pred, params = net(
                        b_gpu["drug_a_ids"], b_gpu["drug_a_mask"], b_gpu["drug_a_morgan"], b_gpu["drug_a_desc"],
                        b_gpu["drug_b_ids"], b_gpu["drug_b_mask"], b_gpu["drug_b_morgan"], b_gpu["drug_b_desc"],
                        b_gpu["cell_line"], b_gpu["doses_a"], b_gpu["doses_b"]
                    )
                    v_loss = loss_fn(y_pred, b_gpu["viability"], params)
                    val_loss_sum += v_loss.item()

            val_loss = val_loss_sum / max(len(val_loader), 1)
            return {"val_loss": val_loss}

    def run_ablation_study(self) -> Dict[str, Dict[str, float]]:
        """Compares drug-drug cross attention enabled vs disabled configurations.

        Returns:
            Dict[str, Dict[str, float]]: Validation performance results of both configurations.
        """
        self.logger.info("Initializing ablation data...")
        m_config, t_config = load_config(self.config_path)
        set_seed(t_config.seed)
        
        train_data, cell_features = generate_mock_data(32)
        val_data, _ = generate_mock_data(8)
        
        from dataset import DescriptorScaler
        exp_scaler = DescriptorScaler()
        exp_scaler.fit([np.zeros(200, dtype=np.float32)])
        train_dataset = DrugComboDataset(train_data, cell_features, descriptor_scaler=exp_scaler)
        val_dataset = DrugComboDataset(val_data, cell_features, descriptor_scaler=exp_scaler)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        results = {}
        
        # Test Case 1: Drug-Drug Attention Disabled
        self.logger.info("--- Ablation Case 1: Drug-Drug Cross-Attention DISABLED ---")
        val_res_1 = self._run_single_ablation(
            enable_dd_attn=False,
            m_config=m_config,
            t_config=t_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        results["disabled"] = val_res_1
        
        # Test Case 2: Drug-Drug Attention Enabled
        self.logger.info("--- Ablation Case 2: Drug-Drug Cross-Attention ENABLED ---")
        val_res_2 = self._run_single_ablation(
            enable_dd_attn=True,
            m_config=m_config,
            t_config=t_config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        results["enabled"] = val_res_2
        
        self.logger.info("Ablation Study Results Comparison:")
        self.logger.info(f"  Disabled DD Attention - Val Loss: {val_res_1.get('val_loss'):.4f}")
        self.logger.info(f"  Enabled DD Attention  - Val Loss: {val_res_2.get('val_loss'):.4f}")
        
        return results

if __name__ == "__main__":
    exp = Experimenter()
    exp.run_ablation_study()
