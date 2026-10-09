import torch
from config import ModelConfig
from cancercombo import CancerCombo
from dataset import SMILESTokenizer
from preprocessor import MolecularPreprocessor
import numpy as np
from typing import Dict, Any

class SynergyPredictor:
    """Inference class loaded from a trained checkpoint to run predictions on drug combinations."""
    
    def __init__(self, checkpoint_path: str, config: ModelConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = CancerCombo(config)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        # Strip PyTorch Lightning 'model.' prefix if present
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        trainable_missing = [k for k in incompatible.missing_keys if not k.startswith("molformer_enc.pretrained_")]
        if trainable_missing:
            raise RuntimeError(f"Checkpoint is missing required trainable keys: {trainable_missing}")
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = SMILESTokenizer(max_len=128)
        self.preprocessor = MolecularPreprocessor()
        
    def predict(
        self,
        smiles_a: str,
        smiles_b: str,
        cell_line_gene_expr: np.ndarray,
        doses_a: np.ndarray,
        doses_b: np.ndarray
    ) -> Dict[str, Any]:
        """Run single-combination prediction.

        Args:
            smiles_a: SMILES string of Drug A.
            smiles_b: SMILES string of Drug B.
            cell_line_gene_expr: Array of size (cell_in_dim,).
            doses_a: Dose level list/array of Drug A.
            doses_b: Dose level list/array of Drug B.

        Returns:
            Dict[str, Any]: Predicted matrix and pharmacological parameter parameters values.
        """
        # Preprocess features
        morgan_a, desc_a, _ = self.preprocessor.process_smiles(smiles_a)
        morgan_b, desc_b, _ = self.preprocessor.process_smiles(smiles_b)
        
        ids_a, mask_a = self.tokenizer.tokenize(smiles_a)
        ids_b, mask_b = self.tokenizer.tokenize(smiles_b)
        
        # Prepare inputs
        t_ids_a = torch.tensor([ids_a], dtype=torch.long, device=self.device)
        t_mask_a = torch.tensor([mask_a], dtype=torch.float32, device=self.device)
        t_morgan_a = torch.tensor(np.array([morgan_a]), dtype=torch.float32, device=self.device)
        t_desc_a = torch.tensor(np.array([desc_a]), dtype=torch.float32, device=self.device)
        
        t_ids_b = torch.tensor([ids_b], dtype=torch.long, device=self.device)
        t_mask_b = torch.tensor([mask_b], dtype=torch.float32, device=self.device)
        t_morgan_b = torch.tensor(np.array([morgan_b]), dtype=torch.float32, device=self.device)
        t_desc_b = torch.tensor(np.array([desc_b]), dtype=torch.float32, device=self.device)
        
        t_cell = torch.tensor(np.array([cell_line_gene_expr]), dtype=torch.float32, device=self.device)
        t_doses_a = torch.tensor(np.array([doses_a]), dtype=torch.float32, device=self.device)
        t_doses_b = torch.tensor(np.array([doses_b]), dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            y_pred, params = self.model(
                t_ids_a, t_mask_a, t_morgan_a, t_desc_a,
                t_ids_b, t_mask_b, t_morgan_b, t_desc_b,
                t_cell, t_doses_a, t_doses_b
            )
            
        e1, e2, e3, log_c1, log_c2, h1, h2, alpha = params
        
        return {
            "predicted_viability_matrix": y_pred.squeeze(0).cpu().numpy(),
            "pharmacological_parameters": {
                "e1": float(e1.item()),
                "e2": float(e2.item()),
                "e3": float(e3.item()),
                "log_c1": float(log_c1.item()),
                "log_c2": float(log_c2.item()),
                "c1": float(torch.exp(log_c1).item()),
                "c2": float(torch.exp(log_c2).item()),
                "h1": float(h1.item()),
                "h2": float(h2.item()),
                "alpha": float(alpha.item())
            }
        }
