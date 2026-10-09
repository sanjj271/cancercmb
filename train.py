import os
for _k in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_k] = "1"

from helpers import enforce_single_thread
enforce_single_thread()

import torch
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

try:
    import pytorch_lightning as pl  # type: ignore # pyrefly: ignore [missing-import]
    from pytorch_lightning.callbacks import ModelCheckpoint  # type: ignore # pyrefly: ignore [missing-import]
except ImportError:
    pl = None
    class ModelCheckpoint:
        def __init__(self, *args, **kwargs): pass

from typing import Optional, Any, Iterable
from config import load_config, RAW_FEATURE_STORE_PATH, get_scenario_split_path, get_scenario_scaler_path
from dataset import DrugComboDataset, load_nci60_gex, load_synergy_dataset, load_precomputed_drug_features
from trainer import CancerComboLightningModule
from cancercombo import CancerCombo
from losses import CancerComboLoss
from helpers import set_seed, generate_mock_data
from logger import setup_logger

import argparse


def _iter_tensors(obj: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item)


def _register_backward_debug_hooks(net: torch.nn.Module, logger):
    """Attach gradient hooks to the main backward path for first-batch debugging."""

    handles = []

    def attach_output_hooks(module: torch.nn.Module, name: str):
        def _forward_hook(_module, _inputs, outputs):
            for idx, tensor in enumerate(_iter_tensors(outputs)):
                if not tensor.requires_grad:
                    continue

                tensor.retain_grad()

                def _grad_hook(grad, module_name=name, output_index=idx):
                    grad_norm = float(grad.norm().item()) if grad.numel() > 0 else 0.0
                    logger.info(
                        f"[BACKWARD DEBUG] grad reached {module_name}[{output_index}] "
                        f"shape={tuple(grad.shape)} dtype={grad.dtype} norm={grad_norm:.6f}"
                    )
                    return grad

                tensor.register_hook(_grad_hook)

        handles.append(module.register_forward_hook(_forward_hook))

    target_module_names = [
        "molformer_enc",
        "molformer_enc.transformer",
        "molformer_enc.transformer.layers.0",
        "molformer_enc.transformer.layers.0.self_attn",
        "molformer_enc.transformer.layers.1",
        "molformer_enc.transformer.layers.1.self_attn",
        "fusion",
        "fusion.self_attn",
        "cell_enc",
        "drug_cell_attn",
        "drug_cell_attn.cross_attn",
        "symmetric_fusion",
        "heads",
        "hill_solver",
    ]

    for name in target_module_names:
        try:
            mod = net
            for part in name.split('.'):
                if part.isdigit():
                    mod = mod[int(part)]
                else:
                    mod = getattr(mod, part)
            attach_output_hooks(mod, name)
        except (AttributeError, IndexError):
            logger.debug(f"[BACKWARD DEBUG] Module {name} not found in model architecture, skipping hook registration.")
            continue

    logger.info("[BACKWARD DEBUG] Registered gradient hooks for the main backward path.")
    return handles

def run_training(
    config_path: str = "config.yaml",
    epochs: Optional[int] = None,
    max_samples: Optional[int] = None,
    scenario: int = 1,
    engine: str = "auto",
    debug_backprop: bool = False,
    resume: Optional[str] = None
):
    """Initializes dataset generators and executes full model training.

    Args:
        config_path: Path to configuration file.
        epochs: Optional epoch override.
        max_samples: Optional maximum dataset samples limit.
        scenario: The split scenario to use (1, 2, or 3).
        engine: Execution engine: 'auto', 'lightning', or 'native'.
        debug_backprop: Attach gradient debug hooks.
        resume: Optional path to checkpoint to resume training from.
    """
    logger = setup_logger("CancerCombo Train")
    logger.info("Loading configs and setting seed...")
    
    m_config, t_config = load_config(config_path)
    if epochs is not None:
        t_config.epochs = epochs
    set_seed(t_config.seed)
    
    # Configure PyTorch CUDA backends to avoid hangs/deadlocks on GPU container setups
    if torch.cuda.is_available():
        logger.info("Configuring PyTorch CUDA settings...")
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            logger.info("  [SUCCESS] Disabled FlashAttention and MemEfficient Attention SDP backends (preventing CUDA compiler & padding mask hangs).")
        except Exception as e:
            logger.warning(f"  [WARNING] Failed to configure SDPA kernels: {e}")
    
    logger.info("Attempting to load real dataset archives...")
    real_gex = load_nci60_gex("data/features/NCI-60_landmark_gex.csv", target_dim=m_config.cell_in_dim)
    
    # Map scenario number to split file path
    split_path = get_scenario_split_path(scenario)
    logger.info(f"Loading split scenario from {split_path}...")
    if scenario == 3:
        print(f"Scenario 3 split file: {split_path}")
        logger.info(f"Scenario 3 split file: {split_path}")

    import pandas as pd
    split_df = pd.read_csv(split_path)
    
    if "split" not in split_df.columns:
        logger.error(f"Split file missing 'split' column: {split_path}")
        return
        
    train_df = split_df[split_df["split"] == 1].copy()
    val_df = split_df[split_df["split"] == 2].copy()
    
    if not hasattr(t_config, "checkpoint_dir") or not t_config.checkpoint_dir:
        t_config.checkpoint_dir = "checkpoints/drugcell_crossattention"

    feature_store_path = RAW_FEATURE_STORE_PATH
    drug_features = load_precomputed_drug_features(feature_path=feature_store_path)
    if not drug_features:
        raise FileNotFoundError(f"Required Morgan 1024 r=3 feature store missing at '{feature_store_path}'. Fail loudly.")
    logger.info(f"Loaded Morgan (1024-dim) + RDKit (200-dim) precomputed features for {len(drug_features)} keys from {feature_store_path}.")

    # Extract scenario-specific TRAIN, VAL, and TEST NSC IDs strictly from existing split artifact
    from dataset import extract_scenario_nscs
    scenario_train_nscs, scenario_val_nscs, scenario_test_nscs = extract_scenario_nscs(split_df)
    logger.info(
        f"Scenario {scenario} unique drugs: "
        f"TRAIN={len(scenario_train_nscs)}, VAL={len(scenario_val_nscs)}, TEST={len(scenario_test_nscs)}"
    )

    if scenario == 3:
        # Strict unseen-drug partition invariant check
        assert not (set(scenario_train_nscs) & set(scenario_val_nscs)), "Scenario 3 TRAIN and VAL drugs must be strictly disjoint!"
        assert not (set(scenario_train_nscs) & set(scenario_test_nscs)), "Scenario 3 TRAIN and TEST drugs must be strictly disjoint!"
        logger.info("  [VERIFIED] Scenario 3 drug sets are 100% strictly disjoint (Zero Leakage Guarantee).")

    # Fit DescriptorScaler strictly on TRAIN drugs using common fit_scenario_scaler
    from precompute_molecular_features import fit_scenario_scaler
    scaler_save_path = get_scenario_scaler_path(t_config.checkpoint_dir, scenario)
    desc_scaler = fit_scenario_scaler(
        split_csv=split_df,
        feature_store_or_path=drug_features,
        scaler_output=scaler_save_path
    )
    logger.info(f"Fitted DescriptorScaler on ALL {len(scenario_train_nscs)} unique TRAIN-drug descriptor vectors for Scenario {scenario} -> '{scaler_save_path}'.")
    from dataset import load_dti_embeddings, load_pathway_embeddings
    dti_store = load_dti_embeddings("data/drug_target_embeddings_2048.xlsx")
    pathway_store = load_pathway_embeddings("data/drug_pathway_embeddings_1152.xlsx")
    logger.info(f"Loaded DTI embeddings for {len(dti_store)} keys from data/drug_target_embeddings_2048.xlsx.")
    logger.info(f"Loaded Pathway embeddings for {len(pathway_store)} keys from data/drug_pathway_embeddings_1152.xlsx.")

    if max_samples is not None:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max(1, max_samples // 4))
        
    from dataset import parse_dataframe_to_records
    train_data = parse_dataframe_to_records(train_df, known_gex_dict=real_gex)
    val_data = parse_dataframe_to_records(val_df, known_gex_dict=real_gex)
    cell_features = real_gex

    import re
    def log_split_bio_availability(split_name: str, records: list, dti_s: dict, pw_s: dict):
        dti_missing_cnt = 0
        pw_missing_cnt = 0
        missing_dti_nscs = set()
        missing_pw_nscs = set()
        for rec in records:
            for key_nsc in ['nsc_a', 'nsc_b']:
                nsc_val = str(rec.get(key_nsc, '')).strip()
                clean_nsc = re.sub(r'^(NSC[-_]?)', '', nsc_val, flags=re.IGNORECASE).strip()
                keys_to_try = [nsc_val, clean_nsc, f"NSC-{clean_nsc}"]
                if not any(k in dti_s for k in keys_to_try):
                    dti_missing_cnt += 1
                    missing_dti_nscs.add(clean_nsc)
                if not any(k in pw_s for k in keys_to_try):
                    pw_missing_cnt += 1
                    missing_pw_nscs.add(clean_nsc)
        logger.info(
            f"Split '{split_name}' Bio Availability Audit ({len(records)} records) | "
            f"DTI-missing drug instances: {dti_missing_cnt} (NSCs: {sorted(list(missing_dti_nscs))}) | "
            f"Pathway-missing drug instances: {pw_missing_cnt} (NSCs: {sorted(list(missing_pw_nscs))})"
        )

    log_split_bio_availability("train", train_data, dti_store, pathway_store)
    log_split_bio_availability("val", val_data, dti_store, pathway_store)

    train_dataset = DrugComboDataset(
        train_data, cell_features, drug_feature_store=drug_features, descriptor_scaler=desc_scaler,
        feature_store_path=feature_store_path,
        dti_store=dti_store, pathway_store=pathway_store,
        use_pretrained_molformer=m_config.use_pretrained_molformer,
        molformer_model_name=m_config.molformer_model_name
    )
    val_dataset = DrugComboDataset(
        val_data, cell_features, drug_feature_store=drug_features, descriptor_scaler=desc_scaler,
        feature_store_path=feature_store_path,
        dti_store=dti_store, pathway_store=pathway_store,
        use_pretrained_molformer=m_config.use_pretrained_molformer,
        molformer_model_name=m_config.molformer_model_name
    )


    logger.info(
        "Dataset sizes | "
        f"train_records={len(train_data)} train_dataset={len(train_dataset)} "
        f"val_records={len(val_data)} val_dataset={len(val_dataset)}"
    )
    
    num_workers = getattr(t_config, "num_workers", 0)
    pin_mem = False
    
    loader_kwargs = {
        "batch_size": t_config.batch_size,
        "pin_memory": pin_mem,
        "num_workers": num_workers
    }
    if num_workers > 0 and os.name != 'nt':
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
        import torch.multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
            logger.info("  Set multiprocessing start method to 'spawn' for safe CUDA training with multiple workers.")
        except RuntimeError as e:
            logger.warning(f"  [WARNING] Could not set multiprocessing start method to 'spawn': {e}")
        
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    logger.info(
        "DataLoader sizes | "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={t_config.batch_size} num_workers={num_workers}"
    )
    
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    
    if engine == "lightning" and pl is None:
        logger.warning("PyTorch Lightning is not installed but '--engine lightning' was requested. Falling back to native PyTorch engine.")
        use_lightning = False
    else:
        use_lightning = (engine == "lightning" or (engine == "auto" and pl is not None and hasattr(pl, "Trainer")))
    
    if use_lightning:
        logger.info("Initializing LightningModule...")
        model = CancerComboLightningModule(m_config, t_config)
        checkpoint_callback = ModelCheckpoint(
            dirpath=t_config.checkpoint_dir,
            filename="cancercombo_best",
            save_top_k=t_config.save_top_k,
            monitor="val_loss",
            mode="min"
        )
        try:
            from pytorch_lightning.callbacks import TQDMProgressBar
            pbar_callback = TQDMProgressBar(refresh_rate=1, leave=True)
            callbacks_list = [checkpoint_callback, pbar_callback]
        except Exception:
            callbacks_list = [checkpoint_callback]
            
        logger.info(f"Starting PyTorch Lightning trainer fit on {accelerator.upper()} for {t_config.epochs} epochs...")
        trainer_kwargs = {
            "max_epochs": t_config.epochs,
            "accelerator": accelerator,
            "devices": 1,
            "gradient_clip_val": 5.0,
            "callbacks": callbacks_list,
            "enable_checkpointing": True,
            "log_every_n_steps": 1,
            "precision": "32-true"
        }
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume)
    else:
        logger.info(f"Starting Native PyTorch Training Engine on {accelerator.upper()} for {t_config.epochs} epochs...")
        if len(train_loader) == 0:
            logger.error(
                "Training DataLoader is empty. Epoch loop will not run. "
                "Check split parsing, cell-line matching, and filter conditions in dataset.py."
            )
            return
        from tqdm import tqdm
        from metrics import calculate_metrics
        import numpy as np
        import time

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = CancerCombo(m_config).to(device)
        loss_fn = CancerComboLoss()
        opt_name = getattr(t_config, "optimizer_name", "AdamW").lower()
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(net.parameters(), lr=t_config.lr, weight_decay=t_config.weight_decay, momentum=0.9)
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=t_config.lr, weight_decay=t_config.weight_decay)
        else:
            optimizer = torch.optim.AdamW(net.parameters(), lr=t_config.lr, weight_decay=t_config.weight_decay)
        
        sched_name = getattr(t_config, "scheduler_name", "ReduceLROnPlateau")
        if sched_name == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=t_config.scheduler_factor,
                patience=t_config.scheduler_patience,
                min_lr=getattr(t_config, "min_lr", 1.0e-6)
            )
        elif sched_name == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_config.epochs, eta_min=getattr(t_config, "min_lr", 1.0e-6)
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=t_config.scheduler_factor, patience=t_config.scheduler_patience, min_lr=getattr(t_config, "min_lr", 1.0e-6)
            )


        
        best_val_loss = float("inf")
        start_epoch = 1

        # Phase 10: Checkpoint Resume Support
        if resume and os.path.exists(resume):
            logger.info(f"Resuming checkpoint state from '{resume}'...")
            ckpt = torch.load(resume, map_location=device)
            if "state_dict" in ckpt:
                net.load_state_dict(ckpt["state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            logger.info(f"Resumed successfully at Epoch {start_epoch} (Best Val Loss: {best_val_loss:.6f}).")

        os.makedirs(t_config.checkpoint_dir, exist_ok=True)
        hook_handles = []
        if debug_backprop:
            hook_handles = _register_backward_debug_hooks(net, logger)

        try:
            for epoch in range(start_epoch, t_config.epochs + 1):
                epoch_start_time = time.time()
                net.train()
                train_loss_sum = 0.0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{t_config.epochs}", leave=True)

                last_grad_norm = 0.0
                last_pct_with_grad = 0.0
                last_pct_zero_grad = 0.0
                total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

                for batch_idx, batch in enumerate(pbar):
                    optimizer.zero_grad()
                    b_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    target = b_gpu.get("viability", b_gpu.get("viability_matrix"))
                    if target is None:
                        logger.warning(f"Batch {batch_idx} is missing target keys. Skipping.")
                        continue

                    if batch_idx == 0 and epoch == 1:
                        d_a_t = b_gpu.get("doses_a")
                        d_b_t = b_gpu.get("doses_b")
                        logger.info("=" * 60)
                        logger.info("FIRST BATCH TENSOR SHAPE VERIFICATION:")
                        logger.info(f"  doses_a shape         : {list(d_a_t.shape) if d_a_t is not None else None}")
                        logger.info(f"  doses_b shape         : {list(d_b_t.shape) if d_b_t is not None else None}")
                        logger.info(f"  viability_matrix shape: {list(target.shape) if target is not None else None}")
                        logger.info("=" * 60)

                    # Phase 3 Multimodal forward pass
                    y_pred, params = net(
                        drug_a_morgan=b_gpu.get("drug_a_morgan"),
                        drug_a_desc=b_gpu.get("drug_a_desc"),
                        drug_a_dti=b_gpu.get("drug_a_dti"),
                        drug_a_pathway=b_gpu.get("drug_a_pathway"),
                        drug_a_m_dti=b_gpu.get("drug_a_m_dti"),
                        drug_a_m_pathway=b_gpu.get("drug_a_m_pathway"),
                        drug_b_morgan=b_gpu.get("drug_b_morgan"),
                        drug_b_desc=b_gpu.get("drug_b_desc"),
                        drug_b_dti=b_gpu.get("drug_b_dti"),
                        drug_b_pathway=b_gpu.get("drug_b_pathway"),
                        drug_b_m_dti=b_gpu.get("drug_b_m_dti"),
                        drug_b_m_pathway=b_gpu.get("drug_b_m_pathway"),
                        cell_line=b_gpu.get("cell_line"),
                        doses_a=b_gpu.get("doses_a"),
                        doses_b=b_gpu.get("doses_b")
                    )

                    params_true = {p: b_gpu[p] for p in ["e1", "e2", "e3", "log_c1", "log_c2", "h1", "h2", "alpha"] if p in b_gpu}
                    loss = loss_fn(y_pred, target, params, params_true if params_true else None)

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0).item()
                    optimizer.step()

                    # Compute detailed parameter coverage diagnostics on first/last batch
                    if batch_idx == 0 or batch_idx == len(train_loader) - 1:
                        grads = [p.grad.detach() for p in net.parameters() if p.requires_grad and p.grad is not None]
                        if grads:
                            params_with_grad_count = sum(p.numel() for p in net.parameters() if p.requires_grad and p.grad is not None)
                            zero_grad_count = sum((g == 0).sum().item() for g in grads)
                            last_pct_with_grad = (params_with_grad_count / max(1, total_trainable_params)) * 100.0
                            last_pct_zero_grad = (zero_grad_count / max(1, total_trainable_params)) * 100.0
                        last_grad_norm = grad_norm

                    train_loss_sum += loss.item()
                    pbar.set_postfix({"train_loss": f"{loss.item():.4f}", "grad_norm": f"{grad_norm:.2f}"})


                train_loss = train_loss_sum / max(len(train_loader), 1)

                # Validation step
                net.eval()
                val_loss_sum = 0.0
                val_preds_list, val_trues_list = [], []
                val_param_records = {k: [] for k in ["e1", "e2", "e3", "log_c1", "log_c2", "h1", "h2", "alpha"]}

                with torch.no_grad():
                    for batch_idx_val, batch in enumerate(val_loader):
                        b_gpu = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        target_val = b_gpu.get("viability", b_gpu.get("viability_matrix"))
                        if target_val is None:
                            continue

                        y_pred, params = net(
                            drug_a_morgan=b_gpu.get("drug_a_morgan"),
                            drug_a_desc=b_gpu.get("drug_a_desc"),
                            drug_a_dti=b_gpu.get("drug_a_dti"),
                            drug_a_pathway=b_gpu.get("drug_a_pathway"),
                            drug_a_m_dti=b_gpu.get("drug_a_m_dti"),
                            drug_a_m_pathway=b_gpu.get("drug_a_m_pathway"),
                            drug_b_morgan=b_gpu.get("drug_b_morgan"),
                            drug_b_desc=b_gpu.get("drug_b_desc"),
                            drug_b_dti=b_gpu.get("drug_b_dti"),
                            drug_b_pathway=b_gpu.get("drug_b_pathway"),
                            drug_b_m_dti=b_gpu.get("drug_b_m_dti"),
                            drug_b_m_pathway=b_gpu.get("drug_b_m_pathway"),
                            cell_line=b_gpu.get("cell_line"),
                            doses_a=b_gpu.get("doses_a"),
                            doses_b=b_gpu.get("doses_b")
                        )
                        
                        v_loss = loss_fn(y_pred, target_val, params)
                        val_loss_sum += v_loss.item()
                        val_preds_list.append(y_pred.detach().cpu().numpy())
                        val_trues_list.append(target_val.detach().cpu().numpy())

                        # Task 4: Store Pharmacological Parameters for logging
                        e1_v, e2_v, e3_v, log_c1_v, log_c2_v, h1_v, h2_v, alpha_v = params
                        val_param_records["e1"].append(e1_v.detach().cpu().numpy())
                        val_param_records["e2"].append(e2_v.detach().cpu().numpy())
                        val_param_records["e3"].append(e3_v.detach().cpu().numpy())
                        val_param_records["log_c1"].append(log_c1_v.detach().cpu().numpy())
                        val_param_records["log_c2"].append(log_c2_v.detach().cpu().numpy())
                        val_param_records["h1"].append(h1_v.detach().cpu().numpy())
                        val_param_records["h2"].append(h2_v.detach().cpu().numpy())
                        val_param_records["alpha"].append(alpha_v.detach().cpu().numpy())

                val_loss = val_loss_sum / max(len(val_loader), 1)
                
                # Task 2: Step scheduler and log LR in scientific notation (e.g. LR = 1.234567890123e-05)
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                lr_str = f"{current_lr:.12e}"


                if val_preds_list:
                    v_preds = np.concatenate(val_preds_list, axis=0).flatten()
                    v_trues = np.concatenate(val_trues_list, axis=0).flatten()
                    val_metrics = calculate_metrics(v_preds, v_trues)
                    v_rmse = val_metrics["rmse"]
                    v_pearson = val_metrics["pearson"]
                    v_spearman = val_metrics["spearman"]
                else:
                    v_rmse, v_pearson, v_spearman = 0.0, 0.0, 0.0

                epoch_time = time.time() - epoch_start_time
                gpu_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if torch.cuda.is_available() else 0.0

                # Task 2, 3, 12: Log Epoch Metrics, LR in Scientific Notation, and Gradient Diagnostics
                logger.info(
                    f"Epoch [{epoch}/{t_config.epochs}] Complete | Time: {epoch_time:.2f}s | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val RMSE: {v_rmse:.4f} | Val Pearson: {v_pearson:.4f} | Val Spearman: {v_spearman:.4f} | "
                    f"LR = {lr_str}"
                )
                logger.info(
                    f"  [Gradient Diagnostics] Global Norm: {last_grad_norm:.6f} | Trainable Params: {total_trainable_params} | "
                    f"Params w/ Grad: {last_pct_with_grad:.2f}% | Zero Grads: {last_pct_zero_grad:.2f}% | GPU Mem: {gpu_mem_mb:.1f} MB"
                )

                # Task 4: Log Pharmacological Parameter Statistics (mean, std, min, max) every validation epoch
                param_stats_lines = []
                for p_name, p_list in val_param_records.items():
                    if p_list:
                        arr = np.concatenate(p_list, axis=0).flatten()
                        param_stats_lines.append(
                            f"{p_name:6s} -> mean: {arr.mean():.4f}, std: {arr.std():.4f}, min: {arr.min():.4f}, max: {arr.max():.4f}"
                        )
                logger.info("  [Pharmacological Parameters Val Stats]:\n    " + "\n    ".join(param_stats_lines))

                # Phase 10: Checkpoint Improvements (Save best model)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_path = os.path.join(t_config.checkpoint_dir, "cancercombo_best.ckpt")
                    torch.save({
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "val_loss": val_loss,
                        "val_rmse": v_rmse,
                        "val_pearson": v_pearson,
                        "val_spearman": v_spearman,
                        "current_lr": current_lr,
                        "config": m_config,
                    }, ckpt_path)
                    logger.info(f"Saved BEST checkpoint -> {ckpt_path} (Epoch {epoch}, Val Loss: {val_loss:.4f}, Val RMSE: {v_rmse:.4f})")

                if epoch % 200 == 0:
                    ckpt_path = os.path.join(t_config.checkpoint_dir, f"epoch_{epoch}.ckpt")
                    torch.save({
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_rmse": v_rmse,
                        "val_pearson": v_pearson,
                        "val_spearman": v_spearman,
                        "current_lr": current_lr,
                        "config": m_config,
                    }, ckpt_path)
                    logger.info(f"Saved periodic checkpoint -> {ckpt_path}")
        
        finally:
            for handle in hook_handles:
                handle.remove()
                
    logger.info("Training finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CancerCombo")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--scenario", type=int, default=1, help="Split scenario (1, 2, or 3)")
    parser.add_argument("--engine", type=str, default="native", choices=["auto", "lightning", "native"], help="Training engine: auto, lightning, or native (default: native — avoids PyTorch Lightning TQDMProgressBar hangs seen on this hardware)")

    parser.add_argument("--debug_backprop", action="store_true", help="Attach backward trace hooks to the native training path")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training")
    args = parser.parse_args()
    
    run_training(
        config_path=args.config,
        epochs=args.epochs,
        max_samples=args.max_samples,
        scenario=args.scenario,
        engine=args.engine,
        debug_backprop=args.debug_backprop,
        resume=args.resume
    )

