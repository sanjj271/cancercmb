import os
import re
import json
import ast
import zipfile
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from preprocessor import MolecularPreprocessor
from config import RAW_FEATURE_STORE_PATH
from typing import List, Dict, Tuple, Any, Optional

def _parse_list_or_json(val: Any) -> Any:
    """Parse a value into a list or numpy array, handling JSON strings, ast literal strings, and delimiter splits."""
    if val is None:
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    if isinstance(val, (int, float, np.number)):
        return [float(val)]
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str:
            return None
        # 1. Try json.loads
        try:
            res = json.loads(val_str)
            if isinstance(res, (list, tuple)):
                return list(res)
            if isinstance(res, (int, float)):
                return [float(res)]
        except Exception:
            pass
        # 2. Try ast.literal_eval
        try:
            res = ast.literal_eval(val_str)
            if isinstance(res, (list, tuple)):
                return list(res)
            if isinstance(res, (int, float)):
                return [float(res)]
        except Exception:
            pass
        # 3. Try splitting by comma
        cleaned = val_str.strip("[]() ")
        if cleaned:
            try:
                items = [float(x.strip()) for x in cleaned.split(",") if x.strip()]
                if items:
                    return items
            except Exception:
                pass
    return val


def _to_tensor(x: Any, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if x.dtype != dtype else x
    return torch.tensor(x, dtype=dtype)

SMILES_REGEX = r"(\[[^\]]+\]|Br|Cl|Si|Se|B|C|N|O|P|S|F|I|b|c|n|o|p|s|==|#|%[0-9]{2}|[0-9]|\+|-|=|/|\\|\@|\.|\(|\)|~|\*)"

class SMILESTokenizer:
    """Robust regex-based tokenizer for SMILES chemical structures."""
    
    def __init__(self, max_len: int = 128):
        self.max_len = max_len
        self.regex = re.compile(SMILES_REGEX)
        tokens = [
            '[PAD]', '[UNK]', 'C', 'c', 'O', 'o', 'N', 'n', 'S', 's', 'F', 'P', 'p',
            'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'H', '=', '#', '(', ')', '[', ']',
            '/', '\\', '+', '-', '@', '.', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'
        ]
        self.vocab = {t: idx for idx, t in enumerate(tokens)}
        self.pad_idx = self.vocab['[PAD]']
        self.unk_idx = self.vocab['[UNK]']

    def tokenize(self, smiles: str) -> Tuple[List[int], List[int]]:
        """Tokenize a SMILES string using regex matching, mapping tokens to IDs.

        Args:
            smiles: Raw SMILES string.

        Returns:
            Tuple[List[int], List[int]]: Token IDs and attention mask.
        """
        matched_tokens = self.regex.findall(smiles)
        ids = [self.vocab.get(t, self.unk_idx) for t in matched_tokens]
        
        # Padding & Truncation
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            attention_mask = [1] * self.max_len
        else:
            padding_len = self.max_len - len(ids)
            attention_mask = [1] * len(ids) + [0] * padding_len
            ids = ids + [self.pad_idx] * padding_len
            
        return ids, attention_mask


def load_nci60_gex(csv_path: str = "data/features/NCI-60_landmark_gex.csv", target_dim: int = 976) -> Dict[str, np.ndarray]:
    """Load cell line gene expression matrix from NCI-60 CSV file.

    Args:
        csv_path: Path to NCI-60 gene expression CSV file.
        target_dim: Expected dimension size (for validation only).

    Returns:
        Dict[str, np.ndarray]: Dict mapping cell line names to float32 expression vectors.
    """
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        # First column is gene names, remaining columns are canonical cell line names
        cell_lines = df.columns[1:]
        # Impute missing gene values with gene-wise median across cell lines
        df[cell_lines] = df[cell_lines].apply(lambda row: row.fillna(row.median()), axis=1)
        df[cell_lines] = df[cell_lines].fillna(0.0)
        
        gex_dict = {}
        for cell in cell_lines:
            vec = df[cell].values.astype(np.float32)
            if target_dim and len(vec) != target_dim:
                raise ValueError(f"Gene expression vector for {cell} has dimension {len(vec)}, expected {target_dim}")
            
            # Map canonical name and normalized name
            gex_dict[cell] = vec
            norm_key = re.sub(r'[^A-Z0-9]', '', str(cell).upper())
            gex_dict[norm_key] = vec
        return gex_dict
    except Exception as e:
        print(f"Error loading gene expression: {e}")
        return {}


def resolve_cell_line(sample_str: str, known_cells_map: Dict[str, str]) -> Optional[str]:
    """Robustly extract biological cell line name from a complex Sample string and return canonical name.
    
    Args:
        sample_str: The full sample string (e.g., 'DrugA_DrugB_786-0', 'DrugA_DrugB_A549/ATCC', 'DrugA_DrugB_HL-60(TB)')
        known_cells_map: Mapping from normalized cell line string to canonical cell line name.
        
    Returns:
        Optional[str]: The canonical cell line name if matched, else None.
    """
    if not isinstance(sample_str, str) or not sample_str.strip():
        return None
    s_clean = re.sub(r'\(.*?\)', '', str(sample_str)).strip()
    
    # Protect NCI/ADR-RES while handling provider slash suffixes (e.g. A549/ATCC -> A549)
    if '/' in s_clean and 'NCI/ADR' not in s_clean.upper():
        s_clean = s_clean.split('/')[0].strip()
        
    norm_sample = re.sub(r'[^A-Z0-9]', '', s_clean.upper())
    
    # Sort normalized keys by length descending to match longest cell line name first (preventing prefix collision)
    sorted_keys = sorted(known_cells_map.keys(), key=len, reverse=True)
    for norm_k in sorted_keys:
        if norm_sample.endswith(norm_k):
            return known_cells_map[norm_k]
            
    return None


def match_cell_line(sample_str: str, known_cells_map: Dict[str, str]) -> Optional[str]:
    """Alias to resolve_cell_line for backward compatibility."""
    return resolve_cell_line(sample_str, known_cells_map)


import hashlib
from typing import Union


def normalize_clean_nsc(nsc: Any) -> str:
    """Normalize raw NSC string to clean integer digits string."""
    raw = str(nsc).strip()
    return re.sub(r'^(NSC[-_]?)', '', raw, flags=re.IGNORECASE).strip()


def extract_scenario_nscs(
    split_df_or_path: Union[pd.DataFrame, str]
) -> Tuple[List[str], List[str], List[str]]:
    """Extract strictly partitioned unique NSC IDs for TRAIN (split 1), VAL (split 2), and TEST (split 3).
    
    Reads existing scenario split data and extracts clean canonical NSC IDs across all known NSC columns:
    ['nsc_a', 'nsc_b', 'Drug1_NSC_ID', 'Drug2_NSC_ID', 'drug1', 'drug2'].
    
    Returns:
        Tuple[List[str], List[str], List[str]]: (scenario_train_nscs, scenario_val_nscs, scenario_test_nscs)
    """
    if isinstance(split_df_or_path, str):
        if not os.path.exists(split_df_or_path):
            raise FileNotFoundError(f"Scenario split CSV missing at: '{split_df_or_path}'")
        split_df = pd.read_csv(split_df_or_path)
    else:
        split_df = split_df_or_path

    if "split" not in split_df.columns:
        raise ValueError("Split DataFrame does not contain required 'split' column!")

    nsc_cols = [c for c in ["nsc_a", "nsc_b", "Drug1_NSC_ID", "Drug2_NSC_ID", "drug1", "drug2"] if c in split_df.columns]
    if not nsc_cols:
        raise ValueError(f"No recognizable NSC ID column found in split data! Available columns: {list(split_df.columns)}")

    def _get_nscs_for_split(val_code: int) -> List[str]:
        sub = split_df[split_df["split"] == val_code]
        nsc_set = set()
        for c in nsc_cols:
            for v in sub[c].dropna():
                clean_k = normalize_clean_nsc(v)
                if clean_k:
                    nsc_set.add(clean_k)
        return sorted(list(nsc_set))

    train_nscs = _get_nscs_for_split(1)
    val_nscs = _get_nscs_for_split(2)
    test_nscs = _get_nscs_for_split(3)

    return train_nscs, val_nscs, test_nscs


class DescriptorScaler:
    """Scaler for 200 continuous RDKit physical descriptors fitted on TRAIN drugs only.
    
    Pipeline:
      detect NaN/+Inf/-Inf -> TRAIN median imputation -> TRAIN mean/std standardization -> clip [-10, 10] -> assert finite
    """
    def __init__(self):
        self.median_vec = None
        self.mean_vec = None
        self.std_vec = None
        self.is_fitted = False

    def fit(self, raw_descriptors_list: List[Any]):
        """Fit median, mean, and std statistics using TRAIN-drug descriptors only."""
        if not raw_descriptors_list:
            return
        clean_list = []
        for x in raw_descriptors_list:
            if isinstance(x, torch.Tensor):
                clean_list.append(x.detach().cpu().numpy())
            else:
                clean_list.append(np.asarray(x))
        arr = np.array(clean_list, dtype=np.float64)
        arr[np.isinf(arr)] = np.nan
        arr = np.clip(arr, -1e6, 1e6)
        
        median_vec64 = np.nanmedian(arr, axis=0)
        median_vec64 = np.nan_to_num(median_vec64, nan=0.0)
        
        imputed_arr = arr.copy()
        inds = np.where(np.isnan(imputed_arr))
        imputed_arr[inds] = np.take(median_vec64, inds[1])
        
        mean_vec64 = np.mean(imputed_arr, axis=0)
        std_vec64 = np.std(imputed_arr, axis=0)
        std_vec64[std_vec64 < 1e-8] = 1.0

        self.median_vec = median_vec64.astype(np.float32)
        self.mean_vec = mean_vec64.astype(np.float32)
        self.std_vec = std_vec64.astype(np.float32)
        self.is_fitted = True

    def fit_from_feature_store(
        self,
        feature_store: Dict[str, Any],
        train_nsc_ids: List[str]
    ):
        """Fit median, mean, and std statistics strictly on TRAIN drugs using explicit NSC mapping.
        
        No positional correspondence, ordering, or row-position assumptions allowed.
        Fails loudly if any Train NSC ID is missing from the feature store.
        """
        if not train_nsc_ids:
            raise ValueError("Cannot fit DescriptorScaler on empty training NSC list!")
        train_raw_descs = []
        for nsc in train_nsc_ids:
            clean_nsc = normalize_clean_nsc(nsc)
            feat = (
                feature_store.get(nsc) or
                feature_store.get(f"NSC-{nsc}") or
                feature_store.get(clean_nsc) or
                feature_store.get(f"NSC-{clean_nsc}")
            )
            if feat is not None:
                desc = feat.get("descriptors", feat.get("desc")) if isinstance(feat, dict) else feat[1]
                if isinstance(desc, torch.Tensor):
                    desc = desc.detach().cpu().numpy()
                train_raw_descs.append(np.array(desc, dtype=np.float32))
            else:
                raise KeyError(
                    f"Train NSC '{nsc}' (normalized '{clean_nsc}') not found in molecular feature store! "
                    f"Explicit NSC mapping failed. Positional or silent substitution is strictly prohibited."
                )
        self.fit(train_raw_descs)

    def transform(self, raw_desc: Any) -> np.ndarray:
        """Transform raw descriptor vector using fitted TRAIN-drug statistics."""
        if not self.is_fitted or self.median_vec is None:
            raise RuntimeError("Cannot transform using an unfitted DescriptorScaler!")
        if isinstance(raw_desc, torch.Tensor):
            raw_desc = raw_desc.detach().cpu().numpy()
        vec = np.asarray(raw_desc, dtype=np.float32).copy()
        vec[np.isinf(vec)] = np.nan
        vec = np.clip(vec, -1e6, 1e6)
        
        nans = np.isnan(vec)
        vec[nans] = self.median_vec[nans]
        vec = (vec - self.mean_vec) / self.std_vec
        vec = np.clip(vec, -10.0, 10.0)
            
        assert np.all(np.isfinite(vec)), "Non-finite descriptor value detected after scaling!"
        return vec.astype(np.float32)


def compute_nsc_hash(train_nsc_ids: List[str]) -> str:
    """Compute deterministic SHA256 hash of sorted unique Train NSC IDs."""
    sorted_unique = sorted(list(set(normalize_clean_nsc(x) for x in train_nsc_ids if str(x).strip())))
    return hashlib.sha256(",".join(sorted_unique).encode("utf-8")).hexdigest()


def save_descriptor_scaler(scaler: DescriptorScaler, path: str, train_nsc_ids: List[str]):
    """Persist DescriptorScaler parameters and Train NSC hash as plain NPZ arrays."""
    if not scaler.is_fitted or scaler.median_vec is None:
        raise ValueError("Cannot save an unfitted DescriptorScaler!")
        
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    train_hash = compute_nsc_hash(train_nsc_ids)
    np.savez(
        path,
        median_vec=scaler.median_vec,
        mean_vec=scaler.mean_vec,
        std_vec=scaler.std_vec,
        train_nsc_hash=np.array([train_hash]),
        num_train_nscs=np.array([len(set(train_nsc_ids))])
    )


def load_descriptor_scaler(path: str, train_nsc_ids: Optional[List[str]] = None) -> DescriptorScaler:
    """Load persisted DescriptorScaler parameters and validate finite values and Train NSC hash."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required DescriptorScaler artifact missing at '{path}'. Fail loudly.")
        
    if train_nsc_ids is None:
        raise ValueError("train_nsc_ids must be provided to load_descriptor_scaler to verify scenario correspondence.")

    data = np.load(path, allow_pickle=True)
    if "median_vec" not in data or "mean_vec" not in data or "std_vec" not in data:
        raise ValueError(f"Corrupted DescriptorScaler file at '{path}': missing required arrays.")
        
    median_vec = data["median_vec"].astype(np.float32)
    mean_vec = data["mean_vec"].astype(np.float32)
    std_vec = data["std_vec"].astype(np.float32)
    
    if len(median_vec) != 200 or len(mean_vec) != 200 or len(std_vec) != 200:
        raise ValueError(f"DescriptorScaler array dimension mismatch: expected 200, got {len(median_vec)}")
        
    if not (np.all(np.isfinite(median_vec)) and np.all(np.isfinite(mean_vec)) and np.all(np.isfinite(std_vec))):
        raise ValueError(f"DescriptorScaler at '{path}' contains non-finite values.")
        
    if "train_nsc_hash" not in data:
        raise ValueError(f"Corrupted or legacy DescriptorScaler file at '{path}': missing required 'train_nsc_hash'.")

    expected_hash = compute_nsc_hash(train_nsc_ids)
    saved_hash = str(data["train_nsc_hash"][0])
    if expected_hash != saved_hash:
        raise ValueError(
            f"DescriptorScaler Train NSC hash mismatch!\n"
            f"  Saved Scaler Hash : {saved_hash}\n"
            f"  Expected Hash     : {expected_hash}"
        )
        
    scaler = DescriptorScaler()
    scaler.median_vec = median_vec
    scaler.mean_vec = mean_vec
    scaler.std_vec = std_vec
    scaler.is_fitted = True
    return scaler


def parse_dataframe_to_records(df: pd.DataFrame, known_gex_dict: Dict[str, np.ndarray] = None) -> List[Dict[str, Any]]:
    """Parse pandas DataFrame into a list of 2D dose-response matrix records grouped by Sample.
    
    Args:
        df: Input dataframe containing drug and dose-response data.
        known_gex_dict: Dictionary mapping cell lines to gene expression vectors.
        
    Returns:
        List[Dict]: List of parsed sample records containing 4x4 viability matrices.
    """
    if df.empty:
        return []
        
    # Build normalized lookup for cell matching
    known_cells_map = {}
    if known_gex_dict:
        for k in known_gex_dict.keys():
            norm_k = re.sub(r'[^A-Z0-9]', '', str(k).upper())
            known_cells_map[norm_k] = k

    # First check if the dataframe is already pre-grouped records (e.g., has doses_a/dose_a as list or matrix)
    if any(col in df.columns for col in ['viability_matrix', 'viability', 'doses_a', 'dose_a', 'doses_b', 'dose_b']):
        records = df.to_dict('records')
        data_list = []
        for row in records:
            s_a = row.get('smiles_a', row.get('Drug1_SMILES', row.get('drug_a', '')))
            s_b = row.get('smiles_b', row.get('Drug2_SMILES', row.get('drug_b', '')))
            nsc_a = str(row.get('nsc_a', row.get('Drug1_NSC_ID', ''))).strip()
            nsc_b = str(row.get('nsc_b', row.get('Drug2_NSC_ID', ''))).strip()
            cell_raw = str(row.get('cell_line_name', row.get('Sample', row.get('cell_line', ''))))
            
            if pd.isna(s_a) or pd.isna(s_b) or not str(s_a).strip() or not str(s_b).strip():
                continue
                
            cell = match_cell_line(cell_raw, known_cells_map) if known_cells_map else cell_raw
            if known_cells_map and (cell is None or cell not in known_gex_dict):
                continue
                
            d_a = row.get('doses_a', row.get('dose_a'))
            d_b = row.get('doses_b', row.get('dose_b'))
            viab = row.get('viability_matrix', row.get('viability'))
            
            d_a_parsed = _parse_list_or_json(d_a)
            d_b_parsed = _parse_list_or_json(d_b)
            viab_parsed = _parse_list_or_json(viab)

            if d_a_parsed is None or d_b_parsed is None or viab_parsed is None:
                continue
                
            data_list.append({
                "smiles_a": str(s_a),
                "smiles_b": str(s_b),
                "nsc_a": nsc_a,
                "nsc_b": nsc_b,
                "cell_line_name": str(cell),
                "doses_a": d_a_parsed,
                "doses_b": d_b_parsed,
                "viability_matrix": viab_parsed
            })
        return data_list

    # Vectorized fast grouping for raw 16-row per Sample dataframe
    df_clean = df.dropna(subset=['Response']).copy()
    df_sorted = df_clean.sort_values(['Sample', 'Drug1_Dose', 'Drug2_Dose']).reset_index(drop=True)
    
    first_rows = df_sorted.drop_duplicates(subset=['Sample'], keep='first').copy()
    
    matched_count = 0
    unmatched_count = 0
    unique_cells = set()
    unmatched_names = set()
    
    sample_to_cell = {}
    for sample_name in first_rows['Sample'].values:
        if known_cells_map:
            cell = match_cell_line(sample_name, known_cells_map)
            if cell is None or cell not in known_gex_dict:
                unmatched_count += 1
                unmatched_names.add(sample_name)
                continue
            matched_count += 1
            unique_cells.add(cell)
            sample_to_cell[sample_name] = cell
        else:
            sample_to_cell[sample_name] = str(sample_name)

    valid_samples = [s for s in first_rows['Sample'].values if s in sample_to_cell]
    valid_set = set(valid_samples)
    
    df_valid = df_sorted[df_sorted['Sample'].isin(valid_set)].copy()
    
    sample_counts = df_valid['Sample'].value_counts()
    valid_16_samples = set(sample_counts[sample_counts == 16].index)
    df_valid = df_valid[df_valid['Sample'].isin(valid_16_samples)].copy()
    
    first_rows_valid = first_rows[first_rows['Sample'].isin(valid_16_samples)].copy()
    
    s_a_col = 'Drug1_SMILES' if 'Drug1_SMILES' in first_rows_valid.columns else 'Drug1'
    s_b_col = 'Drug2_SMILES' if 'Drug2_SMILES' in first_rows_valid.columns else 'Drug2'
    
    grouped = df_valid.groupby('Sample', sort=False)
    data_list = []
    
    for sample_name, sample_df in grouped:
        if sample_name not in sample_to_cell:
            continue
        cell = sample_to_cell[sample_name]
        
        row_first = sample_df.iloc[0]
        s_a = str(row_first[s_a_col])
        s_b = str(row_first[s_b_col])
        nsc_a = str(row_first['Drug1_NSC_ID']).strip() if 'Drug1_NSC_ID' in row_first else ''
        nsc_b = str(row_first['Drug2_NSC_ID']).strip() if 'Drug2_NSC_ID' in row_first else ''

        if pd.isna(s_a) or pd.isna(s_b) or not s_a.strip() or not s_b.strip():
            continue
            
        d1_doses = np.sort(sample_df['Drug1_Dose'].unique())
        d2_doses = np.sort(sample_df['Drug2_Dose'].unique())
        
        if len(d1_doses) != 4 or len(d2_doses) != 4:
            continue
            
        coord_map = {}
        valid_grid = True
        for da_val, db_val, resp_val in zip(sample_df['Drug1_Dose'].values, sample_df['Drug2_Dose'].values, sample_df['Response'].values):
            da_f, db_f, resp_f = float(da_val), float(db_val), float(resp_val)
            if np.isnan(resp_f) or np.isinf(resp_f):
                valid_grid = False
                break
            coord_map[(da_f, db_f)] = resp_f
            
        if not valid_grid or len(coord_map) != 16:
            continue
            
        matrix = np.zeros((4, 4), dtype=np.float32)
        complete = True
        for i, da in enumerate(d1_doses):
            for j, db in enumerate(d2_doses):
                coord_key = (float(da), float(db))
                if coord_key not in coord_map:
                    complete = False
                    break
                matrix[i, j] = coord_map[coord_key]
            if not complete:
                break
                
        if not complete:
            continue
            
        data_list.append({
            "smiles_a": s_a,
            "smiles_b": s_b,
            "nsc_a": nsc_a,
            "nsc_b": nsc_b,
            "cell_line_name": cell,
            "doses_a": d1_doses.tolist(),
            "doses_b": d2_doses.tolist(),
            "viability_matrix": matrix.tolist()
        })

    return data_list


def load_dti_embeddings(excel_path: str = "data/drug_target_embeddings_2048.xlsx") -> Dict[str, np.ndarray]:
    """Load pre-pooled 2048-D DTI embeddings keyed by canonical NSC ID with strict validation."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Required DTI embedding file missing at '{excel_path}'. Fail loudly.")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        raise ValueError(f"Failed to read DTI Excel file at '{excel_path}': {e}")

    if "NSC_ID" not in df.columns:
        raise ValueError(f"DTI embedding table at '{excel_path}' missing required 'NSC_ID' column. Fail loudly.")

    feat_cols = [c for c in df.columns if str(c).startswith("dti_emb_")]
    if len(feat_cols) != 2048:
        raise ValueError(f"DTI embedding table at '{excel_path}' column count mismatch: expected exactly 2048 'dti_emb_*' columns, got {len(feat_cols)}.")

    dti_dict = {}
    seen_clean_nscs = set()

    for idx, row in df.iterrows():
        raw_nsc = str(row["NSC_ID"]).strip()
        clean_nsc = re.sub(r'^(NSC[-_]?)', '', raw_nsc, flags=re.IGNORECASE).strip()
        if not clean_nsc:
            raise ValueError(f"Invalid/empty NSC_ID at row {idx} in '{excel_path}'. Fail loudly.")

        if clean_nsc in seen_clean_nscs:
            raise ValueError(f"Duplicate NSC detected in DTI table: '{raw_nsc}' (normalized '{clean_nsc}') at row {idx}. Fail loudly.")
        seen_clean_nscs.add(clean_nsc)

        vec = row[feat_cols].values.astype(np.float32)
        if len(vec) != 2048:
            raise ValueError(f"Inconsistent DTI vector length for NSC '{raw_nsc}': expected 2048, got {len(vec)}.")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"Non-finite values (NaN/Inf) detected in DTI vector for NSC '{raw_nsc}'. Fail loudly.")

        dti_dict[raw_nsc] = vec
        dti_dict[clean_nsc] = vec
        dti_dict[f"NSC-{clean_nsc}"] = vec

    return dti_dict


def load_pathway_embeddings(excel_path: str = "data/drug_pathway_embeddings_1152.xlsx") -> Dict[str, np.ndarray]:
    """Load precomputed 1152-D pathway embeddings keyed by canonical NSC ID with strict validation."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Required Pathway embedding file missing at '{excel_path}'. Fail loudly.")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        raise ValueError(f"Failed to read Pathway Excel file at '{excel_path}': {e}")

    if "NSC_ID" not in df.columns:
        raise ValueError(f"Pathway embedding table at '{excel_path}' missing required 'NSC_ID' column. Fail loudly.")

    feat_cols = [c for c in df.columns if str(c).startswith("embedding_")]
    if len(feat_cols) != 1152:
        raise ValueError(f"Pathway embedding table at '{excel_path}' column count mismatch: expected exactly 1152 'embedding_*' columns, got {len(feat_cols)}.")

    pw_dict = {}
    seen_clean_nscs = set()

    for idx, row in df.iterrows():
        raw_nsc = str(row["NSC_ID"]).strip()
        clean_nsc = re.sub(r'^(NSC[-_]?)', '', raw_nsc, flags=re.IGNORECASE).strip()
        if not clean_nsc:
            raise ValueError(f"Invalid/empty NSC_ID at row {idx} in '{excel_path}'. Fail loudly.")

        if clean_nsc in seen_clean_nscs:
            raise ValueError(f"Duplicate NSC detected in Pathway table: '{raw_nsc}' (normalized '{clean_nsc}') at row {idx}. Fail loudly.")
        seen_clean_nscs.add(clean_nsc)

        vec = row[feat_cols].values.astype(np.float32)
        if len(vec) != 1152:
            raise ValueError(f"Inconsistent Pathway vector length for NSC '{raw_nsc}': expected 1152, got {len(vec)}.")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"Non-finite values (NaN/Inf) detected in Pathway vector for NSC '{raw_nsc}'. Fail loudly.")

        pw_dict[raw_nsc] = vec
        pw_dict[clean_nsc] = vec
        pw_dict[f"NSC-{clean_nsc}"] = vec

    return pw_dict


class DrugComboDataset(Dataset):
    """PyTorch Dataset class representing cell line and drug combination inputs."""
    
    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        cell_line_features: Dict[str, np.ndarray],
        drug_feature_store: Dict[str, Dict[str, Any]] | None = None,
        descriptor_scaler: DescriptorScaler | None = None,
        feature_store_path: str = RAW_FEATURE_STORE_PATH,
        dti_store: Dict[str, np.ndarray] | None = None,
        pathway_store: Dict[str, np.ndarray] | None = None,
        max_smiles_len: int = 128,
        use_pretrained_molformer: bool = False,
        molformer_model_name: str = "ibm/MoLFormer-XL-CIMA-100M"
    ):
        self.data = data_list
        self.cell_line_features = cell_line_features
        self.drug_feature_store = drug_feature_store
        self.dti_store = dti_store if dti_store is not None else {}
        self.pathway_store = pathway_store if pathway_store is not None else {}
        if descriptor_scaler is None:
            raise ValueError("DrugComboDataset requires a fitted DescriptorScaler; descriptor_scaler cannot be None.")
        if not getattr(descriptor_scaler, "is_fitted", False):
            raise ValueError("DrugComboDataset requires a fitted DescriptorScaler; provided scaler is not fitted.")
        self.descriptor_scaler = descriptor_scaler
        self.feature_store_path = feature_store_path
        self.max_smiles_len = max_smiles_len
        self.tokenizer = SMILESTokenizer(max_len=max_smiles_len)
        self.preprocessor = MolecularPreprocessor()
        self.use_pretrained_molformer = use_pretrained_molformer
        self.hf_tokenizer = None
        if self.use_pretrained_molformer:
            try:
                from transformers import AutoTokenizer
                self.hf_tokenizer = AutoTokenizer.from_pretrained(molformer_model_name, trust_remote_code=True)
            except Exception as e:
                print(f"Warning: Failed to load pretrained HF AutoTokenizer for '{molformer_model_name}': {e}. Falling back to custom SMILESTokenizer.")
                self.use_pretrained_molformer = False

    def _tokenize_smiles(self, smiles: str) -> Tuple[List[int], List[int]]:
        if self.use_pretrained_molformer and self.hf_tokenizer is not None:
            encoded = self.hf_tokenizer(smiles, max_length=self.max_smiles_len, padding="max_length", truncation=True)
            return encoded["input_ids"], encoded["attention_mask"]
        return self.tokenizer.tokenize(smiles)
        
    def _get_drug_features(self, nsc_id: str, smiles: str) -> Tuple[Any, Any]:
        """Fetch drug features using NSC ID primary key with strict validation.
        
        No on-the-fly dynamic generation or zero-vector fallbacks allowed for the active pipeline.
        """
        if self.drug_feature_store is None or len(self.drug_feature_store) == 0:
            raise ValueError(f"Required Morgan 1024 feature store at '{self.feature_store_path}' is empty or missing.")

        feat = None
        if nsc_id:
            feat = self.drug_feature_store.get(nsc_id) or self.drug_feature_store.get(f"NSC-{nsc_id}")
            
        if feat is None and smiles:
            feat = self.drug_feature_store.get(smiles)

        if feat is None:
            missing_id = nsc_id if nsc_id else smiles
            raise KeyError(
                f"Missing molecular feature for NSC {missing_id} in {self.feature_store_path}. "
                f"On-the-fly feature generation is disabled for the active pipeline."
            )

        morgan, desc = None, None
        if isinstance(feat, dict):
            morgan = feat.get("morgan", feat.get("morgan_fp", feat.get("fingerprint")))
            desc = feat.get("descriptors", feat.get("desc"))
        elif isinstance(feat, (tuple, list)) and len(feat) >= 2:
            morgan, desc = feat[0], feat[1]

        if morgan is None or desc is None:
            raise KeyError(
                f"Corrupted molecular feature vector for NSC {nsc_id} in {self.feature_store_path}."
            )

        if len(morgan) != 1024:
            raise ValueError(
                f"Morgan feature store dimension mismatch: expected 1024, got {len(morgan)} in {self.feature_store_path}."
            )

        scaled_desc = self.descriptor_scaler.transform(desc)
        return morgan, scaled_desc

    def _get_bio_features(self, nsc_id: str, smiles: str) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """Fetch biological embeddings (DTI and Pathway) with availability mask."""
        clean_nsc = re.sub(r'^(NSC[-_]?)', '', str(nsc_id).strip(), flags=re.IGNORECASE).strip()
        keys_to_try = [str(nsc_id).strip(), clean_nsc, f"NSC-{clean_nsc}"]

        # DTI
        dti_vec = None
        for k in keys_to_try:
            if k in self.dti_store:
                dti_vec = self.dti_store[k]
                break
        if dti_vec is not None and len(dti_vec) == 2048:
            dti_arr = np.array(dti_vec, dtype=np.float32)
            m_dti = 1.0
        else:
            dti_arr = np.zeros(2048, dtype=np.float32)
            m_dti = 0.0

        # Pathway
        pathway_vec = None
        for k in keys_to_try:
            if k in self.pathway_store:
                pathway_vec = self.pathway_store[k]
                break
        if pathway_vec is not None and len(pathway_vec) == 1152:
            pathway_arr = np.array(pathway_vec, dtype=np.float32)
            m_pathway = 1.0
        else:
            pathway_arr = np.zeros(1152, dtype=np.float32)
            m_pathway = 0.0

        return dti_arr, m_dti, pathway_arr, m_pathway

    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        smiles_a = item['smiles_a']
        smiles_b = item['smiles_b']
        nsc_a = item.get('nsc_a', '')
        nsc_b = item.get('nsc_b', '')
        cell_name = item['cell_line_name']
        doses_a = np.array(item['doses_a'], dtype=np.float32) * 1e6
        doses_b = np.array(item['doses_b'], dtype=np.float32) * 1e6
        viability = np.array(item['viability_matrix'], dtype=np.float32)
        if viability.ndim == 1 and viability.size == 16:
            viability = viability.reshape(4, 4)
        
        morgan_a, desc_a = self._get_drug_features(nsc_a, smiles_a)
        morgan_b, desc_b = self._get_drug_features(nsc_b, smiles_b)

        dti_a, m_dti_a, pw_a, m_pw_a = self._get_bio_features(nsc_a, smiles_a)
        dti_b, m_dti_b, pw_b, m_pw_b = self._get_bio_features(nsc_b, smiles_b)

        norm_cell = re.sub(r'[^A-Z0-9]', '', str(cell_name).upper())
        cell_vec = self.cell_line_features.get(cell_name, self.cell_line_features.get(norm_cell))
        if cell_vec is None:
            raise KeyError(f"Cell line features for '{cell_name}' not found. Do not zero-pad.")
        assert cell_vec.shape[-1] == 976, f"Expected 976-dim gene expression vector, got {cell_vec.shape[-1]}"

        res_dict = {
            "drug_a_morgan": _to_tensor(morgan_a, dtype=torch.float32),
            "drug_a_desc": _to_tensor(desc_a, dtype=torch.float32),
            "drug_a_dti": _to_tensor(dti_a, dtype=torch.float32),
            "drug_a_pathway": _to_tensor(pw_a, dtype=torch.float32),
            "drug_a_m_dti": torch.tensor(m_dti_a, dtype=torch.float32),
            "drug_a_m_pathway": torch.tensor(m_pw_a, dtype=torch.float32),

            "drug_b_morgan": _to_tensor(morgan_b, dtype=torch.float32),
            "drug_b_desc": _to_tensor(desc_b, dtype=torch.float32),
            "drug_b_dti": _to_tensor(dti_b, dtype=torch.float32),
            "drug_b_pathway": _to_tensor(pw_b, dtype=torch.float32),
            "drug_b_m_dti": torch.tensor(m_dti_b, dtype=torch.float32),
            "drug_b_m_pathway": torch.tensor(m_pw_b, dtype=torch.float32),

            "cell_line": _to_tensor(cell_vec, dtype=torch.float32),
            "doses_a": _to_tensor(doses_a, dtype=torch.float32),
            "doses_b": _to_tensor(doses_b, dtype=torch.float32),
            "viability": _to_tensor(viability, dtype=torch.float32)
        }

        for p in ["e1", "e2", "e3", "log_c1", "log_c2", "h1", "h2", "alpha"]:
            if p in item:
                res_dict[p] = _to_tensor([float(item[p])], dtype=torch.float32)
                
        return res_dict


def load_synergy_dataset(zip_or_csv_path: str = "data/DrugCombination_with_SMILES.zip", known_cells: set = None) -> List[Dict[str, Any]]:
    """Load drug combination dataset from ZIP or CSV archive.

    Args:
        zip_or_csv_path: Path to dataset ZIP archive or CSV file.
        known_cells: Optional set of known cell lines.

    Returns:
        List[Dict[str, Any]]: Parsed list of sample dictionaries.
    """
    if not os.path.exists(zip_or_csv_path):
        return []
    try:
        if zip_or_csv_path.endswith('.zip'):
            with zipfile.ZipFile(zip_or_csv_path, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    return []
                dfs = []
                for csv_file in csv_files:
                    with z.open(csv_file) as f:
                        dfs.append(pd.read_csv(f))
                df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(zip_or_csv_path)
            
        return parse_dataframe_to_records(df, known_cells)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def load_precomputed_drug_features(feature_path: str, config: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    """Load a saved drug feature store from .pt or .pkl with strict metadata and dimension validation."""
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Required Morgan feature store artifact missing at '{feature_path}'")

    store = {}
    try:
        if feature_path.endswith(".pt"):
            store = torch.load(feature_path, map_location="cpu", weights_only=False)
        elif feature_path.endswith(".pkl"):
            with open(feature_path, "rb") as f:
                store = pickle.load(f)
        else:
            raise ValueError(f"Unsupported feature store extension: {feature_path}")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise e
        raise ValueError(f"Failed to load feature store from '{feature_path}': {e}")

    if not store or not isinstance(store, dict):
        raise ValueError(f"Feature store at '{feature_path}' is empty or invalid!")

    # 1. Require metadata header
    metadata = store.get("metadata") or store.get("__metadata__")
    if not metadata or not isinstance(metadata, dict):
        raise ValueError(f"Feature store at '{feature_path}' is missing required metadata header! Fail loudly.")

    # Determine expected dimensions from config (or default to active config 1024 / r=3)
    expected_nbits = getattr(config, "morgan_in_dim", 1024) if config else 1024
    expected_radius = getattr(config, "morgan_radius", 3) if config else 3

    meta_nbits = metadata.get("morgan_nbits")
    meta_radius = metadata.get("morgan_radius")
    key_type = metadata.get("key_type")

    # 2. Validate metadata fields against active configuration
    if meta_nbits is None or meta_nbits != expected_nbits:
        raise ValueError(
            f"Morgan feature store dimension mismatch: artifact has {meta_nbits}-bit, "
            f"active configuration requires {expected_nbits}-bit in '{feature_path}'."
        )

    if meta_radius is None or meta_radius != expected_radius:
        raise ValueError(
            f"Morgan feature store radius mismatch: artifact has radius {meta_radius}, "
            f"active configuration requires radius {expected_radius} in '{feature_path}'."
        )

    if key_type != "NSC":
        raise ValueError(
            f"Feature store key_type mismatch: expected 'NSC', got '{key_type}' in '{feature_path}'."
        )

    # 3. Require descriptor_names and rdkit_version in metadata
    meta_desc_names = metadata.get("descriptor_names")
    if meta_desc_names is None or not isinstance(meta_desc_names, (list, tuple)):
        raise ValueError(
            f"Feature store at '{feature_path}' is missing required 'descriptor_names' in metadata header! Fail loudly."
        )

    meta_rdkit_ver = metadata.get("rdkit_version")
    if meta_rdkit_ver is None or not str(meta_rdkit_ver).strip():
        raise ValueError(
            f"Feature store at '{feature_path}' is missing required 'rdkit_version' in metadata header! Fail loudly."
        )

    expected_desc_dim = metadata.get("n_descriptors", 200)
    preprocessor = MolecularPreprocessor(morgan_nbits=expected_nbits, morgan_radius=expected_radius)
    expected_desc_names = preprocessor.descriptor_names[:expected_desc_dim]
    if list(meta_desc_names[:expected_desc_dim]) != list(expected_desc_names):
        raise ValueError(
            f"Feature store descriptor_names mismatch with active preprocessor in '{feature_path}'."
        )

    # Extract feature entries (excluding metadata keys)
    features = {k: v for k, v in store.items() if k not in ("metadata", "__metadata__")}

    if not features:
        raise ValueError(f"Feature store at '{feature_path}' contains no feature entries!")

    # 4. Validate EVERY feature-store entry's Morgan & descriptor vector length, finite values, and bounds
    for k, feat in features.items():
        if not isinstance(feat, dict):
            raise ValueError(f"Malformed feature entry for key '{k}' in '{feature_path}'")
        m_fp = feat.get("morgan", feat.get("morgan_fp", feat.get("fingerprint")))
        d_fp = feat.get("descriptors", feat.get("desc"))
        
        if m_fp is None or len(m_fp) != expected_nbits:
            m_len = len(m_fp) if m_fp is not None else "None"
            raise ValueError(
                f"Inconsistent Morgan vector dimension for key '{k}': expected {expected_nbits}, got {m_len} in '{feature_path}'."
            )
            
        if d_fp is None or len(d_fp) != expected_desc_dim:
            d_len = len(d_fp) if d_fp is not None else "None"
            raise ValueError(
                f"Inconsistent descriptor vector dimension for key '{k}': expected {expected_desc_dim}, got {d_len} in '{feature_path}'."
            )

        m_arr = m_fp.detach().cpu().numpy() if isinstance(m_fp, torch.Tensor) else np.asarray(m_fp)
        if not np.all(np.isfinite(m_arr)):
            raise ValueError(f"Non-finite Morgan vector for key '{k}' in '{feature_path}'!")
        if not np.all((m_arr == 0.0) | (m_arr == 1.0)):
            raise ValueError(f"Non-binary Morgan vector for key '{k}' in '{feature_path}'!")

        d_arr = d_fp.detach().cpu().numpy() if isinstance(d_fp, torch.Tensor) else np.asarray(d_fp)
        if np.any(np.isinf(d_arr)):
            raise ValueError(f"Inf value detected in descriptors for key '{k}' in '{feature_path}'!")
        finite_d = d_arr[np.isfinite(d_arr)]
        if len(finite_d) > 0 and np.any(np.abs(finite_d) > 1e6):
            raise ValueError(f"Descriptor value exceeding 1e6 for key '{k}' in '{feature_path}'!")

    return features


