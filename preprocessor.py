import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from typing import Tuple, Optional
from functools import lru_cache

try:
    from rdkit.Chem import rdFingerprintGenerator
    HAS_GENERATOR = True
except ImportError:
    HAS_GENERATOR = False

METALS = {'Pt', 'Fe', 'Cu', 'Zn', 'Au', 'As', 'Mg', 'Ca', 'Li', 'Na', 'K', 'Bi'}

def strip_salts_keep_largest_fragment(smiles: str) -> Tuple[str, bool]:
    """Strip salts by keeping largest heavy-atom fragment while retaining metal-containing drugs.

    Args:
        smiles: Input SMILES string.

    Returns:
        Tuple[str, bool]: (normalized_smiles, was_normalized)
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return "", False

    smiles_clean = smiles.strip()
    if "." not in smiles_clean:
        return smiles_clean, False

    mol_orig = Chem.MolFromSmiles(smiles_clean)
    if mol_orig is not None:
        atom_symbols = {a.GetSymbol() for a in mol_orig.GetAtoms()}
        if atom_symbols.intersection(METALS):
            return smiles_clean, False

    fragments = smiles_clean.split(".")
    best_frag = smiles_clean
    best_heavy_count = -1

    for frag in fragments:
        f_mol = Chem.MolFromSmiles(frag)
        heavy_count = f_mol.GetNumHeavyAtoms() if f_mol is not None else 0
        if heavy_count > best_heavy_count:
            best_heavy_count = heavy_count
            best_frag = frag
        elif heavy_count == best_heavy_count:
            if frag < best_frag:
                best_frag = frag

    return best_frag, best_frag != smiles_clean


class MolecularPreprocessor:
    """RDKit chemistry preprocessor mapping SMILES strings to numerical representations with LRU caching."""
    
    def __init__(self, morgan_nbits: int = 1024, morgan_radius: int = 3, cache_size: int = 10000):
        self.morgan_nbits = morgan_nbits
        self.morgan_radius = morgan_radius
        self.descriptor_names = [desc[0] for desc in Descriptors._descList][:200]
        self.cache_size = cache_size
        self._cache = {}
        self._generator = None

    def _get_generator(self):
        if self._generator is None and HAS_GENERATOR:
            self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=self.morgan_radius, fpSize=self.morgan_nbits)
        return self._generator

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_generator"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._generator = None
        
    def smiles_to_mol(self, smiles: str) -> Optional[Chem.Mol]:
        """Convert SMILES to RDKit Mol object after applying salt stripping.

        Args:
            smiles: Raw SMILES string.

        Returns:
            Optional[Chem.Mol]: RDKit Mol object or None if invalid.
        """
        try:
            norm_smiles, _ = strip_salts_keep_largest_fragment(smiles)
            mol = Chem.MolFromSmiles(norm_smiles)
            if mol is not None:
                mol = Chem.AddHs(mol)
            return mol
        except Exception:
            return None

    def get_morgan_fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        """Generate 1024-bit Morgan fingerprint vector (radius=3)."""
        if mol is None:
            return np.zeros(self.morgan_nbits, dtype=np.float32)
        try:
            generator = self._get_generator()
            if generator is not None:
                fp = generator.GetFingerprint(mol)
            else:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius=self.morgan_radius, nBits=self.morgan_nbits
                )
            arr = np.zeros((1,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            return arr.astype(np.float32)
        except Exception:
            return np.zeros(self.morgan_nbits, dtype=np.float32)

    def get_physical_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """Generate 200 continuous RDKit physical descriptor values with raw sanitization.
        
        Contract:
          1. +Inf and -Inf converted to NaN
          2. Finite raw descriptor values clipped to [-1e6, +1e6]
        """
        desc_vec = []
        if mol is None:
            return np.full(len(self.descriptor_names), np.nan, dtype=np.float32)
        for name in self.descriptor_names:
            try:
                func = getattr(Descriptors, name)
                val = func(mol)
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    desc_vec.append(np.nan)
                else:
                    desc_vec.append(float(np.clip(val, -1e6, 1e6)))
            except Exception:
                desc_vec.append(np.nan)
        res = np.array(desc_vec, dtype=np.float32)
        if len(res) < 200:
            res = np.pad(res, (0, 200 - len(res)), constant_values=np.nan)
        elif len(res) > 200:
            res = res[:200]
        res[np.isinf(res)] = np.nan
        finite_mask = np.isfinite(res)
        res[finite_mask] = np.clip(res[finite_mask], -1e6, 1e6)
        return res

    def process_smiles(self, smiles: str) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Processes SMILES string into Morgan fingerprint and continuous descriptors."""
        if smiles in self._cache:
            m_fp, desc, ok = self._cache[smiles]
            return m_fp.copy(), desc.copy(), ok
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            res = (np.zeros(self.morgan_nbits, dtype=np.float32), np.full(200, np.nan, dtype=np.float32), False)
        else:
            morgan_fp = self.get_morgan_fingerprint(mol)
            descriptors = self.get_physical_descriptors(mol)
            res = (morgan_fp, descriptors, True)
        if len(self._cache) < self.cache_size:
            self._cache[smiles] = res
        return res[0].copy(), res[1].copy(), res[2]
