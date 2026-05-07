"""
DeePHF Inference Engine
=======================
Loads three trained CorrNet checkpoints (electronic-only, SOAP-only, combined)
and runs all three in a single call after the shared HF computation.

Model paths are set via environment variables:
  MODEL_ELEC  — electronic-only checkpoint  (default: /app/model_elec.pt)
  MODEL_SOAP  — SOAP-only checkpoint        (default: /app/model_soap.pt)
  MODEL_COMB  — combined checkpoint         (default: /app/model_comb.pt)
  HF_BASIS    — PySCF basis set             (default: cc-pvdz)

Any missing checkpoint is silently skipped — the interface shows
"model not loaded" for that column and the others still work.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

HARTREE_TO_EV   = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631
EV_TO_KCAL      = 23.0609

_Z_TO_SYM = {
    1:"H", 2:"He", 3:"Li", 4:"Be", 5:"B", 6:"C", 7:"N", 8:"O",
    9:"F", 10:"Ne", 11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P",
    16:"S", 17:"Cl", 18:"Ar", 35:"Br", 53:"I",
}


# ── CorrNet (mirrors deephf_train.py exactly) ─────────────────────────────
class CorrNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(100, 100, 100)):
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer("input_shift", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))

        self.linear = nn.Linear(input_dim, 1)
        sizes = [input_dim, *hidden_sizes]
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
            + [nn.Linear(sizes[-1], 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.input_shift) / (self.input_scale + 1e-8)
        l = self.linear(x)
        h = x
        for i, layer in enumerate(self.layers):
            h_out = layer(h)
            if i < len(self.layers) - 1:
                h_out = F.gelu(h_out)
                if h.shape[-1] == h_out.shape[-1]:
                    h_out = h + h_out
            h = h_out
        return (h + l).sum(dim=1).squeeze(-1)


# ── .xyz parser ────────────────────────────────────────────────────────────
def parse_xyz(xyz_text: str) -> Tuple[np.ndarray, np.ndarray, int, str]:
    _SYM_TO_Z = {v: k for k, v in _Z_TO_SYM.items()}
    _SYM_TO_Z.update({"h":1,"c":6,"n":7,"o":8,"f":9,"s":16,"cl":17,"br":35})

    lines = [l.strip() for l in xyz_text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("File too short to be a valid .xyz")
    try:
        n_atoms = int(lines[0])
    except ValueError:
        raise ValueError(f"First line must be atom count, got: '{lines[0]}'")

    atom_lines = lines[2: 2 + n_atoms]
    if len(atom_lines) < n_atoms:
        raise ValueError(f"Expected {n_atoms} atoms, found {len(atom_lines)} lines")

    atomic_numbers, coords = [], []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed atom line: '{line}'")
        sym = parts[0].strip()
        Z = _SYM_TO_Z.get(sym) or _SYM_TO_Z.get(sym.capitalize())
        if Z is None:
            raise ValueError(f"Unknown element symbol: '{sym}'")
        atomic_numbers.append(Z)
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    atomic_numbers = np.array(atomic_numbers, dtype=int)
    coords = np.array(coords, dtype=float)

    from collections import Counter
    counts = Counter(atomic_numbers)
    formula = "".join(
        f"{_Z_TO_SYM.get(z, f'Z{z}')}{c if c > 1 else ''}"
        for z, c in sorted(counts.items(), key=lambda x: (-x[0] != 6, x[0]))
    )
    return atomic_numbers, coords, n_atoms, formula

def smiles_to_xyz(smiles: str) -> str:
    """Convert a SMILES string to XYZ-format text via RDKit ETKDG + MMFF."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: '{smiles}'")
    mol = Chem.AddHs(mol)                         # explicit hydrogens
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        raise ValueError("3D embedding failed — try a simpler SMILES or add stereo info.")
    AllChem.MMFFOptimizeMolecule(mol)             # quick force-field polish
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    lines = [str(n), smiles]                      # standard XYZ header
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()}  {p.x:.6f}  {p.y:.6f}  {p.z:.6f}")
    return "\n".join(lines)


# ── Model loader ───────────────────────────────────────────────────────────
def _load_model(path: Optional[str], label: str) -> Optional[CorrNet]:
    if not path or not os.path.exists(path):
        logger.warning(f"[{label}] No checkpoint at {path} — column will show 'model not loaded'")
        return None
    try:
        state = torch.load(path, map_location="cpu")
        input_dim = state["linear.weight"].shape[1]
        model = CorrNet(input_dim=input_dim)
        model.load_state_dict(state)
        model.eval()
        logger.info(f"[{label}] Loaded from {path} (input_dim={input_dim})")
        return model
    except Exception as e:
        logger.error(f"[{label}] Failed to load: {e}")
        return None


# ── Main inference class ───────────────────────────────────────────────────
class DeePHFInference:
    """
    Orchestrates the full pipeline for a single molecule:
      1. Electronic descriptors via PySCF (shared, computed once)
      2. SOAP descriptors via DScribe (if SOAP or combined model present)
      3. CorrNet inference for each available model
    """

    def __init__(
        self,
        model_elec_path: Optional[str] = None,
        model_soap_path: Optional[str] = None,
        model_comb_path: Optional[str] = None,
        cache_dir: str = "/app/cache_electronic",
        hf_basis: str = "cc-pvdz",
        w_atomic: float = 1.0,
    ):
        self.cache_dir = cache_dir
        self.hf_basis  = hf_basis
        self.w_atomic  = w_atomic
        os.makedirs(cache_dir, exist_ok=True)

        self.model_elec = _load_model(model_elec_path, "electronic")
        self.model_soap = _load_model(model_soap_path, "SOAP")
        self.model_comb = _load_model(model_comb_path, "combined")

    @property
    def any_model_loaded(self):
        return any(m is not None for m in [self.model_elec, self.model_soap, self.model_comb])

    # ── Descriptor computation ──
    def _electronic_desc(
        self, atomic_numbers: np.ndarray, coords: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
        """Run PySCF HF + density matrix projection. Returns (desc, E_HF, E_corr_mp2)."""
        try:
            from deephf_electronic import ElectronicDescriptor, _mol_hash

            h = _mol_hash(atomic_numbers, coords)
            cache_file = os.path.join(self.cache_dir, f"{h}.npz")

            if os.path.exists(cache_file):
                cached = np.load(cache_file, allow_pickle=True)
                desc   = cached["desc"]   if cached["desc"].shape != ()   else None
                E_HF   = float(cached["E_HF"])   if cached["E_HF"] != -1   else None
                E_corr = float(cached["E_corr"]) if cached["E_corr"] != -1 else None
                logger.info("Electronic descriptors loaded from cache")
                return desc, E_HF, E_corr

            edesc = ElectronicDescriptor(hf_basis=self.hf_basis, compute_mp2=True)
            desc, E_HF, E_corr = edesc.compute(atomic_numbers, coords)

            if desc is not None:
                np.savez(
                    cache_file,
                    desc=desc,
                    E_HF=E_HF   if E_HF   is not None else -1,
                    E_corr=E_corr if E_corr is not None else -1,
                )
            return desc, E_HF, E_corr

        except ImportError:
            raise RuntimeError(
                "deephf_electronic not found. "
                "Ensure the project files are mounted at /app/project/."
            )

    def _soap_desc(
        self, atomic_numbers: np.ndarray, coords: np.ndarray
    ) -> Optional[np.ndarray]:
        """Compute SOAP descriptors via DScribe."""
        try:
            from deephf_descriptors import SOAPDescriptor, SOAPConfig
            from deephf_dataloader import MoleculeData

            mol = MoleculeData(atomic_numbers=atomic_numbers, coords=coords, energy=None)
            soap = SOAPDescriptor(SOAPConfig())
            return soap.compute([mol])[0]   # (n_atoms, soap_dim)
        except Exception as e:
            logger.warning(f"SOAP descriptor computation failed: {e}")
            return None

    @staticmethod
    def _infer(model: CorrNet, desc: np.ndarray) -> float:
        X = torch.tensor(desc[np.newaxis, ...], dtype=torch.float32)
        with torch.no_grad():
            return float(model(X).item())

    # ── Public predict method ──
    def predict(self, atomic_numbers: np.ndarray, coords: np.ndarray) -> Dict[str, Any]:
        """
        Run the full pipeline. Returns a dict with energies for all three models,
        the HF reference, the MP2 reference, and metadata.
        Missing models return None for their energy fields.
        """

        # ── Step 1: Electronic descriptors (always computed — needed for E_HF, E_MP2,
        #            and the electronic-only model) ──
        elec_desc, E_HF, E_corr_mp2 = self._electronic_desc(atomic_numbers, coords)

        if elec_desc is None:
            return {
                "success": False,
                "error": "Hartree-Fock calculation failed to converge. "
                         "Check that your geometry is physically reasonable.",
                "E_HF_hartree": None, "E_HF_eV": None,
                "E_corr_mp2_eV": None, "E_corr_mp2_kcal": None,
                "E_corr_elec_eV": None, "E_corr_elec_kcal": None,
                "E_corr_soap_eV": None, "E_corr_soap_kcal": None,
                "E_corr_comb_eV": None, "E_corr_comb_kcal": None,
                "basis": self.hf_basis,
            }

        # ── Step 2: SOAP descriptors (only if at least one SOAP-dependent model loaded) ──
        soap_desc = None
        if self.model_soap is not None or self.model_comb is not None:
            soap_desc = self._soap_desc(atomic_numbers, coords)

        # ── Step 3: Build combined descriptor ──
        # Z-score normalise both blocks independently, then concatenate with w_atomic weight.
        comb_desc = None
        if soap_desc is not None and elec_desc is not None:
            e_norm = (elec_desc - elec_desc.mean(0)) / (elec_desc.std(0) + 1e-8)
            s_norm = (soap_desc - soap_desc.mean(0)) / (soap_desc.std(0) + 1e-8)
            comb_desc = np.concatenate([e_norm, self.w_atomic * s_norm], axis=1)

        # ── Step 4: Run each model ──
        def run(model, desc):
            if model is None or desc is None:
                return None, None
            ev = self._infer(model, desc)
            return round(ev, 6), round(ev * EV_TO_KCAL, 4)

        elec_ev, elec_kcal = run(self.model_elec, elec_desc)
        soap_ev, soap_kcal = run(self.model_soap, soap_desc)
        comb_ev, comb_kcal = run(self.model_comb, comb_desc)

        return {
            "success": True,
            # HF reference
            "E_HF_hartree": E_HF,
            "E_HF_eV":      E_HF * HARTREE_TO_EV   if E_HF else None,
            # MP2 reference
            "E_corr_mp2_eV":   E_corr_mp2 * HARTREE_TO_EV   if E_corr_mp2 else None,
            "E_corr_mp2_kcal": E_corr_mp2 * HARTREE_TO_KCAL if E_corr_mp2 else None,
            # Model predictions
            "E_corr_elec_eV":   elec_ev,   "E_corr_elec_kcal":  elec_kcal,
            "E_corr_soap_eV":   soap_ev,   "E_corr_soap_kcal":  soap_kcal,
            "E_corr_comb_eV":   comb_ev,   "E_corr_comb_kcal":  comb_kcal,
            # Metadata
            "basis": self.hf_basis,
            "descriptor_shape_elec": list(elec_desc.shape) if elec_desc is not None else None,
            "descriptor_shape_soap": list(soap_desc.shape) if soap_desc is not None else None,
            "descriptor_shape_comb": list(comb_desc.shape) if comb_desc is not None else None,
        }
