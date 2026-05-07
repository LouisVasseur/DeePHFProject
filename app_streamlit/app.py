"""
DeePHFProject — Streamlit Interface
=====================================
Vera Dias Gomes · Cédric Nathanaël Rossboth · Octavian Susanu · Louis James Vasseur
Team 3 · AI for Chemistry (CH-457) · EPFL

Predicts molecular correlation energy using four parallel models:
  1. Electronic-only CorrNet  (HF wavefunction descriptors)
  2. SOAP-only CorrNet        (geometric descriptors)
  3. Combined CorrNet         (electronic + SOAP)
  4. GNN                      (molecular graph, NNConv + LSTM)

Supports four input methods (XYZ upload, paste, SMILES, structure drawing via
Ketcher), live 3D molecule preview, and on-demand HF molecular orbital
visualisation as cube-file isosurfaces.
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import streamlit as st

# ── Streamlit page config — must be first ─────────────────────────────────
st.set_page_config(
    page_title  = "DeePHFProject",
    page_icon   = "⚛️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Path management ────────────────────────────────────────────────────────
# _HERE        = deephf_streamlit/  (directory containing this app.py)
# PROJECT_ROOT = DeePHFProject/     (one level up — contains deephf_electronic.py etc.)
_HERE        = Path(__file__).parent.resolve()
PROJECT_ROOT = _HERE.parent

# Inject into sys.path (this process) AND os.environ["PYTHONPATH"] (subprocesses /
# st.cache_data workers) so that deephf_electronic and friends are always importable.
for _p in [str(_HERE), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_existing_pypath = os.environ.get("PYTHONPATH", "")
_extra = os.pathsep.join(
    p for p in [str(PROJECT_ROOT), str(_HERE)]
    if p not in _existing_pypath
)
if _extra:
    os.environ["PYTHONPATH"] = (
        _extra + (os.pathsep + _existing_pypath if _existing_pypath else "")
    )

# Pre-import project modules at module level so they are available inside
# @st.cache_data without relying on sys.path being re-inherited.
try:
    from deephf_electronic import ElectronicDescriptor as _ElectronicDescriptor
    _HAVE_ELECTRONIC = True
except ImportError:
    _ElectronicDescriptor = None
    _HAVE_ELECTRONIC = False

try:
    from deephf_descriptors import SOAPDescriptor as _SOAPDescriptor, SOAPConfig as _SOAPConfig
    from deephf_dataloader  import MoleculeData   as _MoleculeData
    _HAVE_SOAP = True
except ImportError:
    _SOAPDescriptor = _SOAPConfig = _MoleculeData = None
    _HAVE_SOAP = False

try:
    from graph_atomic_descriptors import GraphAtomicDescriptors as _GraphAtomicDescriptors
    _HAVE_GRAPH = True
    _GRAPH_ERR  = None
except Exception as _e:
    _GraphAtomicDescriptors = None
    _HAVE_GRAPH = False
    _GRAPH_ERR  = str(_e)

try:
    from torch_geometric.data import Data as _PyGData, Batch as _PyGBatch
    _HAVE_PYG = True
    _PYG_ERR  = None
except Exception as _e:
    _PyGData = _PyGBatch = None
    _HAVE_PYG = False
    _PYG_ERR  = str(_e)

# Optional deps for the new features. App should still load even if these
# are missing — we degrade gracefully and tell the user what to install.
try:
    from streamlit_ketcher import st_ketcher
    _HAVE_KETCHER = True
    _KETCHER_ERR  = None
except Exception as _e:
    st_ketcher    = None
    _HAVE_KETCHER = False
    _KETCHER_ERR  = str(_e)

try:
    import py3Dmol
    _HAVE_PY3DMOL = True
    _PY3DMOL_ERR  = None
except Exception as _e:
    py3Dmol       = None
    _HAVE_PY3DMOL = False
    _PY3DMOL_ERR  = str(_e)


def _find_model(filename: str) -> str:
    """
    Look for a model file in _HERE first, then PROJECT_ROOT.
    Returns the first path that actually exists, or the PROJECT_ROOT
    location as a default (callers handle missing files gracefully).
    """
    for candidate in [_HERE / filename, PROJECT_ROOT / filename]:
        if candidate.is_file():
            return str(candidate)
    return str(PROJECT_ROOT / filename)


# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("DeePHFProject")

# ══════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════
HARTREE_TO_EV    = 27.211386245988
HARTREE_TO_KCAL  = 627.5094740631
EV_TO_KCAL       = 23.0609
CHEM_ACCURACY_EV = 1.0 / EV_TO_KCAL   # ~0.0434 eV

_Z_TO_SYM = {
    1: "H",  2: "He", 3: "Li", 4: "Be", 5: "B",  6: "C",  7: "N",  8: "O",
    9: "F",  10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 35: "Br", 53: "I",
}

# ── Default model paths: search _HERE then PROJECT_ROOT ───────────────────
_DEFAULT_ELEC = _find_model("model_elec.pt")
_DEFAULT_SOAP = _find_model("model_soap.pt")
_DEFAULT_COMB = _find_model("model_comb.pt")
_DEFAULT_GNN  = _find_model("model_gnn.ckpt")
_DEFAULT_CACHE = str(_HERE / "cache_electronic")
_DEFAULT_ORB_CACHE = str(_HERE / "cache_orbitals")

# ══════════════════════════════════════════════════════════════════════════
# XYZ utilities
# ══════════════════════════════════════════════════════════════════════════

def parse_xyz(xyz_text: str):
    _SYM_TO_Z = {v: k for k, v in _Z_TO_SYM.items()}
    _SYM_TO_Z.update({"h": 1, "c": 6, "n": 7, "o": 8, "f": 9,
                       "s": 16, "cl": 17, "br": 35})

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
    coords         = np.array(coords,         dtype=float)

    from collections import Counter
    counts  = Counter(atomic_numbers)
    formula = "".join(
        f"{_Z_TO_SYM.get(z, f'Z{z}')}{c if c > 1 else ''}"
        for z, c in sorted(counts.items(), key=lambda x: (-x[0] != 6, x[0]))
    )
    return atomic_numbers, coords, n_atoms, formula


def smiles_to_xyz(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: '{smiles}'")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        raise ValueError("3D embedding failed — try a simpler SMILES.")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    n    = mol.GetNumAtoms()
    lines = [str(n), smiles]
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()}  {p.x:.6f}  {p.y:.6f}  {p.z:.6f}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# CorrNet architecture  (mirrors deephf_train.py)
# ══════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(100, 100, 100)):
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer("input_shift", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))
        self.linear = nn.Linear(input_dim, 1)
        sizes = [input_dim, *hidden_sizes]
        self.layers = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
            + [nn.Linear(sizes[-1], 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = (x - self.input_shift) / (self.input_scale + 1e-8)
        l  = self.linear(x)
        h  = x
        for i, layer in enumerate(self.layers):
            h_out = layer(h)
            if i < len(self.layers) - 1:
                h_out = F.gelu(h_out)
                if h.shape[-1] == h_out.shape[-1]:
                    h_out = h + h_out
            h = h_out
        return (h + l).sum(dim=1).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════
# Model loaders (cached so Streamlit only loads once per session)
# ══════════════════════════════════════════════════════════════════════════

def _load_corrnet(path: str, label: str) -> Optional[CorrNet]:
    if not path or not os.path.isfile(path):
        return None
    try:
        state     = torch.load(path, map_location="cpu", weights_only=True)
        input_dim = state["linear.weight"].shape[1]
        model     = CorrNet(input_dim=input_dim)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as e:
        logger.warning(f"[{label}] Failed to load: {e}")
        return None


def _load_gnn(path: str, label: str = "GNN"):
    if not path or not os.path.isfile(path):
        logger.warning(f"[{label}] No checkpoint found at {path!r}")
        return None, None
    try:
        from gnn_model import corr_gnn

        # map_location="cpu" handles checkpoints saved on cuda:0
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg  = ckpt.get("hyper_parameters", {}).get("config", {}).get("model", {})

        model = corr_gnn(
            in_dimension     = cfg.get("in_dimension",     17),
            hidden_dimension = cfg.get("hidden_dimension", 64),
            out_dimension    = cfg.get("out_dimension",     1),
            T                = cfg.get("T",                 5),
            version          = cfg.get("version",          "v2"),
        )
        # PyTorch Lightning prefixes every key with "model." — strip it
        stripped = {k.removeprefix("model."): v
                    for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(stripped)
        model.eval()

        # Normalisation stats embedded in checkpoint (if any)
        stats = ckpt.get("hyper_parameters", {}).get("gnn_stats", None)
        logger.info(f"[{label}] Loaded successfully from {path!r}")
        return model, stats

    except Exception as e:
        logger.warning(f"[{label}] Failed to load: {e}", exc_info=True)
        return None, None


@st.cache_resource(show_spinner="Loading models…")
def load_models(elec_path, soap_path, comb_path, gnn_path):
    model_elec            = _load_corrnet(elec_path, "electronic")
    model_soap            = _load_corrnet(soap_path, "SOAP")
    model_comb            = _load_corrnet(comb_path, "combined")
    model_gnn, gnn_stats  = _load_gnn(gnn_path)
    return model_elec, model_soap, model_comb, model_gnn, gnn_stats


# ══════════════════════════════════════════════════════════════════════════
# Electronic descriptors + HF/MP2 via PySCF
# ══════════════════════════════════════════════════════════════════════════

def _mol_hash(atomic_numbers, coords, basis):
    raw = atomic_numbers.tobytes() + coords.tobytes() + basis.encode()
    return hashlib.md5(raw).hexdigest()


@st.cache_data(show_spinner=False)
def compute_electronic(_an_bytes, _co_bytes, basis, cache_dir):
    """Cached HF + density matrix descriptors."""
    atomic_numbers = np.frombuffer(_an_bytes, dtype=int)
    coords         = np.frombuffer(_co_bytes, dtype=float).reshape(-1, 3)

    h     = _mol_hash(atomic_numbers, coords, basis)
    cfile = Path(cache_dir) / f"{h}.npz"

    if cfile.exists():
        d    = np.load(cfile, allow_pickle=True)
        desc = d["desc"]  if d["desc"].shape != ()  else None
        E_HF = float(d["E_HF"])   if d["E_HF"]   != -1 else None
        E_c  = float(d["E_corr"]) if d["E_corr"]  != -1 else None
        return desc, E_HF, E_c

    if not _HAVE_ELECTRONIC:
        raise RuntimeError(
            "deephf_electronic could not be imported. "
            f"PROJECT_ROOT={PROJECT_ROOT} — check that deephf_electronic.py is there."
        )
    edesc = _ElectronicDescriptor(hf_basis=basis, compute_mp2=True)
    desc, E_HF, E_corr = edesc.compute(atomic_numbers, coords)

    if desc is not None:
        cfile.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cfile,
                 desc   = desc,
                 E_HF   = E_HF   if E_HF   is not None else -1,
                 E_corr = E_corr if E_corr is not None else -1)
    return desc, E_HF, E_corr


def compute_soap(atomic_numbers, coords):
    if not _HAVE_SOAP:
        logger.warning("SOAP dependencies not available — skipping SOAP descriptors")
        return None
    try:
        mol  = _MoleculeData(atomic_numbers=atomic_numbers, coords=coords, energy=None)
        soap = _SOAPDescriptor(_SOAPConfig())
        return soap.compute([mol])[0]
    except Exception as e:
        logger.warning(f"SOAP failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# HF molecular orbital cube generation
# ══════════════════════════════════════════════════════════════════════════
#
# We re-run an HF calculation here (cheap relative to MP2 for small molecules)
# so we can hold onto the SCF object in memory and call pyscf.tools.cubegen on
# its MOs.  The descriptor cache stores only the eigenvalue array — it does
# not preserve mo_coeff/mo_energy — so we use a separate cube cache keyed by
# (geometry, basis).  Cubes are stored as plain text (small for the 4 frontier
# orbitals at default grid resolution: ~50–200 KB each).
#
# Selected orbitals: HOMO-1, HOMO, LUMO, LUMO+1 (the four frontier MOs).
# ──────────────────────────────────────────────────────────────────────────

# Default cube grid resolution — coarser than PySCF's default for speed
_CUBE_NX = _CUBE_NY = _CUBE_NZ = 60


def _cube_cache_key(atomic_numbers, coords, basis) -> str:
    """Same hash scheme as the descriptor cache — stable across runs."""
    return _mol_hash(atomic_numbers, coords, basis)


@st.cache_data(show_spinner=False)
def compute_hf_orbitals(
    _an_bytes: bytes,
    _co_bytes: bytes,
    basis: str,
    orb_cache_dir: str,
    nx: int = _CUBE_NX,
    ny: int = _CUBE_NY,
    nz: int = _CUBE_NZ,
) -> Optional[Dict[str, Any]]:
    """
    Run a fresh RHF calculation and dump cube data for HOMO−1, HOMO, LUMO, LUMO+1.

    Returns a dict
        {
            "labels":       ["HOMO-1", "HOMO", "LUMO", "LUMO+1"],
            "indices":      [i_h-1, i_h, i_l, i_l+1],   # 0-based MO indices
            "energies_eV":  [...],                       # MO energies
            "occupations":  [...],                       # 2.0 / 0.0
            "cubes":        ["<cube text>", "<cube text>", ...],
            "homo_idx":     i_h,
            "n_mo":         total number of MOs,
        }
    or None on failure.

    Disk-cache behaviour:
      • If a cube cache file already exists for this (geometry, basis), it is
        loaded and returned. (Reading the user's earlier explicit save.)
      • Otherwise the calculation is done fully in memory and the result is
        returned WITHOUT being written to disk. The caller decides whether
        to persist via save_hf_orbitals_to_cache().

    Streamlit's @st.cache_data still memoises across reruns within a session,
    so the user only pays for the HF + cubegen work once per molecule per
    session even when nothing is on disk.
    """
    atomic_numbers = np.frombuffer(_an_bytes, dtype=int)
    coords         = np.frombuffer(_co_bytes, dtype=float).reshape(-1, 3)

    Path(orb_cache_dir).mkdir(parents=True, exist_ok=True)
    h          = _cube_cache_key(atomic_numbers, coords, basis)
    cache_file = Path(orb_cache_dir) / f"{h}.npz"

    # ── Read from disk if already cached (user previously saved) ──────────
    if cache_file.exists():
        try:
            d = np.load(cache_file, allow_pickle=True)
            return {
                "labels":      [str(x) for x in d["labels"]],
                "indices":     [int(x) for x in d["indices"]],
                "energies_eV": [float(x) for x in d["energies_eV"]],
                "occupations": [float(x) for x in d["occupations"]],
                "cubes":       [str(c) for c in d["cubes"]],
                "homo_idx":    int(d["homo_idx"]),
                "n_mo":        int(d["n_mo"]),
            }
        except Exception as e:
            logger.warning(f"Could not read orbital cache {cache_file}: {e}")

    # ── Otherwise compute in memory only — DO NOT write to disk ───────────
    try:
        from pyscf import gto, scf
        from pyscf.tools import cubegen
    except ImportError as e:
        raise RuntimeError(f"PySCF not available for orbital generation: {e}")

    # Build PySCF molecule
    atom_list = []
    for i in range(len(atomic_numbers)):
        sym = _Z_TO_SYM.get(int(atomic_numbers[i]))
        if sym is None:
            raise RuntimeError(f"Unknown element Z={atomic_numbers[i]}")
        atom_list.append(
            f"{sym} {coords[i,0]:.10f} {coords[i,1]:.10f} {coords[i,2]:.10f}"
        )
    mol = gto.Mole(atom="; ".join(atom_list), basis=basis, verbose=0)
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-8
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(
            "HF did not converge during orbital cube generation. "
            "Try a different basis set."
        )

    mo_occ    = mf.mo_occ
    mo_energy = mf.mo_energy
    mo_coeff  = mf.mo_coeff
    n_mo      = mo_coeff.shape[1]

    # HOMO is the highest-index occupied MO (occ > 0). For a closed-shell
    # singlet this is unambiguous; mo_occ holds 2.0 for occupied and 0.0
    # for virtual.
    occupied = np.where(mo_occ > 0)[0]
    if len(occupied) == 0:
        raise RuntimeError("No occupied orbitals found.")
    homo_idx = int(occupied[-1])
    lumo_idx = homo_idx + 1

    # Pick the four frontier orbitals, clamped to legal range
    candidates = [
        ("HOMO-1", homo_idx - 1),
        ("HOMO",   homo_idx    ),
        ("LUMO",   lumo_idx    ),
        ("LUMO+1", lumo_idx + 1),
    ]
    selected: List[Tuple[str, int]] = [
        (lab, idx) for (lab, idx) in candidates if 0 <= idx < n_mo
    ]

    labels:      List[str]   = []
    indices:     List[int]   = []
    energies_eV: List[float] = []
    occs:        List[float] = []
    cubes:       List[str]   = []

    # Generate cube data for each selected MO. We write to a temp file
    # because cubegen.orbital writes to disk; we then read the text back
    # and immediately delete the temp file. The user-facing disk cache
    # is written separately, only when explicitly requested.
    for lab, idx in selected:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cube", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            cubegen.orbital(
                mol, tmp_path, mo_coeff[:, idx],
                nx=nx, ny=ny, nz=nz,
            )
            with open(tmp_path, "r") as fh:
                cube_text = fh.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        labels.append(lab)
        indices.append(int(idx))
        energies_eV.append(float(mo_energy[idx]) * HARTREE_TO_EV)
        occs.append(float(mo_occ[idx]))
        cubes.append(cube_text)

    return {
        "labels":      labels,
        "indices":     indices,
        "energies_eV": energies_eV,
        "occupations": occs,
        "cubes":       cubes,
        "homo_idx":    homo_idx,
        "n_mo":        int(n_mo),
    }


def orbital_cache_path(
    atomic_numbers: np.ndarray, coords: np.ndarray, basis: str, orb_cache_dir: str
) -> Path:
    """Filename where an orbital result for this (geometry, basis) would live."""
    h = _cube_cache_key(atomic_numbers, coords, basis)
    return Path(orb_cache_dir) / f"{h}.npz"


def is_orbital_cached(
    atomic_numbers: np.ndarray, coords: np.ndarray, basis: str, orb_cache_dir: str
) -> bool:
    """Return True if an orbital cube file is already on disk for this molecule."""
    return orbital_cache_path(atomic_numbers, coords, basis, orb_cache_dir).is_file()


def save_hf_orbitals_to_cache(
    orb_data: Dict[str, Any],
    atomic_numbers: np.ndarray,
    coords: np.ndarray,
    basis: str,
    orb_cache_dir: str,
) -> Tuple[bool, str]:
    """
    Persist an in-memory orb_data dict (as returned by compute_hf_orbitals)
    to the on-disk orbital cache.

    Returns (success, message).
    """
    if not orb_data or not orb_data.get("cubes"):
        return False, "No orbital data to store."

    cache_file = orbital_cache_path(atomic_numbers, coords, basis, orb_cache_dir)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_file,
            labels      = np.array(orb_data["labels"]),
            indices     = np.array(orb_data["indices"], dtype=int),
            energies_eV = np.array(orb_data["energies_eV"], dtype=float),
            occupations = np.array(orb_data["occupations"], dtype=float),
            cubes       = np.array(orb_data["cubes"], dtype=object),
            homo_idx    = np.int64(orb_data["homo_idx"]),
            n_mo        = np.int64(orb_data["n_mo"]),
        )
        size_kb = cache_file.stat().st_size / 1024
        return True, f"Stored {size_kb:.0f} KB to {cache_file.name}"
    except Exception as e:
        return False, f"Save failed: {e}"


# ══════════════════════════════════════════════════════════════════════════
# GNN inference
# ══════════════════════════════════════════════════════════════════════════

def infer_gnn(model_gnn, gnn_stats, atomic_numbers, coords, xyz_text: str) -> Optional[float]:
    """
    Run the GNN.  graph_atomic_descriptors.build_molecular_graph() requires a
    file path (OpenBabel pybel.readfile), so we write a temp XYZ then delete it.

    Uses module-level pre-imported classes (_GraphAtomicDescriptors, _PyGData,
    _PyGBatch) so this works reliably inside st.cache_data contexts.

    Raises RuntimeError on any failure so the caller can surface it to the UI.
    """
    if model_gnn is None:
        return None

    if not _HAVE_GRAPH:
        raise RuntimeError(
            f"graph_atomic_descriptors import failed: {_GRAPH_ERR}. "
            f"PROJECT_ROOT={PROJECT_ROOT}."
        )
    if not _HAVE_PYG:
        raise RuntimeError(
            f"torch_geometric import failed: {_PYG_ERR}. "
            "Run: pip install torch_geometric"
        )

    # Write a temporary XYZ file for OpenBabel inside build_molecular_graph
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as tmp:
        tmp.write(xyz_text)
        xyz_path = tmp.name

    try:
        gad = _GraphAtomicDescriptors(remove_h=False)
        node_feat, edge_feat, edge_idx = gad.build_molecular_graph(xyz_path)
    except Exception as e:
        raise RuntimeError(f"Graph construction failed: {e}") from e
    finally:
        os.unlink(xyz_path)

    if node_feat is None:
        raise RuntimeError(
            "build_molecular_graph returned None — OpenBabel could not parse the structure. "
            "Check that openbabel-wheel is installed: pip install openbabel-wheel"
        )

    data = _PyGData(
        x          = torch.tensor(node_feat, dtype=torch.float),
        edge_attr  = torch.tensor(edge_feat, dtype=torch.float),
        edge_index = torch.tensor(edge_idx,  dtype=torch.long),
    )
    batch = _PyGBatch.from_data_list([data])

    with torch.no_grad():
        out = model_gnn(
            node_feature_matrix = batch.x,
            edge_indices_matrix = batch.edge_index,
            edge_feature_matrix = batch.edge_attr,
            batch               = batch.batch,
        )

    z_score = float(out.item())

    if gnn_stats and "mean" in gnn_stats and "std" in gnn_stats:
        return z_score * float(gnn_stats["std"]) + float(gnn_stats["mean"])
    return z_score


# ══════════════════════════════════════════════════════════════════════════
# Full prediction pipeline
# ══════════════════════════════════════════════════════════════════════════

def run_prediction(
    atomic_numbers, coords,
    model_elec, model_soap, model_comb, model_gnn, gnn_stats,
    basis, w_atomic, cache_dir,
    xyz_text: str = "",
) -> Dict[str, Any]:

    # ── 1. Electronic descriptors (HF + MP2 references) ──────────────────
    elec_desc, E_HF, E_corr_mp2 = compute_electronic(
        atomic_numbers.tobytes(),
        coords.tobytes(),
        basis,
        cache_dir,
    )

    if elec_desc is None:
        return {
            "success": False,
            "error": "Hartree-Fock calculation failed to converge. "
                     "Check that your geometry is physically reasonable.",
        }

    # ── 2. SOAP descriptors ───────────────────────────────────────────────
    soap_desc = None
    if model_soap is not None or model_comb is not None:
        soap_desc = compute_soap(atomic_numbers, coords)

    # ── 3. Combined descriptor ────────────────────────────────────────────
    comb_desc = None
    if soap_desc is not None and elec_desc is not None:
        e_norm    = (elec_desc - elec_desc.mean(0)) / (elec_desc.std(0) + 1e-8)
        s_norm    = (soap_desc - soap_desc.mean(0)) / (soap_desc.std(0) + 1e-8)
        comb_desc = np.concatenate([e_norm, w_atomic * s_norm], axis=1)

    # ── 4. CorrNet inference ──────────────────────────────────────────────
    def run(model, desc):
        if model is None or desc is None:
            return None, None
        X  = torch.tensor(desc[np.newaxis, ...], dtype=torch.float32)
        with torch.no_grad():
            ev = float(model(X).item())
        return round(ev, 6), round(ev * EV_TO_KCAL, 4)

    elec_ev, elec_kcal = run(model_elec, elec_desc)
    soap_ev, soap_kcal = run(model_soap, soap_desc)
    comb_ev, comb_kcal = run(model_comb, comb_desc)

    # ── 5. GNN inference ──────────────────────────────────────────────────
    gnn_ev = gnn_kcal = None
    gnn_warning = None
    try:
        gnn_ev_raw = infer_gnn(model_gnn, gnn_stats, atomic_numbers, coords, xyz_text)
        if gnn_ev_raw is not None:
            gnn_ev   = round(gnn_ev_raw, 6)
            gnn_kcal = round(gnn_ev_raw * EV_TO_KCAL, 4)
    except RuntimeError as e:
        gnn_warning = str(e)
        logger.warning(f"GNN skipped: {e}")

    return {
        "success": True,
        "E_HF_hartree":    E_HF,
        "E_HF_eV":         E_HF * HARTREE_TO_EV   if E_HF else None,
        "E_corr_mp2_eV":   E_corr_mp2 * HARTREE_TO_EV   if E_corr_mp2 else None,
        "E_corr_mp2_kcal": E_corr_mp2 * HARTREE_TO_KCAL if E_corr_mp2 else None,
        "E_corr_elec_eV":  elec_ev,  "E_corr_elec_kcal": elec_kcal,
        "E_corr_soap_eV":  soap_ev,  "E_corr_soap_kcal": soap_kcal,
        "E_corr_comb_eV":  comb_ev,  "E_corr_comb_kcal": comb_kcal,
        "E_corr_gnn_eV":   gnn_ev,   "E_corr_gnn_kcal":  gnn_kcal,
        "gnn_warning": gnn_warning,
        "basis": basis,
    }


# ══════════════════════════════════════════════════════════════════════════
# 3D visualisation (py3Dmol → HTML embed)
# ══════════════════════════════════════════════════════════════════════════
#
# Two viewers:
#   • render_molecule_3d        — sticks + balls only, used as the live preview
#                                 next to the input panel before/after running
#                                 the prediction.
#   • render_molecule_with_orbital  — same molecule + a cube-file isosurface
#                                     overlaid (positive lobe = blue, negative
#                                     lobe = red), used after the prediction
#                                     when "Show MO" is toggled on.
# ──────────────────────────────────────────────────────────────────────────

def render_molecule_3d(
    xyz_text: str,
    height_px: int = 460,
    width_px: Optional[int] = None,
) -> Optional[str]:
    """Build a self-contained HTML snippet that renders the molecule in 3D."""
    if not _HAVE_PY3DMOL:
        return None

    # py3Dmol works best with a fixed pixel width; we let it stretch via JS.
    view = py3Dmol.view(width=(width_px or 480), height=height_px)
    view.addModel(xyz_text, "xyz")
    view.setStyle({"stick":  {"radius": 0.12, "colorscheme": "Jmol"},
                   "sphere": {"scale": 0.22, "colorscheme": "Jmol"}})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.zoom(1.15)

    html = view._make_html()
    # Make the viewer responsive: stretch the container to fit Streamlit's column.
    responsive_css = """
    <style>
      .mol-container, .mol-container > div, .mol-container canvas {
        width: 100% !important;
      }
    </style>
    <div class="mol-container">
    """
    return responsive_css + html + "</div>"


def render_molecule_with_orbital(
    xyz_text: str,
    cube_text: str,
    isovalue: float = 0.04,
    height_px: int = 480,
    width_px: Optional[int] = None,
) -> Optional[str]:
    """Molecule + MO isosurface overlay (positive lobe blue, negative lobe red)."""
    if not _HAVE_PY3DMOL:
        return None

    view = py3Dmol.view(width=(width_px or 480), height=height_px)
    view.addModel(xyz_text, "xyz")
    view.setStyle({"stick":  {"radius": 0.10, "colorscheme": "Jmol"},
                   "sphere": {"scale": 0.18, "colorscheme": "Jmol"}})
    view.setBackgroundColor("white")

    # Positive lobe (blue) and negative lobe (red) at ±isovalue
    view.addVolumetricData(
        cube_text, "cube",
        {"isoval":  isovalue, "color": "blue", "opacity": 0.78},
    )
    view.addVolumetricData(
        cube_text, "cube",
        {"isoval": -isovalue, "color": "red",  "opacity": 0.78},
    )

    view.zoomTo()
    view.zoom(1.05)

    html = view._make_html()
    responsive_css = """
    <style>
      .mol-container, .mol-container > div, .mol-container canvas {
        width: 100% !important;
      }
    </style>
    <div class="mol-container">
    """
    return responsive_css + html + "</div>"


# ══════════════════════════════════════════════════════════════════════════
# CSS  — EPFL white/red academic palette
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=JetBrains+Mono:wght@300;400;500&display=swap');

  /* ── Force white background and dark text everywhere ── */
  html, body { background-color: #ffffff !important; color: #1a1a1a !important; }
  .stApp { background-color: #ffffff !important; }
  .stApp * { color: #1a1a1a; }

  /* Main content and markdown text */
  .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th,
  div[data-testid="stMarkdownContainer"] p,
  div[data-testid="stMarkdownContainer"] li { color: #1a1a1a !important; }

  /* Headings */
  h1, h2, h3, h4, h5, h6 { color: #1a1a1a !important; }

  /* Inputs, selectboxes, sliders */
  .stTextInput label, .stSelectbox label, .stSlider label,
  .stFileUploader label, .stTextArea label, .stRadio label { color: #1a1a1a !important; }
  .stTextInput input, .stTextArea textarea { color: #1a1a1a !important; background: #ffffff !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background-color: #f9f9f9 !important; border-right: 1px solid #e8e8e8; }
  section[data-testid="stSidebar"] * { color: #1a1a1a !important; }
  section[data-testid="stSidebar"] h3 { color: #1a1a1a !important; }

  /* Expander */
  details summary { color: #1a1a1a !important; }
  details > div { color: #1a1a1a !important; }

  /* Code blocks */
  code, pre { color: #1a1a1a !important; background: #f4f4f4 !important; }

  /* Table */
  table { color: #1a1a1a !important; }
  th { color: #1a1a1a !important; background: #f4f4f4 !important; }
  td { color: #1a1a1a !important; }

  /* Radio buttons */
  .stRadio > div label { color: #1a1a1a !important; }

  /* Spinner / status messages */
  .stSpinner p { color: #555555 !important; }

  /* Global serif font */
  html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background-color: #f9f9f9; border-right: 1px solid #e8e8e8; }

  /* ── EPFL red accent ── */
  :root {
    --epfl-red:   #ff0000;
    --epfl-dark:  #1a1a1a;
    --epfl-mid:   #555555;
    --epfl-dim:   #888888;
    --epfl-line:  #e0e0e0;
    --epfl-bg:    #f7f7f7;
    --mono:       'JetBrains Mono', monospace;
  }

  /* ── Header ── */
  .deephf-header {
    text-align: center;
    padding: 2rem 0 1.2rem;
    border-bottom: 2px solid var(--epfl-red);
    margin-bottom: 1.8rem;
  }
  .deephf-header h1 {
    font-size: 1.85rem;
    font-weight: 600;
    color: var(--epfl-dark);
    letter-spacing: -0.5px;
    margin-bottom: 0.15rem;
  }
  .deephf-header .subtitle {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--epfl-red);
    margin-bottom: 0.8rem;
  }
  .deephf-header .authors {
    font-size: 0.88rem;
    color: var(--epfl-mid);
    line-height: 1.8;
    font-style: italic;
  }
  .deephf-header .course {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 1px;
    color: var(--epfl-dim);
    font-style: normal;
    margin-top: 0.15rem;
  }

  /* ── Section labels ── */
  .section-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--epfl-dim);
    margin: 1.4rem 0 0.6rem;
    border-left: 3px solid var(--epfl-red);
    padding-left: 0.5rem;
  }

  /* ── Energy result cards ── */
  .energy-card {
    background: #ffffff;
    border: 1px solid var(--epfl-line);
    border-top: 3px solid var(--epfl-line);
    border-radius: 4px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.8rem;
  }
  .energy-card .model-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
    color: var(--epfl-dim);
  }
  .energy-card .value-ev {
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--epfl-dark);
    letter-spacing: -0.3px;
  }
  .energy-card .value-kcal {
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--epfl-mid);
    margin-top: 0.1rem;
  }
  .energy-card .delta {
    font-family: var(--mono);
    font-size: 0.77rem;
    margin-top: 0.5rem;
    padding: 0.18rem 0.5rem;
    border-radius: 3px;
    display: inline-block;
  }
  .delta-good { background: #fff0f0; color: #8b0000; border: 1px solid #ffcccc; }
  .delta-bad  { background: #f5f5f5; color: #555555; border: 1px solid #dddddd; }
  .delta-na   { background: #f5f5f5; color: var(--epfl-dim); border: 1px solid #e0e0e0; }
  .not-loaded {
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--epfl-dim);
    font-style: italic;
  }

  /* ── Model colour accents (subtle, for labels only) ── */
  .col-elec { color: #c0392b; }
  .col-soap { color: #333333; }
  .col-comb { color: #c0392b; }
  .col-gnn  { color: #333333; }
  .col-mp2  { color: #888888; }

  /* ── Status pills ── */
  .status-pill {
    display: inline-block;
    padding: 0.12rem 0.55rem;
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.5px;
  }
  .pill-ok  { background: #fff0f0; color: #c0392b; border: 1px solid #ffbbbb; }
  .pill-off { background: #f5f5f5; color: #999999; border: 1px solid #e0e0e0; }

  /* ── Info / instruction box ── */
  .info-box {
    background: #fafafa;
    border: 1px solid var(--epfl-line);
    border-left: 3px solid var(--epfl-red);
    border-radius: 3px;
    padding: 0.9rem 1.1rem;
    font-size: 0.87rem;
    color: var(--epfl-dark);
    line-height: 1.7;
    margin: 0.8rem 0 1.2rem;
  }
  .info-box code {
    font-family: var(--mono);
    font-size: 0.82rem;
    background: #f0f0f0;
    padding: 0.05rem 0.3rem;
    border-radius: 2px;
  }

  /* ── 3D viewer panel ── */
  .viewer-panel {
    background: #ffffff;
    border: 1px solid var(--epfl-line);
    border-top: 3px solid var(--epfl-red);
    border-radius: 4px;
    padding: 0.6rem 0.6rem 0.4rem;
    margin-bottom: 0.8rem;
  }
  .viewer-caption {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--epfl-dim);
    text-align: center;
    margin-top: 0.3rem;
  }
  .viewer-empty {
    height: 460px;
    border: 1px dashed var(--epfl-line);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--epfl-dim);
    background: #fafafa;
    text-align: center;
    padding: 1rem;
  }

  /* ── Footer / citation ── */
  .citation-box {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--epfl-line);
    font-size: 0.78rem;
    color: var(--epfl-dim);
    line-height: 1.7;
    font-style: italic;
  }
  .citation-box strong {
    font-style: normal;
    color: var(--epfl-mid);
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Sidebar — configuration
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")

    st.markdown("**Model checkpoints**")
    elec_path = st.text_input(
        "Electronic model (.pt)",
        value=_DEFAULT_ELEC,
        help="CorrNet trained on HF electronic descriptors."
    )
    soap_path = st.text_input(
        "SOAP model (.pt)",
        value=_DEFAULT_SOAP,
        help="CorrNet trained on SOAP geometric descriptors."
    )
    comb_path = st.text_input(
        "Combined model (.pt)",
        value=_DEFAULT_COMB,
        help="CorrNet trained on electronic + SOAP descriptors."
    )
    gnn_path = st.text_input(
        "GNN model (.ckpt)",
        value=_DEFAULT_GNN,
        help="PyTorch Lightning checkpoint — NNConv + LSTM message passing."
    )

    st.markdown("---")
    st.markdown("**HF / descriptor settings**")
    hf_basis = st.selectbox(
        "HF basis set",
        ["cc-pvdz", "cc-pvtz", "sto-3g", "6-31g", "6-31g*"],
        index=0,
        help="Basis set for the PySCF Hartree-Fock calculation."
    )
    w_atomic = st.slider(
        "SOAP weight (w_atomic)",
        min_value=0.1, max_value=4.0, value=1.0, step=0.1,
        help="Relative weight of SOAP features in the combined descriptor."
    )

    st.markdown("---")
    st.markdown("**GNN normalisation** *(optional)*")
    gnn_mean_str = st.text_input(
        "Mean (eV)", value="", placeholder="e.g. −0.312"
    )
    gnn_std_str  = st.text_input(
        "Std  (eV)", value="", placeholder="e.g. 0.085"
    )
    _gnn_stats_override = None
    if gnn_mean_str.strip() and gnn_std_str.strip():
        try:
            _gnn_stats_override = {
                "mean": float(gnn_mean_str),
                "std":  float(gnn_std_str),
            }
        except ValueError:
            st.warning("GNN stats must be valid floats.")

    st.markdown("---")
    cache_dir = st.text_input(
        "Cache directory",
        value=_DEFAULT_CACHE,
        help="HF descriptor computations are cached here to avoid recomputation."
    )
    os.makedirs(cache_dir, exist_ok=True)

    orb_cache_dir = st.text_input(
        "Orbital cache directory",
        value=_DEFAULT_ORB_CACHE,
        help="Cube files for HF molecular orbitals are cached here. Each "
             "molecule + basis combination caches ~1 MB of data."
    )
    os.makedirs(orb_cache_dir, exist_ok=True)

    st.markdown("---")
    st.markdown(
        "<span style='font-family:monospace;font-size:0.68rem;color:#aaaaaa'>"
        "DeePHFProject · CH-457 · EPFL · 2025</span>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# Load models (cached across reruns)
# ══════════════════════════════════════════════════════════════════════════

model_elec, model_soap, model_comb, model_gnn, _gnn_stats_ckpt = load_models(
    elec_path, soap_path, comb_path, gnn_path
)
gnn_stats = _gnn_stats_override if _gnn_stats_override is not None else _gnn_stats_ckpt


# ══════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="deephf-header">
  <h1>DeePHFProject</h1>
  <div class="subtitle">Correlation Energy Predictor</div>
  <div class="authors">
    Vera Dias Gomes &middot; Cédric Nathanaël Rossboth &middot; Octavian Susanu &middot; Louis James Vasseur<br>
    <span class="course">Team 3 &middot; AI for Chemistry (CH-457) &middot; EPFL</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Model status row ───────────────────────────────────────────────────────
cols_status = st.columns(4)
model_status = [
    ("Electronic", model_elec,  "col-elec"),
    ("SOAP",       model_soap,  "col-soap"),
    ("Combined",   model_comb,  "col-comb"),
    ("GNN",        model_gnn,   "col-gnn"),
]
for col, (name, m, cls) in zip(cols_status, model_status):
    ok   = m is not None
    pill = "pill-ok" if ok else "pill-off"
    icon = "✓" if ok else "–"
    col.markdown(
        f'<div style="text-align:center;margin-bottom:0.5rem">'
        f'<span class="{cls}" style="font-family:monospace;font-size:0.72rem;'
        f'font-weight:600;text-transform:uppercase;letter-spacing:0.5px">{name}</span><br>'
        f'<span class="status-pill {pill}">{icon} {"loaded" if ok else "not loaded"}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# Input section + live 3D viewer  (two-column layout)
# ══════════════════════════════════════════════════════════════════════════
#
# Left column  : input picker (upload / SMILES / paste / draw)
# Right column : live 3D molecule preview that updates whenever xyz_text
#                changes. After a prediction has been run, the same panel
#                can additionally render an MO isosurface.
# ──────────────────────────────────────────────────────────────────────────

# Initialise session state once. We persist xyz_text, mol_label and the last
# prediction result between reruns so toggling unrelated widgets (e.g. the
# orbital checkbox) doesn't clear what the user was looking at.
for _k in ("xyz_text", "mol_label", "last_result"):
    if _k not in st.session_state:
        st.session_state[_k] = None

col_input, col_viewer = st.columns([1, 1], gap="large")

# ─────────────────────────────────────────────────────────────────────────
# LEFT — input picker
# ─────────────────────────────────────────────────────────────────────────
with col_input:
    st.markdown('<div class="section-label">Input molecule</div>', unsafe_allow_html=True)

    input_modes = ["Upload .xyz file", "Enter SMILES", "Paste XYZ text", "Draw structure"]
    input_mode = st.radio(
        "Input method",
        input_modes,
        horizontal=True,
        label_visibility="collapsed",
    )

    xyz_text  = None
    mol_label = None

    # ── Mode 1: Upload .xyz file ───────────────────────────────────────────
    if input_mode == "Upload .xyz file":
        uploaded = st.file_uploader(
            "Drop a .xyz file here",
            type=["xyz"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            try:
                xyz_text  = uploaded.read().decode("utf-8")
                mol_label = uploaded.name
                # Only invalidate any previous prediction if the XYZ actually changed.
                # (Without this guard, every rerun would clear the results.)
                if st.session_state.get("xyz_text") != xyz_text:
                    st.session_state["last_result"] = None
                st.session_state["xyz_text"]  = xyz_text
                st.session_state["mol_label"] = mol_label
            except Exception:
                st.error("Could not decode the uploaded file. Ensure it is UTF-8 text.")

    # ── Mode 2: Enter SMILES ───────────────────────────────────────────────
    elif input_mode == "Enter SMILES":
        smiles_input = st.text_input(
            "SMILES string",
            placeholder="e.g. O   or   CC   or   c1ccccc1",
            label_visibility="collapsed",
        )
        if smiles_input.strip():
            if st.button("Generate 3D structure", type="primary", key="smiles_gen"):
                with st.spinner("Generating 3D geometry via RDKit ETKDG + MMFF…"):
                    try:
                        xyz_text  = smiles_to_xyz(smiles_input.strip())
                        mol_label = smiles_input.strip()
                        st.session_state["xyz_text"]  = xyz_text
                        st.session_state["mol_label"] = mol_label
                        st.session_state["last_result"] = None
                    except ValueError as e:
                        st.error(str(e))

    # ── Mode 3: Paste XYZ text ─────────────────────────────────────────────
    elif input_mode == "Paste XYZ text":
        xyz_input = st.text_area(
            "Paste XYZ content",
            height=200,
            placeholder="3\nWater\nO  0.000  0.000  0.000\nH  0.000  0.757  0.587\nH  0.000 -0.757  0.587",
            label_visibility="collapsed",
        )
        if xyz_input.strip():
            xyz_text  = xyz_input.strip()
            mol_label = "pasted molecule"
            if st.session_state.get("xyz_text") != xyz_text:
                st.session_state["last_result"] = None
            st.session_state["xyz_text"]  = xyz_text
            st.session_state["mol_label"] = mol_label

    # ── Mode 4: Draw structure (streamlit_ketcher) ─────────────────────────
    else:  # "Draw structure"
        if not _HAVE_KETCHER:
            st.warning(
                "`streamlit-ketcher` is not installed. Install it with:\n"
                "```\npip install streamlit-ketcher\n```\n"
                f"Underlying error: `{_KETCHER_ERR}`"
            )
        else:
            st.markdown(
                "<span style='font-size:0.82rem;color:#555'>"
                "Draw a structure below, then click <b>Apply</b> in the editor "
                "to capture the SMILES.</span>",
                unsafe_allow_html=True,
            )
            # st_ketcher returns the current SMILES whenever the user clicks Apply
            drawn_smiles = st_ketcher("", height=420, key="ketcher_editor")

            if drawn_smiles and drawn_smiles.strip():
                st.markdown(
                    f"**Drawn SMILES:** `{drawn_smiles}`"
                )
                # Auto-generate 3D the first time we see a new SMILES, but allow
                # the user to re-trigger via a button. We compare to last drawn
                # to avoid rebuilding on every rerun.
                last_drawn = st.session_state.get("last_drawn_smiles")
                regen_btn = st.button(
                    "Generate 3D structure from drawing",
                    type="primary",
                    key="ketcher_gen",
                )
                if regen_btn or last_drawn != drawn_smiles:
                    with st.spinner("Generating 3D geometry via RDKit ETKDG + MMFF…"):
                        try:
                            xyz_text  = smiles_to_xyz(drawn_smiles.strip())
                            mol_label = f"drawn: {drawn_smiles.strip()}"
                            st.session_state["xyz_text"]         = xyz_text
                            st.session_state["mol_label"]        = mol_label
                            st.session_state["last_drawn_smiles"] = drawn_smiles
                            st.session_state["last_result"]      = None
                        except ValueError as e:
                            st.error(str(e))

    # If the user navigates away and back (or the rerun lost xyz_text),
    # restore it from session state. This keeps the right-hand viewer alive.
    if xyz_text is None and st.session_state.get("xyz_text"):
        xyz_text  = st.session_state["xyz_text"]
        mol_label = st.session_state.get("mol_label")

    # ── XYZ preview ────────────────────────────────────────────────────────
    if xyz_text is not None:
        with st.expander("XYZ preview", expanded=False):
            st.code(
                xyz_text[:2000] + (" …" if len(xyz_text) > 2000 else ""),
                language="text",
            )

    # ── Run-prediction button (still inside left column) ──────────────────
    st.markdown("")
    run_btn = st.button(
        "Run prediction",
        type="primary",
        disabled=(xyz_text is None),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# RIGHT — live 3D viewer (and, after a prediction, MO overlay controls)
# ─────────────────────────────────────────────────────────────────────────
with col_viewer:
    st.markdown('<div class="section-label">3D structure</div>', unsafe_allow_html=True)

    # Pull the most recent prediction (if any) so we can show MO controls
    # in this same column right below the viewer.
    _viewer_last = st.session_state.get("last_result")
    _orbitals_available = (
        _viewer_last is not None
        and _HAVE_PY3DMOL
        and xyz_text is not None
    )

    if xyz_text is None:
        st.markdown(
            '<div class="viewer-empty">'
            'No molecule yet — provide an input on the left and the 3D structure '
            'will appear here.'
            '</div>',
            unsafe_allow_html=True,
        )
    elif not _HAVE_PY3DMOL:
        st.warning(
            "`py3Dmol` is not installed — cannot render 3D viewer.\n\n"
            "Install with: `pip install py3Dmol`"
        )
    else:
        # If a prediction has been run AND the user toggled orbitals on,
        # we render the molecule + MO isosurface in this same viewer.
        # Otherwise: plain sticks-and-balls preview.
        show_mo = bool(st.session_state.get("show_orbitals", False)) and _orbitals_available

        # We need to compute orb_data BEFORE rendering the viewer when
        # show_mo is on, because the viewer HTML embeds the cube text.
        orb_data = None
        if show_mo:
            try:
                with st.spinner(
                    "Computing orbital cubes "
                    "(first time on this molecule: 5–30 s; cached afterwards)…"
                ):
                    orb_data = compute_hf_orbitals(
                        _viewer_last["an_bytes"],
                        _viewer_last["co_bytes"],
                        _viewer_last["basis"],
                        orb_cache_dir,
                    )
            except Exception as e:
                st.error(f"Orbital generation failed: {e}")
                orb_data = None

        # Resolve which orbital + isovalue the user picked (if MO mode).
        # We read these from session_state so the viewer can render with
        # the current selection on every rerun.
        sel_idx_in_list = 0
        isovalue        = 0.04
        if show_mo and orb_data is not None and orb_data["cubes"]:
            sel_label = st.session_state.get("orbital_choice", "HOMO")
            if sel_label not in orb_data["labels"]:
                sel_label = (
                    "HOMO" if "HOMO" in orb_data["labels"]
                    else orb_data["labels"][0]
                )
            sel_idx_in_list = orb_data["labels"].index(sel_label)
            isovalue = float(st.session_state.get("orbital_isovalue", 0.04))

        # ── Render the viewer ────────────────────────────────────────────
        try:
            if show_mo and orb_data is not None and orb_data["cubes"]:
                cube_text = orb_data["cubes"][sel_idx_in_list]
                html = render_molecule_with_orbital(
                    xyz_text, cube_text,
                    isovalue=isovalue,
                    height_px=460,
                )
            else:
                html = render_molecule_3d(xyz_text, height_px=460)

            if html is not None:
                st.components.v1.html(html, height=480, scrolling=False)

                if show_mo and orb_data is not None and orb_data["cubes"]:
                    sel_label = orb_data["labels"][sel_idx_in_list]
                    mo_idx    = orb_data["indices"][sel_idx_in_list]
                    mo_e_ev   = orb_data["energies_eV"][sel_idx_in_list]
                    mo_occ    = orb_data["occupations"][sel_idx_in_list]
                    st.markdown(
                        f'<div class="viewer-caption">'
                        f'{sel_label} · MO #{mo_idx + 1}/{orb_data["n_mo"]} · '
                        f'ε = {mo_e_ev:.3f} eV · occ = {mo_occ:.1f} · '
                        f'<span style="color:#1e6fd6">■</span> ψ &gt; +{isovalue:.3f}'
                        f' &nbsp; <span style="color:#c0392b">■</span> ψ &lt; −{isovalue:.3f}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif mol_label:
                    st.markdown(
                        f'<div class="viewer-caption">{mol_label}</div>',
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"3D render failed: {e}")

    # ── Orbital controls (right under the viewer in this same column) ────
    # Only shown after a prediction has been run.
    if _orbitals_available:
        st.checkbox(
            "Show calculated HF molecular orbitals",
            help="Overlay the four frontier orbitals (HOMO−1, HOMO, LUMO, "
                 "LUMO+1) on the molecule above. Cubes are generated in "
                 "memory from the same HF wave function that produced E_HF. "
                 "They are NOT written to disk unless you click "
                 "“Store orbitals in cache” below.",
            key="show_orbitals",
        )

        if st.session_state.get("show_orbitals") and orb_data is not None and orb_data["cubes"]:
            ctrl_a, ctrl_b = st.columns([1, 1])
            with ctrl_a:
                st.selectbox(
                    "Orbital",
                    orb_data["labels"],
                    index=orb_data["labels"].index(
                        orb_data["labels"][sel_idx_in_list]
                    ),
                    key="orbital_choice",
                    help="Select which frontier orbital to display.",
                )
            with ctrl_b:
                st.slider(
                    "Isovalue",
                    min_value=0.005, max_value=0.10, value=0.04, step=0.005,
                    key="orbital_isovalue",
                    help="Larger values show smaller, denser lobes; "
                         "smaller values show larger, more diffuse lobes.",
                )

            # ── Manual "Store in cache" button ────────────────────────────
            # Showing orbitals is decoupled from saving them. The user may
            # only want to inspect them once; we don't waste disk on that.
            # The button is disabled if a cache file already exists for
            # this (geometry, basis), and re-enables for new molecules.
            an_arr = np.frombuffer(_viewer_last["an_bytes"], dtype=int)
            co_arr = np.frombuffer(
                _viewer_last["co_bytes"], dtype=float
            ).reshape(-1, 3)
            already_cached = is_orbital_cached(
                an_arr, co_arr, _viewer_last["basis"], orb_cache_dir
            )

            store_col, status_col = st.columns([1, 2])
            with store_col:
                store_clicked = st.button(
                    "Store orbitals in cache",
                    disabled=already_cached,
                    help=(
                        "Write the four frontier-orbital cube files to disk "
                        "so they are reused on subsequent runs without "
                        "re-running HF."
                        if not already_cached
                        else "Already cached on disk for this molecule "
                             "and basis."
                    ),
                    key="store_orbitals_btn",
                    use_container_width=True,
                )
            with status_col:
                if already_cached:
                    cf = orbital_cache_path(
                        an_arr, co_arr,
                        _viewer_last["basis"], orb_cache_dir,
                    )
                    st.markdown(
                        f'<div style="font-family:monospace;font-size:0.72rem;'
                        f'color:#888;padding-top:0.5rem">'
                        f'✓ already on disk · <code>{cf.name}</code>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif store_clicked:
                    ok, msg = save_hf_orbitals_to_cache(
                        orb_data, an_arr, co_arr,
                        _viewer_last["basis"], orb_cache_dir,
                    )
                    if ok:
                        st.markdown(
                            f'<div style="font-family:monospace;font-size:0.72rem;'
                            f'color:#1a8c3a;padding-top:0.5rem">'
                            f'✓ {msg}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(msg)
                else:
                    st.markdown(
                        '<div style="font-family:monospace;font-size:0.72rem;'
                        'color:#888;padding-top:0.5rem">'
                        'in memory only — click to persist'
                        '</div>',
                        unsafe_allow_html=True,
                    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# Run prediction
# ══════════════════════════════════════════════════════════════════════════

# When the button is pressed we run the pipeline once and stash everything
# we need for the results display in session_state.  All result rendering
# below reads from session_state["last_result"], so flipping the "show MO"
# toggle (which causes a Streamlit rerun) does NOT recompute the energies.

if run_btn and xyz_text is not None:
    try:
        atomic_numbers, coords, n_atoms, formula = parse_xyz(xyz_text)
    except ValueError as e:
        st.error(f"XYZ parse error: {e}")
        st.stop()

    st.markdown(f"**Molecule:** `{formula}` · {n_atoms} atoms · basis: `{hf_basis}`")

    t0 = time.time()
    with st.spinner(
        f"Running HF/{hf_basis} + descriptor computation… "
        "(first run: 10–60 s depending on molecule size)"
    ):
        try:
            result = run_prediction(
                atomic_numbers, coords,
                model_elec, model_soap, model_comb, model_gnn, gnn_stats,
                hf_basis, w_atomic, cache_dir,
                xyz_text=xyz_text,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    elapsed = time.time() - t0

    if not result["success"]:
        st.error(result["error"])
        st.stop()

    st.success(f"Completed in {elapsed:.1f} s")

    # Persist for subsequent reruns (orbital toggle, etc.)
    st.session_state["last_result"] = {
        "result":          result,
        "formula":         formula,
        "n_atoms":         n_atoms,
        "elapsed":         elapsed,
        "basis":           hf_basis,
        # numpy arrays don't survive a session_state round-trip well —
        # keep them as bytes so we can re-feed compute_hf_orbitals later.
        "an_bytes":        atomic_numbers.tobytes(),
        "co_bytes":        coords.tobytes(),
        "xyz_text":        xyz_text,
    }

    # The right-hand viewer column rendered ABOVE this block — it captured
    # the previous (or absent) value of last_result. Trigger one extra rerun
    # so the orbital toggle and updated state appear in that column without
    # the user needing to interact again.
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Results display (driven by st.session_state["last_result"])
# ══════════════════════════════════════════════════════════════════════════

_last = st.session_state.get("last_result")

if _last is not None:
    result   = _last["result"]
    formula  = _last["formula"]
    n_atoms  = _last["n_atoms"]
    basis    = _last["basis"]

    # ── Reference energies ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Reference energies</div>', unsafe_allow_html=True)
    ref_col1, ref_col2 = st.columns(2)

    E_HF_eV  = result.get("E_HF_eV")
    E_HF_ha  = result.get("E_HF_hartree")
    mp2_ev   = result.get("E_corr_mp2_eV")
    mp2_kcal = result.get("E_corr_mp2_kcal")

    with ref_col1:
        st.markdown(
            f'<div class="energy-card">'
            f'<div class="model-label">E<sub>HF</sub> — Hartree-Fock</div>'
            f'<div class="value-ev">{E_HF_eV:.4f} eV</div>'
            f'<div class="value-kcal">{E_HF_ha:.6f} Hartree</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with ref_col2:
        if mp2_ev is not None:
            st.markdown(
                f'<div class="energy-card">'
                f'<div class="model-label col-mp2">E<sub>corr</sub> — MP2 reference</div>'
                f'<div class="value-ev">{mp2_ev:.4f} eV</div>'
                f'<div class="value-kcal">{mp2_kcal:.2f} kcal/mol</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="energy-card">'
                '<div class="model-label col-mp2">E<sub>corr</sub> — MP2 reference</div>'
                '<div class="not-loaded">not computed</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── Model predictions ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Model predictions</div>', unsafe_allow_html=True)

    models_info = [
        ("Electronic", "col-elec",
         result.get("E_corr_elec_eV"), result.get("E_corr_elec_kcal"),
         "CorrNet · HF density matrix eigenvalues (108/atom)", None),
        ("SOAP", "col-soap",
         result.get("E_corr_soap_eV"), result.get("E_corr_soap_kcal"),
         "CorrNet · Smooth Overlap of Atomic Positions (geometric only, no QM)", None),
        ("Combined", "col-comb",
         result.get("E_corr_comb_eV"), result.get("E_corr_comb_kcal"),
         f"CorrNet · Electronic + SOAP (w_atomic = {w_atomic})", None),
        ("GNN", "col-gnn",
         result.get("E_corr_gnn_eV"), result.get("E_corr_gnn_kcal"),
         "Graph Neural Network · NNConv + LSTM message passing (T = 5)",
         result.get("gnn_warning")),
    ]

    pred_cols = st.columns(2)
    for idx, (name, cls, ev, kcal, desc, warning) in enumerate(models_info):
        with pred_cols[idx % 2]:
            if ev is None:
                warn_html = (
                    f'<div style="font-size:0.73rem;color:#b00;margin-top:0.4rem;'
                    f'font-family:monospace;white-space:pre-wrap">{warning}</div>'
                    if warning else ""
                )
                st.markdown(
                    f'<div class="energy-card">'
                    f'<div class="model-label {cls}">{name}</div>'
                    f'<div class="not-loaded">model not loaded</div>'
                    f'{warn_html}'
                    f'<div style="font-size:0.73rem;color:#aaa;margin-top:0.35rem">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                if mp2_ev is not None:
                    delta_ev   = abs(ev - mp2_ev)
                    delta_kcal = delta_ev * EV_TO_KCAL
                    ok         = delta_ev < CHEM_ACCURACY_EV
                    delta_cls  = "delta-good" if ok else "delta-bad"
                    chk        = " ✓ chemical accuracy" if ok else ""
                    delta_html = (
                        f'<div class="delta {delta_cls}">'
                        f'Δ vs MP2: {delta_ev:.4f} eV · {delta_kcal:.2f} kcal/mol{chk}'
                        f'</div>'
                    )
                else:
                    delta_html = '<div class="delta delta-na">Δ vs MP2: —</div>'

                st.markdown(
                    f'<div class="energy-card">'
                    f'<div class="model-label {cls}">{name}</div>'
                    f'<div class="value-ev">{ev:.4f} eV</div>'
                    f'<div class="value-kcal">{kcal:.2f} kcal/mol</div>'
                    f'{delta_html}'
                    f'<div style="font-size:0.73rem;color:#aaa;margin-top:0.4rem">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Comparison chart ──────────────────────────────────────────────────
    has_preds = any(
        result.get(k) is not None
        for k in ["E_corr_elec_eV", "E_corr_soap_eV", "E_corr_comb_eV", "E_corr_gnn_eV"]
    )
    if has_preds and mp2_ev is not None:
        st.markdown('<div class="section-label">Comparison</div>', unsafe_allow_html=True)
        import pandas as pd
        chart_data = {"MP2 ref": mp2_ev}
        for label, key in [
            ("Electronic", "E_corr_elec_eV"),
            ("SOAP",       "E_corr_soap_eV"),
            ("Combined",   "E_corr_comb_eV"),
            ("GNN",        "E_corr_gnn_eV"),
        ]:
            if result.get(key) is not None:
                chart_data[label] = result[key]
        df = pd.DataFrame({"Model": list(chart_data), "E_corr (eV)": list(chart_data.values())})
        st.bar_chart(df.set_index("Model"), use_container_width=True)

    # Note: HF molecular orbital visualisation is rendered in the right-hand
    # 3D viewer column at the top of the page (under the section "3D
    # structure"), driven by st.session_state["show_orbitals"]. It is not
    # repeated here.

else:
    # ── Landing / instructions ────────────────────────────────────────────
    st.markdown(
        '<div class="info-box">'
        'Provide a molecular geometry using one of the four input methods above '
        '(upload, SMILES, paste, or draw), then click <strong>Run prediction</strong>.<br><br>'
        'The pipeline will:<br>'
        '&nbsp;&nbsp;1. Run a Hartree-Fock calculation (PySCF) to obtain E<sub>HF</sub> '
        'and the MP2 correlation energy reference.<br>'
        '&nbsp;&nbsp;2. Compute <strong>electronic descriptors</strong> (projected density matrix '
        'eigenvalues, 108 per atom) and <strong>SOAP descriptors</strong> (smooth overlap of atomic '
        'positions, geometric only).<br>'
        '&nbsp;&nbsp;3. Run three <strong>CorrNet</strong> models — one per descriptor type '
        '(electronic, SOAP, combined) — and one <strong>GNN</strong> (NNConv + LSTM message '
        'passing on the molecular graph).<br><br>'
        'After the prediction, you can also visualise the four frontier '
        '<strong>HF molecular orbitals</strong> (HOMO−1, HOMO, LUMO, LUMO+1) as 3D '
        'isosurfaces.<br><br>'
        'All four model predictions are shown side by side with their error relative to the '
        'MP2 reference. A result within 1 kcal/mol is considered chemically accurate.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### About this tool")
    st.markdown("""
**DeePHFProject** predicts the **correlation energy** E_corr = E_MP2 − E_HF of small
organic molecules from their 3D geometry.

| Model | Descriptor | Notes |
|---|---|---|
| **Electronic** | HF density matrix eigenvalues (108/atom) | Baseline — requires PySCF |
| **SOAP** | Smooth Overlap of Atomic Positions | Geometric only, no QM needed |
| **Combined** | Electronic + SOAP (z-score normalised) | Best accuracy on diverse sets |
| **GNN** | Molecular graph — NNConv + LSTM (T = 5) | End-to-end, no hand-crafted descriptors |

All computations run locally. Molecules already computed are cached to disk.

> *Chemical accuracy:* |ΔE| < 1 kcal/mol ≈ 0.043 eV relative to the MP2 reference.
""")

# ── Citation footer ────────────────────────────────────────────────────────
st.markdown(
    '<div class="citation-box">'
    '<strong>Reference</strong><br>'
    'Built on DeePHF — Chen, Y., Zhang, L., Wang, H., &amp; E, W. (2020). '
    'DeePHF: A machine learning-based electron correlation and excitation energy predictor. '
    '<em>The Journal of Chemical Physics</em>, 152, 034102.'
    '</div>',
    unsafe_allow_html=True,
)
