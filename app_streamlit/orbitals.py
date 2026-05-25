"""HF molecular orbital cube generation + 3D rendering helpers.

Self-contained module that runs a fresh RHF calculation via PySCF and
generates cube files for HOMO-1, HOMO, LUMO, LUMO+1. Cubes are returned
in memory as plain-text strings (suitable for py3Dmol's addVolumetricData).
A disk cache keyed by (geometry, basis) hash makes repeat views instant.

The orbital pipeline is independent of the inference pipeline — it
re-runs HF rather than sharing state with the descriptor cache (the
descriptor cache stores eigenvalues only, not mo_coeff/mo_energy).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import py3Dmol
    _HAVE_PY3DMOL = True
except ImportError:
    _HAVE_PY3DMOL = False


HARTREE_TO_EV = 27.211386245988

# Coarser than PySCF's default (80^3) for speed on Spaces CPU
_CUBE_NX = _CUBE_NY = _CUBE_NZ = 60

_Z_TO_SYM = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    14: "Si", 15: "P", 16: "S", 17: "Cl",
    35: "Br", 53: "I",
}


def mol_hash(Z: np.ndarray, coords: np.ndarray, basis: str) -> str:
    raw = (
        np.asarray(Z, dtype=int).tobytes()
        + np.asarray(coords, dtype=float).tobytes()
        + basis.encode()
    )
    return hashlib.md5(raw).hexdigest()


def orbital_cache_path(Z, coords, basis, orb_cache_dir) -> Path:
    h = mol_hash(Z, coords, basis)
    return Path(orb_cache_dir) / f"{h}.npz"


def is_orbital_cached(Z, coords, basis, orb_cache_dir) -> bool:
    return orbital_cache_path(Z, coords, basis, orb_cache_dir).is_file()


def compute_hf_orbitals(
    Z: np.ndarray,
    coords: np.ndarray,
    basis: str,
    orb_cache_dir: str,
    nx: int = _CUBE_NX,
    ny: int = _CUBE_NY,
    nz: int = _CUBE_NZ,
) -> Dict[str, Any]:
    """Compute (or load from cache) HOMO-1, HOMO, LUMO, LUMO+1 cube data.

    Returns dict with keys: labels, indices, energies_eV, occupations,
    cubes (list of cube-format text), homo_idx, n_mo.
    """
    Z = np.asarray(Z, dtype=int)
    coords = np.asarray(coords, dtype=float)

    Path(orb_cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = orbital_cache_path(Z, coords, basis, orb_cache_dir)

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
        except Exception:
            pass  # corrupt cache → recompute

    from pyscf import gto, scf
    from pyscf.tools import cubegen

    atom_list = []
    for i in range(len(Z)):
        sym = _Z_TO_SYM.get(int(Z[i]))
        if sym is None:
            raise RuntimeError(f"Unknown element Z={Z[i]} for orbital cube generation.")
        atom_list.append(
            f"{sym} {coords[i, 0]:.10f} {coords[i, 1]:.10f} {coords[i, 2]:.10f}"
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

    mo_occ, mo_energy, mo_coeff = mf.mo_occ, mf.mo_energy, mf.mo_coeff
    n_mo = mo_coeff.shape[1]

    occupied = np.where(mo_occ > 0)[0]
    if len(occupied) == 0:
        raise RuntimeError("No occupied orbitals found.")
    homo_idx = int(occupied[-1])
    lumo_idx = homo_idx + 1

    candidates: List[Tuple[str, int]] = [
        ("HOMO-1", homo_idx - 1),
        ("HOMO",   homo_idx),
        ("LUMO",   lumo_idx),
        ("LUMO+1", lumo_idx + 1),
    ]
    selected = [(lab, idx) for lab, idx in candidates if 0 <= idx < n_mo]

    labels:      List[str]   = []
    indices:     List[int]   = []
    energies_eV: List[float] = []
    occs:        List[float] = []
    cubes:       List[str]   = []

    for lab, idx in selected:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cube", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cubegen.orbital(mol, tmp_path, mo_coeff[:, idx], nx=nx, ny=ny, nz=nz)
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


def save_hf_orbitals_to_cache(
    orb_data: Dict[str, Any],
    Z: np.ndarray,
    coords: np.ndarray,
    basis: str,
    orb_cache_dir: str,
) -> Tuple[bool, str]:
    """Persist in-memory orb_data dict to disk for fast subsequent loads."""
    if not orb_data or not orb_data.get("cubes"):
        return False, "No orbital data to store."

    cache_file = orbital_cache_path(Z, coords, basis, orb_cache_dir)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_file,
            labels=np.array(orb_data["labels"]),
            indices=np.array(orb_data["indices"], dtype=int),
            energies_eV=np.array(orb_data["energies_eV"], dtype=float),
            occupations=np.array(orb_data["occupations"], dtype=float),
            cubes=np.array(orb_data["cubes"], dtype=object),
            homo_idx=np.int64(orb_data["homo_idx"]),
            n_mo=np.int64(orb_data["n_mo"]),
        )
        size_kb = cache_file.stat().st_size / 1024
        return True, f"Stored {size_kb:.0f} KB to {cache_file.name}"
    except Exception as e:
        return False, f"Save failed: {e}"


def render_molecule_3d(
    xyz_text: str,
    height_px: int = 460,
    width_px: Optional[int] = None,
) -> Optional[str]:
    """Plain stick-and-ball 3D viewer (no orbitals)."""
    if not _HAVE_PY3DMOL:
        return None
    view = py3Dmol.view(width=(width_px or 480), height=height_px)
    view.addModel(xyz_text, "xyz")
    view.setStyle({"stick":  {"radius": 0.12, "colorscheme": "Jmol"},
                   "sphere": {"scale": 0.22, "colorscheme": "Jmol"}})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.zoom(1.15)
    html = view._make_html()
    css = """
    <style>
      .mol-container, .mol-container > div, .mol-container canvas {
        width: 100% !important;
      }
    </style>
    <div class="mol-container">
    """
    return css + html + "</div>"


def render_molecule_with_orbital(
    xyz_text: str,
    cube_text: str,
    isovalue: float = 0.04,
    height_px: int = 480,
    width_px: Optional[int] = None,
) -> Optional[str]:
    """3D viewer with MO isosurface overlay (positive blue, negative red)."""
    if not _HAVE_PY3DMOL:
        return None
    view = py3Dmol.view(width=(width_px or 480), height=height_px)
    view.addModel(xyz_text, "xyz")
    view.setStyle({"stick":  {"radius": 0.10, "colorscheme": "Jmol"},
                   "sphere": {"scale": 0.18, "colorscheme": "Jmol"}})
    view.setBackgroundColor("white")
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
    css = """
    <style>
      .mol-container, .mol-container > div, .mol-container canvas {
        width: 100% !important;
      }
    </style>
    <div class="mol-container">
    """
    return css + html + "</div>"
