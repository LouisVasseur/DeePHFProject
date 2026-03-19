"""
DeePHF Unified Data Loader
===========================
Reads all benchmark datasets into a common format for the DeePHF
atomic descriptor ablation study.

Supported datasets:
  - QM7-X     (HDF5, Zenodo)
  - rMD17     (npz, Figshare/Materials Cloud)
  - QM7b      (mat, DeepChem)
  - GMTKN55   (xyz + csv, ACCDB GitHub)
  - Water     (npy, deepks-kit examples)

Common output format (MoleculeData):
  - atomic_numbers: np.ndarray (N,) int
  - coords:         np.ndarray (N, 3) float, Angstrom
  - energy:         float or None, eV (converted from source units)
  - forces:         np.ndarray (N, 3) or None, eV/Angstrom
  - properties:     dict of any extra per-molecule data
  - source:         str, dataset name
  - mol_id:         str, unique identifier

Usage:
    from deephf_dataloader import load_dataset, list_datasets

    # Load a small subset for testing
    mols = load_dataset("qm7x", data_dir="./deephf_datasets", max_samples=100)
    mols = load_dataset("rmd17", data_dir="./deephf_datasets", molecule="ethanol", max_samples=500)
    mols = load_dataset("qm7b", data_dir="./deephf_datasets")
    mols = load_dataset("gmtkn55", data_dir="./deephf_datasets", subset="S22")
    mols = load_dataset("water", data_dir="./deephf_datasets")

    # Each mol is a dict:
    # {
    #   'atomic_numbers': array([8, 1, 1]),
    #   'coords': array([[0., 0., 0.], ...]),  # Angstrom
    #   'energy': -76.026,                      # eV
    #   'forces': array([[...], ...]) or None,   # eV/Ang
    #   'properties': {'E_corr': ..., ...},
    #   'source': 'qm7x',
    #   'mol_id': 'qm7x_1000_mol001_conf003'
    # }
"""

from __future__ import annotations

import os
import glob
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)

# ── Unit conversions ──────────────────────────────────────────────────────
KCAL_TO_EV = 0.0433641153087705
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903


# ── Common output format ──────────────────────────────────────────────────
@dataclass
class MoleculeData:
    """Standardized molecule representation."""
    atomic_numbers: np.ndarray          # (N,) int
    coords: np.ndarray                  # (N, 3) Angstrom
    energy: Optional[float] = None      # eV
    forces: Optional[np.ndarray] = None # (N, 3) eV/Angstrom
    properties: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    mol_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def elements(self) -> List[str]:
        """Return element symbols."""
        from ase.data import chemical_symbols
        return [chemical_symbols[z] for z in self.atomic_numbers]

    @property
    def formula(self) -> str:
        """Simple molecular formula."""
        from collections import Counter
        counts = Counter(self.elements)
        return "".join(f"{el}{c if c > 1 else ''}" for el, c in sorted(counts.items()))


# ══════════════════════════════════════════════════════════════════════════
# QM7-X Loader
# ══════════════════════════════════════════════════════════════════════════
def load_qm7x(
    data_dir: str,
    max_samples: Optional[int] = None,
    files: Optional[List[str]] = None,
    equilibrium_only: bool = False,
) -> List[MoleculeData]:
    """
    Load QM7-X from HDF5 files.

    Args:
        data_dir: Path to deephf_datasets/qm7x/ or directory containing *.hdf5
        max_samples: Limit number of conformations loaded
        files: Specific HDF5 files to load (e.g. ["8000.hdf5"] for quick test)
        equilibrium_only: If True, only load equilibrium structures (sRMSD == 0)
    """
    import h5py

    qm7x_dir = _find_subdir(data_dir, "qm7x")

    if files is None:
        hdf5_files = sorted(glob.glob(os.path.join(qm7x_dir, "*.hdf5")))
    else:
        hdf5_files = [os.path.join(qm7x_dir, f) for f in files]

    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {qm7x_dir}")

    molecules = []
    count = 0

    for fpath in hdf5_files:
        fname = os.path.basename(fpath)
        logger.info(f"Loading QM7-X: {fname}")

        with h5py.File(fpath, "r") as f:
            for mol_id in f.keys():
                for conf_id in f[mol_id].keys():
                    if max_samples and count >= max_samples:
                        return molecules

                    data = f[mol_id][conf_id]

                    # Filter equilibrium if requested
                    if equilibrium_only:
                        rmsd = float(np.asarray(data["sRMSD"]).flat[0])
                        if rmsd > 1e-6:
                            continue

                    Z = data["atNUM"][:]
                    R = data["atXYZ"][:]  # already in Angstrom

                    # Energies are in eV in QM7-X
                    # Use .flat[0] to safely extract scalar (avoids NumPy deprecation)
                    _s = lambda d: float(np.asarray(d).flat[0])
                    e_pbe0 = _s(data["ePBE0"])
                    e_corr = _s(data["eC"])
                    e_x = _s(data["eX"])
                    e_xc = _s(data["eXC"])

                    forces_raw = data["totFOR"][:]  # eV/Angstrom

                    props = {
                        "E_PBE0": e_pbe0,
                        "E_corr": e_corr,
                        "E_exchange": e_x,
                        "E_XC": e_xc,
                        "E_atomization": _s(data["eAT"]),
                        "HOMO": _s(data["eH"]),
                        "LUMO": _s(data["eL"]),
                        "sRMSD": _s(data["sRMSD"]),
                    }

                    # Add Hirshfeld charges if available
                    if "hRAT" in data:
                        props["hirshfeld_charges"] = data["hRAT"][:]

                    molecules.append(MoleculeData(
                        atomic_numbers=Z.astype(int),
                        coords=R,
                        energy=e_pbe0,
                        forces=forces_raw,
                        properties=props,
                        source="qm7x",
                        mol_id=f"qm7x_{fname.replace('.hdf5','')}_{mol_id}_{conf_id}",
                    ))
                    count += 1

    logger.info(f"Loaded {len(molecules)} structures from QM7-X")
    return molecules


# ══════════════════════════════════════════════════════════════════════════
# rMD17 Loader
# ══════════════════════════════════════════════════════════════════════════
AVAILABLE_RMD17_MOLECULES = [
    "aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
    "naphthalene", "paracetamol", "salicylic", "toluene", "uracil",
]

def load_rmd17(
    data_dir: str,
    molecule: str = "ethanol",
    max_samples: Optional[int] = None,
    stride: int = 1,
) -> List[MoleculeData]:
    """
    Load revised MD17 dataset.

    Args:
        data_dir: Path to deephf_datasets/
        molecule: One of AVAILABLE_RMD17_MOLECULES
        max_samples: Limit number of structures
        stride: Subsample every N-th frame (recommended: stride >= 100
                due to MD autocorrelation)
    """
    md17_dir = _find_subdir(data_dir, "md17")

    # Try different possible locations (Figshare zip can nest differently)
    candidates = [
        os.path.join(md17_dir, "rmd17", "npz_data", f"rmd17_{molecule}.npz"),
        os.path.join(md17_dir, "rmd17", f"rmd17_{molecule}.npz"),
        os.path.join(md17_dir, f"rmd17_{molecule}.npz"),
        os.path.join(md17_dir, "rmd17", f"{molecule}.npz"),
    ]
    # Also search recursively — handles any nesting from Figshare extraction
    found = glob.glob(os.path.join(md17_dir, "**", f"*{molecule}*.npz"), recursive=True)
    if found:
        # Prefer files with "rmd17" in name
        rmd17_found = [f for f in found if "rmd17" in os.path.basename(f)]
        candidates = (rmd17_found or found) + candidates

    fpath = None
    for c in candidates:
        if os.path.exists(c):
            fpath = c
            break

    if fpath is None:
        raise FileNotFoundError(
            f"rMD17 file for '{molecule}' not found in {md17_dir}. "
            f"Available: {AVAILABLE_RMD17_MOLECULES}"
        )

    logger.info(f"Loading rMD17: {molecule} from {fpath}")
    data = np.load(fpath)

    Z = data["nuclear_charges"].astype(int)
    all_coords = data["coords"]         # (N_frames, N_atoms, 3) Angstrom
    all_E = data["energies"]            # (N_frames,) kcal/mol
    all_F = data["forces"]              # (N_frames, N_atoms, 3) kcal/mol/Ang

    # Subsample
    indices = np.arange(0, len(all_E), stride)
    if max_samples:
        indices = indices[:max_samples]

    molecules = []
    for i in indices:
        molecules.append(MoleculeData(
            atomic_numbers=Z.copy(),
            coords=all_coords[i],
            energy=float(all_E[i]) * KCAL_TO_EV,
            forces=all_F[i] * KCAL_TO_EV,
            properties={
                "energy_kcal": float(all_E[i]),
            },
            source="rmd17",
            mol_id=f"rmd17_{molecule}_{i:06d}",
        ))

    logger.info(f"Loaded {len(molecules)} structures from rMD17/{molecule}")
    return molecules


# ══════════════════════════════════════════════════════════════════════════
# QM7b Loader
# ══════════════════════════════════════════════════════════════════════════
def load_qm7b(
    data_dir: str,
    max_samples: Optional[int] = None,
) -> List[MoleculeData]:
    """
    Load QM7b dataset from .mat file.

    Note: QM7b only provides Coulomb matrices, not explicit xyz coordinates.
    We reconstruct atomic numbers from the diagonal of the Coulomb matrix
    (diagonal elements = 0.5 * Z^2.4). Coordinates are NOT available —
    you'll need to retrieve them from GDB-13 or run geometry optimization.

    For DeePHF descriptor computation, you need actual 3D coordinates.
    Consider using QM7-X instead, which provides full xyz + properties.
    """
    from scipy.io import loadmat

    qm7b_dir = _find_subdir(data_dir, "qm7b")
    fpath = os.path.join(qm7b_dir, "qm7b.mat")

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"QM7b file not found at {fpath}")

    logger.info(f"Loading QM7b from {fpath}")
    mat = loadmat(fpath)

    X = mat["X"]  # (7211, 23, 23) Coulomb matrices
    T = mat["T"]  # (7211, 14) properties

    # Property names (from the QM7b paper)
    prop_names = [
        "E_PBE0", "E_HF", "E_ZINDO", "dipole_PBE0",
        "IP_ZINDO", "EA_ZINDO", "E1_ZINDO", "Emax_ZINDO",
        "Imax_ZINDO", "HOMO_GW", "LUMO_GW", "HOMO_PBE0",
        "LUMO_PBE0", "alpha_PBE0",
    ]

    n_samples = min(len(X), max_samples) if max_samples else len(X)
    molecules = []

    for i in range(n_samples):
        # Extract atomic numbers from Coulomb matrix diagonal
        diag = np.diag(X[i])
        # diagonal = 0.5 * Z^2.4, so Z = (2*diag)^(1/2.4)
        nonzero = diag > 0.01
        Z_float = (2.0 * diag[nonzero]) ** (1.0 / 2.4)
        Z = np.round(Z_float).astype(int)

        props = {name: float(T[i, j]) for j, name in enumerate(prop_names)}
        props["coulomb_matrix"] = X[i]

        molecules.append(MoleculeData(
            atomic_numbers=Z,
            coords=np.zeros((len(Z), 3)),  # NOT AVAILABLE in QM7b
            energy=props.get("E_PBE0"),
            forces=None,
            properties=props,
            source="qm7b",
            mol_id=f"qm7b_{i:05d}",
        ))

    logger.info(
        f"Loaded {len(molecules)} molecules from QM7b. "
        f"WARNING: No 3D coordinates — use QM7-X for descriptor computation."
    )
    return molecules


# ══════════════════════════════════════════════════════════════════════════
# GMTKN55 / ACCDB Loader
# ══════════════════════════════════════════════════════════════════════════
GMTKN55_SUBSETS = [
    "BH76", "ISO34", "S22", "S66", "W4-11", "DARC", "IDISP", "WATER27",
]

def load_gmtkn55(
    data_dir: str,
    subset: str = "S22",
    max_samples: Optional[int] = None,
) -> List[MoleculeData]:
    """
    Load GMTKN55 subset from ACCDB repository.

    Returns molecules with reaction reference energies in properties.
    Each molecule's properties dict includes:
      - 'subset': subset name
      - 'reactions': list of dicts, each with:
          - 'reaction_id': str
          - 'coefficient': float (this molecule's stoichiometry in that reaction)
          - 'ref_energy_hartree': float (total reaction reference energy)
          - 'ref_energy_eV': float
          - 'partners': list of (coeff, mol_name) for all molecules in the reaction

    Args:
        data_dir: Path to deephf_datasets/
        subset: GMTKN55 subset name (e.g. "S22", "BH76", "W4-11")
    """
    accdb_dir = _find_subdir(data_dir, "gmtkn55_accdb")
    geom_dir = os.path.join(accdb_dir, "Geometries")

    if not os.path.isdir(geom_dir):
        raise FileNotFoundError(f"ACCDB Geometries dir not found at {geom_dir}")

    # ACCDB stores GMTKN55 subsets as .list files: GMTKN_S22.list, GMTKN_BH76.list, etc.
    gmtkn55_dir = os.path.join(accdb_dir, "Databases", "GMTKN", "GMTKN55")
    list_file = None

    if os.path.isdir(gmtkn55_dir):
        candidates = [
            os.path.join(gmtkn55_dir, f"GMTKN_{subset}.list"),
            os.path.join(gmtkn55_dir, f"{subset}.list"),
        ]
        for c in candidates:
            if os.path.exists(c):
                list_file = c
                break

    if list_file is None:
        available = []
        if os.path.isdir(gmtkn55_dir):
            for f in sorted(os.listdir(gmtkn55_dir)):
                if f.startswith("GMTKN_") and f.endswith(".list"):
                    available.append(f.replace("GMTKN_", "").replace(".list", ""))
        raise FileNotFoundError(
            f"GMTKN55 subset '{subset}' not found. Available: {available}"
        )

    # ── Parse DatasetEval.csv for reaction definitions and reference energies ──
    # Format: reaction_id, coeff1, mol1, coeff2, mol2, ..., ref_energy_Hartree
    # e.g.: S22_1,-1,S22_2a,-1,S22_2b,1,S22_2,0.005251
    reactions = {}  # reaction_id -> {ref_energy, participants: [(coeff, mol_name), ...]}
    mol_to_reactions = {}  # mol_name -> [reaction dicts]

    csv_file = os.path.join(gmtkn55_dir, "DatasetEval.csv")
    if os.path.exists(csv_file):
        with open(csv_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    continue

                rxn_id = parts[0].strip()
                # Only include reactions belonging to this subset
                if not rxn_id.startswith(subset + "_") and not rxn_id.startswith(subset.replace("-", "") + "_"):
                    # Also try without the subset prefix for subsets like W4-11
                    prefix_match = False
                    for p in [subset, subset.replace("-", ""), subset.replace("-", "_")]:
                        if rxn_id.startswith(p + "_") or rxn_id.startswith(p):
                            prefix_match = True
                            break
                    if not prefix_match:
                        continue

                ref_energy = float(parts[-1].strip())

                # Parse coefficient-molecule pairs from the middle columns
                participants = []
                i = 1
                while i < len(parts) - 1:
                    coeff = float(parts[i].strip())
                    mol_name = parts[i + 1].strip()
                    participants.append((coeff, mol_name))
                    i += 2

                rxn = {
                    "reaction_id": rxn_id,
                    "ref_energy_hartree": ref_energy,
                    "ref_energy_eV": ref_energy * HARTREE_TO_EV,
                    "participants": participants,
                }
                reactions[rxn_id] = rxn

                # Map each molecule to the reactions it participates in
                for coeff, mol_name in participants:
                    if mol_name not in mol_to_reactions:
                        mol_to_reactions[mol_name] = []
                    mol_to_reactions[mol_name].append({
                        "reaction_id": rxn_id,
                        "coefficient": coeff,
                        "ref_energy_hartree": ref_energy,
                        "ref_energy_eV": ref_energy * HARTREE_TO_EV,
                        "partners": participants,
                    })

        logger.info(f"  Parsed {len(reactions)} reactions for {subset} from DatasetEval.csv")
    else:
        logger.warning(f"  DatasetEval.csv not found at {csv_file}")

    # ── Read molecule filenames from the .list file ──
    mol_files = []
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                mol_files.append(line)

    logger.info(f"Loading GMTKN55/{subset}: {len(mol_files)} geometry files from {os.path.basename(list_file)}")

    molecules = []
    count = 0

    for mol_file in mol_files:
        if max_samples and count >= max_samples:
            break

        xyz_path = os.path.join(geom_dir, mol_file)
        if not xyz_path.endswith(".xyz"):
            xyz_path += ".xyz"

        if not os.path.exists(xyz_path):
            logger.debug(f"Skipping missing file: {xyz_path}")
            continue

        Z, R = _read_xyz(xyz_path)

        # Strip .xyz extension for matching against CSV mol names
        mol_name = mol_file.replace(".xyz", "") if mol_file.endswith(".xyz") else mol_file
        rxn_list = mol_to_reactions.get(mol_name, [])

        molecules.append(MoleculeData(
            atomic_numbers=Z,
            coords=R,
            energy=None,  # Per-molecule absolute energies not provided
            forces=None,
            properties={
                "subset": subset,
                "filename": mol_file,
                "reactions": rxn_list,
                "n_reactions": len(rxn_list),
            },
            source="gmtkn55",
            mol_id=f"gmtkn55_{subset}_{mol_name}",
        ))
        count += 1

    n_with_rxn = sum(1 for m in molecules if m.properties.get("n_reactions", 0) > 0)
    logger.info(f"Loaded {len(molecules)} structures from GMTKN55/{subset} "
                f"({n_with_rxn} linked to {len(reactions)} reactions)")
    return molecules


def load_gmtkn55_reactions(
    data_dir: str,
    subset: str = "S22",
) -> List[dict]:
    """
    Load just the reaction definitions for a GMTKN55 subset.

    Returns list of dicts:
      {
        'reaction_id': 'S22_1',
        'ref_energy_hartree': 0.005251,
        'ref_energy_eV': 0.1429,
        'participants': [(-1, 'S22_2a'), (-1, 'S22_2b'), (1, 'S22_2')],
      }
    """
    accdb_dir = _find_subdir(data_dir, "gmtkn55_accdb")
    gmtkn55_dir = os.path.join(accdb_dir, "Databases", "GMTKN", "GMTKN55")
    csv_file = os.path.join(gmtkn55_dir, "DatasetEval.csv")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"DatasetEval.csv not found at {csv_file}")

    reactions = []
    with open(csv_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue

            rxn_id = parts[0].strip()
            # Filter by subset prefix
            match = False
            for p in [subset, subset.replace("-", ""), subset.replace("-", "_")]:
                if rxn_id.startswith(p + "_") or rxn_id == p:
                    match = True
                    break
            if not match:
                continue

            ref_energy = float(parts[-1].strip())
            participants = []
            i = 1
            while i < len(parts) - 1:
                coeff = float(parts[i].strip())
                mol_name = parts[i + 1].strip()
                participants.append((coeff, mol_name))
                i += 2

            reactions.append({
                "reaction_id": rxn_id,
                "ref_energy_hartree": ref_energy,
                "ref_energy_eV": ref_energy * HARTREE_TO_EV,
                "participants": participants,
            })

    return reactions


# ══════════════════════════════════════════════════════════════════════════
# deepks-kit Water Loader
# ══════════════════════════════════════════════════════════════════════════
def load_water(
    data_dir: str,
    max_samples: Optional[int] = None,
) -> List[MoleculeData]:
    """
    Load water molecules from deepks-kit examples.
    Format: directories containing coord.npy + energy.npy.
    atom.npy may be in the same dir or a sibling/parent dir.
    """
    water_dir = _find_subdir(data_dir, "water")

    # Search for deepks-kit data directories (need at least coord.npy)
    data_dirs = []
    for root, dirs, files in os.walk(water_dir):
        if "coord.npy" in files:
            data_dirs.append(root)

    if not data_dirs:
        # Fallback: check for xyz files
        xyz_files = glob.glob(os.path.join(water_dir, "**", "*.xyz"), recursive=True)
        if xyz_files:
            logger.info(f"Found {len(xyz_files)} xyz files in water dir")
            molecules = []
            for i, xyz in enumerate(xyz_files[:max_samples]):
                Z, R = _read_xyz(xyz)
                molecules.append(MoleculeData(
                    atomic_numbers=Z, coords=R,
                    source="water", mol_id=f"water_xyz_{i:04d}",
                ))
            return molecules

        raise FileNotFoundError(
            f"No deepks-kit data (coord.npy) or xyz files in {water_dir}"
        )

    def _find_atom_npy(ddir: str) -> Optional[str]:
        """Search for atom.npy path in this dir, then siblings, then parents."""
        local = os.path.join(ddir, "atom.npy")
        if os.path.exists(local):
            return local
        parent = os.path.dirname(ddir)
        if os.path.isdir(parent):
            for sibling in sorted(os.listdir(parent)):
                sib_path = os.path.join(parent, sibling, "atom.npy")
                if os.path.exists(sib_path):
                    return sib_path
        return None

    molecules = []
    count = 0

    for ddir in sorted(data_dirs):
        dir_name = os.path.basename(ddir)
        has_atom = os.path.exists(os.path.join(ddir, "atom.npy"))
        has_coord = os.path.exists(os.path.join(ddir, "coord.npy"))

        # Load energy/forces if available
        energy_path = os.path.join(ddir, "energy.npy")
        force_path = os.path.join(ddir, "force.npy")
        energies = np.load(energy_path) if os.path.exists(energy_path) else None
        forces_all = np.load(force_path) if os.path.exists(force_path) else None

        if has_atom:
            # atom.npy: (N_conf, N_atoms_ref, 4) — col 0 = Z, cols 1:4 = coords (Bohr)
            # BUT for clusters, atom.npy only stores ONE water molecule's info
            # while coord.npy stores the FULL cluster coords
            atom_data = np.load(os.path.join(ddir, "atom.npy"))

            if atom_data.ndim == 3 and atom_data.shape[2] == 4:
                Z_template = atom_data[0, :, 0].astype(int)  # e.g. [8, 1, 1]

                if has_coord:
                    # Prefer coord.npy — it has the full cluster geometry
                    coords_all = np.load(os.path.join(ddir, "coord.npy"))
                    if coords_all.ndim == 3:
                        n_conf = coords_all.shape[0]
                        n_atoms_coord = coords_all.shape[1]
                        # Check Bohr
                        if n_conf > 0 and n_atoms_coord >= 2:
                            d01 = np.linalg.norm(coords_all[0, 0] - coords_all[0, 1])
                            if d01 > 1.5:
                                coords_all = coords_all * BOHR_TO_ANG
                        # Tile Z to match coord size
                        if len(Z_template) < n_atoms_coord and n_atoms_coord % len(Z_template) == 0:
                            Z_template = np.tile(Z_template, n_atoms_coord // len(Z_template))
                        elif len(Z_template) != n_atoms_coord:
                            logger.warning(f"Z/coord mismatch in {ddir}: {len(Z_template)} vs {n_atoms_coord}, skipping")
                            continue
                        Z_all = np.tile(Z_template, (n_conf, 1))
                    else:
                        logger.warning(f"Unexpected coord shape {coords_all.shape} in {ddir}, skipping")
                        continue
                else:
                    # No coord.npy — use atom.npy cols 1:4 as coords
                    n_conf = atom_data.shape[0]
                    Z_all = atom_data[:, :, 0].astype(int)
                    coords_all = atom_data[:, :, 1:4] * BOHR_TO_ANG

            elif atom_data.ndim == 1:
                # Simple 1D array of atomic numbers
                Z_template = atom_data.astype(int)
                if has_coord:
                    coords_all = np.load(os.path.join(ddir, "coord.npy"))
                    if coords_all.ndim == 3:
                        n_conf = coords_all.shape[0]
                        coords_all = coords_all * BOHR_TO_ANG
                    else:
                        continue
                    Z_all = np.tile(Z_template, (n_conf, 1))
                else:
                    continue
            else:
                logger.warning(f"Unexpected atom.npy shape {atom_data.shape} in {ddir}, skipping")
                continue

        elif has_coord:
            # No atom.npy — get Z from a sibling's atom.npy
            atom_path = _find_atom_npy(ddir)
            if atom_path is None:
                logger.warning(f"No atom.npy found for {ddir}, skipping")
                continue

            ref_atom = np.load(atom_path)
            if ref_atom.ndim == 3 and ref_atom.shape[2] == 4:
                Z_template = ref_atom[0, :, 0].astype(int)
            elif ref_atom.ndim == 1:
                Z_template = ref_atom.astype(int)
            else:
                logger.warning(f"Unexpected ref atom.npy shape {ref_atom.shape}, skipping {ddir}")
                continue

            coords_all = np.load(os.path.join(ddir, "coord.npy"))
            if coords_all.ndim == 3:
                n_conf = coords_all.shape[0]
                # Check if Bohr
                if n_conf > 0 and coords_all.shape[1] >= 2:
                    d01 = np.linalg.norm(coords_all[0, 0] - coords_all[0, 1])
                    if d01 > 1.5:
                        coords_all = coords_all * BOHR_TO_ANG
            else:
                logger.warning(f"Unexpected coord shape {coords_all.shape} in {ddir}, skipping")
                continue

            n_atoms_coord = coords_all.shape[1]
            # For clusters (n2, n3...), Z_template is for 1 water (3 atoms),
            # but coords has more atoms. Tile Z accordingly.
            if len(Z_template) < n_atoms_coord and n_atoms_coord % len(Z_template) == 0:
                n_repeat = n_atoms_coord // len(Z_template)
                Z_template = np.tile(Z_template, n_repeat)

            Z_all = np.tile(Z_template, (n_conf, 1))
        else:
            continue

        for j in range(n_conf):
            if max_samples and count >= max_samples:
                return molecules

            # Extract per-conformation Z and coords
            if Z_all.ndim == 2:
                Z_j = Z_all[j]
            else:
                Z_j = Z_all  # 1D, same for all confs

            R_j = coords_all[j]

            # Validate shapes match
            if len(Z_j) != R_j.shape[0]:
                logger.debug(
                    f"Shape mismatch in {dir_name} conf {j}: "
                    f"Z={len(Z_j)}, coords={R_j.shape[0]}, skipping"
                )
                continue

            E = None
            if energies is not None and energies.ndim > 0 and j < len(energies):
                E = float(np.asarray(energies[j]).flat[0]) * HARTREE_TO_EV

            F = None
            if forces_all is not None and forces_all.ndim >= 2:
                F_raw = forces_all[j] if forces_all.ndim == 3 else forces_all
                F = np.asarray(F_raw).reshape(-1, 3) * (HARTREE_TO_EV / BOHR_TO_ANG)

            molecules.append(MoleculeData(
                atomic_numbers=np.asarray(Z_j).ravel().astype(int),
                coords=R_j,
                energy=E,
                forces=F,
                properties={"dir": dir_name},
                source="water",
                mol_id=f"water_{dir_name}_{j:04d}",
            ))
            count += 1

    logger.info(f"Loaded {len(molecules)} structures from deepks-kit water")
    return molecules


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
# Caltech MOB-ML Dataset Loader
# ══════════════════════════════════════════════════════════════════════════
MOBML_SUBSETS = ["water", "alkanes", "qm7b_T", "gdb13_T"]

def load_mobml(
    data_dir: str,
    subset: str = "water",
    max_samples: Optional[int] = None,
) -> List[MoleculeData]:
    """
    Load Caltech MOB-ML dataset (the DeePHF benchmark dataset).

    Contains HF and MP2/CCSD(T) energies + geometries for:
      - water: 1000 water molecule geometries (350K)
      - alkanes: 2304 geometries (ethane, propane, butane, isobutane)
      - qm7b_T: 7212 organic molecules thermalized at 350K
      - gdb13_T: GDB-13 subset thermalized at 350K

    Target for DeePHF: E_corr = E_MP2 - E_HF (correlation energy in Hartree).
    Stored in mol.properties['E_corr_hartree'] and mol.energy = E_corr in eV.

    Args:
        data_dir: Path to deephf_datasets/
        subset: One of 'water', 'alkanes', 'qm7b_T', 'gdb13_T'
        max_samples: Limit number of molecules
    """
    # Find the MOB-ML data directory
    mobml_dir = None
    candidates = [
        os.path.join(data_dir, "caltech_mobml", "data", subset),
        os.path.join(data_dir, "caltech_mobml", "data"),
        os.path.join(data_dir, "caltech_mobml"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            # Check if this dir has energy.dat
            if os.path.exists(os.path.join(c, "energy.dat")):
                mobml_dir = c
                break
            # Maybe subset is a subdir
            sub = os.path.join(c, subset)
            if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "energy.dat")):
                mobml_dir = sub
                break

    if mobml_dir is None:
        # Try recursive search
        for root, dirs, files in os.walk(os.path.join(data_dir, "caltech_mobml")):
            if "energy.dat" in files and subset in root:
                mobml_dir = root
                break

    if mobml_dir is None:
        raise FileNotFoundError(
            f"MOB-ML subset '{subset}' not found. "
            f"Expected at: {data_dir}/caltech_mobml/data/{subset}/"
        )

    # Parse energy.dat
    # Format varies by subset:
    #   water:   mol_id  E_HF  E_MP2  E_CCSD  E_CCSD(T)   (4 energy cols)
    #   alkanes: mol_id  E_HF  E_MP2                       (2 energy cols)
    #   qm7b_T:  mol_id  E_HF  E_MP2                       (2 energy cols)
    # Target: best available correlation energy = E_best - E_HF
    energy_file = os.path.join(mobml_dir, "energy.dat")
    energies = {}  # geom_id -> dict of energies

    with open(energy_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            mol_id = parts[0]
            try:
                E_HF = float(parts[1])
                E_MP2 = float(parts[2])
                E_CCSD = float(parts[3]) if len(parts) > 3 else None
                E_CCSDT = float(parts[4]) if len(parts) > 4 else None

                # Best available: CCSD(T) > CCSD > MP2
                E_best = E_CCSDT if E_CCSDT is not None else (E_CCSD if E_CCSD is not None else E_MP2)
                E_corr = E_best - E_HF  # correlation energy in Hartree

                # Extract geometry file ID by stripping subset prefix
                # e.g. "water_00000" -> "00000", "ethane_00001" -> "ethane_00001"
                # For water: strip "water_" prefix
                # For alkanes: keep full name (files may be named differently)
                geom_id = mol_id
                # Try stripping common prefixes
                for prefix in [f"{subset}_", "water_"]:
                    if mol_id.startswith(prefix):
                        geom_id = mol_id[len(prefix):]
                        break

                energies[mol_id] = {
                    "geom_id": geom_id,
                    "E_HF": E_HF,
                    "E_MP2": E_MP2,
                    "E_CCSD": E_CCSD,
                    "E_CCSDT": E_CCSDT,
                    "E_best": E_best,
                    "E_corr": E_corr,
                }
            except (ValueError, IndexError):
                continue

    level = "CCSD(T)" if any(e["E_CCSDT"] is not None for e in energies.values()) else "MP2"
    logger.info(f"  Parsed {len(energies)} energies from {energy_file} (level={level})")

    # Find geometry directory
    geom_dir = None
    for gd in ["geometry", "geometries"]:
        candidate = os.path.join(mobml_dir, gd)
        if os.path.isdir(candidate):
            geom_dir = candidate
            break

    if geom_dir is None:
        raise FileNotFoundError(f"No geometry directory in {mobml_dir}")

    # Load xyz files matched to energy entries
    molecules = []
    count = 0

    for mol_id, edata in sorted(energies.items()):
        if max_samples and count >= max_samples:
            break

        geom_id = edata["geom_id"]
        E_corr = edata["E_corr"]

        # Find xyz file — try geom_id.xyz, then mol_id.xyz, then glob
        xyz_candidates = [
            os.path.join(geom_dir, f"{geom_id}.xyz"),
            os.path.join(geom_dir, f"{mol_id}.xyz"),
        ]
        xyz_path = None
        for xc in xyz_candidates:
            if os.path.exists(xc):
                xyz_path = xc
                break

        if xyz_path is None:
            # Try glob with geom_id
            matches = glob.glob(os.path.join(geom_dir, f"*{geom_id}*"))
            if matches:
                xyz_path = matches[0]

        if xyz_path is None:
            logger.debug(f"No geometry for {mol_id} (geom_id={geom_id}), skipping")
            continue

        try:
            Z, R = _read_xyz(xyz_path)
        except Exception as e:
            logger.debug(f"Failed to read {xyz_path}: {e}")
            continue

        molecules.append(MoleculeData(
            atomic_numbers=Z,
            coords=R,
            energy=E_corr * HARTREE_TO_EV,  # correlation energy in eV
            forces=None,
            properties={
                "E_HF_hartree": edata["E_HF"],
                "E_MP2_hartree": edata["E_MP2"],
                "E_CCSD_hartree": edata["E_CCSD"],
                "E_CCSDT_hartree": edata["E_CCSDT"],
                "E_corr_hartree": E_corr,
                "E_corr_eV": E_corr * HARTREE_TO_EV,
                "E_corr_kcal": E_corr * 627.5094740631,
                "subset": subset,
                "mol_id_original": mol_id,
            },
            source=f"mobml_{subset}",
            mol_id=f"mobml_{subset}_{mol_id}",
        ))
        count += 1

    if not molecules:
        raise ValueError(
            f"No molecules loaded for MOB-ML/{subset}. "
            f"Check geometry file naming: energy.dat has IDs like '{list(energies.keys())[:3]}', "
            f"geometry dir has files like '{os.listdir(geom_dir)[:3]}'"
        )

    corr_kcals = [m.properties["E_corr_kcal"] for m in molecules]
    logger.info(
        f"Loaded {len(molecules)} structures from MOB-ML/{subset} "
        f"(E_corr range: [{min(corr_kcals):.1f}, {max(corr_kcals):.1f}] kcal/mol, "
        f"level={level})"
    )
    return molecules


LOADERS = {
    "qm7x": load_qm7x,
    "rmd17": load_rmd17,
    "md17": load_rmd17,  # alias
    "qm7b": load_qm7b,
    "gmtkn55": load_gmtkn55,
    "water": load_water,
    "mobml": load_mobml,
    "mobml_water": lambda **kw: load_mobml(subset="water", **kw),
    "mobml_alkanes": lambda **kw: load_mobml(subset="alkanes", **kw),
    "mobml_qm7b_t": lambda **kw: load_mobml(subset="qm7b_T", **kw),
}

def list_datasets() -> List[str]:
    """Return available dataset names."""
    return list(LOADERS.keys())


def load_dataset(name: str, data_dir: str = "./deephf_datasets", **kwargs) -> List[MoleculeData]:
    """
    Load a dataset by name.

    Args:
        name: Dataset identifier (qm7x, rmd17, qm7b, gmtkn55, water)
        data_dir: Root directory containing all downloaded datasets
        **kwargs: Dataset-specific arguments (max_samples, molecule, subset, etc.)

    Returns:
        List of MoleculeData objects
    """
    name = name.lower().replace("-", "")
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")

    return LOADERS[name](data_dir=data_dir, **kwargs)


def load_all(
    data_dir: str = "./deephf_datasets",
    max_per_dataset: int = 100,
) -> Dict[str, List[MoleculeData]]:
    """
    Load a sample from every available dataset for quick inspection.

    Args:
        data_dir: Root directory
        max_per_dataset: Max molecules to load per dataset

    Returns:
        Dict mapping dataset name to list of MoleculeData
    """
    results = {}

    configs = [
        ("water", {}),
        ("qm7b", {}),
        ("qm7x", {"files": ["8000.hdf5"]}),  # smallest file
    ]

    # Try rMD17 — try multiple molecules in case some aren't extracted
    for mol in ["ethanol", "benzene", "aspirin"]:
        try:
            mols = load_dataset("rmd17", data_dir=data_dir,
                                molecule=mol, stride=100,
                                max_samples=max_per_dataset)
            results["rmd17"] = mols
            logger.info(f"  rmd17: {len(mols)} molecules loaded ({mol})")
            break
        except FileNotFoundError:
            continue
    else:
        logger.warning("  rmd17: skipped (no npz files found)")

    # GMTKN55 — discover subsets from .list files
    try:
        accdb_dir = _find_subdir(data_dir, "gmtkn55_accdb")
        gmtkn55_dir = os.path.join(accdb_dir, "Databases", "GMTKN", "GMTKN55")
        available = []
        if os.path.isdir(gmtkn55_dir):
            for f in sorted(os.listdir(gmtkn55_dir)):
                if f.startswith("GMTKN_") and f.endswith(".list"):
                    available.append(f.replace("GMTKN_", "").replace(".list", ""))

        if available:
            logger.info(f"  gmtkn55: found {len(available)} subsets")
            for subset in ["S22", "W4-11", "BH76", "WATER27"] + available[:3]:
                if subset in available:
                    try:
                        mols = load_dataset("gmtkn55", data_dir=data_dir,
                                            subset=subset, max_samples=max_per_dataset)
                        results["gmtkn55"] = mols
                        logger.info(f"  gmtkn55: {len(mols)} molecules loaded ({subset})")
                        break
                    except (FileNotFoundError, Exception) as e:
                        logger.debug(f"  gmtkn55/{subset}: {e}")
                        continue
            else:
                logger.warning(f"  gmtkn55: no loadable subset (available: {available[:10]})")
        else:
            logger.warning(f"  gmtkn55: no .list files found in {gmtkn55_dir}")
    except FileNotFoundError as e:
        logger.warning(f"  gmtkn55: skipped ({e})")

    for name, kwargs in configs:
        try:
            mols = load_dataset(name, data_dir=data_dir,
                                max_samples=max_per_dataset, **kwargs)
            results[name] = mols
            logger.info(f"  {name}: {len(mols)} molecules loaded")
        except (FileNotFoundError, ImportError) as e:
            logger.warning(f"  {name}: skipped ({e})")

    return results


# ══════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════
# Inline element lookup (avoids hard dependency on ase for simple cases)
_ELEMENT_MAP = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53,
}

def _sym_to_z(sym: str) -> int:
    """Convert element symbol to atomic number."""
    sym = sym.strip().capitalize()
    if sym in _ELEMENT_MAP:
        return _ELEMENT_MAP[sym]
    try:
        from ase.data import atomic_numbers
        return atomic_numbers[sym]
    except (ImportError, KeyError):
        raise ValueError(f"Unknown element: {sym}")


def _read_xyz(filepath: str):
    """Read an xyz file. Returns (atomic_numbers, coords)."""
    with open(filepath) as f:
        lines = f.readlines()

    # First line: number of atoms (or could be charge/multiplicity in ACCDB)
    # ACCDB xyz files have charge and multiplicity on line 1
    n_atoms = int(lines[0].strip().split()[0])
    # Line 2: comment (skip)
    # Lines 3..N+2: element x y z

    Z = []
    R = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        Z.append(_sym_to_z(sym))
        R.append([x, y, z])

    return np.array(Z, dtype=int), np.array(R, dtype=float)


def _find_subdir(data_dir: str, name: str) -> str:
    """Find a dataset subdirectory, with flexibility for different layouts."""
    # Direct path
    direct = os.path.join(data_dir, name)
    if os.path.isdir(direct):
        return direct

    # Maybe data_dir IS the dataset dir
    if os.path.basename(data_dir) == name:
        return data_dir

    # Search one level up
    parent = os.path.dirname(data_dir)
    alt = os.path.join(parent, name)
    if os.path.isdir(alt):
        return alt

    # Fuzzy match (e.g. "qm7x" matches "qm7x_data")
    if os.path.isdir(data_dir):
        for d in os.listdir(data_dir):
            if name in d.lower() and os.path.isdir(os.path.join(data_dir, d)):
                return os.path.join(data_dir, d)

    raise FileNotFoundError(
        f"Dataset directory '{name}' not found in {data_dir}. "
        f"Contents: {os.listdir(data_dir) if os.path.isdir(data_dir) else 'N/A'}"
    )


# ══════════════════════════════════════════════════════════════════════════
# CLI / Quick test
# ══════════════════════════════════════════════════════════════════════════
def _print_summary(mols: List[MoleculeData], name: str):
    """Print a quick summary of loaded molecules."""
    if not mols:
        print(f"  {name}: empty")
        return

    n = len(mols)
    has_coords = sum(1 for m in mols if m.coords is not None and np.any(m.coords != 0))
    has_energy = sum(1 for m in mols if m.energy is not None)
    has_forces = sum(1 for m in mols if m.forces is not None)

    # Collect unique elements
    all_elements = set()
    for m in mols:
        flat = np.asarray(m.atomic_numbers).ravel()
        for z in flat:
            all_elements.add(int(z))

    elem_syms = []
    for z in sorted(all_elements):
        for sym, num in _ELEMENT_MAP.items():
            if num == z:
                elem_syms.append(sym)
                break
        else:
            elem_syms.append(f"Z={z}")

    size_range = f"{min(m.n_atoms for m in mols)}-{max(m.n_atoms for m in mols)}"

    print(f"  {name}:")
    print(f"    Molecules: {n}")
    print(f"    Atoms/mol: {size_range}")
    print(f"    Elements:  {', '.join(elem_syms)}")
    print(f"    Has coords: {has_coords}/{n}")
    print(f"    Has energy: {has_energy}/{n}")
    print(f"    Has forces: {has_forces}/{n}")

    if has_energy:
        energies = [m.energy for m in mols if m.energy is not None]
        print(f"    Energy range: [{min(energies):.2f}, {max(energies):.2f}] eV")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./deephf_datasets"
    print(f"DeePHF Unified Data Loader — scanning {data_dir}\n")

    results = load_all(data_dir, max_per_dataset=50)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, mols in results.items():
        _print_summary(mols, name)
    print()