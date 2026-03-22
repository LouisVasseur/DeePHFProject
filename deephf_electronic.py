"""
DeePHF Electronic Descriptor Computation
==========================================
Computes the core DeePHF descriptors: projected density matrix eigenvalues.

Pipeline per molecule:
  1. Build PySCF Mole object from (Z, coords)
  2. Run restricted HF (RHF)
  3. Get density matrix DM
  4. Build atom-centered projection basis (difference-of-Gaussians, l_max=2)
  5. For each atom I: PDM^I = S_I^T @ DM @ S_I
  6. Split PDM^I into (n, l) blocks, eigendecompose each
  7. Concatenate eigenvalues → 108-dim descriptor per atom

Usage:
    from deephf_electronic import ElectronicDescriptor, compute_electronic_for_dataset

    # Single molecule
    edesc = ElectronicDescriptor()
    desc, E_HF, E_corr = edesc.compute(atomic_numbers, coords)
    # desc: (N_atoms, 108), E_HF: float (Ha), E_corr: float (Ha, MP2)

    # Batch with caching
    descs = compute_electronic_for_dataset(molecules, cache_dir="./cache")

Requirements:
    pip install pyscf
"""

from __future__ import annotations

import os
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
# Projection basis: 12 radial × (s, p, d) = 12 × 9 = 108 eigenvalues per atom
N_RADIAL = 12
L_MAX = 2
DESC_DIM = N_RADIAL * sum(2 * l + 1 for l in range(L_MAX + 1))  # 12 * 9 = 108

# Gaussian exponents (geometric series from Chen et al. 2020, Appendix A)
PROJ_EXPONENTS = [
    9.8526125336e+02, 1.9461950684e+02, 5.7665039062e+01,
    1.7085937500e+01, 7.5937500000e+00, 3.3750000000e+00,
    2.2500000000e+00, 1.5000000000e+00, 1.0000000000e+00,
    6.6666666667e-01, 4.4444444444e-01, 2.9629629630e-01,
]

# Element symbol lookup
_Z_TO_SYM = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 35: "Br", 53: "I",
}


# ══════════════════════════════════════════════════════════════════════════
# Core descriptor computation
# ══════════════════════════════════════════════════════════════════════════
class ElectronicDescriptor:
    """
    Compute DeePHF electronic descriptors via PySCF.

    Descriptors are projected density matrix eigenvalues:
    108 dimensions per atom (12 radial shells × 9 angular functions).
    """

    def __init__(
        self,
        hf_basis: str = "cc-pvdz",
        proj_exponents: Optional[List[float]] = None,
        l_max: int = L_MAX,
        compute_mp2: bool = True,
        conv_tol: float = 1e-10,
        max_cycle: int = 200,
    ):
        """
        Args:
            hf_basis: GTO basis set for HF calculation
            proj_exponents: Gaussian exponents for projection basis
            l_max: Maximum angular momentum (0=s, 1=p, 2=d)
            compute_mp2: Whether to compute MP2 correlation energy
            conv_tol: HF convergence tolerance
            max_cycle: Maximum HF SCF iterations
        """
        self.hf_basis = hf_basis
        self.proj_exponents = proj_exponents or PROJ_EXPONENTS
        self.l_max = l_max
        self.n_radial = len(self.proj_exponents)
        self.compute_mp2 = compute_mp2
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle

        self.n_features = self.n_radial * sum(2 * l + 1 for l in range(self.l_max + 1))

    def _build_proj_basis(self) -> list:
        """Build PySCF-format projection basis using difference-of-Gaussians."""
        basis_list = []
        for n_idx, exp in enumerate(self.proj_exponents):
            if n_idx < self.n_radial - 1:
                exp_next = self.proj_exponents[n_idx + 1]
                for l in range(self.l_max + 1):
                    basis_list.append([l, [exp, 1.0], [exp_next, -1.0]])
            else:
                # Last function: single Gaussian
                for l in range(self.l_max + 1):
                    basis_list.append([l, [exp, 1.0]])
        return basis_list

    def compute(
        self,
        atomic_numbers: np.ndarray,
        coords: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
        """
        Compute electronic descriptors for a single molecule.

        Args:
            atomic_numbers: (N,) array of atomic numbers
            coords: (N, 3) array of coordinates in Angstrom

        Returns:
            (descriptors, E_HF, E_corr) where:
              descriptors: (N, 108) eigenvalue array, or None if HF fails
              E_HF: Hartree-Fock energy in Hartree
              E_corr: MP2 correlation energy in Hartree (or None)
        """
        from pyscf import gto, scf, mp

        Z = np.asarray(atomic_numbers).ravel().astype(int)
        R = np.asarray(coords).reshape(-1, 3)

        # Build PySCF molecule
        atom_list = []
        for i in range(len(Z)):
            sym = _Z_TO_SYM.get(int(Z[i]))
            if sym is None:
                logger.warning(f"Unknown element Z={Z[i]}, skipping molecule")
                return None, None, None
            atom_list.append(f"{sym} {R[i, 0]:.10f} {R[i, 1]:.10f} {R[i, 2]:.10f}")

        atom_str = "; ".join(atom_list)

        try:
            mol = gto.Mole(atom=atom_str, basis=self.hf_basis, verbose=0)
            mol.build()

            # Run RHF
            mf = scf.RHF(mol)
            mf.conv_tol = self.conv_tol
            mf.max_cycle = self.max_cycle
            E_HF = mf.kernel()

            if not mf.converged:
                logger.warning("HF did not converge")
                return None, None, None

            dm = mf.make_rdm1()

            # MP2 correlation energy
            E_corr = None
            if self.compute_mp2:
                try:
                    mp2 = mp.MP2(mf)
                    E_corr, _ = mp2.kernel()
                except Exception as e:
                    logger.debug(f"MP2 failed: {e}")

            # Build projection basis
            proj_basis = self._build_proj_basis()
            elements = set(mol.atom_symbol(i) for i in range(mol.natm))
            proj_dict = {e: proj_basis for e in elements}

            mol_aux = mol.copy()
            mol_aux.basis = proj_dict
            mol_aux.verbose = 0
            mol_aux.build()

            # Cross-overlap: <mol_AO | proj_AO>
            S_cross = gto.intor_cross("int1e_ovlp", mol, mol_aux)

            # Compute eigenvalues per atom per shell
            all_eigs = []
            for ia in range(mol.natm):
                p0, p1 = mol_aux.aoslice_by_atom()[ia, 2:4]
                S_I = S_cross[:, p0:p1]  # (nao_mol, nproj_atom)
                PDM = S_I.T @ dm @ S_I   # projected density matrix

                atom_eigs = []
                col = 0
                for l in range(self.l_max + 1):
                    for n_idx in range(self.n_radial):
                        n_m = 2 * l + 1  # number of m values
                        block = PDM[col:col + n_m, col:col + n_m]
                        eigs = np.sort(np.linalg.eigvalsh(block))
                        atom_eigs.extend(eigs.tolist())
                        col += n_m

                all_eigs.append(atom_eigs)

            descriptors = np.array(all_eigs, dtype=np.float64)  # (natm, 108)
            return descriptors, E_HF, E_corr

        except Exception as e:
            logger.warning(f"Descriptor computation failed: {e}")
            return None, None, None


# ══════════════════════════════════════════════════════════════════════════
# Batch computation with caching
# ══════════════════════════════════════════════════════════════════════════
def _mol_hash(Z: np.ndarray, R: np.ndarray) -> str:
    """Deterministic hash for a molecule geometry."""
    data = np.concatenate([Z.ravel().astype(np.float64), R.ravel()])
    return hashlib.md5(data.tobytes()).hexdigest()[:12]


def _compute_single(args):
    """Worker function for parallel computation."""
    idx, Z, R, hf_basis, compute_mp2 = args
    edesc = ElectronicDescriptor(hf_basis=hf_basis, compute_mp2=compute_mp2)
    desc, E_HF, E_corr = edesc.compute(Z, R)
    return idx, desc, E_HF, E_corr


def compute_electronic_for_dataset(
    molecules: list,
    hf_basis: str = "cc-pvdz",
    compute_mp2: bool = True,
    cache_dir: Optional[str] = None,
    n_jobs: int = 1,
    max_samples: Optional[int] = None,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[float]], List[Optional[float]]]:
    """
    Compute electronic descriptors for a list of MoleculeData objects.

    Args:
        molecules: From deephf_dataloader.load_dataset()
        hf_basis: Basis set for HF
        compute_mp2: Whether to compute MP2 correlation energies
        cache_dir: Directory to cache results (None = no caching)
        n_jobs: Number of parallel workers (1 = serial)
        max_samples: Limit number of molecules

    Returns:
        (descriptors, hf_energies, corr_energies) where each is a list
        of length len(molecules), with None for failed computations.
    """
    if max_samples:
        molecules = molecules[:max_samples]

    n = len(molecules)
    descriptors = [None] * n
    hf_energies = [None] * n
    corr_energies = [None] * n

    # Setup cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Identify which molecules need computation
    to_compute = []
    for i, mol in enumerate(molecules):
        if mol.coords is None or not np.any(mol.coords != 0):
            continue
        if len(mol.atomic_numbers) != mol.coords.shape[0]:
            continue

        # Check cache
        if cache_dir:
            h = _mol_hash(mol.atomic_numbers, mol.coords)
            cache_file = os.path.join(cache_dir, f"{h}.npz")
            if os.path.exists(cache_file):
                cached = np.load(cache_file, allow_pickle=True)
                descriptors[i] = cached["desc"] if cached["desc"].shape != () else None
                hf_energies[i] = float(cached["E_HF"]) if cached["E_HF"] != -1 else None
                corr_energies[i] = float(cached["E_corr"]) if cached["E_corr"] != -1 else None
                continue

        to_compute.append(i)

    n_cached = n - len(to_compute)
    if n_cached > 0:
        logger.info(f"  {n_cached} molecules loaded from cache")

    if not to_compute:
        logger.info(f"All {n} molecules cached, nothing to compute")
        return descriptors, hf_energies, corr_energies

    logger.info(f"Computing electronic descriptors for {len(to_compute)} molecules "
                f"(basis={hf_basis}, mp2={compute_mp2})...")

    t0 = time.time()

    if n_jobs == 1:
        # Serial computation
        edesc = ElectronicDescriptor(hf_basis=hf_basis, compute_mp2=compute_mp2)
        for count, i in enumerate(to_compute):
            mol = molecules[i]
            desc, E_HF, E_corr = edesc.compute(mol.atomic_numbers, mol.coords)
            descriptors[i] = desc
            hf_energies[i] = E_HF
            corr_energies[i] = E_corr

            # Cache
            if cache_dir and desc is not None:
                h = _mol_hash(mol.atomic_numbers, mol.coords)
                np.savez(
                    os.path.join(cache_dir, f"{h}.npz"),
                    desc=desc,
                    E_HF=E_HF if E_HF is not None else -1,
                    E_corr=E_corr if E_corr is not None else -1,
                )

            if (count + 1) % 10 == 0 or count == 0:
                elapsed = time.time() - t0
                rate = (count + 1) / elapsed
                eta = (len(to_compute) - count - 1) / rate if rate > 0 else 0
                logger.info(
                    f"  [{count+1}/{len(to_compute)}] "
                    f"{rate:.1f} mol/s, ETA {eta:.0f}s"
                )
    else:
        # Parallel computation
        args_list = [
            (i, molecules[i].atomic_numbers, molecules[i].coords, hf_basis, compute_mp2)
            for i in to_compute
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(_compute_single, args): args[0] for args in args_list}
            done = 0
            for future in as_completed(futures):
                idx, desc, E_HF, E_corr = future.result()
                descriptors[idx] = desc
                hf_energies[idx] = E_HF
                corr_energies[idx] = E_corr

                if cache_dir and desc is not None:
                    mol = molecules[idx]
                    h = _mol_hash(mol.atomic_numbers, mol.coords)
                    np.savez(
                        os.path.join(cache_dir, f"{h}.npz"),
                        desc=desc,
                        E_HF=E_HF if E_HF is not None else -1,
                        E_corr=E_corr if E_corr is not None else -1,
                    )

                done += 1
                if done % 10 == 0:
                    logger.info(f"  [{done}/{len(to_compute)}] completed")

    elapsed = time.time() - t0
    n_success = sum(1 for d in descriptors if d is not None)
    logger.info(
        f"Electronic descriptors done: {n_success}/{n} succeeded "
        f"in {elapsed:.1f}s ({n_success/max(elapsed,1):.1f} mol/s)"
    )

    return descriptors, hf_energies, corr_energies


# ══════════════════════════════════════════════════════════════════════════
# CLI / Quick test
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("DeePHF Electronic Descriptor — Test")
    print("=" * 60)

    edesc = ElectronicDescriptor(hf_basis="cc-pvdz")

    # Test on equilibrium water
    print("\n--- Equilibrium water ---")
    Z = np.array([8, 1, 1])
    R = np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]])

    desc, E_HF, E_corr = edesc.compute(Z, R)
    if desc is not None:
        print(f"  Descriptor shape: {desc.shape} (n_atoms × {edesc.n_features})")
        print(f"  E_HF  = {E_HF:.8f} Ha")
        print(f"  E_corr(MP2) = {E_corr:.8f} Ha")
        print(f"  O  descriptor: min={desc[0].min():.6f}, max={desc[0].max():.6f}")
        print(f"  H1 descriptor: min={desc[1].min():.6f}, max={desc[1].max():.6f}")
        print(f"  H2 descriptor: min={desc[2].min():.6f}, max={desc[2].max():.6f}")
    else:
        print("  FAILED — PySCF not installed? pip install pyscf")

    # Test on dataset if available
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./deephf_datasets"
    print(f"\n--- Batch test (rMD17 ethanol, 5 structures) ---")
    try:
        sys.path.insert(0, ".")
        from deephf_dataloader import load_dataset

        mols = load_dataset("rmd17", data_dir=data_dir, molecule="ethanol",
                            stride=10000, max_samples=5)
        descs, hf_es, corr_es = compute_electronic_for_dataset(
            mols, cache_dir="./cache_electronic", n_jobs=1,
        )
        n_ok = sum(1 for d in descs if d is not None)
        print(f"  Success: {n_ok}/{len(mols)}")
        if n_ok > 0:
            first = next(d for d in descs if d is not None)
            print(f"  Shape: {first.shape}")
            print(f"  Range: [{first.min():.4f}, {first.max():.4f}]")
    except ImportError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 60)
