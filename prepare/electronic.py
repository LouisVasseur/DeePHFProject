# AI Use: code made directly from deepks (the repo linked in the DeePHF paper) and adapted by hand to our data format and needs, with optimizations and fixes by Claude. 
# The overall structure of the code was also architected by hand for modularity and clarity.

"""DeePHF electronic descriptors via PySCF HF + density-matrix projection.

108 per-atom features = 12 radial × {s, p, d}. Difference-of-Gaussian
projection basis exponents from Chen et al. 2020 (Appendix A).
"""

import hashlib
import sys
from pathlib import Path

import numpy as np


Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl"}

N_RADIAL = 12
L_MAX = 2  # → 12 × (1+3+5) = 108 features per atom
PROJ_EXPONENTS = [
    9.8526125336e+02, 1.9461950684e+02, 5.7665039062e+01,
    1.7085937500e+01, 7.5937500000e+00, 3.3750000000e+00,
    2.2500000000e+00, 1.5000000000e+00, 1.0000000000e+00,
    6.6666666667e-01, 4.4444444444e-01, 2.9629629630e-01,
]


def _proj_basis():
    out = []
    for i, e in enumerate(PROJ_EXPONENTS):
        if i < N_RADIAL - 1:
            e_next = PROJ_EXPONENTS[i + 1]
            for l in range(L_MAX + 1):
                out.append([l, [e, 1.0], [e_next, -1.0]])
        else:
            for l in range(L_MAX + 1):
                out.append([l, [e, 1.0]])
    return out


# given (Z, R), compute the electronic descriptors by running HF and projecting the density matrix onto the DoG basis. 
# Return (descriptors_per_atom (n_atoms, 108), E_HF, E_corr_MP2). 
# If HF fails to converge, return (None, None, None).
def compute_for_mol(Z, R, basis="cc-pvdz"):
    """Return (descriptors_per_atom (n_atoms, 108), E_HF, E_corr_MP2)."""
    from pyscf import gto, scf, mp

    atom_str = "; ".join(f"{Z_TO_SYMBOL[int(z)]} {x:.10f} {y:.10f} {zc:.10f}"
                         for z, (x, y, zc) in zip(Z, R))
    try:
        mol = gto.Mole(atom=atom_str, basis=basis, verbose=0); mol.build()
        mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.max_cycle = 200
        e_hf = mf.kernel()
        if not mf.converged:
            return None, None, None
        dm = mf.make_rdm1()
        e_corr = float(mp.MP2(mf).kernel()[0])

        elements = set(mol.atom_symbol(i) for i in range(mol.natm))
        mol_aux = mol.copy()
        mol_aux.basis = {e: _proj_basis() for e in elements}
        mol_aux.verbose = 0; mol_aux.build()
        S_cross = gto.intor_cross("int1e_ovlp", mol, mol_aux)

        eigs = []
        for ia in range(mol.natm):
            p0, p1 = mol_aux.aoslice_by_atom()[ia, 2:4]
            S_I = S_cross[:, p0:p1]
            PDM = S_I.T @ dm @ S_I
            atom_eigs, col = [], 0
            for _ in range(N_RADIAL):
                for l in range(L_MAX + 1):
                    n_m = 2 * l + 1
                    block = PDM[col:col + n_m, col:col + n_m]
                    atom_eigs.extend(np.sort(np.linalg.eigvalsh(block)).tolist())
                    col += n_m
            eigs.append(atom_eigs)
        return np.array(eigs, np.float64), float(e_hf), e_corr
    except Exception as e:
        print(f"  HF/MP2 failed: {e}", file=sys.stderr)
        return None, None, None


# Cache class to store computed electronic descriptors on disk, keyed by a hash of (Z, R). This avoids redundant HF calculations for identical molecules.
class Cache:
    """Disk cache keyed by SHA1 of (Z, coords)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, Z, R):
        return hashlib.sha1(
            np.concatenate([Z.astype(np.float64).ravel(), R.ravel()]).tobytes()
        ).hexdigest()

    def get_or_compute(self, Z, R):
        path = self.root / f"{self._key(Z, R)}.npz"
        if path.exists():
            d = np.load(path)
            return d["desc"], float(d["e_hf"]), float(d["e_corr"])
        desc, e_hf, e_corr = compute_for_mol(Z, R)
        if desc is not None:
            np.savez(path, desc=desc, e_hf=e_hf, e_corr=e_corr)
        return desc, e_hf, e_corr
