"""
DeePHF Descriptor Module
=========================
Computes and combines descriptors for the DeePHF atomic descriptor ablation study.

Two descriptor types:
  1. Electronic descriptors (DeePHF): projected density matrix eigenvalues (108-dim per atom)
     - Computed by your existing PySCF pipeline
  2. Atomic descriptors (SOAP): Smooth Overlap of Atomic Positions via DScribe
     - Computed from xyz coordinates + atomic numbers only

Three ablation configurations:
  (a) electronic_only  — 108-dim per atom (baseline DeePHF)
  (b) atomic_only      — SOAP features per atom
  (c) combined         — concatenation with regularization weight w_atomic

Usage:
    from deephf_descriptors import SOAPDescriptor, CombinedDescriptor, DescriptorConfig

    # Compute SOAP for a molecule
    soap = SOAPDescriptor()
    soap_features = soap.compute(atomic_numbers, coords)  # (N_atoms, n_soap_features)

    # Combined descriptor pipeline
    config = DescriptorConfig.combined(w_atomic=0.1)
    desc = CombinedDescriptor(config)
    features = desc.compute(
        atomic_numbers=Z,
        coords=R,
        electronic_desc=e_desc,  # (N_atoms, 108) from your PySCF pipeline
    )  # (N_atoms, 108 + n_soap)

Requirements:
    pip install dscribe ase
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════
class DescriptorMode(Enum):
    ELECTRONIC_ONLY = "electronic_only"
    ATOMIC_ONLY = "atomic_only"
    COMBINED = "combined"


@dataclass
class SOAPConfig:
    """Configuration for SOAP descriptors."""
    # SOAP hyperparameters
    r_cut: float = 6.0          # Cutoff radius in Angstrom
    n_max: int = 8              # Number of radial basis functions
    l_max: int = 6              # Maximum angular momentum
    sigma: float = 0.3          # Gaussian smearing width
    # Species to consider (auto-detected if None)
    species: Optional[List[int]] = None
    # Whether to use crossover terms between species
    crossover: bool = True
    # Compression: average over sites (molecule-level) or keep per-atom
    average: str = "off"        # "off" = per-atom, "inner" or "outer" = averaged
    # Sparse: use only a subset of features
    sparse: bool = False
    # Normalization
    normalize: bool = True

    @property
    def n_features(self) -> int:
        """Estimate number of SOAP features (exact count depends on species)."""
        n_species = len(self.species) if self.species else 5  # estimate
        if self.crossover:
            n_pairs = n_species * (n_species + 1) // 2
        else:
            n_pairs = n_species
        return n_pairs * self.n_max * (self.l_max + 1)


@dataclass
class DescriptorConfig:
    """Full descriptor configuration for the ablation study."""
    mode: DescriptorMode = DescriptorMode.COMBINED
    soap: SOAPConfig = field(default_factory=SOAPConfig)
    # Weight for atomic descriptors in combined mode
    w_atomic: float = 0.1
    # Electronic descriptor dimension (from your DeePHF pipeline)
    electronic_dim: int = 108

    @classmethod
    def electronic_only(cls) -> "DescriptorConfig":
        return cls(mode=DescriptorMode.ELECTRONIC_ONLY)

    @classmethod
    def atomic_only(cls, **soap_kwargs) -> "DescriptorConfig":
        return cls(mode=DescriptorMode.ATOMIC_ONLY, soap=SOAPConfig(**soap_kwargs))

    @classmethod
    def combined(cls, w_atomic: float = 0.1, **soap_kwargs) -> "DescriptorConfig":
        return cls(
            mode=DescriptorMode.COMBINED,
            w_atomic=w_atomic,
            soap=SOAPConfig(**soap_kwargs),
        )


# ══════════════════════════════════════════════════════════════════════════
# SOAP Descriptor
# ══════════════════════════════════════════════════════════════════════════
class SOAPDescriptor:
    """
    Compute SOAP atomic descriptors using DScribe.

    SOAP encodes the local chemical environment around each atom as a
    rotationally invariant power spectrum of the neighbor density.
    """

    def __init__(self, config: Optional[SOAPConfig] = None):
        self.config = config or SOAPConfig()
        self._soap = None  # Lazy init (depends on species)
        self._fitted_species = None

    def _init_soap(self, species: List[int]):
        """Initialize DScribe SOAP calculator for given species."""
        from dscribe.descriptors import SOAP

        # DScribe 2.x API: crossover is default (no separate param)
        # compression={'mode': 'off'} = full crossover (default)
        # compression={'mode': 'crossover'} = no cross-species terms
        compression = {"mode": "off"} if self.config.crossover else {"mode": "crossover"}

        self._soap = SOAP(
            species=sorted(species),
            r_cut=self.config.r_cut,
            n_max=self.config.n_max,
            l_max=self.config.l_max,
            sigma=self.config.sigma,
            compression=compression,
            average=self.config.average,
            sparse=self.config.sparse,
            periodic=False,
        )
        self._fitted_species = sorted(species)
        logger.debug(
            f"SOAP initialized: species={self._fitted_species}, "
            f"n_features={self._soap.get_number_of_features()}"
        )

    def compute(
        self,
        atomic_numbers: np.ndarray,
        coords: np.ndarray,
        species: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute SOAP descriptors for a single molecule.

        Args:
            atomic_numbers: (N,) array of atomic numbers
            coords: (N, 3) array of coordinates in Angstrom
            species: List of all possible species (for consistent feature dim).
                     If None, auto-detected from this molecule.

        Returns:
            (N, n_features) SOAP descriptor matrix
        """
        from ase import Atoms

        Z = np.asarray(atomic_numbers).ravel().astype(int)
        R = np.asarray(coords).reshape(-1, 3)

        # Determine species
        if species is not None:
            sp = sorted(species)
        elif self.config.species is not None:
            sp = sorted(self.config.species)
        else:
            sp = sorted(set(Z.tolist()))

        # Re-init if species changed
        if self._fitted_species != sp:
            self._init_soap(sp)

        # Create ASE Atoms object
        atoms = Atoms(numbers=Z, positions=R)

        # Compute SOAP
        desc = self._soap.create(atoms)  # (N_atoms, n_features) or (1, n_features) if averaged

        if self.config.normalize and desc.shape[0] > 0:
            norms = np.linalg.norm(desc, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            desc = desc / norms

        return np.asarray(desc)

    def compute_batch(
        self,
        molecules: list,
        species: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """
        Compute SOAP for a batch of molecules.

        Args:
            molecules: List of MoleculeData objects (from deephf_dataloader)
            species: Global species list for consistent feature dimensions.
                     If None, auto-detected from all molecules in the batch.

        Returns:
            List of (N_atoms_i, n_features) arrays
        """
        from ase import Atoms

        # Auto-detect species from batch
        if species is None and self.config.species is None:
            all_species = set()
            for mol in molecules:
                all_species.update(np.asarray(mol.atomic_numbers).ravel().astype(int).tolist())
            species = sorted(all_species)

        sp = species if species is not None else sorted(self.config.species)

        # Init SOAP if needed
        if self._fitted_species != sp:
            self._init_soap(sp)

        # Build ASE Atoms list
        atoms_list = []
        for mol in molecules:
            Z = np.asarray(mol.atomic_numbers).ravel().astype(int)
            R = np.asarray(mol.coords).reshape(-1, 3)
            atoms_list.append(Atoms(numbers=Z, positions=R))

        # Batch compute
        all_desc = self._soap.create(atoms_list, n_jobs=1)

        # DScribe returns a single 2D array for batch with same-size molecules,
        # or a list of arrays for variable-size molecules
        if isinstance(all_desc, np.ndarray) and all_desc.ndim == 2:
            # All molecules have same number of atoms
            n_atoms = len(molecules[0].atomic_numbers)
            results = []
            for i in range(len(molecules)):
                chunk = all_desc[i * n_atoms:(i + 1) * n_atoms]
                if self.config.normalize:
                    norms = np.linalg.norm(chunk, axis=1, keepdims=True)
                    norms = np.where(norms > 1e-10, norms, 1.0)
                    chunk = chunk / norms
                results.append(chunk)
            return results
        elif isinstance(all_desc, list):
            results = []
            for desc in all_desc:
                desc = np.asarray(desc)
                if self.config.normalize:
                    norms = np.linalg.norm(desc, axis=1, keepdims=True)
                    norms = np.where(norms > 1e-10, norms, 1.0)
                    desc = desc / norms
                results.append(desc)
            return results
        else:
            # Fallback: compute one by one
            return [self.compute(mol.atomic_numbers, mol.coords, species=sp)
                    for mol in molecules]

    @property
    def n_features(self) -> int:
        """Number of SOAP features per atom."""
        if self._soap is not None:
            return self._soap.get_number_of_features()
        return self.config.n_features


# ══════════════════════════════════════════════════════════════════════════
# Combined Descriptor
# ══════════════════════════════════════════════════════════════════════════
class CombinedDescriptor:
    """
    Combines electronic (DeePHF) and atomic (SOAP) descriptors
    for the ablation study.

    Three modes:
      electronic_only: returns electronic_desc as-is
      atomic_only: returns SOAP features
      combined: returns [electronic_desc, w_atomic * SOAP]
    """

    def __init__(self, config: Optional[DescriptorConfig] = None):
        self.config = config or DescriptorConfig()
        self.soap = SOAPDescriptor(self.config.soap)

    def compute(
        self,
        atomic_numbers: np.ndarray,
        coords: np.ndarray,
        electronic_desc: Optional[np.ndarray] = None,
        species: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute descriptors for a single molecule.

        Args:
            atomic_numbers: (N,) atomic numbers
            coords: (N, 3) coordinates in Angstrom
            electronic_desc: (N, 108) DeePHF electronic descriptors (required
                             for electronic_only and combined modes)
            species: Species list for SOAP

        Returns:
            (N, D) descriptor matrix where D depends on mode:
              electronic_only: D = 108
              atomic_only: D = n_soap_features
              combined: D = 108 + n_soap_features
        """
        mode = self.config.mode

        if mode == DescriptorMode.ELECTRONIC_ONLY:
            if electronic_desc is None:
                raise ValueError("electronic_desc required for electronic_only mode")
            return electronic_desc

        # Compute SOAP
        soap_desc = self.soap.compute(atomic_numbers, coords, species=species)

        if mode == DescriptorMode.ATOMIC_ONLY:
            return soap_desc

        # Combined mode
        if electronic_desc is None:
            raise ValueError("electronic_desc required for combined mode")

        n_atoms = electronic_desc.shape[0]
        if soap_desc.shape[0] != n_atoms:
            raise ValueError(
                f"Shape mismatch: electronic has {n_atoms} atoms, "
                f"SOAP has {soap_desc.shape[0]} atoms"
            )

        # Scale atomic descriptors by w_atomic
        scaled_soap = self.config.w_atomic * soap_desc

        return np.concatenate([electronic_desc, scaled_soap], axis=1)

    def compute_batch(
        self,
        molecules: list,
        electronic_descs: Optional[List[np.ndarray]] = None,
        species: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """
        Compute descriptors for a batch of molecules.

        Args:
            molecules: List of MoleculeData objects
            electronic_descs: List of (N_i, 108) arrays (one per molecule)
            species: Global species list

        Returns:
            List of (N_i, D) descriptor arrays
        """
        mode = self.config.mode

        if mode == DescriptorMode.ELECTRONIC_ONLY:
            if electronic_descs is None:
                raise ValueError("electronic_descs required for electronic_only mode")
            return electronic_descs

        # Compute SOAP batch
        soap_descs = self.soap.compute_batch(molecules, species=species)

        if mode == DescriptorMode.ATOMIC_ONLY:
            return soap_descs

        # Combined
        if electronic_descs is None:
            raise ValueError("electronic_descs required for combined mode")

        results = []
        for e_desc, s_desc in zip(electronic_descs, soap_descs):
            scaled = self.config.w_atomic * s_desc
            results.append(np.concatenate([e_desc, scaled], axis=1))
        return results

    @property
    def n_features(self) -> int:
        """Total descriptor dimension per atom."""
        mode = self.config.mode
        if mode == DescriptorMode.ELECTRONIC_ONLY:
            return self.config.electronic_dim
        elif mode == DescriptorMode.ATOMIC_ONLY:
            return self.soap.n_features
        else:
            return self.config.electronic_dim + self.soap.n_features


# ══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ══════════════════════════════════════════════════════════════════════════
def compute_soap_for_dataset(
    molecules: list,
    soap_config: Optional[SOAPConfig] = None,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Compute SOAP descriptors for a list of MoleculeData objects.

    Args:
        molecules: From deephf_dataloader.load_dataset()
        soap_config: SOAP configuration (uses defaults if None)
        max_samples: Limit number of molecules

    Returns:
        (soap_descriptors, species) where:
          soap_descriptors: List of (N_atoms_i, n_features) arrays
          species: Sorted list of all atomic numbers in the dataset
    """
    if max_samples:
        molecules = molecules[:max_samples]

    # Filter to molecules with valid coords
    valid = [m for m in molecules if m.coords is not None and np.any(m.coords != 0)]
    if len(valid) < len(molecules):
        logger.warning(f"Skipping {len(molecules) - len(valid)} molecules without coordinates")

    soap = SOAPDescriptor(soap_config)

    # Auto-detect species
    all_species = set()
    for mol in valid:
        all_species.update(np.asarray(mol.atomic_numbers).ravel().astype(int).tolist())
    species = sorted(all_species)

    logger.info(f"Computing SOAP for {len(valid)} molecules, species={species}")
    descs = soap.compute_batch(valid, species=species)
    logger.info(f"SOAP done: {soap.n_features} features per atom")

    return descs, species


# ══════════════════════════════════════════════════════════════════════════
# CLI / Quick test
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Import the dataloader
    sys.path.insert(0, ".")
    from deephf_dataloader import load_dataset

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./deephf_datasets"

    print("=" * 60)
    print("DeePHF Descriptor Module — SOAP Integration Test")
    print("=" * 60)

    # Test on water (small, fast)
    print("\n--- Water (deepks-kit) ---")
    try:
        mols = load_dataset("water", data_dir=data_dir, max_samples=10)
        descs, species = compute_soap_for_dataset(mols)
        print(f"  Molecules: {len(descs)}")
        print(f"  Species: {species}")
        print(f"  SOAP features/atom: {descs[0].shape[1]}")
        print(f"  First molecule shape: {descs[0].shape}")
        print(f"  Feature range: [{descs[0].min():.4f}, {descs[0].max():.4f}]")
    except Exception as e:
        print(f"  Error: {e}")

    # Test on QM7-X (diverse chemistry)
    print("\n--- QM7-X (8000.hdf5) ---")
    try:
        mols = load_dataset("qm7x", data_dir=data_dir, files=["8000.hdf5"], max_samples=10)
        descs, species = compute_soap_for_dataset(mols)
        print(f"  Molecules: {len(descs)}")
        print(f"  Species: {species}")
        print(f"  SOAP features/atom: {descs[0].shape[1]}")
        print(f"  First molecule shape: {descs[0].shape}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test on rMD17
    print("\n--- rMD17 (ethanol) ---")
    try:
        mols = load_dataset("rmd17", data_dir=data_dir, molecule="ethanol",
                            stride=1000, max_samples=10)
        descs, species = compute_soap_for_dataset(mols)
        print(f"  Molecules: {len(descs)}")
        print(f"  Species: {species}")
        print(f"  SOAP features/atom: {descs[0].shape[1]}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test combined descriptor with fake electronic descriptors
    print("\n--- Combined descriptor (mock electronic + SOAP) ---")
    try:
        # Use rMD17 ethanol (clean, guaranteed Z/coords match)
        mols = load_dataset("rmd17", data_dir=data_dir, molecule="ethanol",
                            stride=10000, max_samples=1)
        if not mols:
            mols = load_dataset("qm7x", data_dir=data_dir, files=["8000.hdf5"],
                                max_samples=1)
        mol = mols[0]
        n_atoms = len(mol.atomic_numbers)
        print(f"  Test molecule: {mol.mol_id} ({n_atoms} atoms)")

        # Mock electronic descriptors (108-dim, normally from PySCF)
        fake_electronic = np.random.randn(n_atoms, 108)

        for mode_name, config in [
            ("electronic_only", DescriptorConfig.electronic_only()),
            ("atomic_only", DescriptorConfig.atomic_only()),
            ("combined w=0.1", DescriptorConfig.combined(w_atomic=0.1)),
            ("combined w=0.5", DescriptorConfig.combined(w_atomic=0.5)),
        ]:
            desc = CombinedDescriptor(config)
            features = desc.compute(
                mol.atomic_numbers, mol.coords,
                electronic_desc=fake_electronic,
            )
            print(f"  {mode_name:25s} -> ({n_atoms}, {features.shape[1]:4d}) features")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Done. Install DScribe if needed: pip install dscribe ase")
    print("=" * 60)