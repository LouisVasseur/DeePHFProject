"""
CCSD(T)/cc-pVTZ ground-truth lookup from training data.

Usage in app.py:
    from groundtruth import lookup_gt

    gt = lookup_gt(smiles="CCO")
    # or
    gt = lookup_gt(xyz=xyz_string)

    if gt is not None:
        st.metric(
            f"Ground truth (CCSD(T)/cc-pVTZ, from {', '.join(gt['datasets'])} training set)",
            f"{gt['mean_kcal']:.2f} kcal/mol",
            help=f"Mean over {gt['n']} conformers (σ={gt['std_kcal']:.3f} kcal/mol)",
        )
"""
import json
from pathlib import Path
from functools import lru_cache
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

_LOOKUP_FILE = Path(__file__).parent / "ground_truth.json"


@lru_cache(maxsize=1)
def _load_lookup() -> dict:
    if not _LOOKUP_FILE.exists():
        return {}
    return json.loads(_LOOKUP_FILE.read_text())


def _canonical_smiles_from_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _canonical_smiles_from_xyz(xyz_str: str) -> str | None:
    """Parse xyz string, infer bonds, return canonical SMILES (Hs stripped)."""
    try:
        mol = Chem.MolFromXYZBlock(xyz_str)
        if mol is None:
            return None
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
    except Exception:
        return None


def lookup_gt(smiles: str | None = None, xyz: str | None = None) -> dict | None:
    """
    Return CCSD(T)/cc-pVTZ ground-truth stats dict, or None if not in training set.

    Provide exactly one of `smiles` or `xyz`.
    Dict shape: {mean_kcal, std_kcal, min_kcal, max_kcal, n, datasets}
    """
    if smiles is not None:
        canonical = _canonical_smiles_from_smiles(smiles)
    elif xyz is not None:
        canonical = _canonical_smiles_from_xyz(xyz)
    else:
        return None

    if canonical is None:
        return None

    return _load_lookup().get(canonical)
