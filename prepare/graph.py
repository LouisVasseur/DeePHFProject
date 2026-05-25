"""Build the 17-dim per-atom + 5-dim per-bond features from an xyz file.

Node features (17 dims):
    0-5    : one-hot of element (C, Cl, N, O, S, H)
    6      : atomic number
    7      : aromaticity flag
    8-14   : one-hot of hybridisation (SP3, SP2, SP, SP3D2, SP3D, S, UNKNOWN)
    15     : implicit hydrogen count
    16     : Pauling electronegativity

Edge features (5 dims):
    0-3    : one-hot of bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC)
    4      : bond length (Å)
"""

import numpy as np
from rdkit import Chem
from openbabel import pybel

_ELEMENT_OH = {"C": 0, "Cl": 1, "N": 2, "O": 3, "S": 4, "H": 5}
_HYB_OH = {"SP3": 0, "SP2": 1, "SP": 2, "SP3D2": 3, "SP3D": 4, "S": 5, "UNKNOWN": 6}
_BOND_OH = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
_EN = {6: 2.55, 17: 3.16, 7: 3.04, 8: 3.44, 16: 2.58, 1: 2.2}


def _node_features(mol):
    out = np.zeros((mol.GetNumAtoms(), 17), np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        out[i, _ELEMENT_OH[sym]] = 1.0
        out[i, 6] = atom.GetAtomicNum()
        out[i, 7] = float(atom.GetIsAromatic())
        out[i, 8 + _HYB_OH[str(atom.GetHybridization())]] = 1.0
        out[i, 15] = atom.GetValence(Chem.ValenceType.IMPLICIT)
        out[i, 16] = _EN[atom.GetAtomicNum()]
    return out


def _edge_features(mol):
    conf = mol.GetConformer()
    feats, idx = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = np.zeros(5, np.float32)
        f[_BOND_OH[str(bond.GetBondType())]] = 1.0
        pa, pb = np.array(conf.GetAtomPosition(a)), np.array(conf.GetAtomPosition(b))
        f[4] = np.linalg.norm(pa - pb)
        feats.append(f)
        idx.append([a, b])
    if not feats:
        return np.zeros((0, 5), np.float32), np.zeros((0, 2), np.int64)
    e = np.array(feats, np.float32)
    i = np.array(idx, np.int64)
    return np.concatenate([e, e], 0), np.concatenate([i, i[:, ::-1]], 0)


def build_graph(xyz_path):
    """Return (node_features, edge_features, edge_indices) or three Nones."""
    mol_ob = next(pybel.readfile("xyz", xyz_path))
    sdf = mol_ob.write("sdf")
    mol = Chem.MolFromMolBlock(sdf, removeHs=False)
    if mol is None:
        return None, None, None
    return _node_features(mol), *_edge_features(mol)
