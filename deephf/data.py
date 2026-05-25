"""Read prepared HuggingFace datasets into padded numpy arrays."""

from pathlib import Path

import numpy as np
from datasets import load_from_disk

from .constants import CHEM_DIM, ELEC_DIM, DATA_DIR
from .padding import pad_to
from .xyz import read_xyz


# Possible shapes the energy can take after feature extraction
ENERGY_FIELDS = ("E_corr_eV", "E_corr_kcal", "E_corr_Ha", "E_corr",
                 "e_corr", "correlation_energy", "energy", "target", "y")

# Standard conversions to convert to eV
TO_EV = {
    "E_corr_eV": 1.0, "E_corr_kcal": 1.0 / 23.0609, "E_corr_Ha": 27.2114,
    "E_corr": 1.0, "e_corr": 1.0, "correlation_energy": 1.0,
    "energy": 1.0, "target": 1.0, "y": 1.0,
}


def energy_field(ds):
    cols = set(ds.column_names)
    for name in ENERGY_FIELDS:
        if name in cols:
            return name, TO_EV[name]
    raise KeyError(f"No energy field. Available: {sorted(cols)}")


def hf_variant(descriptor):
    """Map descriptor name to the dataset variant we load.
    chemical_soap has no separate on-disk variant; we read chemical_elec_soap
    and skip the 108 electronic dims when slicing features.
    """
    if descriptor in ("chemical", "chemical_elec", "chemical_elec_soap"):
        return descriptor
    if descriptor == "chemical_soap":
        return "chemical_elec_soap"
    raise ValueError(f"Unknown descriptor: {descriptor}")


# get a specific set features needed
def select_features(nf, descriptor):
    if descriptor == "chemical_soap":
        return np.concatenate(
            [nf[:, :CHEM_DIM], nf[:, CHEM_DIM + ELEC_DIM:]], axis=-1)
    return nf


def load_split(dataset, descriptor, split, data_dir=DATA_DIR, quiet=False):
    """Return a dict with X, y, Z, coords, edge_index, edge_type, edge_mask."""


    # load the data from disk and extract energy fields (targets)
    path = Path(data_dir) / f"{dataset}_{hf_variant(descriptor)}" / split
    ds = load_from_disk(str(path))
    e_field, to_ev = energy_field(ds)


    # extract 
    sample = np.asarray(ds[0]["node_feature_matrix"], dtype=np.float32)
    desc_dim = select_features(sample, descriptor).shape[1]
    max_atoms = max(len(ex["node_feature_matrix"]) for ex in ds)
    max_edges_idx = max(len(ex["indices_feature_matrix"]) for ex in ds)
    max_edges_ef = max(len(ex["edge_feature_matrix"]) for ex in ds)
    max_edges = 2 * max(max_edges_idx, max_edges_ef)
    sample_ef = np.asarray(ds[0]["edge_feature_matrix"])
    edge_feat_dim = sample_ef.shape[1] if len(sample_ef) else 5

    if not quiet:
        print(f"  {dataset}/{descriptor}/{split}: n={len(ds)}, "
              f"max_atoms={max_atoms}, max_edges={max_edges}, "
              f"desc_dim={desc_dim}")

    n = len(ds)
    X = np.zeros((n, max_atoms, desc_dim), np.float32)
    y = np.zeros(n, np.float32)
    Z = np.zeros((n, max_atoms), np.int32)
    coords = np.zeros((n, max_atoms, 3), np.float32)
    edge_index = np.zeros((n, 2, max_edges), np.int64)
    edge_type = np.zeros((n, max_edges, edge_feat_dim), np.float32)
    edge_mask = np.zeros((n, max_edges), bool)

    for i, ex in enumerate(ds):
        nf = np.asarray(ex["node_feature_matrix"], np.float32)
        feat = select_features(nf, descriptor)
        X[i, :len(feat)] = feat
        y[i] = float(ex[e_field]) * to_ev

        xyz_path = ex.get("xyz_path")
        if xyz_path and Path(xyz_path).exists():
            Zs, R = read_xyz(xyz_path)
            Z[i, :len(Zs)] = Zs
            coords[i, :len(R)] = R
        else:
            Z[i, :len(nf)] = (np.abs(nf).sum(axis=1) > 0).astype(np.int32)

        _fill_edges(ex, edge_index[i], edge_type[i], edge_mask[i], max_edges)

    return dict(X=X, y=y, Z=Z, coords=coords,
                edge_index=edge_index, edge_type=edge_type, edge_mask=edge_mask,
                desc_dim=desc_dim, edge_feat_dim=edge_feat_dim)


def _fill_edges(ex, edge_index, edge_type, edge_mask, max_edges):
    """Deduplicate undirected pairs and emit both directions for the GAT."""
    idx = np.asarray(ex["indices_feature_matrix"], np.int64)
    ef = np.asarray(ex["edge_feature_matrix"], np.float32)
    if len(idx) == 0:
        return
    seen = {}
    for k in range(min(len(idx), len(ef))):
        a, b = int(idx[k, 0]), int(idx[k, 1])
        if a == b:
            continue
        seen.setdefault((min(a, b), max(a, b)), ef[k])
    e = 0
    for (a, b), feat in seen.items():
        if e >= max_edges:
            return
        edge_index[0, e] = a; edge_index[1, e] = b
        edge_type[e] = feat;  edge_mask[e] = True
        e += 1
        if e >= max_edges:
            return
        edge_index[0, e] = b; edge_index[1, e] = a
        edge_type[e] = feat;  edge_mask[e] = True
        e += 1


def load_dataset(dataset, descriptor, data_dir=DATA_DIR, quiet=False):
    """Load all three splits and pad them to common shapes."""
    train = load_split(dataset, descriptor, "train", data_dir, quiet)
    val   = load_split(dataset, descriptor, "val",   data_dir, quiet=True)
    test  = load_split(dataset, descriptor, "test",  data_dir, quiet=True)
    max_atoms = max(d["X"].shape[1] for d in (train, val, test))
    max_edges = max(d["edge_index"].shape[-1] for d in (train, val, test))
    return (pad_to(train, max_atoms, max_edges),
            pad_to(val,   max_atoms, max_edges),
            pad_to(test,  max_atoms, max_edges))
