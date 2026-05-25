#AI Use: code architected by hand and implemented by hand, optimized and fixed by Claude

"""Pad a loaded split so that all splits in a cell share max_atoms and max_edges."""

import numpy as np


def pad_to(d, max_atoms, max_edges):
    n, A, D = d["X"].shape
    E = d["edge_index"].shape[-1]
    if A < max_atoms:
        a = max_atoms - A
        d["X"] = np.concatenate([d["X"], np.zeros((n, a, D), np.float32)], axis=1)
        d["Z"] = np.concatenate([d["Z"], np.zeros((n, a), np.int32)], axis=1)
        d["coords"] = np.concatenate(
            [d["coords"], np.zeros((n, a, 3), np.float32)], axis=1)
    if E < max_edges:
        e = max_edges - E
        ef_d = d["edge_type"].shape[-1]
        d["edge_index"] = np.concatenate(
            [d["edge_index"], np.zeros((n, 2, e), np.int64)], axis=-1)
        d["edge_type"] = np.concatenate(
            [d["edge_type"], np.zeros((n, e, ef_d), np.float32)], axis=1)
        d["edge_mask"] = np.concatenate(
            [d["edge_mask"], np.zeros((n, e), bool)], axis=1)
    return d
