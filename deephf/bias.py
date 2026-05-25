# AI use: code first made by hand and optimized using Claude (for example, Claude added the use of np.linalg.lstsq)
# Verified against standard implementations


"""Fit a per-element energy bias by ordinary least squares on the train split.

E_corr_i ≈ sum_a B[Z_ia] for each molecule i. The fitted B is frozen and
used as a non-trainable embedding inside every architecture.
"""


import numpy as np
import torch

def fit_element_bias(Z, y, mask, max_z):
    """Return (bias_per_z: torch.Tensor of shape (max_z,), r2: float)."""
    Z_np = Z.cpu().numpy()
    m_np = mask.cpu().numpy()
    y_np = y.cpu().numpy()
    comp = np.zeros((len(Z_np), max_z), np.float64)
    for i in range(len(Z_np)):
        for z in Z_np[i][m_np[i]]:
            if 0 <= z < max_z:
                comp[i, z] += 1
    active = np.where(comp.sum(0) > 0)[0]
    coef, *_ = np.linalg.lstsq(comp[:, active], y_np, rcond=None)
    pred = comp[:, active] @ coef
    r2 = 1.0 - ((y_np - pred) ** 2).sum() / ((y_np - y_np.mean()) ** 2).sum()
    bias = np.zeros(max_z, np.float32)
    bias[active] = coef.astype(np.float32)
    return torch.tensor(bias), float(r2)
