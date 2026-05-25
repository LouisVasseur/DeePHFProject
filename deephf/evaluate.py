"""Evaluate a trained model on a held-out loader."""

import numpy as np
import torch
from torch.cuda.amp import autocast

from .constants import EV_TO_MHA, EV_TO_KCAL


def evaluate(model, loader, device, use_amp=True):
    amp = use_amp and torch.cuda.is_available() and str(device).startswith("cuda")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            b = [t.to(device, non_blocking=True) for t in batch]
            X, y, Z, mask, coords, ei, et, em = b
            with autocast(enabled=amp):
                preds.append(model(X, Z, mask, coords, ei, et, em).float().cpu())
            trues.append(y.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(trues).numpy()
    mae_eV = float(np.mean(np.abs(y_pred - y_true)))
    return {"test_mae_mHa": mae_eV * EV_TO_MHA,
            "test_mae_kcal": mae_eV * EV_TO_KCAL,
            "y_pred": y_pred, "y_true": y_true}
