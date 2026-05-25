# AI Use: training loop code architected by hand and implemented by hand, 
# optimized and fixed by Claude for efficiency and stability 
# (for example, the use of autocast and GradScaler for mixed precision training, and the specific way of computing val MAE to avoid GPU-CPU transfer bottlenecks).

"""Training loop: AdamW + exponential LR decay + early stopping on val MAE."""

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .constants import (LR, WEIGHT_DECAY, LR_DECAY, LR_DECAY_EVERY,
                        ES_PATIENCE, MAX_EPOCHS, EV_TO_MHA)


# _to: helper to move a batch to device, with non_blocking=True for efficiency.
def _to(batch, device):
    return [b.to(device, non_blocking=True) for b in batch]

# train_model: main training loop with AdamW, exponential LR decay, and early stopping on val MAE. Returns best_val_mae_mHa, best_epoch, final_epoch, duration_s, history.
def train_model(model, train_loader, val_loader, device,
                max_epochs=MAX_EPOCHS, patience=ES_PATIENCE,
                use_amp=True, verbose=True):
    """Returns best_val_mae_mHa, best_epoch, final_epoch, duration_s, history."""
    amp = use_amp and torch.cuda.is_available() and str(device).startswith("cuda")
    scaler = GradScaler(enabled=amp)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ExponentialLR(
        opt, gamma=LR_DECAY ** (1.0 / LR_DECAY_EVERY))

    best, best_ep, hist, n_bad = float("inf"), -1, [], 0
    t0 = time.time()
    bar = tqdm(range(max_epochs), desc="train", ncols=100, leave=False,
               file=sys.stdout, mininterval=2.0, disable=not verbose)
    ep = -1
    for ep in bar:
        model.train()
        for batch in train_loader:
            X, y, Z, mask, coords, ei, et, em = _to(batch, device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                loss = F.mse_loss(model(X, Z, mask, coords, ei, et, em), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                X, y, Z, mask, coords, ei, et, em = _to(batch, device)
                with autocast(enabled=amp):
                    preds.append(model(X, Z, mask, coords, ei, et, em).float().cpu())
                trues.append(y.cpu())
        val_mae = float(np.mean(np.abs(
            torch.cat(preds).numpy() - torch.cat(trues).numpy()))) * EV_TO_MHA
        hist.append(val_mae)

        if val_mae < best:
            best, best_ep, n_bad = val_mae, ep, 0
        else:
            n_bad += 1
        bar.set_postfix(val=f"{val_mae:.3f}", best=f"{best:.3f}@{best_ep}", bad=n_bad)
        if n_bad >= patience:
            break

    return {"best_val_mae_mHa": best, "best_epoch": best_ep,
            "final_epoch": ep, "duration_s": time.time() - t0, "history": hist}
