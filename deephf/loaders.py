# AI Use: code made by hand based on our custom data type. The make_loaders function was suggested and written by Claude for modularity

"""Torch Dataset wrapper and DataLoader builder."""

import torch
from torch.utils.data import Dataset, DataLoader

from .constants import BATCH_SIZE


class MolDataset(Dataset):
    def __init__(self, d):
        self.X  = torch.from_numpy(d["X"])
        self.y = torch.from_numpy(d["y"])
        self.Z = torch.from_numpy(d["Z"]).long()
        self.coords = torch.from_numpy(d["coords"])
        self.edge_index = torch.from_numpy(d["edge_index"])
        self.edge_type  = torch.from_numpy(d["edge_type"])
        self.edge_mask  = torch.from_numpy(d["edge_mask"])
        self.mask  = self.Z > 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.X[i], self.y[i], self.Z[i], self.mask[i],
                self.coords[i], self.edge_index[i],
                self.edge_type[i], self.edge_mask[i])


def make_loaders(train, val, test, num_workers=4, pin=True):
    """Return (train_loader, val_loader, test_loader). 128-batch train, 256 eval."""
    def kw(bs, shuf):
        return dict(batch_size=bs, shuffle=shuf, num_workers=num_workers,
                    pin_memory=pin, persistent_workers=num_workers > 0,
                    prefetch_factor=4 if num_workers > 0 else None)
    return (DataLoader(MolDataset(train), **kw(BATCH_SIZE, True)),
            DataLoader(MolDataset(val),   **kw(max(BATCH_SIZE, 256), False)),
            DataLoader(MolDataset(test),  **kw(max(BATCH_SIZE, 256), False)))
