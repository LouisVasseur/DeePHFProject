"""
DeePHF Descriptor Ablation Training
=====================================
Trains CorrNet on three descriptor configurations:
  (a) electronic_only — 108-dim eigenvalues from PySCF (baseline DeePHF)
  (b) atomic_only — SOAP features from DScribe
  (c) combined — electronic + w_atomic * SOAP

Supports multiple datasets (QM7-X, rMD17, GMTKN55, water).
Generates learning curves and per-element error analysis.

Usage:
    # Quick test on rMD17 ethanol
    python deephf_train.py --dataset rmd17 --molecule ethanol --max-samples 500

    # Full ablation on QM7-X
    python deephf_train.py --dataset qm7x --files 8000.hdf5 --max-samples 2000

    # Specific mode only
    python deephf_train.py --dataset rmd17 --molecule ethanol --mode combined --w-atomic 0.1
"""

from __future__ import annotations

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ── Import project modules ────────────────────────────────────────────────
from deephf_dataloader import load_dataset, MoleculeData
from deephf_descriptors import (
    SOAPDescriptor, SOAPConfig, CombinedDescriptor, DescriptorConfig,
    DescriptorMode, compute_soap_for_dataset,
)

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631


# ══════════════════════════════════════════════════════════════════════════
# CorrNet Model (from your existing pipeline)
# ══════════════════════════════════════════════════════════════════════════
class CorrNet(nn.Module):
    """
    Per-atom neural network for correlation energy prediction.

    Architecture: 3×100 GELU ResNet with linear pre-fitting branch.
    Input: (batch, n_atoms, n_desc) descriptors
    Output: (batch,) total energy (sum over atoms)
    """

    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...] = (100, 100, 100)):
        super().__init__()
        self.input_dim = input_dim

        # Input normalization buffers
        self.register_buffer("input_shift", torch.zeros(input_dim))
        self.register_buffer("input_scale", torch.ones(input_dim))

        # Linear pre-fitting branch (initialized via ridge regression)
        self.linear = nn.Linear(input_dim, 1)

        # Nonlinear branch: ResNet with GELU
        layers = []
        sizes = [input_dim, *hidden_sizes]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        # Final projection to scalar
        layers.append(nn.Linear(sizes[-1], 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_atoms, n_desc) descriptor tensor
        Returns:
            (batch,) predicted molecular energy
        """
        # Normalize
        x = (x - self.input_shift) / (self.input_scale + 1e-8)

        # Linear branch
        l = self.linear(x)  # (batch, n_atoms, 1)

        # Nonlinear branch with residual connections
        h = x
        for i, layer in enumerate(self.layers):
            h_out = layer(h)
            if i < len(self.layers) - 1:
                h_out = F.gelu(h_out)
                # Residual when dims match
                if h.shape[-1] == h_out.shape[-1]:
                    h_out = h + h_out
            h = h_out

        # Per-atom energy = linear + nonlinear
        e_atom = h + l  # (batch, n_atoms, 1)

        # Sum over atoms → molecular energy
        return e_atom.sum(dim=1).squeeze(-1)  # (batch,)

    def set_normalization(self, shift: np.ndarray, scale: np.ndarray):
        self.input_shift.copy_(torch.tensor(shift, dtype=torch.float32))
        self.input_scale.copy_(torch.tensor(scale, dtype=torch.float32))

    def set_prefitting(self, X_train: np.ndarray, y_train: np.ndarray, ridge_alpha: float = 10.0):
        """Initialize linear branch via ridge regression on summed descriptors."""
        from sklearn.linear_model import Ridge

        X_sum = X_train.sum(axis=1)  # (N, n_desc) — sum over atoms
        reg = Ridge(alpha=ridge_alpha)
        reg.fit(X_sum, y_train)

        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(reg.coef_, dtype=torch.float32).reshape(1, -1)
            )
            self.linear.bias.copy_(
                torch.tensor(reg.intercept_ / max(X_train.shape[1], 1),
                             dtype=torch.float32).reshape(1)
            )
        logger.info(f"  Pre-fitting: Ridge R² = {reg.score(X_sum, y_train):.4f}")


# ══════════════════════════════════════════════════════════════════════════
# Dataset for PyTorch
# ══════════════════════════════════════════════════════════════════════════
class MolDescDataset(Dataset):
    """PyTorch dataset wrapping descriptor arrays + target energies."""

    def __init__(self, descriptors: np.ndarray, energies: np.ndarray,
                 atomic_numbers: Optional[np.ndarray] = None):
        """
        Args:
            descriptors: (N, n_atoms, n_features) padded descriptor array
            energies: (N,) target energies
            atomic_numbers: (N, n_atoms) for per-element analysis
        """
        self.X = torch.tensor(descriptors, dtype=torch.float32)
        self.y = torch.tensor(energies, dtype=torch.float32)
        self.Z = atomic_numbers

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_sizes: Tuple[int, ...] = (100, 100, 100),
    lr: float = 3e-4,
    epochs: int = 500,
    batch_size: int = 32,
    prefitting: bool = True,
    lr_decay_rate: float = 0.5,
    lr_decay_steps: int = 150,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[CorrNet, Dict]:
    """
    Train CorrNet and return model + training history.

    Returns:
        (model, history) where history has keys:
          train_loss, val_loss, train_mae, val_mae, best_val_mae, best_epoch
    """
    n_desc = X_train.shape[-1]
    model = CorrNet(n_desc, hidden_sizes=hidden_sizes).to(device)

    # Normalization from training data
    flat_X = X_train.reshape(-1, n_desc)
    shift = flat_X.mean(axis=0)
    scale = flat_X.std(axis=0) + 1e-8
    model.set_normalization(shift, scale)

    # Pre-fit linear branch
    if prefitting:
        model.set_prefitting(X_train, y_train)

    # Data loaders
    train_ds = MolDescDataset(X_train, y_train)
    val_ds = MolDescDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_steps, lr_decay_rate)

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = F.mse_loss(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            all_pred, all_true = [], []
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                pred = model(X_batch)
                all_pred.append(pred.cpu().numpy())
                all_true.append(y_batch.numpy())

            val_pred = np.concatenate(all_pred)
            val_true = np.concatenate(all_true)
            val_mae = np.mean(np.abs(val_pred - val_true))
            val_loss = np.mean((val_pred - val_true) ** 2)

        train_mae = np.sqrt(np.mean(train_losses))  # approx
        history["train_loss"].append(np.mean(train_losses))
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"train_loss={np.mean(train_losses):.6f} | "
                f"val_MAE={val_mae:.6f} eV | "
                f"best={best_val_mae:.6f} eV @{best_epoch} | "
                f"lr={current_lr:.2e}"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_mae"] = best_val_mae
    history["best_epoch"] = best_epoch
    return model, history


# ══════════════════════════════════════════════════════════════════════════
# Descriptor preparation (pad variable-size molecules)
# ══════════════════════════════════════════════════════════════════════════
def prepare_descriptors(
    molecules: List[MoleculeData],
    desc_config: DescriptorConfig,
    electronic_descs: Optional[List[np.ndarray]] = None,
    normalize: bool = True,
    target: str = "energy",
    corr_energies: Optional[List[Optional[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute descriptors and pad to uniform size for batching.

    Args:
        normalize: If True, z-score normalize electronic and SOAP descriptors
                   independently before combining.
        target: What to predict. Options:
            "energy" — total energy (mol.energy), default
            "E_corr" — correlation energy from mol.properties['E_corr'] (QM7-X)
            "mp2_corr" — MP2 correlation energy from corr_energies list
        corr_energies: List of MP2 correlation energies (from electronic descriptor
                       computation), used when target="mp2_corr"

    Returns:
        X: (N, max_atoms, n_features) padded descriptors
        y: (N,) target energies in eV
        Z: (N, max_atoms) padded atomic numbers
    """
    # Filter molecules with valid coords and appropriate target
    def _get_target(i, m):
        if target == "energy":
            return m.energy
        elif target == "E_corr":
            return m.properties.get("E_corr")
        elif target == "mp2_corr":
            if corr_energies and i < len(corr_energies) and corr_energies[i] is not None:
                return corr_energies[i] * HARTREE_TO_EV  # Ha → eV
            return None
        return m.energy

    valid = [
        (i, m) for i, m in enumerate(molecules)
        if _get_target(i, m) is not None
        and m.coords is not None
        and np.any(m.coords != 0)
        and len(m.atomic_numbers) == m.coords.shape[0]
    ]

    if not valid:
        raise ValueError("No valid molecules with energy + coords found")

    indices, mols = zip(*valid)
    mols = list(mols)

    mode = desc_config.mode

    # ── Compute raw descriptors per type ──
    e_descs = None
    soap_descs = None

    if mode in (DescriptorMode.ELECTRONIC_ONLY, DescriptorMode.COMBINED):
        if electronic_descs is not None:
            e_descs = [electronic_descs[i] for i in indices]
            # Filter out None entries (failed HF)
            valid_mask = [d is not None for d in e_descs]
            if not all(valid_mask):
                n_bad = sum(1 for v in valid_mask if not v)
                logger.warning(f"  {n_bad} molecules have no electronic descriptors, filtering out")
                keep = [j for j, v in enumerate(valid_mask) if v]
                mols = [mols[j] for j in keep]
                indices = tuple(indices[j] for j in keep)
                e_descs = [e_descs[j] for j in keep]
        else:
            logger.warning("No electronic descriptors provided — using random placeholders!")
            e_descs = [np.random.randn(m.n_atoms, 108) for m in mols]

    if mode in (DescriptorMode.ATOMIC_ONLY, DescriptorMode.COMBINED):
        soap_descs, _ = compute_soap_for_dataset(mols)

    # ── Normalize independently ──
    if normalize:
        if e_descs is not None:
            e_flat = np.concatenate([d.reshape(-1, d.shape[-1]) for d in e_descs], axis=0)
            e_mean = e_flat.mean(axis=0)
            e_std = e_flat.std(axis=0) + 1e-8
            e_descs = [(d - e_mean) / e_std for d in e_descs]
            logger.info(f"  Electronic descriptors normalized: mean~0, std~1 (raw range was [{e_flat.min():.2f}, {e_flat.max():.2f}])")

        if soap_descs is not None:
            s_flat = np.concatenate([d.reshape(-1, d.shape[-1]) for d in soap_descs], axis=0)
            s_mean = s_flat.mean(axis=0)
            s_std = s_flat.std(axis=0) + 1e-8
            soap_descs = [(d - s_mean) / s_std for d in soap_descs]
            logger.info(f"  SOAP descriptors normalized: mean~0, std~1 (raw range was [{s_flat.min():.2f}, {s_flat.max():.2f}])")

    # ── Assemble final descriptors ──
    if mode == DescriptorMode.ELECTRONIC_ONLY:
        descs = e_descs
    elif mode == DescriptorMode.ATOMIC_ONLY:
        descs = soap_descs
    else:  # COMBINED
        descs = []
        for e, s in zip(e_descs, soap_descs):
            # After normalization both are ~N(0,1), so w_atomic controls relative importance
            scaled_s = desc_config.w_atomic * s
            descs.append(np.concatenate([e, scaled_s], axis=1))

    # Pad to uniform size
    max_atoms = max(d.shape[0] for d in descs)
    n_features = descs[0].shape[1]

    X = np.zeros((len(descs), max_atoms, n_features), dtype=np.float32)
    Z = np.zeros((len(descs), max_atoms), dtype=np.int32)
    y = np.array([_get_target(indices[j], mols[j]) for j in range(len(mols))], dtype=np.float32)

    for i, (desc, mol) in enumerate(zip(descs, mols)):
        n = desc.shape[0]
        X[i, :n, :] = desc
        Z[i, :n] = mol.atomic_numbers[:n]

    logger.info(f"Prepared {len(y)} molecules: X={X.shape}, y range=[{y.min():.2f}, {y.max():.2f}] eV")
    return X, y, Z


# ══════════════════════════════════════════════════════════════════════════
# Ablation experiment
# ══════════════════════════════════════════════════════════════════════════
def run_ablation(
    molecules: List[MoleculeData],
    modes: List[str] = ["electronic_only", "atomic_only", "combined"],
    w_atomic: float = 0.1,
    w_atomic_sweep: Optional[List[float]] = None,
    normalize: bool = True,
    target: str = "energy",
    corr_energies: Optional[List[Optional[float]]] = None,
    train_sizes: Optional[List[int]] = None,
    test_frac: float = 0.2,
    epochs: int = 500,
    batch_size: int = 32,
    lr: float = 3e-4,
    hidden_sizes: Tuple[int, ...] = (100, 100, 100),
    seed: int = 42,
    output_dir: str = "./results",
    device: str = "cpu",
    electronic_descs: Optional[List[np.ndarray]] = None,
) -> Dict:
    """
    Run the full descriptor ablation experiment.

    Args:
        molecules: List of MoleculeData from the data loader
        modes: Which ablation configs to run
        w_atomic: SOAP weight for combined mode (ignored if w_atomic_sweep is set)
        w_atomic_sweep: List of w_atomic values to sweep (e.g. [0.01, 0.1, 0.5, 1.0, 5.0])
        normalize: Z-score normalize electronic and SOAP independently before combining
        train_sizes: List of training set sizes for learning curves
        test_frac: Fraction held out for testing
        epochs: Training epochs per run
        output_dir: Where to save results

    Returns:
        Dict with results per mode per train_size
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Expand modes: if sweep requested, add combined_w=X for each w
    expanded_modes = []
    for mode_name in modes:
        if mode_name == "combined" and w_atomic_sweep:
            for w in w_atomic_sweep:
                expanded_modes.append(f"combined_w={w}")
        else:
            expanded_modes.append(mode_name)

    results = {}

    for mode_name in expanded_modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Mode: {mode_name} (normalize={normalize})")
        logger.info(f"{'='*60}")

        # Build config
        if mode_name == "electronic_only":
            config = DescriptorConfig.electronic_only()
        elif mode_name == "atomic_only":
            config = DescriptorConfig.atomic_only()
        elif mode_name.startswith("combined_w="):
            w = float(mode_name.split("=")[1])
            config = DescriptorConfig.combined(w_atomic=w)
        elif mode_name == "combined":
            config = DescriptorConfig.combined(w_atomic=w_atomic)
        else:
            raise ValueError(f"Unknown mode: {mode_name}")

        # Compute descriptors
        t0 = time.time()
        X, y, Z = prepare_descriptors(molecules, config, electronic_descs,
                                       normalize=normalize, target=target,
                                       corr_energies=corr_energies)
        desc_time = time.time() - t0
        logger.info(f"Descriptor computation: {desc_time:.1f}s, shape={X.shape}")

        # Train/test split
        n = len(y)
        n_test = max(int(n * test_frac), 10)
        n_avail = n - n_test

        perm = np.random.permutation(n)
        test_idx = perm[:n_test]
        train_pool = perm[n_test:]

        X_test, y_test, Z_test = X[test_idx], y[test_idx], Z[test_idx]

        # Learning curve sizes
        if train_sizes is None:
            train_sizes_actual = [s for s in [25, 50, 100, 200, 500, 1000] if s <= n_avail]
            if not train_sizes_actual:
                train_sizes_actual = [n_avail]
        else:
            train_sizes_actual = [s for s in train_sizes if s <= n_avail]

        mode_results = {
            "n_features": int(X.shape[-1]),
            "desc_time_s": desc_time,
            "train_sizes": [],
        }

        for n_train in train_sizes_actual:
            logger.info(f"\n--- {mode_name} | n_train={n_train} ---")
            train_idx = train_pool[:n_train]
            X_train, y_train = X[train_idx], y[train_idx]

            t0 = time.time()
            model, history = train_model(
                X_train, y_train, X_test, y_test,
                hidden_sizes=hidden_sizes,
                lr=lr, epochs=epochs, batch_size=min(batch_size, n_train),
                device=device,
            )
            train_time = time.time() - t0

            # Per-element MAE
            element_mae = compute_element_mae(model, X_test, y_test, Z_test, device)

            run_result = {
                "n_train": n_train,
                "n_test": n_test,
                "best_val_mae_eV": float(history["best_val_mae"]),
                "best_val_mae_kcal": float(history["best_val_mae"]) * HARTREE_TO_KCAL / HARTREE_TO_EV,
                "best_epoch": int(history["best_epoch"]),
                "train_time_s": train_time,
                "element_mae": element_mae,
                "final_train_loss": float(history["train_loss"][-1]),
            }
            mode_results["train_sizes"].append(run_result)

            logger.info(
                f"  Result: MAE={run_result['best_val_mae_eV']:.4f} eV "
                f"({run_result['best_val_mae_kcal']:.2f} kcal/mol) "
                f"@ epoch {run_result['best_epoch']} "
                f"[{train_time:.1f}s]"
            )

        results[mode_name] = mode_results

    # Save results
    results_file = os.path.join(output_dir, "ablation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_file}")

    # Print summary table
    print_summary(results)

    return results


def compute_element_mae(
    model: CorrNet,
    X: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    device: str = "cpu",
) -> Dict[int, float]:
    """Compute MAE per element type."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(X_t).cpu().numpy()

    errors = np.abs(pred - y)

    # Group by elements present in each molecule
    element_errors = {}
    unique_elements = set()
    for i in range(len(Z)):
        unique_elements.update(Z[i][Z[i] > 0].tolist())

    for elem in sorted(unique_elements):
        mask = np.array([elem in Z[i][Z[i] > 0] for i in range(len(Z))])
        if mask.sum() > 0:
            element_errors[int(elem)] = float(errors[mask].mean())

    return element_errors


def print_summary(results: Dict):
    """Print a clean summary table of ablation results."""
    print("\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)

    # Element symbol lookup
    elem_map = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 17: "Cl", 9: "F"}

    for mode, data in results.items():
        print(f"\n  {mode} ({data['n_features']} features, desc={data['desc_time_s']:.1f}s)")
        print(f"  {'n_train':>8s} {'MAE(eV)':>10s} {'MAE(kcal)':>10s} {'epoch':>6s} {'time':>6s}")
        print(f"  {'-'*44}")

        for run in data["train_sizes"]:
            print(
                f"  {run['n_train']:8d} "
                f"{run['best_val_mae_eV']:10.4f} "
                f"{run['best_val_mae_kcal']:10.2f} "
                f"{run['best_epoch']:6d} "
                f"{run['train_time_s']:5.1f}s"
            )

            if run.get("element_mae"):
                elems = ", ".join(
                    f"{elem_map.get(z, f'Z={z}')}={mae:.4f}"
                    for z, mae in sorted(run["element_mae"].items())
                )
                print(f"           per-element: {elems}")

    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="DeePHF Descriptor Ablation Training")
    parser.add_argument("--data-dir", default="./deephf_datasets", help="Dataset root")
    parser.add_argument("--dataset", default="mobml",
                        choices=["mobml", "rmd17", "qm7x", "gmtkn55", "water"])
    parser.add_argument("--molecule", default="ethanol", help="rMD17 molecule name")
    parser.add_argument("--mobml-subset", default="water",
                        choices=["water", "alkanes", "qm7b_T", "gdb13_T"],
                        help="MOB-ML subset (DeePHF benchmark)")
    parser.add_argument("--files", nargs="+", default=None, help="QM7-X HDF5 files")
    parser.add_argument("--subset", default="S22", help="GMTKN55 subset")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--stride", type=int, default=100, help="rMD17 subsampling stride")
    parser.add_argument("--modes", nargs="+",
                        default=["electronic_only", "atomic_only", "combined"],
                        help="Ablation modes to run")
    parser.add_argument("--w-atomic", type=float, default=0.1, help="SOAP weight in combined mode")
    parser.add_argument("--w-atomic-sweep", nargs="+", type=float, default=None,
                        help="Sweep w_atomic values (e.g. 0.01 0.1 0.5 1.0 5.0)")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Z-score normalize descriptors independently (default: True)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="Disable descriptor normalization")
    parser.add_argument("--train-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--target", default="auto",
                        choices=["auto", "energy", "E_corr", "mp2_corr"],
                        help="Training target: auto=E_corr for qm7x, mp2_corr for rmd17")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", nargs="+", type=int, default=[100, 100, 100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hf-basis", default="cc-pvdz", help="Basis set for HF (electronic descriptors)")
    parser.add_argument("--cache-dir", default="./cache_electronic", help="Cache dir for HF results")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel HF jobs")
    parser.add_argument("--skip-electronic", action="store_true",
                        help="Skip electronic descriptor computation (use placeholders)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    # Always show INFO for our loggers
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger("deephf_dataloader").setLevel(logging.INFO)
    logging.getLogger("deephf_descriptors").setLevel(logging.INFO)
    logging.getLogger("deephf_electronic").setLevel(logging.INFO)

    # Load dataset
    logger.info(f"Loading {args.dataset}...")
    load_kwargs = {"data_dir": args.data_dir, "max_samples": args.max_samples}

    if args.dataset == "mobml":
        load_kwargs["subset"] = args.mobml_subset
    elif args.dataset == "rmd17":
        load_kwargs["molecule"] = args.molecule
        load_kwargs["stride"] = args.stride
    elif args.dataset == "qm7x":
        if args.files:
            load_kwargs["files"] = args.files
    elif args.dataset == "gmtkn55":
        load_kwargs["subset"] = args.subset

    molecules = load_dataset(args.dataset, **load_kwargs)
    logger.info(f"Loaded {len(molecules)} molecules")

    # Compute electronic descriptors (the expensive step)
    electronic_descs = None
    needs_electronic = any(m in ["electronic_only", "combined"] for m in args.modes)

    if needs_electronic and not args.skip_electronic:
        from deephf_electronic import compute_electronic_for_dataset

        logger.info(f"Computing electronic descriptors (basis={args.hf_basis})...")
        electronic_descs, hf_energies, corr_energies = compute_electronic_for_dataset(
            molecules,
            hf_basis=args.hf_basis,
            cache_dir=args.cache_dir,
            n_jobs=args.n_jobs,
        )

        n_ok = sum(1 for d in electronic_descs if d is not None)
        logger.info(f"Electronic descriptors: {n_ok}/{len(molecules)} succeeded")
    elif needs_electronic:
        logger.warning("Electronic modes requested but --skip-electronic set. Using placeholders.")

    # Auto-select training target
    target = args.target
    corr_energies_list = None

    if target == "auto":
        if args.dataset == "mobml":
            target = "energy"  # MOB-ML loader already sets energy = E_corr in eV
            logger.info("Auto-selected target: energy (E_corr from MOB-ML, already in eV)")
        elif args.dataset == "qm7x":
            target = "E_corr"
            logger.info("Auto-selected target: E_corr (QM7-X DFT correlation energy)")
        elif needs_electronic and not args.skip_electronic:
            target = "mp2_corr"
            logger.info("Auto-selected target: mp2_corr (MP2 correlation energy from HF)")
        else:
            target = "energy"
            logger.info("Auto-selected target: energy (total energy)")

    if target == "mp2_corr" and corr_energies is not None:
        corr_energies_list = corr_energies

    # Run ablation
    if args.dataset == "mobml":
        run_label = f"mobml_{args.mobml_subset}"
    elif args.dataset == "rmd17":
        run_label = f"rmd17_{args.molecule}"
    else:
        run_label = args.dataset
    output_dir = os.path.join(args.output_dir, run_label)

    results = run_ablation(
        molecules=molecules,
        modes=args.modes,
        w_atomic=args.w_atomic,
        w_atomic_sweep=args.w_atomic_sweep,
        normalize=args.normalize,
        target=target,
        corr_energies=corr_energies_list,
        train_sizes=args.train_sizes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_sizes=tuple(args.hidden),
        seed=args.seed,
        output_dir=output_dir,
        device=args.device,
        electronic_descs=electronic_descs,
    )


if __name__ == "__main__":
    main()