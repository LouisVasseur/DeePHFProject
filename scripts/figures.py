# AI Use: code made fully by Claude for simplicity, based on the results JSON structure and the needs of the paper figures.

"""Plot the paper figures from the results JSONs."""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from labels import (DATASETS, DATASET_LABEL, ARCHITECTURES, ARCH_LABEL,
                    DESCRIPTORS, DESC_LABEL, DESC_COLOR,
                    CHEMICAL_ACCURACY, by_key)


# Figure 1 in paper: in-domain performance across dataset/descriptor/arch cells.
def fig_in_domain(grid, out):
    """Per-dataset grouped bars over (architecture, descriptor)."""
    v = by_key(grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(ARCHITECTURES))
    w = 0.21
    for ax, ds in zip(axes.flat, DATASETS):
        for j, desc in enumerate(DESCRIPTORS):
            vals = [v.get((ds, desc, a), np.nan) for a in ARCHITECTURES]
            ax.bar(x + (j - 1.5) * w, vals, w,
                   color=DESC_COLOR[desc], edgecolor="black", linewidth=0.4,
                   label=DESC_LABEL[desc] if ax is axes[0, 0] else None)
        bias = v.get((ds, "chemical", "bias_only"))
        if bias is not None:
            ax.axhline(bias, color="black", linestyle=":", linewidth=1.2,
                       label="bias_only" if ax is axes[0, 0] else None)
        ax.axhline(CHEMICAL_ACCURACY, color="red", linestyle="--", linewidth=1,
                   label="chem. acc." if ax is axes[0, 0] else None)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABEL[a] for a in ARCHITECTURES])
        ax.set_title(DATASET_LABEL[ds])
        ax.set_ylabel("Test MAE (mHa, log)")
        ax.grid(True, alpha=0.3, which="both", axis="y")
    axes[0, 0].legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

# Figure 2 in paper: feature scaling across descriptors, per architecture.
def fig_feature_scaling(grid, out):
    """Geometric-mean MAE across datasets vs descriptor richness, per arch."""
    v = by_key(grid)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(DESCRIPTORS))
    for a in ARCHITECTURES:
        per_desc = []
        for desc in DESCRIPTORS:
            ms = [v.get((ds, desc, a)) for ds in DATASETS]
            ms = [m for m in ms if m is not None]
            per_desc.append(np.exp(np.mean(np.log(ms))) if ms else np.nan)
        ax.plot(x, per_desc, "-o", label=ARCH_LABEL[a], linewidth=1.5)
    ax.axhline(CHEMICAL_ACCURACY, color="red", linestyle="--", linewidth=1,
               label="chem. acc.")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([DESC_LABEL[d] for d in DESCRIPTORS])
    ax.set_ylabel("Test MAE, geo. mean across datasets (mHa)")
    ax.set_title("Feature scaling per architecture")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

# Figure 3 in paper: edge bias ablation across dataset/descriptor cells.
def fig_edge_bias(ab, out):
    """D_gat_edge vs D_no_edge across (dataset, descriptor) cells."""
    v = by_key(ab)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.4
    pairs = [(ds, desc) for ds in DATASETS for desc in DESCRIPTORS]
    x = np.arange(len(pairs))
    d_vals  = [v.get((ds, desc, "D_gat_edge"), np.nan) for ds, desc in pairs]
    nb_vals = [v.get((ds, desc, "D_no_edge"),  np.nan) for ds, desc in pairs]
    ax.bar(x - width / 2, d_vals,  width, color="#4f7fb4", label="D_gat_edge")
    ax.bar(x + width / 2, nb_vals, width, color="#bbbbbb", label="D_no_edge")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{DATASET_LABEL[d][:4]}\n{DESC_LABEL[desc]}"
                        for d, desc in pairs], fontsize=7)
    ax.set_ylabel("Test MAE (mHa)")
    ax.set_yscale("log")
    ax.axhline(CHEMICAL_ACCURACY, color="red", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3, which="both", axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

# Figure 4 in paper: edge-feature ablation (D_gat_edge vs D_no_rbf vs D_no_bond) for alkanes/chem+elec+SOAP.
def fig_edge_features(records, out):
    """D_gat_edge vs D_no_rbf vs D_no_bond (alkanes)."""
    rs = {r["architecture"]: r["test_mae_mHa"] for r in records
          if r.get("test_mae_mHa") is not None and r.get("seed") == 43}
    order = ["D_gat_edge", "D_no_rbf", "D_no_bond"]
    vals = [rs.get(k, np.nan) for k in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(order, vals, color=["#4f7fb4", "#d18b3e", "#5e9e60"],
           edgecolor="black")
    for i, val in enumerate(vals):
        if not np.isnan(val):
            ax.text(i, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Test MAE (mHa)")
    ax.set_title("Edge-feature decomposition (alkanes)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

# Figure not included in paper: K-sweep for late-stream MP layers (alkanes/chem+elec+SOAP).
def fig_k_sweep(records, out):
    """MAE vs K for each late-stream mode."""
    by_late = defaultdict(dict)
    for r in records:
        if r.get("test_mae_mHa") is None:
            continue
        a = r["architecture"]
        try:
            late = a.split("late=")[1].split("_")[0]
            k = int(a.split("k=")[1])
        except (IndexError, ValueError):
            continue
        by_late[late][k] = r["test_mae_mHa"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for late, vals in by_late.items():
        ks = sorted(vals)
        ax.plot(ks, [vals[k] for k in ks], "-o",
                label=f"late={late}", linewidth=1.5)
    ax.set_xlabel("K (late-stream MP layers)")
    ax.set_ylabel("Test MAE (mHa)")
    ax.set_title("K-sweep (alkanes / chem+elec+SOAP)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
