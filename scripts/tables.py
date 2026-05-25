"""Write LaTeX tables for the paper from the results JSON."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from labels import (DATASETS, DATASET_LABEL, ARCHITECTURES, ARCH_LABEL,
                    DESCRIPTORS, DESC_LABEL, by_key)


def table_best_cells(grid, out):
    """Best (descriptor, architecture) per dataset, with bias_only floor."""
    v = by_key(grid)
    rows = []
    for ds in DATASETS:
        best = min(((d, a, v[(ds, d, a)])
                    for d in DESCRIPTORS for a in ARCHITECTURES
                    if (ds, d, a) in v),
                   key=lambda t: t[2], default=None)
        if best is None:
            continue
        d, a, m = best
        bias = v.get((ds, "chemical", "bias_only"), float("nan"))
        rows.append((DATASET_LABEL[ds], DESC_LABEL[d], ARCH_LABEL[a], m, bias))

    lines = [r"\begin{tabular}{lllrr}", r"\toprule",
             r"Dataset & Best descriptor & Best architecture & MAE (mHa) "
             r"& bias\_only (mHa) \\", r"\midrule"]
    for ds, d, a, m, bias in rows:
        lines.append(rf"{ds} & {d} & {a} & {m:.4f} & {bias:.4f} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out).write_text("\n".join(lines))
    print(f"  wrote {out}")


def table_seed_variance(grid, out):
    """For every cell with ≥2 seeds, write mean ± std of test MAE."""
    by_cell = defaultdict(list)
    for r in grid:
        if r.get("test_mae_mHa") is None:
            continue
        by_cell[(r["dataset"], r["descriptor"], r["architecture"])].append(
            r["test_mae_mHa"])
    multi = {k: vs for k, vs in by_cell.items() if len(vs) >= 2}
    if not multi:
        print("  table_seed_variance: no multi-seed cells, skipping")
        return
    lines = [r"\begin{tabular}{lllr}", r"\toprule",
             r"Dataset & Descriptor & Architecture "
             r"& MAE (mean $\pm$ std, mHa) \\", r"\midrule"]
    for (ds, d, a), vs in sorted(multi.items()):
        arr = np.array(vs)
        lines.append(rf"{DATASET_LABEL[ds]} & {DESC_LABEL[d]} & "
                     rf"{ARCH_LABEL[a]} & ${arr.mean():.4f} \pm {arr.std():.4f}$ \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out).write_text("\n".join(lines))
    print(f"  wrote {out}")
