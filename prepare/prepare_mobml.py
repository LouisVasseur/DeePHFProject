# AI Use: code mostly from Claude with some hand edits for data format and efficiency. 
# AI used to iterate on properties of existing datasets to find the right ones to use as node features.


"""Step 1 of feature prep: MOB-ML xyz + energy.dat → HuggingFace dataset.

Output: gnn_data/{subset}/{train,val,test}/ each with columns
    xyz_path, mol_id, E_HF_Ha, E_corr_Ha, E_corr_eV,
    node_feature_matrix (17 dim), edge_feature_matrix (5 dim),
    indices_feature_matrix.

The next script (compute_descriptors.py) reads these and appends the
electronic (108) and SOAP (~1.5k) descriptors to produce the four variants
expected by the training code.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from graph import build_graph

HARTREE_TO_EV = 27.211386245988
SUBSETS = ("water", "alkanes", "qm7b_T", "gdb13_T")


def read_energy_dat(path):
    """energy.dat has columns: mol_id E_HF E_MP2 (in Hartree)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            mol_id, e_hf, e_mp2 = parts[0], float(parts[1]), float(parts[2])
            rows.append((mol_id, e_hf, e_mp2 - e_hf))
    return rows


def find_geometry_dir(base):
    for name in ("geometry", "geometries"):
        d = base / name
        if d.is_dir():
            return d
    raise FileNotFoundError(f"No geometry dir under {base}")


def build_hf_dataset(data_dir, subset):
    base = Path(data_dir) / "caltech_mobml" / "data" / subset
    rows = read_energy_dat(base / "energy.dat")
    geom_dir = find_geometry_dir(base)

    records = []
    for mol_id, e_hf, e_corr in tqdm(rows, desc=f"  scan {subset}", ncols=80):
        xyz = geom_dir / f"{mol_id}.xyz"
        if not xyz.exists():
            continue
        records.append({
            "mol_id": mol_id,
            "xyz_path": str(xyz.resolve()),
            "E_HF_Ha": e_hf,
            "E_corr_Ha": e_corr,
            "E_corr_eV": e_corr * HARTREE_TO_EV,
        })
    return Dataset.from_list(records)


def add_graph_features(ds, num_proc=1):
    def encode(batch):
        nfs, efs, idxs = [], [], []
        for p in batch["xyz_path"]:
            nf, ef, idx = build_graph(p)
            nfs.append(nf); efs.append(ef); idxs.append(idx)
        return {"node_feature_matrix": nfs,
                "edge_feature_matrix": efs,
                "indices_feature_matrix": idxs}

    ds = ds.map(encode, batched=True, batch_size=64, num_proc=num_proc)
    return ds.filter(lambda x: x["node_feature_matrix"] is not None)


def split_save(ds, out_dir, seed=42):
    """80/10/10 split. Saved as gnn_data/{subset}_chemical/{train,val,test}/."""
    s1 = ds.train_test_split(test_size=0.2, seed=seed)
    s2 = s1["test"].train_test_split(test_size=0.5, seed=seed)
    for name, split in [("train", s1["train"]),
                        ("val", s2["train"]), ("test", s2["test"])]:
        split.save_to_disk(os.path.join(out_dir, name))
        print(f"  {name}: {len(split)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="deephf_datasets",
                   help="Root with caltech_mobml/data/{subset}/")
    p.add_argument("--subset", choices=SUBSETS, required=True)
    p.add_argument("--output-dir", default="gnn_data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-proc", type=int, default=1)
    args = p.parse_args()

    ds = build_hf_dataset(args.data_dir, args.subset)
    print(f"Loaded {len(ds)} molecules from {args.subset}")
    ds = add_graph_features(ds, num_proc=args.num_proc)
    print(f"After graph build: {len(ds)} molecules")

    out = Path(args.output_dir) / f"{args.subset}_chemical"
    split_save(ds, out, seed=args.seed)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
