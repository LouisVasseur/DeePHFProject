# AI Use: code inspired from standard SOAP descriptor computation and electronic descriptor computation, 
# but fully implemented by hand and optimized by Claude for efficiency and correctness. 
# The overall structure of the script was also architected by hand for modularity and clarity.

"""Append SOAP + electronic descriptors to the chemical-only HF datasets.

Input  : gnn_data/{subset}_chemical/{split}/             (from prepare_mobml.py)
Output : gnn_data_enriched/{subset}_{variant}/{split}/   for variant in
         {chemical, chemical_elec, chemical_elec_soap}.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deephf.xyz import read_xyz
from deephf.constants import SYMBOL_TO_Z

from electronic import Cache as ElectronicCache
from soap import build_soap, compute_for_mol as compute_soap

Z_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_Z.items()}
SUBSETS = ("water", "alkanes", "qm7b_T", "gdb13_T")


# compute SOAP and electronic descriptors for each example in the dataset split, and return lists of node feature matrices for each variant (chemical, chemical_elec, chemical_elec_soap).
def enrich_split(ds, soap, ecache):
    nf_chem, nf_elec, nf_full = [], [], []
    for ex in tqdm(ds, desc="  enrich", ncols=80):
        nf = np.asarray(ex["node_feature_matrix"], np.float32)          # (n, 17)
        Z, R = read_xyz(ex["xyz_path"])
        soap_v = compute_soap(soap, Z, R)                               # (n, n_soap)
        elec_v, _, _ = ecache.get_or_compute(Z, R)                      # (n, 108) or None
        elec_v = (np.zeros((nf.shape[0], 108), np.float32)
                  if elec_v is None else elec_v.astype(np.float32))
        nf_chem.append(nf)
        nf_elec.append(np.concatenate([nf, elec_v], axis=1))
        nf_full.append(np.concatenate([nf, elec_v, soap_v], axis=1))
    return nf_chem, nf_elec, nf_full


# write_variant: helper to write a dataset variant (with a specific node feature matrix list) to disk, reusing all other columns from the source dataset.
def write_variant(src_ds, nf_list, out_dir):
    cols = {k: src_ds[k] for k in src_ds.column_names if k != "node_feature_matrix"}
    cols["node_feature_matrix"] = nf_list
    Dataset.from_dict(cols).save_to_disk(str(out_dir))

# main: parse args, build SOAP, loop over splits to enrich and write variants.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", choices=SUBSETS, required=True)
    p.add_argument("--input-dir", default="gnn_data")
    p.add_argument("--output-dir", default="gnn_data_enriched")
    p.add_argument("--cache-dir", default="cache_electronic")
    args = p.parse_args()

    in_root = Path(args.input_dir) / f"{args.subset}_chemical"
    out_root = Path(args.output_dir)
    ecache = ElectronicCache(args.cache_dir)

    # Build SOAP from the elements seen in train
    train = load_from_disk(str(in_root / "train"))
    species = set()
    for ex in train:
        Z, _ = read_xyz(ex["xyz_path"])
        species.update(int(z) for z in Z)
    soap = build_soap([Z_TO_SYMBOL[z] for z in species])
    print(f"  SOAP dim = {soap.get_number_of_features()}")

    for split in ("train", "val", "test"):
        print(f"\n--- {args.subset}/{split} ---")
        ds = load_from_disk(str(in_root / split))
        nf_chem, nf_elec, nf_full = enrich_split(ds, soap, ecache)
        write_variant(ds, nf_chem, out_root / f"{args.subset}_chemical" / split)
        write_variant(ds, nf_elec, out_root / f"{args.subset}_chemical_elec" / split)
        write_variant(ds, nf_full, out_root / f"{args.subset}_chemical_elec_soap" / split)

    print(f"\nDone. Variants written under {out_root}")


if __name__ == "__main__":
    main()
