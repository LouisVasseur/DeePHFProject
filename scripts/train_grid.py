# AI Use: code architected by hand and implemented by hand, optimized and fixed by Claude for modularity, efficiency, and correctness.

"""Train the full 4 datasets × 4 descriptors × 4 architectures grid plus
one bias_only baseline per dataset. Resumable: skips cells already present
in the output JSON for the given seed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deephf import constants as C
from train_cell import run_cell, append_result

DATASETS = ("water", "alkanes", "qm7b_T", "gdb13_T")
DESCRIPTORS = ("chemical", "chemical_elec", "chemical_soap", "chemical_elec_soap")
NN_ARCHITECTURES = ("A_mlp", "B_mpnn", "C_gat", "D_gat_edge")


def planned_cells():
    """One bias_only per dataset (descriptor doesn't matter, use chemical) +
    every (dataset, descriptor, arch) NN combination."""
    cells = [(d, "chemical", "bias_only") for d in DATASETS]
    for d in DATASETS:
        for desc in DESCRIPTORS:
            for a in NN_ARCHITECTURES:
                cells.append((d, desc, a))
    return cells


def already_done(path, seed):
    if not Path(path).exists():
        return set()
    done = set()
    for r in json.loads(Path(path).read_text()):
        if r.get("seed") == seed and r.get("test_mae_mHa") is not None:
            done.add((r["dataset"], r["descriptor"], r["architecture"]))
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--output", default=f"{C.RESULTS_DIR}/results.json")
    p.add_argument("--data-dir", default=C.DATA_DIR)
    p.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=C.ES_PATIENCE)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    cells = planned_cells()
    done = already_done(args.output, args.seed) if args.resume else set()
    todo = [c for c in cells if c not in done]

    print(f"Grid: {len(cells)} total, {len(done)} done, {len(todo)} to run.")
    for i, (ds, desc, arch) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {ds}/{desc}/{arch}/seed{args.seed}")
        try:
            rec = run_cell(ds, desc, arch, args.seed,
                           data_dir=args.data_dir,
                           max_epochs=args.max_epochs,
                           patience=args.patience,
                           num_workers=args.num_workers)
            append_result(args.output, rec)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()
            append_result(args.output, {
                "dataset": ds, "descriptor": desc, "architecture": arch,
                "seed": args.seed, "test_mae_mHa": None, "error": str(e),
            })

    print(f"\nDone. Results: {args.output}")


if __name__ == "__main__":
    main()
