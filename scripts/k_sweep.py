# AI Use: code architected by hand and implemented by hand, optimized and fixed by Claude

"""Appendix K-sweep on alkanes / chemical_elec_soap.

Two late-stream modes × four K values:
    late='elec' : main = [chem + soap], late = [elec]   — does elec need MP?
    late='soap' : main = [chem + elec], late = [soap]   — does SOAP need MP?
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deephf import constants as C
from deephf.data import load_dataset
from deephf.dual_stream import DualStreamGATEdge
from deephf.evaluate import evaluate
from deephf.loaders import make_loaders
from deephf.train import train_model

# run_one: helper to run one cell (dataset/descriptor/late/k/seed) and return a record dict with results and metadata.
def run_one(dataset, descriptor, late_kind, k, seed,
            data_dir, max_epochs, patience, num_workers):
    torch.manual_seed(seed); np.random.seed(seed)
    train, val, test = load_dataset(dataset, descriptor, data_dir=data_dir)
    max_z = max(int(d["Z"].max()) for d in (train, val, test)) + 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DualStreamGATEdge(
        total_input_dim=train["desc_dim"], edge_feat_dim=train["edge_feat_dim"],
        late_kind=late_kind, k=k, max_z=max(20, max_z),
    ).to(device)

    Xt = torch.from_numpy(train["X"]).to(device)
    Zt = torch.from_numpy(train["Z"]).long().to(device)
    yt = torch.from_numpy(train["y"]).to(device)
    mt = Zt > 0
    model.init_normalization(Xt, mt)
    r2 = model.init_element_bias(Zt, yt, mt)

    tr_l, va_l, te_l = make_loaders(train, val, test,
                                    num_workers=num_workers,
                                    pin=(device != "cpu"))
    tr = train_model(model, tr_l, va_l, device,
                     max_epochs=max_epochs, patience=patience)
    out = evaluate(model, te_l, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "dataset": dataset, "descriptor": descriptor,
        "architecture": f"D_gat_edge_late={late_kind}_k={k}",
        "seed": seed,
        "test_mae_mHa": out["test_mae_mHa"],
        "test_mae_kcal": out["test_mae_kcal"],
        "best_val_mae_mHa": tr["best_val_mae_mHa"],
        "best_epoch": tr["best_epoch"], "final_epoch": tr["final_epoch"],
        "n_train": len(train["y"]), "main_dim": model.main_dim,
        "late_dim": model.late_dim, "k": k,
        "element_bias_r2": r2, "n_params": n_params,
        "train_duration_s": tr["duration_s"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="alkanes")
    p.add_argument("--descriptor", default="chemical_elec_soap")
    p.add_argument("--lates", nargs="+", default=["elec", "soap"])
    p.add_argument("--ks", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--output", default=f"{C.RESULTS_DIR}/k_sweep.json")
    p.add_argument("--data-dir", default=C.DATA_DIR)
    p.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=C.ES_PATIENCE)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else []

    cells = [(L, k) for L in args.lates for k in args.ks]
    for i, (L, k) in enumerate(cells, 1):
        tag = (args.dataset, args.descriptor,
               f"D_gat_edge_late={L}_k={k}", args.seed)
        if any((r["dataset"], r["descriptor"], r["architecture"], r["seed"])
               == tag for r in results if r.get("test_mae_mHa") is not None):
            print(f"[{i}/{len(cells)}] skip {tag}")
            continue
        print(f"\n[{i}/{len(cells)}] {args.dataset}/{args.descriptor}/"
              f"late={L}/k={k}/seed{args.seed}")
        t0 = time.time()
        rec = run_one(args.dataset, args.descriptor, L, k, args.seed,
                      args.data_dir, args.max_epochs, args.patience,
                      args.num_workers)
        results.append(rec)
        print(f"  test_mae={rec['test_mae_mHa']:.4f} mHa  ({time.time() - t0:.0f}s)")
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\nDone. Results: {args.output}")


if __name__ == "__main__":
    main()
