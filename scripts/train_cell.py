# AI Use: code architected by hand and implemented by hand, optimized and fixed by Claude for modularity, efficiency, and correctness.

"""Train one cell (dataset, descriptor, architecture, seed) and append the
result record to a JSON file."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deephf import constants as C
from deephf.data import load_dataset
from deephf.evaluate import evaluate
from deephf.loaders import make_loaders
from deephf.model import UnifiedModel
from deephf.train import train_model

# main function to run one cell (dataset/descriptor/architecture/seed) and return a record dict with results and metadata.
def run_cell(dataset, descriptor, architecture, seed,
             data_dir=C.DATA_DIR, max_epochs=C.MAX_EPOCHS,
             patience=C.ES_PATIENCE, num_workers=4, device=None,
             use_amp=True, save_preds_to=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)

    train, val, test = load_dataset(dataset, descriptor, data_dir=data_dir)
    max_z = max(int(d["Z"].max()) for d in (train, val, test)) + 1
    model = UnifiedModel(input_dim=train["desc_dim"],
                         architecture=architecture,
                         edge_feat_dim=train["edge_feat_dim"],
                         max_z=max(20, max_z)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    Xt = torch.from_numpy(train["X"]).to(device)
    Zt = torch.from_numpy(train["Z"]).long().to(device)
    yt = torch.from_numpy(train["y"]).to(device)
    mt = Zt > 0
    model.init_normalization(Xt, mt)
    r2 = model.init_element_bias(Zt, yt, mt)
    print(f"  desc_dim={train['desc_dim']}, edge_feat_dim={train['edge_feat_dim']}, "
          f"params={n_params:,}, elem_bias_R²={r2:.4f}", flush=True)

    tr_l, va_l, te_l = make_loaders(train, val, test,
                                    num_workers=num_workers,
                                    pin=(device != "cpu"))

    if architecture == "bias_only":
        out = evaluate(model, te_l, device, use_amp=use_amp)
        print(f"  test_mae={out['test_mae_mHa']:.4f} mHa  (bias-only)", flush=True)
        return _record(dataset, descriptor, architecture, seed, train,
                       out, None, n_params, r2)

    tr = train_model(model, tr_l, va_l, device,
                     max_epochs=max_epochs, patience=patience, use_amp=use_amp)
    out = evaluate(model, te_l, device, use_amp=use_amp)
    print(f"  test_mae={out['test_mae_mHa']:.4f} mHa  "
          f"best_val={tr['best_val_mae_mHa']:.4f}@{tr['best_epoch']}  "
          f"train={tr['duration_s']:.0f}s", flush=True)

    if save_preds_to:
        np.savez(save_preds_to, y_true=out["y_true"], y_pred=out["y_pred"])

    return _record(dataset, descriptor, architecture, seed, train,
                   out, tr, n_params, r2)


# helper to build a record dict from the results of one cell, for appending to the JSON.
def _record(dataset, descriptor, architecture, seed, train, out, tr,
            n_params, r2):
    return {
        "dataset": dataset, "descriptor": descriptor,
        "architecture": architecture, "seed": seed,
        "test_mae_mHa": out["test_mae_mHa"],
        "test_mae_kcal": out["test_mae_kcal"],
        "best_val_mae_mHa": tr["best_val_mae_mHa"] if tr else None,
        "best_epoch":       tr["best_epoch"]       if tr else None,
        "final_epoch":      tr["final_epoch"]      if tr else None,
        "n_train": len(train["y"]), "desc_dim": train["desc_dim"],
        "n_params": n_params, "element_bias_r2": r2,
        "train_duration_s": tr["duration_s"] if tr else 0.0,
    }


# helper to append a record to a JSON file, creating it if it doesn't exist.
def append_result(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(path.read_text()) if path.exists() else []
    records.append(record)
    path.write_text(json.dumps(records, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=("water", "alkanes", "qm7b_T", "gdb13_T"))
    p.add_argument("--descriptor", required=True,
                   choices=("chemical", "chemical_elec",
                            "chemical_soap", "chemical_elec_soap"))
    p.add_argument("--architecture", required=True,
                   choices=("bias_only", "A_mlp", "B_mpnn", "C_gat", "D_gat_edge"))
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--data-dir", default=C.DATA_DIR)
    p.add_argument("--output", default=f"{C.RESULTS_DIR}/results.json")
    p.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=C.ES_PATIENCE)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--save-preds", default=None)
    args = p.parse_args()

    print(f"[{args.dataset}/{args.descriptor}/{args.architecture}/seed{args.seed}]")
    rec = run_cell(args.dataset, args.descriptor, args.architecture, args.seed,
                   data_dir=args.data_dir, max_epochs=args.max_epochs,
                   patience=args.patience, num_workers=args.num_workers,
                   device=args.device, use_amp=not args.no_amp,
                   save_preds_to=args.save_preds)
    append_result(args.output, rec)
    print(f"  → appended to {args.output}")


if __name__ == "__main__":
    main()
