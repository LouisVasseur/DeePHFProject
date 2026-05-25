"""Edge-bias ablation: D_gat_edge with vs without the edge-bias term.

`D_no_edge` is the same multi-head GAT-with-edge-features class as
D_gat_edge, but the additive edge term in the attention score is dropped
(set `use_edge_bias=False`). All other code paths are shared so any gap is
attributable to that single mechanism.

Trains the 2 variants on (dataset × descriptor) pairs and writes to
results/edge_bias_ablation.json.
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
from deephf.evaluate import evaluate
from deephf.loaders import make_loaders
from deephf.model import UnifiedModel
from deephf.train import train_model

DATASETS = ("water", "alkanes", "qm7b_T", "gdb13_T")
DESCRIPTORS = ("chemical", "chemical_elec", "chemical_soap", "chemical_elec_soap")


def run_variant(dataset, descriptor, variant, seed, data_dir, max_epochs,
                patience, num_workers):
    torch.manual_seed(seed); np.random.seed(seed)
    train, val, test = load_dataset(dataset, descriptor, data_dir=data_dir)

    max_z = max(int(d["Z"].max()) for d in (train, val, test)) + 1
    use_edge_bias = (variant == "D_gat_edge")
    model = UnifiedModel(
        input_dim=train["desc_dim"], architecture="D_gat_edge",
        edge_feat_dim=train["edge_feat_dim"], use_edge_bias=use_edge_bias,
        max_z=max(20, max_z),
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

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
        "architecture": variant, "seed": seed,
        "test_mae_mHa": out["test_mae_mHa"],
        "test_mae_kcal": out["test_mae_kcal"],
        "best_val_mae_mHa": tr["best_val_mae_mHa"],
        "best_epoch": tr["best_epoch"], "final_epoch": tr["final_epoch"],
        "n_train": len(train["y"]), "desc_dim": train["desc_dim"],
        "n_params": n_params, "element_bias_r2": r2,
        "train_duration_s": tr["duration_s"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--descriptors", nargs="+", default=list(DESCRIPTORS))
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--output", default=f"{C.RESULTS_DIR}/edge_bias_ablation.json")
    p.add_argument("--data-dir", default=C.DATA_DIR)
    p.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=C.ES_PATIENCE)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else []

    cells = [(ds, desc, v) for ds in args.datasets
             for desc in args.descriptors
             for v in ("D_gat_edge", "D_no_edge")]

    for i, (ds, desc, v) in enumerate(cells, 1):
        key = (ds, desc, v, args.seed)
        if any((r["dataset"], r["descriptor"], r["architecture"], r["seed"])
               == key for r in results if r.get("test_mae_mHa") is not None):
            print(f"[{i}/{len(cells)}] skip {key}")
            continue
        print(f"[{i}/{len(cells)}] {ds}/{desc}/{v}")
        t0 = time.time()
        try:
            rec = run_variant(ds, desc, v, args.seed,
                              args.data_dir, args.max_epochs,
                              args.patience, args.num_workers)
            results.append(rec)
            print(f"  test_mae={rec['test_mae_mHa']:.4f} mHa  "
                  f"({time.time() - t0:.0f}s)")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"dataset": ds, "descriptor": desc,
                            "architecture": v, "seed": args.seed,
                            "test_mae_mHa": None, "error": str(e)})
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\nDone. Results: {args.output}")


if __name__ == "__main__":
    main()
