"""Decompose the edge-bias mechanism by masking either the geometric (RBF)
or the chemical (bond-type + bond length) channel of the 21-dim edge
feature.

The same D_gat_edge architecture is used in all three runs; only the
post-build edge feature is masked. Trained on alkanes/chemical_elec_soap
(cleanest signal, fastest dataset).
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
from deephf.model import UnifiedModel, EdgeMaskedModel
from deephf.train import train_model


def make_variant(base, variant):
    if variant == "D_gat_edge":
        return base
    if variant == "D_no_rbf":
        return EdgeMaskedModel(base, drop="rbf")
    if variant == "D_no_bond":
        return EdgeMaskedModel(base, drop="bond")
    raise ValueError(variant)


def run_variant(dataset, descriptor, variant, seed, data_dir, max_epochs,
                patience, num_workers):
    torch.manual_seed(seed); np.random.seed(seed)
    train, val, test = load_dataset(dataset, descriptor, data_dir=data_dir)
    max_z = max(int(d["Z"].max()) for d in (train, val, test)) + 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = UnifiedModel(
        input_dim=train["desc_dim"], architecture="D_gat_edge",
        edge_feat_dim=train["edge_feat_dim"],
        max_z=max(20, max_z),
    ).to(device)

    Xt = torch.from_numpy(train["X"]).to(device)
    Zt = torch.from_numpy(train["Z"]).long().to(device)
    yt = torch.from_numpy(train["y"]).to(device)
    mt = Zt > 0
    base.init_normalization(Xt, mt)
    r2 = base.init_element_bias(Zt, yt, mt)

    model = make_variant(base, variant).to(device)
    tr_l, va_l, te_l = make_loaders(train, val, test,
                                    num_workers=num_workers,
                                    pin=(device != "cpu"))
    tr = train_model(model, tr_l, va_l, device,
                     max_epochs=max_epochs, patience=patience)
    out = evaluate(model, te_l, device)
    return {
        "dataset": dataset, "descriptor": descriptor,
        "architecture": variant, "seed": seed,
        "test_mae_mHa": out["test_mae_mHa"],
        "test_mae_kcal": out["test_mae_kcal"],
        "best_val_mae_mHa": tr["best_val_mae_mHa"],
        "best_epoch": tr["best_epoch"], "final_epoch": tr["final_epoch"],
        "n_train": len(train["y"]), "desc_dim": train["desc_dim"],
        "element_bias_r2": r2,
        "train_duration_s": tr["duration_s"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="alkanes")
    p.add_argument("--descriptor", default="chemical_elec_soap")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--variants", nargs="+",
                   default=["D_gat_edge", "D_no_rbf", "D_no_bond"])
    p.add_argument("--output", default=f"{C.RESULTS_DIR}/edge_features_ablation.json")
    p.add_argument("--data-dir", default=C.DATA_DIR)
    p.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=C.ES_PATIENCE)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else []

    for v in args.variants:
        print(f"\n[{args.dataset}/{args.descriptor}/{v}/seed{args.seed}]")
        t0 = time.time()
        rec = run_variant(args.dataset, args.descriptor, v, args.seed,
                          args.data_dir, args.max_epochs, args.patience,
                          args.num_workers)
        results.append(rec)
        print(f"  test_mae={rec['test_mae_mHa']:.4f} mHa  ({time.time() - t0:.0f}s)")
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\nDone. Results: {args.output}")


if __name__ == "__main__":
    main()
