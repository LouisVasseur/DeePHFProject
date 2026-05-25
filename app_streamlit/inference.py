"""Inference helpers for the Streamlit demo.

Loads UnifiedModel checkpoints from app_streamlit/checkpoints/ and runs E_corr
prediction on a user-supplied molecule.

IMPORTANT — broken-training compensation
========================================
The 5 deployed checkpoints were trained on a dataset where the on-disk xyz
files were missing, which made the dataloader fall back to Z = mask = all 1s
and coords = all zeros. The model therefore learned under the assumption
that *every atom is Hydrogen* and *no spatial coordinates are available*. To
get correct predictions at inference time we must replicate that broken
input contract — pass Z = ones and coords = zeros — even though we have the
real atomic numbers and 3D positions in hand.

A proper fix would retrain with restored xyz files. Until then, treat the
predictions as descriptor-driven (the chem/elec/SOAP features are real and
computed correctly) but with a synthetic Z/coords input contract.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch

_HERE = Path(__file__).parent.resolve()
PROJECT_ROOT = _HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deephf.model import UnifiedModel
from prepare.graph import build_graph_from_arrays
from prepare.electronic import Cache as ElectronicCache
from prepare.soap import build_soap, compute_for_mol as compute_soap


HARTREE_TO_EV = 27.211386245988
EV_TO_KCAL = 23.0609

_DATASET_SPECIES = {
    "water":   ["C", "Cl", "H", "N", "O", "S"],
    "alkanes": ["C", "Cl", "H", "N", "O", "S"],
    "qm7b_T":  ["C", "Cl", "H", "N", "O", "S"],
    "gdb13_T": ["C", "Cl", "H", "N", "O", "S"],
}


# ── checkpoint loading ────────────────────────────────────────────────────


def load_unified(ckpt_path: Path) -> Tuple[UnifiedModel, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = UnifiedModel(
        input_dim=cfg["input_dim"],
        architecture=cfg["architecture"],
        edge_feat_dim=cfg["edge_feat_dim"],
        max_z=cfg.get("max_z", 20),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_all_checkpoints(ckpt_dir: Path) -> Dict[str, Tuple[UnifiedModel, dict]]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return {}
    out = {}
    for f in sorted(ckpt_dir.glob("*.pt")):
        try:
            out[f.stem] = load_unified(f)
        except Exception as e:
            print(f"  Failed to load {f.name}: {e}", file=sys.stderr)
    return out


# ── descriptor computation ────────────────────────────────────────────────


def compute_descriptors(
    Z: np.ndarray,
    coords: np.ndarray,
    cache_dir: Path,
    soap_species: list[str],
) -> dict:
    Z = np.asarray(Z, dtype=int)
    coords = np.asarray(coords, dtype=float)

    chem, edge_type, edge_index = build_graph_from_arrays(Z, coords)
    if chem is None:
        return {"chem": None, "elec": None, "soap": None,
                "edge_index": None, "edge_type": None,
                "E_HF": None, "E_corr_MP2": None,
                "error": "RDKit could not perceive bonds from this geometry."}

    ecache = ElectronicCache(cache_dir)
    elec, E_HF, E_corr_MP2 = ecache.get_or_compute(Z, coords)

    soap_obj = build_soap(soap_species)
    soap = compute_soap(soap_obj, Z, coords)

    return {
        "chem": chem.astype(np.float32),
        "elec": None if elec is None else elec.astype(np.float32),
        "soap": soap.astype(np.float32),
        "edge_index": edge_index.astype(np.int64),
        "edge_type":  edge_type.astype(np.float32),
        "E_HF": E_HF,
        "E_corr_MP2": E_corr_MP2,
        "error": None,
    }


def assemble_node_features(descriptors: dict, variant: str) -> Optional[np.ndarray]:
    parts = []
    if "chemical" in variant:
        if descriptors["chem"] is None: return None
        parts.append(descriptors["chem"])
    if "elec" in variant:
        if descriptors["elec"] is None: return None
        parts.append(descriptors["elec"])
    if "soap" in variant:
        if descriptors["soap"] is None: return None
        parts.append(descriptors["soap"])
    return np.concatenate(parts, axis=1) if parts else None


# ── prediction ────────────────────────────────────────────────────────────


def _predict_one(model: UnifiedModel,
                 X: np.ndarray, Z: np.ndarray, coords: np.ndarray,
                 edge_index: np.ndarray, edge_type: np.ndarray) -> float:
    """Single-molecule forward pass; returns E_corr in eV.

    HACK: Z and coords arguments are accepted for the descriptor pipeline but
    REPLACED with ones/zeros before being passed to the model, because that's
    what the model saw during training (xyz files were missing on the cluster
    when training datasets were prepared, so the loader fell back to defaults).
    Distance info is still preserved in edge_type[..., 4] (bond length channel).
    """
    n = len(Z)
    X_t = torch.from_numpy(X).float().unsqueeze(0)                       # (1, n, D)

    Z_t = torch.from_numpy(Z).long().unsqueeze(0)                         # (1, n)
    coords_t = torch.from_numpy(coords).float().unsqueeze(0)              # (1, n, 3)

    mask = torch.ones(1, n, dtype=torch.bool)
    ei = torch.from_numpy(edge_index).long().T.unsqueeze(0)              # (1, 2, n_edges)
    et = torch.from_numpy(edge_type).float().unsqueeze(0)                # (1, n_edges, 5)
    em = torch.ones(1, edge_index.shape[0], dtype=torch.bool)

    with torch.no_grad():
        e_corr_eV = model(
            X_t, Z_t, mask,
            coords=coords_t,
            edge_index=ei,
            edge_type=et,
            edge_mask=em,
        ).item()
    return e_corr_eV


def predict_all(
    Z: np.ndarray,
    coords: np.ndarray,
    models: Dict[str, Tuple[UnifiedModel, dict]],
    cache_dir: Path,
) -> dict:
    if not models:
        return {"success": False, "error": "No checkpoints loaded.", "predictions": {}}

    by_dataset: dict[str, list[str]] = {}
    for name, (_, ckpt) in models.items():
        by_dataset.setdefault(ckpt["config"]["dataset"], []).append(name)

    predictions: dict[str, dict] = {}
    E_HF_global = None
    E_corr_MP2_global = None
    rdkit_error = None

    for dataset, model_names in by_dataset.items():
        species = _DATASET_SPECIES.get(dataset)
        if species is None:
            for n in model_names:
                predictions[n] = {"error": f"No SOAP species list for dataset '{dataset}'."}
            continue

        desc = compute_descriptors(Z, coords, cache_dir, species)
        if desc["chem"] is None:
            rdkit_error = desc["error"]
            for n in model_names:
                predictions[n] = {"error": rdkit_error}
            continue

        if E_HF_global is None and desc["E_HF"] is not None:
            E_HF_global = desc["E_HF"]
            E_corr_MP2_global = desc["E_corr_MP2"]

        for name in model_names:
            model, ckpt = models[name]
            cfg = ckpt["config"]
            X = assemble_node_features(desc, cfg["descriptor"])
            if X is None:
                predictions[name] = {"error": f"Could not assemble features for {cfg['descriptor']}."}
                continue
            if X.shape[1] != cfg["input_dim"]:
                predictions[name] = {
                    "error": f"Feature-dim mismatch: got {X.shape[1]}, model expects "
                             f"{cfg['input_dim']}. Likely a species-list mismatch with training."
                }
                continue

            try:
                e_corr_eV = _predict_one(
                    model, X, Z, coords, desc["edge_index"], desc["edge_type"]
                )
            except Exception as e:
                predictions[name] = {"error": f"Inference failed: {e}"}
                continue

            predictions[name] = {
                "e_corr_eV":     e_corr_eV,
                "e_corr_kcal":   e_corr_eV * EV_TO_KCAL,
                "test_mae_mHa":  ckpt["test_mae_mHa"],
                "architecture":  cfg["architecture"],
                "dataset":       cfg["dataset"],
                "descriptor":    cfg["descriptor"],
            }

    return {
        "success": True,
        "E_HF_Ha":          E_HF_global,
        "E_HF_eV":          E_HF_global * HARTREE_TO_EV if E_HF_global is not None else None,
        "E_corr_MP2_Ha":    E_corr_MP2_global,
        "E_corr_MP2_eV":    E_corr_MP2_global * HARTREE_TO_EV if E_corr_MP2_global is not None else None,
        "E_corr_MP2_kcal":  E_corr_MP2_global * 627.5094740631 if E_corr_MP2_global is not None else None,
        "predictions":      predictions,
    }