"""DeePHF+SOAP — Streamlit demo

Predicts molecular correlation energy using UnifiedModel checkpoints trained on
MOB-ML. The four alkanes models (D_gat_edge across the four descriptor variants)
visualise the descriptor lift — same architecture, progressively richer features.
The QM7b-T model (A_mlp + chemical_elec) demonstrates the inversion regime,
where the simpler architecture with the simpler descriptor wins on more
chemically diverse data.

Run from the repository root:
    streamlit run app_streamlit/app.py
"""

from __future__ import annotations

import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

# ── Streamlit page config — MUST be the first Streamlit call ──────────────
st.set_page_config(
    page_title="DeePHF+SOAP demo",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Path management ───────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
PROJECT_ROOT = _HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import our inference module (it injects project root again, no-op)
from inference import (
    HARTREE_TO_EV,
    EV_TO_KCAL,
    load_all_checkpoints,
    predict_all,
)

# Optional dependencies — degrade gracefully when missing
try:
    import py3Dmol
    _HAVE_PY3DMOL = True
except ImportError:
    _HAVE_PY3DMOL = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    _HAVE_RDKIT = True
except ImportError:
    _HAVE_RDKIT = False


# ── Constants ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("DeePHF-demo")

CHEM_ACCURACY_KCAL = 1.0
CHEM_ACCURACY_EV = CHEM_ACCURACY_KCAL / EV_TO_KCAL  # ~0.0434 eV

_SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
}
_Z_TO_SYMBOL = {v: k for k, v in _SYMBOL_TO_Z.items()}

_DEFAULT_CKPT_DIR = _HERE / "checkpoints_best"
_DEFAULT_CACHE_DIR = _HERE / "cache_electronic"


# ══════════════════════════════════════════════════════════════════════════
# XYZ parsing
# ══════════════════════════════════════════════════════════════════════════

def parse_xyz(xyz_text: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Parse XYZ text → (Z, coords). Returns None on parse failure."""
    try:
        lines = xyz_text.strip().splitlines()
        if len(lines) < 3:
            return None
        n = int(lines[0].strip())
        body = lines[2:2 + n]
        Z, coords = [], []
        for line in body:
            parts = line.split()
            if len(parts) < 4:
                return None
            sym = parts[0]
            z = _SYMBOL_TO_Z.get(sym)
            if z is None:
                return None
            Z.append(z)
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(Z, dtype=int), np.array(coords, dtype=float)
    except Exception:
        return None


def coords_to_xyz_text(Z: np.ndarray, coords: np.ndarray, comment: str = "") -> str:
    """Inverse of parse_xyz — produce an XYZ-format string."""
    lines = [str(len(Z)), comment]
    for z, (x, y, zc) in zip(Z, coords):
        sym = _Z_TO_SYMBOL.get(int(z), "X")
        lines.append(f"{sym} {x:.10f} {y:.10f} {zc:.10f}")
    return "\n".join(lines)


def smiles_to_xyz(smiles: str) -> Optional[str]:
    """Generate 3D structure from SMILES using RDKit ETKDG + MMFF."""
    if not _HAVE_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    Z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=int)
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return coords_to_xyz_text(Z, coords, comment=f"from SMILES: {smiles}")


# ══════════════════════════════════════════════════════════════════════════
# 3D visualisation
# ══════════════════════════════════════════════════════════════════════════

def render_molecule_3d(xyz_text: str, width: int = 400, height: int = 400) -> Optional[str]:
    """Return an HTML embed of a py3Dmol viewer showing the molecule."""
    if not _HAVE_PY3DMOL:
        return None
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_text, "xyz")
    view.setStyle({}, {"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    view.setBackgroundColor("0xffffff")
    view.zoomTo()
    return view._make_html()


# ══════════════════════════════════════════════════════════════════════════
# Model loading (cached across reruns)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model checkpoints...")
def get_models(ckpt_dir: str):
    return load_all_checkpoints(Path(ckpt_dir))


# ══════════════════════════════════════════════════════════════════════════
# UI: sidebar
# ══════════════════════════════════════════════════════════════════════════

def sidebar() -> dict:
    st.sidebar.title("⚛️ DeePHF+SOAP")
    st.sidebar.markdown(
        "Demo of correlation-energy prediction using `UnifiedModel` checkpoints "
        "from the MOB-ML grid (alkanes + QM7b-T).\n\n"
        "Input a small molecule (XYZ or SMILES) and see predictions from each "
        "trained model with reference Hartree–Fock and MP2 energies."
    )
    st.sidebar.divider()

    ckpt_dir = st.sidebar.text_input("Checkpoint directory", str(_DEFAULT_CKPT_DIR))
    cache_dir = st.sidebar.text_input("HF cache directory", str(_DEFAULT_CACHE_DIR))

    st.sidebar.divider()
    st.sidebar.caption(
        "Note: alkanes-trained models only accept H/C atoms. "
        "QM7b-T model accepts H/C/N/O/S/Cl."
    )
    return {"ckpt_dir": ckpt_dir, "cache_dir": cache_dir}


# ══════════════════════════════════════════════════════════════════════════
# UI: input panel
# ══════════════════════════════════════════════════════════════════════════

def input_panel() -> Optional[str]:
    """Render the input selection UI, return the XYZ text or None."""
    mode = st.radio(
        "Input method",
        ("Upload .xyz file", "Paste XYZ text", "Enter SMILES"),
        horizontal=True,
    )
    xyz_text = None

    if mode == "Upload .xyz file":
        f = st.file_uploader("Drop an .xyz file", type=["xyz"])
        if f is not None:
            xyz_text = f.read().decode("utf-8")
            st.code(xyz_text, language="text")

    elif mode == "Paste XYZ text":
        xyz_text = st.text_area(
            "Paste XYZ content",
            height=160,
            placeholder="3\nWater molecule\nO 0.000 0.000 0.000\nH 0.000 0.757 0.587\nH 0.000 -0.757 0.587",
        )
        if not xyz_text.strip():
            xyz_text = None

    elif mode == "Enter SMILES":
        if not _HAVE_RDKIT:
            st.error("RDKit is not installed; SMILES input unavailable.")
        else:
            col1, col2 = st.columns([3, 1])
            smiles = col1.text_input("SMILES", placeholder="CC for ethane")
            if col2.button("Generate 3D", use_container_width=True) and smiles:
                xyz_text = smiles_to_xyz(smiles)
                if xyz_text is None:
                    st.error("RDKit could not embed this SMILES into 3D.")
                else:
                    st.code(xyz_text, language="text")
                    st.session_state["last_xyz"] = xyz_text
            # Keep the last generated structure between reruns
            xyz_text = xyz_text or st.session_state.get("last_xyz")

    return xyz_text


# ══════════════════════════════════════════════════════════════════════════
# UI: results display
# ══════════════════════════════════════════════════════════════════════════

def display_results(result: dict, xyz_text: Optional[str] = None) -> None:
    # CCSD(T)/cc-pVTZ ground-truth lookup. Matches user's molecule against the
    # training set via canonical SMILES; when matched, we know the true label
    # the models were trained against, so chemical-accuracy badges become
    # meaningful (the previous "Δ vs MP2" badge was structurally biased by the
    # method gap MP2/cc-pVDZ → CCSD(T)/cc-pVTZ, ~20–100 kcal/mol on organics).
    try:
        from groundtruth import lookup_gt
        gt = lookup_gt(xyz=xyz_text) if xyz_text else None
    except Exception as e:
        logger.warning(f"Ground-truth lookup unavailable: {e}")
        gt = None

    if gt is not None:
        st.subheader("Reference: CCSD(T)/cc-pVTZ (training-set lookup)")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "E_corr (true)",
            f"{gt['mean_kcal']:.2f} kcal/mol",
            help=(
                f"Mean over {gt['n']} training-set conformers "
                f"(σ = {gt['std_kcal']:.3f} kcal/mol). "
                f"SMILES matched against: {', '.join(gt['datasets'])}."
            ),
        )
        c2.metric("σ across conformers", f"{gt['std_kcal']:.3f} kcal/mol")
        c3.metric("n conformers", str(gt['n']))
        if result.get("E_corr_MP2_kcal") is not None:
            gap = gt["mean_kcal"] - result["E_corr_MP2_kcal"]
            st.caption(
                f"PySCF reference (MP2/cc-pVDZ): E_corr = {result['E_corr_MP2_kcal']:.2f} kcal/mol. "
                f"Method-gap to CCSD(T)/cc-pVTZ: {gap:+.2f} kcal/mol (expected — "
                "models target CCSD(T)/cc-pVTZ, MP2/cc-pVDZ recovers less correlation)."
            )
    else:
        st.subheader("Reference: PySCF MP2/cc-pVDZ (no training-set match)")
        c1, c2, _ = st.columns(3)
        if result.get("E_HF_Ha") is not None:
            c1.metric("E_HF", f"{result['E_HF_Ha']:.6f} Ha", help="Hartree–Fock total energy")
        if result.get("E_corr_MP2_Ha") is not None:
            c2.metric(
                "E_corr (MP2)",
                f"{result['E_corr_MP2_Ha']:.6f} Ha",
                f"{result['E_corr_MP2_kcal']:.2f} kcal/mol",
            )
        st.caption(
            "This molecule's SMILES isn't in the training set, so no CCSD(T)/cc-pVTZ "
            "ground truth available. Model predictions target CCSD(T)/cc-pVTZ — expect "
            "a method gap of 20–100 kcal/mol from the MP2/cc-pVDZ reference above. "
            "No chemical-accuracy badge shown in this mode."
        )

    st.divider()
    st.subheader("Predictions")

    preds = result["predictions"]
    if not preds:
        st.warning("No model predictions produced.")
        return

    rows = []
    for name, p in preds.items():
        if "error" in p:
            rows.append({"name": name, "error": p["error"]})
            continue
        delta_vs_gt = (p["e_corr_kcal"] - gt["mean_kcal"]) if gt is not None else None
        delta_vs_mp2 = (
            p["e_corr_kcal"] - result["E_corr_MP2_kcal"]
            if result.get("E_corr_MP2_kcal") is not None else None
        )
        rows.append({
            "name":          name,
            "dataset":       p["dataset"],
            "descriptor":    p["descriptor"],
            "architecture":  p["architecture"],
            "e_corr_kcal":   p["e_corr_kcal"],
            "e_corr_eV":     p["e_corr_eV"],
            "delta_vs_gt":   delta_vs_gt,
            "delta_vs_mp2":  delta_vs_mp2,
            "test_mae_mHa":  p["test_mae_mHa"],
        })

    valid = [r for r in rows if "error" not in r]
    if valid:
        for r in valid:
            with st.container(border=True):
                # 4 columns (dropped redundant eV) to give kcal value room to render
                cols = st.columns([3, 2, 2, 2])
                cols[0].markdown(
                    f"**{r['descriptor']}** — `{r['architecture']}` ({r['dataset']})\n\n"
                    f"<small>Test MAE on holdout: {r['test_mae_mHa']:.3f} mHa</small>",
                    unsafe_allow_html=True,
                )
                cols[1].metric("E_corr (pred)", f"{r['e_corr_kcal']:.2f} kcal/mol")

                if r["delta_vs_gt"] is not None:
                    # GT mode: compare to true CCSD(T)/cc-pVTZ label, show CA badge
                    sign = "+" if r["delta_vs_gt"] >= 0 else ""
                    cols[2].metric(
                        "Δ vs CCSD(T)",
                        f"{sign}{r['delta_vs_gt']:.3f} kcal/mol",
                        delta_color="off",
                        help="Model prediction minus true CCSD(T)/cc-pVTZ label.",
                    )
                    within_ca = abs(r["delta_vs_gt"]) < CHEM_ACCURACY_KCAL
                    cols[3].markdown(
                        "✅ within CA" if within_ca else "❌ above CA",
                        help="|Δ| < 1 kcal/mol vs true CCSD(T)/cc-pVTZ.",
                    )
                elif r["delta_vs_mp2"] is not None:
                    # Fallback: show method gap to MP2, no CA judgment
                    sign = "+" if r["delta_vs_mp2"] >= 0 else ""
                    cols[2].metric(
                        "vs MP2 (method gap)",
                        f"{sign}{r['delta_vs_mp2']:.2f} kcal/mol",
                        delta_color="off",
                        help=(
                            "Method gap: model targets CCSD(T)/cc-pVTZ, reference is MP2/cc-pVDZ. "
                            "20–100 kcal/mol gap is expected, not a model error."
                        ),
                    )
                    cols[3].markdown(
                        "<small>ℹ️ no GT — CA undecidable</small>",
                        unsafe_allow_html=True,
                    )

    errors = [r for r in rows if "error" in r]
    if errors:
        with st.expander(f"{len(errors)} model(s) skipped", expanded=False):
            for r in errors:
                st.warning(f"**{r['name']}**: {r['error']}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    settings = sidebar()

    st.title("DeePHF+SOAP — correlation-energy demo")
    st.markdown(
        "Each row below shows one trained `UnifiedModel`. The four alkanes models "
        "demonstrate the descriptor lift; the QM7b-T model demonstrates the inversion."
    )

    # Load models once (cached)
    models = get_models(settings["ckpt_dir"])
    if not models:
        st.error(
            f"No checkpoints found in `{settings['ckpt_dir']}`. "
            "Run `sbatch slurm/train_app_models.sbatch` from the repo root first."
        )
        with st.expander("Checkpoint summary"):
            st.write({})
        return

    with st.expander(f"Loaded {len(models)} checkpoint(s)", expanded=False):
        rows = []
        for name, (_, ckpt) in models.items():
            cfg = ckpt["config"]
            rows.append({
                "name":         name,
                "dataset":      cfg["dataset"],
                "descriptor":   cfg["descriptor"],
                "architecture": cfg["architecture"],
                "test_mae_mHa": round(ckpt["test_mae_mHa"], 4),
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

    # Input + 3D viewer
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Input")
        xyz_text = input_panel()

    with right:
        st.subheader("3D preview")
        if xyz_text:
            html = render_molecule_3d(xyz_text)
            if html:
                st.components.v1.html(html, height=420)
            else:
                st.info("Install `py3Dmol` (`pip install py3Dmol`) to enable 3D preview.")
        else:
            st.info("Provide a molecule on the left to see its 3D structure here.")

    # Run prediction
    if xyz_text and st.button("Run prediction", type="primary", use_container_width=True):
        parsed = parse_xyz(xyz_text)
        if parsed is None:
            st.error("Could not parse XYZ. Check the format.")
            return
        Z, coords = parsed
        with st.spinner("Running HF + descriptor pipeline + UnifiedModel inference..."):
            result = predict_all(Z, coords, models, Path(settings["cache_dir"]))
        if not result.get("success"):
            st.error(result.get("error", "Prediction failed for an unknown reason."))
            return
        display_results(result, xyz_text=xyz_text)


if __name__ == "__main__":
    main()