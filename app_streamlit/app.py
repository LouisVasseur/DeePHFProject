"""DeePHF+SOAP — Streamlit Interface (full version)

Vera Dias Gomes · Cédric Nathanaël Rossboth · Octavian Susanu · Louis James Vasseur
Team 3 · AI for Chemistry (CH-457) · EPFL

Predicts molecular correlation energy using UnifiedModel checkpoints trained
on MOB-ML CCSD(T)/cc-pVTZ data. Features:

  • Four input modes (xyz upload / SMILES / paste / draw via Ketcher)
  • Live 3D molecule preview with optional HF molecular orbital overlay
    (HOMO-1, HOMO, LUMO, LUMO+1 cube isosurfaces)
  • CCSD(T)/cc-pVTZ ground-truth lookup from training-set SMILES
  • Reference HF + MP2 energies via PySCF
  • Per-checkpoint predictions with chemical-accuracy badge
  • Side-by-side comparison bar chart
"""

from __future__ import annotations

import hashlib
import io
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

# ── Page config (must be the first Streamlit call) ────────────────────────
st.set_page_config(
    page_title="DeePHF+SOAP",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Path management ───────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
PROJECT_ROOT = _HERE.parent
for _p in [str(_HERE), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference import HARTREE_TO_EV, EV_TO_KCAL, load_all_checkpoints, predict_all
from orbitals import (
    compute_hf_orbitals,
    is_orbital_cached,
    save_hf_orbitals_to_cache,
    render_molecule_3d,
    render_molecule_with_orbital,
)

# Optional dependencies
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

try:
    from streamlit_ketcher import st_ketcher
    _HAVE_KETCHER = True
    _KETCHER_ERR = None
except Exception as _e:
    st_ketcher = None
    _HAVE_KETCHER = False
    _KETCHER_ERR = str(_e)


# ── Constants ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("DeePHF-demo")

CHEM_ACCURACY_KCAL = 1.0
CHEM_ACCURACY_EV = CHEM_ACCURACY_KCAL / EV_TO_KCAL

_SYMBOL_TO_Z = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
}
_Z_TO_SYMBOL = {v: k for k, v in _SYMBOL_TO_Z.items()}

_DEFAULT_CKPT_DIR = _HERE / "checkpoints_best"
_DEFAULT_CACHE_DIR = _HERE / "cache_electronic"
_DEFAULT_ORB_CACHE_DIR = _HERE / "cache_orbitals"


# ══════════════════════════════════════════════════════════════════════════
# XYZ parsing + helpers
# ══════════════════════════════════════════════════════════════════════════

def parse_xyz(xyz_text: str):
    """Parse XYZ text → (Z, coords, n_atoms, formula). Raises ValueError."""
    lines = xyz_text.strip().splitlines()
    if len(lines) < 3:
        raise ValueError("XYZ requires at least 3 lines.")
    n = int(lines[0].strip())
    body = lines[2:2 + n]
    Z, coords = [], []
    for line in body:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad atom line: {line!r}")
        sym = parts[0]
        z = _SYMBOL_TO_Z.get(sym)
        if z is None:
            raise ValueError(f"Unknown element: {sym}")
        Z.append(z)
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    Z_arr = np.array(Z, dtype=int)
    coords_arr = np.array(coords, dtype=float)
    formula = _build_formula(Z_arr)
    return Z_arr, coords_arr, n, formula


def _build_formula(Z: np.ndarray) -> str:
    """Hill notation: C first, H second, rest alphabetical."""
    counts: dict[str, int] = {}
    for z in Z:
        sym = _Z_TO_SYMBOL.get(int(z), f"Z{z}")
        counts[sym] = counts.get(sym, 0) + 1
    ordered: list[tuple[str, int]] = []
    if "C" in counts:
        ordered.append(("C", counts.pop("C")))
    if "H" in counts:
        ordered.append(("H", counts.pop("H")))
    for sym in sorted(counts):
        ordered.append((sym, counts[sym]))
    return "".join(f"{s}{c if c > 1 else ''}" for s, c in ordered)


def coords_to_xyz_text(Z: np.ndarray, coords: np.ndarray, comment: str = "") -> str:
    lines = [str(len(Z)), comment]
    for z, (x, y, zc) in zip(Z, coords):
        sym = _Z_TO_SYMBOL.get(int(z), "X")
        lines.append(f"{sym} {x:.10f} {y:.10f} {zc:.10f}")
    return "\n".join(lines)


def smiles_to_xyz(smiles: str) -> Optional[str]:
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
# Cached model loading + cached orbital generation
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model checkpoints…")
def get_models(ckpt_dir: str):
    return load_all_checkpoints(Path(ckpt_dir))


@st.cache_data(show_spinner=False)
def cached_hf_orbitals(_an_bytes: bytes, _co_bytes: bytes, basis: str, orb_cache_dir: str):
    """Streamlit-cached wrapper around orbitals.compute_hf_orbitals."""
    Z = np.frombuffer(_an_bytes, dtype=int)
    coords = np.frombuffer(_co_bytes, dtype=float).reshape(-1, 3)
    return compute_hf_orbitals(Z, coords, basis, orb_cache_dir)


# ══════════════════════════════════════════════════════════════════════════
# Custom CSS — EPFL white/red palette
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=JetBrains+Mono:wght@300;400;500&display=swap');

  html, body { background-color: #ffffff !important; color: #1a1a1a !important; }
  .stApp { background-color: #ffffff !important; }

  .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th,
  div[data-testid="stMarkdownContainer"] p,
  div[data-testid="stMarkdownContainer"] li { color: #1a1a1a !important; }

  h1, h2, h3, h4, h5, h6 { color: #1a1a1a !important; }

  .stTextInput label, .stSelectbox label, .stSlider label,
  .stFileUploader label, .stTextArea label, .stRadio label { color: #1a1a1a !important; }
  .stTextInput input, .stTextArea textarea { color: #1a1a1a !important; background: #ffffff !important; }

  section[data-testid="stSidebar"] { background-color: #f9f9f9 !important; border-right: 1px solid #e8e8e8; }
  section[data-testid="stSidebar"] * { color: #1a1a1a !important; }

  details summary { color: #1a1a1a !important; }
  details > div { color: #1a1a1a !important; }
  code, pre { color: #1a1a1a !important; background: #f4f4f4 !important; }
  table { color: #1a1a1a !important; }
  th { color: #1a1a1a !important; background: #f4f4f4 !important; }
  td { color: #1a1a1a !important; }
  .stRadio > div label { color: #1a1a1a !important; }
  .stSpinner p { color: #555555 !important; }

  html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }

  :root {
    --epfl-red:  #ff0000;
    --epfl-dark: #1a1a1a;
    --epfl-mid:  #555555;
    --epfl-dim:  #888888;
    --epfl-line: #e0e0e0;
    --epfl-bg:   #f7f7f7;
    --mono:      'JetBrains Mono', monospace;
  }

  .deephf-header {
    text-align: center;
    padding: 1.8rem 0 1.0rem;
    border-bottom: 2px solid var(--epfl-red);
    margin-bottom: 1.5rem;
  }
  .deephf-header h1 {
    font-size: 1.85rem; font-weight: 600;
    color: var(--epfl-dark); letter-spacing: -0.5px;
    margin-bottom: 0.15rem;
  }
  .deephf-header .subtitle {
    font-family: var(--mono); font-size: 0.68rem;
    letter-spacing: 2px; text-transform: uppercase;
    color: var(--epfl-red); margin-bottom: 0.8rem;
  }
  .deephf-header .authors {
    font-size: 0.88rem; color: var(--epfl-mid);
    line-height: 1.8; font-style: italic;
  }
  .deephf-header .course {
    font-family: var(--mono); font-size: 0.65rem;
    letter-spacing: 1px; color: var(--epfl-dim);
    font-style: normal; margin-top: 0.15rem;
  }

  .section-label {
    font-family: var(--mono); font-size: 0.62rem;
    letter-spacing: 2px; text-transform: uppercase;
    color: var(--epfl-dim);
    margin: 1.4rem 0 0.6rem;
    border-left: 3px solid var(--epfl-red);
    padding-left: 0.5rem;
  }

  .energy-card {
    background: #ffffff;
    border: 1px solid var(--epfl-line);
    border-top: 3px solid var(--epfl-line);
    border-radius: 4px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.8rem;
  }
  .energy-card .model-label {
    font-family: var(--mono); font-size: 0.68rem;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-bottom: 0.35rem; color: var(--epfl-dim);
  }
  .energy-card .value-main {
    font-size: 1.45rem; font-weight: 600;
    color: var(--epfl-dark); letter-spacing: -0.3px;
  }
  .energy-card .value-sub {
    font-family: var(--mono); font-size: 0.82rem;
    color: var(--epfl-mid); margin-top: 0.1rem;
  }
  .energy-card .delta {
    font-family: var(--mono); font-size: 0.77rem;
    margin-top: 0.5rem; padding: 0.18rem 0.5rem;
    border-radius: 3px; display: inline-block;
  }
  .delta-good { background: #fff0f0; color: #8b0000; border: 1px solid #ffcccc; }
  .delta-bad  { background: #f5f5f5; color: #555555; border: 1px solid #dddddd; }
  .delta-na   { background: #f5f5f5; color: var(--epfl-dim); border: 1px solid #e0e0e0; }
  .card-footer { font-size: 0.73rem; color: #aaa; margin-top: 0.4rem; }
  .not-loaded {
    font-family: var(--mono); font-size: 0.8rem;
    color: var(--epfl-dim); font-style: italic;
  }

  .col-alkanes { color: #c0392b; }
  .col-water   { color: #2980b9; }
  .col-qm7b_T  { color: #8e44ad; }
  .col-gdb13_T { color: #27ae60; }
  .col-ref     { color: #888888; }

  .status-pill {
    display: inline-block; padding: 0.12rem 0.55rem;
    border-radius: 2px; font-family: var(--mono);
    font-size: 0.62rem; letter-spacing: 0.5px;
  }
  .pill-ok  { background: #fff0f0; color: #c0392b; border: 1px solid #ffbbbb; }
  .pill-off { background: #f5f5f5; color: #999999; border: 1px solid #e0e0e0; }

  .info-box {
    background: #fafafa;
    border: 1px solid var(--epfl-line);
    border-left: 3px solid var(--epfl-red);
    border-radius: 3px;
    padding: 0.9rem 1.1rem;
    font-size: 0.87rem; color: var(--epfl-dark);
    line-height: 1.7; margin: 0.8rem 0 1.2rem;
  }
  .info-box code {
    font-family: var(--mono); font-size: 0.82rem;
    background: #f0f0f0; padding: 0.05rem 0.3rem; border-radius: 2px;
  }

  .viewer-caption {
    font-family: var(--mono); font-size: 0.65rem;
    letter-spacing: 1px; text-transform: uppercase;
    color: var(--epfl-dim); text-align: center;
    margin-top: 0.3rem;
  }
  .viewer-empty {
    height: 460px;
    border: 1px dashed var(--epfl-line);
    border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: 0.78rem;
    color: var(--epfl-dim); background: #fafafa;
    text-align: center; padding: 1rem;
  }

  .citation-box {
    margin-top: 2.5rem; padding-top: 1rem;
    border-top: 1px solid var(--epfl-line);
    font-size: 0.78rem; color: var(--epfl-dim);
    line-height: 1.7; font-style: italic;
  }
  .citation-box strong { font-style: normal; color: var(--epfl-mid); }

  div[data-testid="stButton"] button[kind="primary"] {
    background: var(--epfl-red) !important; border: none !important;
  }
  div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #cc0000 !important;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")

    st.markdown("**Checkpoints**")
    ckpt_dir = st.text_input(
        "Checkpoint directory",
        value=str(_DEFAULT_CKPT_DIR),
        help="Each .pt file is loaded as one UnifiedModel checkpoint.",
    )

    st.markdown("---")
    st.markdown("**HF / descriptor settings**")
    hf_basis = st.selectbox(
        "HF basis set",
        ["cc-pvdz", "cc-pvtz", "sto-3g", "6-31g", "6-31g*"],
        index=0,
        help="Basis for PySCF Hartree-Fock (reference MP2 + HF orbital cubes).",
    )

    st.markdown("---")
    st.markdown("**Caches**")
    cache_dir = st.text_input(
        "Descriptor cache",
        value=str(_DEFAULT_CACHE_DIR),
        help="HF electronic descriptors cached as .npz keyed by (geometry, basis).",
    )
    orb_cache_dir = st.text_input(
        "Orbital cube cache",
        value=str(_DEFAULT_ORB_CACHE_DIR),
        help="HF MO cubes cached as .npz (~200-500 KB per molecule).",
    )
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(orb_cache_dir).mkdir(parents=True, exist_ok=True)

    st.markdown("---")
    st.markdown(
        "<span style='font-family:monospace;font-size:0.68rem;color:#aaaaaa'>"
        "DeePHFProject · CH-457 · EPFL · 2025</span>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="deephf-header">
  <h1>DeePHF+SOAP</h1>
  <div class="subtitle">Correlation Energy Predictor</div>
  <div class="authors">
    Vera Dias Gomes &middot; Cédric Nathanaël Rossboth &middot; Octavian Susanu &middot; Louis James Vasseur<br>
    <span class="course">Team 3 &middot; AI for Chemistry (CH-457) &middot; EPFL</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────
models = get_models(ckpt_dir)

if not models:
    st.markdown(
        f'<div class="info-box">No checkpoints found in <code>{ckpt_dir}</code>. '
        f'Place trained <code>.pt</code> files in that directory.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ── Status row: loaded checkpoints by dataset ────────────────────────────
by_dataset: dict[str, list[str]] = {}
for name, (_, ckpt) in models.items():
    by_dataset.setdefault(ckpt["config"]["dataset"], []).append(name)

datasets = sorted(by_dataset)
cols_status = st.columns(len(datasets))
for col, ds in zip(cols_status, datasets):
    n_in_ds = len(by_dataset[ds])
    col.markdown(
        f'<div style="text-align:center;margin-bottom:0.5rem">'
        f'<span class="col-{ds}" style="font-family:monospace;font-size:0.72rem;'
        f'font-weight:600;text-transform:uppercase;letter-spacing:0.5px">{ds}</span><br>'
        f'<span class="status-pill pill-ok">✓ {n_in_ds} loaded</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# Persistent session state
# ══════════════════════════════════════════════════════════════════════════

for _k in ("xyz_text", "mol_label", "last_result", "show_orbitals",
           "orbital_choice", "orbital_isovalue", "last_drawn_smiles"):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ══════════════════════════════════════════════════════════════════════════
# Input + 3D viewer (two-column)
# ══════════════════════════════════════════════════════════════════════════

col_input, col_viewer = st.columns([1, 1], gap="large")

# ── LEFT: input picker ───────────────────────────────────────────────────
with col_input:
    st.markdown('<div class="section-label">Input molecule</div>', unsafe_allow_html=True)

    input_modes = ["Upload .xyz file", "Enter SMILES", "Paste XYZ text", "Draw structure"]
    input_mode = st.radio(
        "Input method",
        input_modes,
        horizontal=True,
        label_visibility="collapsed",
    )

    xyz_text = None
    mol_label = None

    if input_mode == "Upload .xyz file":
        uploaded = st.file_uploader(
            "Drop a .xyz file here",
            type=["xyz"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            try:
                xyz_text = uploaded.read().decode("utf-8")
                mol_label = uploaded.name
                if st.session_state.get("xyz_text") != xyz_text:
                    st.session_state["last_result"] = None
                st.session_state["xyz_text"] = xyz_text
                st.session_state["mol_label"] = mol_label
            except Exception:
                st.error("Could not decode the uploaded file.")

    elif input_mode == "Enter SMILES":
        smiles_input = st.text_input(
            "SMILES string",
            placeholder="e.g. O   or   CC   or   c1ccccc1",
            label_visibility="collapsed",
        )
        if smiles_input.strip():
            if st.button("Generate 3D structure", type="primary", key="smiles_gen"):
                with st.spinner("Generating 3D geometry via RDKit ETKDG + MMFF…"):
                    try:
                        xyz_text = smiles_to_xyz(smiles_input.strip())
                        if xyz_text is None:
                            st.error("RDKit could not embed this SMILES into 3D.")
                        else:
                            mol_label = smiles_input.strip()
                            st.session_state["xyz_text"] = xyz_text
                            st.session_state["mol_label"] = mol_label
                            st.session_state["last_result"] = None
                    except Exception as e:
                        st.error(str(e))

    elif input_mode == "Paste XYZ text":
        xyz_input = st.text_area(
            "Paste XYZ content",
            height=200,
            placeholder="3\nWater\nO  0.000  0.000  0.000\nH  0.000  0.757  0.587\nH  0.000 -0.757  0.587",
            label_visibility="collapsed",
        )
        if xyz_input.strip():
            xyz_text = xyz_input.strip()
            mol_label = "pasted molecule"
            if st.session_state.get("xyz_text") != xyz_text:
                st.session_state["last_result"] = None
            st.session_state["xyz_text"] = xyz_text
            st.session_state["mol_label"] = mol_label

    else:  # Draw structure
        if not _HAVE_KETCHER:
            st.warning(
                "`streamlit-ketcher` not installed.\n\n"
                "Install with: `pip install streamlit-ketcher`\n\n"
                f"Underlying error: `{_KETCHER_ERR}`"
            )
        else:
            st.markdown(
                "<span style='font-size:0.82rem;color:#555'>"
                "Draw a structure below, then click <b>Apply</b> in the editor "
                "to capture the SMILES.</span>",
                unsafe_allow_html=True,
            )
            drawn_smiles = st_ketcher("", height=420, key="ketcher_editor")
            if drawn_smiles and drawn_smiles.strip():
                st.markdown(f"**Drawn SMILES:** `{drawn_smiles}`")
                last_drawn = st.session_state.get("last_drawn_smiles")
                regen = st.button(
                    "Generate 3D structure from drawing",
                    type="primary",
                    key="ketcher_gen",
                )
                if regen or last_drawn != drawn_smiles:
                    with st.spinner("Generating 3D geometry…"):
                        try:
                            xyz_text = smiles_to_xyz(drawn_smiles.strip())
                            if xyz_text is None:
                                st.error("RDKit could not embed this SMILES into 3D.")
                            else:
                                mol_label = f"drawn: {drawn_smiles.strip()}"
                                st.session_state["xyz_text"] = xyz_text
                                st.session_state["mol_label"] = mol_label
                                st.session_state["last_drawn_smiles"] = drawn_smiles
                                st.session_state["last_result"] = None
                        except Exception as e:
                            st.error(str(e))

    # Restore from session state if this rerun lost xyz_text
    if xyz_text is None and st.session_state.get("xyz_text"):
        xyz_text = st.session_state["xyz_text"]
        mol_label = st.session_state.get("mol_label")

    if xyz_text is not None:
        with st.expander("XYZ preview", expanded=False):
            st.code(xyz_text[:2000] + (" …" if len(xyz_text) > 2000 else ""), language="text")

    st.markdown("")
    run_btn = st.button(
        "Run prediction",
        type="primary",
        disabled=(xyz_text is None),
        use_container_width=True,
    )


# ── RIGHT: 3D viewer ──────────────────────────────────────────────────────
with col_viewer:
    st.markdown('<div class="section-label">3D structure</div>', unsafe_allow_html=True)

    _viewer_last = st.session_state.get("last_result")
    _orbitals_available = (
        _viewer_last is not None and _HAVE_PY3DMOL and xyz_text is not None
    )

    if xyz_text is None:
        st.markdown(
            '<div class="viewer-empty">No molecule yet — provide an input on the left.</div>',
            unsafe_allow_html=True,
        )
    elif not _HAVE_PY3DMOL:
        st.warning("`py3Dmol` not installed. `pip install py3Dmol`")
    else:
        show_mo = bool(st.session_state.get("show_orbitals", False)) and _orbitals_available

        orb_data = None
        sel_label = "HOMO"
        isovalue = 0.04
        if show_mo:
            try:
                with st.spinner("Computing HF molecular orbitals (HOMO-1 → LUMO+1)…"):
                    orb_data = cached_hf_orbitals(
                        _viewer_last["an_bytes"],
                        _viewer_last["co_bytes"],
                        _viewer_last["basis"],
                        orb_cache_dir,
                    )
            except Exception as e:
                st.error(f"Orbital generation failed: {e}")
                orb_data = None

        if show_mo and orb_data and orb_data.get("cubes"):
            sel_label = st.session_state.get("orbital_choice") or "HOMO"
            if sel_label not in orb_data["labels"]:
                sel_label = "HOMO" if "HOMO" in orb_data["labels"] else orb_data["labels"][0]
            sel_idx = orb_data["labels"].index(sel_label)
            cube_text = orb_data["cubes"][sel_idx]
            isovalue = float(st.session_state.get("orbital_isovalue") or 0.04)

            html = render_molecule_with_orbital(xyz_text, cube_text, isovalue=isovalue)
            if html:
                st.components.v1.html(html, height=500)
            mo_e = orb_data["energies_eV"][sel_idx]
            occ = orb_data["occupations"][sel_idx]
            caption = (
                f"{sel_label} · ε = {mo_e:+.3f} eV · "
                f"{'occupied' if occ > 0 else 'virtual'} · isoval = ±{isovalue:.3f}"
            )
            st.markdown(f'<div class="viewer-caption">{caption}</div>',
                        unsafe_allow_html=True)
        else:
            html = render_molecule_3d(xyz_text)
            if html:
                st.components.v1.html(html, height=480)
            if mol_label:
                st.markdown(f'<div class="viewer-caption">{mol_label}</div>',
                            unsafe_allow_html=True)

    # ── Orbital controls (only after a prediction has been run) ──────────
    if _orbitals_available:
        st.checkbox(
            "Show HF molecular orbitals",
            help="Overlay HOMO-1/HOMO/LUMO/LUMO+1 cube isosurfaces on the molecule. "
                 "First view: 5-30 s for HF + cube generation. Subsequent: cached.",
            key="show_orbitals",
        )

        if st.session_state.get("show_orbitals") and orb_data and orb_data.get("cubes"):
            ctrl_a, ctrl_b = st.columns([1, 1])
            with ctrl_a:
                st.selectbox(
                    "Orbital",
                    orb_data["labels"],
                    index=orb_data["labels"].index(sel_label),
                    key="orbital_choice",
                )
            with ctrl_b:
                st.slider(
                    "Isovalue",
                    min_value=0.005, max_value=0.10,
                    value=isovalue, step=0.005,
                    key="orbital_isovalue",
                    help="Larger = denser, smaller lobes; smaller = diffuse, larger lobes.",
                )

            an_arr = np.frombuffer(_viewer_last["an_bytes"], dtype=int)
            co_arr = np.frombuffer(_viewer_last["co_bytes"], dtype=float).reshape(-1, 3)
            already_cached = is_orbital_cached(
                an_arr, co_arr, _viewer_last["basis"], orb_cache_dir
            )

            store_col, status_col = st.columns([1, 2])
            with store_col:
                clicked = st.button(
                    "Store orbitals in cache",
                    disabled=already_cached,
                    help="Persist cube files to disk for instant subsequent loads."
                    if not already_cached else "Already on disk.",
                    use_container_width=True,
                )
            with status_col:
                if already_cached:
                    raw = an_arr.tobytes() + co_arr.tobytes() + _viewer_last["basis"].encode()
                    fname = hashlib.md5(raw).hexdigest() + ".npz"
                    st.markdown(
                        f'<div style="font-family:monospace;font-size:0.72rem;'
                        f'color:#888;padding-top:0.5rem">'
                        f'✓ cached · <code>{fname}</code></div>',
                        unsafe_allow_html=True,
                    )
                elif clicked:
                    ok, msg = save_hf_orbitals_to_cache(
                        orb_data, an_arr, co_arr, _viewer_last["basis"], orb_cache_dir
                    )
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)


# ══════════════════════════════════════════════════════════════════════════
# Run prediction
# ══════════════════════════════════════════════════════════════════════════

if run_btn and xyz_text is not None:
    try:
        Z, coords, n_atoms, formula = parse_xyz(xyz_text)
    except ValueError as e:
        st.error(f"XYZ parse error: {e}")
        st.stop()

    st.markdown(f"**Molecule:** `{formula}` · {n_atoms} atoms · basis: `{hf_basis}`")

    t0 = time.time()
    with st.spinner(f"Running HF/{hf_basis} + descriptors + model inference…"):
        try:
            result = predict_all(Z, coords, models, Path(cache_dir))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()
    elapsed = time.time() - t0

    if not result.get("success"):
        st.error(result.get("error", "Prediction failed."))
        st.stop()

    st.success(f"Completed in {elapsed:.1f} s")

    st.session_state["last_result"] = {
        "result":    result,
        "formula":   formula,
        "n_atoms":   n_atoms,
        "elapsed":   elapsed,
        "basis":     hf_basis,
        "an_bytes":  Z.tobytes(),
        "co_bytes":  coords.tobytes(),
        "xyz_text":  xyz_text,
    }
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Results display
# ══════════════════════════════════════════════════════════════════════════

_last = st.session_state.get("last_result")

if _last is not None:
    result = _last["result"]
    formula = _last["formula"]
    n_atoms = _last["n_atoms"]
    basis = _last["basis"]
    xyz_text_result = _last["xyz_text"]

    # CCSD(T) ground-truth lookup
    try:
        from groundtruth import lookup_gt
        gt = lookup_gt(xyz=xyz_text_result)
    except Exception as e:
        logger.warning(f"GT lookup unavailable: {e}")
        gt = None

    # ── Reference energies ───────────────────────────────────────────
    st.markdown('<div class="section-label">Reference energies</div>',
                unsafe_allow_html=True)

    E_HF_Ha = result.get("E_HF_Ha")
    E_HF_eV = result.get("E_HF_eV")
    mp2_kcal = result.get("E_corr_MP2_kcal")
    mp2_Ha = result.get("E_corr_MP2_Ha")

    if gt is not None:
        ref_c1, ref_c2, ref_c3 = st.columns(3)
        with ref_c1:
            st.markdown(
                f'<div class="energy-card">'
                f'<div class="model-label col-ref">CCSD(T)/cc-pVTZ — ground truth</div>'
                f'<div class="value-main">{gt["mean_kcal"]:.2f} kcal/mol</div>'
                f'<div class="value-sub">σ = {gt["std_kcal"]:.3f} · n = {gt["n"]} conformers</div>'
                f'<div class="card-footer">SMILES matched against: {", ".join(gt["datasets"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with ref_c2:
            if E_HF_Ha is not None:
                st.markdown(
                    f'<div class="energy-card">'
                    f'<div class="model-label">E<sub>HF</sub> — Hartree-Fock</div>'
                    f'<div class="value-main">{E_HF_Ha:.6f} Ha</div>'
                    f'<div class="value-sub">{E_HF_eV:.4f} eV</div>'
                    f'<div class="card-footer">PySCF · basis: {basis}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with ref_c3:
            if mp2_kcal is not None:
                gap = gt["mean_kcal"] - mp2_kcal
                st.markdown(
                    f'<div class="energy-card">'
                    f'<div class="model-label col-ref">E<sub>corr</sub> — MP2/cc-pVDZ</div>'
                    f'<div class="value-main">{mp2_kcal:.2f} kcal/mol</div>'
                    f'<div class="value-sub">{mp2_Ha:.6f} Ha</div>'
                    f'<div class="card-footer">Method gap vs CCSD(T)/cc-pVTZ: {gap:+.2f} kcal/mol (expected)</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div class="info-box">'
            "This molecule's SMILES isn't in the training set, so no CCSD(T)/cc-pVTZ "
            "ground-truth lookup. Reference is PySCF MP2/cc-pVDZ — expect a method-gap "
            "of 20–100 kcal/mol from the model's CCSD(T)/cc-pVTZ target. "
            "No chemical-accuracy badge in this mode."
            "</div>",
            unsafe_allow_html=True,
        )
        ref_c1, ref_c2 = st.columns(2)
        with ref_c1:
            if E_HF_Ha is not None:
                st.markdown(
                    f'<div class="energy-card">'
                    f'<div class="model-label">E<sub>HF</sub></div>'
                    f'<div class="value-main">{E_HF_Ha:.6f} Ha</div>'
                    f'<div class="value-sub">{E_HF_eV:.4f} eV</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with ref_c2:
            if mp2_kcal is not None:
                st.markdown(
                    f'<div class="energy-card">'
                    f'<div class="model-label col-ref">E<sub>corr</sub> — MP2/cc-pVDZ</div>'
                    f'<div class="value-main">{mp2_kcal:.2f} kcal/mol</div>'
                    f'<div class="value-sub">{mp2_Ha:.6f} Ha</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Model predictions ────────────────────────────────────────────
    st.markdown('<div class="section-label">Model predictions</div>',
                unsafe_allow_html=True)

    preds = result["predictions"]
    valid_preds = [(n, p) for n, p in preds.items() if "error" not in p]
    error_preds = [(n, p) for n, p in preds.items() if "error" in p]

    if not valid_preds:
        st.warning("No valid predictions produced.")
    else:
        pred_cols = st.columns(2)
        for i, (name, p) in enumerate(valid_preds):
            ds = p["dataset"]
            with pred_cols[i % 2]:
                if gt is not None:
                    delta_kcal = p["e_corr_kcal"] - gt["mean_kcal"]
                    within = abs(delta_kcal) < CHEM_ACCURACY_KCAL
                    cls = "delta-good" if within else "delta-bad"
                    chk = " ✓ within CA" if within else ""
                    delta_html = (
                        f'<div class="delta {cls}">'
                        f'Δ vs CCSD(T): {delta_kcal:+.3f} kcal/mol{chk}'
                        f'</div>'
                    )
                elif mp2_kcal is not None:
                    delta_kcal = p["e_corr_kcal"] - mp2_kcal
                    delta_html = (
                        f'<div class="delta delta-na">'
                        f'vs MP2: {delta_kcal:+.2f} kcal/mol (method gap)'
                        f'</div>'
                    )
                else:
                    delta_html = '<div class="delta delta-na">Δ: —</div>'

                st.markdown(
                    f'<div class="energy-card">'
                    f'<div class="model-label col-{ds}">{p["descriptor"]} · {p["architecture"]} · {ds}</div>'
                    f'<div class="value-main">{p["e_corr_kcal"]:.2f} kcal/mol</div>'
                    f'<div class="value-sub">{p["e_corr_eV"]:.4f} eV</div>'
                    f'{delta_html}'
                    f'<div class="card-footer">Test MAE on holdout: {p["test_mae_mHa"]:.3f} mHa</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    if error_preds:
        with st.expander(f"{len(error_preds)} model(s) skipped", expanded=False):
            for name, p in error_preds:
                st.warning(f"**{name}**: {p['error']}")

    # ── Comparison bar chart ────────────────────────────────────────
    if valid_preds and (gt is not None or mp2_kcal is not None):
        st.markdown('<div class="section-label">Comparison (kcal/mol)</div>',
                    unsafe_allow_html=True)
        import pandas as pd
        chart_data = {}
        if gt is not None:
            chart_data["CCSD(T) true"] = gt["mean_kcal"]
        if mp2_kcal is not None:
            chart_data["MP2 ref"] = mp2_kcal
        for name, p in valid_preds:
            short = f"{p['descriptor'][:18]} · {p['dataset']}"
            chart_data[short] = p["e_corr_kcal"]
        df = pd.DataFrame({
            "Source": list(chart_data),
            "E_corr (kcal/mol)": list(chart_data.values()),
        })
        st.bar_chart(df.set_index("Source"), use_container_width=True)

else:
    # Landing instructions
    st.markdown(
        '<div class="info-box">'
        'Provide a molecular geometry using one of the four input methods above '
        '(upload, SMILES, paste, or draw), then click <strong>Run prediction</strong>.<br><br>'
        'The pipeline will:<br>'
        '&nbsp;&nbsp;1. Run a Hartree-Fock calculation (PySCF) to obtain E<sub>HF</sub> '
        'and an MP2 correlation-energy reference.<br>'
        '&nbsp;&nbsp;2. Compute <strong>chemical</strong>, <strong>electronic</strong> '
        '(projected density-matrix eigenvalues, 108/atom), and <strong>SOAP</strong> '
        '(smooth overlap of atomic positions, 1512-dim) descriptors.<br>'
        '&nbsp;&nbsp;3. Run each loaded <strong>UnifiedModel</strong> checkpoint and compare '
        'its predicted E<sub>corr</sub> against either the CCSD(T)/cc-pVTZ ground truth '
        '(if the SMILES is in the training set) or the MP2 reference (fallback).<br><br>'
        'After the prediction, toggle <strong>Show HF molecular orbitals</strong> in the right '
        'column to overlay HOMO−1, HOMO, LUMO, LUMO+1 as 3D cube isosurfaces.<br><br>'
        'A prediction within 1 kcal/mol of the true CCSD(T)/cc-pVTZ label is considered '
        '<strong>chemically accurate</strong>.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### About this tool")
    st.markdown("""
**DeePHF+SOAP** predicts the **CCSD(T)/cc-pVTZ correlation energy** of small organic
molecules from 3D geometry, using `UnifiedModel` checkpoints trained on the
Caltech MOB-ML dataset (alkanes, water, QM7b-T, GDB-13-T).

The model combines three descriptor families:
- **Chemical**: bond connectivity + atom types (17 dims, RDKit)
- **Electronic**: projected density-matrix eigenvalues from HF (108 dims)
- **SOAP**: smooth overlap of atomic positions (1512 dims, dscribe)

Architectures span MLPs (`A_mlp`), MPNNs (`B_mpnn`), GATs (`C_gat`), and edge-bias
GATs (`D_gat_edge`). The strongest combo (chemical+elec+SOAP × D_gat_edge) achieves
**0.053 mHa test MAE** on alkanes — well under chemical accuracy (1.6 mHa).
""")


# ── Footer ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="citation-box">'
    '<strong>Cite as:</strong> Dias Gomes, Rossboth, Susanu &amp; Vasseur (2025). '
    '<em>DeePHF+SOAP: descriptor-lift and inversion in correlation-energy prediction.</em> '
    'CH-457, EPFL.<br>'
    'Built on: PySCF, RDKit, PyTorch, dscribe, py3Dmol, Streamlit.'
    '</div>',
    unsafe_allow_html=True,
)
