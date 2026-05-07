# DeePHFProject — Streamlit Interface

**Vera Dias Gomes · Cédric Nathanaël Rossboth · Octavian Susanu · Louis James Vasseur**  
Team 3 · AI for Chemistry (CH-457) · EPFL

---

## Overview

This is a **Streamlit** replacement for the Docker/FastAPI web interface.
It provides the same prediction capabilities in a self-contained Python script,
plus three additional features:

1. **Drawing-based input** — sketch your molecule directly in the browser via
   the embedded **Ketcher** chemical editor; the SMILES is captured and passed
   through the same RDKit ETKDG + MMFF pipeline as typed SMILES input.
2. **Live 3D molecule preview** — every input method now renders the resulting
   geometry in an interactive **py3Dmol** viewer next to the input panel,
   *before* you click *Run prediction*.
3. **HF molecular orbital visualisation** — after a prediction, a checkbox
   reveals the four frontier orbitals (HOMO−1, HOMO, LUMO, LUMO+1) as cube-file
   isosurfaces (positive lobe blue, negative lobe red), with adjustable
   isovalue and a selector for which orbital to display.

---

## Installation

```bash
# 1. Clone the repository (or unzip the project)
git clone https://github.com/LouisVasseur/DeePHFProject.git
cd DeePHFProject

# 2. Install dependencies (now includes streamlit, streamlit-ketcher, py3Dmol)
pip install -r requirements.txt

# 3. (Optional) install torch_geometric for the GNN
pip install torch_geometric
```

> **Note:** `pyscf` is Linux-only. On macOS/Windows you can still use the GNN
> and view the app, but the HF calculation (and therefore the orbital
> visualisation) will fail.

---

## Running the app

From the **repository root**, run:

```bash
streamlit run app_streamlit/app.py
```

The app opens automatically at **http://localhost:8501**.

---

## File structure

```
app_streamlit/
├── app.py                     ← Main Streamlit application
├── cache_electronic/          ← Auto-created: HF descriptor cache (.npz)
├── cache_orbitals/            ← Auto-created: HF orbital cube cache (.npz)
└── README.md                  ← This file
```

---

## Using the app

### Input methods (4 modes)

1. **Upload .xyz file** — drag & drop or browse a standard XYZ geometry file.
2. **Enter SMILES** — type a SMILES string (e.g. `O`, `CC`, `c1ccccc1`) and
   click *Generate 3D structure* to create a geometry via RDKit ETKDG + MMFF.
3. **Paste XYZ text** — paste the XYZ content directly into the text area.
4. **Draw structure** — sketch the molecule in the embedded Ketcher
   editor. Click **Apply** in the editor toolbar to capture the SMILES; the
   3D geometry is then generated automatically (or click *Generate 3D structure
   from drawing* to refresh).

The right-hand panel always shows the live 3D structure as soon as a geometry
is available — drag to rotate, scroll to zoom.

### Sidebar settings

| Setting | Description |
|---|---|
| Model paths | Paths to the four checkpoint files |
| HF basis set | PySCF basis set (default: `cc-pvdz`) |
| SOAP weight | `w_atomic` for combined descriptor (default: 1.0) |
| GNN stats (mean/std) | Training-set normalisation constants for GNN output (eV) |
| Cache directory | Where to store HF descriptor caches |
| Orbital cache directory | Where to store HF orbital cube files |

### Results

- **Reference energies** — E_HF and E_corr(MP2) from PySCF
- **Model predictions** — E_corr in eV and kcal/mol for each model
- **Δ vs MP2** — error relative to the MP2 reference, coloured green if within
  chemical accuracy (1 kcal/mol) and red otherwise
- **Comparison bar chart** — visual comparison of all predictions
- **Molecular orbitals** — toggle the *Show calculated HF molecular
  orbitals* checkbox, then choose between HOMO−1, HOMO, LUMO, LUMO+1 and
  adjust the isovalue (default 0.04 a.u.). Generation takes 5–30 s the first
  time per molecule and is held in memory (across reruns) for the rest of
  the session. Click **Store orbitals in cache** to persist the cube files
  to disk so they are reused on later sessions; until you click that
  button, no orbital data is written to disk.

---

## Performance notes

- HF + electronic descriptors are cached **automatically** under
  `cache_electronic/` keyed by geometry + basis set.
- HF orbital cubes are cached **manually**, on demand: viewing orbitals only
  generates them in memory; click *Store orbitals in cache* to persist them
  under `cache_orbitals/` (~11 MB per molecule for the four frontier orbitals
  at the default 60×60×60 grid). Until then nothing is written to disk for
  orbitals.
- Within a single session, Streamlit's `@st.cache_data` keeps the in-memory
  orbital result so toggling/selecting between MOs is instant after the
  first generation.
- The orbital toggle does **not** trigger a re-prediction — predictions are
  persisted in `st.session_state["last_result"]` and survive reruns triggered
  by the toggle/selector/slider/store-button.

---

## XYZ format reference

```
3
Water molecule
O   0.000000   0.000000   0.000000
H   0.000000   0.757000   0.587000
H   0.000000  -0.757000   0.587000
```

Line 1: atom count. Line 2: comment (any text). Lines 3+: `Symbol  x  y  z` (Å).
