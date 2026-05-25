# DeePHF+SOAP — Streamlit demo

Interactive front-end for the `UnifiedModel` checkpoints from the AI4Chem
report. Predicts correlation energy on a user-supplied molecule and shows
the **descriptor lift** (same architecture, four progressively richer
descriptors) and the **QM7b-T inversion** (simpler architecture wins on
diverse chemistry).

This app replaces the legacy CorrNet × 3 + GNN pipeline with the new
single-class `UnifiedModel` (A_mlp / B_mpnn / C_gat / D_gat_edge) from
`deephf/`.

---

## What's in here

```
app_streamlit/
├── app.py                  ← Streamlit entry point
├── inference.py            ← model loading + descriptor pipeline + predict()
├── checkpoints/            ← .pt files (gitignored — produced by training)
├── cache_electronic/       ← auto-created HF/MP2 cache (.npz keyed by geometry)
└── README.md               ← this file
```

---

## Quick deploy (already-trained checkpoints in place)

From the **repository root**:

```bash
# 1. install deps
pip install -r requirements.txt

# 2. launch the app
streamlit run app_streamlit/app.py
```

Open `http://localhost:8501`. Provide a molecule (XYZ upload, paste, or SMILES)
and click **Run prediction**.

The app expects checkpoint files in `app_streamlit/checkpoints/`. If absent,
the app shows an error directing you to train them first (see below).

---

## Training the checkpoints (one-time)

The app needs five checkpoint files produced by `train_cell.py` with
`--save-checkpoint-to`. Submit the bundled sbatch from the repo root:

```bash
sbatch slurm/train_app_models.sbatch
```

This trains 5 cells (~45–60 min on a V100):

- `alkanes_chemical_D_gat_edge_seed43.pt`
- `alkanes_chemical_elec_D_gat_edge_seed43.pt`
- `alkanes_chemical_soap_D_gat_edge_seed43.pt`
- `alkanes_chemical_elec_soap_D_gat_edge_seed43.pt`  ← headline cell
- `qm7b_T_chemical_elec_A_mlp_seed43.pt`             ← inversion case

After the job finishes, all five `.pt` files appear in
`app_streamlit/checkpoints/`. Smoke-test:

```bash
python -c "
import torch, glob
for f in sorted(glob.glob('app_streamlit/checkpoints/*.pt')):
    ckpt = torch.load(f, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    print(f'{f.split(\"/\")[-1]:55s}  {cfg[\"architecture\"]:11s}  test_mae={ckpt[\"test_mae_mHa\"]:.4f} mHa')"
```

---

## How it works

1. User provides an XYZ structure (file upload, paste, or generated from SMILES).
2. `inference.compute_descriptors` runs:
   - `prepare.graph.build_graph_from_arrays` — RDKit-perceived bonds + 17-dim
     chemical features per atom + 5-dim edge features per bond.
   - `prepare.electronic.Cache.get_or_compute` — PySCF Hartree–Fock + MP2,
     density-matrix projection → 108-dim per-atom electronic descriptor.
     Cached on disk under `cache_electronic/` keyed by SHA1 of `(Z, coords)`.
   - `prepare.soap.compute_for_mol` — DScribe SOAP at
     `r_cut=6 Å, n_max=8, l_max=6`, species list matching the training dataset.
3. For each loaded checkpoint, the descriptors are concatenated according to
   the checkpoint's `descriptor` field (e.g., `chemical_elec_soap` →
   `[chem | elec | soap]`), then run through `UnifiedModel.forward`.
4. The predicted E_corr (in eV) is converted to kcal/mol and compared against
   the MP2 reference; green ✅ if within chemical accuracy.

The model does NOT use 3D coordinates directly at inference — bond distances
are taken from the `edge_type` feature (column 4, computed by RDKit from the
input geometry), and RBF-expanded inside the model. This matches the training
pipeline.

---

## Constraints

- **Element coverage**:
  - Alkanes-trained models only accept H and C atoms. Other elements →
    species-list mismatch in SOAP → prediction skipped with a clear error.
  - QM7b-T model accepts H, C, N, O, S, Cl.
- **PySCF availability**: HF and MP2 require PySCF (`pip install pyscf`),
  which is Linux-only. On macOS/Windows the predictions still work for any
  checkpoint that doesn't need `chemical_elec` (i.e., the alkanes `chemical`
  and `chemical_soap` variants), but `chemical_elec` and `chemical_elec_soap`
  will fail. The MP2 reference will also be unavailable.
- **RDKit bond perception**: Geometry-only input (XYZ) is converted to a
  bonded graph via OpenBabel → SDF → RDKit. Unusual valences may fail.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `No checkpoints found` | Haven't run `sbatch slurm/train_app_models.sbatch` yet. |
| `Feature-dim mismatch` for one model | Input molecule has elements outside the model's training species set (e.g., nitrogen on an alkanes model). |
| `RDKit could not perceive bonds` | Geometry too distorted, or unsupported element. |
| `HF/MP2 failed: ...` | PySCF can't converge (poor geometry, exotic valence). The non-electronic models will still run. |
| All predictions identical | You're probably running the same architecture on a tiny molecule where the descriptors don't differentiate much — try a larger alkane. |

---

## Extending

Want more checkpoints? Add another `train_cell.py --save-checkpoint-to ...`
line to `slurm/train_app_models.sbatch`. The app auto-detects whatever `.pt`
files are present and renders one row per model.

To add a new dataset's species list (e.g., `water` or `gdb13_T`), edit the
`_DATASET_SPECIES` dict in `inference.py`. The species must match the order
that DScribe sorts them internally — alphabetical by element symbol, which
matches what `prepare/compute_descriptors.py` produces.
