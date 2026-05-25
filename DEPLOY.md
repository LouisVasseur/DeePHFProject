# Deploying the DeePHF+SOAP demo

End-to-end deployment from a clean clone of the repository to a running
Streamlit app. Two paths depending on whether you have the trained
checkpoints already.

---

## Prerequisites

- Linux (Ubuntu 22+ tested) — PySCF and openbabel-wheel are Linux-first
- Python 3.9–3.12
- ~6 GB disk free for venv + caches
- GPU not required for inference (CPU is fine), but training the checkpoints
  needs a CUDA GPU (V100 / A100 class)

---

## Path A: Deploy from a clone with checkpoints already trained

Use this path when you've trained the five checkpoints (e.g., on a SLURM
cluster) and want to launch the app locally or on a different host.

```bash
# 1. clone the repo
git clone https://github.com/<you>/deephf-soap.git
cd deephf-soap

# 2. set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. drop the checkpoints into place
mkdir -p app_streamlit/checkpoints
# copy each .pt file produced by train_cell.py --save-checkpoint-to
scp <cluster>:/path/to/checkpoints/*.pt app_streamlit/checkpoints/

# 4. launch
streamlit run app_streamlit/app.py
```

Open `http://localhost:8501`.

---

## Path B: Full deploy from raw data

Use this when you don't yet have the trained checkpoints.

```bash
# 1. clone + env (same as Path A steps 1–2)

# 2. download raw MOB-ML data from Caltech
#    https://data.caltech.edu/records/waft4-tww64
#    unpack so you have:
#    deephf_datasets/caltech_mobml/data/{water,alkanes,qm7b_T,gdb13_T}/

# 3. preprocess into HuggingFace datasets format
python prepare/prepare_mobml.py --subset alkanes
python prepare/prepare_mobml.py --subset qm7b_T

# 4. compute electronic + SOAP descriptors (~1–3 hours per dataset)
python prepare/compute_descriptors.py --subset alkanes
python prepare/compute_descriptors.py --subset qm7b_T

# 5. train the five app models (on a GPU)
sbatch slurm/train_app_models.sbatch
# or locally (single GPU):
bash slurm/train_app_models.sbatch

# 6. launch
streamlit run app_streamlit/app.py
```

---

## Verifying the deployment

After step 4 in Path A or step 5 in Path B:

```bash
# checkpoint integrity
python -c "
import torch, glob
files = sorted(glob.glob('app_streamlit/checkpoints/*.pt'))
print(f'Found {len(files)} checkpoint(s).')
for f in files:
    ckpt = torch.load(f, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    print(f'  {f.split(\"/\")[-1]:55s}  {cfg[\"architecture\"]:11s}  '
          f'test_mae={ckpt[\"test_mae_mHa\"]:.4f} mHa')"
```

Expected output (with the bundled `slurm/train_app_models.sbatch`):

```
Found 5 checkpoint(s).
  alkanes_chemical_D_gat_edge_seed43.pt            D_gat_edge   test_mae=~1.4 mHa
  alkanes_chemical_elec_D_gat_edge_seed43.pt       D_gat_edge   test_mae=~0.14 mHa
  alkanes_chemical_elec_soap_D_gat_edge_seed43.pt  D_gat_edge   test_mae=~0.07 mHa
  alkanes_chemical_soap_D_gat_edge_seed43.pt       D_gat_edge   test_mae=~0.26 mHa
  qm7b_T_chemical_elec_A_mlp_seed43.pt             A_mlp        test_mae=~1.1 mHa
```

If any value is off by more than 30 % from these targets, the training was
probably wrong (e.g., broken data loader, mismatched seed). Re-train that
specific cell.

---

## Smoke test

Once the app is running at `http://localhost:8501`:

1. Switch to **Enter SMILES** mode.
2. Type `CC` and click **Generate 3D**. The 3D viewer should show ethane.
3. Click **Run prediction**.
4. The "Reference (PySCF)" section should show `E_HF ≈ -79.2 Ha` and
   `E_corr (MP2) ≈ -0.22 Ha`.
5. The four alkanes models should produce predictions ranging from poor
   (`chemical` alone, ~5–10 kcal/mol off MP2) to excellent
   (`chemical_elec_soap`, sub-kcal/mol from MP2). This is the descriptor lift,
   visible per-molecule.

---

## Deploying to a remote host (Streamlit Community Cloud or self-hosted)

The cleanest options for a public demo:

- **Streamlit Community Cloud** (free): point it at the repo with checkpoints
  committed (they're ~1–10 MB each, well under GitHub LFS limits). Be aware
  that PySCF on Streamlit Cloud is slow — the first HF run per molecule
  dominates response time. SOAP-only and chemical-only models will still
  feel snappy.
- **Self-hosted (Docker / VM)**:
  ```bash
  # minimal Dockerfile
  FROM python:3.11-slim
  RUN apt-get update && apt-get install -y build-essential libopenbabel-dev
  COPY . /app
  WORKDIR /app
  RUN pip install -r requirements.txt
  EXPOSE 8501
  CMD ["streamlit", "run", "app_streamlit/app.py", "--server.address=0.0.0.0"]
  ```
  Build with `docker build -t deephf-app .` and run with
  `docker run -p 8501:8501 deephf-app`.

For both, make sure `app_streamlit/checkpoints/*.pt` ships with the image or
is mounted at runtime.
