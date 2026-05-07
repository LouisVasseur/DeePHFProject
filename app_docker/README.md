# DeePHFProject — Correlation Energy Predictor

**Vera Dias Gomes · Cédric Nathanaël Rossboth · Octavian Susanu · Louis James Vasseur**  
Team 1 · AI for Chemistry (CH-457) · EPFL

---

## What is this?

DeePHFProject is a tool that predicts the **correlation energy** of a molecule from its geometry alone. You provide a molecular structure file and the app returns correlation energy predictions from four different models side by side, so you can directly compare their accuracy against an MP2 reference.

The four models differ in what information they use to make their prediction:

| Model | What it uses |
|---|---|
| **Electronic only** | Quantum-mechanical descriptors derived from the Hartree-Fock wavefunction (108 eigenvalues per atom) |
| **SOAP only** | Local 3D geometry around each atom (no quantum chemistry) |
| **Combined** | Both of the above together |
| **GNN** | Molecular graph directly *(coming soon)* |

All computation runs on your machine. Nothing is sent to an external server.

---

## Prerequisites

The application is distributed through Docker.  
Before launching the interface, install Docker and Docker Compose.

### Ubuntu / Debian

```bash
sudo apt install docker.io
sudo apt install docker-compose-v2
sudo usermod -aG docker $USER
newgrp docker
```

---

## Launching the interface

First, clone the repository and navigate into the `app/` folder:

```bash
git clone https://github.com/LouisVasseur/DeePHFProject.git
cd DeePHFProject/app_docker
```

Then, for the first launch:

```bash
docker compose up --build
```

**The first time only**, this will take **5–15 minutes** — Docker builds the environment image and installs all dependencies (PySCF, PyTorch, DScribe, and others). You will see a long stream of installation output. This is normal.

Once you see a line like `Uvicorn running on http://0.0.0.0:8080`, the app is ready. Open your browser and go to:

**[http://localhost:8080](http://localhost:8080)**

For every subsequent launch, the image is already built so you can simply run:

```bash
docker compose up
```

This starts in a few seconds. All calculations run entirely on your local machine — nothing is sent to an external server.

To stop the app, press `Ctrl + C` in the terminal.

---

## What you need

A molecular structure in **standard XYZ format** — a plain text file with the `.xyz` extension structured like this:

```
3
Water molecule
O   0.000000   0.000000   0.000000
H   0.000000   0.757000   0.587000
H   0.000000  -0.757000   0.587000
```

The first line is the number of atoms, the second is a comment (can be left blank), and each following line gives an element symbol and its x, y, z coordinates in Ångström. Most quantum chemistry packages (Gaussian, ORCA, Avogadro, ASE) can export this format directly.

---

## How to use it

Open [http://localhost:8080](http://localhost:8080) in your browser.

**Step 1 — Drop your file**  
Drag your `.xyz` file onto the drop zone, or click it to browse for a file.

**Step 2 — Wait for the calculation**  
The app first runs a Hartree-Fock calculation using PySCF. This is the slow step — expect **10 to 60 seconds** depending on the size of your molecule. A live timer shows how long it has been running. Larger molecules with more heavy atoms take longer.

**Step 3 — Read the results**  
Once complete, the interface shows:

- **E_HF** — the Hartree-Fock total energy, in eV and Hartree
- **E_corr (MP2)** — the MP2 correlation energy, used as the ground truth reference
- **Three model predictions** — each showing E_corr in eV and kcal/mol, with a Δ vs MP2 error displayed below. The error turns **green** if it is below 1 kcal/mol (chemical accuracy) and **red** if it exceeds it.

You can drop a new file at any time to run a new prediction.

---

## Tips

**If the HF calculation fails to converge**, your geometry is likely unphysical — atoms may be too close together or in an unusual bonding arrangement. Try geometry-optimising your structure first with a force field or a semi-empirical method (xTB works well for this) before using it here.

**If a model column shows "model not loaded"**, the checkpoint file for that model has not been provided yet. The other models and the reference energies are still computed normally.

**Molecules that have already been computed are cached** — if you drop the same structure again (same geometry, same basis set), the HF step is skipped and results appear immediately.

---

## What the numbers mean

**Correlation energy (E_corr)** is the difference between the exact ground-state energy and the Hartree-Fock energy. It captures electron-electron interactions that HF theory ignores. It is typically a small negative number — for water it is around −0.3 eV — but it is chemically important: reaction barriers, bond dissociation energies, and non-covalent interactions all depend on it.

**MP2** (Møller-Plesset perturbation theory, second order) is the reference method used here. It is a standard post-HF method that captures most of the correlation energy at a tractable computational cost.

**Chemical accuracy** is conventionally defined as an error below 1 kcal/mol (~0.04 eV) relative to a high-level reference. This is the threshold below which a prediction is considered reliable for most chemistry applications.

---

## Reference

Built on DeePHF — Chen, Y., Zhang, L., Wang, H., & E, W. (2020). *DeePHF: A machine learning-based electron correlation and excitation energy predictor.* The Journal of Chemical Physics, 152, 034102.
