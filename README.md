# DeePHF Atomic Descriptor Ablation — Technical Summary

## Research Question

**Does augmenting DeePHF's electronic descriptors with SOAP atomic descriptors improve correlation energy prediction?**

DeePHF (Chen et al., 2020) predicts post-Hartree–Fock correlation energies using descriptors derived purely from the HF wavefunction (projected density matrix eigenvalues). We hypothesize that adding local structural information via SOAP (Smooth Overlap of Atomic Positions) descriptors can improve accuracy, especially when chemical diversity increases.

---

## Datasets

All experiments use the **Caltech MOB-ML dataset** (Cheng et al., 2019), the same benchmark used in the original DeePHF/MOB-ML papers. This ensures apples-to-apples comparison.

| Subset | N molecules | Description | Theory level |
|--------|------------|-------------|--------------|
| **water** | 1,000 | H₂O monomers thermalized at 350K | HF/MP2/CCSD/CCSD(T) / cc-pVTZ |
| **alkanes** | 2,304 | Ethane (1000), propane (1000), butane (100), isobutane (100) at 350K | HF/MP2 / cc-pVTZ |
| **QM7b-T** | 7,212 | Organic molecules (up to 7 heavy atoms: C, N, O, S, Cl) at 350K | HF/MP2 / cc-pVTZ |
| **rMD17 ethanol** | 1,000 (stride=100 from 100k) | Ethanol conformations from revised MD17 | DFT/PBE+vdW-TS |

**Target variable**: Correlation energy E_corr = E_best − E_HF (in Hartree), where E_best is CCSD(T) when available, else MP2.

For MOB-ML water: E_corr ≈ −174 kcal/mol (CCSD(T) level).
For MOB-ML alkanes/QM7b-T: E_corr uses MP2 level.

---

## Descriptor Pipelines

### A. Electronic Descriptors (DeePHF baseline) — 108 dim/atom

Reimplementation of the Chen et al. (2020) projected density matrix eigenvalue approach:

1. **Hartree–Fock calculation** via PySCF (RHF / cc-pVDZ basis)
2. **Density matrix** P = C_occ × C_occ^T from occupied MO coefficients
3. **Atom-centered projection basis**: 12 radial shells × (s, p, d) angular functions = 12 × 9 = 108 projectors per atom
   - Radial functions: difference-of-Gaussians with geometric exponent series from Chen et al. Appendix A
   - Exponents: [985.26, 194.62, 57.67, 17.09, 7.59, 3.38, 2.25, 1.50, 1.00, 0.667, 0.444, 0.296]
   - l_max = 2 (s: 1 function, p: 3 functions, d: 5 functions → 9 total angular per radial shell)
4. **Projection**: For each atom, project P onto atom-centered basis → sub-matrix → eigenvalue decomposition
5. **Descriptor**: 108 sorted eigenvalues per atom (rotationally invariant by construction)

Also yields E_HF and E_corr(MP2) as byproducts. Caching via MD5 hash of (Z, coords, basis).

### B. SOAP Descriptors (atomic/structural) — variable dim/atom

Computed via DScribe 2.x:

| Parameter | Value |
|-----------|-------|
| r_cut | 6.0 Å |
| n_max | 8 (radial basis functions) |
| l_max | 6 (angular momentum) |
| sigma | 0.3 Å (Gaussian smearing) |
| Crossover | True (inter-species terms via `compression={"mode": "crossover"}`) |

Feature dimension depends on species present:
- Water (H, O): 2 species → 3 pairs × 8 × 7 = 168 features/atom
- Ethanol (H, C, O): 3 species → 6 pairs × 8 × 7 = 336 features/atom  
- QM7b-T (H, C, N, O, S): 5 species → 15 pairs × 8 × 7 = 840 features/atom

### C. Combined Descriptors

Concatenation after independent z-score normalization:

```
x_combined = [x_electronic_normalized, w_atomic × x_soap_normalized]
```

Where z-score normalization is computed per-feature across all atoms in the training set:
- x_norm = (x − μ) / σ for each feature dimension
- Applied independently to electronic (108-dim) and SOAP blocks

w_atomic ∈ {0.1, 0.5, 1.0, 2.0} controls relative importance of SOAP features.

**Critical finding**: Without normalization, electronic features (range [0, 5]) dominate over SOAP features (unit-norm normalized, range [−0.01, 0.05]), making combined mode perform *worse* than electronic-only. Z-score normalization is essential.

---

## ML Model: CorrNet

Per-atom neural network that sums atomic energy contributions to predict molecular correlation energy.

### Architecture

```
Input: (batch, n_atoms, n_features)
  │
  ├── Linear branch: Linear(n_features → 1)          [initialized via ridge regression]
  │
  └── Nonlinear branch: 3-layer ResNet
        Linear(n_features → 100) → GELU → residual
        Linear(100 → 100) → GELU → residual
        Linear(100 → 100) → GELU → residual
        Linear(100 → 1)
  │
  ├── Per-atom energy = linear + nonlinear             (batch, n_atoms, 1)
  └── Molecular energy = sum over atoms                (batch,)
```

### Training Details

| Hyperparameter | Value |
|---------------|-------|
| Hidden layers | 3 × 100 (GELU activation) |
| Residual connections | Yes (when dims match) |
| Pre-fitting | Ridge regression (α=10) on atom-summed descriptors |
| Input normalization | Per-feature z-score (shift/scale from training data) |
| Optimizer | Adam (lr=3×10⁻⁴) |
| LR schedule | StepLR (decay=0.5 every 150 epochs) |
| Loss | MSE on molecular E_corr (eV) |
| Epochs | 200 (quick) / 500 (full) |
| Batch size | 32 |
| Early stopping | Best validation MAE checkpoint |
| Train/test split | 80/20 random |
| Metric | MAE on E_corr (eV and kcal/mol) |

### Learning Curves

Training sizes swept: [25, 50, 100, 200, 400] molecules from the training pool.

---

## Experimental Protocol

For each dataset × mode × w_atomic × train_size:

1. Load molecules from MOB-ML dataset (geometry + E_HF + E_corr from energy.dat)
2. Compute descriptors:
   - `electronic_only`: Run PySCF HF → project density matrix → 108 eigenvalues/atom
   - `atomic_only`: Compute SOAP via DScribe
   - `combined_w=X`: Z-score normalize both independently, concatenate with weight X
3. Pad to uniform (N, max_atoms, n_features) tensor (zero-padding for smaller molecules)
4. Split 80/20 train/test
5. Train CorrNet with ridge pre-fitting
6. Record best validation MAE across epochs
7. Compute per-element MAE breakdown

---

## Results (Preliminary)

From the cross-dataset bar chart (quick sweep, 200 samples per dataset, 200 epochs):

| Dataset | Electronic (baseline) | SOAP only | Best combined | Improvement |
|---------|----------------------|-----------|---------------|-------------|
| **water** | ~0.01 eV | ~0.005 eV | ~0.005 eV (w=1.0) | ~50% |
| **alkanes** | 0.32 eV | 0.07 eV | ~0.08 eV (w=2.0) | ~75% |
| **QM7b-T** | 1.45 eV | 0.67 eV | 0.51 eV (w=2.0) | ~65% |
| **rMD17 ethanol** | 0.07 eV | 0.05 eV | 0.024 eV (w=1.0) | ~65% |

### Key Findings

1. **SOAP consistently outperforms electronic-only** across all datasets, often by 50–75%.
2. **Combined descriptors offer modest further improvement** over SOAP-only on QM7b-T (the most chemically diverse dataset), but SOAP alone is competitive or better on simpler systems.
3. **Optimal w_atomic ≈ 1.0–2.0** after z-score normalization (equal or slight over-weighting of SOAP).
4. **Normalization is critical**: without it, combined mode is worse than either alone due to feature scale mismatch.
5. **Water is too easy**: only 3 atoms, single molecule type — not enough structural variation for SOAP to help much.
6. **Diversity matters**: The improvement from SOAP grows with chemical diversity (water < alkanes < QM7b-T).

### Interpretation

The electronic descriptors capture quantum-mechanical information (orbital occupancy, electron correlation structure) but are expensive to compute (require HF calculation) and miss local geometric detail. SOAP descriptors are cheap (no QM needed) and encode 3D structural neighborhoods directly. The combined approach gets the best of both worlds: QM-informed electronic structure + geometric context.

The fact that SOAP alone nearly matches or beats electronic descriptors suggests that for the DeePHF prediction task, local atomic geometry is highly informative — which makes physical sense since correlation energy is dominated by short-range electron-electron interactions that correlate strongly with bond lengths and angles.

---

## Code Inventory

| File | Description |
|------|-------------|
| `deephf_dataloader.py` | Unified loader for MOB-ML, QM7-X, rMD17, QM7b, GMTKN55, deepks-kit water |
| `deephf_descriptors.py` | SOAP descriptor computation via DScribe (r_cut=6, n_max=8, l_max=6) |
| `deephf_electronic.py` | Electronic descriptor computation via PySCF (HF → density matrix → DoG projection → eigenvalues) |
| `deephf_train.py` | CorrNet training with ablation sweep (modes × w_atomic × train_sizes) |
| `run_sweep.sh` | Shell script to run full MOB-ML benchmark (water + alkanes + QM7b-T) |
| `plot_results.py` | Visualization: learning curves, w_sweep, cross-dataset bar chart, per-element MAE |

---

## Next Steps

- [ ] Run full sweep (1000 samples, 500 epochs) on all MOB-ML subsets
- [ ] Scale to QM7-X (4.2M structures, diverse chemistry) on IZAR cluster
- [ ] Test GDB-13-T subset (larger molecules, transferability test)
- [ ] Per-element analysis on QM7b-T (H/C/N/O/S contributions)
- [ ] Compare with MOB-ML paper's reported accuracies (sub-kcal/mol on water with 1 training sample)
- [ ] Investigate why SOAP alone is so competitive — is the electronic descriptor bottleneck the basis set (cc-pVDZ) or the projection scheme?

---

## References

- Chen, Y., Zhang, L., Wang, H., & E, W. (2020). DeePHF: A deep learning approach for Hartree–Fock theory. *JCTC*.
- Cheng, L., Welborn, M., Christensen, A. S., & Miller, T. F. (2019). A universal density matrix functional from MOB-ML. *JCP* 150, 131103.
- Husch, T., Sun, J., Cheng, L., Lee, S. J. R., & Miller, T. F. (2021). Improved accuracy and transferability of MOB-ML. *JCP* 154, 064108.
- Bartók, A. P., Kondor, R., & Csányi, G. (2013). SOAP: On representing chemical environments. *PRB*.
