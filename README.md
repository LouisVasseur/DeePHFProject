# Augmenting DeePHF with geometric descriptors

Code for the AI4Chemistry course project at EPFL.

The report can be found here, on the **main** branch, under the name `report.pdf`.

For the web interface, visit the link: https://huggingface.co/spaces/LouisVasseur/deephf-soap-demo

This branch contains a combined implementation of the different architectures that were studied in this project and of the web-interface. If you want to see the initial implementation of the GNNs, check-out the dedicated branch **graph_branch**. If you want to see the initial implementation of the interface, check-out the dedicated branch **add-web-interface**.

The project re-implements the DeePHF correlation-energy predictor of Chen et al. (2020) and tests
whether adding SOAP geometric descriptors to the original electronic
descriptors improves accuracy, and whether the gain depends on the
inductive bias of the prediction network.

The four prediction networks span a spectrum of inductive bias on top of a
shared per-element OLS energy baseline:

| Tag         | Architecture                                                |
|-------------|-------------------------------------------------------------|
| `A_mlp`     | per-atom MLP, no message passing                            |
| `B_mpnn`    | sum-aggregation MPNN with RBF + bond-type edge features     |
| `C_gat`     | multi-head GAT, attention over neighbours (no edge features)|
| `D_gat_edge`| multi-head GAT with edge-bias attention (Uni-Mol style)     |

The four descriptor variants are concatenations onto the same 17-dim RDKit
chemical base:

| Tag                  | Dims (water example) | Contents                          |
|----------------------|----------------------|-----------------------------------|
| `chemical`           | 17                   | RDKit only                        |
| `chemical_elec`      | 17 + 108             | RDKit + 108-dim electronic        |
| `chemical_soap`      | 17 + N_soap          | RDKit + DScribe SOAP              |
| `chemical_elec_soap` | 17 + 108 + N_soap    | RDKit + electronic + SOAP         |

Datasets are the four MOB-ML subsets of Cheng et al. 2019:
`water`, `alkanes`, `qm7b_T`, `gdb13_T`. Target is the correlation energy
`E_corr = E_MP2 - E_HF` at cc-pVTZ.

## Layout

```
app_streamlit/ code for the Streamlit app used for demo/inference with best model checkpoints
deephf/      library: data loader, model, training loop
prepare/     one-off feature preparation (RDKit graph, SOAP, electronic)
scripts/     experiments and figure generation
```

## Quick start

```bash
# 0. install
pip install -r requirements.txt

# 1. prepare features (slow; needs the MOB-ML raw data) NOTE THAT the MOB-ML dataset must be downloaded for this to work
python prepare/prepare_mobml.py --data-dir ./deephf_datasets --subset water
python prepare/compute_descriptors.py --subset water
# repeat for alkanes, qm7b_T, gdb13_T

# 2. train one cell
python scripts/train_cell.py \
    --dataset alkanes --descriptor chemical_elec_soap \
    --architecture D_gat_edge --seed 43

# 3. full grid (4 datasets × 4 descriptors × 4 architectures + bias_only)
python scripts/train_grid.py --seed 43 --resume

# 4. ablations
python scripts/ablate_edge_bias.py
python scripts/ablate_edge_features.py --dataset alkanes
python scripts/test_invariance.py
python scripts/k_sweep.py # not in the paper due to lack of space

# 5. figures + LaTeX tables
python scripts/make_artifacts.py
```

## Hyperparameters

Fixed across every cell (so architecture and descriptor are the only sources
of variation):

- Hidden node dim 128, 3 message-passing rounds, 4 attention heads
- AdamW lr 3e-4, weight decay 1e-4, batch size 128, AMP enabled
- Exponential LR decay (factor 0.96 per 500 epochs)
- Early stopping on validation MAE with patience 500, max 5000 epochs
- Per-element energy bias initialised by OLS on the training split and frozen
- 80/10/10 train/val/test split, fixed at feature-preparation time
- Default seed 43 (primary), seed 44 for a subset of headline cells
