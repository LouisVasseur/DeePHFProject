# Graph Branch - Employing GNNs for $E_{corr}$ determination

This branch demonstrates 3 types of GNN architectures used in the determination of correlation energy, together with the scripts necessary in order to construct graph features, preprocess the MOB-ML dataset and train the networks. The graph features used in this branch are called in the paper `chemical` descriptors.

## Graph Features Extraction

In GNNs, molecules are encoded as graphs, with the atoms corresponding to nodes and the bonds to edges. The code corresponding to feature extraction may be found in `graph_atomic_descriptors.py`

The 17-dimensional node (atomic) features are as follows:

| Feature | Encoding Type | Description |
|--------|------------|-------------|
| **Identity** | One-hot | Encodes identity of the atom between H,C,N,O,S,Cl |
| **Atomic number** | Integer | Z |
|**Aromaticity** | Binary | 1 if aromatic, 0 else |
|**Implicit H count** | Integer | Number of neighboring Hs |
| **Hybridization** | One-hot | Encodes hybridization of the atom |
| **Electronegativity** | Float | Pauling electronegativity of the atom |

The 5-dimensional edge (bond) features are as follows:

| Feature | Encoding Type | Description |
|--------|------------|-------------|
| **Identity** | One-hot | Encodes identity of the bond between single, double, triple, aromatic |
| **Length** | Float | Encodes the length of the bond |

## Preprocessing the MOB-ML Datasets

Follow the following commands for one of your chosen MOB-ML partitions (water, alkanes, qm7b_T, gdb13_T)

The script for converting the chosen MOB-ML dataset into a HuggingFace (HF) dataset is in `hf_dataset.py`. To run it, enter the following command:

```
bash run_scripts/convert_to_hf.sh
```

after having modified the variables in the `.sh` script to match your environment. The scripts that follow operate only on HF datasets

To extract the atom and bond features, *openbabel* is employed to convert .xyz format to .sdf format, which is then processed by RDKit. If conversion fails, the datapoint is discarded. The script for extracting these features from the datasets can be found in `process_mobml_for_graph`. To run it, enter the following command:

```bash
bash run_scripts/process_mobml_graph.sh
```

after having modified the variables in the `.sh` script to match your environment. 

To obtain a deterministic 80/10/10 - train/val/test split, you can run the following command:

```bash
bash run_scripts/split_dataset.sh
```

after having modified the variables in the `.sh` script to match your environment. This will run the `split_dataset.py` script.


## GNN Architectures

This branch supports 3 types of GNN architectures. The implementation of the architectures is found in `gnn_model.py`. 

1. **V1 - Graph Convolutional Network (GCN)** - mimics the architecture proposed by Gilmer et al. [1]. Thus, the message passing is done through graph-convolutions (shared weights across message passing steps) and the message updates (shared weights across message passing steps) is done using a GRU. After each message passing step, a normalization is applied. In the readout, a soft-attention mechanism is applied between the initial node state and the processed node state, in order to yield the final node state. Finally, these are aggregated by summation, producing a graph embedding that is then passed through an MLP to predict the correlation energy.

2. **V2** - Takes the architecture presented in V1 but uses an LSTM instead of a GRU (goal was reducing noisy train loss)

3. **V3 - Graph Attention Network (GAT)** - closely follows the architecture proposed by Velickovic et al. [2]. The message passing is done through a graph tailored attention mechanism. The update of the nodes is done by a linear combination of the hidden states of the neighboring nodes with the attention coefficients weighing the combination, after which the ELU nonlinearity is applied. After each message passing steps, a normalization layer with skip connections is applied. The attention layers use weight sharing across the message passing  In the "readout phase" (since no longer a readout phase as described for previous architectures), the same attention mechanism as the one in message passing is applied. As for the other architectures, a graph embedding is produced by summation of the nodes, and the embedding is passed through an MLP to predict the target.

## GNN Training

In order to train a GNN model, run the following command:

```bash
bash run_scripts/train.sh
```

after having modified the variables in the `.sh` script. This will launch the training script found in `train.py`, which makes use of `gnn_dataloader.py` to prepares the data in `TochDataset` format and `gnn_wrapper.py` to wrap the desired GNN architecture into the `pl.LightningModule`. You can then visualize the evolution of the training on **wandb**.

To modify the parameters used to train the model, edit the `conv_gnn.yml` file. Currently, V1-V3 are trained using a **hidden node dimension** of `64`, `5` **message passing steps**, **AdamW** optimizer, an **initial learning rate** or `3e-4` with a "reduce on plateau" scheduler. The **loss used is the mean squarred loss** (MSE). Training is done for `500` **epochs**.

## References

[1]. Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for quantum chemistry. *arXiv preprint: https://arxiv.org/abs/1704.01212*

[2]. Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li`o, and Yoshua Bengio. Graph attention networks. *arXiv preprint: https://arxiv.org/abs/1710.10903*
