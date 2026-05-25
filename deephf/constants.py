# AI Use: none, parameters found via testing
# Verified using standard chemical formulas

"""Constants and default hyperparameters."""

# Unit conversions
EV_TO_KCAL = 23.0609
EV_TO_HA = 1.0 / 27.2114
EV_TO_MHA = EV_TO_HA * 1000

# Element table used everywhere in the project
SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}

# Feature layout in the prepared HF datasets
CHEM_DIM = 17 # RDKit per-atom features (cf prepare/graph_features.py)
ELEC_DIM = 108 # DeePHF electronic descriptor (DoG projection eigenvalues)
RDKIT_EDGE_DIM = 5 # 4-dim bond-type one-hot + 1-dim bond length

# RBF expansion for interatomic distances
RBF_N = 16
RBF_RANGE = (0.0, 10.0)

# Architecture hyperparameters (fixed across all cells)
NODE_DIM = 128
MP_ROUNDS = 3
N_HEADS = 4
DROPOUT = 0.0

# Training protocol (fixed across all cells)
LR = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
LR_DECAY = 0.96
LR_DECAY_EVERY = 500
ES_PATIENCE = 500
MAX_EPOCHS = 5000

# Default dir paths
DATA_DIR = "gnn_data_enriched"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
TABLES_DIR = "tables"
