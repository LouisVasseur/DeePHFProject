# AI Use: code architected by hand and implemented by hand, optimized and fixed by Claude for modularity, efficiency, and correctness. 

"""UnifiedModel: one class with five architecture variants.

  bias_only     : per-element OLS bias only (no NN)
  A_mlp         : per-atom MLP, no message passing
   B_mpnn        : sum-aggregation MPNN with RBF + bond-type edges
                  (update fn: MLP by default, GRUCell if mpnn_update="gru" for
                   canonical Gilmer 2017 MPNN)
  C_gat         : multi-head GAT, no edge information
  D_gat_edge    : multi-head GAT with edge-bias attention (Uni-Mol style)

Common to all NN variants: input normalisation, node embed, skip connection
h_T + h_0, MLP readout. Element bias is initialised by OLS (see bias.py) and
frozen during training.
"""

import torch
import torch.nn as nn

from .bias import fit_element_bias
from .constants import NODE_DIM, MP_ROUNDS, N_HEADS, RBF_N
from .layers import (RBFExpansion, MPNNLayer,
                     MultiHeadGATLayer, MultiHeadGATEdgeLayer)

ARCHITECTURES = ("bias_only", "A_mlp", "B_mpnn", "C_gat", "D_gat_edge")

# The UnifiedModel class implements five architecture variants in a single class, with shared components like input normalization and element bias. 

class UnifiedModel(nn.Module):

    def __init__(self, input_dim, architecture, edge_feat_dim,
                node_dim=NODE_DIM, rounds=MP_ROUNDS, n_heads=N_HEADS,
                use_edge_bias=True, max_z=20,
                mpnn_update="mlp"): 
        super().__init__()
        assert architecture in ARCHITECTURES
        self.architecture = architecture
        self.input_dim = input_dim
        self.use_edge_bias = use_edge_bias

        self.elem_bias = nn.Embedding(max_z, 1)
        nn.init.zeros_(self.elem_bias.weight)
        self.elem_bias.weight.requires_grad_(False)

        self.register_buffer("shift", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))

        if architecture == "bias_only":
            return

        self.node_embed = nn.Sequential(
            nn.Linear(input_dim, node_dim), nn.GELU(),
            nn.LayerNorm(node_dim), nn.Linear(node_dim, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)

        if architecture == "A_mlp":
            self.mp = None
        elif architecture == "B_mpnn":
            self.rbf = RBFExpansion()
            ed = RBF_N + edge_feat_dim
            self.mp = nn.ModuleList(
                [MPNNLayer(node_dim, ed, update=mpnn_update)
                 for _ in range(rounds)])
        elif architecture == "C_gat":
            self.mp = nn.ModuleList(
                [MultiHeadGATLayer(node_dim, n_heads=n_heads)
                 for _ in range(rounds)])
        elif architecture == "D_gat_edge":
            self.rbf = RBFExpansion()
            ed = RBF_N + edge_feat_dim
            self.mp = nn.ModuleList(
                [MultiHeadGATEdgeLayer(node_dim, ed, n_heads=n_heads,
                                       use_edge_bias=use_edge_bias)
                 for _ in range(rounds)])

        self.readout = nn.Sequential(
            nn.Linear(node_dim, node_dim), nn.GELU(),
            nn.LayerNorm(node_dim), nn.Linear(node_dim, 1))

    # initialisation from the train split
    @torch.no_grad()
    def init_normalization(self, X, mask):
        flat = X[mask]
        self.shift.copy_(flat.mean(dim=0))
        self.scale.copy_(flat.std(dim=0).clamp(min=1e-6))

    @torch.no_grad()
    def init_element_bias(self, Z, y, mask):
        bias, r2 = fit_element_bias(Z, y, mask, self.elem_bias.num_embeddings)
        self.elem_bias.weight.copy_(bias.unsqueeze(-1))
        self.elem_bias.weight.requires_grad_(False)
        return r2

    # forward 
    def build_edge_feat(self, coords, edge_index, edge_type):
        """[RBF(d_ij), bond features] concat. Public for ablation wrappers."""
        B, E = coords.shape[0], edge_index.shape[-1]
        src, dst = edge_index[:, 0], edge_index[:, 1]
        b = torch.arange(B, device=coords.device).unsqueeze(-1).expand(B, E)
        r = torch.norm(coords[b, src] - coords[b, dst], dim=-1)
        return torch.cat([self.rbf(r), edge_type], dim=-1)

    def forward(self, X, Z, mask, coords=None, edge_index=None,
                edge_type=None, edge_mask=None, edge_feat=None):
        atom_e = self.elem_bias(Z.long()).squeeze(-1)

        if self.architecture == "bias_only":
            return (atom_e * mask.float()).sum(dim=-1)

        h0 = self.node_norm(self.node_embed((X - self.shift) / self.scale))
        h = h0
        if self.architecture == "B_mpnn":
            ef = edge_feat if edge_feat is not None \
                else self.build_edge_feat(coords, edge_index, edge_type)
            for layer in self.mp:
                h = layer(h, edge_index, ef, edge_mask)
        elif self.architecture == "C_gat":
            for layer in self.mp:
                h = layer(h, edge_index, edge_mask)
        elif self.architecture == "D_gat_edge":
            ef = edge_feat if edge_feat is not None \
                else self.build_edge_feat(coords, edge_index, edge_type)
            for layer in self.mp:
                h = layer(h, edge_index, ef, edge_mask)
        # A_mlp: no message passing

        atom_e = atom_e + self.readout(h + h0).squeeze(-1)
        return (atom_e * mask.float()).sum(dim=-1)


# The EdgeMaskedModel is a wrapper that allows ablation of edge features by zeroing out specific slices of the concatenated edge feature vector.

class EdgeMaskedModel(nn.Module):
    """Drop a slice of the 21-dim concatenated edge feature.

    The full feature is [RBF(d) (16), bond-type one-hot (4), bond length (1)].
    drop='rbf' zeroes the first 16 (geometric channel); drop='bond' zeroes
    the last 5 (chemical channel).
    """

    def __init__(self, base, drop):
        super().__init__()
        assert drop in ("rbf", "bond")
        self.base = base
        self.drop = drop

    def forward(self, X, Z, mask, coords, edge_index, edge_type, edge_mask):
        ef = self.base.build_edge_feat(coords, edge_index, edge_type)
        m = torch.ones(ef.shape[-1], device=ef.device, dtype=ef.dtype)
        if self.drop == "rbf":
            m[:RBF_N] = 0.0
        else:
            m[RBF_N:] = 0.0
        ef = ef * m
        return self.base(X, Z, mask, coords=coords, edge_index=edge_index,
                         edge_type=edge_type, edge_mask=edge_mask, edge_feat=ef)
