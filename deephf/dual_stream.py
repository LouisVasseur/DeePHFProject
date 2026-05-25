#AI Use: architecture designed by hand, implemented by hand 
#and optimized using Claude (eg the way the main and late streams are split, the way edge features are computed, etc...)


"""Dual-stream GAT-edge architecture used by the K-sweep (appendix).

The 17+108+N_soap input is split into a main and a late block:
    late='elec' : main=[chem | soap],  late=[elec]
    late='soap' : main=[chem | elec],  late=[soap]

Both blocks are embedded and then processed by GAT-edge layers (3 rounds for
the main stream, K rounds for the late stream); their final node embeddings
are concatenated and fed to the readout. K=0 means the late embedding is
passed straight to the readout.
"""

import torch
import torch.nn as nn

from .bias import fit_element_bias
from .constants import CHEM_DIM, ELEC_DIM, NODE_DIM, MP_ROUNDS, N_HEADS, RBF_N
from .layers import RBFExpansion, MultiHeadGATEdgeLayer


'''Dual-stream GAT-edge architecture used by the K-sweep (appendix) used to fit the bias and in an ablation study. The 17+108+N_soap input is split into a main and a late block:
    late='elec' : main=[chem | soap],  late=[elec]
    late='soap' : main=[chem | elec],  late=[soap]
Both blocks are embedded and then processed by GAT-edge layers (3 rounds for the main stream, K rounds for the late stream); their final node embeddings
are concatenated and fed to the readout. K=0 means the late embedding is passed straight to the readout.
'''
class DualStreamGATEdge(nn.Module):


    def __init__(self, total_input_dim, edge_feat_dim, late_kind, k,
                 node_dim=NODE_DIM, n_heads=N_HEADS, max_z=20):
        super().__init__()
        assert late_kind in ("elec", "soap")
        self.late_kind = late_kind
        self.k = k

        if late_kind == "elec":
            self.main_dim = total_input_dim - ELEC_DIM
            self.late_dim = ELEC_DIM
        else:
            self.main_dim = CHEM_DIM + ELEC_DIM
            self.late_dim = total_input_dim - self.main_dim

        self.elem_bias = nn.Embedding(max_z, 1)
        nn.init.zeros_(self.elem_bias.weight)
        self.elem_bias.weight.requires_grad_(False)

        self.register_buffer("shift", torch.zeros(total_input_dim))
        self.register_buffer("scale", torch.ones(total_input_dim))

        self.rbf = RBFExpansion()
        ed = RBF_N + edge_feat_dim

        self.embed_main = nn.Sequential(
            nn.Linear(self.main_dim, node_dim), nn.GELU(),
            nn.LayerNorm(node_dim), nn.Linear(node_dim, node_dim))
        self.embed_late = nn.Sequential(
            nn.Linear(self.late_dim, node_dim), nn.GELU(),
            nn.LayerNorm(node_dim), nn.Linear(node_dim, node_dim))

        self.mp_main = nn.ModuleList(
            [MultiHeadGATEdgeLayer(node_dim, ed, n_heads=n_heads)
             for _ in range(MP_ROUNDS)])
        self.mp_late = nn.ModuleList(
            [MultiHeadGATEdgeLayer(node_dim, ed, n_heads=n_heads)
             for _ in range(k)])

        self.readout = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim), nn.GELU(),
            nn.LayerNorm(node_dim), nn.Linear(node_dim, 1))


    # initialization methods (called once on the train split, then frozen)
    @torch.no_grad()
    def init_normalization(self, X, mask):
        flat = X[mask]
        self.shift.copy_(flat.mean(dim=0))
        self.scale.copy_(flat.std(dim=0).clamp(min=1e-6))

    # fit the element bias and freeze it (called once on train split, then frozen).
    # Returns the R^2 of the fit, which is a measure of how well bias alone can explain the target.
    @torch.no_grad()
    def init_element_bias(self, Z, y, mask):
        bias, r2 = fit_element_bias(Z, y, mask, self.elem_bias.num_embeddings)
        self.elem_bias.weight.copy_(bias.unsqueeze(-1))
        self.elem_bias.weight.requires_grad_(False)
        return r2

    # split the input features into main and late blocks, depending on late_kind.
    def _split(self, X_norm):
        if self.late_kind == "elec":
            main = torch.cat([X_norm[..., :CHEM_DIM],
                              X_norm[..., CHEM_DIM + ELEC_DIM:]], dim=-1)
            late = X_norm[..., CHEM_DIM:CHEM_DIM + ELEC_DIM]
        else:
            main = X_norm[..., :CHEM_DIM + ELEC_DIM]
            late = X_norm[..., CHEM_DIM + ELEC_DIM:]
        return main, late

    # compute edge features by concatenating RBF expansion of interatomic distances and the input edge features.
    def _edge_feat(self, coords, edge_index, edge_type):
        B, E = coords.shape[0], edge_index.shape[-1]
        src, dst = edge_index[:, 0], edge_index[:, 1]
        b = torch.arange(B, device=coords.device).unsqueeze(-1).expand(B, E)
        r = torch.norm(coords[b, src] - coords[b, dst], dim=-1)
        return torch.cat([self.rbf(r), edge_type], dim=-1)

    # forward method: embed main and late features, process them with GAT-edge layers, add the element bias, concatenate and read out.
    def forward(self, X, Z, mask, coords, edge_index, edge_type, edge_mask):
        X_norm = (X - self.shift) / self.scale
        main, late = self._split(X_norm)
        h_main = self.embed_main(main)
        h_late = self.embed_late(late)
        ef = self._edge_feat(coords, edge_index, edge_type)

        for layer in self.mp_main:
            h_main = layer(h_main, edge_index, ef, edge_mask)
        for layer in self.mp_late:
            h_late = layer(h_late, edge_index, ef, edge_mask)

        atom_e = self.elem_bias(Z.long()).squeeze(-1) \
            + self.readout(torch.cat([h_main, h_late], dim=-1)).squeeze(-1)
        return (atom_e * mask.float()).sum(dim=-1)
