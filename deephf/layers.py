# AI Use: architectures designed by hand, implemented by hand and optimized using Claude (for example, the gat_softmax implementation was optimized by Claude for numerical stability and efficiency).
# Verified against standard implementations and other AI for chemistry codebases (eg PyG, DimeNet, etc...)

"""Per-layer building blocks used by the model architectures.

Common interface:
    layer(h, edge_index, edge_mask)                # GAT (no edge features)
    layer(h, edge_index, edge_feat, edge_mask)     # MPNN, GAT-edge

`h` is (B, A, D) padded node embeddings; edges live on flat lists of length E
masked by `edge_mask`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import N_HEADS, RBF_N, RBF_RANGE

# RBF expansion for interatomic distances, used to compute edge features from coordinates
class RBFExpansion(nn.Module):
    """Gaussian RBF expansion of pairwise distances."""

    def __init__(self, n=RBF_N, rmin=RBF_RANGE[0], rmax=RBF_RANGE[1]):
        super().__init__()
        self.register_buffer("centers", torch.linspace(rmin, rmax, n))
        self.gamma = 1.0 / ((rmax - rmin) / n) ** 2

    def forward(self, r):
        return torch.exp(-self.gamma * (r.unsqueeze(-1) - self.centers) ** 2)

# Sum-aggregation MPNN layer: m_ij = MLP([h_src, h_dst, e_ij]).
class MPNNLayer(nn.Module):
    """Sum-aggregation MPNN: m_ij = MLP([h_src, h_dst, e_ij])."""

    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim), nn.GELU(),
            nn.Linear(node_dim, node_dim))
        self.upd = nn.Sequential(
            nn.Linear(node_dim, node_dim), nn.GELU(),
            nn.Linear(node_dim, node_dim))
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, h, edge_index, edge_feat, edge_mask):
        B, A, D = h.shape
        E = edge_index.shape[-1]
        src, dst = edge_index[:, 0], edge_index[:, 1]
        b = torch.arange(B, device=h.device).unsqueeze(-1).expand(B, E)

        m = self.msg(torch.cat([h[b, src], h[b, dst], edge_feat], dim=-1))
        m = m * edge_mask.unsqueeze(-1).to(m.dtype)
        agg = torch.zeros_like(h, dtype=m.dtype)
        agg.scatter_add_(1, dst.unsqueeze(-1).expand(-1, -1, D), m)
        return self.norm(h + self.upd(agg.to(h.dtype)))

# gat_softmax: numerically stable masked softmax of attention scores over neighbors, per head. Useful for GAT and GAT-edge layers.
def gat_softmax(scores, dst, edge_mask, B, A, H):
    """Masked softmax of attention scores over neighbours, per head."""
    scores = scores.masked_fill(~edge_mask.unsqueeze(-1), float("-inf"))
    dst_h = dst.unsqueeze(-1).expand(-1, -1, H)
    mx = torch.full((B, A, H), float("-inf"),
                    device=scores.device, dtype=scores.dtype)
    mx = mx.scatter_reduce(1, dst_h, scores, reduce="amax", include_self=False)
    mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
    b = torch.arange(B, device=scores.device).unsqueeze(-1).expand(B, scores.shape[1])
    s = scores - mx[b, dst]
    exp = torch.exp(s) * edge_mask.unsqueeze(-1).to(s.dtype)
    denom = torch.zeros(B, A, H, device=scores.device, dtype=exp.dtype)
    denom.scatter_add_(1, dst_h, exp)
    return exp / denom[b, dst].clamp(min=1e-12)


# Multi-head GAT layer without edge features, used in an ablation study.
class MultiHeadGATLayer(nn.Module):
    """Multi-head GAT (Velickovic et al. 2018), no edge information."""

    def __init__(self, node_dim, n_heads=N_HEADS, neg_slope=0.2):
        super().__init__()
        assert node_dim % n_heads == 0
        self.H, self.Dh = n_heads, node_dim // n_heads
        self.W = nn.Linear(node_dim, node_dim, bias=False)
        self.a_src = nn.Linear(self.Dh, 1, bias=False)
        self.a_dst = nn.Linear(self.Dh, 1, bias=False)
        self.out = nn.Linear(node_dim, node_dim, bias=False)
        self.norm = nn.LayerNorm(node_dim)
        self.neg_slope = neg_slope

    def forward(self, h, edge_index, edge_mask):
        B, A, D = h.shape; H, Dh = self.H, self.Dh
        E = edge_index.shape[-1]
        src, dst = edge_index[:, 0], edge_index[:, 1]
        b = torch.arange(B, device=h.device).unsqueeze(-1).expand(B, E)

        Wh = self.W(h).view(B, A, H, Dh)
        scores = F.leaky_relu(
            self.a_dst(Wh).squeeze(-1)[b, dst]
            + self.a_src(Wh).squeeze(-1)[b, src],
            self.neg_slope)
        alpha = gat_softmax(scores, dst, edge_mask, B, A, H)

        msgs = Wh[b, src] * alpha.unsqueeze(-1)
        msgs = msgs * edge_mask.unsqueeze(-1).unsqueeze(-1).to(msgs.dtype)
        agg = torch.zeros(B, A, H, Dh, device=h.device, dtype=msgs.dtype)
        agg.scatter_add_(1, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, Dh),
                         msgs)
        return self.norm(h + self.out(F.elu(agg.view(B, A, D).to(h.dtype))))


# Multi-head GAT layer with an additive edge term in the attention score, used in the main architecture.
class MultiHeadGATEdgeLayer(nn.Module):
    """Multi-head GAT with an additive edge term in the attention score."""

    def __init__(self, node_dim, edge_dim, n_heads=N_HEADS, neg_slope=0.2,
                 use_edge_bias=True):
        super().__init__()
        assert node_dim % n_heads == 0
        self.H, self.Dh = n_heads, node_dim // n_heads
        self.use_edge_bias = use_edge_bias
        self.W = nn.Linear(node_dim, node_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, node_dim, bias=False)
        self.a_src = nn.Linear(self.Dh, 1, bias=False)
        self.a_dst = nn.Linear(self.Dh, 1, bias=False)
        self.a_edge = nn.Linear(self.Dh, 1, bias=False)
        self.out = nn.Linear(node_dim, node_dim, bias=False)
        self.norm = nn.LayerNorm(node_dim)
        self.neg_slope = neg_slope

    def forward(self, h, edge_index, edge_feat, edge_mask):
        B, A, D = h.shape; H, Dh = self.H, self.Dh
        E = edge_index.shape[-1]
        src, dst = edge_index[:, 0], edge_index[:, 1]
        b = torch.arange(B, device=h.device).unsqueeze(-1).expand(B, E)

        Wh = self.W(h).view(B, A, H, Dh)
        We = self.W_edge(edge_feat).view(B, E, H, Dh)
        edge_bias = self.a_edge(We).squeeze(-1) if self.use_edge_bias else 0.0
        scores = F.leaky_relu(
            self.a_dst(Wh).squeeze(-1)[b, dst]
            + self.a_src(Wh).squeeze(-1)[b, src]
            + edge_bias,
            self.neg_slope)
        alpha = gat_softmax(scores, dst, edge_mask, B, A, H)

        msgs = Wh[b, src] * alpha.unsqueeze(-1)
        msgs = msgs * edge_mask.unsqueeze(-1).unsqueeze(-1).to(msgs.dtype)
        agg = torch.zeros(B, A, H, Dh, device=h.device, dtype=msgs.dtype)
        agg.scatter_add_(1, dst.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, Dh),
                         msgs)
        return self.norm(h + self.out(F.elu(agg.view(B, A, D).to(h.dtype))))
