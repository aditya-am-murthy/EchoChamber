"""
Temporal attention module for post→post precedence edges.

This layer is designed to work with the (post, 'precedes', post) relation
built in GraphBuilder.build_temporal_edges(), where edge_attr encodes
exponential time-decay weights.

The attention mechanism uses these decay weights as priors and learns
additional modulation via a small MLP over [h_src, h_tgt, edge_weight]:

    score_ij = MLP([h_i, h_j, w_ij])
    alpha_ij = softmax_j(score_ij)
    h'_j = sum_i alpha_ij * h_i
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionLayer(nn.Module):
    """Temporal attention for (post, 'precedes', post) edges."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        """
        Args:
            hidden_dim: Dimension of node embeddings h_i, h_j
            dropout: Dropout applied to attention scores
        """
        super().__init__()
        # Input: [h_src, h_tgt, edge_weight] → scalar score
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node embeddings for 'post' nodes [num_posts, hidden_dim]
            edge_index: Indices [2, num_edges] for (src, tgt) in (post, 'precedes', post)
            edge_weight: Time-decay weights [num_edges, 1] or [num_edges]

        Returns:
            Updated node embeddings [num_posts, hidden_dim] with temporal attention.
        """
        if edge_index.numel() == 0:
            return x

        src, tgt = edge_index  # [num_edges]
        h_src = x[src]  # [E, H]
        h_tgt = x[tgt]  # [E, H]

        if edge_weight.dim() == 1:
            w = edge_weight.unsqueeze(-1)  # [E, 1]
        else:
            w = edge_weight  # [E, 1]

        # Concatenate features for MLP
        concat = torch.cat([h_src, h_tgt, w], dim=-1)  # [E, 2H+1]
        scores = self.mlp(concat).squeeze(-1)  # [E]

        # Softmax over incoming edges per target node
        num_nodes = x.size(0)
        alpha = self._softmax_edge(scores, tgt, num_nodes)  # [E]

        if self.dropout > 0 and self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Aggregate messages: sum_i alpha_ij * h_src_i
        out = torch.zeros_like(x)
        out.index_add_(0, tgt, alpha.unsqueeze(-1) * h_src)

        # Residual connection
        return x + out

    @staticmethod
    def _softmax_edge(
        scores: torch.Tensor, tgt_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Compute softmax over edges grouped by target node."""
        # For numerical stability, subtract max per target
        max_per_tgt = torch.full((num_nodes,), -float("inf"), device=scores.device)
        max_per_tgt.index_put_(
            (tgt_index,), scores, accumulate=True
        )  # running max via accumulate
        max_per_tgt = torch.where(
            max_per_tgt == -float("inf"),
            torch.zeros_like(max_per_tgt),
            max_per_tgt,
        )

        scores_centered = scores - max_per_tgt[tgt_index]
        exp_scores = scores_centered.exp()

        # Sum exp per target
        sum_exp_per_tgt = torch.zeros(num_nodes, device=scores.device)
        sum_exp_per_tgt.index_add_(0, tgt_index, exp_scores)
        sum_exp_per_tgt = torch.clamp(sum_exp_per_tgt, min=1e-12)

        alpha = exp_scores / sum_exp_per_tgt[tgt_index]
        return alpha


