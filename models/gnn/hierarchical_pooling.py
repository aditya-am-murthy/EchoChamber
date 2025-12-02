"""
Hierarchical pooling for heterogeneous social graphs.

Groups:
    - Posts
    - Regular users
    - Influencer users
    - Tags

Outputs:
    - post_pool:        [hidden_dim]
    - user_regular:     [hidden_dim]
    - user_influencer:  [hidden_dim]
    - tag_pool:         [hidden_dim]
    - graph_repr:       concatenation of the above [hidden_dim * 4]
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class HierarchicalPooling(nn.Module):
    """Hierarchical pooling by node type and user type."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        user_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: Dict of node embeddings {node_type: [num_nodes_t, hidden_dim]}
            user_types: Optional tensor of shape [num_users] with:
                        - 1 for influencer users
                        - 0 for regular users

        Returns:
            Dict with keys:
                - 'post'
                - 'user_regular'
                - 'user_influencer'
                - 'tag'
                - 'graph'
        """
        device = next(iter(x_dict.values())).device

        # ----- Post pooling -----
        if "post" in x_dict and x_dict["post"].numel() > 0:
            post_pool = x_dict["post"].mean(dim=0)
        else:
            post_pool = torch.zeros(self.hidden_dim, device=device)

        # ----- User pooling (regular vs influencer) -----
        if "user" in x_dict and x_dict["user"].numel() > 0:
            user_x = x_dict["user"]
            num_users = user_x.size(0)

            if user_types is None:
                # All users treated as regular by default
                regular_mask = torch.ones(num_users, dtype=torch.bool, device=device)
                influencer_mask = torch.zeros(num_users, dtype=torch.bool, device=device)
            else:
                user_types = user_types.to(device)
                influencer_mask = user_types.bool()
                regular_mask = ~influencer_mask

            if regular_mask.any():
                user_regular = user_x[regular_mask].mean(dim=0)
            else:
                user_regular = torch.zeros(self.hidden_dim, device=device)

            if influencer_mask.any():
                user_influencer = user_x[influencer_mask].mean(dim=0)
            else:
                user_influencer = torch.zeros(self.hidden_dim, device=device)
        else:
            user_regular = torch.zeros(self.hidden_dim, device=device)
            user_influencer = torch.zeros(self.hidden_dim, device=device)

        # ----- Tag pooling -----
        if "tag" in x_dict and x_dict["tag"].numel() > 0:
            tag_pool = x_dict["tag"].mean(dim=0)
        else:
            tag_pool = torch.zeros(self.hidden_dim, device=device)

        # Concatenate for overall graph representation
        graph_repr = torch.cat(
            [post_pool, user_regular, user_influencer, tag_pool], dim=-1
        )

        return {
            "post": post_pool,
            "user_regular": user_regular,
            "user_influencer": user_influencer,
            "tag": tag_pool,
            "graph": graph_repr,
        }


