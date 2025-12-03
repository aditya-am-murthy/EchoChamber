"""
Heterogeneous Graph Neural Network model for social network graphs.

Phase 1:
    - Define HeteroGNNConfig
    - Implement HeteroGNNModel using torch_geometric.nn.HeteroConv
    - Support basic forward pass:
        (x_dict, edge_index_dict) -> {post_repr, user_repr, tag_repr, graph_repr}

Later phases will add:
    - Temporal attention over (post, precedes, post) edges
    - Hierarchical pooling by node / user type
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, GATConv, SAGEConv

from .temporal_attention import TemporalAttentionLayer
from .hierarchical_pooling import HierarchicalPooling


EdgeType = Tuple[str, str, str]


@dataclass
class HeteroGNNConfig:
    """Configuration for heterogeneous GNN."""

    hidden_dim: int = 256
    num_layers: int = 3
    output_dim: int = 128
    dropout: float = 0.3

    # Relation-specific layer types, keyed by (src, rel, dst)
    relation_layers: Dict[EdgeType, str] = field(
        default_factory=lambda: {
            ("user", "authors", "post"): "SAGE",
            ("user", "likes", "post"): "SAGE",
            ("user", "comments", "post"): "GAT",  # GAT for important comments
            ("post", "mentions", "user"): "SAGE",
            ("post", "has_tag", "tag"): "SAGE",
            ("post", "replies_to", "post"): "GCN",
            ("post", "precedes", "post"): "GAT",  # Placeholder for TemporalGAT in later phase
        }
    )

    use_temporal_attention: bool = True
    use_hierarchical_pooling: bool = True
    pooling_aggr: str = "mean"  # 'mean', 'max', 'attention' (attention added in later phase)


class HeteroGNNModel(nn.Module):
    """Heterogeneous GNN for social network graphs.

    This model is compatible with HeteroData returned by GraphBuilder.build_hetero_graph().

    Design (Phase 1):
        - Per-node-type linear projection to a shared hidden_dim
        - num_layers stacked HeteroConv layers with relation-specific conv types
        - Simple type-wise global pooling (mean over nodes of each type)
        - Concatenated graph representation across node types
    """

    def __init__(self, config: HeteroGNNConfig, metadata: Tuple[List[str], List[EdgeType]]):
        """
        Args:
            config: HeteroGNNConfig
            metadata: (node_types, edge_types) tuple from HeteroData.metadata()
        """
        super().__init__()
        self.config = config

        node_types, edge_types = metadata
        self.node_types: List[str] = list(node_types)
        self.edge_types: List[EdgeType] = list(edge_types)

        # Per-node-type input projections are built lazily on first forward
        self.input_proj = nn.ModuleDict()
        self._input_proj_initialized = False

        # Optional temporal attention for (post, 'precedes', 'post') relation
        self.temporal_attention: Optional[TemporalAttentionLayer] = None

        # Build stacked HeteroConv layers
        self.convs = nn.ModuleList()
        for _ in range(self.config.num_layers):
            conv_dict = nn.ModuleDict()
            for edge_type in self.edge_types:
                conv_name = self.config.relation_layers.get(edge_type, "SAGE")
                conv = self._make_conv_for_relation(conv_name)
                conv_dict[self._edge_type_to_str(edge_type)] = conv

            hetero_conv = HeteroConv(
                {
                    self._str_to_edge_type(name): conv
                    for name, conv in conv_dict.items()
                },
                aggr="sum",
            )
            self.convs.append(hetero_conv)

        # Graph-level projection from pooled representations
        # For hierarchical pooling we fix 4 groups: post, regular_user, influencer_user, tag
        self.graph_proj: Optional[nn.Module] = None
        self.pooling = HierarchicalPooling(hidden_dim=self.config.hidden_dim)

    # ---------------------------------------------------------------------
    # Helper builders
    # ---------------------------------------------------------------------
    def _make_conv_for_relation(self, conv_name: str) -> nn.Module:
        """Create a convolution layer for a single relation.

        Uses lazy input channels (-1) so it can adapt to the hidden_dim at runtime.
        """
        out_dim = self.config.hidden_dim
        conv_name = conv_name.upper()

        if conv_name == "GCN":
            # For homogeneous use (src and dst share feature space)
            return GCNConv(-1, out_dim, add_self_loops=False)
        elif conv_name == "GAT":
            # heads=4 with concat=False keeps feature size = out_dim
            # Disable internal self-loop handling for hetero use.
            return GATConv(
                (-1, -1),
                out_dim,
                heads=4,
                concat=False,
                add_self_loops=False,
            )
        elif conv_name == "SAGE":
            # SAGEConv supports bipartite (src, dst) feature tuples
            return SAGEConv((-1, -1), out_dim)
        else:
            # Fallback to SAGE for unknown types (e.g. placeholder "TemporalGAT" in Phase 1)
            return SAGEConv((-1, -1), out_dim)

    @staticmethod
    def _edge_type_to_str(edge_type: EdgeType) -> str:
        """Serialize edge type tuple to string key."""
        return "__".join(edge_type)

    @staticmethod
    def _str_to_edge_type(key: str) -> EdgeType:
        """Deserialize string key back to edge type tuple."""
        src, rel, dst = key.split("__")
        return src, rel, dst

    def _init_input_proj(self, x_dict: Dict[str, torch.Tensor]):
        """Initialize per-node-type input projections based on observed feature dims."""
        for node_type in self.node_types:
            if node_type not in x_dict:
                continue
            in_dim = x_dict[node_type].size(-1)
            if node_type not in self.input_proj:
                self.input_proj[node_type] = nn.Linear(in_dim, self.config.hidden_dim)
        self._input_proj_initialized = True

    def _get_graph_proj(self) -> nn.Module:
        """Lazily build graph-level projection based on pooling strategy."""
        if self.graph_proj is None:
            if self.config.use_hierarchical_pooling:
                in_dim = self.config.hidden_dim * 4  # post, user_regular, user_influencer, tag
            else:
                in_dim = self.config.hidden_dim * len(self.node_types)

            self.graph_proj = nn.Sequential(
                nn.Linear(in_dim, self.config.output_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.output_dim, self.config.output_dim),
            )
        return self.graph_proj

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        edge_attr_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
        user_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: Dict of node features {node_type: [num_nodes_t, feat_dim_t]}
            edge_index_dict: Dict of edge indices {edge_type: [2, num_edges]}

        Returns:
            Dict with:
                - 'post_repr': pooled representation for post nodes
                - 'user_repr': pooled representation for user nodes
                - 'tag_repr': pooled representation for tag nodes
                - 'graph_repr': concatenated graph-level representation
        """
        # Lazily build input projections on first call
        if not self._input_proj_initialized:
            self._init_input_proj(x_dict)

        # Project inputs to common hidden_dim
        h_dict: Dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            if node_type not in x_dict:
                continue
            h = self.input_proj[node_type](x_dict[node_type])
            h = F.relu(h)
            h = F.dropout(h, p=self.config.dropout, training=self.training)
            h_dict[node_type] = h

        # Ensure every node type from metadata has an entry in h_dict
        # (even if there are currently zero nodes of that type).
        device = next(iter(h_dict.values())).device if h_dict else self.input_proj[
            self.node_types[0]
        ].weight.device
        for node_type in self.node_types:
            if node_type not in h_dict:
                h_dict[node_type] = torch.zeros(
                    0, self.config.hidden_dim, device=device
                )

        # Filter out relations with no edges to avoid PyG aggregation on empty inputs
        filtered_edge_index_dict: Dict[EdgeType, torch.Tensor] = {}
        filtered_edge_attr_dict: Dict[EdgeType, torch.Tensor] = {}
        for edge_type, edge_index in edge_index_dict.items():
            if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
                filtered_edge_index_dict[edge_type] = edge_index
                if edge_attr_dict is not None and edge_type in edge_attr_dict:
                    filtered_edge_attr_dict[edge_type] = edge_attr_dict[edge_type]

        # Message passing layers
        for layer_idx, conv in enumerate(self.convs):
            h_dict = conv(h_dict, filtered_edge_index_dict)

            # Optional temporal attention over (post, 'precedes', 'post') edges
            if (
                self.config.use_temporal_attention
                and ("post", "precedes", "post") in filtered_edge_index_dict
                and "post" in h_dict
            ):
                # Lazily build temporal attention layer once hidden_dim is known
                if self.temporal_attention is None:
                    self.temporal_attention = TemporalAttentionLayer(
                        hidden_dim=self.config.hidden_dim,
                        dropout=self.config.dropout,
                    )

                post_edge_index = filtered_edge_index_dict[("post", "precedes", "post")]

                # Edge weights typically come from hetero[('post','precedes','post')].edge_attr
                if (
                    filtered_edge_attr_dict is not None
                    and ("post", "precedes", "post") in filtered_edge_attr_dict
                ):
                    edge_weight = filtered_edge_attr_dict[("post", "precedes", "post")].to(
                        post_edge_index.device
                    )
                else:
                    # Fallback to ones if weights are not provided
                    if isinstance(post_edge_index, torch.Tensor):
                        num_edges = post_edge_index.size(1)
                        edge_weight = torch.ones(num_edges, device=post_edge_index.device)
                    else:
                        edge_weight = torch.ones(
                            0, device=next(iter(h_dict.values())).device
                        )

                h_post = h_dict["post"]
                h_post_updated = self.temporal_attention(h_post, post_edge_index, edge_weight)
                h_dict["post"] = h_post_updated

            # Non-linearity and dropout per node type
            for node_type, h in h_dict.items():
                h = F.relu(h)
                h = F.dropout(h, p=self.config.dropout, training=self.training)
                h_dict[node_type] = h

        # Pooling
        if self.config.use_hierarchical_pooling:
            pools = self.pooling(h_dict, user_types=user_types)
            post_repr = pools["post"]
            user_regular_repr = pools["user_regular"]
            user_influencer_repr = pools["user_influencer"]
            tag_repr = pools["tag"]
            graph_input = pools["graph"]
        else:
            pooled: Dict[str, torch.Tensor] = {}
            for node_type, h in h_dict.items():
                if h.numel() == 0:
                    continue
                pooled[node_type] = h.mean(dim=0)  # [hidden_dim]

            device = next(iter(h_dict.values())).device
            post_repr = pooled.get("post", torch.zeros(self.config.hidden_dim, device=device))
            user_regular_repr = pooled.get("user", torch.zeros(self.config.hidden_dim, device=device))
            user_influencer_repr = torch.zeros(self.config.hidden_dim, device=device)
            tag_repr = pooled.get("tag", torch.zeros(self.config.hidden_dim, device=device))
            graph_input = torch.cat(
                [post_repr, user_regular_repr, user_influencer_repr, tag_repr], dim=-1
            )

        # Project to final graph representation
        graph_proj = self._get_graph_proj()
        graph_repr = graph_proj(graph_input)

        return {
            "post_repr": post_repr,
            "user_regular_repr": user_regular_repr,
            "user_influencer_repr": user_influencer_repr,
            "tag_repr": tag_repr,
            "graph_repr": graph_repr,
        }


