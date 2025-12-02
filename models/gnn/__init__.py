"""
Graph Neural Network models.

Exports:
    - GNNModel / GNNConfig: legacy homogeneous GNN
    - HeteroGNNModel / HeteroGNNConfig: heterogeneous GNN for HeteroData graphs
"""

from .gnn_model import GNNModel, GNNConfig
from .hetero_gnn_model import HeteroGNNModel, HeteroGNNConfig
from .temporal_attention import TemporalAttentionLayer
from .hierarchical_pooling import HierarchicalPooling

__all__ = [
    "GNNModel",
    "GNNConfig",
    "HeteroGNNModel",
    "HeteroGNNConfig",
    "TemporalAttentionLayer",
    "HierarchicalPooling",
]

