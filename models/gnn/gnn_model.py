"""
Graph Neural Network model for learning social network patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from dataclasses import dataclass
from typing import Optional


@dataclass
class GNNConfig:
    """Configuration for GNN model"""
    input_dim: int = 768  # BERT embedding dimension
    hidden_dim: int = 256
    num_layers: int = 3
    output_dim: int = 128
    dropout: float = 0.3
    gnn_type: str = 'GCN'  # 'GCN', 'GAT', or 'SAGE'
    use_batch_norm: bool = True
    pooling: str = 'mean'  # 'mean', 'max', or 'both'


class GNNModel(nn.Module):
    """Graph Neural Network for learning user-post interaction patterns"""
    
    def __init__(self, config: GNNConfig):
        super(GNNModel, self).__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if config.use_batch_norm else None
        
        for i in range(config.num_layers):
            if i == 0:
                in_dim = config.hidden_dim
            else:
                in_dim = config.hidden_dim
            
            if config.gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(in_dim, config.hidden_dim))
            elif config.gnn_type == 'GAT':
                self.gnn_layers.append(GATConv(in_dim, config.hidden_dim, heads=4, concat=False))
            elif config.gnn_type == 'SAGE':
                self.gnn_layers.append(SAGEConv(in_dim, config.hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {config.gnn_type}")
            
            if config.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(config.hidden_dim))
        
        # Output projection
        if config.pooling == 'both':
            pool_dim = config.hidden_dim * 2
        else:
            pool_dim = config.hidden_dim
        
        self.output_proj = nn.Sequential(
            nn.Linear(pool_dim, config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.output_dim, config.output_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for nodes (optional)
        
        Returns:
            Graph-level representation [batch_size, output_dim]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_new = gnn_layer(x, edge_index)
            
            if self.config.use_batch_norm:
                x_new = self.batch_norms[i](x_new)
            
            x_new = F.relu(x_new)
            x = F.dropout(x_new, p=self.config.dropout, training=self.training)
        
        # Graph-level pooling
        if batch is None:
            # Single graph
            if self.config.pooling == 'mean':
                graph_repr = global_mean_pool(x, batch=None)
            elif self.config.pooling == 'max':
                graph_repr = global_max_pool(x, batch=None)
            elif self.config.pooling == 'both':
                mean_pool = global_mean_pool(x, batch=None)
                max_pool = global_max_pool(x, batch=None)
                graph_repr = torch.cat([mean_pool, max_pool], dim=1)
            else:
                raise ValueError(f"Unknown pooling: {self.config.pooling}")
        else:
            # Batched graphs
            if self.config.pooling == 'mean':
                graph_repr = global_mean_pool(x, batch)
            elif self.config.pooling == 'max':
                graph_repr = global_max_pool(x, batch)
            elif self.config.pooling == 'both':
                mean_pool = global_mean_pool(x, batch)
                max_pool = global_max_pool(x, batch)
                graph_repr = torch.cat([mean_pool, max_pool], dim=1)
            else:
                raise ValueError(f"Unknown pooling: {self.config.pooling}")
        
        # Output projection
        output = self.output_proj(graph_repr)
        
        return output

