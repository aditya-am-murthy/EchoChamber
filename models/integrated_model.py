"""
Integrated NLP + GNN model for virality prediction
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from torch_geometric.data import Data

from .nlp.text_processor import TextProcessor
from .gnn.gnn_model import GNNModel, GNNConfig
from .graph.graph_builder import GraphBuilder


class IntegratedNLPGNNModel(nn.Module):
    """
    Combined NLP + GNN model for predicting post virality
    
    Architecture:
    1. NLP module processes text to extract features
    2. GNN module learns social network patterns
    3. Features are combined and passed to prediction head
    """
    
    def __init__(
        self,
        nlp_model_name: str = "bert-base-uncased",
        gnn_config: Optional[GNNConfig] = None,
        nlp_feature_dim: int = 768,
        gnn_output_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        include_news_context: bool = False,
        news_context_dim: int = 0,
        device: str = None
    ):
        """
        Initialize integrated model
        
        Args:
            nlp_model_name: HuggingFace model for NLP
            gnn_config: Configuration for GNN
            nlp_feature_dim: Dimension of NLP features
            gnn_output_dim: Output dimension of GNN
            hidden_dim: Hidden dimension for fusion layer
            dropout: Dropout rate
            include_news_context: Whether to include news headlines (Experiment 1)
            news_context_dim: Dimension of news context features
            device: Device to run on
        """
        super(IntegratedNLPGNNModel, self).__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.include_news_context = include_news_context
        
        # NLP module (frozen during training, used for feature extraction)
        self.text_processor = TextProcessor(nlp_model_name, device=self.device)
        
        # Calculate input dimensions
        # NLP features: embeddings + linguistic + sentiment + topic
        # Actual: 768 (embeddings) + 11 (linguistic) + 4 (sentiment) + 3 (topic) = 786
        nlp_total_dim = nlp_feature_dim + 11 + 4 + 3  # embeddings + linguistic (11) + sentiment + topic
        self.nlp_total_dim = nlp_total_dim
        
        # GNN config
        if gnn_config is None:
            gnn_config = GNNConfig(
                input_dim=7,  # Node features: 6 node types + 1 interaction count
                hidden_dim=256,
                num_layers=3,
                output_dim=gnn_output_dim,
                dropout=dropout
            )
        
        # GNN module
        self.gnn = GNNModel(gnn_config).to(self.device)
        
        # Feature fusion
        fusion_input_dim = nlp_total_dim + gnn_output_dim
        if include_news_context:
            fusion_input_dim += news_context_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output virality score [0, 1]
        )
    
    def extract_nlp_features(self, text: str) -> torch.Tensor:
        """Extract all NLP features from text"""
        features = self.text_processor.extract_all_features(text)
        
        # Concatenate all features
        nlp_features = np.concatenate([
            features['embeddings'],
            features['linguistic'],
            features['sentiment'],
            features['topic']
        ])
        
        return torch.tensor(nlp_features, dtype=torch.float32).to(self.device)
    
    def forward(
        self,
        text: str,
        graph: Data,
        news_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            text: Post text
            graph: PyTorch Geometric graph
            news_context: Optional news headline features (Experiment 1)
        
        Returns:
            Virality score [0, 1]
        """
        # Extract NLP features
        nlp_features = self.extract_nlp_features(text)
        
        # GNN forward pass
        graph = graph.to(self.device)
        # Ensure graph.x has the correct dimension
        # Clone to avoid modifying the original graph
        graph_x = graph.x.clone()
        if graph_x.shape[1] != self.gnn.config.input_dim:
            # If dimension mismatch, create a projection layer
            if not hasattr(self, 'node_feature_proj'):
                self.node_feature_proj = nn.Linear(graph_x.shape[1], self.gnn.config.input_dim).to(self.device)
            graph_x = self.node_feature_proj(graph_x)
        
        gnn_features = self.gnn(graph_x, graph.edge_index)
        
        # Combine features
        # Ensure dimensions match
        if gnn_features.dim() > 1:
            gnn_features = gnn_features.squeeze(0)
        if gnn_features.dim() == 0:
            gnn_features = gnn_features.unsqueeze(0)
        
        # Debug: print dimensions if mismatch
        if nlp_features.shape[0] + gnn_features.shape[0] != self.fusion[0].in_features:
            print(f"Warning: Dimension mismatch - NLP: {nlp_features.shape[0]}, GNN: {gnn_features.shape[0]}, Expected: {self.fusion[0].in_features}")
        
        combined = torch.cat([nlp_features, gnn_features])
        
        if self.include_news_context and news_context is not None:
            news_context = news_context.to(self.device)
            combined = torch.cat([combined, news_context])
        
        # Fusion
        fused = self.fusion(combined)
        
        # Prediction
        virality_score = self.predictor(fused)
        
        return virality_score
    
    def predict(
        self,
        text: str,
        graph: Data,
        news_context: Optional[torch.Tensor] = None
    ) -> float:
        """Make prediction (inference mode)"""
        self.eval()
        with torch.no_grad():
            score = self.forward(text, graph, news_context)
            return score.item()

