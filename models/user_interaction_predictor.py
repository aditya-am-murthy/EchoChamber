"""
Model for predicting which users will interact with a post
This is a link prediction problem: predict edges (user-post interactions)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from torch_geometric.data import Data

from .nlp.text_processor import TextProcessor
from .gnn.gnn_model import GNNModel, GNNConfig


class UserInteractionPredictor(nn.Module):
    """
    Predicts which users will interact with a post
    
    Architecture:
    1. NLP processes post text → extracts features
    2. GNN processes social network → learns user embeddings
    3. For each user, compute interaction probability based on:
       - User embedding from GNN
       - Post text features from NLP
       - User's historical interaction patterns
    """
    
    def __init__(
        self,
        nlp_model_name: str = "bert-base-uncased",
        gnn_config: Optional[GNNConfig] = None,
        nlp_feature_dim: int = 768,
        user_embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        device: str = None
    ):
        """
        Initialize user interaction predictor
        
        Args:
            nlp_model_name: HuggingFace model for NLP
            gnn_config: Configuration for GNN
            nlp_feature_dim: Dimension of NLP features
            user_embedding_dim: Dimension of user embeddings from GNN
            hidden_dim: Hidden dimension for interaction predictor
            dropout: Dropout rate
            device: Device to run on
        """
        super(UserInteractionPredictor, self).__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # NLP module
        self.text_processor = TextProcessor(nlp_model_name, device=self.device)
        
        # Calculate NLP feature dimensions
        nlp_total_dim = nlp_feature_dim + 11 + 4 + 3  # embeddings + linguistic + sentiment + topic
        
        # GNN config
        if gnn_config is None:
            gnn_config = GNNConfig(
                input_dim=7,  # Node features: 6 node types + 1 interaction count
                hidden_dim=256,
                num_layers=3,
                output_dim=user_embedding_dim,
                dropout=dropout,
                use_batch_norm=False
            )
        
        # GNN module - outputs user embeddings
        self.gnn = GNNModel(gnn_config).to(self.device)
        
        # Interaction predictor: takes user embedding + post features → interaction probability
        interaction_input_dim = user_embedding_dim + nlp_total_dim
        self.interaction_predictor = nn.Sequential(
            nn.Linear(interaction_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Probability of interaction [0, 1]
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
        user_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            text: Post text
            graph: PyTorch Geometric graph with users and post
            user_indices: Optional tensor of user node indices to predict for
                         If None, predicts for all user nodes
        
        Returns:
            Interaction probabilities for each user [num_users]
        """
        # Extract NLP features
        nlp_features = self.extract_nlp_features(text)
        
        # GNN forward pass - get user embeddings
        graph = graph.to(self.device)
        graph_x = graph.x.clone()
        if graph_x.shape[1] != self.gnn.config.input_dim:
            if not hasattr(self, 'node_feature_proj'):
                self.node_feature_proj = nn.Linear(graph_x.shape[1], self.gnn.config.input_dim).to(self.device)
            graph_x = self.node_feature_proj(graph_x)
        
        # Get node embeddings from GNN
        node_embeddings = self.gnn(graph_x, graph.edge_index)
        
        # Identify user nodes (exclude post nodes)
        # Assuming post nodes are marked or we can identify them
        # For now, use all nodes except the first one (assuming first is post)
        if user_indices is None:
            # Get all user node indices (all except post nodes)
            # This is a simplification - in practice, you'd track node types
            num_nodes = node_embeddings.shape[0]
            user_indices = torch.arange(1, num_nodes, device=self.device)  # Skip first node (post)
        
        user_embeddings = node_embeddings[user_indices]
        
        # For each user, combine user embedding with post features
        # Expand post features to match number of users
        post_features = nlp_features.unsqueeze(0).expand(user_embeddings.shape[0], -1)
        
        # Concatenate user embedding + post features
        combined = torch.cat([user_embeddings, post_features], dim=1)
        
        # Predict interaction probabilities
        interaction_probs = self.interaction_predictor(combined).squeeze()
        
        return interaction_probs
    
    def predict_interactions(
        self,
        text: str,
        graph: Data,
        user_indices: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict which users will interact
        
        Args:
            text: Post text
            graph: Social network graph
            user_indices: Optional user indices to predict for
            threshold: Probability threshold for interaction
        
        Returns:
            Tuple of (interaction_probs, predicted_interactions)
            - interaction_probs: [num_users] probabilities
            - predicted_interactions: [num_users] binary predictions
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(text, graph, user_indices)
            predictions = (probs > threshold).long()
        
        return probs, predictions

