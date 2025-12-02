"""
Integrated NLP + GNN model for virality prediction
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from torch_geometric.data import Data
from torch_geometric.data import HeteroData

from .nlp.text_processor import TextProcessor
from .gnn import GNNModel, GNNConfig, HeteroGNNModel, HeteroGNNConfig
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
        device: str = None,
        use_hetero_gnn: bool = False,
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
            use_hetero_gnn: Whether to use heterogeneous HeteroGNNModel (True)
                            or legacy homogeneous GNNModel (False, default)
        """
        super(IntegratedNLPGNNModel, self).__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.include_news_context = include_news_context
        self.use_hetero_gnn = use_hetero_gnn
        
        # NLP module (frozen during training, used for feature extraction)
        self.text_processor = TextProcessor(nlp_model_name, device=self.device)
        
        # Calculate input dimensions
        # NLP features: embeddings + linguistic + sentiment + topic
        # Actual: 768 (embeddings) + 11 (linguistic) + 4 (sentiment) + 3 (topic) = 786
        nlp_total_dim = nlp_feature_dim + 11 + 4 + 3  # embeddings + linguistic (11) + sentiment + topic
        self.nlp_total_dim = nlp_total_dim
        
        # GNN configs
        if gnn_config is None:
            gnn_config = GNNConfig(
                input_dim=7,  # Node features: 6 node types + 1 interaction count
                hidden_dim=256,
                num_layers=3,
                output_dim=gnn_output_dim,
                dropout=dropout,
            )

        self.gnn_config = gnn_config
        self.gnn_output_dim = gnn_output_dim

        # Legacy homogeneous GNN module
        self.gnn = None
        if not self.use_hetero_gnn:
            self.gnn = GNNModel(self.gnn_config).to(self.device)

        # Heterogeneous GNN module (lazy init once metadata is known)
        self.hetero_gnn: Optional[HeteroGNNModel] = None
        self.hetero_gnn_config = HeteroGNNConfig(
            hidden_dim=256,
            num_layers=3,
            output_dim=gnn_output_dim,
            dropout=dropout,
        )
        
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
        graph,
        news_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            text: Post text
            graph: PyTorch Geometric graph (Data or HeteroData,
                   depending on use_hetero_gnn flag)
            news_context: Optional news headline features (Experiment 1)
        
        Returns:
            Virality score [0, 1]
        """
        # Extract NLP features
        nlp_features = self.extract_nlp_features(text)
        
        # GNN forward pass
        graph = graph.to(self.device)

        if not self.use_hetero_gnn:
            # Legacy homogeneous graph path
            graph_x = graph.x.clone()
            if graph_x.shape[1] != self.gnn_config.input_dim:
                if not hasattr(self, "node_feature_proj"):
                    self.node_feature_proj = nn.Linear(
                        graph_x.shape[1], self.gnn_config.input_dim
                    ).to(self.device)
                graph_x = self.node_feature_proj(graph_x)

            gnn_features_vec = self.gnn(graph_x, graph.edge_index)
        else:
            # Heterogeneous graph path
            assert isinstance(
                graph, HeteroData
            ), "use_hetero_gnn=True requires a HeteroData graph."

            # Lazily initialize hetero GNN once metadata is known
            if self.hetero_gnn is None:
                metadata = graph.metadata()
                self.hetero_gnn = HeteroGNNModel(
                    self.hetero_gnn_config, metadata
                ).to(self.device)

            # Build edge_index_dict and edge_attr_dict
            edge_index_dict = graph.edge_index_dict
            edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
            for edge_type in graph.edge_types:
                data = graph[edge_type]
                if hasattr(data, "edge_attr") and data.edge_attr is not None:
                    edge_attr_dict[edge_type] = data.edge_attr

            # Derive user_types from user node features (influencer flag at index 4)
            user_types = None
            if "user" in graph.node_types and graph["user"].x.numel() > 0:
                user_x = graph["user"].x
                if user_x.size(1) >= 5:
                    influencer_flag = user_x[:, 4]  # 1.0 if influencer else 0.0
                    user_types = (influencer_flag > 0.5).long()

            gnn_out = self.hetero_gnn(
                graph.x_dict,
                edge_index_dict,
                edge_attr_dict=edge_attr_dict,
                user_types=user_types,
            )
            gnn_features_vec = gnn_out["graph_repr"]
        
        # Combine features
        # Ensure dimensions match
        if gnn_features_vec.dim() > 1:
            gnn_features_vec = gnn_features_vec.squeeze(0)
        if gnn_features_vec.dim() == 0:
            gnn_features_vec = gnn_features_vec.unsqueeze(0)
        
        # Debug: print dimensions if mismatch
        if (
            nlp_features.shape[0] + gnn_features_vec.shape[0]
            != self.fusion[0].in_features
        ):
            print(
                f"Warning: Dimension mismatch - NLP: {nlp_features.shape[0]}, "
                f"GNN: {gnn_features_vec.shape[0]}, "
                f"Expected: {self.fusion[0].in_features}"
            )

        combined = torch.cat([nlp_features, gnn_features_vec])
        
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

