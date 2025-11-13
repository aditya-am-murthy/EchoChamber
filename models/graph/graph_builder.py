"""
Build social network graphs from interaction data
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import networkx as nx
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import to_networkx
from .node_types import NodeType


class GraphBuilder:
    """Builds graph representations of social network interactions"""
    
    def __init__(self, include_news_agencies: bool = False):
        """
        Initialize graph builder
        
        Args:
            include_news_agencies: Whether to include news agency nodes (Experiment 2)
        """
        self.include_news_agencies = include_news_agencies
        self.node_id_to_type: Dict[str, NodeType] = {}
        self.node_id_to_index: Dict[str, int] = {}
        self.index_to_node_id: Dict[int, str] = {}
        self.user_interaction_counts: Dict[str, int] = defaultdict(int)
        self.post_engagement: Dict[str, Dict[str, int]] = {}
        
        # News agency identifiers (can be expanded)
        self.news_agencies = {
            'cnn', 'foxnews', 'fox', 'msnbc', 'abc', 'cbs', 'nbc',
            'reuters', 'ap', 'associated press', 'bbc', 'wsj', 'wall street journal',
            'nytimes', 'new york times', 'washington post', 'wapo',
            'politico', 'the hill', 'breitbart', 'daily wire'
        }
    
    def is_news_agency(self, username: str) -> bool:
        """Check if username belongs to a news agency"""
        username_lower = username.lower()
        return any(agency in username_lower for agency in self.news_agencies)
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load post data from CSV"""
        df = pd.read_csv(csv_path, delimiter=';')
        return df
    
    def parse_interactions(self, row: pd.Series) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Parse likes and comments from a row"""
        try:
            likes = json.loads(row['likes']) if isinstance(row['likes'], str) else []
            comments = json.loads(row['comments']) if isinstance(row['comments'], str) else []
        except (json.JSONDecodeError, KeyError):
            likes = []
            comments = []
        
        # Extract usernames from comments (comments are tuples of (username, content))
        comment_users = [c[0] if isinstance(c, (list, tuple)) and len(c) > 0 else '' 
                        for c in comments]
        comment_users = [u for u in comment_users if u]
        
        return likes, comment_users
    
    def build_node_mapping(self, df: pd.DataFrame):
        """Build mapping from node IDs to indices and types"""
        node_counter = 0
        
        # Add post nodes
        for post_id in df['post_id']:
            node_id = f"post_{post_id}"
            if node_id not in self.node_id_to_index:
                self.node_id_to_index[node_id] = node_counter
                self.index_to_node_id[node_counter] = node_id
                self.node_id_to_type[node_id] = NodeType.POST
                node_counter += 1
        
        # Add user nodes from interactions
        for _, row in df.iterrows():
            likes, comment_users = self.parse_interactions(row)
            
            # Add users who liked
            for username in likes:
                if username:
                    node_id = f"user_{username}"
                    if node_id not in self.node_id_to_index:
                        self.node_id_to_index[node_id] = node_counter
                        self.index_to_node_id[node_counter] = node_id
                        
                        # Determine node type
                        if self.include_news_agencies and self.is_news_agency(username):
                            self.node_id_to_type[node_id] = NodeType.NEWS_AGENCY
                        else:
                            self.node_id_to_type[node_id] = NodeType.USER
                        
                        node_counter += 1
                    
                    self.user_interaction_counts[username] += 1
            
            # Add users who commented
            for username in comment_users:
                if username:
                    node_id = f"user_{username}"
                    if node_id not in self.node_id_to_index:
                        self.node_id_to_index[node_id] = node_counter
                        self.index_to_node_id[node_counter] = node_id
                        
                        if self.include_news_agencies and self.is_news_agency(username):
                            self.node_id_to_type[node_id] = NodeType.NEWS_AGENCY
                        else:
                            self.node_id_to_type[node_id] = NodeType.USER
                        
                        node_counter += 1
                    
                    self.user_interaction_counts[username] += 1
        
        # Mark high-engagement users as influencers
        if len(self.user_interaction_counts) > 0:
            threshold = np.percentile(list(self.user_interaction_counts.values()), 90)
            for username, count in self.user_interaction_counts.items():
                node_id = f"user_{username}"
                if node_id in self.node_id_to_type and count >= threshold:
                    if self.node_id_to_type[node_id] == NodeType.USER:
                        self.node_id_to_type[node_id] = NodeType.INFLUENCER
    
    def build_edges(self, df: pd.DataFrame) -> List[Tuple[int, int, Dict]]:
        """
        Build edge list with edge features
        
        Returns:
            List of (source_index, target_index, edge_attrs) tuples
        """
        edges = []
        
        for _, row in df.iterrows():
            post_id = row['post_id']
            post_node_id = f"post_{post_id}"
            post_idx = self.node_id_to_index.get(post_node_id)
            
            if post_idx is None:
                continue
            
            likes, comment_users = self.parse_interactions(row)
            
            # Add like edges (user -> post)
            for username in likes:
                if username:
                    user_node_id = f"user_{username}"
                    user_idx = self.node_id_to_index.get(user_node_id)
                    
                    if user_idx is not None:
                        edges.append((
                            user_idx,
                            post_idx,
                            {'type': 'like', 'weight': 1.0}
                        ))
            
            # Add comment edges (user -> post)
            for username in comment_users:
                if username:
                    user_node_id = f"user_{username}"
                    user_idx = self.node_id_to_index.get(user_node_id)
                    
                    if user_idx is not None:
                        edges.append((
                            user_idx,
                            post_idx,
                            {'type': 'comment', 'weight': 2.0}  # Comments weighted higher
                        ))
        
        return edges
    
    def build_pyg_graph(self, df: pd.DataFrame, node_features: Optional[Dict[str, np.ndarray]] = None) -> Data:
        """
        Build PyTorch Geometric graph
        
        Args:
            df: DataFrame with post data
            node_features: Optional dict mapping node_id to feature vector
        
        Returns:
            PyTorch Geometric Data object
        """
        # Build node mappings
        self.build_node_mapping(df)
        
        # Build edges
        edges = self.build_edges(df)
        
        if not edges:
            # Return empty graph
            num_nodes = len(self.node_id_to_index)
            return Data(
                x=torch.zeros((num_nodes, 768)),  # Default feature size
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=num_nodes
            )
        
        # Extract edge indices and attributes
        edge_index = []
        edge_attr = []
        
        for src, tgt, attrs in edges:
            edge_index.append([src, tgt])
            edge_attr.append([attrs['weight']])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Build node features
        num_nodes = len(self.node_id_to_index)
        if node_features is None:
            # Default features: one-hot encoding of node type
            node_feat_dim = len(NodeType) + 1  # +1 for interaction count
            x = torch.zeros((num_nodes, node_feat_dim))
            
            for node_id, idx in self.node_id_to_index.items():
                node_type = self.node_id_to_type.get(node_id, NodeType.USER)
                # One-hot encode node type
                type_idx = list(NodeType).index(node_type)
                x[idx, type_idx] = 1.0
                
                # Add interaction count (normalized)
                if node_id.startswith('user_'):
                    username = node_id.replace('user_', '')
                    interaction_count = self.user_interaction_counts.get(username, 0)
                    x[idx, -1] = min(interaction_count / 100.0, 1.0)  # Normalize
        else:
            # Use provided features
            feat_dim = next(iter(node_features.values())).shape[0]
            x = torch.zeros((num_nodes, feat_dim))
            
            for node_id, idx in self.node_id_to_index.items():
                if node_id in node_features:
                    x[idx] = torch.tensor(node_features[node_id], dtype=torch.float)
        
        # Build graph
        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )
        
        # Store metadata
        graph.node_id_to_index = self.node_id_to_index
        graph.index_to_node_id = self.index_to_node_id
        graph.node_id_to_type = self.node_id_to_type
        
        return graph
    
    def get_node_type_features(self) -> Dict[str, np.ndarray]:
        """Get one-hot encoded node type features"""
        features = {}
        num_types = len(NodeType)
        
        for node_id, node_type in self.node_id_to_type.items():
            type_vector = np.zeros(num_types)
            type_idx = list(NodeType).index(node_type)
            type_vector[type_idx] = 1.0
            features[node_id] = type_vector
        
        return features

