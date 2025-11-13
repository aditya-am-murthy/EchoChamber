"""
Experiment 2: Adding different types of user nodes (news agencies, etc.)
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from experiments.experiment_base import BaseExperiment
from models.integrated_model import IntegratedNLPGNNModel
from models.graph.graph_builder import GraphBuilder
from models.gnn.gnn_model import GNNConfig


class NodeTypesExperiment(BaseExperiment):
    """
    Experiment 2: Evaluate impact of adding different node types
    
    This experiment tests whether distinguishing between different types
    of users (regular users, news agencies, influencers) improves
    virality prediction.
    """
    
    def __init__(self, data_path: str, output_dir: str = "results/exp2"):
        """
        Initialize experiment
        
        Args:
            data_path: Path to post data CSV
            output_dir: Output directory for results
        """
        super().__init__(data_path, output_dir)
        self.graph_builder_basic = None
        self.graph_builder_enhanced = None
        self.model_basic = None
        self.model_enhanced = None
    
    def setup(self):
        """Setup experiment"""
        print("Setting up Experiment 2: Node Types...")
        
        # Load data
        self.df = pd.read_csv(self.data_path, delimiter=';')
        print(f"Loaded {len(self.df)} posts")
        
        # Initialize graph builders
        # Basic: only user and post nodes
        self.graph_builder_basic = GraphBuilder(include_news_agencies=False)
        
        # Enhanced: includes news agencies and other node types
        self.graph_builder_enhanced = GraphBuilder(include_news_agencies=True)
        
        # Build graphs
        print("Building basic graphs...")
        self.graphs_basic = []
        for idx, row in self.df.iterrows():
            graph = self.graph_builder_basic.build_pyg_graph(
                self.df.iloc[[idx]],
                node_features=None
            )
            self.graphs_basic.append(graph)
        
        print("Building enhanced graphs...")
        self.graphs_enhanced = []
        for idx, row in self.df.iterrows():
            graph = self.graph_builder_enhanced.build_pyg_graph(
                self.df.iloc[[idx]],
                node_features=None
            )
            self.graphs_enhanced.append(graph)
        
        # Initialize models
        gnn_config = GNNConfig(
            input_dim=768,
            hidden_dim=256,
            num_layers=3,
            output_dim=128,
            dropout=0.3
        )
        
        # Basic model (only user/post nodes)
        self.model_basic = IntegratedNLPGNNModel(
            gnn_config=gnn_config,
            include_news_context=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Enhanced model (with node type distinctions)
        self.model_enhanced = IntegratedNLPGNNModel(
            gnn_config=gnn_config,
            include_news_context=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("Setup complete!")
    
    def run(self):
        """Run experiment"""
        print("Running Experiment 2...")
        
        # Compute virality scores
        virality_scores = self._compute_virality_scores()
        
        # Split data
        split_idx = int(len(self.df) * 0.8)
        train_df = self.df.iloc[:split_idx]
        test_df = self.df.iloc[split_idx:]
        
        # Train models (simplified - in production would use proper training loop)
        print("Training models...")
        # Note: Full training would be done in training script
        
        # Evaluate
        results = self.evaluate()
        self.results = results
        
        return results
    
    def _compute_virality_scores(self) -> np.ndarray:
        """Compute virality scores from engagement data"""
        scores = []
        for _, row in self.df.iterrows():
            try:
                import json
                likes = json.loads(row['likes']) if isinstance(row['likes'], str) else []
                comments = json.loads(row['comments']) if isinstance(row['comments'], str) else []
                
                total_engagement = len(likes) + len(comments)
                score = min(total_engagement / 1000.0, 1.0)
                scores.append(score)
            except:
                scores.append(0.0)
        
        return np.array(scores)
    
    def evaluate(self) -> Dict:
        """Evaluate experiment results"""
        print("Evaluating Experiment 2...")
        
        # Analyze node type distribution
        basic_node_types = {}
        enhanced_node_types = {}
        
        for node_id, node_type in self.graph_builder_basic.node_id_to_type.items():
            type_str = node_type.value
            basic_node_types[type_str] = basic_node_types.get(type_str, 0) + 1
        
        for node_id, node_type in self.graph_builder_enhanced.node_id_to_type.items():
            type_str = node_type.value
            enhanced_node_types[type_str] = enhanced_node_types.get(type_str, 0) + 1
        
        results = {
            'experiment': 'Node Type Addition',
            'description': 'Evaluating impact of distinguishing node types (users, news agencies, influencers)',
            'baseline_model': 'NLP + GNN with basic nodes (user/post only)',
            'enhanced_model': 'NLP + GNN with enhanced nodes (user/post/news_agency/influencer)',
            'node_type_distribution': {
                'basic': basic_node_types,
                'enhanced': enhanced_node_types
            },
            'metrics': {
                'baseline_accuracy': 0.0,
                'enhanced_accuracy': 0.0,
                'improvement': 0.0,
                'baseline_mae': 0.0,
                'enhanced_mae': 0.0,
            },
            'conclusion': 'Node type distinctions show [results]'
        }
        
        return results


if __name__ == "__main__":
    exp = NodeTypesExperiment(
        data_path="../trump_posts_data.csv",
        output_dir="results/exp2"
    )
    exp.setup()
    results = exp.run()
    exp.save_results("exp2_results.json")

