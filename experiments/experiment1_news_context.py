"""
Experiment 1: Adding news headlines as additional context
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from experiments.experiment_base import BaseExperiment
from models.integrated_model import IntegratedNLPGNNModel
from models.graph.graph_builder import GraphBuilder
from models.nlp.text_processor import TextProcessor
from models.gnn.gnn_model import GNNConfig


class NewsContextExperiment(BaseExperiment):
    """
    Experiment 1: Evaluate impact of adding news headlines as context
    
    This experiment tests whether incorporating current news headlines
    improves virality prediction accuracy.
    """
    
    def __init__(self, data_path: str, news_data_path: Optional[str] = None, output_dir: str = "results/exp1"):
        """
        Initialize experiment
        
        Args:
            data_path: Path to post data CSV
            news_data_path: Optional path to news headlines data
            output_dir: Output directory for results
        """
        super().__init__(data_path, output_dir)
        self.news_data_path = news_data_path
        self.news_headlines: List[str] = []
        self.text_processor = None
        self.graph_builder = None
        self.model_with_news = None
        self.model_without_news = None
    
    def setup(self):
        """Setup experiment"""
        print("Setting up Experiment 1: News Context...")
        
        # Load data
        self.df = pd.read_csv(self.data_path, delimiter=';')
        print(f"Loaded {len(self.df)} posts")
        
        # Load or generate news headlines
        if self.news_data_path and Path(self.news_data_path).exists():
            news_df = pd.read_csv(self.news_data_path)
            self.news_headlines = news_df['headline'].tolist() if 'headline' in news_df.columns else []
        else:
            # Generate synthetic news headlines for demonstration
            # In production, these would come from actual news APIs
            self.news_headlines = self._generate_synthetic_news()
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.graph_builder = GraphBuilder(include_news_agencies=False)
        
        # Build graphs
        print("Building graphs...")
        self.graphs = []
        for idx, row in self.df.iterrows():
            graph = self.graph_builder.build_pyg_graph(
                self.df.iloc[[idx]],
                node_features=None
            )
            self.graphs.append(graph)
        
        # Initialize models
        gnn_config = GNNConfig(
            input_dim=768,
            hidden_dim=256,
            num_layers=3,
            output_dim=128,
            dropout=0.3
        )
        
        # Model with news context
        self.model_with_news = IntegratedNLPGNNModel(
            gnn_config=gnn_config,
            include_news_context=True,
            news_context_dim=768,  # BERT embedding dimension
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Model without news context (baseline)
        self.model_without_news = IntegratedNLPGNNModel(
            gnn_config=gnn_config,
            include_news_context=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print("Setup complete!")
    
    def _generate_synthetic_news(self) -> List[str]:
        """Generate synthetic news headlines for demonstration"""
        headlines = [
            "Breaking: Major policy announcement expected",
            "Political tensions rise ahead of key decision",
            "Economic indicators show mixed signals",
            "International relations take center stage",
            "Technology sector faces regulatory scrutiny",
            "Healthcare policy debate intensifies",
            "Climate change discussions continue",
            "Education reform proposals gain traction",
            "Immigration policy changes announced",
            "Trade negotiations reach critical phase"
        ]
        return headlines
    
    def get_news_context(self, post_text: str) -> torch.Tensor:
        """
        Get relevant news context for a post
        
        In production, this would use semantic similarity to find
        relevant headlines. For now, we use a simple approach.
        """
        if not self.news_headlines:
            return torch.zeros(768)  # Zero vector if no news
        
        # Simple approach: embed all headlines and take mean
        # In production, would select most relevant headlines
        embeddings = []
        for headline in self.news_headlines[:5]:  # Use top 5 headlines
            emb = self.text_processor.get_embeddings(headline)
            embeddings.append(emb)
        
        if embeddings:
            mean_embedding = np.mean(embeddings, axis=0)
            return torch.tensor(mean_embedding, dtype=torch.float32)
        else:
            return torch.zeros(768)
    
    def run(self):
        """Run experiment"""
        print("Running Experiment 1...")
        
        # For demonstration, we'll create synthetic virality scores
        # In production, these would come from actual engagement metrics
        virality_scores = self._compute_virality_scores()
        
        # Split data
        split_idx = int(len(self.df) * 0.8)
        train_df = self.df.iloc[:split_idx]
        test_df = self.df.iloc[split_idx:]
        
        # Train models (simplified - in production would use proper training loop)
        print("Training models...")
        # Note: Full training would be done in training script
        # This is just for experiment structure
        
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
                
                # Simple virality: normalize by number of interactions
                total_engagement = len(likes) + len(comments)
                # Normalize to [0, 1] (assuming max engagement ~1000)
                score = min(total_engagement / 1000.0, 1.0)
                scores.append(score)
            except:
                scores.append(0.0)
        
        return np.array(scores)
    
    def evaluate(self) -> Dict:
        """Evaluate experiment results"""
        print("Evaluating Experiment 1...")
        
        # This would contain actual evaluation metrics
        # For now, return structure
        results = {
            'experiment': 'News Context Addition',
            'description': 'Evaluating impact of adding news headlines as context',
            'baseline_model': 'NLP + GNN without news',
            'enhanced_model': 'NLP + GNN with news context',
            'metrics': {
                'baseline_accuracy': 0.0,  # Would be computed during training
                'enhanced_accuracy': 0.0,
                'improvement': 0.0,
                'baseline_mae': 0.0,
                'enhanced_mae': 0.0,
            },
            'conclusion': 'News context addition shows [results]'
        }
        
        return results


if __name__ == "__main__":
    exp = NewsContextExperiment(
        data_path="../trump_posts_data.csv",
        output_dir="results/exp1"
    )
    exp.setup()
    results = exp.run()
    exp.save_results("exp1_results.json")

