"""
Prediction script for new text prompts
"""

import torch
import argparse
from pathlib import Path
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.integrated_model import IntegratedNLPGNNModel
from models.graph.graph_builder import GraphBuilder
from models.gnn.gnn_model import GNNConfig
from models.nlp.text_processor import TextProcessor


def load_model(checkpoint_path: str, include_news: bool, device: str):
    """Load trained model from checkpoint"""
    gnn_config = GNNConfig(
        input_dim=768,
        hidden_dim=256,
        num_layers=3,
        output_dim=128,
        dropout=0.3
    )
    
    model = IntegratedNLPGNNModel(
        gnn_config=gnn_config,
        include_news_context=include_news,
        device=device
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def predict_virality(
    text: str,
    model,
    graph_builder: GraphBuilder,
    text_processor: TextProcessor,
    device: str,
    news_context: torch.Tensor = None
) -> float:
    """Predict virality for a text prompt"""
    # Build a minimal graph for the prediction
    # In practice, you'd use historical interaction data
    import pandas as pd
    
    # Create a dummy post entry
    dummy_data = pd.DataFrame({
        'post_id': ['dummy_prediction'],
        'likes': ['[]'],
        'comments': ['[]']
    })
    
    graph = graph_builder.build_pyg_graph(dummy_data, node_features=None)
    
    # Predict
    with torch.no_grad():
        virality_score = model.predict(text, graph, news_context)
    
    return virality_score


def main():
    parser = argparse.ArgumentParser(description='Predict virality for text prompt')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--text', type=str, required=True,
                       help='Text prompt to predict')
    parser.add_argument('--include_news', action='store_true',
                       help='Model includes news context')
    parser.add_argument('--include_news_agencies', action='store_true',
                       help='Graph includes news agency nodes')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.include_news, device)
    
    # Initialize components
    text_processor = TextProcessor(device=device)
    graph_builder = GraphBuilder(include_news_agencies=args.include_news_agencies)
    
    # Predict
    print(f"\nPredicting virality for: {args.text[:100]}...")
    virality_score = predict_virality(
        args.text,
        model,
        graph_builder,
        text_processor,
        device
    )
    
    print(f"\nPredicted Virality Score: {virality_score:.4f} (0-1 scale)")
    print(f"Interpretation: {'High' if virality_score > 0.7 else 'Medium' if virality_score > 0.4 else 'Low'} virality potential")


if __name__ == "__main__":
    main()

