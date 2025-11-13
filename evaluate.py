"""
Evaluation script for trained models
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.integrated_model import IntegratedNLPGNNModel
from models.graph.graph_builder import GraphBuilder
from models.gnn.gnn_model import GNNConfig
from train import ViralityDataset


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
    
    return model, checkpoint


def evaluate_model(model, dataloader, device):
    """Evaluate model and return metrics"""
    predictions = []
    targets = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            texts = batch['text']
            graphs = batch['graph']
            news_contexts = batch['news_context']
            target = batch['virality'].cpu().numpy()
            
            batch_preds = []
            for i, text in enumerate(texts):
                graph = graphs[i].to(device)
                news_ctx = news_contexts[i].to(device) if news_contexts[i] is not None else None
                
                pred = model(text, graph, news_ctx)
                batch_preds.append(pred.cpu().item())
            
            predictions.extend(batch_preds)
            targets.extend(target)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions.tolist(),
        'targets': targets.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='trump_posts_data.csv',
                       help='Path to data CSV')
    parser.add_argument('--include_news', action='store_true',
                       help='Model includes news context')
    parser.add_argument('--include_news_agencies', action='store_true',
                       help='Graph includes news agency nodes')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, checkpoint = load_model(args.checkpoint, args.include_news, device)
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Load data
    df = pd.read_csv(args.data, delimiter=';')
    print(f"Loaded {len(df)} posts")
    
    # Create dataset
    from models.nlp.text_processor import TextProcessor
    text_processor = TextProcessor(device=device)
    graph_builder = GraphBuilder(include_news_agencies=args.include_news_agencies)
    
    dataset = ViralityDataset(
        df,
        graph_builder,
        text_processor,
        include_news=args.include_news
    )
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, dataloader, device)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"MAE: {results['mae']:.4f}")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R²: {results['r2']:.4f}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

