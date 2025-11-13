"""
Training script for NLP + GNN virality prediction model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.integrated_model import IntegratedNLPGNNModel
from models.graph.graph_builder import GraphBuilder
from models.gnn.gnn_model import GNNConfig


def custom_collate_fn(batch):
    """Custom collate function for handling PyTorch Geometric Data objects"""
    texts = [item['text'] for item in batch]
    graphs = [item['graph'] for item in batch]  # Keep as list, process individually
    news_contexts = [item['news_context'] for item in batch]
    virality = torch.stack([item['virality'] for item in batch])
    post_ids = [item['post_id'] for item in batch]
    
    # Stack news contexts if they exist
    news_batch = None
    if news_contexts[0] is not None:
        news_batch = torch.stack(news_contexts)
    
    return {
        'text': texts,
        'graph': graphs,  # List of Data objects
        'news_context': news_batch,
        'virality': virality,
        'post_id': post_ids
    }


class ViralityDataset(Dataset):
    """Dataset for virality prediction"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        graph_builder: GraphBuilder,
        text_processor,
        include_news: bool = False,
        news_headlines: List[str] = None
    ):
        self.df = df
        self.graph_builder = graph_builder
        self.text_processor = text_processor
        self.include_news = include_news
        self.news_headlines = news_headlines or []
        
        # Compute virality scores
        self.virality_scores = self._compute_virality_scores()
        
        # Build graphs - need to reset graph builder for each post to avoid node ID conflicts
        self.graphs = []
        for idx in range(len(df)):
            # Create a fresh graph builder for each post to avoid node ID conflicts
            from models.graph.graph_builder import GraphBuilder
            single_post_builder = GraphBuilder(include_news_agencies=graph_builder.include_news_agencies)
            graph = single_post_builder.build_pyg_graph(
                df.iloc[[idx]],
                node_features=None
            )
            self.graphs.append(graph)
    
    def _compute_virality_scores(self) -> np.ndarray:
        """Compute virality scores from engagement"""
        scores = []
        for _, row in self.df.iterrows():
            try:
                likes = json.loads(row['likes']) if isinstance(row['likes'], str) else []
                comments = json.loads(row['comments']) if isinstance(row['comments'], str) else []
                
                total_engagement = len(likes) + len(comments)
                # Normalize to [0, 1]
                score = min(total_engagement / 1000.0, 1.0)
                scores.append(score)
            except:
                scores.append(0.0)
        
        return np.array(scores)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get post text (would need to be in CSV or fetched)
        # For now, use a placeholder
        post_text = getattr(row, 'text', '') or f"Post {row['post_id']}"
        
        graph = self.graphs[idx]
        virality_score = torch.tensor(self.virality_scores[idx], dtype=torch.float32)
        
        # News context (if enabled)
        news_context = None
        if self.include_news and self.news_headlines:
            # Simple: use mean of all headlines
            embeddings = [self.text_processor.get_embeddings(h) for h in self.news_headlines[:5]]
            if embeddings:
                mean_emb = np.mean(embeddings, axis=0)
                news_context = torch.tensor(mean_emb, dtype=torch.float32)
        
        return {
            'text': post_text,
            'graph': graph,
            'news_context': news_context,
            'virality': virality_score,
            'post_id': row['post_id']
        }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        texts = batch['text']
        graphs = batch['graph']  # List of Data objects
        news_contexts = batch['news_context']
        targets_batch = batch['virality'].to(device)
        
        # Forward pass - process each text with its graph
        batch_predictions = []
        for i, text in enumerate(texts):
            # Clone graph to avoid reusing computational graph across epochs
            graph = graphs[i].clone().to(device)
            news_ctx = None
            if news_contexts is not None:
                news_ctx = news_contexts[i].clone().to(device) if news_contexts[i] is not None else None
            
            pred = model(text, graph, news_ctx)
            batch_predictions.append(pred)
        
        batch_predictions = torch.stack(batch_predictions).squeeze()
        
        # Compute loss
        loss = criterion(batch_predictions, targets_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions and targets for metrics
        pred_np = batch_predictions.detach().cpu().numpy()
        target_np = targets_batch.detach().cpu().numpy()
        if pred_np.ndim == 0:
            pred_np = np.array([pred_np])
        if target_np.ndim == 0:
            target_np = np.array([target_np])
        predictions.extend(pred_np)
        targets.extend(target_np)
        
        num_batches += 1
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Calculate accuracy using median threshold (more appropriate for imbalanced data)
    threshold = np.median(targets) if len(targets) > 0 else 0.5
    pred_binary = (predictions > threshold).astype(int)
    target_binary = (targets > threshold).astype(int)
    accuracy = np.mean(pred_binary == target_binary)
    
    return avg_loss, mae, r2_score, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            texts = batch['text']
            graphs = batch['graph']  # List of Data objects
            news_contexts = batch['news_context']
            target = batch['virality'].to(device)
            
            batch_preds = []
            for i, text in enumerate(texts):
                # Clone graph to avoid reusing computational graph
                graph = graphs[i].clone().to(device)
                news_ctx = None
                if news_contexts is not None:
                    news_ctx = news_contexts[i].clone().to(device) if news_contexts[i] is not None else None
                
                pred = model(text, graph, news_ctx)
                batch_preds.append(pred)
            
            batch_preds = torch.stack(batch_preds)
            if batch_preds.dim() > 1:
                batch_preds = batch_preds.squeeze()
            if batch_preds.dim() == 0:
                batch_preds = batch_preds.unsqueeze(0)
            
            loss = criterion(batch_preds, target)
            
            total_loss += loss.item()
            pred_np = batch_preds.cpu().numpy()
            target_np = target.cpu().numpy()
            if pred_np.ndim == 0:
                pred_np = np.array([pred_np])
            if target_np.ndim == 0:
                target_np = np.array([target_np])
            predictions.extend(pred_np)
            targets.extend(target_np)
            num_batches += 1
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate R² score (coefficient of determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Calculate accuracy using median threshold (more appropriate for imbalanced data)
    threshold = np.median(targets) if len(targets) > 0 else 0.5
    pred_binary = (predictions > threshold).astype(int)
    target_binary = (targets > threshold).astype(int)
    accuracy = np.mean(pred_binary == target_binary)
    
    return avg_loss, mae, r2_score, accuracy, predictions, targets


def main():
    parser = argparse.ArgumentParser(description='Train NLP + GNN virality prediction model')
    parser.add_argument('--data', type=str, default='trump_posts_data.csv',
                       help='Path to data CSV')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--include_news', action='store_true',
                       help='Include news context (Experiment 1)')
    parser.add_argument('--include_news_agencies', action='store_true',
                       help='Include news agency nodes (Experiment 2)')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    df = pd.read_csv(args.data, delimiter=';')
    print(f"Loaded {len(df)} posts")
    
    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # Initialize components
    from models.nlp.text_processor import TextProcessor
    text_processor = TextProcessor(device=device)
    
    graph_builder = GraphBuilder(include_news_agencies=args.include_news_agencies)
    
    # Create datasets
    train_dataset = ViralityDataset(
        train_df,
        graph_builder,
        text_processor,
        include_news=args.include_news
    )
    
    val_dataset = ViralityDataset(
        val_df,
        graph_builder,
        text_processor,
        include_news=args.include_news
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Initialize model
    # Node features are: 6 node types (one-hot) + 1 interaction count = 7 dimensions
    gnn_config = GNNConfig(
        input_dim=7,  # Match graph node feature dimension
        hidden_dim=256,
        num_layers=3,
        output_dim=128,
        dropout=0.3,
        use_batch_norm=False  # Disable batch norm for small graphs
    )
    
    model = IntegratedNLPGNNModel(
        gnn_config=gnn_config,
        include_news_context=args.include_news,
        device=device
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_mae, train_r2, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train R²: {train_r2:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_mae, val_r2, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'train_acc': train_acc,
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

