"""
Training script for predicting user interactions (which users will like/comment)
This is a link prediction problem in the graph
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

from models.graph.graph_builder import GraphBuilder
from models.gnn.gnn_model import GNNModel, GNNConfig
from models.nlp.text_processor import TextProcessor


class UserInteractionDataset(Dataset):
    """Dataset for user interaction prediction"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        graph_builder: GraphBuilder,
        text_processor: TextProcessor,
    ):
        self.df = df
        self.graph_builder = graph_builder
        self.text_processor = text_processor
        
        # Build graphs and extract interaction labels
        self.graphs = []
        self.interaction_labels = []  # List of dicts: {user_idx: 1 if interacted, 0 otherwise}
        self.user_indices_list = []  # List of user node indices for each graph
        self.graph_builders = []  # Store graph builders to access node mappings
        
        for idx in range(len(df)):
            # Create fresh graph builder for each post
            single_post_builder = GraphBuilder(include_news_agencies=graph_builder.include_news_agencies)
            graph = single_post_builder.build_pyg_graph(df.iloc[[idx]], node_features=None)
            
            # Store graph builder to access node mappings
            self.graph_builders.append(single_post_builder)
            
            # Extract interaction labels
            row = df.iloc[idx]
            likes, comment_users = self._parse_interactions(row)
            
            # Create interaction labels: 1 if user interacted, 0 otherwise
            interaction_dict = {}
            user_indices = []
            
            # Get all user nodes using the graph builder's node mapping
            for node_id, node_idx in single_post_builder.node_id_to_index.items():
                if node_id.startswith('user_'):
                    user_indices.append(node_idx)
                    username = node_id.replace('user_', '')
                    # Check if this user liked or commented
                    interacted = (username in likes) or (username in comment_users)
                    interaction_dict[node_idx] = 1 if interacted else 0
            
            self.graphs.append(graph)
            self.interaction_labels.append(interaction_dict)
            self.user_indices_list.append(torch.tensor(user_indices, dtype=torch.long))
    
    def _parse_interactions(self, row: pd.Series) -> Tuple[List[str], List[str]]:
        """Parse likes and comments from a row"""
        try:
            likes = json.loads(row['likes']) if isinstance(row['likes'], str) else []
            comments = json.loads(row['comments']) if isinstance(row['comments'], str) else []
        except (json.JSONDecodeError, KeyError):
            likes = []
            comments = []
        
        # Extract usernames from comments
        comment_users = [c[0] if isinstance(c, (list, tuple)) and len(c) > 0 else '' 
                        for c in comments]
        comment_users = [u for u in comment_users if u]
        
        return likes, comment_users
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get post text
        post_text = getattr(row, 'text', '') or f"Post {row['post_id']}"
        
        graph = self.graphs[idx]
        interaction_labels = self.interaction_labels[idx]
        user_indices = self.user_indices_list[idx]
        
        # Convert interaction labels to tensor
        labels = torch.zeros(len(user_indices), dtype=torch.float32)
        for i, user_idx in enumerate(user_indices):
            labels[i] = interaction_labels.get(user_idx.item(), 0)
        
        return {
            'text': post_text,
            'graph': graph,
            'user_indices': user_indices,
            'labels': labels,
            'post_id': row['post_id']
        }


def custom_collate_fn(batch):
    """Custom collate function"""
    texts = [item['text'] for item in batch]
    graphs = [item['graph'] for item in batch]
    user_indices = [item['user_indices'] for item in batch]
    labels = [item['labels'] for item in batch]
    post_ids = [item['post_id'] for item in batch]
    
    return {
        'text': texts,
        'graph': graphs,
        'user_indices': user_indices,
        'labels': labels,
        'post_id': post_ids
    }


# Simplified model for user interaction prediction
class SimpleInteractionModel(nn.Module):
    """Simple model that predicts user interactions"""
    
    def __init__(self, text_processor, gnn_config, device):
        super().__init__()
        self.text_processor = text_processor
        self.device = device
        
        # GNN for user embeddings
        self.gnn = GNNModel(gnn_config).to(device)
        
        # Project node embeddings to desired dimension
        self.node_embedding_proj = nn.Linear(gnn_config.hidden_dim, gnn_config.output_dim).to(device)
        
        # Interaction predictor
        nlp_dim = 768 + 11 + 4 + 3  # NLP features
        user_dim = gnn_config.output_dim
        self.predictor = nn.Sequential(
            nn.Linear(user_dim + nlp_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text, graph, user_indices):
        # Extract NLP features
        features = self.text_processor.extract_all_features(text)
        nlp_features = np.concatenate([
            features['embeddings'],
            features['linguistic'],
            features['sentiment'],
            features['topic']
        ])
        nlp_tensor = torch.tensor(nlp_features, dtype=torch.float32).to(self.device)
        
        # Get user embeddings from GNN
        graph = graph.to(self.device)
        graph_x = graph.x.clone()
        if graph_x.shape[1] != self.gnn.config.input_dim:
            if not hasattr(self, 'node_proj'):
                self.node_proj = nn.Linear(graph_x.shape[1], self.gnn.config.input_dim).to(self.device)
            graph_x = self.node_proj(graph_x)
        
        # Get node embeddings (before pooling)
        # We need to modify GNN to return node-level embeddings, not graph-level
        # For now, let's create a simple node embedding extractor
        node_embeddings = self._get_node_embeddings(graph_x, graph.edge_index)
        user_embeddings = node_embeddings[user_indices]
        
        # Combine features
        nlp_expanded = nlp_tensor.unsqueeze(0).expand(user_embeddings.shape[0], -1)
        combined = torch.cat([user_embeddings, nlp_expanded], dim=1)
        
        # Predict
        probs = self.predictor(combined).squeeze()
        return probs
    
    def _get_node_embeddings(self, x, edge_index):
        """Get node-level embeddings from GNN (before pooling)"""
        # Manually run through GNN layers to get node embeddings
        x = self.gnn.input_proj(x)
        x = torch.relu(x)
        
        for i, gnn_layer in enumerate(self.gnn.gnn_layers):
            x_new = gnn_layer(x, edge_index)
            if self.gnn.config.use_batch_norm and self.gnn.batch_norms is not None:
                x_new = self.gnn.batch_norms[i](x_new)
            x_new = torch.relu(x_new)
            x = torch.dropout(x_new, p=self.gnn.config.dropout, train=self.training)
        
        # Project to desired dimension
        x = self.node_embedding_proj(x)
        
        # Return node embeddings (not pooled)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        texts = batch['text']
        graphs = batch['graph']
        user_indices_list = batch['user_indices']
        labels_list = batch['labels']
        
        batch_loss = 0
        batch_size = len(texts)
        
        for i in range(batch_size):
            graph = graphs[i].clone().to(device)
            user_indices = user_indices_list[i].to(device)
            labels = labels_list[i].to(device)
            
            # Skip if no users
            if len(user_indices) == 0 or len(labels) == 0:
                continue
            
            # Forward
            preds = model(texts[i], graph, user_indices)
            
            # Ensure predictions are valid
            preds = torch.clamp(preds, min=1e-7, max=1-1e-7)
            
            # Loss
            loss = criterion(preds, labels)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            batch_loss += loss
            
            # Store for metrics
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
        
        # Backward
        if batch_size > 0 and not torch.isnan(batch_loss):
            optimizer.zero_grad()
            avg_batch_loss = batch_loss / batch_size
            avg_batch_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += avg_batch_loss.item()
            num_batches += 1
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Accuracy
    pred_binary = (all_preds > 0.5).astype(int)
    accuracy = np.mean(pred_binary == all_labels)
    
    return total_loss / num_batches, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            texts = batch['text']
            graphs = batch['graph']
            user_indices_list = batch['user_indices']
            labels_list = batch['labels']
            
            batch_loss = 0
            batch_size = len(texts)
            valid_samples = 0
            
            for i in range(batch_size):
                graph = graphs[i].clone().to(device)
                user_indices = user_indices_list[i].to(device)
                labels = labels_list[i].to(device)
                
                # Skip if no users
                if len(user_indices) == 0 or len(labels) == 0:
                    continue
                
                preds = model(texts[i], graph, user_indices)
                preds = torch.clamp(preds, min=1e-7, max=1-1e-7)
                
                loss = criterion(preds, labels)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    if isinstance(loss, torch.Tensor):
                        batch_loss += loss.item()
                    else:
                        batch_loss += loss
                    valid_samples += 1
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            if valid_samples > 0:
                total_loss += batch_loss / valid_samples
                num_batches += 1
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    avg_loss = total_loss / num_batches
    pred_binary = (all_preds > 0.5).astype(int)
    accuracy = np.mean(pred_binary == all_labels)
    
    # Precision, Recall, F1
    tp = np.sum((pred_binary == 1) & (all_labels == 1))
    fp = np.sum((pred_binary == 1) & (all_labels == 0))
    fn = np.sum((pred_binary == 0) & (all_labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return avg_loss, accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='Train user interaction prediction model')
    parser.add_argument('--data', type=str, default='trump_posts_data.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    df = pd.read_csv(args.data, delimiter=';')
    print(f"Loaded {len(df)} posts")
    
    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # Initialize
    text_processor = TextProcessor(device=device)
    graph_builder = GraphBuilder(include_news_agencies=False)
    
    gnn_config = GNNConfig(
        input_dim=7,
        hidden_dim=256,
        num_layers=3,
        output_dim=128,
        dropout=0.3,
        use_batch_norm=False
    )
    
    # Create datasets
    train_dataset = UserInteractionDataset(train_df, graph_builder, text_processor)
    val_dataset = UserInteractionDataset(val_df, graph_builder, text_processor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Model
    model = SimpleInteractionModel(text_processor, gnn_config, device).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Prec: {val_prec:.4f}, Val Rec: {val_rec:.4f}, Val F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_dir / 'best_interaction_model.pt')
            print(f"Saved best model (val_acc: {val_acc:.4f})")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

