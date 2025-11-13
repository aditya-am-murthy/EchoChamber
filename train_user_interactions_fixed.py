"""
Training script for predicting user interactions with fixed set of top 500 users
Each prediction is a binary vector of length 500 indicating which users interact
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Set
import argparse
from tqdm import tqdm
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.graph.graph_builder import GraphBuilder
from models.gnn.gnn_model import GNNModel, GNNConfig
from models.nlp.text_processor import TextProcessor


def load_top_users(csv_path: str) -> List[str]:
    """Load top users from CSV"""
    df = pd.read_csv(csv_path)
    return df['username'].tolist()


class FixedUserInteractionDataset(Dataset):
    """Dataset with fixed set of top users"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        top_users: List[str],
        text_processor: TextProcessor,
    ):
        self.df = df
        self.top_users = top_users
        self.num_users = len(top_users)
        self.user_to_idx = {user: idx for idx, user in enumerate(top_users)}
        self.text_processor = text_processor
        
        # Build interaction labels: for each post, binary vector of which top users interacted
        self.interaction_labels = []
        self.post_texts = []
        self.post_metadata = []
        
        for idx, row in df.iterrows():
            # Get post text
            post_text = getattr(row, 'text', '') or f"Post {row['post_id']}"
            self.post_texts.append(post_text)
            
            # Parse interactions
            likes, comment_users = self._parse_interactions(row)
            interacted_users = set(likes) | set(comment_users)
            
            # Create binary label vector for top 500 users
            label_vector = torch.zeros(self.num_users, dtype=torch.float32)
            for username in interacted_users:
                if username in self.user_to_idx:
                    user_idx = self.user_to_idx[username]
                    label_vector[user_idx] = 1.0
            
            self.interaction_labels.append(label_vector)
            
            # Extract metadata
            metadata = {
                'post_id': row['post_id'],
                'timestamp': getattr(row, 'timestamp', None),
                'num_likes': len(likes),
                'num_comments': len(comment_users),
            }
            self.post_metadata.append(metadata)
    
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
        return {
            'text': self.post_texts[idx],
            'labels': self.interaction_labels[idx],
            'metadata': self.post_metadata[idx],
        }


def custom_collate_fn(batch):
    """Custom collate function"""
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    metadata = [item['metadata'] for item in batch]
    
    return {
        'text': texts,
        'labels': labels,
        'metadata': metadata
    }


class FixedUserInteractionModel(nn.Module):
    """Model that predicts interactions for fixed set of 500 users using GNN"""
    
    def __init__(self, text_processor, num_users: int, user_graph: Data, device):
        super().__init__()
        self.text_processor = text_processor
        self.num_users = num_users
        self.device = device
        
        # NLP feature dimension
        nlp_dim = 768 + 11 + 4 + 3  # embeddings + linguistic + sentiment + topic
        
        # Add metadata features (post_id, num_likes, num_comments, etc.)
        metadata_dim = 3  # num_likes, num_comments, timestamp features
        
        # GNN config for user graph
        gnn_config = GNNConfig(
            input_dim=user_graph.x.shape[1],  # Node feature dimension
            hidden_dim=256,
            num_layers=3,
            output_dim=128,
            dropout=0.3,
            use_batch_norm=False
        )
        
        # GNN to learn user embeddings from interaction graph
        self.user_gnn = GNNModel(gnn_config).to(device)
        self.user_graph = user_graph.to(device)
        
        # Post feature extractor (NLP + metadata)
        self.post_encoder = nn.Sequential(
            nn.Linear(nlp_dim + metadata_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Interaction predictor: combines user embedding + post features
        interaction_input_dim = gnn_config.output_dim + 128  # user_emb + post_features
        self.interaction_predictor = nn.Sequential(
            nn.Linear(interaction_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
            # No sigmoid - using BCEWithLogitsLoss
        )
        
        # Pre-compute user embeddings from graph (static)
        self._compute_user_embeddings()
    
    def _compute_user_embeddings(self):
        """Compute user embeddings from the graph using GNN"""
        # Get node embeddings before pooling
        x = self.user_graph.x
        edge_index = self.user_graph.edge_index
        
        # Run through GNN layers manually to get node-level embeddings
        x = self.user_gnn.input_proj(x)
        x = torch.relu(x)
        
        for i, gnn_layer in enumerate(self.user_gnn.gnn_layers):
            x_new = gnn_layer(x, edge_index)
            if self.user_gnn.config.use_batch_norm and self.user_gnn.batch_norms is not None:
                x_new = self.user_gnn.batch_norms[i](x_new)
            x_new = torch.relu(x_new)
            x = torch.dropout(x_new, p=self.user_gnn.config.dropout, train=False)
        
        # Project to output dimension using the output projection
        # The GNN's output_proj expects pooled features, but we want node-level
        # So we'll create a separate projection
        if not hasattr(self, 'node_embedding_proj'):
            self.node_embedding_proj = nn.Linear(
                self.user_gnn.config.hidden_dim, 
                self.user_gnn.config.output_dim
            ).to(self.device)
        
        user_embeddings = self.node_embedding_proj(x)
        
        # Make it a parameter so it can be fine-tuned
        self.user_embeddings_static = nn.Parameter(user_embeddings)
    
    def forward(self, text, metadata):
        # Extract NLP features
        features = self.text_processor.extract_all_features(text)
        nlp_features = np.concatenate([
            features['embeddings'],
            features['linguistic'],
            features['sentiment'],
            features['topic']
        ])
        nlp_tensor = torch.tensor(nlp_features, dtype=torch.float32).to(self.device)
        
        # Extract metadata features
        metadata_features = torch.tensor([
            metadata.get('num_likes', 0) / 100.0,  # Normalize
            metadata.get('num_comments', 0) / 100.0,
            0.0  # Placeholder for timestamp features
        ], dtype=torch.float32).to(self.device)
        
        # Combine NLP + metadata
        post_features = torch.cat([nlp_tensor, metadata_features])
        post_encoded = self.post_encoder(post_features)  # [128]
        
        # Get user embeddings from GNN (pre-computed)
        user_embs = self.user_embeddings_static  # [num_users, user_embedding_dim]
        
        # Combine user embeddings with post features
        # Expand post features to match number of users
        post_expanded = post_encoded.unsqueeze(0).expand(self.num_users, -1)  # [num_users, 128]
        combined = torch.cat([user_embs, post_expanded], dim=1)  # [num_users, user_emb_dim + 128]
        
        # Predict interactions for all users (logits)
        interaction_logits = self.interaction_predictor(combined).squeeze()  # [num_users]
        
        return interaction_logits


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        texts = batch['text']
        labels = batch['labels'].to(device)  # [batch_size, num_users]
        metadata_list = batch['metadata']
        
        batch_size = len(texts)
        batch_predictions = []
        
        for i in range(batch_size):
            preds = model(texts[i], metadata_list[i])
            batch_predictions.append(preds)
        
        # Stack predictions: [batch_size, num_users]
        predictions = torch.stack(batch_predictions)
        
        # Loss: BCEWithLogitsLoss (takes logits, not probabilities)
        loss = criterion(predictions, labels)
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store for metrics (convert logits to probabilities)
        pred_probs = torch.sigmoid(predictions)
        all_preds.extend(pred_probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
        num_batches += 1
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Accuracy: percentage of correct predictions across all user-post pairs
    # Use adaptive threshold based on positive rate
    pos_rate = np.mean(all_labels)
    threshold = max(0.1, min(0.5, pos_rate * 2))  # Adaptive threshold
    pred_binary = (all_preds > threshold).astype(int)
    accuracy = np.mean(pred_binary == all_labels)
    
    return total_loss / num_batches if num_batches > 0 else 0.0, accuracy


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
            labels = batch['labels'].to(device)
            metadata_list = batch['metadata']
            
            batch_size = len(texts)
            batch_predictions = []
            
            for i in range(batch_size):
                preds = model(texts[i], metadata_list[i])
                batch_predictions.append(preds)
            
            predictions = torch.stack(batch_predictions)
            loss = criterion(predictions, labels)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                # Convert logits to probabilities for metrics
                pred_probs = torch.sigmoid(predictions)
                all_preds.extend(pred_probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                num_batches += 1
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Accuracy with adaptive threshold
    pos_rate = np.mean(all_labels) if len(all_labels) > 0 else 0.04
    threshold = max(0.1, min(0.5, pos_rate * 2))
    pred_binary = (all_preds > threshold).astype(int)
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
    parser = argparse.ArgumentParser(description='Train fixed user interaction prediction model')
    parser.add_argument('--data', type=str, default='trump_posts_data.csv')
    parser.add_argument('--top_users', type=str, default='top_500_users.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load top users
    top_users = load_top_users(args.top_users)
    num_users = len(top_users)
    print(f"Loaded {num_users} top users")
    
    # Load data
    df = pd.read_csv(args.data, delimiter=';')
    print(f"Loaded {len(df)} posts")
    
    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    # Initialize
    text_processor = TextProcessor(device=device)
    
    # Create datasets
    train_dataset = FixedUserInteractionDataset(train_df, top_users, text_processor)
    val_dataset = FixedUserInteractionDataset(val_df, top_users, text_processor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Load user graph
    graph_data = torch.load('user_graph.pt', map_location=device, weights_only=False)
    user_graph = graph_data['graph']
    print(f"Loaded user graph: {user_graph.num_nodes} nodes, {user_graph.edge_index.shape[1]} edges")
    
    # Model
    model = FixedUserInteractionModel(text_processor, num_users, user_graph, device).to(device)
    
    # Loss and optimizer
    # Use weighted BCE to handle class imbalance (3.78% positive)
    pos_weight = torch.tensor([(25000 - 946) / 946], device=device)  # negative / positive
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Note: We'll need to remove sigmoid from predictor and use logits
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
                'top_users': top_users,
            }, output_dir / 'best_fixed_user_model.pt')
            print(f"Saved best model (val_acc: {val_acc:.4f})")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

