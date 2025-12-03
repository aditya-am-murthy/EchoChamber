# Heterogeneous GNN Implementation Guide

## Overview

Your project now has a **fully functional heterogeneous Graph Neural Network (GNN)** implementation that can handle multiple node types (users, posts, tags) and multiple edge types (likes, comments, authors, mentions, etc.) for more powerful virality prediction.

## ✅ What's Already Implemented

### 1. **HeteroGNNModel** (`models/gnn/hetero_gnn_model.py`)

- Handles 3 node types: `user`, `post`, `tag`
- Handles 7 edge types:
  - `(user, authors, post)` - authorship
  - `(user, likes, post)` - like interactions
  - `(user, comments, post)` - comment interactions
  - `(post, mentions, user)` - user mentions
  - `(post, has_tag, tag)` - hashtag associations
  - `(post, replies_to, post)` - reply threads
  - `(post, precedes, post)` - temporal relationships

### 2. **GraphBuilder** (`models/graph/graph_builder.py`)

- `build_hetero_graph()` method creates HeteroData objects
- Automatically constructs all 7 edge types from your data
- Generates type-specific node features:
  - **User nodes**: 6 features (followers, following, verified, influencer, avg_likes, posting_freq)
  - **Post nodes**: 21 features (favorites, reblogs, replies, media_count, hashtag_count, etc.)
  - **Tag nodes**: 1 feature (occurrence_count)

### 3. **IntegratedNLPGNNModel** (`models/integrated_model.py`)

- Already supports both homogeneous and heterogeneous GNN modes
- Set `use_hetero_gnn=True` to use heterogeneous GNN
- Combines NLP features with graph features for virality prediction

### 4. **Training Script** (`train.py`)

- Already configured to use `build_hetero_graph()`
- Already sets `use_hetero_gnn=True` by default
- Ready to train out of the box

## 🚀 How to Run the Model

### Quick Start (Basic Training)

```bash
# Train with heterogeneous GNN (default)
python train.py --data trump_posts_data.csv --epochs 10 --batch_size 4
```

### Advanced Options

```bash
# Full training with all features
python train.py \
  --data trump_posts_data.csv \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-4 \
  --output_dir checkpoints \
  --include_news \
  --include_news_agencies
```

### Command-Line Arguments

| Argument                  | Default                | Description                                  |
| ------------------------- | ---------------------- | -------------------------------------------- |
| `--data`                  | `trump_posts_data.csv` | Path to your data CSV file                   |
| `--epochs`                | `10`                   | Number of training epochs                    |
| `--batch_size`            | `4`                    | Batch size (reduce if OOM)                   |
| `--lr`                    | `1e-4`                 | Learning rate                                |
| `--output_dir`            | `checkpoints`          | Directory to save model checkpoints          |
| `--include_news`          | `False`                | Include news headline context (Experiment 1) |
| `--include_news_agencies` | `False`                | Include news agency nodes (Experiment 2)     |

## 📊 What the Model Does

### Input

- **Text**: Post content (processed by NLP module)
- **Graph**: Social network structure with multiple node/edge types
- **Optional News Context**: News headlines for contextual awareness

### Processing Pipeline

1. **NLP Module** (BERT-based)

   - Extracts 768-dim embeddings from post text
   - Computes linguistic features (12-dim)
   - Analyzes sentiment (4-dim)
   - Identifies topics (3-dim)
   - **Total NLP features: 787-dim**

2. **Heterogeneous GNN Module**

   - Projects each node type to common hidden dimension (256-dim)
   - Performs relation-specific message passing:
     - Different GNN layers for different edge types
     - SAGEConv for likes/authorship/mentions
     - GATConv for comments (attention-weighted)
   - Includes temporal attention for post sequences
   - Pools post node embeddings to graph-level representation
   - **Output: 128-dim graph features**

3. **Fusion & Prediction**
   - Combines NLP (787-dim) + GNN (128-dim) features
   - Passes through fusion layers (256-dim → 128-dim)
   - Final prediction head outputs virality score [0, 1]

### Output

- **Virality Score**: Float value between 0 and 1
  - Trained on normalized engagement (likes + 2×reblogs + 1.5×replies)
  - Higher score = predicted higher engagement

## 📁 Project Structure

```
EchoChamber/
├── models/
│   ├── gnn/
│   │   ├── hetero_gnn_model.py       ✅ Heterogeneous GNN implementation
│   │   ├── gnn_model.py              (Legacy homogeneous GNN)
│   │   ├── temporal_attention.py     ✅ Temporal attention layer
│   │   └── hierarchical_pooling.py   ✅ Hierarchical pooling
│   ├── graph/
│   │   └── graph_builder.py          ✅ HeteroData graph construction
│   ├── nlp/
│   │   └── text_processor.py         ✅ BERT-based NLP features
│   └── integrated_model.py           ✅ Main model combining NLP + GNN
├── train.py                          ✅ Training script
├── predict.py                        (Inference script)
└── HETEROGENEOUS_GNN_GUIDE.md        📖 This guide
```

## 🔧 Configuration

### GNN Configuration

The model uses `HeteroGNNConfig` with these default parameters:

```python
config = HeteroGNNConfig(
    user_input_dim=6,      # User node features
    post_input_dim=21,     # Post node features
    tag_input_dim=1,       # Tag node features
    hidden_dim=256,        # Hidden dimension for all node types
    num_layers=3,          # Number of GNN layers
    output_dim=128,        # Output dimension
    dropout=0.3,           # Dropout probability
    gnn_type='SAGE',       # GNN type (SAGE, GCN, or GAT)
    pooling='mean'         # Pooling strategy (mean, max, or both)
)
```

You can modify these in `train.py` lines 330-338.

### Training Configuration

Default training configuration:

- **Loss Function**: MSE (Mean Squared Error)
- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Device**: Automatic (CUDA if available, else CPU)

## 📈 Training Metrics

During training, you'll see these metrics:

- **Loss**: Mean squared error between predicted and actual virality
- **MAE**: Mean absolute error (average prediction error)
- **R²**: Coefficient of determination (how well model fits data)
- **Accuracy**: Binary classification accuracy (using median threshold)

Example output:

```
Epoch 1/10
Training: 100%|████████████| 25/25 [00:45<00:00]
Train Loss: 0.0234, Train MAE: 0.1123, Train R²: 0.7456, Train Acc: 0.8125
Evaluating: 100%|████████████| 6/6 [00:08<00:00]
Val Loss: 0.0289, Val MAE: 0.1345, Val R²: 0.7012, Val Acc: 0.7812
Saved best model (val_loss: 0.0289, val_acc: 0.7812)
```

## 🎯 Key Advantages of Heterogeneous GNN

### Compared to Homogeneous GNN:

1. **Relation-Specific Learning**

   - Different GNN layers for likes vs. comments vs. mentions
   - Model learns that comments indicate deeper engagement than likes

2. **Richer Node Features**

   - Post nodes: 21 features (vs. 7 in homogeneous)
   - User nodes: 6 features capturing influence and activity
   - Tag nodes: Topic-level virality patterns

3. **Semantic Awareness**

   - Understands authorship vs. engagement vs. mention relationships
   - Captures temporal momentum through `(post, precedes, post)` edges

4. **Better Generalization**

   - Learns structural patterns across different interaction types
   - Can predict engagement from unseen combinations of features

5. **Interpretable**
   - Can analyze which edge types contribute most to virality
   - Can inspect attention weights on comments vs. likes

## 🔍 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Solution**: Reduce batch size

```bash
python train.py --batch_size 2
```

#### 2. CUDA not available

**Solution**: The model automatically falls back to CPU. For GPU:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Import Errors

**Solution**: Ensure all dependencies are installed

```bash
pip install -r requirements.txt
```

#### 4. Data Loading Errors

**Solution**: Check your CSV file format

- Must use `;` as delimiter
- Must have columns: `post_id`, `likes`, `comments`, `author_username`, etc.
- JSON fields (likes, comments) should be properly formatted

#### 5. Dimension Mismatch

**Solution**: This usually means the graph builder produced unexpected feature dimensions. Check:

```python
# In Python console
from models.graph.graph_builder import GraphBuilder
import pandas as pd

df = pd.read_csv('trump_posts_data.csv', delimiter=';')
builder = GraphBuilder()
graph = builder.build_hetero_graph(df.iloc[[0]])

print("User features:", graph['user'].x.shape)  # Should be [num_users, 6]
print("Post features:", graph['post'].x.shape)  # Should be [num_posts, 21]
print("Tag features:", graph['tag'].x.shape)    # Should be [num_tags, 1]
```

## 📝 Next Steps

### 1. Basic Training

Start with default parameters to verify everything works:

```bash
python train.py --epochs 5 --batch_size 4
```

### 2. Hyperparameter Tuning

Experiment with different configurations:

```bash
# Try different learning rates
python train.py --lr 5e-5 --epochs 10

# Try different batch sizes
python train.py --batch_size 8 --epochs 10

# Try with news context
python train.py --include_news --epochs 10
```

### 3. Evaluation

After training, evaluate on test set:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt
```

### 4. Prediction

Make predictions on new posts:

```bash
python predict.py --checkpoint checkpoints/best_model.pt --post "Your post text here"
```

## 🔬 Advanced: Understanding the Architecture

### Message Passing Flow

```
1. Input Projection
   user[6] → hidden[256]
   post[21] → hidden[256]
   tag[1] → hidden[256]

2. Heterogeneous Convolution (x3 layers)
   Layer 1:
     (user, authors, post): SAGEConv
     (user, likes, post): SAGEConv
     (user, comments, post): GATConv [attention-weighted]
     (post, mentions, user): SAGEConv
     (post, has_tag, tag): SAGEConv
     (post, replies_to, post): GCNConv
     (post, precedes, post): GATConv + TemporalAttention

   Layer 2: [same structure]
   Layer 3: [same structure]

3. Hierarchical Pooling
   - Separate pools for: post, regular_user, influencer_user, tag
   - Concatenate: [256, 256, 256, 256] = 1024-dim
   - Project to output_dim: 1024 → 128

4. Integration with NLP
   NLP[787] + GNN[128] = 915-dim
   → Fusion[256] → Fusion[128] → Prediction[1]
```

### Edge Type Semantics

| Edge Type                  | Meaning              | GNN Layer          | Why?                               |
| -------------------------- | -------------------- | ------------------ | ---------------------------------- |
| `(user, authors, post)`    | User created post    | SAGE               | Aggregate author influence         |
| `(user, likes, post)`      | User liked post      | SAGE               | Shallow engagement signal          |
| `(user, comments, post)`   | User commented       | **GAT**            | **Deep engagement, use attention** |
| `(post, mentions, user)`   | Post mentions user   | SAGE               | Targeted engagement                |
| `(post, has_tag, tag)`     | Post has hashtag     | SAGE               | Topic association                  |
| `(post, replies_to, post)` | Post replies to post | GCN                | Thread structure                   |
| `(post, precedes, post)`   | Post before post     | **GAT + Temporal** | **Viral momentum**                 |

## 📚 References

- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **HeteroData Documentation**: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html
- **GraphSAGE Paper**: https://arxiv.org/abs/1706.02216
- **GAT Paper**: https://arxiv.org/abs/1710.10903

## 🆘 Support

If you encounter issues:

1. Check the error message carefully
2. Verify your data format matches the expected schema
3. Try reducing batch size or number of epochs
4. Check CUDA availability if using GPU
5. Ensure all dependencies are correctly installed

## ✨ Summary

Your heterogeneous GNN is **ready to use**! Simply run:

```bash
python train.py --data trump_posts_data.csv --epochs 10 --batch_size 4
```

The model will:

- ✅ Load your data
- ✅ Build heterogeneous graphs with 3 node types and 7 edge types
- ✅ Extract NLP features from post text
- ✅ Learn relation-specific message passing
- ✅ Combine graph and text features
- ✅ Predict virality scores
- ✅ Save the best model to `checkpoints/best_model.pt`

**Good luck with your training!** 🚀
