# ML Models and Experiments Setup Guide

This guide explains how to set up and use the NLP + GNN models for virality prediction.

## Overview

The system combines:
- **NLP**: Text decomposition and feature extraction (BERT embeddings, sentiment, linguistic features)
- **GNN**: Social network graph learning (user-post interactions, node types)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (if not already downloaded):
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Project Structure

```
EchoChamber/
├── models/              # ML model implementations
│   ├── nlp/            # NLP components
│   ├── graph/         # Graph construction
│   ├── gnn/           # GNN models
│   └── integrated_model.py
├── experiments/        # Experiment frameworks
│   ├── experiment1_news_context.py
│   └── experiment2_node_types.py
├── train.py           # Training script
├── evaluate.py        # Evaluation script
├── predict.py         # Prediction script
└── requirements.txt
```

## Quick Start

### 1. Training a Model

Train the baseline model:
```bash
python train.py --data trump_posts_data.csv --epochs 10 --batch_size 4
```

Train with news context (Experiment 1):
```bash
python train.py --data trump_posts_data.csv --include_news --epochs 10
```

Train with news agency nodes (Experiment 2):
```bash
python train.py --data trump_posts_data.csv --include_news_agencies --epochs 10
```

### 2. Evaluating a Model

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --data trump_posts_data.csv
```

### 3. Making Predictions

```bash
python predict.py --checkpoint checkpoints/best_model.pt --text "Your text prompt here"
```

## Experiments

### Experiment 1: News Context

Tests whether adding current news headlines improves predictions.

```bash
cd experiments
python experiment1_news_context.py
```

Results saved to `results/exp1/exp1_results.json`

### Experiment 2: Node Types

Tests whether distinguishing node types (users, news agencies, influencers) improves predictions.

```bash
cd experiments
python experiment2_node_types.py
```

Results saved to `results/exp2/exp2_results.json`

## Model Architecture

### NLP Component
- **Text Decomposition**: Extracts words, sentences, hashtags, mentions
- **Feature Extraction**:
  - BERT embeddings (768-dim)
  - Linguistic features (12-dim): word count, readability, etc.
  - Sentiment features (4-dim): VADER scores
  - Topic features (3-dim): controversy, political, emotional scores

### GNN Component
- **Graph Structure**:
  - Nodes: Users, posts, news agencies (optional)
  - Edges: Like (weight=1.0), Comment (weight=2.0)
- **Architecture**: GCN/GAT/GraphSAGE with 3 layers
- **Output**: 128-dim graph representation

### Fusion & Prediction
- Concatenate NLP + GNN features
- Optional: Add news context embeddings
- Feed through MLP → Sigmoid → Virality score [0, 1]

## Data Format

Expected CSV format:
```csv
post_id;likes;comments
12345;"[""user1"", ""user2""]";"[[""user3"", ""comment text""]]"
```

- `post_id`: Unique post identifier
- `likes`: JSON array of usernames who liked
- `comments`: JSON array of [username, comment_text] tuples

## Configuration

### GNN Config
```python
from models.gnn.gnn_model import GNNConfig

config = GNNConfig(
    input_dim=768,      # BERT embedding dimension
    hidden_dim=256,     # Hidden layer size
    num_layers=3,       # Number of GNN layers
    output_dim=128,     # Graph representation size
    dropout=0.3,        # Dropout rate
    gnn_type='GCN'      # 'GCN', 'GAT', or 'SAGE'
)
```

## Performance Tips

1. **GPU**: Use CUDA for faster training (automatically detected)
2. **Batch Size**: Adjust based on GPU memory (default: 4)
3. **Learning Rate**: Default 1e-4, adjust if loss doesn't decrease
4. **Graph Size**: Large graphs may require batching or sampling

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python path includes project root

### CUDA Errors
- Model falls back to CPU if CUDA unavailable
- Check `torch.cuda.is_available()` for GPU status

### Memory Issues
- Reduce batch size
- Use smaller GNN hidden dimensions
- Process graphs in smaller batches

## Next Steps

1. **Data Collection**: Expand dataset with more posts and interactions
2. **Feature Engineering**: Add more linguistic/topic features
3. **Hyperparameter Tuning**: Optimize GNN architecture and training params
4. **News Integration**: Connect to real news APIs for Experiment 1
5. **Evaluation**: Add more metrics (precision, recall, F1 for classification)

## References

- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- BERT Paper: https://arxiv.org/abs/1810.04805
- GCN Paper: https://arxiv.org/abs/1609.02907

