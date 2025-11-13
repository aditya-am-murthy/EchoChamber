# ML Models for EchoChamber

This directory contains the machine learning models for predicting post virality using NLP and Graph Neural Networks.

## Architecture

### NLP Module (`models/nlp/`)
- **TextProcessor**: Extracts linguistic, sentiment, and topic features from text
- **TextEmbedder**: Generates BERT embeddings for text
- Features extracted:
  - Linguistic: word count, readability, punctuation usage
  - Sentiment: VADER sentiment scores
  - Topic: controversy, political, emotional keywords
  - Embeddings: BERT-based semantic embeddings

### Graph Module (`models/graph/`)
- **GraphBuilder**: Constructs social network graphs from interaction data
- **NodeType**: Defines node types (USER, POST, NEWS_AGENCY, INFLUENCER, etc.)
- Graph structure:
  - Nodes: Users, posts, news agencies (optional)
  - Edges: Like and comment interactions (weighted)

### GNN Module (`models/gnn/`)
- **GNNModel**: Graph Neural Network for learning social patterns
- Supports: GCN, GAT, and GraphSAGE architectures
- Learns user-post interaction patterns

### Integrated Model (`models/integrated_model.py`)
- **IntegratedNLPGNNModel**: Combines NLP and GNN
- Architecture:
  1. NLP processes text → extracts features
  2. GNN processes graph → learns network patterns
  3. Features fused → prediction head → virality score

## Usage

```python
from models.integrated_model import IntegratedNLPGNNModel
from models.graph.graph_builder import GraphBuilder
from models.gnn.gnn_model import GNNConfig

# Initialize
graph_builder = GraphBuilder(include_news_agencies=True)
gnn_config = GNNConfig(hidden_dim=256, num_layers=3)
model = IntegratedNLPGNNModel(gnn_config=gnn_config)

# Predict
virality_score = model.predict(text, graph, news_context)
```

## Experiments

See `../experiments/` for:
- Experiment 1: News context addition
- Experiment 2: Node type distinctions

