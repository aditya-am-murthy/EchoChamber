# Experiments

This directory contains experiments for evaluating different model configurations.

## Experiment 1: News Context Addition

**File**: `experiment1_news_context.py`

**Goal**: Evaluate the impact of adding current news headlines as additional context.

**Methodology**:
- Baseline: NLP + GNN model without news context
- Enhanced: NLP + GNN model with news headline embeddings
- Compare prediction accuracy and error metrics

**Usage**:
```bash
python experiments/experiment1_news_context.py
```

## Experiment 2: Node Type Distinctions

**File**: `experiment2_node_types.py`

**Goal**: Evaluate the impact of distinguishing different node types (users, news agencies, influencers).

**Methodology**:
- Baseline: Graph with only USER and POST nodes
- Enhanced: Graph with USER, POST, NEWS_AGENCY, and INFLUENCER nodes
- Compare how node type information affects predictions

**Usage**:
```bash
python experiments/experiment2_node_types.py
```

## Running Experiments

Both experiments follow the same structure:
1. `setup()`: Load data and initialize models
2. `run()`: Execute experiment
3. `evaluate()`: Compute metrics and compare models
4. Results saved to `results/` directory

