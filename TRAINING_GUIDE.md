# Training Guide

## Quick Start

### 1. Find Top Users (if not already done)
```bash
python3 find_top_users.py --data trump_posts_data.csv --top_n 500 --output top_500_users.csv
```

### 2. Build User Graph
```bash
python3 build_user_graph.py --data trump_posts_data.csv --top_users top_500_users.csv --output user_graph.pt --min_co_interactions 1
```

### 3. Train the Model
```bash
python3 train_user_interactions_fixed.py \
    --data trump_posts_data.csv \
    --top_users top_500_users.csv \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-4 \
    --output_dir checkpoints
```

## Full Training Command

```bash
python3 train_user_interactions_fixed.py \
    --data trump_posts_data.csv \
    --top_users top_500_users.csv \
    --epochs 20 \
    --batch_size 4 \
    --lr 1e-4 \
    --output_dir checkpoints
```

## What the Model Does

1. **Fixed Set of 500 Users**: Only predicts interactions for the top 500 most interactive users
2. **GNN Learning**: Uses a user-user graph (based on co-interaction patterns) to learn user embeddings
3. **NLP Processing**: Processes post text + metadata (timestamp, engagement counts) to extract features
4. **Binary Predictions**: Outputs a 500-dim binary vector indicating which users will interact (like/comment)

## Output

- **Model checkpoint**: `checkpoints/best_fixed_user_model.pt`
- **Metrics displayed**: Loss, Accuracy, Precision, Recall, F1
- **Predictions**: Binary vector of length 500 (one per user)

## Notes

- The model uses weighted BCE loss to handle class imbalance (3.78% positive rate)
- Validation accuracy may be low due to extreme imbalance, but precision/recall are more informative
- The GNN learns user relationships from historical interaction patterns

