# Quick Start: Heterogeneous GNN for Virality Prediction

## ✅ Your Setup is Complete!

Your project already has a fully functional heterogeneous GNN implementation. Everything is configured and ready to run.

## 🚀 Run Training (3 Steps)

### Step 1: Validate Setup

```bash
python validate_hetero_gnn.py
```

This checks that all components work correctly.

### Step 2: Train Model

```bash
python train.py --data trump_posts_data.csv --epochs 10 --batch_size 4
```

### Step 3: Check Results

Model will be saved to `checkpoints/best_model.pt`

## 📊 What You'll See

```
Epoch 1/10
Training: 100%|████████████| 25/25 [00:45<00:00]
Train Loss: 0.0234, Train MAE: 0.1123, Train R²: 0.7456, Train Acc: 0.8125
Evaluating: 100%|████████████| 6/6 [00:08<00:00]
Val Loss: 0.0289, Val MAE: 0.1345, Val R²: 0.7012, Val Acc: 0.7812
Saved best model ✓
```

## 🎯 Key Metrics

- **Loss**: Lower is better (target: < 0.05)
- **MAE**: Mean error in virality prediction (target: < 0.15)
- **R²**: How well model fits data (target: > 0.7)
- **Accuracy**: Binary classification accuracy (target: > 0.75)

## 🔧 Adjust if Needed

### Out of Memory?

```bash
python train.py --batch_size 2
```

### Want More Epochs?

```bash
python train.py --epochs 20
```

### Include News Context?

```bash
python train.py --include_news --epochs 10
```

## 📖 Architecture Overview

### Node Types (3)

- **user**: Social media users (6 features)
- **post**: Social media posts (21 features)
- **tag**: Hashtags/topics (1 feature)

### Edge Types (7)

- `(user, authors, post)` - Authorship
- `(user, likes, post)` - Likes
- `(user, comments, post)` - Comments (with attention)
- `(post, mentions, user)` - Mentions
- `(post, has_tag, tag)` - Hashtags
- `(post, replies_to, post)` - Reply threads
- `(post, precedes, post)` - Temporal momentum (with attention)

### Processing Flow

```
Text → NLP (BERT) → 787-dim features
Graph → Hetero-GNN → 128-dim features
Combined → Fusion → Virality Score [0, 1]
```

## 🔍 Files Modified/Created

✅ **Already Implemented:**

- `models/gnn/hetero_gnn_model.py` - Heterogeneous GNN
- `models/graph/graph_builder.py` - HeteroData construction
- `models/integrated_model.py` - NLP + GNN integration
- `train.py` - Training script (uses hetero by default)

📖 **Documentation:**

- `HETEROGENEOUS_GNN_GUIDE.md` - Complete guide
- `HETERO_GNN_QUICK_START.md` - This file
- `validate_hetero_gnn.py` - Validation script

## ⚡ Quick Commands

```bash
# Validate setup
python validate_hetero_gnn.py

# Train (default)
python train.py --epochs 10 --batch_size 4

# Train with news
python train.py --epochs 10 --include_news

# Train longer
python train.py --epochs 20 --batch_size 8 --lr 5e-5

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🆘 Troubleshooting

| Issue              | Solution                          |
| ------------------ | --------------------------------- |
| Out of memory      | Reduce `--batch_size`             |
| CUDA not found     | Install CUDA-enabled PyTorch      |
| Import errors      | `pip install -r requirements.txt` |
| Data errors        | Check CSV delimiter is `;`        |
| Dimension mismatch | Run `validate_hetero_gnn.py`      |

## 📚 More Information

- **Full Guide**: See `HETEROGENEOUS_GNN_GUIDE.md`
- **Training Guide**: See `TRAINING_GUIDE.md`
- **ML Setup**: See `ML_SETUP.md`

## ✨ Summary

Your heterogeneous GNN is **ready to use**. Just run:

```bash
python train.py --data trump_posts_data.csv --epochs 10 --batch_size 4
```

**That's it!** The model will train and save the best checkpoint automatically. 🎉
