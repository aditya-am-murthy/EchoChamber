#!/bin/bash
# Setup script for ML models

echo "Setting up ML environment for EchoChamber..."

# Create virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"

# Create directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p results/exp1
mkdir -p results/exp2
mkdir -p data

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Ensure your data CSV is in the project root (trump_posts_data.csv)"
echo "2. Train a model: python train.py --data trump_posts_data.csv"
echo "3. Run experiments: python run_experiments.py"
echo "4. Make predictions: python predict.py --checkpoint checkpoints/best_model.pt --text 'Your text here'"

