"""
Text embedding utilities
"""

import numpy as np
import torch
from typing import List, Optional
from transformers import AutoTokenizer, AutoModel


class TextEmbedder:
    """Handles text embeddings for posts"""
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        """
        Initialize text embedder
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
    
    def embed_batch(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Embed a batch of texts"""
        if not texts:
            return np.zeros((0, self.embedding_dim))
        
        # Tokenize batch
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def embed(self, text: str, max_length: int = 512) -> np.ndarray:
        """Embed a single text"""
        return self.embed_batch([text])[0]

