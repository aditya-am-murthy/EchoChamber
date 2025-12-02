"""
Text processing and feature extraction for posts
"""

import re
import string
from typing import Dict, List, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from textstat import flesch_reading_ease, flesch_kincaid_grade
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextProcessor:
    """Processes text posts to extract linguistic and semantic features"""
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        """
        Initialize text processor
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Load GoEmotions model & tokenizer
        model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.emotion_model.to(self.device)
        self.emotion_model.eval()

    
    def clean_text(self, text: str) -> str:
        """Clean HTML tags and normalize text"""
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features from text"""
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return {
                'word_count': 0,
                'char_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'uppercase_ratio': 0,
                'hashtag_count': 0,
                'mention_count': 0,
                'readability_score': 0,
                'reading_grade': 0,
            }
        
        words = cleaned.split()
        sentences = nltk.sent_tokenize(cleaned)
        
        features = {
            'word_count': len(words),
            'char_count': len(cleaned),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'exclamation_count': cleaned.count('!'),
            'question_count': cleaned.count('?'),
            'uppercase_ratio': sum(1 for c in cleaned if c.isupper()) / len(cleaned) if cleaned else 0,
            'hashtag_count': len(re.findall(r'#\w+', text)),
            'mention_count': len(re.findall(r'@\w+', text)),
            'readability_score': flesch_reading_ease(cleaned),
            'reading_grade': flesch_kincaid_grade(cleaned),
        }
        
        return features
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features using VADER"""
        cleaned = self.clean_text(text)
        sentiment = self.sentiment_analyzer.polarity_scores(cleaned)
        
        return {
            'sentiment_compound': sentiment['compound'],
            'sentiment_pos': sentiment['pos'],
            'sentiment_neu': sentiment['neu'],
            'sentiment_neg': sentiment['neg'],
        }
    
    def extract_topic_features(self, text: str) -> Dict[str, float]:
        """Extract topic-related features (controversy indicators, keywords)"""
        cleaned = self.clean_text(text).lower()
        
        # Controversy keywords (can be expanded)
        controversy_keywords = [
            'fake', 'fraud', 'rigged', 'corrupt', 'witch hunt',
            'deep state', 'conspiracy', 'cover up', 'scandal'
        ]
        
        # Political keywords
        political_keywords = [
            'election', 'vote', 'democrat', 'republican', 'biden', 'trump',
            'congress', 'senate', 'president', 'policy', 'law'
        ]
        
        # Emotional keywords
        emotional_keywords = [
            'amazing', 'terrible', 'disaster', 'incredible', 'horrible',
            'fantastic', 'worst', 'best', 'love', 'hate'
        ]
        
        controversy_count = sum(1 for kw in controversy_keywords if kw in cleaned)
        political_count = sum(1 for kw in political_keywords if kw in cleaned)
        emotional_count = sum(1 for kw in emotional_keywords if kw in cleaned)
        
        return {
            'controversy_score': controversy_count / max(len(cleaned.split()), 1),
            'political_score': political_count / max(len(cleaned.split()), 1),
            'emotional_score': emotional_count / max(len(cleaned.split()), 1),
        }
    
    def extract_emotion_features(self, text: str) -> Dict[str, float]:
        """
        Extract emotion features using the GoEmotions model
        (joeddav/distilbert-base-uncased-go-emotions-student).

        Returns aggregated probabilities over a few macro-emotion groups.
        """
        cleaned = self.clean_text(text)

        # Tokenize
        inputs = self.emotion_tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
            logits = outputs.logits  # shape: [1, num_labels]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Map id -> label -> probability
        id2label = self.emotion_model.config.id2label
        emotion_scores = {
            id2label[i]: float(probs[i]) for i in range(len(probs))
        }

        # Helper to sum groups safely
        def group_prob(labels):
            return float(sum(emotion_scores.get(lbl, 0.0) for lbl in labels))

        # Aggregate into macro emotion dimensions
        emo_joy = group_prob(["joy", "amusement", "excitement"])
        emo_sadness = group_prob(["sadness", "grief", "disappointment", "remorse"])
        emo_anger = group_prob(["anger", "annoyance", "disgust"])
        emo_fear = group_prob(["fear", "nervousness"])
        emo_love = group_prob([
            "love", "admiration", "caring", "gratitude",
            "pride", "relief", "optimism", "approval",
        ])
        emo_surprise = group_prob(["surprise", "realization", "confusion"])
        emo_neutral = emotion_scores.get("neutral", 0.0)

        # Summary stats
        max_emotion_score = float(max(emotion_scores.values()))
        # entropy-like "emotional spread"
        emo_spread = float(
            -sum(p * (0 if p <= 0 else torch.log(torch.tensor(p))).item()
                 for p in emotion_scores.values())
        )

        return {
            "emo_joy": emo_joy,
            "emo_sadness": emo_sadness,
            "emo_anger": emo_anger,
            "emo_fear": emo_fear,
            "emo_love": emo_love,
            "emo_surprise": emo_surprise,
            "emo_neutral": emo_neutral,
            "emo_max_score": max_emotion_score,
            "emo_entropy": emo_spread,
        }
    
    def get_embeddings(self, text: str, max_length: int = 512) -> np.ndarray:
        """Get BERT embeddings for text"""
        cleaned = self.clean_text(text)
        
        if not cleaned:
            # Return zero vector
            return np.zeros(self.model.config.hidden_size)
        
        # Tokenize and encode
        encoded = self.tokenizer(
            cleaned,
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
        
        return embeddings[0]
    
    def extract_all_features(self, text: str) -> Dict[str, np.ndarray]:
        """Extract all features from text"""
        linguistic = self.extract_linguistic_features(text)
        sentiment = self.extract_sentiment_features(text)
        topic = self.extract_topic_features(text)
        emotion = self.extract_emotion_features(text) 
        embeddings = self.get_embeddings(text)
        
        return {
            'linguistic': np.array(list(linguistic.values())),
            'sentiment': np.array(list(sentiment.values())),
            'topic': np.array(list(topic.values())),
            'emotion': np.array(list(emotion.values())),
            'embeddings': embeddings,
        }
    
    def decompose_text(self, text: str) -> Dict:
        """
        Decompose text into structured components
        
        Returns:
            Dictionary with decomposed text components
        """
        cleaned = self.clean_text(text)
        
        # Extract components
        hashtags = re.findall(r'#(\w+)', text)
        mentions = re.findall(r'@(\w+)', text)
        urls = re.findall(r'http\S+|www\.\S+', text)
        
        # Tokenize
        words = nltk.word_tokenize(cleaned)
        sentences = nltk.sent_tokenize(cleaned)
        
        return {
            'original': text,
            'cleaned': cleaned,
            'words': words,
            'sentences': sentences,
            'hashtags': hashtags,
            'mentions': mentions,
            'urls': urls,
            'features': self.extract_all_features(text),
        }