"""
Base experiment class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import pandas as pd
from pathlib import Path


class BaseExperiment(ABC):
    """Base class for experiments"""
    
    def __init__(self, data_path: str, output_dir: str = "results"):
        """
        Initialize experiment
        
        Args:
            data_path: Path to CSV data
            output_dir: Directory to save results
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    @abstractmethod
    def setup(self):
        """Setup experiment (load data, prepare features, etc.)"""
        pass
    
    @abstractmethod
    def run(self):
        """Run experiment"""
        pass
    
    @abstractmethod
    def evaluate(self):
        """Evaluate experiment results"""
        pass
    
    def save_results(self, filename: str = "results.json"):
        """Save experiment results"""
        import json
        results_path = self.output_dir / filename
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {results_path}")

