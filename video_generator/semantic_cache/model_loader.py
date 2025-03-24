"""
Safe model loading utilities for SentenceTransformers.

This module provides utilities for safely loading SentenceTransformer models in environments
with multiple processes (like when using Celery workers) to avoid segmentation faults.
"""
import os
import logging
import numpy as np

# Default embedding dimension for the model we use
DEFAULT_EMBEDDING_DIM = 384

class ModelLoader:
    """Safely loads SentenceTransformer models in multiprocessing environments."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_folder='cache/models'):
        """Initialize with model parameters but don't load the model yet."""
        self.model_name = model_name
        self.cache_folder = cache_folder
        self._model = None
        self._pid = None
        
    def get_model(self):
        """
        Safely get the model, reloading if the process ID has changed.
        This prevents segmentation faults when a process is forked.
        """
        current_pid = os.getpid()
        
        # If we're in a new process or model hasn't been loaded yet
        if self._model is None or self._pid != current_pid:
            self._pid = current_pid
            try:
                # Delay import until needed to avoid import-time problems
                from sentence_transformers import SentenceTransformer
                
                # Force CPU to avoid CUDA issues in forked processes
                self._model = SentenceTransformer(
                    self.model_name, 
                    cache_folder=self.cache_folder,
                    device="cpu"
                )
                logging.info(f"Initialized SentenceTransformer (pid={self._pid}): {self.model_name}")
            except Exception as e:
                logging.error(f"Failed to initialize SentenceTransformer: {e}")
                self._model = None
                
        return self._model
        
    def encode(self, texts, **kwargs):
        """Safely encode texts, with fallback for failures."""
        model = self.get_model()
        if model is None:
            # Return zero vectors as fallback
            if isinstance(texts, str):
                return np.zeros(DEFAULT_EMBEDDING_DIM)
            else:
                return np.zeros((len(texts), DEFAULT_EMBEDDING_DIM))
                
        try:
            return model.encode(texts, **kwargs)
        except Exception as e:
            logging.error(f"Error encoding texts: {e}")
            # Return zero vectors as fallback
            if isinstance(texts, str):
                return np.zeros(DEFAULT_EMBEDDING_DIM)
            else:
                return np.zeros((len(texts), DEFAULT_EMBEDDING_DIM))
    
    def get_dimension(self):
        """Get the embedding dimension of the model."""
        model = self.get_model()
        if model is None:
            return DEFAULT_EMBEDDING_DIM
            
        try:
            return model.get_sentence_embedding_dimension()
        except Exception:
            return DEFAULT_EMBEDDING_DIM


# Singleton instance for reuse
model_loader = ModelLoader()