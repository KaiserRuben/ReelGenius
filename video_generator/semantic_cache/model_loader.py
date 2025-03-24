"""
Safe model loading utilities for SentenceTransformers.

This module provides utilities for safely loading SentenceTransformer models in environments
with multiple processes (like when using Celery workers) to avoid segmentation faults.
"""
import os
import logging
import numpy as np
import multiprocessing
import threading
import gc

# Force CPU only mode and disable CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_CUDA"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism in tokenizers

# Default embedding dimension for the model we use
DEFAULT_EMBEDDING_DIM = 384

# Global lock for thread safety
_model_lock = threading.RLock()

class ModelLoader:
    """Safely loads SentenceTransformer models in multiprocessing environments."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_folder='cache/models'):
        """Initialize with model parameters but don't load the model yet."""
        self.model_name = model_name
        self.cache_folder = cache_folder
        self._model = None
        self._pid = None
        self._model_process = None
        
    def get_model(self):
        """
        Safely get the model, reloading if the process ID has changed.
        This prevents segmentation faults when a process is forked.
        """
        with _model_lock:
            current_pid = os.getpid()
            
            # If we're in a new process or model hasn't been loaded yet
            if self._model is None or self._pid != current_pid:
                # Clean up old model
                if self._model is not None:
                    del self._model
                    # Explicitly collect garbage
                    gc.collect()
                
                self._pid = current_pid
                try:
                    # Ensure imports happen after forking
                    from sentence_transformers import SentenceTransformer
                    import torch
                    
                    # Force CPU to avoid CUDA issues in forked processes
                    # Must happen before model is loaded
                    torch.set_num_threads(1)  # Limit thread count
                    
                    # Create model with strict CPU-only settings
                    self._model = SentenceTransformer(
                        self.model_name, 
                        cache_folder=self.cache_folder,
                        device="cpu"
                    )
                    
                    # Disable gradient calculations completely
                    for param in self._model.parameters():
                        param.requires_grad = False
                    
                    # Set model to eval mode
                    self._model.eval()
                    
                    logging.info(f"Initialized SentenceTransformer (pid={self._pid}): {self.model_name}")
                except Exception as e:
                    logging.error(f"Failed to initialize SentenceTransformer: {e}")
                    self._model = None
                    
            return self._model
        
    def encode(self, texts, **kwargs):
        """Safely encode texts, with fallback for failures."""
        # We'll use a separate process for encoding to avoid segfaults
        try:
            model = self.get_model()
            if model is None:
                # Return zero vectors as fallback
                if isinstance(texts, str):
                    return np.zeros(DEFAULT_EMBEDDING_DIM)
                else:
                    return np.zeros((len(texts), DEFAULT_EMBEDDING_DIM))
            
            with _model_lock:    
                # Run the encoding in the current process with lock protection
                try:
                    # Use lighter encoding settings
                    kwargs.setdefault('batch_size', 1)
                    kwargs.setdefault('show_progress_bar', False)
                    kwargs.setdefault('convert_to_numpy', True)
                    
                    result = model.encode(texts, **kwargs)
                    return result
                except Exception as e:
                    logging.error(f"Error encoding texts: {e}")
                    # Return zero vectors as fallback
                    if isinstance(texts, str):
                        return np.zeros(DEFAULT_EMBEDDING_DIM)
                    else:
                        return np.zeros((len(texts), DEFAULT_EMBEDDING_DIM))
        except Exception as e:
            logging.error(f"Unexpected error in encode: {e}")
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