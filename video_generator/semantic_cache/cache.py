"""
Caching system that provides function result caching using Redis or in-memory fallback.

This module provides decorators for caching function results with automatic expiration.
It supports both synchronous and asynchronous functions, and will automatically fall back
to in-memory caching if Redis is unavailable.

Example usage:
    @cached("user_data", 3600)
    async def get_user(user_id: int):
        return await db.fetch_user(user_id)

    @cached("pricing", 1800)
    def calculate_price(items: list[str]):
        return sum(price_lookup[item] for item in items)

Note:
    - cached_sync is maintained for backwards compatibility
    - cache keys are automatically generated based on function arguments
    - cache entries are automatically expired after the specified time
    - pickle is used for serialization, so cached values must be pickle-able
"""
import datetime
import hashlib
import logging
import os
import pickle
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from inspect import iscoroutinefunction
from typing import Callable, Any, Union, TypeVar, ParamSpec, List

import numpy as np

from video_generator.semantic_cache.my_redis import get_redis_client
from video_generator.semantic_cache.model_loader import model_loader

# Type variables for better generic type hints
T = TypeVar('T')  # Return type
P = ParamSpec('P')  # Parameter specification


class CacheManager:
    """Handles cache operations and key management."""

    def __init__(self, cleanup_interval: int = 300):
        """
        Initialize cache manager with separate Redis databases for cache and embeddings.

        Args:
            cleanup_interval: Cleanup interval for cache entries in seconds
        """
        # DB 0 for actual cache entries (with expiration)
        self.redis_cache = get_redis_client(cleanup_interval=cleanup_interval, db=0)
        # DB 1 for embeddings (no expiration)
        self.redis_embeddings = get_redis_client(cleanup_interval=cleanup_interval, db=1)

        # Use the safe model loader instead of direct initialization
        self._embedding_cache = OrderedDict()
        self._max_cache_size = 10000
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Optimized embedding retrieval with better caching."""
        # First check the cache
        if text in self._embedding_cache:
            self._embedding_cache.move_to_end(text)
            return self._embedding_cache[text]

        # Use the safe model loader that handles forking correctly
        embedding = model_loader.encode(
            [text],
            normalize_embeddings=True,  # Precompute normalization
            convert_to_numpy=True,      # Ensure numpy output
            show_progress_bar=False     # Disable progress bar for single items
        )[0]
        
        # Cache the result
        if len(self._embedding_cache) >= self._max_cache_size:
            self._embedding_cache.popitem(last=False)
            
        self._embedding_cache[text] = embedding
        return embedding

    def find_similar_keys(self, query: str, query_embedding: np.ndarray, threshold: float = 0.85) -> list[
        dict[str, str | float]]:
        """Find similar keys by comparing stored embeddings."""
        if ':' not in query:
            return []

        prefix, query_hash = query.split(':', 1)
        logging.info(f"CACHE - SIMILARITY SEARCH - Prefix: '{prefix}', Hash: '{query_hash}'")

        similar_keys = []
        cursor = 0
        batch_size = 1000

        while True:
            # Get batch of keys
            cursor, keys = self.redis_cache.scan(cursor=cursor, match=f'{prefix}:*', count=batch_size)
            if not keys:
                break

            # Batch process embeddings
            embedding_keys = []
            cache_keys = []
            for key in keys:
                key_str = key.decode('utf-8')
                key_hash = key_str.split(':', 1)[1]
                embedding_keys.append(f"{prefix}:embeddings:{key_hash}")
                cache_keys.append(key_str)

            # Pipeline the embedding retrievals
            pipeline = self.redis_embeddings.pipeline()
            for emb_key in embedding_keys:
                pipeline.get(emb_key)
            stored_embeddings_bytes = pipeline.execute()

            # Process embeddings in parallel using numpy
            valid_embeddings = []
            valid_keys = []

            for idx, emb_bytes in enumerate(stored_embeddings_bytes):
                if emb_bytes is not None:
                    valid_embeddings.append(np.frombuffer(emb_bytes, dtype=np.float32))
                    valid_keys.append(cache_keys[idx])

            if valid_embeddings:
                # Stack all embeddings for vectorized computation
                embeddings_matrix = np.vstack(valid_embeddings)

                # Vectorized similarity computation
                query_normalized = query_embedding / np.linalg.norm(query_embedding)
                embeddings_normalized = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
                similarities = np.dot(embeddings_normalized, query_normalized)

                # Find matches above threshold
                matches = similarities > threshold
                for idx, is_match in enumerate(matches):
                    if is_match:
                        similar_keys.append({
                            'key': valid_keys[idx],
                            'similarity': float(similarities[idx])
                        })

            if cursor == 0 or len(similar_keys) >= 5:  # Early stopping if we have enough matches
                break

        similar_keys.sort(key=lambda x: x['similarity'], reverse=True)
        if len(similar_keys) > 0:
            logging.info(f"CACHE - SIMILARITY SEARCH - Found {len(similar_keys)} similar keys:")
        for key in similar_keys:
            logging.info(f"\t- {key['key']} (similarity: {key['similarity']:.3f})")
        return similar_keys[:5]

    def generate_cache_key(self, prefix: str, args: tuple, kwargs: dict, use_similarity=None) -> tuple[str, np.ndarray]:
        """Generate cache key and store embedding for later similarity comparison."""

        def clean_message_object(obj):
            str_content = str(obj)
            content_pattern = r"content='([^']*)'|content=\"([^\"]*)\""
            matches = re.findall(content_pattern, str_content)
            cleaned_content = ' '.join(match[0] or match[1] for match in matches if match[0] or match[1])

            if not cleaned_content:
                str_content = re.sub(r"Human(?:Message)?|AI(?:Message)?|content=?", " ", str_content)
                str_content = re.sub(r"id='[^']*'", " ", str_content)
                str_content = re.sub(r'additional_kwargs={}', ' ', str_content)
                str_content = re.sub(r'response_metadata={}', ' ', str_content)
                # # remove True/False
                # str_content = re.sub(r'True|False', ' ', str_content)
                cleaned_content = str_content

            cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
            cleaned_content = re.sub(r'[^\w\s]', '', cleaned_content)
            return cleaned_content.strip()

        # Clean up the input content
        cleaned_args = clean_message_object(args)
        cleaned_kwargs = clean_message_object(kwargs)
        content = f"{cleaned_args} {cleaned_kwargs}".strip()

        logging.info(f"CACHE - KEY GENERATION - Cleaned content: '{content}'")

        # Generate embedding
        embedding = self._get_embedding(content)

        # Create hash from raw embedding bytes
        embedding_bytes = embedding.tobytes()
        semantic_hash = hashlib.sha256(embedding_bytes).hexdigest()

        # Store the embedding in DB 1 (embeddings database)
        embedding_key = f"{prefix}:embeddings:{semantic_hash}"
        self.redis_embeddings.set(embedding_key, embedding_bytes)

        final_cache_key = f"{prefix}:{semantic_hash}"

        # Also log the mapping to the same file
        # if os.getenv("MODE", "production") == "development":
        with open("cache_keys.csv", "a") as f:
            time = datetime.datetime.now()
            f.write(f"{time},{prefix},{semantic_hash},{use_similarity},{content}\n")

        return final_cache_key, embedding

    def get_cached_value(self, cache_key: Union[str, tuple], use_similarity: bool = False,
                         similarity_threshold: float = 0.85) -> tuple[bool, Any]:
        """Get value from cache DB, optionally checking for similar keys."""
        if isinstance(cache_key, tuple):
            cache_key, query_embedding = cache_key
        else:
            query_embedding = None

        # Try exact match from cache DB first
        cached_result = self.redis_cache.get(cache_key)
        if cached_result is not None:
            try:
                data = pickle.loads(cached_result)
                logging.info(f"CACHE - HIT for key {cache_key}")
                return True, data
            except pickle.PickleError:
                logging.error(f"CACHE - FAILED to decode cached result for key {cache_key}")
                self.redis_cache.delete(cache_key)

        # If no exact match and similarity is enabled, try similar keys
        if use_similarity and query_embedding is not None:
            similar_keys = self.find_similar_keys(cache_key, query_embedding, threshold=similarity_threshold)
            for similar_key in similar_keys:
                cached_result = self.redis_cache.get(similar_key['key'])
                if cached_result is not None:
                    try:
                        data = pickle.loads(cached_result)
                        logging.info(
                            f"CACHE - HIT - SIMILAR KEY {similar_key['key']} (similarity: {similar_key['similarity']:.3f})")
                        return True, data
                    except pickle.PickleError:
                        self.redis_cache.delete(similar_key['key'])

        return False, None

    def set_cached_value(self, cache_key: str, value: Any, expire_seconds: int) -> None:
        """Set value in cache DB with expiration."""
        try:
            # Store the actual cache entry in DB 0 with expiration
            if expire_seconds > 0:
                self.redis_cache.setex(cache_key, expire_seconds, pickle.dumps(value))
            else:
                self.redis_cache.set(cache_key, pickle.dumps(value))
            logging.info(f"CACHE - SET key {cache_key} with expiry {expire_seconds}s")
        except (TypeError, pickle.PickleError) as e:
            logging.error(f"Failed to cache result for key {cache_key}: {e}")
            
    def get_sentence_embedding_dimension(self):
        """Get the dimension of sentence embeddings using the safe model loader."""
        return model_loader.get_dimension()


# Initialize a shared cache manager
_cache_manager = CacheManager()


def get_cached(
        cache_key: str,
        args: tuple = (),
        kwargs: dict = {},
        use_similarity: bool = False,
        similarity_threshold: float = 0.85,
        use_as_cache_key=None
):
    """
    Manually retrieve a value from the cache.

    Args:
        cache_key: Base cache key
        args: Arguments to include in the cache key
        kwargs: Keyword arguments to include in the cache key
        use_similarity: Whether to check for similar keys on cache miss
        similarity_threshold: Similarity threshold for matching
        use_as_cache_key: Optional function or value to use as key

    Returns:
        Tuple of (success, value)
    """
    # Handle use_as_cache_key appropriately
    if use_as_cache_key is not None:
        if callable(use_as_cache_key):
            # It's a function, call it with the provided args
            custom_key = use_as_cache_key(*args, **kwargs)
            cache_entry_key, embedding = _cache_manager.generate_cache_key(
                cache_key, (custom_key,), {}, use_similarity)
        else:
            # It's a static value, use it directly
            cache_entry_key, embedding = _cache_manager.generate_cache_key(
                cache_key, (use_as_cache_key,), {}, use_similarity)
    else:
        cache_entry_key, embedding = _cache_manager.generate_cache_key(
            cache_key, args, kwargs, use_similarity)

    return _cache_manager.get_cached_value(
        (cache_entry_key, embedding),
        use_similarity=use_similarity,
        similarity_threshold=similarity_threshold
    )


def set_cached(
        cache_key: str,
        value,
        args: tuple = (),
        kwargs: dict = {},
        expire_seconds: int = -1,
        use_similarity: bool = False,
        use_as_cache_key=None
):
    """
    Manually set a value in the cache.

    Args:
        cache_key: Base cache key
        value: Value to cache
        args: Arguments to include in the cache key
        kwargs: Keyword arguments to include in the cache key
        expire_seconds: Time in seconds until cache entry expires. Set -1 for no expiration.
        use_similarity: Whether to enable similarity checking for this key
        use_as_cache_key: Optional function or value to use as key

    Returns:
        The cache key used
    """
    # Handle use_as_cache_key appropriately
    if use_as_cache_key is not None:
        if callable(use_as_cache_key):
            # It's a function, call it with the provided args
            custom_key = use_as_cache_key(*args, **kwargs)
            cache_entry_key, _ = _cache_manager.generate_cache_key(
                cache_key, (custom_key,), {}, use_similarity)
        else:
            # It's a static value, use it directly
            cache_entry_key, _ = _cache_manager.generate_cache_key(
                cache_key, (use_as_cache_key,), {}, use_similarity)
    else:
        cache_entry_key, _ = _cache_manager.generate_cache_key(
            cache_key, args, kwargs, use_similarity)

    logging.info(f"CACHE - Setting value for key {cache_entry_key}")
    _cache_manager.set_cached_value(cache_entry_key, value, expire_seconds)
    return cache_entry_key

def cached(
        cache_key: str,
        expire_seconds: int,
        use_similarity: bool = False,
        similarity_threshold: float = 0.85,
        use_as_cache_key=None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Universal cache decorator that handles both sync and async functions.

    Args:
        cache_key: Base cache key for this function
        expire_seconds: Time in seconds until cache entries expire. Set -1 for no expiration.
        use_similarity: Whether to check for similar keys on cache miss
        similarity_threshold: Similarity threshold for matching (only used if use_similarity=True)
        use_as_cache_key: (Optional) Either a function that generates a key from args, or a static value to use as key
    Returns:
        Decorated function with caching behavior
    Note:
        Automatically detects if the decorated function is async
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        is_async = iscoroutinefunction(func)

        if is_async:
            @wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # Handle use_as_cache_key appropriately
                if use_as_cache_key is not None:
                    if callable(use_as_cache_key):
                        # It's a function, call it with the same args as the decorated function
                        custom_key = use_as_cache_key(*args, **kwargs)
                        cache_entry_key, embedding = _cache_manager.generate_cache_key(
                            cache_key, (custom_key,), {}, use_similarity)
                    else:
                        # It's a static value, use it directly
                        cache_entry_key, embedding = _cache_manager.generate_cache_key(
                            cache_key, (use_as_cache_key,), {}, use_similarity)
                else:
                    cache_entry_key, embedding = _cache_manager.generate_cache_key(
                        cache_key, args, kwargs, use_similarity)

                success, cached_value = _cache_manager.get_cached_value(
                    (cache_entry_key, embedding),
                    use_similarity=use_similarity,
                    similarity_threshold=similarity_threshold
                )
                if success:
                    return cached_value

                logging.info(f"CACHE - MISS for key {cache_entry_key}")
                result = await func(*args, **kwargs)
                _cache_manager.set_cached_value(cache_entry_key, result, expire_seconds)
                return result
        else:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                # Handle use_as_cache_key appropriately
                if use_as_cache_key is not None:
                    if callable(use_as_cache_key):
                        # It's a function, call it with the same args as the decorated function
                        custom_key = use_as_cache_key(*args, **kwargs)
                        cache_entry_key, embedding = _cache_manager.generate_cache_key(
                            cache_key, (custom_key,), {}, use_similarity)
                    else:
                        # It's a static value, use it directly
                        cache_entry_key, embedding = _cache_manager.generate_cache_key(
                            cache_key, (use_as_cache_key,), {}, use_similarity)
                else:
                    cache_entry_key, embedding = _cache_manager.generate_cache_key(
                        cache_key, args, kwargs, use_similarity)

                success, cached_value = _cache_manager.get_cached_value(
                    (cache_entry_key, embedding),
                    use_similarity=use_similarity,
                    similarity_threshold=similarity_threshold
                )
                if success:
                    return cached_value

                logging.info(f"CACHE - MISS for key {cache_entry_key}")
                result = func(*args, **kwargs)  # Direct call, no async/await
                _cache_manager.set_cached_value(cache_entry_key, result, expire_seconds)
                return result

        return wrapper

    return decorator