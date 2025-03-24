import os
import time as sys_time
from threading import Thread, Lock
import logging
from typing import Optional, Union, Dict
import redis
import asyncio

logging.basicConfig(level=logging.INFO)

# Global dictionaries to store connection pools and fake redis instances
_redis_connection_pools: Dict[int, redis.ConnectionPool] = {}
_fake_redis_instances: Dict[int, 'FakeRedis'] = {}
_fake_redis_pools: Dict[int, Dict[int, 'FakeRedis']] = {}


class FakeRedisPipeline:
    """Simple implementation of Redis pipeline interface for development."""

    def __init__(self, fake_redis):
        self.fake_redis = fake_redis
        self.commands = []

    def get(self, key: str) -> 'FakeRedisPipeline':
        """Queue a GET command."""
        self.commands.append(('get', [key]))
        return self

    def set(self, name: str, value: bytes) -> 'FakeRedisPipeline':
        """Queue a SET command."""
        self.commands.append(('set', [name, value]))
        return self

    def setex(self, name: str, time: int, value: bytes) -> 'FakeRedisPipeline':
        """Queue a SETEX command."""
        self.commands.append(('setex', [name, time, value]))
        return self

    def delete(self, *names: str) -> 'FakeRedisPipeline':
        """Queue a DELETE command."""
        self.commands.append(('delete', names))
        return self

    def execute(self) -> list:
        """Execute commands and return their results."""
        results = []
        for cmd, args in self.commands:
            method = getattr(self.fake_redis, cmd)
            results.append(method(*args))
        self.commands = []
        return results


class FakeRedisPool:
    """Manages a pool of FakeRedis instances for a specific DB."""

    def __init__(self, db: int, max_connections: int = 10, cleanup_interval: int = 300):
        self.db = db
        self.max_connections = max_connections
        self.cleanup_interval = cleanup_interval
        self.instances = {}
        self.available_ids = set(range(max_connections))
        self.lock = Lock()

    def get_connection(self) -> 'FakeRedis':
        """Get a FakeRedis instance from the pool or create a new one."""
        with self.lock:
            if not self.available_ids:
                # If pool is full, return a shared instance
                logging.warning(f"CACHE: FakeRedis pool for db={self.db} is full, sharing instance")
                instance_id = 0
            else:
                instance_id = self.available_ids.pop()

            if instance_id not in self.instances:
                self.instances[instance_id] = FakeRedis(self.cleanup_interval, instance_id, self)

            return self.instances[instance_id]

    def release_connection(self, instance_id: int):
        """Return a FakeRedis instance to the pool."""
        with self.lock:
            if instance_id < self.max_connections:
                self.available_ids.add(instance_id)
                logging.debug(f"CACHE: Released FakeRedis instance {instance_id} back to pool for db={self.db}")


class FakeRedis:
    """In-memory cache implementation that mimics Redis interface."""

    def __init__(self, cleanup_interval: int = 300, instance_id: int = 0, pool: Optional[FakeRedisPool] = None):
        """
        Initialize an empty in-memory store with cleanup mechanism.

        Args:
            cleanup_interval: Time in seconds between cleanup runs (default 5 minutes)
            instance_id: Unique ID for this instance within its pool
            pool: Reference to the pool this instance belongs to
        """
        self.store = {}
        self.lock = Lock()
        self.instance_id = instance_id
        self.pool = pool
        self._start_cleanup_thread(cleanup_interval)

    def __del__(self):
        """Release the connection when this instance is garbage collected."""
        if self.pool:
            self.pool.release_connection(self.instance_id)

    def pipeline(self) -> FakeRedisPipeline:
        """
        Create a new pipeline for batching commands.

        Returns:
            A new FakeRedisPipeline instance
        """
        return FakeRedisPipeline(self)

    def _start_cleanup_thread(self, interval: int) -> None:
        """Start background thread for periodic cleanup."""

        def cleanup_loop():
            while True:
                self._cleanup_expired()
                sys_time.sleep(interval)

        cleanup_thread = Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()

    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        current_time = sys_time.monotonic()
        with self.lock:
            expired_keys = [
                key for key, (_, expiry) in self.store.items()
                if expiry <= current_time
            ]
            for key in expired_keys:
                del self.store[key]
            if expired_keys:
                logging.info(f"CACHE: Cleaned up {len(expired_keys)} expired entries from instance {self.instance_id}")

    def get(self, key: str) -> Optional[bytes]:
        """
        Retrieve a value by key if it hasn't expired.

        Args:
            key: The cache key to retrieve

        Returns:
            The cached value as bytes if found and not expired, None otherwise
        """
        with self.lock:
            entry = self.store.get(key)
            if entry and entry[1] > sys_time.monotonic():
                return entry[0]
            if entry:  # Entry exists but is expired
                del self.store[key]
            return None

    def setex(self, name: str, time: int, value: bytes) -> None:
        """
        Set a value with an expiration time.

        Args:
            name: The cache key
            time: Expiration time in seconds
            value: The value to cache (as bytes)
        """
        with self.lock:
            self.store[name] = (value, sys_time.monotonic() + time)

    def set(self, name: str, value: bytes) -> None:
        """
        Set a value without an expiration time.

        Args:
            name: The cache key
            value: The value to cache (as bytes)
        """
        with self.lock:
            self.store[name] = (value, float('inf'))

    def scan(self, cursor: int = 0, match: str = None, count: int = None) -> tuple[int, list[bytes]]:
        """
        Mimics Redis SCAN command for iterating over keys.

        Args:
            cursor: Position to start scanning from (0 for initial scan)
            match: Pattern to match (currently supports only '*')
            count: Hint for how many items to return (may return more or less)

        Returns:
            Tuple of (next_cursor, list_of_keys)
            - next_cursor will be 0 when scan is complete
            - keys are returned as bytes to match Redis behavior
        """
        with self.lock:
            # Get all keys as a list
            all_keys = list(self.store.keys())
            if match and match.endswith('*'):
                all_keys = [key for key in all_keys if key.startswith(match[:-1])]

            # If no items or cursor is beyond length, return empty result
            if not all_keys or cursor >= len(all_keys):
                return 0, []

            # Determine batch size (default to 10 if count not specified)
            batch_size = count if count is not None else 10

            # Calculate next cursor and slice of keys to return
            next_cursor = cursor + batch_size
            if next_cursor >= len(all_keys):
                next_cursor = 0

            # Get slice of keys for this batch
            start_idx = cursor
            end_idx = min(cursor + batch_size, len(all_keys))
            batch_keys = all_keys[start_idx:end_idx]

            # Convert keys to bytes to match Redis behavior
            return_keys = [key.encode('utf-8') if isinstance(key, str) else key
                           for key in batch_keys]

            return next_cursor, return_keys

    def delete(self, *names: str) -> None:
        """
        Delete one or more cache entries.

        Args:
            *names: Variable number of cache keys to delete
        """
        with self.lock:
            for name in names:
                if name in self.store:
                    del self.store[name]

    def ping(self) -> bool:
        """Mimics Redis PING command."""
        return True

    def close(self) -> None:
        """Release the connection back to the pool."""
        if self.pool:
            self.pool.release_connection(self.instance_id)


class AsyncRedisWrapper:
    """Wrapper for Redis with periodic cleanup capabilities."""

    def __init__(self, redis_client: redis.Redis, cleanup_interval: int = 300):
        """
        Initialize Redis wrapper with cleanup mechanism.

        Args:
            redis_client: Redis client instance
            cleanup_interval: Time in seconds between cleanup runs (default 5 minutes)
        """
        self.redis = redis_client
        self.cleanup_interval = cleanup_interval
        self.cleanup_task = None

    async def initialize(self) -> None:
        """Initialize the cleanup task in an async context."""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """Continuous cleanup loop."""
        while True:
            await self._cleanup_expired()
            await asyncio.sleep(self.cleanup_interval)

    async def _cleanup_expired(self) -> None:
        """Remove expired entries using Redis SCAN."""
        try:
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(
                    cursor=cursor,
                    match='*',
                    count=1000
                )
                for key in keys:
                    if not self.redis.ttl(key):
                        self.redis.delete(key)
                if cursor == 0:
                    break
            logging.info("CACHE: Completed Redis cleanup cycle")
        except redis.RedisError as e:
            logging.error(f"CACHE: Error during Redis cleanup: {e}")

    async def close(self) -> None:
        """Clean up resources by canceling the cleanup task."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None


def get_redis_client(cleanup_interval: int = 300, db: int = 0, max_connections: int = 100) -> Union[
    redis.Redis, FakeRedis]:
    """
    Get a Redis client from the connection pool or a FakeRedis instance.

    Args:
        cleanup_interval: Time in seconds between cleanup runs
        db: Redis database number to use
        max_connections: Maximum number of connections in the pool

    Returns:
        Redis client if connection successful, FakeRedis instance otherwise
    """
    try:
        # Use connection pool for real Redis connections
        if db not in _redis_connection_pools:
            try:
                _redis_connection_pools[db] = redis.ConnectionPool(
                    host='localhost',
                    port=6379,
                    db=db,
                    decode_responses=False,
                    max_connections=max_connections,
                    socket_keepalive=True,
                )
            except redis.exceptions.ConnectionError:
                _redis_connection_pools[db] = redis.ConnectionPool(
                    host='redis',
                    port=6379,
                    db=db,
                    decode_responses=False,
                    max_connections=max_connections
                )

        # Get a client from the pool
        client = redis.Redis(connection_pool=_redis_connection_pools[db])
        if client.ping():
            logging.info(f"CACHE: Connected to Redis db={db} from connection pool")
            wrapper = AsyncRedisWrapper(client, cleanup_interval)
            return wrapper.redis
        raise ConnectionError("Failed to connect to Redis")

    except (redis.exceptions.ConnectionError, TimeoutError):
        # Use FakeRedis pool for local development
        if db not in _fake_redis_pools:
            logging.info(f"CACHE: Creating FakeRedis pool for db={db}")
            _fake_redis_pools[db] = FakeRedisPool(db, max_connections, cleanup_interval)

        logging.info(f"CACHE: Using FakeRedis from pool for db={db}")
        return _fake_redis_pools[db].get_connection()


def close_all_connections():
    """Close all Redis connections in all pools."""
    # Close real Redis connection pools
    for db, pool in _redis_connection_pools.items():
        logging.info(f"CACHE: Closing Redis connection pool for db={db}")
        pool.disconnect()

    # Clear the pools dictionary
    _redis_connection_pools.clear()

    # Clear FakeRedis pools
    _fake_redis_pools.clear()

    logging.info("CACHE: All Redis connections closed")
