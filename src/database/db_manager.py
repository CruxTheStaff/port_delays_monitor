"""
Database Manager Module

Handles all database operations with MongoDB including:
- Connection pooling with adaptive settings
- Bulk operations with automatic periodic flush
- Dynamic LRU caching with auto-sizing
- Optimized indexing
- Thread-safe operations

Author: Stavroula Kamini
Last Modified: Current Date
"""

from datetime import datetime, UTC
from typing import List, Dict, Optional, Any, Mapping
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError, BulkWriteError
import logging
from functools import wraps
from collections import defaultdict
from threading import Thread, Event, Lock
import os
from cachetools import LRUCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicLRUCache:
    """
    A dynamic LRU cache implementation supporting runtime size adjustments.

    This cache can be used as a decorator and provides thread-safe operations
    with minimal lock contention. It uses cachetools.LRUCache as its underlying
    storage mechanism.

    Attributes:
        maxsize (int): Maximum number of items the cache can hold
        hits (int): Number of cache hits
        misses (int): Number of cache misses

    Example:
        @DynamicLRUCache(maxsize=128)
        def expensive_function(x, y):
            return x + y
    """

    def __init__(self, maxsize=128):
        """
        Initialize the cache with specified maximum size.

        Args:
            maxsize (int): Maximum number of items the cache can hold
        """
        self.maxsize = maxsize
        self._cache = LRUCache(maxsize=self.maxsize)
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def __call__(self, func):
        """
        Decorator implementation for caching function results.

        Args:
            func: The function to cache

        Returns:
            wrapper: Decorated function with caching capability

        Note:
            Function execution occurs outside the lock to prevent blocking
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))

            with self._lock:
                if key in self._cache:
                    self.hits += 1
                    return self._cache[key]

            # Execute function outside lock
            result = func(*args, **kwargs)

            with self._lock:
                self._cache[key] = result
                self.misses += 1
            return result

        return wrapper

    def get(self, key):
        """
        Thread-safe retrieval of cached value.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value or None if not found
        """
        with self._lock:
            return self._cache.get(key, None)

    def set(self, key, value):
        """
        Thread-safe setting of cache value.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            self._cache[key] = value

    def resize(self, new_maxsize: int):
        """
        Resize the cache by creating a new one with the new size.
        Preserves most recently used items up to new size.

        Args:
            new_maxsize (int): New maximum cache size
        """
        with self._lock:
            # Keep most recent items within new size limit
            items = dict(list(self._cache.items())[-new_maxsize:])
            self.maxsize = new_maxsize
            self._cache = LRUCache(maxsize=new_maxsize)
            self._cache.update(items)

    def invalidate(self, predicate):
        """
        Selectively invalidate cache entries based on a predicate.

        Args:
            predicate: Function that takes a cache key and returns bool
                      True if entry should be invalidated
        """
        with self._lock:
            # Create new cache with filtered items
            valid_items = {
                k: v for k, v in self._cache.items()
                if not predicate(k[0])
            }
            new_cache = LRUCache(maxsize=self.maxsize)
            new_cache.update(valid_items)
            self._cache = new_cache

    def get_stats(self):
        """
        Get current cache statistics.

        Returns:
            dict: Dictionary containing:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_ratio: Cache hit ratio
                - maxsize: Maximum cache size
                - current_size: Current number of cached items
        """
        with self._lock:
            total = self.hits + self.misses
            hit_ratio = self.hits / total if total > 0 else 0
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_ratio': hit_ratio,
                'maxsize': self.maxsize,
                'current_size': len(self._cache)
            }


def handle_db_errors(func):
    """Decorator for handling database operations errors with detailed logging"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PyMongoError as e:
            logger.error(f"Database error in {func.__name__}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise
    return wrapper


class DatabaseManager:
    """
    Manages database operations with optimized performance and caching

    Features:
    - Adaptive connection pooling
    - Bulk operations with automatic flush
    - Dynamic LRU caching
    - Optimized indexing
    - Thread-safe operations
    """

    def __init__(self, connection_url: str = None, db_name: str = None):
        """
        Initialize database connection with optimized settings

        Args:
            connection_url: MongoDB connection string (falls back to env var MONGO_URI)
            db_name: Target database name (falls back to env var MONGO_DB)
        """
        # Initialize MongoDB client with retry mechanism and optimized pooling
        self.client = self._connect_with_retry(
            connection_url or os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        )

        # Initialize database and operational parameters
        self.db = self.client[db_name or os.getenv("MONGO_DB", "port_delays_monitor")]
        self._indexed_collections = set()
        self._bulk_ops = defaultdict(list)
        self._bulk_threshold = int(os.getenv("BULK_THRESHOLD", 1000))

        # Initialize caching system
        self._cache_stats = {'hits': 0, 'misses': 0}
        self._cache_check_interval = 1000
        self._cache = DynamicLRUCache(maxsize=128)

        # Decorate methods with the dynamic cache
        self.get_latest_port_data = self._cache(self._get_latest_port_data_impl)
        self.get_port_data = self._cache(self._get_port_data_impl)

        # Initialize background flush thread
        self._flush_event = Event()
        self._flush_thread = Thread(target=self._auto_flush, daemon=True)
        self._flush_thread.start()

    @property
    def ports(self):
        return self._get_collection("ports")

    @handle_db_errors
    def get_all_ports(self) -> List[Dict]:
        """Retrieve all ports"""
        return list(self.ports.find())

    def _get_collection(self, collection_name: str):
        """
        Get or create collection with proper indexing

        Args:
            collection_name: Name of the collection to get/create

        Returns:
            MongoDB collection object with ensured indexes
        """
        if collection_name not in self.db.list_collection_names():
            collection = self.db.create_collection(collection_name)
            self._setup_indexes_for_collection(collection_name, collection)
        return self.db[collection_name]

    @staticmethod
    def _setup_indexes_for_collection(collection_name: str, collection) -> None:
        """
        Setup optimized indexes with existence check

        Args:
            collection_name: Name of the collection
            collection: MongoDB collection object
        """
        if '_' in collection_name:  # Port data collections
            required_indexes = [
                {
                    "keys": [("timestamp", DESCENDING)],
                    "opts": {}
                },
                {
                    "keys": [
                        ("port_id", ASCENDING),
                        ("timestamp", DESCENDING)
                    ],
                    "opts": {}
                },
                {
                    "keys": [
                        ("port_id", ASCENDING),
                        ("port_name", ASCENDING),
                        ("data_type", ASCENDING)
                    ],
                    "opts": {"unique": True}
                }
            ]

            existing_indexes = collection.index_information()

            for idx in required_indexes:
                idx_key = tuple((k, d) for k, d in idx["keys"])
                if not any(idx_key == tuple((k, d) for k, d in existing['key'])
                          for existing in existing_indexes.values()):
                    logger.info(f"Creating missing index {idx_key} on {collection_name}")
                    collection.create_index(idx["keys"], **idx["opts"])

    def _track_cache_access(self, hit: bool):
        """Update cache statistics"""
        self._cache_stats['hits' if hit else 'misses'] += 1

        # Auto-adjust cache size periodically
        total_ops = self._cache_stats['hits'] + self._cache_stats['misses']
        if total_ops >= self._cache_check_interval:
            self._adjust_cache_size()
            self._cache_stats = {'hits': 0, 'misses': 0}  # Reset

    def _adjust_cache_size(self):
        """Adjust cache size based on hit ratio"""
        total_ops = self._cache_stats['hits'] + self._cache_stats['misses']
        if total_ops == 0:
            return

        hit_ratio = self._cache_stats['hits'] / total_ops
        current_size = self._cache.maxsize

        if hit_ratio < 0.5 and current_size < 512:
            new_size = min(current_size * 2, 512)
        elif hit_ratio > 0.8 and current_size > 128:
            new_size = max(current_size // 2, 128)
        else:
            return

        self._cache.resize(new_size)
        logger.info(f"Cache resized to {new_size} (hit ratio: {hit_ratio:.2f})")

    def _auto_flush(self):
        """Background thread function for periodic bulk operations flush"""
        while not self._flush_event.wait(300):  # 5 minutes
            try:
                self.flush_bulk_ops()
            except Exception as e:
                logger.error(f"Error in background flush: {e}")

    def _add_to_bulk(self, collection_name: str, operation: dict) -> None:
        """Add operation to bulk operations queue"""
        self._bulk_ops[collection_name].append(operation)
        if len(self._bulk_ops[collection_name]) >= self._bulk_threshold:
            self.flush_bulk_ops(collection_name)

    @staticmethod
    def _connect_with_retry(connection_url: str, max_attempts: int = 3) -> MongoClient[Mapping[str, Any] | Any] | None:
        """
        Establish database connection with retry mechanism

        Args:
            connection_url: MongoDB connection string
            max_attempts: Maximum number of connection attempts

        Returns:
            MongoDB client instance
        """
        attempts = 0
        while attempts < max_attempts:
            try:
                client = MongoClient(
                    connection_url,
                    maxPoolSize=int(os.getenv("MONGO_MAX_POOL", 50)),
                    minPoolSize=int(os.getenv("MONGO_MIN_POOL", 10)),
                    maxIdleTimeMS=int(os.getenv("MONGO_IDLE_MS", 45000)),
                    retryWrites=True,
                    serverSelectionTimeoutMS=5000,
                    waitQueueTimeoutMS=2500
                )
                client.admin.command('ping')
                logger.info("Successfully connected to MongoDB")
                return client
            except PyMongoError:
                attempts += 1
                if attempts == max_attempts:
                    logger.error(f"Failed to connect to MongoDB after {max_attempts} attempts")
                    raise
                logger.warning(f"Connection attempt {attempts} failed, retrying...")

    @handle_db_errors
    def _get_latest_port_data_impl(self, port_name: str, data_type: str) -> Optional[Dict]:
        """
        Actual implementation of latest port data retrieval

        Args:
            port_name: Name of the port
            data_type: Type of data to retrieve

        Returns:
            Most recent port data document or None
        """
        collection_name = f"{port_name.lower()}_{data_type}"
        collection = self._get_collection(collection_name)

        result = collection.find_one(
            sort=[("timestamp", DESCENDING)]
        )
        self._track_cache_access(result is not None)
        return result

    @handle_db_errors
    def _get_port_data_impl(self, port_name: str, data_type: str,
                           start_date: datetime = None,
                           end_date: datetime = None) -> List[Dict]:
        """
        Actual implementation of port data retrieval

        Args:
            port_name: Name of the port
            data_type: Type of data to retrieve
            start_date: Optional start of timeframe
            end_date: Optional end of timeframe

        Returns:
            List of port data documents
        """
        collection_name = f"{port_name.lower()}_{data_type}"
        collection = self._get_collection(collection_name)

        query = {}
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date

        result = list(collection.find(query).sort("timestamp", DESCENDING))
        self._track_cache_access(len(result) > 0)
        return result

    def flush_bulk_ops(self, collection_name: str = None) -> None:
        """
        Execute pending bulk operations

        Args:
            collection_name: Optional specific collection to flush
        """
        try:
            collections = [collection_name] if collection_name else list(self._bulk_ops.keys())
            for coll in collections:
                if self._bulk_ops[coll]:
                    result = self.db[coll].bulk_write(
                        self._bulk_ops[coll],
                        ordered=False
                    )
                    logger.info(
                        f"Bulk operation completed for {coll}: "
                        f"inserted={result.inserted_count}, "
                        f"modified={result.modified_count}, "
                        f"upserted={len(result.upserted_ids)}"
                    )
                    self._bulk_ops[coll].clear()
        except BulkWriteError as bwe:
            logger.error(f"Bulk write error: {bwe.details}")
            raise
        except Exception as e:
            logger.error(f"Error in bulk operation: {str(e)}")
            raise

    def save_port_data(self, port_id: str, port_name: str, data_type: str,
                       vessels: List[Dict], timestamp: datetime = None) -> str:
        """Update port data file with new vessel information"""
        try:
            if timestamp is None:
                timestamp = datetime.now(UTC)

            timestamp = timestamp.replace(second=0, microsecond=0)
            collection_name = f"{port_name.lower()}_{data_type}"
            collection = self._get_collection(collection_name)

            # Create or update the single document for this port/type
            document = {
                "port_id": port_id,
                "port_name": port_name,
                "data_type": data_type,
                "last_updated": timestamp,
                "vessels": vessels
            }

            # Update or create if doesn't exist
            collection.replace_one(
                {
                    "port_id": port_id,
                    "port_name": port_name,
                    "data_type": data_type
                },
                document,
                upsert=True
            )

            logger.info(f"Updated {collection_name} with {len(vessels)} vessels")
            return collection_name

        except Exception as e:
            logger.error(f"Failed to save port data: {e}")
            raise

    def invalidate_cache(self, port_name: str = None, data_type: str = None):
        """
        Invalidate cache entries for specific port/data type

        Args:
            port_name: Optional port name to invalidate
            data_type: Optional data type to invalidate
        """

        def predicate(args):
            """Check if cache entry matches criteria"""
            if port_name and data_type:
                return args[0] == port_name and args[1] == data_type
            return True  # If no criteria, invalidate all

        self._cache.invalidate(predicate)

    def close(self):
        """
        Safely close database connection

        Features:
            - Graceful shutdown of background thread
            - Final flush of pending operations
            - Proper connection cleanup
        """
        try:
            self._flush_event.set()  # Stop background thread
            self._flush_thread.join(timeout=5)
            self.flush_bulk_ops()    # Final flush
        finally:
            if self.client:
                self.client.close()

