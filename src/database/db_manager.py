from datetime import datetime
from typing import List, Dict, Optional, Any, Mapping
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import PyMongoError
import logging
from functools import wraps

from ..models.port import Port
from ..models.vessel_call import VesselCall
from ..models.delay_record import DelayRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_db_errors(func):
    """Decorator for handling database operations errors"""

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
    def __init__(self, connection_url: str = "mongodb://localhost:27017/",
                 db_name: str = "port_delays_monitor"):
        """Initialize database connection and collections"""
        self.client = self._connect_with_retry(connection_url)
        self.db = self.client[db_name]

        # Initialize collections
        self.ports = self.db.ports
        self.vessel_calls = self.db.vessel_calls
        self.delay_records = self.db.delay_records

        # Setup indexes
        self._setup_indexes()

    @staticmethod
    def _connect_with_retry(connection_url: str, max_attempts: int = 3) -> MongoClient[Mapping[str, Any] | Any] | None:
        """Establish database connection with retry mechanism"""
        attempts = 0
        while attempts < max_attempts:
            try:
                client = MongoClient(connection_url)
                # Test connection
                client.admin.command('ping')
                logger.info("Successfully connected to MongoDB")
                return client
            except PyMongoError:
                attempts += 1
                if attempts == max_attempts:
                    logger.error(f"Failed to connect to MongoDB after {max_attempts} attempts")
                    raise
                logger.warning(f"Connection attempt {attempts} failed, retrying...")

    def _setup_indexes(self) -> None:
        """Setup necessary indexes for collections"""
        # Ports indexes
        self.ports.create_index([("port_id", ASCENDING)], unique=True)
        self.ports.create_index([("name", ASCENDING)])

        # Vessel calls indexes
        self.vessel_calls.create_index([("call_id", ASCENDING)], unique=True)
        self.vessel_calls.create_index([("port_id", ASCENDING)])
        self.vessel_calls.create_index([("vessel.vessel_id", ASCENDING)])
        self.vessel_calls.create_index([("eta", ASCENDING)])

        # Delay records indexes
        self.delay_records.create_index([("record_id", ASCENDING)], unique=True)
        self.delay_records.create_index([("port_call_id", ASCENDING)])
        self.delay_records.create_index([("port_id", ASCENDING)])
        self.delay_records.create_index([("created_at", DESCENDING)])

    # Port operations
    @handle_db_errors
    def save_port(self, port: Port) -> str:
        """Save or update port information"""
        port_data = port.model_dump()
        self.ports.update_one(
            {"port_id": port.port_id},
            {"$set": port_data},
            upsert=True
        )
        return str(port.port_id)

    @handle_db_errors
    def get_port(self, port_id: str) -> Optional[Dict]:
        """Retrieve port information"""
        return self.ports.find_one({"port_id": port_id})

    @handle_db_errors
    def get_all_ports(self) -> List[Dict]:
        """Retrieve all ports"""
        return list(self.ports.find())

        # Vessel call operations

    @handle_db_errors
    def save_vessel_call(self, vessel_call: VesselCall) -> str:
        """Save or update vessel call"""
        call_data = vessel_call.model_dump()
        self.vessel_calls.update_one(
            {"call_id": vessel_call.call_id},
            {"$set": call_data},
            upsert=True
        )
        return str(vessel_call.call_id)

    @handle_db_errors
    def get_vessel_call(self, call_id: str) -> Optional[Dict]:
        """Retrieve vessel call information"""
        return self.vessel_calls.find_one({"call_id": call_id})

    @handle_db_errors
    def get_port_vessel_calls(self, port_id: str,
                              start_date: datetime = None,
                              end_date: datetime = None) -> List[Dict]:
        """Retrieve vessel calls for a specific port within date range"""
        query = {"port_id": port_id}
        if start_date or end_date:
            query["eta"] = {}
            if start_date:
                query["eta"]["$gte"] = start_date
            if end_date:
                query["eta"]["$lte"] = end_date

        return list(self.vessel_calls.find(query).sort("eta", ASCENDING))

        # Delay record operations

    @handle_db_errors
    def save_delay_record(self, delay_record: DelayRecord) -> str:
        """Save or update delay record"""
        delay_data = delay_record.model_dump()
        self.delay_records.update_one(
            {"record_id": delay_record.record_id},
            {"$set": delay_data},
            upsert=True
        )
        return str(delay_record.record_id)

    @handle_db_errors
    def get_delay_record(self, record_id: str) -> Optional[Dict]:
        """Retrieve delay record"""
        return self.delay_records.find_one({"record_id": record_id})

    @handle_db_errors
    def get_port_delays(self, port_id: str,
                        start_date: datetime = None,
                        end_date: datetime = None) -> List[Dict]:
        """Retrieve delay records for a specific port within date range"""
        query = {"port_id": port_id}
        if start_date or end_date:
            query["created_at"] = {}
            if start_date:
                query["created_at"]["$gte"] = start_date
            if end_date:
                query["created_at"]["$lte"] = end_date

        return list(self.delay_records.find(query).sort("created_at", DESCENDING))

    @handle_db_errors
    def get_vessel_delays(self, vessel_id: str) -> List[Dict]:
        """Retrieve all delay records for a specific vessel"""
        return list(self.delay_records.find(
            {"vessel_id": vessel_id}
        ).sort("created_at", DESCENDING))

    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
