from pymongo import MongoClient
import logging
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_collection_name(collection_name: str) -> tuple[str, str]:
    """Safely parse collection name into port name and data type"""
    try:
        if '_' in collection_name:
            parts = collection_name.rsplit('_', 1)
            if len(parts) == 2:
                return parts[0], parts[1]
        # If no underscore or parsing fails, return original name and 'unknown'
        return collection_name, 'unknown'
    except Exception as e:
        logger.warning(f"Error parsing collection name '{collection_name}': {e}")
        return collection_name, 'unknown'


def analyze_mongodb():
    """Analyze MongoDB collections and data"""
    client = None
    try:
        # Connect to MongoDB
        client = MongoClient()
        db = client['port_delays_monitor']

        # Get all collections
        collections = db.list_collection_names()
        logger.info(f"Found {len(collections)} collections")

        # Analyze each collection
        for coll_name in collections:
            try:
                collection = db[coll_name]
                doc_count = collection.count_documents({})

                # Get port name and data type
                port_name, data_type = parse_collection_name(coll_name)

                # Get latest document
                latest_doc = collection.find_one(sort=[('timestamp', -1)])

                print(f"\n=== {port_name.upper()} - {data_type} ===")
                print(f"Total documents: {doc_count}")

                if latest_doc:
                    print(f"Latest timestamp: {latest_doc.get('timestamp', 'N/A')}")
                    vessels = latest_doc.get('vessels', [])
                    print(f"Vessels in latest record: {len(vessels)}")

                    if vessels:
                        print("\nSample vessel data:")
                        pprint(vessels[0])
                else:
                    print("No documents found in collection")

            except Exception as e:
                logger.error(f"Error analyzing collection '{coll_name}': {e}")
                continue

    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")

    finally:
        if client:
            try:
                client.close()
                logger.debug("MongoDB connection closed")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}")


if __name__ == "__main__":
    analyze_mongodb()