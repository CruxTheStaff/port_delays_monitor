from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_database():
    client = None
    try:
        # Connect to MongoDB
        client = MongoClient()
        db = client['port_delays_monitor']

        # Get all collections
        collections = db.list_collection_names()
        logger.info(f"Found {len(collections)} collections")

        # Drop each collection
        for collection in collections:
            db.drop_collection(collection)
            logger.info(f"Dropped collection: {collection}")

        logger.info("Database cleared successfully")

    except Exception as e:
        logger.error(f"Error clearing database: {e}")
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    clear_database()