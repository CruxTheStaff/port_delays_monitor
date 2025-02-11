# src/database/populate_ports.py
import pandas as pd
from pathlib import Path
import logging
from src.database.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def populate_ports_collection():
    """Populate ports collection from ports_catalog.csv"""
    db = None
    ports_file = None

    try:
        # Get project root directory and file path
        project_root = Path(__file__).resolve().parents[2]
        ports_file = project_root / 'src' / 'data_collection' / 'ports' / 'ports_enriched.csv'

        if not ports_file.exists():
            logger.error(f"Ports catalog file not found at {ports_file}")
            return

        logger.info(f"Reading ports data from: {ports_file}")

        # Read ports catalog
        df = pd.read_csv(ports_file)
        logger.info(f"Found {len(df)} ports in catalog")

        # Initialize database
        db = DatabaseManager()

        # Clear existing ports
        existing_count = db.db.ports.count_documents({})
        if existing_count > 0:
            logger.warning(f"Deleting {existing_count} existing ports")
            db.db.ports.delete_many({})

        # Prepare ports data
        ports = []
        for _, row in df.iterrows():
            port = {
                'name': row['name'],
                'port_id': row['port_id'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'size': row['size']
            }
            ports.append(port)

        if not ports:
            logger.warning("No ports data to insert")
            return

        # Insert ports
        result = db.db.ports.insert_many(ports)

        # Verify insertion
        inserted_count = len(result.inserted_ids)
        logger.info(f"Successfully inserted {inserted_count} ports")

        # Create indexes
        db.db.ports.create_index([("port_id", 1)], unique=True)
        db.db.ports.create_index([("name", 1)])
        logger.info("Created indexes on ports collection")

        # Sample verification
        sample_port = db.db.ports.find_one()
        if sample_port:
            logger.info("Sample port data:")
            logger.info(sample_port)

    except pd.errors.EmptyDataError:
        logger.error(f"The file {ports_file} is empty")
    except Exception as e:
        logger.error(f"Error populating ports: {e}")
    finally:
        if db:
            db.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    populate_ports_collection()