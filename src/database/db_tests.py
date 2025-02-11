from src.database.db_manager import DatabaseManager

db = DatabaseManager()
collections = db.db.list_collection_names()
print("Collections in database:", collections)

# Check a sample port's data
sample_port = db.get_all_ports()[0]
port_name = sample_port['name']
print(f"\nChecking data for port {port_name}:")
print("in_port:", db.get_latest_port_data(port_name, "in_port") is not None)
print("port_calls:", db.get_latest_port_data(port_name, "port_calls") is not None)
print("expected:", db.get_latest_port_data(port_name, "expected") is not None)